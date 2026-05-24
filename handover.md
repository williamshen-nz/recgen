# Handover: Multi-GPU RecGen Serving

Branch: `will/multi-gpu`. Purpose: serve RecGen inference across 4 GPUs on a
workstation, with one model-holding worker process per GPU behind a single
public endpoint. This doc is the deploy + operate + verify guide for moving it
onto the 4-GPU box.

## TL;DR

```bash
git checkout will/multi-gpu
pixi install                 # pulls httpx + aiohttp into the lock/env
pixi run server              # auto-detects all GPUs -> one worker each, public :18324
```

Clients keep talking to a single URL (`http://<host>:18324`); the multi-GPU
fan-out is transparent. Same `/generate` and `/health` msgpack contract as the
old single-GPU server.

## Architecture

```
client (aiohttp, unchanged)
        |  POST /generate            (one stable URL :18324)
        v
  scripts/gateway.py                 async FastAPI: supervises + dispatches
        |  asyncio.Queue of IDLE workers -> pop idle -> httpx proxy -> push back
        +-- spawns / monitors --------------------------------------------+
  scripts/server.py x4 (one per GPU)                                       |
   :18401 CUDA_VISIBLE_DEVICES=0  :18402 =1  :18403 =2  :18404 =3 <--------+
   (each loads its own pipeline at startup)
```

- **Routing = idle-worker queue.** Every request pops a genuinely-free GPU (not
  blind round-robin) and returns it to the pool when done. All GPUs busy ->
  requests wait FIFO (fair across concurrent clients); past `--queue-timeout`
  they get a `503` instead of piling up unbounded.
- **Synchronous (connection held).** The client holds the HTTP connection until
  its result returns. No job IDs / polling / result storage — chosen because the
  client is already async (`aiohttp`, `gather` per object) with a 600 s timeout,
  so it gets cross-GPU parallelism for free with zero client changes.
- **One worker = one GPU = one in-flight generation.** Worker holds the model in
  its own CUDA context; the gateway never sends a worker a second request while
  it's busy. Inside the worker, `generate()` runs in a single-thread executor so
  `/health` stays responsive but generations stay serialized.
- **Self-healing workers.** A worker that crashes (segfault / OOM-kill) or
  wedges its CUDA context is recycled: the in-flight request retries on another
  GPU and the dead worker respawns in the background, rejoining the pool once its
  pipeline reloads. The poisoned-context case is handled at the source — after a
  failed generation the worker probes its CUDA context with a trivial op and, if
  it's unusable (device-side assert / illegal memory access), self-exits so the
  gateway replaces it instead of returning a wedged worker to the pool.
  Recoverable errors (bad input, etc.) come back as an HTTP 500 and the worker
  stays up.

## Design decisions (the why)

Captured here so the next person doesn't re-litigate them.

- **One process per GPU, not one process spanning GPUs.** A PyTorch model lives
  in a single process's CUDA context, and the GIL prevents real parallelism
  across GPUs in one process. Separate processes are the only clean way to drive
  4 GPUs at once, and they also isolate failures (one crash ≠ all down).

- **Server-side gateway, not client-side balancing across 4 ports.** We
  considered just opening 4 ports and letting the client pick. Rejected because:
  multiple clients can't coordinate (two clients would both hammer GPU 0 while
  others idle); the client would have to reimplement health/retry/failover; and
  there'd be no global queue. A central gateway gives one stable URL, global
  fairness, and one place for health/metrics.

- **Idle-worker queue, not round-robin / nginx.** Each generation is *seconds*
  of GPU work, so head-of-line blocking is the enemy: a dumb round-robin (or
  nginx without live busy-tracking) can queue a request behind a busy worker
  while another GPU sits idle. An `asyncio.Queue` of idle workers means every
  request pops a genuinely-free GPU; when all are busy, requests wait FIFO and
  fall back to 503. ~50 lines, no extra infra.

- **Synchronous (hold the connection), not async job+poll.** The client is
  already async (`aiohttp`, `gather` per object) with a long timeout, so holding
  the connection gives cross-GPU parallelism with *zero* client changes. Job IDs
  + a `/status` endpoint + result storage would be real added complexity that
  buys nothing here (the client never needs to survive a disconnect mid-job).

- **Internal HTTP workers, not a `ProcessPoolExecutor` / `multiprocessing`.**
  Reusing `server.py` as the worker keeps each GPU in its own clean CUDA process
  (no fork/spawn-CUDA hazards), avoids pickling multi-MB numpy arrays through a
  pool, and lets you curl a single worker directly when debugging. Cost is 4
  localhost ports + a thin gateway — worth it.

- **No Redis/Celery/queue broker.** Single-host, in-process queue is enough and
  keeps ops trivial (one command, no broker to run). Revisit only if scaling
  across machines.

- **Self-healing at the source for poisoned CUDA contexts.** The nastier GPU
  failure isn't a crash — it's a corrupted-but-alive context (device-side assert
  / illegal access) that returns 500s forever. Detecting that from the gateway by
  parsing error bodies is brittle, so the *worker* probes its own context after a
  failure and self-exits if wedged, funneling it into the same crash→respawn
  path. One mechanism covers both crash and poison.

- **Staggered startup to warm caches once, not a hardcoded pre-warm.** If all 4
  workers boot together on a cold machine they each fetch the same files — and
  it's not just HF weights: the pipeline also pulls DINOv2 via `torch.hub`, whose
  cache isn't as robust to concurrent writers. Rather than enumerate every
  download source (and bake in a repo name), the gateway brings up **one** worker
  first and waits for it to fully load; that single worker populates *every*
  shared cache, then the rest start and load from cache. Robust to whatever the
  pipeline downloads, and there's no `--hf-repo`-style knob to get wrong. Cost: one
  extra sequential model load at startup (even on warm cache) — judged worth it.

- **Client is fully async (`aiohttp`), not thread-pooled.** This mirrors the real
  downstream client: one shared `aiohttp.ClientSession`, all objects POSTed onto
  the event loop, bounded by an `asyncio.Semaphore(--concurrency)` (default 4).
  The client stays torch-free so it runs on a laptop. `--concurrency` should track
  the server's GPU count.

- **Quiet third-party logs.** `huggingface_hub` 1.x (httpx-based) logs a line per
  cache HEAD-check on every model file, and `spconv` emits a `FutureWarning` per
  kernel — both pure noise at startup. Workers raise those loggers to WARNING and
  filter the spconv warning; the gateway quiets httpx too (it'd otherwise log
  every proxied request).

## Files changed on this branch

| File | Change |
| --- | --- |
| `scripts/gateway.py` | **New.** Supervisor + dispatcher: staggered worker startup (warms caches), idle-queue routing, health, retry, background recycle of dead/poisoned workers; quiets httpx logs. |
| `scripts/server.py` | Per-GPU worker. `generate()` moved off the event loop into a single-thread executor; self-exits on a wedged CUDA context so the gateway respawns it; quiets HF/httpx logs + spconv warnings; added `--worker-label`. |
| `scripts/client_tiptop.py` | Demo client rewritten async (`aiohttp` + semaphore, `--concurrency` default 4) to fan out across GPUs; added `--target-faces`; default URL → `:18324`. |
| `pixi.toml` | Added `httpx` + `aiohttp`; the `server` task now launches the gateway (was the single-GPU `serve`). |
| `pyproject.toml` | Added `httpx>=0.27`, `aiohttp>=3.9`. |
| `README.md` | New "Serving (HTTP)" section. |

## Deploying on the 4-GPU workstation

1. **Get the code + env**
   ```bash
   git checkout will/multi-gpu
   pixi install                  # or: pixi install -e cu118  (match the box's CUDA)
   ```
   Pick the env that matches the workstation's CUDA (default is CUDA 12.1; use
   `-e cu118` for CUDA 11.8). `spconv` must match too — see README troubleshooting.

2. **Sanity-check the GPUs are visible**
   ```bash
   nvidia-smi -L                 # expect: GPU 0..3
   ```

3. **Start the service**
   ```bash
   pixi run server                         # all detected GPUs
   # or pin explicitly / change port:
   pixi run server --gpus 0,1,2,3 --port 18324
   ```
   Startup is **staggered**: the gateway brings up one worker first
   (`Bringing up gpu0 first to warm the weight caches...`) and waits for it to
   load — on a cold machine that single worker does all the downloading (HF
   weights *and* the torch.hub/DINOv2 fetch). It then starts the rest
   (`Caches warm; starting remaining N worker(s) from cache`), which load from
   cache in parallel. It only begins serving once at least one worker is up
   (warns if fewer than all 4 came up). Watch for `Gateway ready: 4 worker(s)`.

4. **Point the client at it** — set the client's `server_url` to
   `http://<workstation-host>:18324`. Nothing else changes.

## Configuration knobs (`scripts/gateway.py`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--gpus` | auto (`CUDA_VISIBLE_DEVICES`, else `nvidia-smi`) | Comma-separated GPU ids; one worker each. |
| `--port` | `18324` | Public gateway port. |
| `--worker-base-port` | `18401` | Worker `i` listens on `base+i` (internal, localhost only). |
| `--checkpoint` | `recgen_base.multiview_stereo` | Passed to every worker. |
| `--queue-timeout` | `300` s | Max wait for a free GPU before returning `503`. |
| `--request-timeout` | `600` s | Max time for one worker `/generate`. Keep >= client timeout. |
| `--worker-startup-timeout` | `600` s | Max wait per worker to load its pipeline at boot. |

## Verifying it works (do this on the box)

These couldn't be exercised on the dev machine (1 GPU + heavy model) — please
confirm on the 4-GPU workstation:

1. **All workers load.** Logs show 4x `pipeline ready` and
   `Gateway ready: 4 worker(s)`. `nvidia-smi` shows memory used on all 4 GPUs.
2. **Health:**
   ```bash
   curl -s localhost:18324/health
   # expect workers_total=4, workers_alive=4, workers_idle=4, pipeline_loaded=true
   ```
3. **Single request** returns a mesh (use the existing client against one
   capture).
4. **Fan-out under load** — the key thing to confirm: fire several objects
   concurrently and watch `nvidia-smi` (e.g. `watch -n0.5 nvidia-smi`) — all 4
   GPUs should light up, and `/health`'s `workers_idle` should drop toward 0
   while busy. With the tiptop client, a multi-object capture should now finish
   in roughly `ceil(N/4)` waves instead of `N` serial generations.
5. **Backpressure** — with all GPUs busy and many more requests queued, extra
   ones should eventually 503 after `--queue-timeout` rather than hang forever.
6. **Self-heal** (optional, if you can induce a failure): kill a worker process
   (`kill <pid>` of one `server.py`) mid-load and confirm the gateway logs
   `Recycling worker gpuN` and later `Worker gpuN recovered and back in pool`,
   with `workers_alive` dipping then returning to 4.

## Known limitations / open decisions

- **Recycle reload cost.** A recycled worker is out of the pool for a full model
  reload (the startup cost, for one GPU, in the background). Repeated instant
  recycling of the *same* worker (e.g. a deterministically-crashing input) would
  thrash; there's no backoff/circuit-breaker yet. Hasn't been a problem, flagged
  in case failures cluster.
- **Startup cost.** 4x model load + 4x VRAM. Confirm the box has enough VRAM per
  GPU for one pipeline each (it's the same footprint as the old single server,
  just times four — one model per card, not four on one card).
- **Single-host only.** Workers are local subprocesses. Cross-machine scaling
  would need a real queue (Redis/Celery) — deliberately out of scope to keep
  infra simple.
- **Lock file.** `httpx` (gateway) and `aiohttp` (client) were added to deps; run
  `pixi install` on the box so the lock resolves there.

## How it was tested here

- Both modules compile and import under the pixi env.
- `_cuda_context_broken()` checked to return `False` on a healthy GPU here.
- spconv `FutureWarning` confirmed silenced on `import scripts.server`.
- The committed suite (`tests/`) is still just `test_build_recgen.py` and
  `test_smoke.py`; it does **not** yet cover the gateway. The dispatch and recycle
  paths (idle-queue release, `503` on saturation, retry-on-dead-worker, `502`
  when all are dead, background respawn, the double-recycle guard) were exercised
  during development with throwaway httpx `MockTransport` scripts but not turned
  into committed tests — a good follow-up, and worth adding before relying on
  these paths in production.
- The gateway's *staggered* startup and worker *respawn* paths spawn real
  `server.py` subprocesses, so they're best exercised on the box via the
  verification list above.
- Not yet run end-to-end against real GPUs/model — that's the verification list
  above (including the self-heal step).

## Rollback

The old single-GPU behavior is still available by running a worker directly:
`pixi run python scripts/server.py --port 18324` serves one `scripts/server.py`
on `:18324` exactly as before. Or `git checkout main`.
