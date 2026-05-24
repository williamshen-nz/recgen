# Handover: Multi-GPU RecGen Serving

Branch: `will/multi-gpu`. Purpose: serve RecGen inference across 4 GPUs on a
workstation, with one model-holding worker process per GPU behind a single
public endpoint. This doc is the deploy + operate + verify guide for moving it
onto the 4-GPU box.

## TL;DR

```bash
git checkout will/multi-gpu
pixi install                 # pulls httpx into the lock/env
pixi run serve               # auto-detects all GPUs -> one worker each, public :18324
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

## Files changed on this branch

| File | Change |
| --- | --- |
| `scripts/gateway.py` | **New.** Supervisor + dispatcher (spawns workers, idle-queue routing, health, retry). |
| `scripts/server.py` | Per-GPU worker. `generate()` moved off the event loop into a single-thread executor; added `--worker-label` for log prefixes. Role otherwise unchanged. |
| `pixi.toml` | Added `httpx`; `serve` now launches the gateway; new `serve-worker` task for single-GPU debugging. |
| `pyproject.toml` | Added `httpx>=0.27`. |
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
   pixi run serve                          # all detected GPUs
   # or pin explicitly / change port:
   pixi run serve --gpus 0,1,2,3 --port 18324
   ```
   Startup loads the model 4x (once per worker, in parallel). The gateway logs
   per-worker readiness and only begins serving once at least one worker is up
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

## Known limitations / open decisions

- **No worker auto-respawn (v1).** If a worker crashes mid-request, it's dropped
  from the pool and the request retries on another GPU; capacity stays reduced
  until you restart the gateway. Auto-respawn is a small add-on if wanted — flagged
  for a decision, not yet implemented.
- **Startup cost.** 4x model load + 4x VRAM. Confirm the box has enough VRAM per
  GPU for one pipeline each (it's the same footprint as the old single server,
  just times four — one model per card, not four on one card).
- **Single-host only.** Workers are local subprocesses. Cross-machine scaling
  would need a real queue (Redis/Celery) — deliberately out of scope to keep
  infra simple.
- **Lock file.** `httpx` was added to deps; run `pixi install` on the box so the
  lock resolves there. `httpx` already imports fine in the current env, so this
  is expected to be a no-op-ish resolve.

## How it was tested here

- Both modules compile and import under the pixi env.
- Gateway dispatch logic unit-tested with mocked workers (httpx `MockTransport`):
  happy-path worker release, `503` on saturation, retry-on-dead-worker, and
  `502` when all workers are dead — all pass.
- Not yet run end-to-end against real GPUs/model — that's the verification list
  above.

## Rollback

The old single-GPU behavior is preserved: `pixi run serve-worker` runs one
`scripts/server.py` on `:18324` exactly as before. Or `git checkout main`.
