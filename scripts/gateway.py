"""Multi-GPU gateway for RecGen inference.

Spawns one ``scripts/server.py`` worker per GPU (each pinned via
``CUDA_VISIBLE_DEVICES`` and holding its own pipeline), then exposes a single
public ``/generate`` endpoint that dispatches each request to an *idle* worker.

Routing is an ``asyncio.Queue`` of idle worker URLs: every request pops a
genuinely-free GPU (not blind round-robin), proxies the msgpack body to it over
localhost, and returns the worker to the pool when done. When all GPUs are
busy, requests wait FIFO (fair across concurrent clients); past
``--queue-timeout`` they get a 503 instead of piling up unbounded.

The client speaks to this gateway exactly as it spoke to a single server — same
``/generate`` and ``/health`` contract, same msgpack wire format — so nothing on
the client side changes.

Run with::

    pixi run serve                       # auto-detects GPUs
    pixi run python scripts/gateway.py --gpus 0,1,2,3 --port 18324
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("recgen_inference.gateway")

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = Path(__file__).resolve().parent / "server.py"


@dataclass
class Worker:
    """One GPU-pinned ``server.py`` subprocess."""

    label: str
    gpu: str
    port: int
    proc: subprocess.Popen
    alive: bool = True
    recycling: bool = False  # guards against starting two respawns at once

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class GatewayState:
    cfg: argparse.Namespace
    workers: List[Worker] = field(default_factory=list)
    idle: "asyncio.Queue[Worker]" = field(default_factory=asyncio.Queue)
    client: Optional[httpx.AsyncClient] = None
    # Strong refs to in-flight recycle tasks (asyncio may GC unreferenced ones).
    recycle_tasks: set = field(default_factory=set)


_state: Optional[GatewayState] = None


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

def _start_proc(cfg: argparse.Namespace, gpu: str, port: int, label: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["RECGEN_WORKER_LABEL"] = label
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--checkpoint", cfg.checkpoint,
        # Exactly one GPU is visible to the worker, so plain "cuda" == that GPU.
        "--device", "cuda",
        "--log-level", cfg.log_level,
    ]
    logger.info("Spawning worker %s on port %d (CUDA_VISIBLE_DEVICES=%s)", label, port, gpu)
    # Inherit stdout/stderr so worker logs (prefixed with [gpuN]) interleave here.
    return subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT))


def _spawn_worker(cfg: argparse.Namespace, gpu: str, port: int) -> Worker:
    label = f"gpu{gpu}"
    return Worker(label=label, gpu=gpu, port=port, proc=_start_proc(cfg, gpu, port, label))


async def _kill_proc(proc: subprocess.Popen) -> None:
    """Terminate a process without blocking the event loop."""
    if proc.poll() is not None:
        return
    proc.terminate()
    for _ in range(50):  # up to ~10s
        if proc.poll() is not None:
            return
        await asyncio.sleep(0.2)
    proc.kill()


async def _recycle_worker(worker: Worker) -> None:
    """Replace a dead/poisoned worker's process and re-add it to the pool.

    Runs as a background task: the triggering request has already moved on (it
    retries on another GPU), so the only effect here is that this one GPU sits
    out of the idle queue until its fresh pipeline finishes loading.
    """
    assert _state is not None and _state.client is not None
    if worker.recycling:  # a respawn is already in flight for this worker
        return
    worker.recycling = True
    worker.alive = False
    try:
        logger.warning("Recycling worker %s (pid %s)...", worker.label, worker.proc.pid)
        await _kill_proc(worker.proc)
        worker.proc = _start_proc(_state.cfg, worker.gpu, worker.port, worker.label)
        if await _await_ready(_state.client, worker, _state.cfg.worker_startup_timeout):
            worker.alive = True
            _state.idle.put_nowait(worker)
            logger.info("Worker %s recovered and back in pool", worker.label)
        else:
            logger.error("Worker %s failed to recover; leaving out of pool", worker.label)
    finally:
        worker.recycling = False


def _schedule_recycle(worker: Worker) -> None:
    assert _state is not None
    task = asyncio.create_task(_recycle_worker(worker))
    _state.recycle_tasks.add(task)
    task.add_done_callback(_state.recycle_tasks.discard)


async def _await_ready(client: httpx.AsyncClient, worker: Worker, timeout: float) -> bool:
    """Poll a worker's /health until its pipeline is loaded or we time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.proc.poll() is not None:
            logger.error("Worker %s exited during startup (code %s)", worker.label, worker.proc.returncode)
            return False
        try:
            r = await client.get(f"{worker.url}/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("pipeline_loaded"):
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2.0)
    logger.error("Worker %s did not become ready within %.0fs", worker.label, timeout)
    return False


def _terminate_workers(workers: List[Worker]) -> None:
    for w in workers:
        if w.proc.poll() is None:
            w.proc.terminate()
    for w in workers:
        try:
            w.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Worker %s did not exit; killing", w.label)
            w.proc.kill()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert _state is not None
    cfg = _state.cfg
    _state.client = httpx.AsyncClient()

    # Staggered startup: bring up the first worker alone and wait for it to load.
    # That one worker populates every shared cache (HF weights *and* the
    # torch.hub / DINOv2 download) while the others aren't running yet — so on a
    # cold machine there's a single download pass instead of N workers racing to
    # fetch the same files. Once it's ready, the rest load straight from cache.
    ports = [cfg.worker_base_port + i for i in range(len(cfg.gpus))]
    first = _spawn_worker(cfg, cfg.gpus[0], ports[0])
    _state.workers.append(first)
    logger.info("Bringing up %s first to warm the weight caches...", first.label)
    first_ok = await _await_ready(_state.client, first, cfg.worker_startup_timeout)

    rest = [_spawn_worker(cfg, gpu, port) for gpu, port in zip(cfg.gpus[1:], ports[1:])]
    _state.workers.extend(rest)
    if rest:
        logger.info("Caches warm; starting remaining %d worker(s) from cache", len(rest))
    rest_ok = await asyncio.gather(
        *(_await_ready(_state.client, w, cfg.worker_startup_timeout) for w in rest)
    )

    ready = [w for w, ok in zip(_state.workers, [first_ok, *rest_ok]) if ok]
    for w in ready:
        _state.idle.put_nowait(w)

    if not ready:
        _terminate_workers(_state.workers)
        await _state.client.aclose()
        raise RuntimeError("No workers became ready; aborting gateway startup.")
    if len(ready) < len(_state.workers):
        logger.warning("%d/%d workers ready; serving with reduced capacity.", len(ready), len(_state.workers))
    logger.info("Gateway ready: %d worker(s) serving on %s", len(ready), [w.label for w in ready])

    try:
        yield
    finally:
        logger.info("Shutting down workers...")
        _terminate_workers(_state.workers)
        if _state.client is not None:
            await _state.client.aclose()


app = FastAPI(
    title="RecGen Inference Gateway",
    description="Dispatches single-view reconstruction requests across one worker per GPU.",
    version="0.1.0",
    lifespan=lifespan,
)


def _alive_workers() -> List[Worker]:
    return [w for w in _state.workers if w.alive]  # type: ignore[union-attr]


@app.get("/health")
def health() -> JSONResponse:
    assert _state is not None
    alive = _alive_workers()
    return JSONResponse({
        "status": "ok" if alive else "error",
        # Kept for client compatibility: true once at least one GPU can serve.
        "pipeline_loaded": len(alive) > 0,
        "checkpoint": _state.cfg.checkpoint,
        "workers_total": len(_state.workers),
        "workers_alive": len(alive),
        "workers_idle": _state.idle.qsize(),
    })


async def _acquire(timeout: float) -> Worker:
    assert _state is not None
    try:
        return await asyncio.wait_for(_state.idle.get(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=f"all GPUs busy; no worker free within {timeout:.0f}s",
        )


@app.post("/generate")
async def generate_endpoint(request: Request) -> Response:
    """Proxy one /generate call to an idle GPU worker, queuing if all are busy."""
    assert _state is not None and _state.client is not None
    cfg = _state.cfg
    body = await request.body()
    content_type = request.headers.get("content-type", "application/x-msgpack")

    tried: List[str] = []
    # Retry once on a *different* worker if the worker process dies. A crash
    # (segfault / OOM-kill) OR a poisoned-CUDA-context self-exit both drop the
    # connection here; either way we recycle that worker (background respawn) and
    # retry elsewhere. Recoverable inference errors come back as an HTTP error
    # *status* from a live worker and are forwarded verbatim, not retried.
    for _ in range(2):
        worker = await _acquire(cfg.queue_timeout)
        t0 = time.perf_counter()
        try:
            resp = await _state.client.post(
                f"{worker.url}/generate",
                content=body,
                headers={"Content-Type": content_type},
                timeout=cfg.request_timeout,
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
            tried.append(worker.label)
            logger.error("Worker %s connection failed (%s); recycling", worker.label, type(e).__name__)
            _schedule_recycle(worker)
            continue
        # Worker answered. If its process has since exited (e.g. it self-exited
        # right after responding due to a poisoned context), recycle rather than
        # returning a dead worker to the pool.
        if worker.proc.poll() is not None:
            logger.error("Worker %s exited after responding (code %s); recycling", worker.label, worker.proc.returncode)
            _schedule_recycle(worker)
        else:
            _state.idle.put_nowait(worker)
        logger.info(
            "/generate -> %s  status=%d  proxy=%.3fs  req_bytes=%d resp_bytes=%d",
            worker.label, resp.status_code, time.perf_counter() - t0, len(body), len(resp.content),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/x-msgpack"),
        )

    raise HTTPException(status_code=502, detail=f"all attempted workers failed: {tried}")


# ---------------------------------------------------------------------------
# GPU discovery + CLI
# ---------------------------------------------------------------------------

def _detect_gpus() -> List[str]:
    """Best-effort GPU list: CUDA_VISIBLE_DEVICES, else `nvidia-smi -L`."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if n:
            return [str(i) for i in range(n)]
    except (OSError, subprocess.CalledProcessError):
        pass
    return []


def _parse_gpus(raw: Optional[str]) -> List[str]:
    if raw:
        return [g.strip() for g in raw.split(",") if g.strip()]
    gpus = _detect_gpus()
    if not gpus:
        raise SystemExit(
            "Could not detect any GPUs. Pass --gpus explicitly, e.g. --gpus 0,1,2,3"
        )
    return gpus


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Multi-GPU gateway for RecGen inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18324, help="Public gateway port")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES or all detected). One worker per id.",
    )
    parser.add_argument(
        "--worker-base-port",
        type=int,
        default=18401,
        help="First internal worker port; worker i listens on base+i.",
    )
    parser.add_argument(
        "--checkpoint",
        default="recgen_base.multiview_stereo",
        help="RecGen checkpoint name (passed to every worker).",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=300.0,
        help="Max seconds a request waits for a free GPU before 503.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Max seconds for a single worker /generate to complete.",
    )
    parser.add_argument(
        "--worker-startup-timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for each worker's pipeline to load at startup.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    args.gpus = _parse_gpus(args.gpus)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s [gateway] %(message)s",
    )
    # httpx logs every proxied request at INFO — far too chatty for the gateway.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    global _state
    _state = GatewayState(cfg=args)
    logger.info("Starting gateway for GPUs %s (workers on ports %d..%d)",
                args.gpus, args.worker_base_port, args.worker_base_port + len(args.gpus) - 1)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
