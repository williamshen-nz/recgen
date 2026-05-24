"""FastAPI server for RecGen single-view inference.

The pipeline is loaded once at startup (via FastAPI's lifespan) so the first
request does not pay the model-load cost. Inputs and outputs are msgpack
blobs carrying numpy arrays directly (via msgpack-numpy) — no PNG round-trip.

Run with::

    pixi run serve
    # or
    pixi run python scripts/server.py --host 0.0.0.0 --port 18324
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import msgpack
import msgpack_numpy
import numpy as np
import torch
import trimesh
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from recgen_inference import build_recgen, generate
from recgen_inference._result import RecGenResult

msgpack_numpy.patch()  # teach msgpack to encode/decode np.ndarray natively

logger = logging.getLogger("recgen_inference.server")

# A single worker thread: the heavy generate() call runs off the event loop
# (so /health stays responsive) but generations are serialized within this
# process, so the one pipeline/GPU is never driven by two threads at once —
# even if this worker is hit concurrently outside the gateway.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _cuda_context_broken() -> bool:
    """Probe whether this process's CUDA context is still usable.

    Some failures (device-side assert, illegal memory access) corrupt the CUDA
    context: the process keeps running but every subsequent op raises. A clean
    error (bad input, CPU-side exception, recoverable OOM) leaves the context
    intact. We test with a trivial op so the caller can tell "retry me later" /
    "I'm wedged, kill me" apart.
    """
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.synchronize()
        _ = (torch.zeros(8, device="cuda") + 1).sum().item()
        torch.cuda.synchronize()
        return False
    except Exception:
        return True


def _hard_exit(code: int = 70) -> None:
    """Flush logs and terminate the process so the gateway respawns this worker."""
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    os._exit(code)

_state: Dict[str, Any] = {"pipeline": None, "checkpoint": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = _state["checkpoint"]
    device = _state["device"]
    logger.info("Loading RecGen pipeline (%s) on %s", checkpoint, device)
    _state["pipeline"] = build_recgen.build(checkpoint, device=device)
    logger.info("Pipeline ready.")
    try:
        yield
    finally:
        _state["pipeline"] = None


app = FastAPI(
    title="RecGen Inference",
    description="HTTP wrapper around recgen_inference: single-view RGB-D 3D reconstruction.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_array(payload: Dict[str, Any], key: str, *, ndim: int | tuple[int, ...]) -> np.ndarray:
    if key not in payload:
        raise HTTPException(status_code=400, detail=f"missing field: {key!r}")
    arr = payload[key]
    if not isinstance(arr, np.ndarray):
        raise HTTPException(
            status_code=400,
            detail=f"field {key!r} must be a numpy array (sent via msgpack-numpy); got {type(arr).__name__}",
        )
    expected = (ndim,) if isinstance(ndim, int) else tuple(ndim)
    if arr.ndim not in expected:
        raise HTTPException(
            status_code=400,
            detail=f"field {key!r} must have ndim in {expected}; got shape {arr.shape}",
        )
    return arr


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation of ``mesh`` to roughly ``target_faces``.

    Uses ``trimesh.Trimesh.simplify_quadric_decimation``, which wraps the
    ``fast-simplification`` package (Sven Forstmann's algorithm). Same quadric
    edge-collapse family as the pyvista/VTK path in
    ``recgen_modules.utils.postprocessing_utils.postprocess_mesh`` but typically
    10–100× faster at high reduction ratios. Vertex colors collapse to the mean
    of the original mesh's vertex colors applied uniformly — cheap and matches
    the uniform-color convention used by downstream consumers that don't need
    per-vertex texture.
    """
    n_faces = len(mesh.faces)
    if target_faces >= n_faces:
        return mesh

    decimated = mesh.simplify_quadric_decimation(face_count=int(target_faces))

    orig_colors = getattr(mesh.visual, "vertex_colors", None)
    if orig_colors is not None and len(orig_colors) == len(mesh.vertices):
        mean_color = np.asarray(orig_colors, dtype=np.float64).mean(axis=0).round().astype(np.uint8)
        decimated.visual.vertex_colors = np.tile(mean_color, (len(decimated.vertices), 1))
    return decimated


def _pack_result(result: RecGenResult) -> bytes:
    """Serialize the minimal payload: pose + camera-frame mesh."""
    mesh = result.mesh
    payload: Dict[str, Any] = {
        "pose_matrix": np.ascontiguousarray(result.pose_matrix, dtype=np.float64),
        "pose_quat": np.ascontiguousarray(result.pose_quat, dtype=np.float64),
        "vertices": np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        "faces": np.ascontiguousarray(mesh.faces, dtype=np.int32),
    }
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        payload["vertex_colors"] = np.ascontiguousarray(colors, dtype=np.uint8)
    return msgpack.packb(payload, use_bin_type=True)


def _run_generate(
    pipeline: Any,
    rgb_arr: np.ndarray,
    depth_arr: np.ndarray,
    mask_arr: np.ndarray,
    K: np.ndarray,
    seed: int,
    target_faces: int | None,
) -> tuple[bytes, float, float]:
    """Blocking inference + optional decimation + serialization.

    Returns ``(response_body, inference_s, decimation_s)``. Runs in a worker
    thread (see ``run_in_executor`` in the endpoint), never on the event loop.
    """
    t_inf = time.perf_counter()
    result = generate(
        pipeline,
        image=rgb_arr,
        depth=depth_arr,
        mask=mask_arr,
        intrinsics=K,
        seed=seed,
    )
    inference_s = time.perf_counter() - t_inf

    decimation_s = 0.0
    if target_faces is not None and target_faces > 0:
        t_dec = time.perf_counter()
        result.mesh = _decimate_mesh(result.mesh, target_faces)
        decimation_s = time.perf_counter() - t_dec

    return _pack_result(result), inference_s, decimation_s


def _ensure_pipeline():
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    return pipeline


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "pipeline_loaded": _state.get("pipeline") is not None,
        "checkpoint": _state.get("checkpoint"),
    })


@app.post("/generate")
async def generate_endpoint(request: Request) -> Response:
    """Single-view inference.

    Request body: msgpack blob (Content-Type: application/x-msgpack) with keys:
        - ``rgb``: (H, W, 3) uint8 RGB
        - ``depth``: (H, W) uint16 mm or float32 m
        - ``mask``: (H, W) any int dtype, non-zero = object
        - ``intrinsics``: (3, 3) float
        - ``seed``: int (optional, default 1)

    Response: msgpack blob with keys ``pose_matrix`` (4,4 float64),
    ``pose_quat`` (7 float64), ``vertices`` (N,3 float32),
    ``faces`` (M,3 int32), ``vertex_colors`` (N,4 uint8, optional).
    """
    pipeline = _ensure_pipeline()
    t_start = time.perf_counter()

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body; expected msgpack")
    try:
        payload = msgpack.unpackb(body, raw=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not msgpack-decode body: {e}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="msgpack body must decode to a dict")

    rgb_arr = _require_array(payload, "rgb", ndim=3)
    depth_arr = _require_array(payload, "depth", ndim=2)
    mask_arr = _require_array(payload, "mask", ndim=2)
    K = _require_array(payload, "intrinsics", ndim=2)
    if K.shape != (3, 3):
        raise HTTPException(status_code=400, detail=f"intrinsics must be (3,3); got {K.shape}")
    seed = int(payload.get("seed", 1))
    target_faces_raw = payload.get("target_faces")
    target_faces = int(target_faces_raw) if target_faces_raw is not None else None

    # The GPU work is synchronous and CPU/GPU-blocking. Run it in the dedicated
    # single-thread executor so this process's event loop stays responsive
    # (/health keeps answering, the gateway can probe liveness) while keeping
    # generations serialized — the one pipeline/GPU is never driven concurrently.
    loop = asyncio.get_running_loop()
    try:
        response_body, inference_s, decimation_s = await loop.run_in_executor(
            _executor,
            functools.partial(
                _run_generate,
                pipeline,
                rgb_arr,
                depth_arr,
                mask_arr,
                K,
                seed,
                target_faces,
            ),
        )
    except Exception as e:
        # If the CUDA context is now wedged, this worker is useless for every
        # future request — exit hard so the gateway respawns us with a fresh
        # context. Otherwise it was a recoverable error: report it and stay up.
        if await loop.run_in_executor(_executor, _cuda_context_broken):
            logger.error("CUDA context unusable after %s; exiting for respawn: %s", type(e).__name__, e)
            _hard_exit()
        logger.exception("generation failed (recoverable)")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    total_s = time.perf_counter() - t_start
    logger.info(
        "/generate ok  inference=%.3fs  decimation=%.3fs  total=%.3fs  rgb=%s depth=%s target_faces=%s req_bytes=%d resp_bytes=%d",
        inference_s,
        decimation_s,
        total_s,
        rgb_arr.shape,
        depth_arr.shape,
        target_faces,
        len(body),
        len(response_body),
    )
    return Response(content=response_body, media_type="application/x-msgpack")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve RecGen as a FastAPI app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18324)
    parser.add_argument(
        "--checkpoint",
        default="recgen_base.multiview_stereo",
        help="RecGen checkpoint name (see build_recgen.list_checkpoints())",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--worker-label",
        default=os.environ.get("RECGEN_WORKER_LABEL", ""),
        help="Optional prefix for log lines (set by the gateway, e.g. 'gpu0').",
    )
    args = parser.parse_args()

    _state["checkpoint"] = args.checkpoint
    _state["device"] = args.device

    prefix = f"[{args.worker_label}] " if args.worker_label else ""
    logging.basicConfig(
        level=args.log_level.upper(),
        format=f"%(asctime)s %(levelname)s %(name)s {prefix}%(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
