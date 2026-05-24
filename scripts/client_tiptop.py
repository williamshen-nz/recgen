"""HTTP client demo: send a tiptop capture to the RecGen server.

Spin up the server in one shell, then point this client at it::

    pixi run serve                                   # shell A: gateway on :18324
    pixi run python scripts/client_tiptop.py \\      # shell B: client
        --root data/tiptop/2026-04-29_10-29-39

The client itself does NOT import torch / recgen_inference — it only needs
opencv (for capture loading), numpy, aiohttp, msgpack, and trimesh — so
you can run it on a laptop while the server holds the GPU.

It is fully ``async`` (one shared ``aiohttp.ClientSession``), mirroring the
real downstream client: the objects in a capture are POSTed **concurrently**,
bounded by ``--concurrency`` (default 4) via a semaphore. The gateway hands each
request to an idle GPU and queues the rest, so a multi-object capture fans out
across all GPUs instead of running serially. Point ``--url`` at the gateway
(default ``:18324``); a single-GPU ``serve-worker`` works too, it just
serializes. Wire format is msgpack (numpy arrays via msgpack-numpy) in both
directions — no PNG round-trip. For each object we save the returned mesh + pose;
optionally merge per-object meshes into a scene OBJ.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Sequence

import aiohttp
import cv2
import msgpack
import msgpack_numpy
import numpy as np
import trimesh

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# Capture loading (mirrors scripts/run_tiptop.py, kept local to avoid pulling
# in the torch-heavy recgen_inference imports)
# ---------------------------------------------------------------------------

def load_capture(root: Path):
    rgb = cv2.imread(str(root / "rgb.png"))
    if rgb is None:
        raise FileNotFoundError(root / "rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(str(root / "perception" / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(root / "perception" / "depth.png")

    masks = np.load(root / "perception" / "masks.npz")["arr_0"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    with open(root / "perception" / "bboxes.json") as f:
        bboxes = json.load(f)

    with open(root / "perception" / "intrinsics.json") as f:
        K = np.array(json.load(f)["intrinsics"], dtype=np.float64)

    if len(bboxes) != masks.shape[0]:
        raise ValueError(f"bboxes ({len(bboxes)}) and masks ({masks.shape[0]}) length mismatch")
    return rgb, depth, masks, bboxes, K


def select_indices(bboxes: Sequence[dict], labels: Sequence[str] | None) -> list[int]:
    if not labels:
        return list(range(len(bboxes)))
    wanted = set(labels)
    selection = [i for i, b in enumerate(bboxes) if b["label"] in wanted]
    missing = wanted - {bboxes[i]["label"] for i in selection}
    if missing:
        print(f"[client] WARNING: labels not found in bboxes.json: {sorted(missing)}")
    return selection


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

async def post_generate(
    session: aiohttp.ClientSession,
    url: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    seed: int,
    timeout: float,
    target_faces: int | None = None,
) -> dict:
    """POST one object to /generate. Returns the unpacked msgpack payload.

    Mirrors the real async client: one shared ``aiohttp.ClientSession`` is used
    for all requests so many can be in flight at once over the event loop.
    """
    payload = {
        "rgb": np.ascontiguousarray(rgb),
        "depth": np.ascontiguousarray(depth),
        "mask": np.ascontiguousarray((mask > 0).astype(np.uint8)),
        "intrinsics": np.ascontiguousarray(K, dtype=np.float64),
        "seed": int(seed),
    }
    if target_faces is not None:
        payload["target_faces"] = int(target_faces)
    body = msgpack.packb(payload, use_bin_type=True)
    async with session.post(
        f"{url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        return msgpack.unpackb(await resp.read(), raw=False)


def payload_to_trimesh(payload: dict) -> trimesh.Trimesh:
    kwargs = {"vertices": payload["vertices"], "faces": payload["faces"], "process": False}
    if "vertex_colors" in payload:
        kwargs["vertex_colors"] = payload["vertex_colors"]
    return trimesh.Trimesh(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RecGen HTTP client demo on tiptop data")
    p.add_argument("--root", required=True, help="Tiptop capture directory")
    p.add_argument("--url", default="http://localhost:18324", help="RecGen server base URL")
    p.add_argument("--out", default=None, help="Output dir (default: <root>/recgen_client_outputs)")
    p.add_argument("--labels", nargs="*", default=None, help="Optional label allowlist")
    p.add_argument("--min-mask-pixels", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max in-flight requests; set near the server's GPU count.",
    )
    p.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="If set, server-side decimate each mesh to roughly this many faces.",
    )
    p.add_argument("--no-merge", action="store_true", help="Skip merged scene mesh export")
    return p.parse_args()


async def _check_health(session: aiohttp.ClientSession, url: str) -> None:
    try:
        async with session.get(
            f"{url.rstrip('/')}/health", timeout=aiohttp.ClientTimeout(total=5)
        ) as r:
            r.raise_for_status()
            body = await r.json()
    except Exception as e:
        raise SystemExit(f"[client] server health check failed at {url}: {e}")
    print(f"[client] server health: {body}")
    if not body.get("pipeline_loaded"):
        print("[client] WARNING: pipeline not loaded yet — first request will block.")


async def _process_object(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, args, rgb, depth, K,
    out_root: Path, i: int, label: str, mask: np.ndarray,
) -> dict:
    """POST one object, save its mesh + pose, return a summary entry.

    ``sem`` bounds how many requests are in flight at once. The decoded mesh is
    stashed under ``_mesh`` for the optional merge step (popped off before the
    entry is written to summary.json).
    """
    obj_dir = out_root / f"{i:02d}_{label}"
    try:
        # Time only the request itself, not the wait for a free semaphore slot,
        # so round_trip_s reflects server latency rather than client-side queuing.
        async with sem:
            t0 = time.perf_counter()
            payload = await post_generate(
                session, args.url, rgb, depth, mask, K, args.seed, args.timeout, args.target_faces
            )
            elapsed = time.perf_counter() - t0
    except aiohttp.ClientResponseError as e:
        print(f"[client] [{i:02d}] {label}: HTTP {e.status} — {e.message}")
        return {"index": i, "label": label, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[client] [{i:02d}] {label}: FAILED ({type(e).__name__}: {e})")
        return {"index": i, "label": label, "status": "failed", "error": str(e)}

    # Decode/export/save outside the request try so a malformed payload or a
    # filesystem error fails just this object — without it, the exception would
    # propagate out of gather() and cancel every other in-flight request.
    try:
        mesh = payload_to_trimesh(payload)
        obj_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(obj_dir / "mesh.obj")
        np.save(obj_dir / "pose_matrix.npy", payload["pose_matrix"])
    except Exception as e:
        print(f"[client] [{i:02d}] {label}: FAILED post-processing ({type(e).__name__}: {e})")
        return {"index": i, "label": label, "status": "failed", "error": str(e)}

    print(f"[client] [{i:02d}] {label}: {elapsed:.2f}s  "
          f"({mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces)")
    return {
        "index": i,
        "label": label,
        "status": "ok",
        "round_trip_s": elapsed,
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "pose_matrix": payload["pose_matrix"].tolist(),
        "_mesh": mesh,
    }


async def _run(args) -> None:
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve() if args.out else root / "recgen_client_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    rgb, depth, masks, bboxes, K = load_capture(root)
    selection = select_indices(bboxes, args.labels)

    # Filter to objects with enough mask pixels; the rest are skipped up front.
    todo: list[tuple[int, str, np.ndarray]] = []
    for i in selection:
        label = bboxes[i]["label"]
        mask = masks[i].astype(np.uint8)
        n_px = int(mask.sum())
        if n_px < args.min_mask_pixels:
            print(f"[client] [{i:02d}] {label}: SKIP ({n_px} mask pixels)")
            continue
        todo.append((i, label, mask))

    sem = asyncio.Semaphore(max(1, args.concurrency))
    t_all = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        await _check_health(session, args.url)
        print(f"[client] Loaded {root}: rgb={rgb.shape}, depth={depth.shape}, "
              f"{masks.shape[0]} objects")
        print(f"[client] Dispatching {len(todo)} object(s) with concurrency={args.concurrency}")
        # Fire all objects onto the event loop at once; the semaphore caps how
        # many are actually in flight, and the gateway fans them across GPUs.
        results = await asyncio.gather(*(
            _process_object(session, sem, args, rgb, depth, K, out_root, i, label, mask)
            for (i, label, mask) in todo
        ))

    results.sort(key=lambda s: s["index"])
    posed_meshes = [s.pop("_mesh") for s in results if s["status"] == "ok"]
    summary = results
    print(f"[client] All requests done in {time.perf_counter() - t_all:.2f}s wall")

    with open(out_root / "summary.json", "w") as f:
        json.dump({"server_url": args.url, "objects": summary}, f, indent=2)
    print(f"[client] Wrote summary to {out_root / 'summary.json'}")

    ok = [s for s in summary if s["status"] == "ok"]
    if ok:
        times = np.array([s["round_trip_s"] for s in ok])
        print(f"[client] {len(ok)} ok / {len(summary) - len(ok)} failed; "
              f"round-trip mean {times.mean():.2f}s (min {times.min():.2f}s, max {times.max():.2f}s)")

    if posed_meshes and not args.no_merge:
        scene_path = out_root / "scene_mesh.obj"
        trimesh.util.concatenate(posed_meshes).export(scene_path)
        print(f"[client] Wrote merged scene mesh: {scene_path}")


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
