"""HTTP client demo: send a tiptop capture to the RecGen server.

Spin up the server in one shell, then point this client at it::

    pixi run serve                                   # shell A: starts server on :18324
    pixi run python scripts/client_tiptop.py \\      # shell B: client
        --root data/tiptop/2026-04-29_10-29-39

The client itself does NOT import torch / recgen_inference — it only needs
opencv (for capture loading), numpy, requests, msgpack, and trimesh — so
you can run it on a laptop while the server holds the GPU.

Wire format is msgpack (numpy arrays via msgpack-numpy) in both directions —
no PNG round-trip. For each object in the capture we POST one /generate
request and save the returned mesh + pose. Optional: merge per-object
meshes into a scene OBJ.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import requests
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

def post_generate(
    url: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    seed: int,
    timeout: float,
) -> dict:
    """POST one object to /generate. Returns the unpacked msgpack payload."""
    body = msgpack.packb(
        {
            "rgb": np.ascontiguousarray(rgb),
            "depth": np.ascontiguousarray(depth),
            "mask": np.ascontiguousarray((mask > 0).astype(np.uint8)),
            "intrinsics": np.ascontiguousarray(K, dtype=np.float64),
            "seed": int(seed),
        },
        use_bin_type=True,
    )
    r = requests.post(
        f"{url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=timeout,
    )
    r.raise_for_status()
    return msgpack.unpackb(r.content, raw=False)


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
    p.add_argument("--no-merge", action="store_true", help="Skip merged scene mesh export")
    return p.parse_args()


def _check_health(url: str) -> None:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"[client] server health check failed at {url}: {e}")
    body = r.json()
    print(f"[client] server health: {body}")
    if not body.get("pipeline_loaded"):
        print("[client] WARNING: pipeline not loaded yet — first request will block.")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve() if args.out else root / "recgen_client_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    _check_health(args.url)
    rgb, depth, masks, bboxes, K = load_capture(root)
    print(f"[client] Loaded {root}: rgb={rgb.shape}, depth={depth.shape}, {masks.shape[0]} objects")

    selection = select_indices(bboxes, args.labels)

    summary: list[dict] = []
    posed_meshes: list[trimesh.Trimesh] = []
    for i in selection:
        label = bboxes[i]["label"]
        mask = masks[i].astype(np.uint8)
        n_px = int(mask.sum())
        obj_dir = out_root / f"{i:02d}_{label}"

        if n_px < args.min_mask_pixels:
            print(f"[client] [{i:02d}] {label}: SKIP ({n_px} mask pixels)")
            continue

        print(f"[client] [{i:02d}] {label}: POSTing ({n_px} mask pixels)")
        try:
            t0 = time.perf_counter()
            payload = post_generate(args.url, rgb, depth, mask, K, args.seed, args.timeout)
            elapsed = time.perf_counter() - t0
        except requests.HTTPError as e:
            body = e.response.text[:500] if e.response is not None else ""
            print(f"[client] [{i:02d}] {label}: HTTP {e.response.status_code if e.response else '?'} — {body}")
            summary.append({"index": i, "label": label, "status": "failed", "error": str(e)})
            continue
        except Exception as e:
            print(f"[client] [{i:02d}] {label}: FAILED ({type(e).__name__}: {e})")
            summary.append({"index": i, "label": label, "status": "failed", "error": str(e)})
            continue

        mesh = payload_to_trimesh(payload)
        obj_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(obj_dir / "mesh.obj")
        np.save(obj_dir / "pose_matrix.npy", payload["pose_matrix"])

        print(f"[client] [{i:02d}] {label}: {elapsed:.2f}s  ({mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces)")
        summary.append({
            "index": i,
            "label": label,
            "status": "ok",
            "round_trip_s": elapsed,
            "n_vertices": int(mesh.vertices.shape[0]),
            "n_faces": int(mesh.faces.shape[0]),
            "pose_matrix": payload["pose_matrix"].tolist(),
        })
        posed_meshes.append(mesh)

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


if __name__ == "__main__":
    main()
