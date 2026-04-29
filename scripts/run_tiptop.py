"""Run RecGen single-view inference on a tiptop capture.

A tiptop capture has the layout:

    <root>/
      rgb.png                       # BGR uint8, HxWx3
      perception/
        depth.png                   # uint16 mm
        masks.npz                   # bool, [N, 1, H, W] in bboxes.json order
        bboxes.json                 # list of {label, box_2d}
        intrinsics.json             # {"intrinsics": 3x3}

Loads the pipeline once and reconstructs every object (or a filtered subset),
then merges the per-object outputs into a single scene mesh, scene Gaussian
splat, and a multi-object overlay on the input RGB.

Examples:
    pixi run python scripts/run_tiptop.py --root data/tiptop/2026-04-29_10-29-39 --save-splat
    pixi run python scripts/run_tiptop.py --root data/tiptop/tiptop_run \\
        --labels red_block_with_letter_O blue_striped_plate
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import trimesh

from recgen_inference import build_recgen, generate


# ---------------------------------------------------------------------------
# Capture loading
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
        masks = masks[:, 0]  # -> [N, H, W]

    with open(root / "perception" / "bboxes.json") as f:
        bboxes = json.load(f)

    with open(root / "perception" / "intrinsics.json") as f:
        K = np.array(json.load(f)["intrinsics"], dtype=np.float64)

    if len(bboxes) != masks.shape[0]:
        raise ValueError(
            f"bboxes ({len(bboxes)}) and masks ({masks.shape[0]}) length mismatch"
        )
    return rgb, depth, masks, bboxes, K


def select_indices(bboxes: Sequence[dict], labels: Sequence[str] | None) -> list[int]:
    if not labels:
        return list(range(len(bboxes)))
    wanted = set(labels)
    selection = [i for i, b in enumerate(bboxes) if b["label"] in wanted]
    missing = wanted - {bboxes[i]["label"] for i in selection}
    if missing:
        print(f"[tiptop] WARNING: labels not found in bboxes.json: {sorted(missing)}")
    return selection


# ---------------------------------------------------------------------------
# Per-object inference
# ---------------------------------------------------------------------------

def run_object(pipeline, rgb, depth, K, mask, label, idx, out_root, args):
    """Run one object. Returns (summary_dict, posed_mesh_or_None, posed_ply_or_None)."""
    n_px = int(mask.sum())
    obj_dir = out_root / f"{idx:02d}_{label}"

    if n_px < args.min_mask_pixels:
        print(f"[tiptop] [{idx:02d}] {label}: SKIP (only {n_px} mask pixels)")
        return None, None, None

    print(f"[tiptop] [{idx:02d}] {label}: {n_px} mask pixels -> {obj_dir}")
    try:
        t0 = time.perf_counter()
        result = generate(pipeline, image=rgb, depth=depth, mask=mask,
                          intrinsics=K, seed=args.seed)
        inference_s = time.perf_counter() - t0
    except Exception as e:
        print(f"[tiptop] [{idx:02d}] {label}: FAILED ({type(e).__name__}: {e})")
        return ({"index": idx, "label": label, "status": "failed", "error": str(e)},
                None, None)

    obj_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result.save(obj_dir, save_splat=args.save_splat, save_glb=args.save_glb)
    save_s = time.perf_counter() - t0
    print(f"[tiptop] [{idx:02d}] {label}: inference {inference_s:.2f}s, save {save_s:.2f}s")

    posed_ply = obj_dir / "posed_gaussian.ply" if args.save_splat else None
    if posed_ply is not None and not posed_ply.exists():
        posed_ply = None

    return (
        {
            "index": idx,
            "label": label,
            "status": "ok",
            "n_mask_pixels": n_px,
            "n_vertices": int(result.mesh.vertices.shape[0]),
            "n_faces": int(result.mesh.faces.shape[0]),
            "inference_s": inference_s,
            "save_s": save_s,
            "pose_matrix": result.pose_matrix.tolist(),
        },
        result.mesh.copy(),
        posed_ply,
    )


# ---------------------------------------------------------------------------
# Scene-level merging
# ---------------------------------------------------------------------------

def merge_meshes(meshes: list, out_path: Path) -> None:
    """Concatenate camera-frame meshes into a single OBJ, preserving vertex colors."""
    trimesh.util.concatenate(meshes).export(out_path)


def merge_gaussian_plys(paths: list, out_path: Path) -> None:
    """Concatenate Gaussian-splat PLYs by stacking their vertex elements.

    All PLYs share the exporter's vertex schema, so we just append rows.
    """
    from plyfile import PlyData, PlyElement

    chunks, dtype = [], None
    for p in paths:
        v = PlyData.read(str(p))["vertex"].data
        if dtype is None:
            dtype = v.dtype
        elif v.dtype != dtype:
            raise ValueError(f"PLY vertex dtype mismatch in {p}: {v.dtype} vs {dtype}")
        chunks.append(v)
    PlyData([PlyElement.describe(np.concatenate(chunks), "vertex")]).write(str(out_path))


def _palette(n: int) -> np.ndarray:
    """n distinct RGB colors via HSV-spaced hues."""
    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = (np.arange(n) * (179 / max(n, 1))).astype(np.uint8)
    hsv[0, :, 1] = 200
    hsv[0, :, 2] = 230
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def render_scene_overlay(meshes: list, K: np.ndarray, rgb: np.ndarray,
                         labels: list, opacity: float = 0.6) -> np.ndarray:
    """Project multiple camera-frame meshes into `rgb` with cross-object occlusion.

    Each mesh gets its own color from a hue-spaced palette; faces from all meshes
    are sorted together by depth so a closer object correctly occludes a farther one.
    """
    H, W = rgb.shape[:2]
    K = np.asarray(K, dtype=np.float64)
    colors = _palette(len(meshes))

    pix_chunks, face_chunks, z_chunks, valid_chunks, color_chunks = [], [], [], [], []
    vert_offset = 0
    for mesh, color in zip(meshes, colors):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        z = verts[:, 2]
        valid = z > 1e-3
        z_safe = np.where(valid, z, 1.0)
        u = K[0, 0] * verts[:, 0] / z_safe + K[0, 2]
        v = K[1, 1] * verts[:, 1] / z_safe + K[1, 2]

        v0, v1, v2 = (verts[mesh.faces[:, k]] for k in range(3))
        normals = np.cross(v1 - v0, v2 - v0)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(nlen > 0, nlen, 1.0)
        diffuse = np.clip(0.25 + 0.75 * np.abs(normals[:, 2]), 0, 1)

        pix_chunks.append(np.stack([u, v], axis=1))
        face_chunks.append(np.asarray(mesh.faces) + vert_offset)
        z_chunks.append(z[mesh.faces].mean(axis=1))
        valid_chunks.append(valid[mesh.faces].all(axis=1))
        color_chunks.append((color[None, :] * diffuse[:, None]).astype(np.uint8))
        vert_offset += verts.shape[0]

    pixels = np.concatenate(pix_chunks)
    faces = np.concatenate(face_chunks)
    face_z = np.concatenate(z_chunks)
    face_valid = np.concatenate(valid_chunks)
    face_color = np.concatenate(color_chunks)

    # Painter's algorithm across all objects for correct cross-object occlusion.
    rendered = np.zeros((H, W, 3), dtype=np.uint8)
    mask_img = np.zeros((H, W), dtype=np.uint8)
    for fi in np.argsort(-face_z):
        if not face_valid[fi]:
            continue
        pts = pixels[faces[fi]].astype(np.int32)
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= W:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= H:
            continue
        cv2.fillPoly(rendered, [pts.reshape(-1, 1, 2)], face_color[fi].tolist())
        cv2.fillPoly(mask_img, [pts.reshape(-1, 1, 2)], 255)

    bg = rgb.astype(np.float32) / 255.0
    mf = (mask_img > 0).astype(np.float32)[:, :, None]
    comp = bg * (1 - opacity * mf) + (rendered.astype(np.float32) / 255.0) * opacity * mf
    overlay = np.clip(comp * 255, 0, 255).astype(np.uint8)
    _draw_legend(overlay, labels, colors)
    return overlay


def _draw_legend(overlay: np.ndarray, labels: list, colors: np.ndarray) -> None:
    pad, sw, row_h, legend_w = 8, 18, 22, 240
    legend_h = pad * 2 + row_h * len(labels)
    if legend_h > overlay.shape[0] or legend_w > overlay.shape[1]:
        return
    legend = np.full((legend_h, legend_w, 3), 32, dtype=np.uint8)
    for i, (label, color) in enumerate(zip(labels, colors)):
        y = pad + i * row_h
        legend[y:y + sw, pad:pad + sw] = color
        cv2.putText(legend, str(label)[:30], (pad + sw + 8, y + sw - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    overlay[:legend_h, :legend_w] = legend


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def report_stats(summary: list, pipeline_load_s: float) -> dict:
    ok = [s for s in summary if s["status"] == "ok"]
    failed = [s for s in summary if s["status"] != "ok"]
    times = np.array([s["inference_s"] for s in ok], dtype=np.float64)

    stats = {"pipeline_load_s": pipeline_load_s, "n_ok": len(ok), "n_failed": len(failed)}
    if times.size:
        stats.update({
            "inference_total_s": float(times.sum()),
            "inference_mean_s": float(times.mean()),
            "inference_median_s": float(np.median(times)),
            "inference_min_s": float(times.min()),
            "inference_max_s": float(times.max()),
            "inference_std_s": float(times.std()),
        })

    print("")
    print("[tiptop] === inference timing (excludes pipeline load) ===")
    print(f"[tiptop] objects: {len(ok)} ok, {len(failed)} failed")
    print(f"[tiptop] pipeline load: {pipeline_load_s:.2f}s")
    if times.size:
        print(f"[tiptop] inference total: {times.sum():.2f}s")
        print(f"[tiptop] inference mean:   {times.mean():.2f}s "
              f"(median {np.median(times):.2f}s, std {times.std():.2f}s)")
        print(f"[tiptop] inference range:  {times.min():.2f}s - {times.max():.2f}s")
        slowest = max(ok, key=lambda s: s["inference_s"])
        fastest = min(ok, key=lambda s: s["inference_s"])
        print(f"[tiptop] slowest: [{slowest['index']:02d}] {slowest['label']} "
              f"({slowest['inference_s']:.2f}s)")
        print(f"[tiptop] fastest: [{fastest['index']:02d}] {fastest['label']} "
              f"({fastest['inference_s']:.2f}s)")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RecGen single-view inference on tiptop data")
    p.add_argument("--root", required=True, help="Tiptop capture directory (see module docstring)")
    p.add_argument("--out", default=None,
                   help="Output directory. Defaults to <root>/recgen_outputs.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Optional label allowlist (matches bboxes.json 'label'). Default: all.")
    p.add_argument("--min-mask-pixels", type=int, default=200,
                   help="Skip objects with fewer than this many mask pixels.")
    p.add_argument("--checkpoint", default="recgen_base.multiview_stereo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-splat", action="store_true")
    p.add_argument("--save-glb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve() if args.out else root / "recgen_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    rgb, depth, masks, bboxes, K = load_capture(root)
    print(f"[tiptop] Loaded capture from {root}: rgb={rgb.shape}, depth={depth.shape}, "
          f"{masks.shape[0]} objects")
    print(f"[tiptop] Intrinsics:\n{K}")

    selection = select_indices(bboxes, args.labels)

    print(f"[tiptop] Loading pipeline: {args.checkpoint}")
    t0 = time.perf_counter()
    pipeline = build_recgen.build(args.checkpoint)
    pipeline_load_s = time.perf_counter() - t0
    print(f"[tiptop] Pipeline loaded in {pipeline_load_s:.2f}s")

    summary: list[dict] = []
    posed_meshes: list[trimesh.Trimesh] = []
    posed_labels: list[str] = []
    posed_ply_paths: list[Path] = []
    for i in selection:
        label = bboxes[i]["label"]
        mask = masks[i].astype(np.uint8)
        entry, mesh, ply = run_object(pipeline, rgb, depth, K, mask, label, i, out_root, args)
        if entry is None:
            continue
        summary.append(entry)
        if mesh is not None:
            posed_meshes.append(mesh)
            posed_labels.append(f"{i:02d} {label}")
        if ply is not None:
            posed_ply_paths.append(ply)

    stats = report_stats(summary, pipeline_load_s)

    with open(out_root / "summary.json", "w") as f:
        json.dump({"stats": stats, "objects": summary}, f, indent=2)
    print(f"[tiptop] Wrote summary to {out_root / 'summary.json'}")

    if posed_meshes:
        merge_meshes(posed_meshes, out_root / "scene_mesh.obj")
        print(f"[tiptop] Wrote merged scene mesh ({len(posed_meshes)} objects)")

        overlay = render_scene_overlay(posed_meshes, K, rgb, posed_labels)
        cv2.imwrite(str(out_root / "scene_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[tiptop] Wrote scene overlay")

    if posed_ply_paths:
        merge_gaussian_plys(posed_ply_paths, out_root / "scene_gaussian.ply")
        print(f"[tiptop] Wrote merged scene gaussian ({len(posed_ply_paths)} objects)")


if __name__ == "__main__":
    main()
