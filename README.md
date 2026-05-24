# RecGen Inference

This repository contains inference code for **RecGen**, a model for single-view and multi-view 3D reconstruction from RGB-D.
Given an RGB image, a depth map, an object mask, and camera intrinsics, RecGen produces a textured mesh, a Gaussian splat, and the object's 6-DoF pose in the camera frame.

Training code will be released in this same repository in a future update.

## Setup

RecGen depends on `torch`, a few 3D libraries, and two CUDA extensions (`spconv`, `diff-gaussian-rasterization`). We recommend [pixi](https://pixi.sh), which pins Python, CUDA, and PyTorch in a single lockfile:

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # one-time
pixi install                                    # CUDA 12.1
pixi install -e cu118                           # CUDA 11.8
```

Optional for speedups and additional visualizations (after `pixi install`):

```bash
pixi run post-install                 # flash-attn (falls back to xformers if the build fails)
pixi run build-nvdiffrast             # nvdiffrast, needed for mp4 vis of the rendering as well as glb (--save-glb)
pixi run build-gaussian-rasterizer    # diff_gaussian_rasterization, needed for turntable.mp4 and --save-glb
```

Without these, `mesh.obj`, `overlay.png`, and `metadata.json` are still produced; the turntable video is skipped.

<details>
<summary>Manual install (no pixi)</summary>

```bash
pip install -e .
bash scripts/setup_cuda.sh            # spconv + flash-attn + diff-gaussian-rasterization
bash scripts/setup_cuda.sh --all      # also builds nvdiffrast
```

You are responsible for providing a working PyTorch + CUDA toolchain.
</details>

<details>
<summary>Docker (no local install)</summary>

If you'd rather not install anything on the host, a CUDA 12.1 image is provided. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t recgen-inference .
docker run --rm --gpus all -v $PWD/data:/data recgen-inference
```

Or via Compose (persists the HuggingFace cache in a named volume):

```bash
docker compose run --rm recgen bash
```

</details>

## Quick Start

```python
from recgen_inference import build_recgen, generate

pipeline = build_recgen.build("recgen_base.multiview_stereo")

result = generate(
    pipeline,
    image=rgb,           # HxWx3 uint8
    depth=depth,         # HxW uint16 (mm) or float32 (m) 
    mask=mask,           # HxW uint8, non-zero = object
    intrinsics=K,        # 3x3
)

print(result.pose_matrix)           # (4, 4) float64 pose in the camera frame
print(result.pose_quat)              # (7,) [tx, ty, tz, qx, qy, qz, qw]
print(result.mesh.vertices.shape)    # final textured mesh

result.save("./out", save_splat=True, save_glb=False)
```

`result` is a [`RecGenResult`](recgen_inference/_result.py) dataclass that carries the final mesh, the raw object-frame mesh, the 6-DoF pose (both as a 4×4 matrix and a 7-vector).

See [notebooks/example.ipynb](notebooks/example.ipynb) for a runnable walkthrough, or run the CLI script:

```bash
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --out ./out --save-splat
```

`intrinsics.yaml` contains `fu`, `fv`, `pu`, `pv` as top-level keys.

## Multi-view Inference

```python
from recgen_inference import generate_multiview

result = generate_multiview(
    pipeline,
    anchor_view={"rgb": rgb0, "depth": d0, "mask": m0, "camera_intrinsics": K},
    second_views=[
        {"rgb": rgb1, "depth": d1, "mask": m1, "camera_intrinsics": K},
    ],
)
```

The result is expressed in the anchor view's camera frame.

## Input Format

- **RGB** — any common format readable by OpenCV (PNG/JPG).
- **Depth** — 16-bit PNG in millimeters *or* float32 in meters. The unit is auto-detected.
- **Mask** — single-channel PNG; non-zero pixels mark the object.
- **Intrinsics** — `(3, 3)` matrix. Helper: `recgen_inference.utils.intrinsics_from_params(fx, fy, cx, cy)`.

## Output Files (`result.save(...)`)

| File | Description |
| --- | --- |
| `mesh.obj` | mesh in object frame (with vertex colors baked into MTL) |
| `posed_mesh.obj` | mesh in camera frame (with vertex colors baked into MTL) |
| `overlay.png` | posed mesh rendered over the input RGB (purple, software rasterizer) |
| `turntable.mp4` | 4 s side-by-side turntable: gaussian color \| mesh normals (skipped with a warning if nvdiffrast + diff_gaussian_rasterization are not installed) |
| `gaussian.ply` | Gaussian splat in object frame (if `save_splat=True`) |
| `posed_gaussian.ply` | Gaussian splat in camera frame (if `save_splat=True`) |
| `textured_mesh.glb` | mesh with baked texture in object frame (if `save_glb=True` and nvdiffrast + diff_gaussian_rasterization are installed) |
| `metadata.json` | predicted pose + camera info |

## Viewing Gaussian Splats

The `gaussian.ply` produced with `--save-splat` is compatible with [SuperSplat](https://superspl.at/editor), a browser-based viewer and editor. Drag the file into the editor window — no upload or install required, everything runs locally in the browser. SuperSplat is also useful for cropping, cleaning, and re-exporting splats (`.ply`, `.splat`, or compressed `.ply`).

## Serving (HTTP)

Run RecGen as an HTTP service. Requests and responses are msgpack blobs carrying
numpy arrays directly (no PNG round-trip); see `scripts/client_tiptop.py` for a
client example.

```bash
pixi run serve                              # gateway: one worker per GPU, public :18324
pixi run serve --gpus 0,1,2,3 --port 18324  # pin the GPU set explicitly
```

`pixi run serve` starts a **gateway** that spawns one GPU-pinned worker process
per GPU (auto-detected from `CUDA_VISIBLE_DEVICES`, else `nvidia-smi`) and
exposes a single `/generate` endpoint. Each request is dispatched to an idle
GPU; when all GPUs are busy, requests queue FIFO and fall back to `503` after
`--queue-timeout` (default 300 s). The client only ever talks to the gateway —
the multi-GPU fan-out is transparent. To fan out, the client must send requests
**concurrently** (the async `scripts/client_tiptop.py` does this); a single
client looping one request at a time will only ever use one GPU.

Startup is staggered: one worker loads first to warm the shared weight caches
(HF + the `torch.hub`/DINOv2 download), then the rest load from cache — so a cold
machine does a single download pass instead of N workers racing.

| Endpoint | Description |
| --- | --- |
| `POST /generate` | one object: msgpack `{rgb, depth, mask, intrinsics, seed?, target_faces?}` → `{vertices, faces, vertex_colors?, pose_matrix, pose_quat}` |
| `GET /health` | gateway status + `workers_total` / `workers_alive` / `workers_idle` |

For single-GPU debugging, `pixi run serve-worker` runs one worker directly
(no gateway) on `:18324`.

> A worker that crashes (segfault/OOM-kill) or wedges its CUDA context is
> recycled automatically: the in-flight request retries on another GPU while the
> dead worker respawns in the background (its GPU rejoins the pool once the fresh
> pipeline finishes loading). A worker detects an unrecoverable CUDA context
> itself and self-exits so the gateway can replace it.

## Troubleshooting

- **`spconv` import error** — wrong CUDA variant. Reinstall with `pip install spconv-cu118` or `spconv-cu120` to match your CUDA.
- **`flash-attn` build fails** — safe to ignore; RecGen falls back to PyTorch SDPA.
- **`diff_gaussian_rasterization` error on CUDA 12+** — rebuild from source via `bash scripts/setup_cuda.sh`; the prebuilt wheel targets CUDA 11.
- **GLB export fails** — run `pixi run build-nvdiffrast` and `pixi run build-gaussian-rasterizer`, or set `save_glb=False`.
- **OpenGL / EGL errors on headless servers** — `PYOPENGL_PLATFORM=egl` is set automatically; ensure your driver has EGL support.

## License

This project is released for non-commercial use only. See [LICENSE](LICENSE) for full terms.
