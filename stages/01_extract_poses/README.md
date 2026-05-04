# Stage 1 — extract_poses

Detects DW Pose skeletons from a source video (or image directory), one frame at a time, and saves them as PNGs sized to the generation resolution. These skeletons are the ControlNet conditioning input for stage 3.

## What it does

1. Reads `cfg.input.pose_source` (mp4 or directory of frames).
2. Computes a stride to downsample to `cfg.generation.fps_generate` (default 8 FPS).
3. For each sampled frame, runs `rtmlib.Wholebody` — the official DWPose ONNX models (YOLOX person detector + dw-ll_ucoco_384 whole-body keypoints) executed via onnxruntime-gpu. Models auto-download to `/root/.cache/rtmlib/` on first run.
4. Renders the keypoints onto a black canvas with `rtmlib.draw_skeleton(..., openpose_skeleton=True)` so the output matches the openpose convention ControlNet expects.
5. Resizes each skeleton to `(cfg.generation.width, cfg.generation.height)` — by default **512×768**, NOT the source video resolution. The output must match stage 3's latent resolution.
6. Pads with the last frame if the source runs out before reaching `fps_generate × duration_sec` frames (default 80).

## Inputs

| Path | What |
|---|---|
| `inputs/pose_source.mp4` | Video that drives the motion. Resolved against repo root. |
| `config.yaml` | Knobs: `generation.width`, `generation.height`, `generation.fps_generate`, `generation.duration_sec`, `run_id`. |

## Outputs

`outputs/<run_id>/`:

| File | What |
|---|---|
| `poses/0001.png … NNNN.png` | Skeletons on black bg, sized to `(width, height)`. NNNN = `fps_generate × duration_sec`. |
| `poses/skeleton.json` | Per-frame metadata (output frame index, source frame index, padded flag). |
| `run.log` | Append-only log shared with all stages. |

## Run locally with Docker

```bash
# 1. Build (from repo root). Pure-wheel install — no native compilation,
#    no CUDA toolkit, no OpenMMLab toolchain.
docker build -t video-pipeline-stage1 -f stages/01_extract_poses/Dockerfile .

# 2. Run. Mount config + inputs (ro) and outputs (rw). --gpus all for the GPU.
docker run --rm --gpus all \
    -v "$PWD/config.yaml":/workspace/config.yaml:ro \
    -v "$PWD/inputs":/workspace/inputs:ro \
    -v "$PWD/outputs":/workspace/outputs \
    video-pipeline-stage1

# Optional: have outputs land owned by you instead of root
docker run --rm --gpus all --user "$(id -u):$(id -g)" ...
```

## Run locally without Docker (advanced)

Requires CUDA 12.x runtime libraries with cuDNN 9 and Python 3.11.

```bash
uv sync --project stages/01_extract_poses
VIDEO_PIPELINE_ROOT=$PWD \
    uv run --project stages/01_extract_poses \
    python stages/01_extract_poses/main.py --config config.yaml
```

## Notes

- The first run downloads the YOLOX + DWPose ONNX models (~600 MB) into `/root/.cache/rtmlib/`. To avoid re-downloading on every container run, mount a cache volume:
  ```bash
  -v video-pipeline-rtmlib-cache:/root/.cache
  ```
- If you change `cfg.generation.width/height`, re-run stage 1: the skeletons are pre-sized for stage 3 and won't match if you skip this.
