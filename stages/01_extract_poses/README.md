# Stage 1 — extract_poses

Detects DW Pose skeletons from a source video (or image directory), one frame at a time, and saves them as PNGs sized to the generation resolution. These skeletons are the ControlNet conditioning input for stage 3.

## What it does

1. Reads `cfg.input.pose_source` (mp4 or directory of frames).
2. Computes a stride to downsample to `cfg.generation.fps_generate` (default 8 FPS).
3. For each sampled frame, runs `controlnet_aux.DWposeDetector` (which lazy-loads `mmpose` + `mmdet` + a YOLOX detector and dw-ll_ucoco_384 keypoint head — both downloaded from openmmlab/HuggingFace on first run, cached in `/root/.cache/torch/hub/`).
4. Resizes each detected skeleton to `(cfg.generation.width, cfg.generation.height)` — by default **512×768**, NOT the source video resolution. The output must match stage 3's latent resolution.
5. Pads with the last frame if the source runs out before reaching `fps_generate × duration_sec` frames (default 80).

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
# 1. Build (from repo root). First build ~10 min — mmcv compiles from sdist.
#    Subsequent builds reuse the cached wheel from BuildKit's uv cache mount.
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

Requires CUDA toolkit (nvcc on PATH), gcc≤12 (CUDA 12.1 rejects gcc-13), Python 3.11.

```bash
uv sync --project stages/01_extract_poses
VIDEO_PIPELINE_ROOT=$PWD \
    uv run --project stages/01_extract_poses \
    python stages/01_extract_poses/main.py --config config.yaml
```

## Notes

- The first run downloads ~500 MB of detector weights into the container's `/root/.cache/torch/hub/`. These are lost when the container exits unless you also mount a cache volume:
  ```bash
  -v video-pipeline-hf-cache:/root/.cache
  ```
- Stage 1 is the only stage that needs the OpenMMLab toolchain (`mmcv`, `mmdet`, `mmpose`); they're isolated to this venv so other stages never pay their conflict cost.
- If you change `cfg.generation.width/height`, re-run stage 1: the skeletons are pre-sized for stage 3 and won't match if you skip this.
