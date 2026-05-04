# Stage 1 — generate

Wan 2.2 Animate 14B. Takes one reference image (identity) plus one driving video (motion) and renders the reference person performing the driving video's motion at native photoreal quality. Pose extraction is internal — no separate skeleton step or face-embedding step. Output is the dense per-frame PNG sequence + a contact sheet + a raw-fps mp4 for fast inspection. Stage 2 (interpolation) consumes the PNGs.

## What it does

1. Reads `inputs/face_ref.png` and `inputs/pose_source.mp4` (paths from `config.yaml`). Center-crops both to `(width, height)` so identity and motion arrive in the model's coordinate system without letterbox bands.
2. Samples the driving video at `generation.fps_generate`, padding with the last frame if the source runs short, until exactly `n = duration_sec * fps_generate` frames are available.
3. Loads `WanAnimatePipeline` (purpose-built ref-image + driving-video animator). Falls back to `WanImageToVideoPipeline` if the installed diffusers doesn't expose Animate yet — that path loses pose adherence but keeps photoreal motion (documented downgrade).
4. Enables `model_cpu_offload` (transformer blocks stream GPU↔CPU per forward) and `vae.enable_tiling` (decoder fits 720p without OOM). Even on 48 GB you want headroom for T5 + VAE + activations across 80 frames.
5. Runs one `pipe(...)` call: prompt + neg_prompt, `num_frames=n`, reference image + driving frame list, single seed. Wan extracts pose from the driving video inside the model.
6. Writes per-frame PNGs, a 4×N contact sheet (every 8th frame), and a raw-fps mp4 preview muxed via PyAV.

## Inputs

| Path | What |
|---|---|
| `inputs/face_ref.png` | Single-image identity reference. Frontal, well-lit photos work best. |
| `inputs/pose_source.mp4` | Driving video — any person performing the motion you want rendered. Wan extracts pose internally. |
| `config.yaml` | Knobs: `seed`, `generation.{model,i2v_fallback_model,width,height,steps,guidance_scale,fps_generate,duration_sec,prompt,negative_prompt}`. |

## Outputs

`outputs/<run_id>/raw_frames/`:

| File | What |
|---|---|
| `0001.png … NNNN.png` | Dense per-frame RGB. Stage 2 (interpolation) reads these. |
| `_contactsheet.jpg` | Every 8th frame, 4-wide grid. Eyeball for identity drift, motion mismatch, color shift. |
| `_preview.mp4` | Raw `fps_generate` playback (8 fps, libx264, no interpolation). Watch for flicker — interp can hide some, but glaring per-frame breakage is your problem. |

## Run locally with Docker

```bash
# 1. Build (from repo root). ~3 min — wheels-only, no sdist compile.
docker build -t video-pipeline-stage1 -f stages/01_generate/Dockerfile .

# 2. Run. First invocation downloads ~28 GB of HF weights (Wan2.2-Animate-14B
#    + T5 + VAE) plus ~50 MB of DW Pose weights for inline pose+face
#    preprocessing. Mount both caches so subsequent runs hit them.
docker run --rm --gpus all \
    -v "$PWD/config.yaml":/workspace/config.yaml:ro \
    -v "$PWD/inputs":/workspace/inputs:ro \
    -v "$PWD/outputs":/workspace/outputs \
    -v video-pipeline-hf:/root/.cache/huggingface \
    -v video-pipeline-rtmlib:/root/.cache/rtmlib \
    video-pipeline-stage1

# Optional: outputs owned by you instead of root
docker run --rm --gpus all --user "$(id -u):$(id -g)" ...
```

## Run locally without Docker (advanced)

Requires Python 3.11, an NVIDIA GPU with a CUDA 12.4 driver, and `ffmpeg` on `PATH` for the mp4 preview muxer.

```bash
uv sync --project stages/01_generate
VIDEO_PIPELINE_ROOT=$PWD \
    uv run --project stages/01_generate \
    python stages/01_generate/main.py --config config.yaml
```

## Notes

- **bf16 only.** Wan 2.2 was trained in bfloat16; fp16 underflows the rotary embedding scales and produces black/blocky frames. Stay in bf16 (`DTYPE = torch.bfloat16` on GPU). Cards without bf16 (T4, V100) won't work.
- **CFG 5 is the sweet spot.** Wan was tuned for low-CFG sampling per its docs. >6 oversaturates and stiffens motion; <4 desaturates. The default `5.0` is what their team published with.
- **Steps 30 is enough.** Wan converges around 30 inference steps; >40 is rarely visibly different.
- **VRAM.** ~22–24 GB peak at 512×768, 80 frames, with cpu-offload + vae tiling. Dropping height to 512 cuts ~5 GB. Disabling cpu-offload would push above 40 GB on this clip length.
- **Reference image quality dominates identity.** Wan-Animate locks identity tightly when the reference is a clean frontal portrait; a cluttered or low-light reference yields drift. Single image is enough — no LoRA training needed.
- **Pose adherence is internal.** Unlike the old AnimateDiff + ControlNet approach, you can't tune a `controlnet_scale`. The model decides how literally to follow the driving motion. If you need looser following, lower `guidance_scale`; for tighter, use a driving video that more closely matches your target framing.
- **Aspect ratios.** The model handles arbitrary aspects, but reference and driving frames are center-cropped to `(width, height)` here so geometry is consistent. Set `width`/`height` to match your driving video's aspect for the cleanest crop.
