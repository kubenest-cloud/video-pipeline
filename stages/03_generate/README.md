# Stage 3 — generate

Renders the raw frame sequence: SD1.5 + AnimateDiff v3 motion adapter + ControlNet (OpenPose) + IP-Adapter FaceID Plus v2, with FreeNoise context scheduling so we can generate >16 frames in one call without temporal seams. Output is the dense per-frame PNG sequence + a contact sheet + a raw-fps mp4 for fast inspection. Stage 4 (interpolation) consumes the PNGs.

## What it does

1. Reads `outputs/<run_id>/poses/0001.png … NNNN.png` (stage 1) and `outputs/<run_id>/reference/{face_embedding.pt,face_clip_embeds.pt}` (stage 2). Bails fast if any are missing.
2. Computes `n = duration_sec * fps_generate` and loads exactly that many pose frames in order. Mismatched pose count is a hard error (a missing `0042.png` aborts).
3. Builds `AnimateDiffControlNetPipeline`: motion adapter + ControlNet OpenPose + base SD1.5 + IP-Adapter FaceID Plus v2.
4. Stashes stage 2's CLIP-H embeds onto the FaceID Plus v2 projection layer (`proj.clip_embeds`, `shortcut=True`, `shortcut_scale=1.0`). The diffusers pipeline has no kwarg for this — it's read directly off the module.
5. Enables FreeNoise (`context_length=context_frames`, `context_stride=context_overlap`) so cross-window denoising is consistent across the full clip.
6. Runs one `pipe(...)` call: prompt + neg_prompt, `num_frames=n`, pose frames as `conditioning_frames`, the InsightFace embedding as `ip_adapter_image_embeds`. Single seed — temporal coherence comes from the motion adapter + FreeNoise, not from per-frame seed scheduling.
7. Writes per-frame PNGs, a 4×N contact sheet (every 8th frame), and a raw-fps mp4 preview muxed via PyAV.

## Inputs

| Path | What |
|---|---|
| `outputs/<run_id>/poses/NNNN.png` | OpenPose-rendered keypoint frames from stage 1. |
| `outputs/<run_id>/reference/face_embedding.pt` | `(2, 1, 512)` antelopev2 ID embedding from stage 2 (CFG-packed: dim 0 is `[neg, pos]`). |
| `outputs/<run_id>/reference/face_clip_embeds.pt` | `(2, 1, 257, 1280)` CLIP-H penultimate hidden states from stage 2. |
| `config.yaml` | Knobs: `seed`, `generation.{base_model,motion_adapter,controlnet,ip_adapter_repo,ip_adapter_weight,width,height,steps,guidance_scale,controlnet_scale,ip_adapter_scale,context_frames,context_overlap,fps_generate,duration_sec,prompt,negative_prompt}`. |

## Outputs

`outputs/<run_id>/raw_frames/`:

| File | What |
|---|---|
| `0001.png … NNNN.png` | Dense per-frame RGB. Stage 4 (interpolation) reads these. |
| `_contactsheet.jpg` | Every 8th frame, 4-wide grid. Eyeball for identity drift, pose mismatch, color shift. |
| `_preview.mp4` | Raw `fps_generate` playback (8 fps, libx264, no interpolation). Watch for flicker — interp can hide some, but glaring per-frame breakage is your problem. |

## Run locally with Docker

```bash
# 1. Build (from repo root). ~3 min — wheels-only, no sdist compile.
docker build -t video-pipeline-stage3 -f stages/03_generate/Dockerfile .

# 2. Run. Stages 1 + 2 must have completed for this `run_id` first.
#    First run downloads ~5 GB of HF weights (motion adapter + ControlNet OpenPose
#    + base SD1.5 + IP-Adapter FaceID Plus v2). Mount the same `video-pipeline-hf`
#    volume stage 2 used so the cache is shared.
docker run --rm --gpus all \
    -v "$PWD/config.yaml":/workspace/config.yaml:ro \
    -v "$PWD/outputs":/workspace/outputs \
    -v video-pipeline-hf:/root/.cache/huggingface \
    video-pipeline-stage3

# Optional: outputs owned by you instead of root
docker run --rm --gpus all --user "$(id -u):$(id -g)" ...
```

Stage 3 doesn't read `inputs/` — it only consumes prior stages' output artifacts plus `config.yaml`. No `inputs/` mount needed.

## Run locally without Docker (advanced)

Requires Python 3.11, an NVIDIA GPU with a CUDA 12.1 driver, and `ffmpeg` on `PATH` for the mp4 preview muxer.

```bash
uv sync --project stages/03_generate
VIDEO_PIPELINE_ROOT=$PWD \
    uv run --project stages/03_generate \
    python stages/03_generate/main.py --config config.yaml
```

## Notes

- **Hard prereqs.** Stages 1 and 2 must finish first for this `run_id`. Stage 3 fails fast with a clear error if pose frames or the two `.pt` files are missing.
- **The IP-Adapter weight + scale must match stage 2.** Stage 2's preview is your gate — if the preview was identity-correct, do not change `ip_adapter_weight` / `ip_adapter_scale` for stage 3 or you'll re-introduce drift you already eliminated.
- **Plus v2 wiring is non-obvious.** Diffusers gives no kwarg path for `clip_embeds` — they're read directly off `pipe.unet.encoder_hid_proj.image_projection_layers[0]`. The `shortcut=True, shortcut_scale=1.0` pair is the canonical Plus v2 residual setting; don't omit either.
- **VRAM budget.** At `512×512`, 80 frames, `context_frames=16`, `context_overlap=4`: ~10 GB peak on a 3080. FreeNoise is what makes this fit — it processes overlapping 16-frame windows instead of all 80 at once. Bump `context_frames` to 24 and you'll OOM on a 3080.
- **FreeNoise requires diffusers ≥ 0.31** — the pin (`<0.32`) is in `pyproject.toml` (see [the project memory on diffusers 0.31 + torch 2.4](../../packages/common) for why we don't go higher).
- **Single seed by design.** Per-frame seed scheduling is what stage 2's preview grid does (4 stills at `seed`, `seed+1`, `seed+2`, `seed+3`); stage 3 wants temporal coherence, so the seed is fixed and motion comes from the AnimateDiff adapter.
- **`controlnet_conditioning_scale` is the pose-faithfulness knob.** Default `0.8` lets the model deviate slightly from rigid pose adherence (helps avoid stiff, pasted-on figures). Drop to `0.6` if pose looks ignored; raise to `1.0` if the model is hallucinating off-pose limbs.
- **Identity drift in the preview mp4** is most often the prompt — Realistic_Vision's prior fights the IP-Adapter on under-specified prompts. Match stage 2's prompt verbatim once you've found one that works there.
