# Stage 2 — reference_image

Extracts a 512-d face embedding from a single reference photo and renders a 2×2 preview grid through the same SD1.5 + IPAdapter FaceID Plus v2 pipeline that stage 3 will use. The preview is a **gate**: if the face in the grid doesn't look like the input, fix the reference (lighting, framing, occlusion) before spending GPU on stage 3.

## What it does

1. Reads `cfg.input.face_ref` (a portrait PNG/JPG, frontal works best).
2. Runs `insightface.FaceAnalysis(name="antelopev2")` with CUDAExecutionProvider → CPU fallback. **Must be antelopev2, not buffalo_l** — IP-Adapter FaceID was trained on antelopev2's `glintr100` embedding space; buffalo_l's `w600k_r50` produces 512-d vectors in a different space and the projection maps them to a wrong-but-consistent identity (the original symptom that motivated this stage's redesign was a male reference rendering as a consistent female across all 4 stills).
3. Picks the largest detected face by bbox area, extracts its 512-d normed embedding.
4. Saves the bbox crop and the keypoint-aligned 224×224 crop for visual audit.
5. Encodes the aligned crop with CLIP-H (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) and saves the penultimate hidden states. Plus v2 fuses these CLIP features with the InsightFace ID embedding via a Perceiver Resampler — that's where most of the gender/ethnicity preservation comes from.
6. Builds the SD1.5 + IP-Adapter FaceID Plus v2 pipeline (no motion adapter — stills only) at the config resolution, stashes `clip_embeds` onto the projection layer.
7. Generates 4 stills at `seed`, `seed+1`, `seed+2`, `seed+3` so the IPAdapter wiring matches stage 3 exactly.
8. Composes a 2×2 grid and writes it.

## Inputs

| Path | What |
|---|---|
| `inputs/face_ref.png` | Single reference portrait. Resolved against repo root. |
| `config.yaml` | Knobs: `input.face_ref`, `seed`, `generation.{base_model,motion_adapter,ip_adapter_repo,ip_adapter_weight,ip_adapter_image_encoder,width,height,steps,guidance_scale,ip_adapter_scale,prompt,negative_prompt}`. |

## Outputs

`outputs/<run_id>/reference/`:

| File | What |
|---|---|
| `face_embedding.pt` | Torch tensor of shape `(2, 1, 512)` (CFG-packed: dim 0 is `[neg, pos]`), fed to stage 3 as `ip_adapter_image_embeds`. |
| `face_clip_embeds.pt` | Torch tensor of shape `(2, 1, 257, 1280)`, the CLIP-H penultimate hidden states. Stage 3 stashes this on the FaceID Plus v2 projection's `clip_embeds`. |
| `face_crop.png` | Bbox crop from the reference photo. Eyeball to confirm InsightFace picked the right face. |
| `face_aligned.png` | Keypoint-aligned 224×224 crop, the exact image fed to CLIP. If this looks weird (wrong rotation, partial occlusion), CLIP features will be off. |
| `preview.png` | 2×2 grid of stills, identity-conditioned by the embeddings. **Inspect this before running stage 3.** |

## Run locally with Docker

```bash
# 1. Build (from repo root). ~3 min — wheels-only, no sdist compile besides insightface 0.7.3 (no cp311 wheel on PyPI).
docker build -t video-pipeline-stage2 -f stages/02_reference_image/Dockerfile .

# 2. Run. First run downloads ~5 GB SD1.5/IP-Adapter weights + ~4 GB CLIP-H + ~500 MB insightface buffalo_l.
#    Add cache volumes so weights survive container restarts (and stage 3 reuses them).
docker run --rm --gpus all \
    -v "$PWD/config.yaml":/workspace/config.yaml:ro \
    -v "$PWD/inputs":/workspace/inputs:ro \
    -v "$PWD/outputs":/workspace/outputs \
    -v video-pipeline-hf:/root/.cache/huggingface \
    -v video-pipeline-insightface:/root/.insightface \
    video-pipeline-stage2

# Optional: outputs owned by you instead of root
docker run --rm --gpus all --user "$(id -u):$(id -g)" ...
```

## Run locally without Docker (advanced)

Requires Python 3.11, gcc/g++ (for insightface sdist build), an NVIDIA GPU with CUDA 12.1 driver.

```bash
uv sync --project stages/02_reference_image
VIDEO_PIPELINE_ROOT=$PWD \
    uv run --project stages/02_reference_image \
    python stages/02_reference_image/main.py --config config.yaml
```

## Notes

- Stage 2 doesn't depend on stage 1's output — they can run in parallel. But stage 3 needs both.
- Stage 3 consumes `face_embedding.pt` and `face_clip_embeds.pt`. The PNGs are debug aids — look at them every time you change the reference photo.
- Each preview still is roughly the cost of one SD1.5 generation. Four stills at 25 steps takes ~30s on a 3080, longer on cold start (first call recompiles attention modules + downloads CLIP-H).
- VRAM budget at `512×768`: comfortably fits in 10 GB. Bump to `768×768` and you're at ~9 GB peak — fine but tight on a 3080. CLIP-H is freed before the diffusion pipeline runs, so it doesn't compete for VRAM.
- `insightface 0.7.3` ships no cp311 wheel on PyPI; the Dockerfile installs `build-essential` so the sdist compiles. On Vast.ai's standard image this is already there.
- The first run also auto-downloads the antelopev2 model pack (~360 MB) into the `video-pipeline-insightface` volume. insightface 0.7.3 has a known bug where the antelopev2 zip extracts into `models/antelopev2/antelopev2/` (nested) instead of `models/antelopev2/`. `main.py` detects this and flattens automatically — but if you ever pre-populate the volume, mirror the flat layout: `models/antelopev2/{glintr100,scrfd_10g_bnkps,1k3d68,2d106det,genderage}.onnx`.
- If `det_score` in the log is < 0.5, the face detector picked something it's not confident about — re-shoot or crop the reference.
- Identity not matching despite `det_score` being high? `ip_adapter_scale` and prompt specificity are the main knobs. Plus v2 at scale 1.0 with explicit gender/ethnicity in the prompt + a gendered negative gives the best fidelity. Realistic_Vision_V6 has a strong female prior on under-specified prompts — you almost always want gender words in either the prompt or negative.
