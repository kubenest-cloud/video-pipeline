# video-pipeline

Realistic single-person 8–10s video clips. **Wan 2.2 Animate 14B** generates the frames from one reference image + one driving video, **RIFE 4.18** interpolates 8 → 24 FPS, **ffmpeg** encodes. Designed for one A6000 (48GB) on Vast.ai.

## Stack (why)

- **Wan 2.2 Animate 14B** — purpose-built character-animation model: takes one reference image (identity) plus a driving video (motion) and renders the reference person performing the driving motion at native photoreal quality. Pose extraction is internal — no separate skeleton step needed. ~28 GB model d/l on first run, fits 48 GB A6000 with cpu-offload.
- **RIFE 4.18** — generate at 8 FPS, interpolate to 24 FPS (3×). Cheap, sharp.
- **ffmpeg** — h264 + optional h265 + preview gif.

## Setup (Vast.ai A6000 box)

Each stage has its own uv project under `stages/0N_<name>/` with its own `.venv` so dependency conflicts between stages stay contained. The orchestrator (`pipeline.py`) spawns each stage via `uv run --project <stage_dir>`, which auto-syncs the stage's venv on first run.

```bash
# 1. install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. drop your inputs in
cp /path/to/source_video.mp4 inputs/pose_source.mp4
cp /path/to/face_photo.jpg   inputs/face_ref.png

# 3. run the pipeline — each stage's venv builds on first invocation.
#    First run also downloads Wan2.2-Animate-14B (~28 GB) into HF_HOME.
python pipeline.py --config config.yaml
```

To pre-build a single stage (e.g. for debugging):
```bash
uv sync --project stages/01_generate
```

To add a dependency to a stage:
```bash
uv add --project stages/01_generate <pkg>
```

## Run

Each stage runs alone, dumps debug artifacts, and is rerunnable in isolation. **Inspect after every stage** before paying for the next.

```bash
# All in one
python pipeline.py --config config.yaml

# Stage by stage (run from repo root). Set VIDEO_PIPELINE_ROOT so the common
# package can locate outputs/. pipeline.py sets this for you automatically.
export VIDEO_PIPELINE_ROOT=$PWD
uv run --project stages/01_generate    python main.py --config $PWD/config.yaml   # → outputs/<run>/raw_frames/_contactsheet.jpg
uv run --project stages/02_interpolate python main.py --config $PWD/config.yaml   # → outputs/<run>/interpolated/_compare.mp4
uv run --project stages/03_encode      python main.py --config $PWD/config.yaml   # → outputs/<run>/final.mp4

# Run a subset
python pipeline.py --from-stage 2 --to-stage 3
```

For interactive debugging in a single stage:
```bash
source stages/01_generate/.venv/bin/activate
export VIDEO_PIPELINE_ROOT=$PWD
python -i stages/01_generate/main.py --config config.yaml
```

## RIFE setup (one-time)

Stage 2 clones [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) into `stages/02_interpolate/third_party/Practical-RIFE/`. You then need to **download the v4.18 model weights** per its README and place them in `stages/02_interpolate/third_party/Practical-RIFE/train_log/`. Re-run stage 2 after that.

## Knobs (config.yaml)

| Key | Default | Effect |
|---|---|---|
| `generation.model` | Wan-AI/Wan2.2-Animate-14B-Diffusers | Wan-Animate diffusers repo. |
| `generation.width` × `height` | 512 × 768 | Wan handles arbitrary aspect; match your driving video for best results. |
| `generation.steps` | 30 | Wan converges around 30; >40 rarely visible. |
| `generation.guidance_scale` | 5.0 | Wan's documented sweet spot. >6 oversaturates and stiffens motion. |
| `generation.fps_generate` | 8 | Generate fewer frames → faster. RIFE 3× → 24 FPS. |
| `generation.duration_sec` | 10 | 10 × 8 = 80 generated frames. |

## Troubleshooting

| Symptom | First place to look |
|---|---|
| Identity drift across the clip | Use a frontal, well-lit reference photo. Wan-Animate locks identity tightly when the reference is clean. |
| Output ignores the driving motion | `WanAnimatePipeline` not installed — check stage 1 logs for "falling back to WanImageToVideoPipeline". Upgrade diffusers to ≥0.34. |
| Black or blocky frames | Make sure the venv is bf16-capable; fp16 underflows Wan's rotary embeds. |
| OOM | Drop resolution (height → 512), drop `num_frames` (lower `duration_sec`). Cpu-offload + vae tiling are already on. |
| RIFE step fails | Did you download the model weights into `third_party/Practical-RIFE/train_log/`? |

## Layout

Each stage is its own uv project (own `pyproject.toml`, `uv.lock`, `.venv`). The shared `video_pipeline_common` package handles paths/config/logging and is consumed by every stage via `[tool.uv.sources]` path dependency.

```
video-pipeline/
├── config.yaml                       # all knobs
├── pipeline.py                       # orchestrator: uv run --project per stage
├── packages/common/                  # video_pipeline_common (paths, config, logging)
├── stages/
│   ├── 01_generate/                  # Wan 2.2 Animate 14B
│   ├── 02_interpolate/               # torch + opencv + RIFE deps
│   │   └── third_party/Practical-RIFE/   # cloned on first run
│   └── 03_encode/                    # ffmpeg only
├── inputs/                           # pose_source.mp4, face_ref.png  (gitignored)
└── outputs/<run_id>/                 # all artifacts (gitignored)
```
