# video-pipeline

Realistic single-person 8–10s video clips. SD1.5 + AnimateDiff v3 + ControlNet (DW Pose) + IPAdapter FaceID, then RIFE 4.x interpolation, then ffmpeg encode. Designed for one A6000 (48GB) on Vast.ai.

## Stack (why)

- **SD1.5 + AnimateDiff v3** — strongest open-source motion module; SDXL motion adapters are weaker. Revisit SDXL once baseline works.
- **ControlNet OpenPose + DW Pose** — best skeleton extractor → tight pose adherence.
- **IPAdapter FaceID Plus v2** — single-image identity, no LoRA training needed in v1.
- **FreeNoise** — seamless >16-frame generation without retraining.
- **RIFE 4.18** — generate at 8 FPS, interpolate to 24 FPS (3×). Cheap, sharp.

See `/home/romero/.claude/plans/you-re-an-ai-engineer-iterative-barto.md` for the full rationale.

## Setup (Vast.ai A6000 box)

Each stage has its own uv project under `stages/0N_<name>/` with its own `.venv` so dependency conflicts between stages stay contained. The orchestrator (`pipeline.py`) spawns each stage via `uv run --project <stage_dir>`, which auto-syncs the stage's venv on first run.

```bash
# 1. install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. drop your inputs in
cp /path/to/source_video.mp4 inputs/pose_source.mp4
cp /path/to/face_photo.jpg   inputs/face_ref.jpg

# 3. run the pipeline — each stage's venv builds on first invocation.
#    Stage 1's mmcv compiles from sdist (5–15 min, needs nvcc on PATH).
python pipeline.py --config config.yaml
```

To pre-build a single stage (e.g. for debugging):
```bash
uv sync --project stages/03_generate
```

To add a dependency to a stage:
```bash
uv add --project stages/03_generate <pkg>
```

## Run

Each stage runs alone, dumps debug artifacts, and is rerunnable in isolation. **Inspect after every stage** before paying for the next.

```bash
# All in one
python pipeline.py --config config.yaml

# Stage by stage (run from repo root). Set VIDEO_PIPELINE_ROOT so _common can
# locate outputs/. pipeline.py sets this for you automatically.
export VIDEO_PIPELINE_ROOT=$PWD
uv run --project stages/01_extract_poses   python main.py --config $PWD/config.yaml   # → outputs/<run>/poses/
uv run --project stages/02_reference_image python main.py --config $PWD/config.yaml   # → outputs/<run>/reference/preview.png  ← STOP IF IDENTITY WRONG
uv run --project stages/03_generate        python main.py --config $PWD/config.yaml   # → outputs/<run>/raw_frames/_contactsheet.jpg
uv run --project stages/04_interpolate     python main.py --config $PWD/config.yaml   # → outputs/<run>/interpolated/_compare.mp4
uv run --project stages/05_encode          python main.py --config $PWD/config.yaml   # → outputs/<run>/final.mp4

# Run a subset
python pipeline.py --from-stage 3 --to-stage 5
```

For interactive debugging in a single stage:
```bash
source stages/03_generate/.venv/bin/activate
export VIDEO_PIPELINE_ROOT=$PWD
python -i stages/03_generate/main.py --config config.yaml
```

## RIFE setup (one-time)

Stage 4 clones [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) into `stages/04_interpolate/third_party/Practical-RIFE/`. You then need to **download the v4.18 model weights** per its README and place them in `stages/04_interpolate/third_party/Practical-RIFE/train_log/`. Re-run stage 4 after that.

## Knobs (config.yaml)

All hyperparameters live in `config.yaml`. The interesting ones:

| Key | Default | Effect |
|---|---|---|
| `generation.width` × `height` | 512 × 768 | SD1.5 sweet spot is 512-base. 768×768 also works on 48GB. |
| `generation.steps` | 25 | 25 with DPM++ 2M Karras; 30+ rarely worth it. |
| `generation.controlnet_scale` | 0.8 | Lower → more identity freedom; higher → tighter pose. |
| `generation.ip_adapter_scale` | 0.7 | Lower → more prompt freedom; higher → harder identity lock. |
| `generation.context_frames` / `context_overlap` | 16 / 4 | FreeNoise window. Don't change unless you know why. |
| `generation.fps_generate` | 8 | Generate fewer frames → faster. RIFE 3× → 24 FPS. |
| `generation.duration_sec` | 10 | 10 × 8 = 80 generated frames. |

## Troubleshooting

| Symptom | First place to look |
|---|---|
| Identity morphs across the clip | `outputs/<run>/raw_frames/_contactsheet.jpg`. Bump `ip_adapter_scale` to 0.8–0.9. Try a frontal, well-lit reference. |
| Pose ignored | `outputs/<run>/poses/` — are the skeletons clean? If yes, raise `controlnet_scale` to 0.9. |
| Flicker between context windows | Increase `context_overlap` to 6. Check FreeNoise actually enabled in stage 3 logs. |
| OOM | Drop resolution first (768→512 height), then `context_frames` to 12. Attention slicing usually slower than just lowering res. |
| RIFE step fails | Did you download the model weights into `third_party/Practical-RIFE/train_log/`? |

## Layout

Each stage is its own uv project (own `pyproject.toml`, `uv.lock`, `.venv`). The shared `video_pipeline_common` package handles paths/config/logging and is consumed by every stage via `[tool.uv.sources]` path dependency.

```
video-pipeline/
├── config.yaml                       # all knobs
├── pipeline.py                       # orchestrator: uv run --project per stage
├── packages/common/                  # video_pipeline_common (paths, config, logging)
├── stages/
│   ├── 01_extract_poses/             # OpenMMLab + controlnet_aux (mmcv built from sdist)
│   ├── 02_reference_image/           # diffusers + insightface + onnxruntime-gpu
│   ├── 03_generate/                  # diffusers + controlnet
│   ├── 04_interpolate/               # torch + opencv + RIFE deps
│   │   └── third_party/Practical-RIFE/   # cloned on first run
│   └── 05_encode/                    # ffmpeg only
├── inputs/                           # pose_source.mp4, face_ref.jpg  (gitignored)
└── outputs/<run_id>/                 # all artifacts (gitignored)
```
