# Stage 1 — Wan 2.2 Animate generation (ref image + driving video → frames).
#
# Base: cu124 RUNTIME (not devel). Wan 2.2 deps are wheels-only — torch 2.6 is
# pinned to cu124 wheels. No nvcc needed.
#
# Build (from repo root, so the build context can see packages/common):
#   docker build -t video-pipeline-stage1 -f stages/01_generate/Dockerfile .
#
# Run. Stage 1 reads `inputs/face_ref.png` (identity) and `inputs/pose_source.mp4`
# (motion). First run also downloads ~28 GB of HF weights (Wan2.2-Animate-14B
# diffusers repo + T5 text encoder + VAE). Mount the shared `video-pipeline-hf`
# volume so subsequent runs hit the cache.
#   docker run --rm --gpus all \
#     -v "$PWD/config.yaml":/workspace/config.yaml:ro \
#     -v "$PWD/inputs":/workspace/inputs:ro \
#     -v "$PWD/outputs":/workspace/outputs \
#     -v video-pipeline-hf:/root/.cache/huggingface \
#     video-pipeline-stage1
#
# Tip: outputs owned by you instead of root → add `--user "$(id -u):$(id -g)"`.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.11 (project pin) + ffmpeg for imageio's mp4 muxing of the raw-fps
# preview. PyAV wheels bundle their own libav, but imageio[ffmpeg] resolves to
# the imageio-ffmpeg shim which uses the system binary when present —
# installing ffmpeg here avoids the runtime download path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /workspace

# Copy only what stage 1 needs. Layer order optimized so editing main.py
# doesn't bust the venv build cache.
COPY packages/common /workspace/packages/common
COPY stages/01_generate/pyproject.toml stages/01_generate/uv.lock* /workspace/stages/01_generate/

# Build the venv — pure-wheel install (~3 min cold, BuildKit cache makes
# rebuilds near-instant).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /workspace/stages/01_generate

COPY stages/01_generate/main.py /workspace/stages/01_generate/main.py

ENV VIDEO_PIPELINE_ROOT=/workspace
WORKDIR /workspace/stages/01_generate
CMD ["uv", "run", "--project", "/workspace/stages/01_generate", "python", "main.py", "--config", "/workspace/config.yaml"]
