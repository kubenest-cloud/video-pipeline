# Stage 3 — AnimateDiff + ControlNet (pose) + IPAdapter FaceID Plus v2 generation.
#
# Base: cu121 RUNTIME (not devel). Stage 3's deps are wheels-only — torch 2.4.1
# is pinned to cu121 wheels (the matching base is cu121.1, runtime variant). No
# nvcc needed.
#
# Build (from repo root, so the build context can see packages/common):
#   docker build -t video-pipeline-stage3 -f stages/03_generate/Dockerfile .
#
# Run. Stage 3 reads stage 1's pose frames (outputs/<run_id>/poses) and stage
# 2's embedding artifacts (outputs/<run_id>/reference) — both must exist before
# stage 3. First run also downloads ~5 GB of HF weights (motion adapter,
# ControlNet OpenPose, base SD1.5, IP-Adapter FaceID Plus v2). Mount the same
# `video-pipeline-hf` volume stage 2 used to share the cache.
#   docker run --rm --gpus all \
#     -v "$PWD/config.yaml":/workspace/config.yaml:ro \
#     -v "$PWD/outputs":/workspace/outputs \
#     -v video-pipeline-hf:/root/.cache/huggingface \
#     video-pipeline-stage3
#
# Tip: outputs owned by you instead of root → add `--user "$(id -u):$(id -g)"`.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.11 (project pin) + ffmpeg for imageio's mp4 muxing of the raw-fps
# preview. PyAV wheels bundle their own libav, but imageio[ffmpeg] resolves to
# the imageio-ffmpeg shim which uses the system binary when present — installing
# ffmpeg here avoids the runtime download path.
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

# Copy only what stage 3 needs. Layer order optimized so editing main.py
# doesn't bust the venv build cache.
COPY packages/common /workspace/packages/common
COPY stages/03_generate/pyproject.toml stages/03_generate/uv.lock /workspace/stages/03_generate/

# Build the venv — pure-wheel install (~3 min cold, BuildKit cache makes
# rebuilds near-instant).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /workspace/stages/03_generate

COPY stages/03_generate/main.py /workspace/stages/03_generate/main.py

ENV VIDEO_PIPELINE_ROOT=/workspace
WORKDIR /workspace/stages/03_generate

CMD ["uv", "run", "--project", "/workspace/stages/03_generate", \
     "python", "main.py", "--config", "/workspace/config.yaml"]
