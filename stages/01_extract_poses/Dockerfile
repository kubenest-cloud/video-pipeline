# Stage 1 — DW Pose extraction.
#
# Base: cu121 devel ships nvcc and ubuntu 22.04's gcc-11, which is exactly
# what mmcv's sdist build needs (CUDA 12.1's nvcc rejects gcc>=13).
#
# Build (from repo root, so the build context can see packages/common):
#   docker build -t video-pipeline-stage1 -f stages/01_extract_poses/Dockerfile .
#
# Run (mounts config.yaml + inputs/outputs):
#   docker run --rm --gpus all \
#     -v "$PWD/config.yaml":/workspace/config.yaml:ro \
#     -v "$PWD/inputs":/workspace/inputs:ro \
#     -v "$PWD/outputs":/workspace/outputs \
#     video-pipeline-stage1

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 ships python 3.10 + gcc 11. Project pins python 3.11; deadsnakes for that.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        build-essential \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /workspace

# Copy only what stage 1 needs. Layer order optimized for cache hits when only
# main.py changes vs when the pyproject/lockfile change.
COPY packages/common /workspace/packages/common
COPY stages/01_extract_poses/pyproject.toml stages/01_extract_poses/uv.lock /workspace/stages/01_extract_poses/

# Build the venv (mmcv sdist compile — 5–15 min on first build).
# BuildKit cache mount keeps uv's wheel cache across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /workspace/stages/01_extract_poses

# OpenCV (pulled transitively by controlnet_aux as the non-headless build) needs
# libGL + libglib at runtime. Installed late so the mmcv build layer stays cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY stages/01_extract_poses/main.py /workspace/stages/01_extract_poses/main.py

ENV VIDEO_PIPELINE_ROOT=/workspace
WORKDIR /workspace/stages/01_extract_poses

CMD ["uv", "run", "--project", "/workspace/stages/01_extract_poses", \
     "python", "main.py", "--config", "/workspace/config.yaml"]
