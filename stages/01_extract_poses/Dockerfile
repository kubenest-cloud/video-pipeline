# Stage 1 — DW Pose extraction.
#
# Inference is pure onnxruntime (rtmlib wraps the official DWPose ONNX models),
# so the image only needs the CUDA + cuDNN runtime libraries — no toolkit, no
# nvcc, no compilation. onnxruntime-gpu 1.18+ requires cuDNN 9, hence the
# `cudnn-runtime` (== cuDNN 9) tag rather than the `cudnn8-runtime` variant.
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

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 ships python 3.10. Project pins python 3.11; deadsnakes for that.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /workspace

COPY packages/common /workspace/packages/common
COPY stages/01_extract_poses/pyproject.toml stages/01_extract_poses/uv.lock /workspace/stages/01_extract_poses/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /workspace/stages/01_extract_poses

# opencv-python (pulled transitively by rtmlib) needs libGL + libglib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY stages/01_extract_poses/main.py /workspace/stages/01_extract_poses/main.py

ENV VIDEO_PIPELINE_ROOT=/workspace
WORKDIR /workspace/stages/01_extract_poses

CMD ["uv", "run", "--project", "/workspace/stages/01_extract_poses", \
     "python", "main.py", "--config", "/workspace/config.yaml"]
