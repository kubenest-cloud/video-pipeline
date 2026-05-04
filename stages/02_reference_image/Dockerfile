# Stage 2 — face embedding + IPAdapter FaceID preview grid.
#
# Base: cu121 RUNTIME (not devel) — no sdist compiles, so nvcc isn't needed.
#
# Build (from repo root):
#   docker build -t video-pipeline-stage2 -f stages/02_reference_image/Dockerfile .
#
# Run:
#   docker run --rm --gpus all \
#     -v "$PWD/config.yaml":/workspace/config.yaml:ro \
#     -v "$PWD/inputs":/workspace/inputs:ro \
#     -v "$PWD/outputs":/workspace/outputs \
#     video-pipeline-stage2
#
# Tip: mount a HF cache volume to avoid re-downloading ~5 GB of model weights
# every container restart and to share the cache with stage 3:
#     -v video-pipeline-hf:/root/.cache/huggingface \
#     -v video-pipeline-insightface:/root/.insightface

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.11, libGL/libglib for cv2 (insightface uses it), and build-essential
# because insightface 0.7.3 has no cp311 wheel on PyPI and falls back to sdist build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

WORKDIR /workspace

COPY packages/common /workspace/packages/common
COPY stages/02_reference_image/pyproject.toml stages/02_reference_image/uv.lock /workspace/stages/02_reference_image/

# Build the venv (torch + diffusers + insightface + onnxruntime-gpu — ~5 min, all wheels).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /workspace/stages/02_reference_image

COPY stages/02_reference_image/main.py /workspace/stages/02_reference_image/main.py

ENV VIDEO_PIPELINE_ROOT=/workspace
WORKDIR /workspace/stages/02_reference_image

CMD ["uv", "run", "--project", "/workspace/stages/02_reference_image", \
     "python", "main.py", "--config", "/workspace/config.yaml"]
