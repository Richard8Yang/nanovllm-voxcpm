# syntax=docker/dockerfile:1

# CUDA-enabled image intended for k8s GPU nodes.
# Build from repo root:
#   docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
# Run:
#   docker run --rm -p 8000:8000 -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 -v /path/to/models:/models nano-vllm-voxcpm-deployment:latest

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Build deps for native extensions (e.g. flash-attn) and audio libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      ninja-build \
      git \
      python3 \
      python3-pip \
      python3-venv \
      pkg-config \
      libsndfile1-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip \
 && python3 -m pip install --no-cache-dir uv \
 && ln -sf /usr/bin/python3 /usr/local/bin/python

COPY . .

# Install workspace packages from the frozen lockfile.
RUN uv sync --all-packages --frozen --no-dev

ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm

RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR"

EXPOSE 8000

USER appuser

# k8s probes can use /health and /ready.
CMD ["uv", "run", "fastapi", "run", "deployment/app/main.py", "--host", "0.0.0.0", "--port", "8000"]
