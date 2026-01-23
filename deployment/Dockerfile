# syntax=docker/dockerfile:1

# CUDA-enabled image intended for k8s GPU nodes.
# Build from repo root:
#   docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
# Run:
#   docker run --rm -p 8000:8000 -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 -v /path/to/models:/models nano-vllm-voxcpm-deployment:latest

ARG PYTORCH_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

FROM ${PYTORCH_IMAGE}-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Build deps for native extensions (e.g. flash-attn) and audio libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ninja-build \
      git \
      curl \
      pkg-config \
      libsndfile1-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip install --no-cache-dir uv

# Copy only what uv needs for dependency resolution first.
COPY pyproject.toml uv.lock ./
COPY deployment/pyproject.toml ./deployment/pyproject.toml

# Workspace source (uv workspaces typically install local packages editable).
COPY nanovllm_voxcpm/ ./nanovllm_voxcpm/
COPY deployment/app/ ./deployment/app/

# Install only the deployment service (and its workspace deps) from the frozen lockfile.
RUN uv sync --frozen --no-dev --package nano-vllm-voxcpm-deployment


FROM ${PYTORCH_IMAGE}-runtime AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    # Prefer an explicit writable cache path in containers.
    NANOVLLM_CACHE_DIR=/var/cache/nanovllm

WORKDIR /app

# Runtime libs for soundfile; keep this minimal.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR"

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

EXPOSE 8000

USER appuser

# k8s probes can use /health and /ready.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
