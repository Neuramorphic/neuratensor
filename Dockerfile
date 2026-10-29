# ─────────────────────────────────────────────────────────
#  NeuraTensor v3 — AWS ECS / EKS production image
#
#  Build:
#    docker build -t neuratensor:3.0.0 .
#
#  Run (CPU):
#    docker run -p 8080:8080 -e NEURATENSOR_PRESET=small neuratensor:3.0.0
#
#  Run (GPU):
#    docker run --gpus all -p 8080:8080 -e NEURATENSOR_PRESET=medium neuratensor:3.0.0
# ─────────────────────────────────────────────────────────

# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir build wheel

COPY pyproject.toml .
COPY neuratensor/ neuratensor/

RUN python -m build --wheel --outdir /dist


# Stage 2: runtime (CPU)
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="NeuraTensor v3"
LABEL org.opencontainers.image.description="Neuromorphic SNN-SSM inference API"
LABEL org.opencontainers.image.vendor="Neuramorphic, Inc."
LABEL org.opencontainers.image.version="3.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEURATENSOR_PRESET=small \
    NEURATENSOR_COMPILE=0

WORKDIR /app

# Install PyTorch CPU (much smaller than CUDA build)
RUN pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install NeuraTensor wheel + server extras
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/neuratensor-*.whl[server]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

# Non-root user
RUN useradd -m -u 1000 neuratensor
USER neuratensor

CMD ["neuratensor", "serve", "--host", "0.0.0.0", "--port", "8080"]
