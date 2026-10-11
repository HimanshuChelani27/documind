# Build from the repo root:  docker build -f backend/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    DOCUMIND_MODEL_CACHE_DIR=/app/models \
    DOCUMIND_MODEL_THREADS=2 \
    MALLOC_ARENA_MAX=2

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/app ./app
COPY samples /app/samples
# Download both ONNX models and prepare the sample library + baseline eval at build time.
RUN uv run --no-sync python -m app.prebuild

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
