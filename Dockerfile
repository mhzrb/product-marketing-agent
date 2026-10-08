# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# install the package (and only runtime dependencies)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# run as an unprivileged user; traces are written to a volume-able directory
RUN useradd --create-home --uid 10001 app && mkdir -p /data/traces && chown -R app /data
USER app

ENV TRACE_DIR=/data/traces \
    LOG_FORMAT=json \
    LLM_PROVIDER=mock

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "pma.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
