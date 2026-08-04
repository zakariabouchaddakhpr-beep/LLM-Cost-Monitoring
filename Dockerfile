FROM python:3.12-slim

# Unbuffered so container logs appear immediately rather than sitting in a buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# gcc is sometimes needed to build wheels for compiled Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The library is deliberately not pip-installed: WORKDIR is /app and every
# command runs from there, so Python already finds `config` and `core`.

# results/ holds the example JSON call logs; it is normally a bind mount. The
# trace database itself lives in the clickhouse service, not here. uid 1000
# usually matches the host user who owns ./results, which keeps it writable.
RUN mkdir -p /app/results \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# CMD, not ENTRYPOINT, so docker-compose can replace it per service.
CMD ["python", "main.py"]
