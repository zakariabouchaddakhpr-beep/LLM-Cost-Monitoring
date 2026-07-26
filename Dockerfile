FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (gcc sometimes needed for compiled Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Ensure the results/ folder exists and is writable
RUN mkdir -p /app/results

# Run the example when the container starts
CMD ["python", "main.py"]
