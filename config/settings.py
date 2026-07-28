"""
Configuration for the library: environment variables and constants.
No logic here — just settings other modules import.

Provider/model are NOT set here — they're set directly in quickstart.py
(or wherever LLMMonitor is instantiated), since that's the one place
that actually decides which LLM is being called for a given run.
"""

import os
from dotenv import load_dotenv

try:
    load_dotenv()
except OSError:
    # If a local .env file cannot be read (for example, a OneDrive placeholder
    # file that is not downloaded), continue with environment variables.
    pass

# Model used as the "judge" to score response quality (optional feature).
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"


MAX_RETRIES = 2              # number of retries AFTER the first attempt
RETRY_BACKOFF_SECONDS = 1.5  # base delay, doubled on each subsequent retry

# OpenTelemetry configuration
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llm-monitor")

# DuckDB persistence
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "results/llm_traces.duckdb")
