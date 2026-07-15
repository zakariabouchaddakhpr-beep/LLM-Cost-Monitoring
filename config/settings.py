"""
Configuration for the library: environment variables and constants.
No logic here — just settings other modules import.
"""

from dotenv import load_dotenv

load_dotenv()

# Model used as the "judge" to score response quality (optional feature).
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

# Instrumentation-layer retry behavior for transient failures
# (rate limits, brief timeouts, connection blips).
MAX_RETRIES = 2              # number of retries AFTER the first attempt
RETRY_BACKOFF_SECONDS = 1.5  # base delay, doubled on each subsequent retry

# Default model used by each provider adapter if the caller doesn't
# specify one explicitly. Centralized here so changing a default model
# doesn't require touching provider code.
# NOTE: verify exact model strings against LiteLLM's supported models
# list before relying on this in production — provider model IDs
# (especially version-dated ones) change over time.
DEFAULT_MODELS = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-5-sonnet-20241022",
    "gemini": "gemini/gemini-3.5-flash",
    "groq": "groq/llama-3.3-70b-versatile",
    "cerebras": "cerebras/gpt-oss-120b",
}