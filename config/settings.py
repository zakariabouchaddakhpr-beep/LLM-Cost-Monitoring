"""
Configuration for the benchmark: environment variables and constants.
No logic here — just settings other modules import.
"""

from dotenv import load_dotenv

load_dotenv()

# Model used as the "judge" to score response quality.
# Kept separate from the providers under test to reduce self-scoring bias.
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

# Providers under test: (provider_name, model_string)
PROVIDERS = [
    ("groq", "groq/llama-3.3-70b-versatile"),
    ("gemini", "gemini/gemini-3.5-flash"),
    ("cerebras", "cerebras/gpt-oss-120b"),
]