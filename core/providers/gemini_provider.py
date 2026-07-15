import litellm

from core.base_provider import LLMProvider
from config.settings import DEFAULT_MODELS


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = None):
        self._model = model or DEFAULT_MODELS["gemini"]

    def get_provider_name(self) -> str:
        return "gemini"

    def get_model_string(self) -> str:
        return self._model

    def call(self, prompt: str):
        return litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}]
        )