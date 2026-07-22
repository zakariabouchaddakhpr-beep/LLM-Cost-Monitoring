import litellm

from core.base_provider import LLMProvider


class CerebrasProvider(LLMProvider):
    def __init__(self, model: str):
        self._model = model

    def get_provider_name(self) -> str:
        return "cerebras"

    def get_model_string(self) -> str:
        return self._model

    def call(self, prompt: str):
        return litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}]
        )