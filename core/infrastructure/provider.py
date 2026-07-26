"""
Single provider adapter around LiteLLM.

LiteLLM already handles provider routing via the model-string prefix
(e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet"), so there is no
need for a separate class per provider. This is the only adapter
LLMMonitor needs.
"""

import litellm


class LiteLLMProvider:
    def __init__(self, model: str):
        self._model = model

    def call(self, prompt: str):
        """Send prompt to LiteLLM and return the raw response object."""
        return litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}]
        )

    def get_model_string(self) -> str:
        """The litellm model string actually used."""
        return self._model
