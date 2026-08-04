"""
Single provider adapter around LiteLLM.

LiteLLM already handles provider routing via the model-string prefix
(e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet"), so there is no
need for a separate class per provider. This is the only adapter
LLMMonitor needs.
"""

import litellm


class LiteLLMProvider:
    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        # Held on the instance and passed per call rather than written into
        # os.environ: a library that mutates the process environment leaks into
        # everything else running in it. None means "let LiteLLM find the key
        # itself", which is the existing behaviour.
        self._api_key = api_key

    def call(self, prompt: str):
        """Send prompt to LiteLLM and return the raw response object."""
        extra = {"api_key": self._api_key} if self._api_key else {}
        return litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )

    def embed(self, inputs: list[str]):
        """
        Send inputs to LiteLLM's embedding endpoint and return the raw response.

        A list always, even for a single string, because that is the shape every
        provider's embedding API takes — and it keeps the batch case, which is
        the common one for indexing, from being a special path.
        """
        extra = {"api_key": self._api_key} if self._api_key else {}
        return litellm.embedding(model=self._model, input=inputs, **extra)

    def get_model_string(self) -> str:
        """The litellm model string actually used."""
        return self._model
