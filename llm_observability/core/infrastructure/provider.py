import litellm


class LiteLLMProvider:

    def __init__(self, model: str, api_key: str | None = None):
        self._model = model

        self._api_key = api_key

    def _auth(self) -> dict:
        return {"api_key": self._api_key} if self._api_key else {}

    def call(self, prompt: str):
        return litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **self._auth(),
        )

    def embed(self, inputs: list[str]):
        return litellm.embedding(model=self._model, input=inputs,
                                 **self._auth())

    def get_model_string(self) -> str:
        return self._model
