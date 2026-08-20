"""
The one place in the project that actually talks to an AI company's API.

There is no separate class for OpenAI, another for Anthropic, another for
Groq. We don't need them, because the LiteLLM library already handles that.
You tell LiteLLM which company you want by putting its name in front of the
model name, separated by a slash:

    "openai/gpt-4o"
    "anthropic/claude-3-5-sonnet"
    "groq/llama-3.3-70b-versatile"

LiteLLM reads the part before the slash, sends the request to the right
company, and gives every company's answer back in the same shape. So one
small class here is enough for all of them.
"""

import litellm


class LiteLLMProvider:
    """Sends prompts to an AI model and hands back the raw reply."""

    def __init__(self, model: str, api_key: str | None = None):
        self._model = model

        # Why store the key here instead of putting it in os.environ (the
        # program's shared settings)? Because os.environ is shared by
        # EVERYTHING running in the program. A library that quietly writes
        # into it can surprise completely unrelated code. So we keep the key
        # to ourselves and hand it over on each call.
        #
        # None means "we weren't given a key". LiteLLM will then go looking
        # for one itself, usually in a setting named something like
        # OPENAI_API_KEY.
        self._api_key = api_key

    def _auth(self) -> dict:
        """
        Package the API key up so it can be added to a call.

        Gives back either {"api_key": "sk-..."} or an empty {}.

        This exists so the two methods below can both write
        `litellm.completion(..., **self._auth())` and do the right thing
        whether we have a key or not.

        The ** in front of a dictionary unpacks it into named arguments. So
        **{"api_key": "sk-123"} becomes api_key="sk-123", and **{} adds
        nothing at all.
        """
        return {"api_key": self._api_key} if self._api_key else {}

    def call(self, prompt: str):
        """Send one question to the model and hand back its raw reply."""
        return litellm.completion(
            model=self._model,
            # Chat APIs expect a list of messages, not just a piece of text,
            # because they are built for back-and-forth conversations. One
            # message from the "user" is the shortest conversation there is.
            messages=[{"role": "user", "content": prompt}],
            **self._auth(),
        )

    def embed(self, inputs: list[str]):
        """
        Turn a list of texts into lists of numbers, and hand back the reply.

        An "embedding" converts text into numbers that capture its meaning,
        so that similar texts end up with similar numbers. That is how search
        and recommendation features work.

        This is a completely different API endpoint from chat, and it is
        usually far cheaper.
        """
        return litellm.embedding(model=self._model, input=inputs,
                                 **self._auth())

    def get_model_string(self) -> str:
        """Give back the full model name we were set up with."""
        return self._model
