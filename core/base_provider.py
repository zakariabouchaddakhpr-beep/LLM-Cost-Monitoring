"""
The common interface every provider adapter must implement.

This is the core of the Factory pattern here: because every provider
(Claude, GPT, Gemini, Groq, Cerebras...) exposes the exact same three
methods, the rest of the library (retry logic, metrics, the Monitor
class) can work with ANY provider without ever needing to know which
one it actually is. Add a new provider by writing one small class that
implements this interface — nothing else in the library needs to change.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def call(self, prompt: str):
        """
        Sends the prompt to the underlying model and returns the raw
        litellm-style response object (the same shape regardless of
        which real provider answered).
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Short identifier, e.g. 'anthropic', used in logs."""
        pass

    @abstractmethod
    def get_model_string(self) -> str:
        """The litellm model string actually used, e.g. 'anthropic/claude-3-5-sonnet-20241022'."""
        pass