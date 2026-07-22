"""
The common interface every provider adapter must implement.

This is the core of the Factory pattern here: because every provider
exposes the exact same threemethods, the rest of the library 
can work with ANY provider without ever needing to know which
one it actually is. Add a new provider by writing one small class that
implements this interface — nothing else in the library needs to change.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def call(self, prompt: str):
        """
        Sends the prompt to the underlying model and 
        returns the raw litellm-style response object
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Short identifier, used in logs."""
        pass

    @abstractmethod
    def get_model_string(self) -> str:
        """The litellm model string actually used"""
        pass