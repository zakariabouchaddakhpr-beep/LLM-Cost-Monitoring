"""
Factory pattern: given a provider name (a string), return an instance
of the correct provider class — without the caller needing to import
or know about that specific class.

This is what makes the library provider-agnostic: adding a new provider
means writing one small adapter class and registering it here. Nothing
in core/monitor.py, metrics.py, or quality_scoring.py needs to change.
"""

from core.providers.openai_provider import OpenAIProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.gemini_provider import GeminiProvider
from core.providers.groq_provider import GroqProvider
from core.providers.cerebras_provider import CerebrasProvider


class LLMProviderFactory:

    _providers = {}  # {name: class}, populated by register_provider below

    @classmethod
    def register_provider(cls, name: str, provider_class):
        cls._providers[name] = provider_class

    @classmethod
    def get_provider(cls, name: str, **provider_kwargs):
        """
        Returns a NEW instance of the requested provider.
        provider_kwargs are passed straight to the provider's constructor
        (e.g. model="anthropic/claude-3-opus-..." to override the default).
        """
        provider_class = cls._providers.get(name)
        if not provider_class:
            raise ValueError(
                f"Provider '{name}' not found. Available: {list(cls._providers.keys())}"
            )
        return provider_class(**provider_kwargs)

    @classmethod
    def list_providers(cls):
        return list(cls._providers.keys())


# Registration — this is what makes each provider name usable via get_provider().
LLMProviderFactory.register_provider("openai", OpenAIProvider)
LLMProviderFactory.register_provider("anthropic", AnthropicProvider)
LLMProviderFactory.register_provider("gemini", GeminiProvider)
LLMProviderFactory.register_provider("groq", GroqProvider)
LLMProviderFactory.register_provider("cerebras", CerebrasProvider)