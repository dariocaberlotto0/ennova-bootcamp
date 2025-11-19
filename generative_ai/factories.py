from abstracts import GenerativeAIClient

from providers import Provider
from clients import GoogleGenAIClient, OpenAIClient, OpenAIClientAsync

# Factory to create GenerativeAIClient instances based on provider
class GenerativeAIClientFactory:
    @staticmethod
    def create_client(provider: Provider, api_key: str) -> GenerativeAIClient:
        if provider == Provider.GOOGLE:
            return GoogleGenAIClient(api_key=api_key)
        elif provider == Provider.OPENAI:
            return OpenAIClient(api_key=api_key)
        elif provider == Provider.OPENAI_ASYNC:
            return OpenAIClientAsync(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
