from abstracts import GenerativeAIClient
from providers import Provider

import logging

# Factory to create GenerativeAIClient instances based on provider
class GenerativeAIClientFactory:
    @staticmethod
    def create_client(provider: Provider, api_key: str) -> GenerativeAIClient:
        if provider == Provider.GOOGLE:
            from clients import GoogleGenAIClient
            logging.info("Creationg GoogleGenAIClient")
            return GoogleGenAIClient(api_key=api_key)
        elif provider == Provider.OPENAI:
            from clients import OpenAIClient
            logging.info("Creationg OpenAIClient")
            return OpenAIClient(api_key=api_key)
        elif provider == Provider.OPENAI_ASYNC:
            from clients import OpenAIClientAsync
            logging.info("Creationg OpenAIClientAsync")
            return OpenAIClientAsync(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
