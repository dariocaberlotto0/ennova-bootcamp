from .llm import LLMClient
from main import GOOGLE_KEY, OPENAI_KEY

import logging

# Factory to create llm instances based on provider
class ClientFactory:

    @staticmethod
    def create_client(provider: str) -> LLMClient:
        if provider == "gemini":
            from .llm import GeminiLLM
            logging.info("Creating GeminiLLM")
            return GeminiLLM(api_key=GOOGLE_KEY)
        elif provider == "openai":
            from .llm import OpenAILLM
            logging.info("Creating OpenAILLM")
            return OpenAILLM(api_key=OPENAI_KEY)
        elif provider == "openaiasync":
            from .llm import OpenAILLMAsync
            logging.info("Creating OpenAILLMAsync")
            return OpenAILLMAsync(api_key=OPENAI_KEY)
        elif provider == "mock":
            from .llm import MockLLM
            logging.info("Creating MockLLM")
            return MockLLM()
        else:
            raise ValueError(f"Unknown provider: {provider}")
