from .llm import LLMClient
import os

import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    logging.error(e)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logging.warning("⚠️ Set GOOGLE_API_KEY in your environment to run live calls.")
else:
    logging.info("✅ GOOGLE_API_KEY is set.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.warning("⚠️ Set GOOGLE_API_KEY in your environment to run live calls.")
else:
    logging.info("✅ OPENAI_API_KEY is set.")

# Factory to create llm instances based on provider
class ClientFactory:

    @staticmethod
    def create_client(provider: str) -> LLMClient:
        if provider == "gemini":
            from .llm import GeminiLLM
            logging.info("Creating GeminiLLM")
            return GeminiLLM(api_key=GOOGLE_API_KEY)
        elif provider == "openai":
            from .llm import OpenAILLM
            logging.info("Creating OpenAILLM")
            return OpenAILLM(api_key=OPENAI_API_KEY)
        elif provider == "openaiasync":
            from .llm import OpenAILLMAsync
            logging.info("Creating OpenAILLMAsync")
            return OpenAILLMAsync(api_key=OPENAI_API_KEY)
        elif provider == "mock":
            from .llm import MockLLM
            logging.info("Creating MockLLM")
            return MockLLM()
        else:
            raise ValueError(f"Unknown provider: {provider}")
