import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')

OPEN_API_KEY = os.getenv('OPEN_API_KEY')
if not OPEN_API_KEY:
    print('⚠️ Set OPEN_API_KEY in your environment to run live calls.')

import abc

class GenerativeAIClient(abc.ABC):
    
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

from google import genai
from google.genai.types import GenerateContentConfig

class GoogleGenAIClient(GenerativeAIClient):

    def __init__(self, model: str='gemini-2.5-flash', temperature: float=0.7, max_tokens: int=200, retries: int=3, backoff: float=0.8):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff

    def generate(self, prompt: str) -> str:
        """Call the chat completion API with basic retries and timing.
        Returns the model's answer as plain text.
        """

        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")

        client = genai.Client()
        config = GenerateContentConfig(temperature=self.temperature, max_output_tokens=self.max_tokens)

        response = client.models.generate_content(model=self.model, contents=prompt, config=config)
        
        if response.candidates is None:
            raise Exception("No candidates")
        candidate = response.candidates[0]
        if str(candidate.finish_reason) == "FinishReason.STOP":
            if not response or not response.text:
                raise Exception("No valid response from API")
            return response.text
        raise Exception(str(candidate.finish_reason))
    
from openai import OpenAI

class OpenAIClient(GenerativeAIClient):

    def __init__(self, model: str='gpt-4o-mini', temperature: float=0.7, max_tokens: int=200, retries: int=3, backoff: float=0.8):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff
    
    def generate(self, prompt: str) -> str:
        """Call the chat completion API with basic retries and timing.
        Returns the model's answer as plain text.
        """

        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")
        
        client = OpenAI(api_key=OPEN_API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        if response is None:
            raise ValueError("No response from API")
        choices = response.choices
        if not choices or len(choices) == 0:
            raise ValueError("Failed to get a valid response from API")
        first_choice = choices[0]
        message = first_choice.message
        if message.role != "assistant":
            raise ValueError("Invalid role in response")
        if not message.content:
            reason = message.refusal
            raise ValueError("No content in response: " + str(reason))
        
        return message.content
    
from enum import Enum

class Provider(Enum):
    GOOGLE = 'google'
    OPENAI = 'openai'

class GenerativeAIClientFactory:
    @staticmethod
    def create_client(provider: Provider) -> GenerativeAIClient:
        if provider == Provider.GOOGLE:
            return GoogleGenAIClient()
        elif provider == Provider.OPENAI:
            return OpenAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

def main(): 
    google_client = GenerativeAIClientFactory.create_client(Provider.GOOGLE)
    openAI_client = GenerativeAIClientFactory.create_client(Provider.OPENAI)

    response_google = google_client.generate("Hello, how are you?")
    response_openai = openAI_client.generate("Hello, how are you?")

    print(f"Google Response: {response_google}\n")
    print(f"OpenAI Response: {response_openai}")

if __name__ == "__main__":
    main()