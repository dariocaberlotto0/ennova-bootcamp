from dataclasses import dataclass
from decorators import timed, timed_async

from abstracts import GenerativeAIClient

from google import genai
from google.genai.types import GenerateContentConfig

from openai import OpenAI, AsyncOpenAI

@dataclass
class GoogleGenAIClient(GenerativeAIClient):

    api_key: str
    model: str='gemini-2.5-flash'
    temperature: float=0.7
    max_tokens: int=2000
    retries: int=3
    backoff: float=0.8

    # Initialize Google GenAI client
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        self.config = GenerateContentConfig(temperature=self.temperature, max_output_tokens=self.max_tokens)

    def response_check(self, response):
        if response.candidates is None:
            raise Exception("No candidates")
        candidate = response.candidates[0]
        if str(candidate.finish_reason) == "FinishReason.STOP":
            if not response or not response.text:
                raise Exception("No valid response from API")
            return response.text
        return str(candidate.finish_reason)

    @timed_async
    async def generate_async(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")

        response = await self.client.aio.models.generate_content(model=self.model, contents=prompt, config=self.config)
        return self.response_check(response)

    # Generate text using Google's generative AI API
    @timed
    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")

        response = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
        return self.response_check(response)
    

@dataclass
class OpenAIClientAsync(GenerativeAIClient):
    
    api_key: str
    model: str='gpt-4o-mini'
    temperature: float=0.7
    max_tokens: int=2000
    retries: int=3
    backoff: float=0.8

    # Initialize OpenAI client
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPEN_API_KEY environment variable not set")
        
        self.client = AsyncOpenAI(api_key=self.api_key)

    def response_check(self, response):
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

    @timed_async
    async def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self.response_check(response)


@dataclass
class OpenAIClient(GenerativeAIClient):

    api_key: str
    model: str='gpt-4o-mini'
    temperature: float=0.7
    max_tokens: int=2000
    retries: int=3
    backoff: float=0.8

    # Initialize OpenAI client
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("OPEN_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def response_check(self, response):
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

    # Generate text using OpenAI's chat completion API
    @timed
    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self.response_check(response)
