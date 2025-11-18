import os
from dataclasses import dataclass
import time
import asyncio
import random

# Decorator to time function execution
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution: {time.time() - start:.2f}s")
        return result
    return wrapper

def timed_async(func):
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        print(f"Execution: {time.time() - start:.2f}s")
        return result
    return wrapper

import abc

# Abstract base class for Generative AI clients
class GenerativeAIClient(abc.ABC):
    
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

from google import genai
from google.genai.types import GenerateContentConfig

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
    
from openai import OpenAI, AsyncOpenAI

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
    
from enum import Enum

class Provider(Enum):
    GOOGLE = 'google'
    OPENAI = 'openai'
    OPENAI_ASYNC = 'openai_async'

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

def log_event(timeline, task, event):
    """
    Log an event with timestamp to visualize when operations start and end
    You can simply ignore this this does not serve any pupose in the pipeline.
    Args:
        timeline: List to append the event to
        task: Name/ID of the task
        event: Description of what happened (e.g., START or END)
    """
    timestamp = time.monotonic() - initial_timestamp  # Calculate time since start
    timeline.append({
        "task": task,
        "event": event,
        "timestamp": timestamp,
    })
    print(f"[{timestamp:.2f}] {event}: {task}")


async def async_db_operation(step):
    """
    Simulate a database write operation (like saving analytics or logs)
    
    In real applications, this would be writing data to a database,
    which is I/O-bound and benefits from asynchronous execution.
    """
    task_id = f"log_{step}"
    log_event(async_timeline_events, task_id, "START") # just logging time
    # Simulate variable database operation time
    await asyncio.sleep(random.uniform(0.3, 3.0))
    log_event(async_timeline_events, task_id, "END") # just logging time

async def async_google_call(prompt, client):
    """
    Simulate a call to a Large Language Model API
    
    In real applications, this would be a network request to an API like
    OpenAI, Anthropic, or a self-hosted LLM service.
    """
    task_id = f"google_{prompt[:10]}"
    log_event(async_timeline_events, task_id, "START") # just logging time
    # LLM calls often take longer than other API calls
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, how are you?."
    )
    print(f"Google Response: {response.text}\n")
    log_event(async_timeline_events, task_id, "END") # just logging time
    return f"LLM result: {prompt}"

async def async_openai_call(prompt, client):
    """
    Simulate a call to a Large Language Model API
    
    In real applications, this would be a network request to an API like
    OpenAI, Anthropic, or a self-hosted LLM service.
    """
    task_id = f"openai_{prompt[:10]}"
    log_event(async_timeline_events, task_id, "START") # just logging time
    # LLM calls often take longer than other API calls
    response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello, how are you?."}
            ]
    )
    print(f"OpenAI Response: {response.text}\n")
    log_event(async_timeline_events, task_id, "END") # just logging time
    return f"LLM result: {prompt}"

async def async_http_call(endpoint):
    """
    Simulate an HTTP request to an external API
    
    In real applications, this could be fetching data from a REST API,
    microservice, or other web service.
    """
    task_id = f"http_{endpoint.replace('/', '')}"
    log_event(async_timeline_events, task_id, "START") # just logging time
    # Simulate variable network latency
    await asyncio.sleep(random.uniform(0.4, 4.0))
    log_event(async_timeline_events, task_id, "END") # just logging time
    return f"HTTP result from {endpoint}"

async def run_async_pipeline(google_client, openai_client):
    """
    Main workflow that coordinates the execution of different async operations
    
    This demonstrates both:
    1. Dependent tasks (using 'await') - where we need results immediately
    2. Background tasks (using create_task) - where we can fire and forget
    """
    log_event(async_timeline_events, "main_pipeline", "START_PIPELINE")

    # DEPENDENT TASK: We need the LLM result before proceeding
    google_result = await async_google_call("What's the user intent?", google_client)
    # BACKGROUND TASK: Log the result to the database, but don't wait for it
    tasks.append(asyncio.create_task(async_db_operation("google_result")))

    # DEPENDENT TASK: We need the LLM result before proceeding
    # openai_result = await async_openai_call("What's the user intent?", openai_client)
    # BACKGROUND TASK: Log the result to the database, but don't wait for it
    # tasks.append(asyncio.create_task(async_db_operation("openai_result")))

    # DEPENDENT TASK: We need this HTTP result for our business logic
    http1 = await async_http_call("/api/data")
    # BACKGROUND TASK: Log this API call, but don't block on it
    tasks.append(asyncio.create_task(async_db_operation("http1")))

    # DEPENDENT TASK: We need this data too before proceeding
    http2 = await async_http_call("/api/details")
    # BACKGROUND TASK: Another non-blocking logging operation
    tasks.append(asyncio.create_task(async_db_operation("http2")))

    # DEPENDENT TASK: Generate a summary using the LLM with collected data
    summary_google = await async_google_call("Summarize everything", google_client)
    # BACKGROUND TASK: Log the summary generation
    tasks.append(asyncio.create_task(async_db_operation("summary_google")))
    
    # DEPENDENT TASK: Generate a summary using the LLM with collected data
    # summary_openai = await async_openai_call("Summarize everything", openai_client)
    # BACKGROUND TASK: Log the summary generation
    # tasks.append(asyncio.create_task(async_db_operation("summary_openai")))

    log_event(async_timeline_events, "main_pipeline", "END_PIPELINE")

    # Wait for all background tasks to complete before exiting
    # This ensures all logging operations finish properly
    # This will run without gathering all tasks as well, you can try
    await asyncio.gather(*tasks)
    
def sync_db_operation(step):
    """
    Simulate a synchronous database write operation
    
    In real applications, this would be writing data to a database,
    but here we're blocking the entire program while it runs.
    """
    task_id = f"log_{step}"
    log_event(sync_timeline_events, task_id, "START")
    # Block the entire program during this "database operation"
    time.sleep(random.uniform(0.3, 3.0))
    log_event(sync_timeline_events, task_id, "END")

def sync_google_call(prompt, client):
    """
    Simulate a synchronous call to a Large Language Model API
    
    In real applications, this would be waiting for a response from
    an AI service like OpenAI or Anthropic, completely blocking execution.
    """
    task_id = f"google_{prompt[:10]}"
    log_event(sync_timeline_events, task_id, "START")
    # Block the entire program during this "API call"
    response = client.models.generate_content(model="gemini-2.5-flash",
        contents="Hello, how are you?."
    )
    print(f"Google Response: {response.text}\n")
    log_event(sync_timeline_events, task_id, "END")
    return f"LLM result: {prompt}"

def sync_openai_call(prompt, client):
    """
    Simulate a synchronous call to a Large Language Model API
    
    In real applications, this would be waiting for a response from
    an AI service like OpenAI or Anthropic, completely blocking execution.
    """
    task_id = f"openai_{prompt[:10]}"
    log_event(sync_timeline_events, task_id, "START")
    # Block the entire program during this "API call"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
                "role": "user",
                "content": "Hello, how are you?."
        }]
    )
    print(f"OpenAI Response: {response.text}\n")
    log_event(sync_timeline_events, task_id, "END")
    return f"LLM result: {prompt}"

def sync_http_call(endpoint):
    """
    Simulate a synchronous HTTP request to an external API
    
    In real applications, this would be waiting for a network response,
    with the program doing nothing else during this time.
    """
    task_id = f"http_{endpoint.replace('/', '')}"
    log_event(sync_timeline_events, task_id, "START")
    # Block the entire program during this "network request"
    time.sleep(random.uniform(0.4, 4.0))
    log_event(sync_timeline_events, task_id, "END")
    return f"HTTP result from {endpoint}"

def run_sync_pipeline(google_client, openai_client):
    """
    Main workflow that coordinates operations in a strictly sequential manner
    
    Each operation completely blocks the program until it finishes,
    resulting in poor resource utilization and longer total execution time.
    """
    log_event(sync_timeline_events, "main_pipeline", "START_PIPELINE")

    # Step 1: Make LLM call and wait for it to complete
    google_result = sync_google_call("What's the user intent?", google_client)
    # Step 2: Log the result and wait for logging to complete
    sync_db_operation("llm_result")
    
    # Step 1: Make LLM call and wait for it to complete
    # openai_result = sync_openai_call("What's the user intent?", openai_client)
    # Step 2: Log the result and wait for logging to complete
    # sync_db_operation("llm_result")

    # Step 3: Make first HTTP call and wait for it to complete
    http1 = sync_http_call("/api/data")
    # Step 4: Log the HTTP result and wait for logging to complete
    sync_db_operation("http1")

    # Step 5: Make second HTTP call and wait for it to complete
    http2 = sync_http_call("/api/details")
    # Step 6: Log the HTTP result and wait for logging to complete
    sync_db_operation("http2")

    # Step 7: Make final LLM call and wait for it to complete
    google_summary = sync_google_call("Summarize everything", google_client)
    # Step 8: Log the summary and wait for logging to complete
    sync_db_operation("summary")
    
    # Step 7: Make final LLM call and wait for it to complete
    # openai_summary = sync_openai_call("Summarize everything", openai_client)
    # Step 8: Log the summary and wait for logging to complete
    # sync_db_operation("summary")

    log_event(sync_timeline_events, "main_pipeline", "END_PIPELINE")


"""async def main(): 
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    openai_async = True
    genai_async = True

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
    else:
        google_client = GenerativeAIClientFactory.create_client(Provider.GOOGLE, GOOGLE_API_KEY)

    OPEN_API_KEY = os.getenv('OPEN_API_KEY')
    if not OPEN_API_KEY:
        print('⚠️ Set OPEN_API_KEY in your environment to run live calls.')
    else:
        if openai_async:
            openAI_client = GenerativeAIClientFactory.create_client(Provider.OPENAI_ASYNC, OPEN_API_KEY)
        else:
            openAI_client = GenerativeAIClientFactory.create_client(Provider.OPENAI, OPEN_API_KEY)


    if genai_async:
        prompts = [
            "Explain the concept of asynchronous programming.",
            "Write a poem about the ocean.",
            "Summarize the key points of quantum computing."
        ]
        tasks = [google_client.generate_async(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"---------------------- Response {i+1} -----------------------")
            print(f"{result}")
    else:
        response_google = google_client.generate("Hello, how are you?")
        print(f"Google Response: {response_google}\n")


    if openai_async:
        prompts = [
            "Explain the concept of asynchronous programming.",
            "Write a poem about the ocean.",
            "Summarize the key points of quantum computing."
        ]
        tasks = [openAI_client.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"---------------------- Response {i+1} -----------------------")
            print(f"{result}")
    else:
        response_openai = openAI_client.generate("Hello, how are you?")
        print(f"OpenAI Response: {response_openai}")
    """
    

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

openai_async = True
genai_async = True

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
else:
    # google_client = GenerativeAIClientFactory.create_client(Provider.GOOGLE, GOOGLE_API_KEY)
    google_client = genai.Client(api_key=GOOGLE_API_KEY)

"""OPEN_API_KEY = os.getenv('OPEN_API_KEY')
if not OPEN_API_KEY:
    print('⚠️ Set OPEN_API_KEY in your environment to run live calls.')
else:
    openAI_client = GenerativeAIClientFactory.create_client(Provider.OPENAI_ASYNC, OPEN_API_KEY)
"""   
   
# Store all events in a timeline for later analysis
async_timeline_events = []
# Keep track of background tasks we'll gather at the end
tasks = []

# Capture the start time to calculate relative timestamps
initial_timestamp = time.monotonic()


# # Fix for running asyncio in environments like Jupyter notebooks
# import nest_asyncio
# nest_asyncio.apply()

# Reset the timer before starting
initial_timestamp = time.monotonic() 

print("\n--- Running Async Pipeline ---\n")
# Run the main async workflow
asyncio.run(run_async_pipeline(google_client, """openAI_client"""))

# Timeline to track events for visualization and analysis
sync_timeline_events = []
# Capture start time for calculating relative timestamps
initial_timestamp = time.monotonic()

# Reset the timer before starting
initial_timestamp = time.monotonic()

# openAI_client = GenerativeAIClientFactory.create_client(Provider.OPENAI, OPEN_API_KEY)

print("\n--- Running Sync Pipeline ---\n")
# Run the synchronous workflow
run_sync_pipeline(google_client, """openAI_client""")

import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', depending on availability
import matplotlib.pyplot as plt
import tkinter as tk

# Visualization of the timelines using Matplotlib
def plot_timeline(timeline_events, title):
    """
    Plot a timeline of events using Matplotlib
    
    Args:
        timeline_events: List of event dictionaries with 'task', 'event', and 'timestamp'
        title: Title of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign a unique y-position for each task
    tasks = list(set(event['task'] for event in timeline_events))
    task_y_positions = {task: i for i, task in enumerate(tasks)}

    for event in timeline_events:
        y = task_y_positions[event['task']]
        color = 'green' if event['event'] == 'END' else 'red'
        ax.scatter(event['timestamp'], y, color=color, s=100)
        ax.text(event['timestamp'], y + 0.1, f"{event['event']}", ha='center')

    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    plt.grid(True)
    plt.show()

print("\n--- Plotting Timelines ---\n")
plot_timeline(async_timeline_events, "Asynchronous Pipeline Timeline")
plot_timeline(sync_timeline_events, "Synchronous Pipeline Timeline")
    
"""if __name__ == "__main__":
    asyncio.run(main())
"""