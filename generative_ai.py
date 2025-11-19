from collections import defaultdict
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

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_gantt_chart(timeline_events, title="Gantt Chart"):
    """
    Plot a Gantt chart of tasks and their durations using Matplotlib.
    
    The 'main_pipeline' end time is extended to match the completion of the 
    absolute last task in the timeline.

    Args:
        timeline_events: List of event dictionaries with 'task', 'event', and 'timestamp'.
        title: Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Initialize data structures
    task_start_end = defaultdict(lambda: {'start': None, 'end': None})
    # Variable to track the latest completion time across all tasks
    absolute_latest_end_time = 0.0

    # 2. Iterate through the event list to find the start/end times and the latest completion time
    for event in timeline_events:
        task_name = event['task']
        timestamp = event['timestamp']
        event_type = event['event']
        
        # Standardizing event types (mapping specific ones like START_PIPELINE to START)
        if 'START' in event_type:
            # Only record the very first start time for a task
            if task_start_end[task_name]['start'] is None:
                task_start_end[task_name]['start'] = timestamp
                
        elif 'END' in event_type:
            # Record the latest end time for a task (if it has multiple ends)
            if task_start_end[task_name]['end'] is None:
                task_start_end[task_name]['end'] = timestamp
            else:
                task_start_end[task_name]['end'] = max(task_start_end[task_name]['end'], timestamp)
                
                # Track the overall latest completion time (important for main_pipeline alignment)
            if task_start_end[task_name]['end'] > absolute_latest_end_time:
                absolute_latest_end_time = task_start_end[task_name]['end']

    # --- NEW LOGIC: Ensure main_pipeline ends with the absolute last operation ---
    main_pipeline_name = 'main_pipeline' # Assuming this is the top-level task name
    
    if absolute_latest_end_time > 0 and main_pipeline_name in task_start_end:
        # Check if the pipeline's currently recorded end is earlier than the actual latest end time
        pipeline_end_time = task_start_end[main_pipeline_name].get('end')
        
        if pipeline_end_time is None or absolute_latest_end_time > pipeline_end_time:
            # Override the main_pipeline's end time with the absolute latest completion time
            task_start_end[main_pipeline_name]['end'] = absolute_latest_end_time
    # --------------------------------------------------------------------------

    # 3. Filter out tasks and finalize the lists
    tasks = []
    start_times = []
    end_times = []

    for task_name, times in task_start_end.items():
        if times['start'] is not None and times['end'] is not None:
            tasks.append(task_name)
            start_times.append(times['start'])
            end_times.append(times['end'])
    
    # Calculate duration (width of the bar)
    durations = [(end - start) for start, end in zip(start_times, end_times)]
    
    # Determine the task positions on the Y-axis
    y_positions = list(range(len(tasks)))
    
    # 2. Plot the horizontal bars
    ax.barh(y_positions, 
            durations, 
            left=start_times, 
            height=0.6, 
            align='center',
            color='skyblue') 

    # 3. Configure the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()  # To display tasks from top to bottom

    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    
    # Optional: Add gridlines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

plot_gantt_chart(async_timeline_events, title="Async Events Timeline")
plot_gantt_chart(sync_timeline_events, title="Sync Events Timeline")


"""if __name__ == "__main__":
    asyncio.run(main())
"""