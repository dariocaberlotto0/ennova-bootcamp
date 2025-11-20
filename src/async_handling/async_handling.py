import asyncio
import time
import random
import os
from google.genai import Client
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', depending on availability
import matplotlib.pyplot as plt
import tkinter as tk

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
else:
    client = Client(api_key=GOOGLE_API_KEY)
# Store all events in a timeline for later analysis
async_timeline_events = []
# Keep track of background tasks we'll gather at the end
tasks = []

# Capture the start time to calculate relative timestamps
initial_timestamp = time.monotonic()

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

async def async_llm_call(prompt):
    """
    Simulate a call to a Large Language Model API
    
    In real applications, this would be a network request to an API like
    OpenAI, Anthropic, or a self-hosted LLM service.
    """
    task_id = f"llm_{prompt[:10]}"
    log_event(async_timeline_events, task_id, "START") # just logging time
    # LLM calls often take longer than other API calls
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="Write a poem about the ocean."
    )
    print(f"Google Response: {response.text}\n")
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

async def run_async_pipeline():
    """
    Main workflow that coordinates the execution of different async operations
    
    This demonstrates both:
    1. Dependent tasks (using 'await') - where we need results immediately
    2. Background tasks (using create_task) - where we can fire and forget
    """
    log_event(async_timeline_events, "main_pipeline", "START_PIPELINE")

    # DEPENDENT TASK: We need the LLM result before proceeding
    llm_result = await async_llm_call("What's the user intent?")
    # BACKGROUND TASK: Log the result to the database, but don't wait for it
    tasks.append(asyncio.create_task(async_db_operation("llm_result")))

    # DEPENDENT TASK: We need this HTTP result for our business logic
    http1 = await async_http_call("/api/data")
    # BACKGROUND TASK: Log this API call, but don't block on it
    tasks.append(asyncio.create_task(async_db_operation("http1")))

    # DEPENDENT TASK: We need this data too before proceeding
    http2 = await async_http_call("/api/details")
    # BACKGROUND TASK: Another non-blocking logging operation
    tasks.append(asyncio.create_task(async_db_operation("http2")))

    # DEPENDENT TASK: Generate a summary using the LLM with collected data
    summary = await async_llm_call("Summarize everything")
    # BACKGROUND TASK: Log the summary generation
    tasks.append(asyncio.create_task(async_db_operation("summary")))

    log_event(async_timeline_events, "main_pipeline", "END_PIPELINE")

    # Wait for all background tasks to complete before exiting
    # This ensures all logging operations finish properly
    # This will run without gathering all tasks as well, you can try
    await asyncio.gather(*tasks)


# # Fix for running asyncio in environments like Jupyter notebooks
# import nest_asyncio
# nest_asyncio.apply()

# Reset the timer before starting
initial_timestamp = time.monotonic() 

print("\n--- Running Async Pipeline ---\n")
# Run the main async workflow
asyncio.run(run_async_pipeline())

# Timeline to track events for visualization and analysis
sync_timeline_events = []
# Capture start time for calculating relative timestamps
initial_timestamp = time.monotonic()

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

def sync_llm_call(prompt):
    """
    Simulate a synchronous call to a Large Language Model API
    
    In real applications, this would be waiting for a response from
    an AI service like OpenAI or Anthropic, completely blocking execution.
    """
    task_id = f"llm_{prompt[:10]}"
    log_event(sync_timeline_events, task_id, "START")
    # Block the entire program during this "API call"
    response = client.models.generate_content(model="gemini-2.5-flash",
        contents="Write a poem about the ocean."
    )
    print(f"Google Response: {response.text}\n")
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

def run_sync_pipeline():
    """
    Main workflow that coordinates operations in a strictly sequential manner
    
    Each operation completely blocks the program until it finishes,
    resulting in poor resource utilization and longer total execution time.
    """
    log_event(sync_timeline_events, "main_pipeline", "START_PIPELINE")

    # Step 1: Make LLM call and wait for it to complete
    llm_result = sync_llm_call("What's the user intent?")
    # Step 2: Log the result and wait for logging to complete
    sync_db_operation("llm_result")

    # Step 3: Make first HTTP call and wait for it to complete
    http1 = sync_http_call("/api/data")
    # Step 4: Log the HTTP result and wait for logging to complete
    sync_db_operation("http1")

    # Step 5: Make second HTTP call and wait for it to complete
    http2 = sync_http_call("/api/details")
    # Step 6: Log the HTTP result and wait for logging to complete
    sync_db_operation("http2")

    # Step 7: Make final LLM call and wait for it to complete
    summary = sync_llm_call("Summarize everything")
    # Step 8: Log the summary and wait for logging to complete
    sync_db_operation("summary")

    log_event(sync_timeline_events, "main_pipeline", "END_PIPELINE")

# Reset the timer before starting
initial_timestamp = time.monotonic()

print("\n--- Running Sync Pipeline ---\n")
# Run the synchronous workflow
run_sync_pipeline()