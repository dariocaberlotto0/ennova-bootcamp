import time
import asyncio
from google import genai
import openai

from providers import Provider
from factories import GenerativeAIClientFactory
from environment import Environment
from async_pipeline import AsyncPipeline
from sync_pipeline import SyncPipeline
from plot_generating import plot_gantt_chart

# ==================================================================================

async def main():
    GOOGLE_API_KEY, OPEN_API_KEY = Environment.load()
    if not GOOGLE_API_KEY:
        print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
    else:
        google_client = genai.Client(api_key=GOOGLE_API_KEY)

    if not OPEN_API_KEY:
        print('⚠️ Set OPEN_API_KEY in your environment to run live calls.')
    else:
        async_openAI_client = openai.AsyncOpenAI(api_key=OPEN_API_KEY)
        openAI_client = openai.OpenAI(api_key=OPEN_API_KEY)

    # ========================== Async Pipeline ====================================

    async_pipeline = AsyncPipeline()
    # start time measurement
    initial_timestamp = time.monotonic() 
    print("\n--------------- Running Async Pipeline ---------------\n")
    async_timeline_events = await async_pipeline.run_async_pipeline(
            initial_timestamp, google_client, async_openAI_client)

    # ========================== Sync Pipeline =====================================

    sync_pipeline = SyncPipeline()
    # start time measurement
    initial_timestamp = time.monotonic()
    # passing to sync openai client
    if not OPEN_API_KEY:
        raise ValueError("OPEN_API_KEY is required for synchronous OpenAI client.")
    openAI_client = openai.OpenAI(api_key=OPEN_API_KEY)
    print("\n--------------- Running Sync Pipeline ---------------\n")
    sync_timeline_events = sync_pipeline.run_sync_pipeline(
        initial_timestamp, google_client, openAI_client)

    # ============================== Plotting ======================================

    plot_gantt_chart(async_timeline_events, title="Async Events Timeline")
    plot_gantt_chart(sync_timeline_events, title="Sync Events Timeline")

# ==================================================================================

"""
async def main(): 
    openai_async = True
    genai_async = True

    GOOGLE_API_KEY, OPEN_API_KEY = Environment.load()
    
    if not GOOGLE_API_KEY:
        print('⚠️ Set GOOGLE_API_KEY in your environment to run live calls.')
    else:
        google_client = GenerativeAIClientFactory.create_client(
        Provider.GOOGLE, GOOGLE_API_KEY)

    if not OPEN_API_KEY:
        print('⚠️ Set OPEN_API_KEY in your environment to run live calls.')
    else:
        if openai_async:
            openAI_client = GenerativeAIClientFactory.create_client(
            Provider.OPENAI_ASYNC, OPEN_API_KEY)
        else:
            openAI_client = GenerativeAIClientFactory.create_client(
            Provider.OPENAI, OPEN_API_KEY)

    # ================================== GoogleAI ==================================

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

    # =================================== OpenAI ===================================

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

# ==================================================================================

if __name__ == "__main__":
    asyncio.run(main())
