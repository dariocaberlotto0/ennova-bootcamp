import asyncio
import random
import time

from environment import Environment

class AsyncPipeline:

    def log_event(self, timeline, task, event, initial_timestamp):
        timestamp = time.monotonic() - initial_timestamp  # Calculate time since start
        timeline.append({
            "task": task,
            "event": event,
            "timestamp": timestamp,
        })
        print(f"[{timestamp:.2f}] {event}: {task}")

    async def async_db_operation(self, step, initial_timestamp, async_timeline_events):
        task_id = f"log_{step}"
        self.log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
        # Simulate variable database operation time
        await asyncio.sleep(random.uniform(0.3, 3.0))
        self.log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time

    async def async_google_call(self, prompt, client, initial_timestamp, async_timeline_events):
        task_id = f"google_{prompt[:10]}"
        self.log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
        # LLM calls often take longer than other API calls
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello, how are you?."
        )
        print(f"Google Response: {response.text}\n")
        self.log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time
        return f"LLM result: {prompt}"

    async def async_openai_call(self, prompt, client, initial_timestamp, async_timeline_events):
        task_id = f"openai_{prompt[:10]}"
        self.log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
        # LLM calls often take longer than other API calls
        response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Hello, how are you?."}
                ]
        )
        print(f"OpenAI Response: {response.text}\n")
        self.log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time
        return f"LLM result: {prompt}"

    async def async_http_call(self, endpoint, initial_timestamp, async_timeline_events):
        task_id = f"http_{endpoint.replace('/', '')}"
        self.log_event(async_timeline_events, task_id, "START", initial_timestamp) # just logging time
        # Simulate variable network latency
        await asyncio.sleep(random.uniform(0.4, 4.0))
        self.log_event(async_timeline_events, task_id, "END", initial_timestamp) # just logging time
        return f"HTTP result from {endpoint}"

    async def run_async_pipeline(self, initial_timestamp, google_client, openai_client, tasks = [], async_timeline_events = []):
        self.log_event(async_timeline_events, "main_pipeline", "START_PIPELINE", initial_timestamp)

        # DEPENDENT TASK: We need the LLM result before proceeding
        google_result = await self.async_google_call("What's the user intent?", google_client, initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Log the result to the database, but don't wait for it
        tasks.append(asyncio.create_task(self.async_db_operation("google_result", initial_timestamp, async_timeline_events)))

        # DEPENDENT TASK: We need the LLM result before proceeding
        # openai_result = await self.async_openai_call("What's the user intent?", openai_client, initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Log the result to the database, but don't wait for it
        # tasks.append(asyncio.create_task(self.async_db_operation("openai_result", initial_timestamp, async_timeline_events)))

        # DEPENDENT TASK: We need this HTTP result for our business logic
        http1 = await self.async_http_call("/api/data", initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Log this API call, but don't block on it
        tasks.append(asyncio.create_task(self.async_db_operation("http1", initial_timestamp, async_timeline_events)))

        # DEPENDENT TASK: We need this data too before proceeding
        http2 = await self.async_http_call("/api/details", initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Another non-blocking logging operation
        tasks.append(asyncio.create_task(self.async_db_operation("http2", initial_timestamp, async_timeline_events)))

        # DEPENDENT TASK: Generate a summary using the LLM with collected data
        summary_google = await self.async_google_call("Summarize everything", google_client, initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Log the summary generation
        tasks.append(asyncio.create_task(self.async_db_operation("summary_google", initial_timestamp, async_timeline_events)))
        
        # DEPENDENT TASK: Generate a summary using the LLM with collected data
        # summary_openai = await self.async_openai_call("Summarize everything", openai_client, initial_timestamp, async_timeline_events)
        # BACKGROUND TASK: Log the summary generation
        # tasks.append(asyncio.create_task(self.async_db_operation("summary_openai", initial_timestamp, async_timeline_events)))

        self.log_event(async_timeline_events, "main_pipeline", "END_PIPELINE", initial_timestamp)

        # Wait for all background tasks to complete before exiting
        # This ensures all logging operations finish properly
        # This will run without gathering all tasks as well, you can try
        await asyncio.gather(*tasks)

        return async_timeline_events
