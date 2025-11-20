from environment import Environment
import random
import time

class SyncPipeline:

    def log_event(self, timeline, task, event, initial_timestamp):
        timestamp = time.monotonic() - initial_timestamp  # Calculate time since start
        timeline.append({
            "task": task,
            "event": event,
            "timestamp": timestamp,
        })
        print(f"[{timestamp:.2f}] {event}: {task}")

    def sync_db_operation(self, step, initial_timestamp, sync_timeline_events):
        task_id = f"log_{step}"
        self.log_event(sync_timeline_events, task_id, "START", initial_timestamp)
        # Block the entire program during this "database operation"
        time.sleep(random.uniform(0.3, 3.0))
        self.log_event(sync_timeline_events, task_id, "END", initial_timestamp)

    def sync_google_call(self, prompt, client, initial_timestamp, sync_timeline_events):
        task_id = f"google_{prompt[:10]}"
        self.log_event(sync_timeline_events, task_id, "START", initial_timestamp)
        # Block the entire program during this "API call"
        response = client.models.generate_content(model="gemini-2.5-flash",
            contents="Hello, how are you?."
        )
        print(f"Google Response: {response.text}\n")
        self.log_event(sync_timeline_events, task_id, "END", initial_timestamp)
        return f"Google result: {prompt}"

    def sync_openai_call(self, prompt, client, initial_timestamp, sync_timeline_events):
        task_id = f"openai_{prompt[:10]}"
        self.log_event(sync_timeline_events, task_id, "START", initial_timestamp)
        # Block the entire program during this "API call"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                    "role": "user",
                    "content": "Hello, how are you?."
            }]
        )
        print(f"OpenAI Response: {response.choices[0].message.content}\n")
        self.log_event(sync_timeline_events, task_id, "END", initial_timestamp)
        return f"OpenAI result: {prompt}"

    def sync_http_call(self, endpoint, initial_timestamp, sync_timeline_events):
        task_id = f"http_{endpoint.replace('/', '')}"
        self.log_event(sync_timeline_events, task_id, "START", initial_timestamp)
        # Block the entire program during this "network request"
        time.sleep(random.uniform(0.4, 4.0))
        self.log_event(sync_timeline_events, task_id, "END", initial_timestamp)
        return f"HTTP result from {endpoint}"

    def run_sync_pipeline(self, initial_timestamp, google_client, openai_client, tasks = [], sync_timeline_events = []):
        self.log_event(sync_timeline_events, "main_pipeline", "START_PIPELINE", initial_timestamp)

        # Step 1: Make LLM call and wait for it to complete
        google_result = self.sync_google_call("What's the user intent?", google_client, initial_timestamp, sync_timeline_events)
        # Step 2: Log the result and wait for logging to complete
        self.sync_db_operation("llm_result", initial_timestamp, sync_timeline_events)

        # Step 3: Make first HTTP call and wait for it to complete
        http1 = self.sync_http_call("/api/data", initial_timestamp, sync_timeline_events)
        # Step 4: Log the HTTP result and wait for logging to complete
        self.sync_db_operation("http1", initial_timestamp, sync_timeline_events)

        # Step 5: Make second HTTP call and wait for it to complete
        http2 = self.sync_http_call("/api/details", initial_timestamp, sync_timeline_events)
        # Step 6: Log the HTTP result and wait for logging to complete
        self.sync_db_operation("http2", initial_timestamp, sync_timeline_events)
        
        # Step 7: Make final LLM call and wait for it to complete
        openai_summary = self.sync_openai_call("Summarize everything", openai_client, initial_timestamp, sync_timeline_events)
        # Step 8: Log the summary and wait for logging to complete
        self.sync_db_operation("summary", initial_timestamp, sync_timeline_events)

        self.log_event(sync_timeline_events, "main_pipeline", "END_PIPELINE", initial_timestamp)

        return sync_timeline_events
