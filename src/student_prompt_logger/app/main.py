from fastapi import FastAPI
import logging

from app.config.logging import setup_logging
from app.config.errors import http_error_handler

setup_logging()

app = FastAPI()
app.add_exception_handler(Exception, http_error_handler)
log_file = "app.log"

@app.get('/health')
def health():
    return {'status':'ok'}

# Generate a mock response
@app.post('/generate')
def generate(
    prompt: str,
    provider: str
):
    from app.services.client_factory import ClientFactory
    client = ClientFactory.create_client(provider)

    return client.generate(prompt)

# Returns the last 10 logs
@app.get('/history')
def history():
    import os
    db = os.getenv('DB_PATH')
    if not db:
        logging.error("No database found.")
    else:
        import sqlite3
        con = sqlite3.connect(db)
        cur = con.cursor()
    from collections import deque
    with open(log_file, "r") as file:
        lines = deque(file, maxlen=10)
        lines.reverse()
        return lines

# Returns the env variables
@app.get('/config')
def config():
    import os
    db = os.getenv('DB_PATH')
    if not db:
        logging.warning("No database found.")
    log_level = os.getenv('LOG_LEVEL')
    if not log_level:
        logging.warning("No log_level found.")
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        logging.warning("No google_key found.")
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logging.warning("No openai_key found.")
    watsonx_key = os.getenv('IWATSON_API_KEY')
    if not watsonx_key:
        logging.warning("No watsonx_key found.")

    return {
        "app_name": "API Prompt Logger",
        "db_path": db,
        "google_key_present": True if google_key else False,
        "openai_key_present": True if openai_key else False,
        "watsonx_key_present": True if watsonx_key else False
    }


