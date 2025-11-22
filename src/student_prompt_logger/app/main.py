import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    logging.error(e)

import os
DB_PATH = os.getenv('DB_PATH')
if not DB_PATH:
    logging.warning("No database found.")
LOG_LEVEL = os.getenv('LOG_LEVEL')
if not LOG_LEVEL:
    logging.warning("No log_level found.")
GOOGLE_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_KEY:
    logging.warning("No google_key found.")
OPENAI_KEY = os.getenv('OPEN_API_KEY')
if not OPENAI_KEY:
    logging.warning("No openai_key found.")
WATSONX_KEY = os.getenv('IWATSON_API_KEY')
if not WATSONX_KEY:
    logging.warning("No watsonx_key found.")

from fastapi import FastAPI

from config.errors import http_error_handler

app = FastAPI()
app.add_exception_handler(Exception, http_error_handler)
log_file = "service.log"


@app.get('/health')
def health():
    return {'status':'ok'}

# Generate response from the chosen provider ('gemini', 'openai', 'mock', 'openaiasync' (TODO))
@app.post('/generate')
def generate(
    prompt: str,
    provider: str
):
    from services.client_factory import ClientFactory
    client = ClientFactory.create_client(provider)

    return client.generate(prompt)

# Returns the last 10 logs
@app.get('/history')
def history(
    limit: int
):
    lines = []
    if not DB_PATH:
        logging.error("No database found.")
    else:
        import sqlite3
        # Connects to db
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        # Create the table if not exist
        cur.execute("CREATE TABLE IF NOT EXISTS logs(timestamp, level, logger_name, message)")
        logs = cur.execute(f"SELECT timestamp, level, logger_name, message FROM logs ORDER BY id DESC LIMIT {limit}")
        lines = logs.fetchall()
        logging.info(f"Fetched last {limit} logs.")

    return lines
    

# Returns the env variables
@app.get('/config')
def config():
    return {
        "app_name": "API Prompt Logger",
        "db_path": DB_PATH,
        "google_key_present": True if GOOGLE_KEY else False,
        "openai_key_present": True if OPENAI_KEY else False,
        "watsonx_key_present": True if WATSONX_KEY else False
    }
