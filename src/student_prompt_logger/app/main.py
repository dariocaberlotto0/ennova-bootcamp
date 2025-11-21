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
