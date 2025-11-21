import logging
import os

def setup_logging():

    LOGLEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

    logging.basicConfig(filename='app.log', encoding="utf-8", filemode="a", level=LOGLEVEL, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    # Telling uvicorn loggers to print logs in app.log
    uvicorn_loggers = ["uvicorn", "uvicorn.access", "uvicorn.error", "uvicorn.asgi"]
    
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(LOGLEVEL)
        
