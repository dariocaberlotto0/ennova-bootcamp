import os

class Settings:
    DB_PATH = os.getenv('DB_PATH','promptlog.db')
    LOG_LEVEL = os.getenv('LOG_LEVEL','INFO')
    LLM_API_KEY = os.getenv('LLM_API_KEY','')
settings = Settings()
