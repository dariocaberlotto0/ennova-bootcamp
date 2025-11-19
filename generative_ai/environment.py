import os

class Environment:
    @staticmethod
    def load():
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        OPEN_API_KEY = os.getenv('OPEN_API_KEY')

        return GOOGLE_API_KEY, OPEN_API_KEY