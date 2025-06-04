import os

from pydantic import BaseModel
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# Configuration class to hold OpenAI API settings
class Config(BaseModel):
    OPENAI_ENDPOINT : str = os.getenv("OPENAI_ENDPOINT", "")
    OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY", "")

config = Config()