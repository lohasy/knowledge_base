import os

from dotenv import load_dotenv

load_dotenv(override=True)

base_url = os.getenv("OPENAI_BASE_URL")
print(base_url)