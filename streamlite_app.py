from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv() 

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("API KEY:", os.getenv("GROQ_API_KEY"))
