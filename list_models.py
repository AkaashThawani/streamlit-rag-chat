# list_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

    print("--- Your Available Gemini Models ---")
    for m in genai.list_models():
        # We are interested in models that can generate text content
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
    print("------------------------------------")