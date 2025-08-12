# config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env once for the whole project
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Configure Gemini API once
genai.configure(api_key=api_key)

# Create a shared model instance
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_text(response):
    """
    Robust extractor for gemini generate_content response.
    Tries common response shapes, falls back to str(response).
    """
    try:
        # Standard recent shape
        return response.candidates[0].content.parts[0].text
    except Exception:
        try:
            # older variations
            return response.candidates[0].text
        except Exception:
            try:
                return response.output[0].content[0].text
            except Exception:
                return str(response)
