from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from streamlit import secrets

load_dotenv()

def _get_secret(key: str, default: str = None):
    return os.getenv(key) or secrets.get(key) or default


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.2
    )

def create_routing_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.0
    )

def create_synthesizer_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.4
    )