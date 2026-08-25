from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from streamlit import secrets

load_dotenv()

def _get_secret(key: str, default: str = None):
    return os.getenv(key) or secrets.get(key) or default


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=40,
        temperature=0.2,
        max_retries=3,  # evita que el cliente subdivida el deadline < 10s
    )

def create_routing_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=40,
        temperature=0.0,
        max_retries=3,
    )

def create_synthesizer_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=40,
        temperature=0.4,
        max_retries=3,
    )

def create_suggester_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",       # ⚠️ decidir: distinto de flash-lite
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.3,
        timeout=40,        # se corta solo, no hace falta wrapper externo
        max_retries=3      # sin retry en 429 — degradamos a [] nosotros
    )