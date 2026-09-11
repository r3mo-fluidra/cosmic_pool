from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


def _get_secret(key: str, default: str = None):
    """
    Env var primero, Streamlit secrets después.

    El import de `streamlit.secrets` va adentro y envuelto: acceder a
    `st.secrets` sin un .streamlit/secrets.toml lanza
    StreamlitSecretNotFoundError. Estaba a nivel de módulo y sin guarda, así
    que un despliegue con la API key solo en el entorno y sin fichero de
    secrets moría acá dentro del grafo y le llegaba al usuario como
    "El agente falló: ...". Los otros dos módulos que leen secrets
    (agent/tools.py, qdrant_vector_store.py) ya lo envolvían; este era el
    único que no.

    De paso desacopla el módulo de LLMs de Streamlit, que es lo que obligaba
    a tener streamlit instalado para importar cualquier test de config.
    """
    val = os.getenv(key)
    if val:
        return val
    try:
        from streamlit import secrets
        return secrets.get(key) or default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Presupuestos de tiempo
# ---------------------------------------------------------------------------
# El peor caso de una llamada es timeout * max_retries. Estaba en 120 * 3 =
# 360s, contra un TURN_DEADLINE_S de 120: el techo del cliente no acotaba
# nada. Peor, `_run_with_deadline` corta la ESPERA con future.cancel(), que
# no mata el thread — así que una llamada colgada seguía ocupando un slot del
# _STEP_POOL (8 workers) durante esos 360s, invisible, degradando el fan-out
# de todos los turnos siguientes.
#
# Regla: peor caso de una llamada < STEP_DEADLINE_S, para que quien corte sea
# siempre el deadline del grafo y no el del cliente.
#
# Los valores son conservadores y están para calibrarse con la distribución
# real de latencias de Langfuse, no para quedarse fijos.
_FAST_TIMEOUT = 15      # flash-lite: clasificar y sugerir. Peor caso 30s.
_STANDARD_TIMEOUT = 30  # flash: redactar y razonar.        Peor caso 60s.
_MAX_RETRIES = 1        # un reintento. Un 429 no se arregla insistiendo 3
                        # veces contra el mismo modelo con la misma key.


def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=_STANDARD_TIMEOUT,
        temperature=0.2,
        max_retries=_MAX_RETRIES,
    )

def create_routing_llm():
    """
    Modelo del planner (y del supervisor).

    thinking_budget=0 por el mismo argumento que create_synthesis_llm(): el
    planner no investiga ni redacta, clasifica. Su salida es un PlannerOutput
    con schema cerrado — idioma, lista de steps, agente asignado. Con thinking
    dinámico el modelo elige cuánto razonar en cada turno, y esa elección es
    latencia pura en el PRIMER nodo del turno, antes de que el usuario haya
    visto nada.

    0 y no un valor bajo: medido contra la API, cualquier valor > 0 se trata
    como objetivo blando y el modelo lo excede.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=_FAST_TIMEOUT,
        temperature=0.0,
        max_retries=_MAX_RETRIES,
        thinking_budget=0,
    )

def create_synthesizer_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=_STANDARD_TIMEOUT,
        temperature=0.4,
        max_retries=_MAX_RETRIES,
    )



def create_suggester_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",       # ⚠️ decidir: distinto de flash-lite
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.3,
        timeout=_FAST_TIMEOUT,        # se corta solo, no hace falta wrapper externo
        max_retries=_MAX_RETRIES      # sin retry en 429 — degradamos a [] nosotros
    )

def create_fallback_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",       
        google_api_key=_get_secret("GEMINI_API_KEY"),
        temperature=0.3,
        timeout=_STANDARD_TIMEOUT,        # se corta solo, no hace falta wrapper externo
        max_retries=_MAX_RETRIES      # sin retry en 429 — degradamos a [] nosotros
    )

def create_synthesis_llm():
    """
    LLM del nodo synthesizer, con thinking apagado.

    En el trace de referencia la síntesis gastó 1599 tokens de razonamiento
    para producir 367 visibles: 7.7s reformateando material que los
    sub-agentes ya habían resuelto.

    thinking_budget=0 y no un valor bajo: medido contra la API, cualquier
    valor > 0 se trata como objetivo blando y el modelo lo excede (se le
    pide 128, gasta 525). Solo 0 es un corte real.

    El synthesizer no investiga ni decide: reescribe material ya resuelto
    en el formato del contrato. Es el nodo donde apagar el thinking cuesta
    menos.

    Nombre distinto de create_synthesizer_llm(), que a pesar del nombre
    alimenta a los sub-agentes especialistas en agents.py, no a este nodo.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=_STANDARD_TIMEOUT,
        temperature=0.2,
        max_retries=_MAX_RETRIES,
        thinking_budget=0,
        max_tokens=1200,
    )

def create_specialist_llm():
    """
    Modelo de los especialistas de retrieval, separado de
    create_synthesizer_llm() para poder acotarles el thinking sin tocar
    a `math`, que comparte esa factory.

    thinking_budget existe porque tres traces de la MISMA consulta dieron
    1802, 2591 y 3414 tokens de reasoning en la llamada final del
    especialista, con latencias de 12.9s, 18.8s y 19.9s. Con thinking
    dinámico el modelo elige cuánto razonar por llamada, y esa elección es
    la mayor fuente de varianza del turno. 1024 es un valor experimental:
    por debajo del mejor caso observado, para ver si la latencia responde.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=_get_secret("GEMINI_API_KEY"),
        timeout=_STANDARD_TIMEOUT,
        temperature=0.4,
        max_retries=_MAX_RETRIES,
        thinking_budget=1024,
    )