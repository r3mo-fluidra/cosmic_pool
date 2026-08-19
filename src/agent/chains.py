from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from .state import PlannerOutput   # ← Importamos el modelo Pydantic
from ..prompts.prompts import PLANNER_PROMPT
# ================================================================
# PLANNER CHAIN
# ================================================================

def create_planner_chain(llm: BaseChatModel) -> Runnable:
    """
    Crea el chain del Planner con structured output.
    """

    system_prompt = PLANNER_PROMPT

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ])

    # Chain con structured output (forzado)
    planner_chain = (
        prompt 
        | llm.with_structured_output(
            schema=PlannerOutput,
            method="function_calling",   # Mejor opción con modelos modernos
            # method="json_mode"         # Alternativa si function calling no funciona bien
        )
    )

    return planner_chain