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

    method="json_schema" y no "function_calling": con
    langchain-google-genai 4.3.1 el envoltorio de function_calling emite
    una clave que Gemini rechaza ("Key 'parameters' is not supported in
    schema, ignoring"). Los tres métodos producen el plan correcto; este
    es el único que no ensucia el log.
    """

    system_prompt = PLANNER_PROMPT

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ])

    planner_chain = (
        prompt
            | llm.with_structured_output(
            schema=PlannerOutput,
            method="json_schema",
        )
            )

    return planner_chain