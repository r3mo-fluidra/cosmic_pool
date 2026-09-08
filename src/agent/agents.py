"""
agents.py
=========
Construcción de los agentes de cluster.

Solo AQUATIC_CHEM y SYSTEMS necesitan un grafo de agente: declaran tools de
cálculo. GOVERNANCE declara tools=() y corre con _direct_answer en nodes.py,
igual que general y oos.

Cambio estructural respecto de la versión anterior: los agentes ya no se
construyen una vez en _initialize(). Su system prompt depende de los módulos
que el planner eligió para ESTE turno (role framing del primario, contrato de
salida unido, budget por módulo activo), así que se construyen por turno. Lo
que se cachea es el LLM, que es lo caro de instanciar.

Lo que se eliminó:
  - create_supervisor / pool_supervisor: el supervisor operaba sobre
    execution_plan ordenado, avanzando paso a paso. Con un step por turno no
    tiene trabajo.
  - Los agentes de general y oos, y RunnableWithFallbacks: los nodos nunca
    los usaron -- llaman _direct_answer, que es un invoke pelado.
    get_agent_by_name("general") no se llamaba desde el grafo.
  - pool_general_knowledge: era un no-op que devolvía un string pidiéndole al
    modelo responder de su training. Un round trip regalado.
  - RETRIEVAL_TOOLS (vector_search, search_seed_nodes, expand_subgraph): los
    clusters responden con prompt especializado y catálogo de cálculo.
  - SPECIALIST_SPECS y AGENT_REGISTRY: reemplazados por CLUSTER_REGISTRY.
"""

import logging

from langchain.agents import create_agent

from ..tools_math.tools import MATH_TOOLS
from ..config.llm import create_synthesizer_llm
from ..prompts.prompts_sub_agents import CLUSTER_REGISTRY
from ..prompts.prompt_archetype import build_cluster_prompt

logger = logging.getLogger(__name__)


# ================================================================
# CLUSTERS CON GRAFO DE AGENTE
# ================================================================
# GOVERNANCE queda afuera a propósito: tools=() en su ClusterConfig.
# create_agent con lista de tools vacía construye un grafo que nunca puede
# ir al nodo de tools -- un invoke directo hace lo mismo sin el overhead.

TOOLED_CLUSTERS = frozenset({"AQUATIC_CHEM", "SYSTEMS"})


def is_tooled(cluster_name: str) -> bool:
    """True si el cluster necesita grafo de agente en vez de _direct_answer."""
    return cluster_name in TOOLED_CLUSTERS


# ================================================================
# LAZY LLM
# ================================================================

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_synthesizer_llm()
    return _llm


# ================================================================
# CONSTRUCCIÓN POR TURNO
# ================================================================

def build_cluster_agent(
    cluster_name: str,
    primary_module: str,
    support_modules: list[str] | None = None,
    revising: bool = False,
):
    """
    Construye el agente de un cluster para este turno.

    Raises:
        ValueError: cluster desconocido, cluster sin tools, o módulo primario
                    que no pertenece al cluster. Los tres son bugs de ruteo,
                    no condiciones a degradar: el nodo los captura y los
                    reporta como AgentResult fallido.
    """
    cluster = CLUSTER_REGISTRY.get(cluster_name)
    if cluster is None:
        raise ValueError(
            f"Cluster '{cluster_name}' no está registrado. "
            f"Disponibles: {list(CLUSTER_REGISTRY)}"
        )

    if not is_tooled(cluster_name):
        raise ValueError(
            f"Cluster '{cluster_name}' no declara tools; debe correr con "
            "_direct_answer, no con un grafo de agente."
        )

    if primary_module not in cluster.modules:
        raise ValueError(
            f"Módulo '{primary_module}' no pertenece a '{cluster_name}'. "
            f"Módulos del cluster: {list(cluster.modules)}"
        )

    system_prompt = build_cluster_prompt(
        cluster_name=cluster_name,
        primary_module=primary_module,
        support_modules=support_modules,
        revising=revising,
    )

    return create_agent(
        model=_get_llm(),
        tools=MATH_TOOLS,
        name=cluster_name,
        system_prompt=system_prompt,
    )