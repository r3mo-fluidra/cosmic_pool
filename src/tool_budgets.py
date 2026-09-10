"""
Fuente única de verdad del presupuesto de tools por step.

Módulo sin dependencias, en la raíz de src/, porque lo consumen dos paquetes
que no deben importarse entre sí: agent/tools.py lo APLICA en el gate, y
prompts/prompt_archetype.py lo DECLARA al agente. Cualquier otro arreglo crea
un import circular o arrastra los clientes de Neo4j y Qdrant al módulo de
prompts.

Los caps son POR TOOL y NO son intercambiables: gastar la llamada de
vector_search no libera una segunda de search_seed_nodes.

expand_subgraph = 2 porque el camino normativo documentado en RETRIEVAL_CORE
admite un intento por slug + un segundo sobre los seeds reales tras el miss.
"""

RETRIEVAL_TOOL_BUDGETS: dict[str, int] = {
    "vector_search": 1,
    "search_seed_nodes": 1,
    "expand_subgraph": 2,
}

RETRIEVAL_TOTAL = sum(RETRIEVAL_TOOL_BUDGETS.values())