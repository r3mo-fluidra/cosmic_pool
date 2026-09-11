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
    # 2 y no 1: una sola búsqueda de seeds no cubre un panel de análisis. En el
    # trace 6660e14f el agente recibió siete parámetros, gastó su única llamada
    # en una consulta general, y se quedó sin poder resolver los límites de
    # cianúrico, alcalinidad y dureza — de ahí salió un regulatory_limit
    # inventado para la dureza. Después pidió la tool dos veces más y el gate
    # se las rechazó: ~4s de round trips por evidencia que no podía obtener.
    "search_seed_nodes": 2,
    "expand_subgraph": 2,
}

RETRIEVAL_TOTAL = sum(RETRIEVAL_TOOL_BUDGETS.values())