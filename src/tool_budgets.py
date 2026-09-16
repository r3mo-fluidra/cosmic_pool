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
    # --- Retrieval (Neo4j + Qdrant) --------------------------------------
    "vector_search":      1,
    "search_seed_nodes":  2,
    "expand_subgraph":    2,

    # --- Math (catálogo determinista) ------------------------------------
    # Sin entradas acá el default de `_gate` es 1 por tool, y MATH no puede
    # ni resolver un STATUS: CANDIDATES. Dimensionado sobre el trace
    # 3562130029, donde murió contra el recursion_limit a los 13 calls.
    "resolve_formula":    3,
    "calculate":          2,
    "get_constant":       3,
    "convert_units":      2,
    "lookup_product":     1,
    "check_plausibility": 2,
}

#: Solo las tools de retrieval. `RETRIEVAL_TOTAL` sumaba el dict entero y
#: pasó a contar las de math en cuanto entraron: 18 donde debía decir 5.
_RETRIEVAL_TOOLS = ("vector_search", "search_seed_nodes", "expand_subgraph")

RETRIEVAL_TOTAL = sum(RETRIEVAL_TOOL_BUDGETS[t] for t in _RETRIEVAL_TOOLS)