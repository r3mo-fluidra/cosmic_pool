"""
schema_utils.py

Aplanado de JSON Schema para la API de Gemini.

Gemini acepta un subconjunto de OpenAPI 3.0 que no soporta `$ref` ni
`$defs`. Con submodelos anidados, `model_json_schema()` los emite por
referencia y el SDK descarta el bloque entero:

    Key '$defs' is not supported in schema, ignoring

El modelo queda sin el contrato de esos campos: no sabe qué claves lleva
un ExecutionStep ni un DetailSection. De ahí salen las salidas
malformadas intermitentes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _resolve(node: Any, defs: dict, seen: frozenset) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            if name in seen:
                # Ciclo en el schema: no se puede aplanar. Objeto abierto
                # en vez de recursión infinita.
                return {"type": "object"}
            target = defs.get(name)
            if target is None:
                return {"type": "object"}
            resolved = _resolve(deepcopy(target), defs, seen | {name})
            # Las claves hermanas del $ref (description, default) pisan
            # a las del target: son las del punto de uso.
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            resolved.update(_resolve(siblings, defs, seen))
            return resolved
        return {
            k: _resolve(v, defs, seen)
            for k, v in node.items()
            if k != "$defs"
        }
    if isinstance(node, list):
        return [_resolve(i, defs, seen) for i in node]
    return node


def flatten_schema(model) -> dict:
    """Modelo Pydantic -> JSON Schema sin $ref ni $defs."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    flat = _resolve(deepcopy(schema), defs, frozenset())
    flat.pop("$defs", None)
    return flat

def as_function_declaration(model, name: str | None = None) -> dict:
    """
    Modelo Pydantic -> declaración de función lista para Gemini.

    `with_structured_output(method="function_calling")` arma este envoltorio
    solo cuando recibe un modelo Pydantic. Con un dict asume que ya es una
    declaración completa y termina anidando `parameters` dentro de
    `parameters`, así que Gemini descarta el schema entero:

        Key 'parameters' is not supported in schema, ignoring

    Peor que el $defs original: ahí se perdía el contrato de los submodelos,
    acá se pierde todo. Devolver la declaración ya armada es lo que cierra
    las dos cosas — envoltorio correcto y schema aplanado adentro.
    """
    flat = flatten_schema(model)
    description = flat.pop("description", "") or model.__name__
    flat.pop("title", None)
    return {
        "name": name or model.__name__,
        "description": description,
        "parameters": flat,
    }