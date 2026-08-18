"""
graph_context/turn_cache.py
============================
Bolsa de nodos tocados por las tools de retrieval (search_seed_nodes,
expand_subgraph) durante UN turno. Sin lógica de filtrado — eso vive en
suggestions.py. Este módulo solo acumula y devuelve.

Keyed por thread_id (asumo que es la clave de checkpointing de LangGraph,
config["configurable"]["thread_id"] — CONFIRMAR contra el checkpointer real).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TouchedNode:
    id: str       # slug determinístico, mismo id que usa el resto del grafo
    label: str    # label PascalCase de Neo4j
    name: str     # nombre para mostrar / matchear contra texto de respuesta


class _TurnCache:
    """Instancia interna, una por thread_id. No se usa directo — ver API abajo."""

    def __init__(self) -> None:
        self._nodes: Dict[str, TouchedNode] = {}  # dedup por id

    def touch(self, nodes: List[TouchedNode]) -> None:
        for n in nodes:
            self._nodes[n.id] = n  # last-write-wins

    def touched(self) -> List[TouchedNode]:
        return list(self._nodes.values())


# =====================================================================
# API de módulo — thread-safe, un cache por thread_id
# =====================================================================

_lock = threading.Lock()
_caches: Dict[str, _TurnCache] = {}
# NOTA: esto crece sin límite mientras el proceso viva (nunca se borra un
# thread_id viejo). Para un Streamlit single-process con pocos usuarios
# concurrentes no es un problema hoy; si esto pasa a producción con
# muchos threads de larga vida, hace falta un TTL o un LRU acá. Lo dejo
# anotado y no lo resuelvo ahora para no sobre-ingenierizar sin datos.


def reset_turn(thread_id: str) -> None:
    """
    Llamar UNA vez al INICIO de cada turno (primer nodo del grafo —
    presumiblemente planner). Limpia lo del turno anterior para que
    unconsumed_entities nunca filtre entre turnos.
    """
    with _lock:
        _caches[thread_id] = _TurnCache()


def touch(thread_id: str, nodes: List[TouchedNode]) -> None:
    """
    Llamar desde search_seed_nodes / expand_subgraph después de leer
    de Neo4j, con los nodos que la tool efectivamente devolvió.
    """
    with _lock:
        cache = _caches.setdefault(thread_id, _TurnCache())
        cache.touch(nodes)


def get_touched(thread_id: str) -> List[TouchedNode]:
    """Llamar desde el suggester. Lista vacía si no hubo retrieval este turno."""
    with _lock:
        cache = _caches.get(thread_id)
        return cache.touched() if cache else []