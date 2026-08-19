"""
graph_context/turn_cache.py
===========================
Bolsa de nodos que las tools de retrieval tocaron durante UN turno.

Sin lógica de filtrado: solo acumula y devuelve. Decidir qué está
consumido, qué es supernodo y qué es redundante es responsabilidad de
suggestions.py. Una sola responsabilidad por módulo.

Ciclo de vida:
    planner            -> reset_turn(thread_id)      (limpia el turno anterior)
    search_seed_nodes  -> touch(thread_id, nodes)
    expand_subgraph    -> touch(thread_id, nodes)
    suggester          -> get_touched(thread_id)

El reset va al INICIO del turno, no al final: si un turno muere a mitad
de camino (excepción en el orchestrator), el cache queda sucio y el
próximo turno arrancaría contaminado.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TouchedNode:
    """Un nodo que alguna tool de retrieval devolvió en este turno."""

    id: str       # slug determinístico, mismo id que usa el resto del grafo
    label: str    # label de Neo4j (PascalCase)
    name: str     # nombre legible, para matchear contra el texto de la respuesta


class _TurnCache:
    """Instancia interna, una por thread_id. No se usa directo."""

    def __init__(self) -> None:
        self._nodes: Dict[str, TouchedNode] = {}   # dedup por id

    def touch(self, nodes: List[TouchedNode]) -> None:
        for n in nodes:
            if n.id:
                self._nodes[n.id] = n              # last-write-wins

    def touched(self) -> List[TouchedNode]:
        return list(self._nodes.values())


# =====================================================================
# API de módulo — thread-safe, un cache por thread_id
# =====================================================================

_lock = threading.Lock()
_caches: Dict[str, _TurnCache] = {}
# NOTA: _caches crece sin límite mientras viva el proceso — nunca se
# borra un thread_id viejo. Para un Streamlit single-process con pocos
# usuarios concurrentes no es un problema hoy. Si esto escala, hace
# falta TTL o LRU acá. Anotado y deliberadamente no resuelto: sin datos
# de uso, cualquier política sería inventada.


def reset_turn(thread_id: str) -> None:
    """
    Llamar UNA vez al INICIO de cada turno, desde el planner, antes de
    la llamada al LLM. Garantiza que unconsumed_entities nunca filtre
    entidades de turnos anteriores.
    """
    with _lock:
        _caches[thread_id] = _TurnCache()


def touch(thread_id: str, nodes: List[TouchedNode]) -> None:
    """
    Llamar desde search_seed_nodes / expand_subgraph con los nodos que
    la tool efectivamente devolvió (no los que buscó).
    """
    if not thread_id or not nodes:
        return
    with _lock:
        cache = _caches.setdefault(thread_id, _TurnCache())
        cache.touch(nodes)


def get_touched(thread_id: str) -> List[TouchedNode]:
    """
    Llamar desde el suggester. Lista vacía si no hubo retrieval en este
    turno (caso normal para turnos conversacionales).
    """
    with _lock:
        cache = _caches.get(thread_id)
        return cache.touched() if cache else []