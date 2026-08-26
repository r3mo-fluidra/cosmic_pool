"""
agent_names.py
==============
Fuente única de verdad para los nombres de agente.

Vive en su propio módulo para romper la circularidad:
  state.py       importa Suggestion desde suggestions.py
  suggestions.py importa AgentName  desde ... acá, no desde state.py

Mismo criterio que response_contracts.py: sin dependencias del proyecto.
"""

from __future__ import annotations

from typing import Literal

AgentName = Literal[
    "chemistry",
    "equipment",
    "hydraulics",
    "operations",
    "compliance",
    "contamination",
    "facility_design",
    "safety",
    "recovery",
    "records",
    "math",
    "general",
    "oos"
]