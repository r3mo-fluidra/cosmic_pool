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

from typing import Literal, Dict

# Tipo para los nombres de agentes
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

# Diccionario de nombres legibles para humanos
# Esto es lo que probablemente falta y causa el KeyError
AGENT_NAMES: Dict[AgentName, str] = {
    "chemistry": "Chemistry Expert",
    "equipment": "Equipment Specialist", 
    "hydraulics": "Hydraulics Engineer",
    "operations": "Operations Expert",
    "compliance": "Compliance Officer",
    "contamination": "Contamination Specialist",
    "facility_design": "Facility Designer",
    "safety": "Safety Expert",
    "recovery": "Recovery Specialist",
    "records": "Records Manager",
    "math": "Mathematics Expert",
    "general": "General Assistant",
    "oos": "Out of Scope Handler",
}

# Alias para compatibilidad (si algún código espera 'agent_names' en minúscula)
agent_names = AGENT_NAMES

# Lista de todos los nombres de agentes (útil para validación)
ALL_AGENT_NAMES: list[AgentName] = list(AGENT_NAMES.keys())

# Set para validación rápida
AGENT_NAME_SET = set(ALL_AGENT_NAMES)