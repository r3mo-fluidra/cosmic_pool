from __future__ import annotations
import os
from typing import List, Optional, Dict, Any, Literal
from math import log10
# VECTOR STORE AND GRAPH DB CLIENTS 
from ..qdrant_vector_store import cargar_vector_store
from qdrant_client.http import models
from neo4j import GraphDatabase
from dotenv import load_dotenv
from streamlit import secrets
# ==========================================
# 0. NEO4J INITIALIZATION 
# ==========================================

load_dotenv()

_neo4j_driver = None

def _get_secret(key: str, default: str = None) -> Optional[str]:
    """Lee un secret desde env vars o st.secrets, en ese orden."""
    return os.getenv(key) or secrets.get(key) or default

def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        uri      = _get_secret("NEO4J_URI")
        user     = _get_secret("NEO4J_USER", "neo4j")
        password = _get_secret("NEO4J_PASSWORD")

        import streamlit as st
        st.write("DEBUG secrets keys:", list(st.secrets.keys()))
        st.write("NEO4J_URI:", uri)
        
        if not uri:
            raise ValueError(
                "NEO4J_URI no está configurado. "
                "Agrégalo en .streamlit/secrets.toml o en los Secrets de Streamlit Cloud."
            )

        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver


def _execute_cypher(query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Helper function to execute Cypher queries safely."""
    if parameters is None:
        parameters = {}

    # ✅ Usa get_neo4j_driver() en lugar de la variable inexistente `neo4j_driver`
    database = _get_secret("NEO4J_DATABASE")

    results = []
    try:
        with get_neo4j_driver().session(database=database) as session:
            records = session.run(query, parameters)
            for record in records:
                results.append(record.data())
    except Exception as e:
        print(f"Neo4j Execution Error: {e}")
        return [{"error": str(e)}]

    return results

# ==========================================
# 0.0 VECTOR STORE INITIALIZATION
# ==========================================

GLOBAL_VECTOR_STORE = None

def get_vector_store():
    global GLOBAL_VECTOR_STORE

    if GLOBAL_VECTOR_STORE is None:
        GLOBAL_VECTOR_STORE = cargar_vector_store()

    return GLOBAL_VECTOR_STORE

# ==========================================
# 1. DIAGNOSIS AGENT TOOLS
# ==========================================

def query_symptom_graph(symptom_keyword: str) -> List[Dict[str, Any]]:
    """
    Performs a Cypher traversal on the knowledge graph to find direct connections 
    between a physical symptom and pool chemistry parameters.
    """
    cypher_query = """
    MATCH (s:Symptom)-[r:LOW_CAUSES|HIGH_CAUSES|OUT_OF_RANGE_CAUSES|HIGH_CONTRIBUTES]->(p:Parameter)
    WHERE toLower(s.name) CONTAINS toLower($symptom_keyword)
    RETURN p.id AS parameter_id, 
           p.name AS parameter_name, 
           type(r) AS relationship, 
           r.description AS detail
    """
    return _execute_cypher(cypher_query, {"symptom_keyword": symptom_keyword})

def search_troubleshooting_kb(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Executes a dense/sparse hybrid search on the vector store to find unstructured 
    answers when a symptom doesn't have an exact graph node mapping.
    """
    try:
        store = get_vector_store()
    except Exception as e:
        return [{"error": f"Vector store initialization failed: {str(e)}"}]
    
    # In LangChain Qdrant, metadata fields are nested under the "metadata" key in the payload.
    category_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=["Troubleshooting", "Core Parameters"])
            )
        ]
    )
    
    # similarity_search automatically handles the dense + sparse hybrid vectorization
    docs = store.similarity_search(query, k=limit, filter=category_filter)
    
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "category": doc.metadata.get("category")
        } for doc in docs
    ]

# ==========================================
# 2. DOSAGE AGENT TOOLS
# ==========================================

def query_chemical_actions(parameter_id: str, desired_action: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Finds chemicals capable of modifying a given parameter, along with their baseline 
    dosage rates and side effects.
    """
    # Base query for all actions
    cypher_query = """
    MATCH (c:Chemical)-[r:RAISES|LOWERS|RAISES_SIDE_EFFECT|LOWERS_SIDE_EFFECT]->(p:Parameter {id: $parameter_id})
    """
    
    # Filter dynamically based on whether the agent wants to RAISE or LOWER the parameter
    if desired_action == "RAISES":
        cypher_query += " WHERE type(r) STARTS WITH 'RAISES' "
    elif desired_action == "LOWERS":
        cypher_query += " WHERE type(r) STARTS WITH 'LOWERS' "
        
    cypher_query += """
    RETURN c.id AS chemical_id,
           c.name AS chemical_name, 
           type(r) AS action, 
           r.description AS specific_instructions
    """
    return _execute_cypher(cypher_query, {"parameter_id": parameter_id})

def get_dosing_formulas(chemical_id: str, pool_volume_kL: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Retrieves dense instructional chunks from the vector store containing specialized 
    engineering formulas, maximum single-dose thresholds, and application methods.
    """
    
    try:
        store = get_vector_store()
    except Exception as e:
        return [{"error": f"Vector store initialization failed: {str(e)}"}]

    # Convert "C_MuriaticAcid" -> "MuriaticAcid" for better natural language matching
    clean_chemical = chemical_id.replace("C_", "")
    search_query = f"dosing formulas application instructions safety limits for {clean_chemical}"
    
    category_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=["Chemical Adjustments", "Safety"])
            )
        ]
    )
    
    docs = store.similarity_search(search_query, k=2, filter=category_filter)
    
    response = []
    for doc in docs:
        chunk_data = {
            "instructions": doc.page_content,
            "category": doc.metadata.get("category")
        }
        # Append dynamic context if the user provided their pool volume
        if pool_volume_kL:
            chunk_data["context"] = f"Apply these formulas using the user's pool volume: {pool_volume_kL} kL."
            
        response.append(chunk_data)
        
    return response

# ==========================================
# 3. EQUIPMENT AGENT TOOLS
# ==========================================

def query_hardware_impact(parameter_id: Optional[str] = None, equipment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Queries the knowledge graph for hardware components that are actively degraded, 
    corroded, or scaled by specific chemical parameter conditions.
    """
    # Build match condition dynamically based on provided arguments
    if parameter_id and equipment_id:
        match_clause = "MATCH (p:Parameter {id: $parameter_id})-[r:HIGH_DEGRADES|HIGH_CORRODES|LOW_CORRODES|HIGH_SCALES|LOW_DEGRADES]->(e:Equipment {id: $equipment_id})"
    elif parameter_id:
        match_clause = "MATCH (p:Parameter {id: $parameter_id})-[r:HIGH_DEGRADES|HIGH_CORRODES|LOW_CORRODES|HIGH_SCALES|LOW_DEGRADES]->(e:Equipment)"
    elif equipment_id:
        match_clause = "MATCH (p:Parameter)-[r:HIGH_DEGRADES|HIGH_CORRODES|LOW_CORRODES|HIGH_SCALES|LOW_DEGRADES]->(e:Equipment {id: $equipment_id})"
    else:
        return [{"error": "Must provide either parameter_id or equipment_id"}]

    cypher_query = match_clause + """
    RETURN p.id AS parameter_id,
           p.name AS parameter_name, 
           e.id AS equipment_id,
           e.name AS equipment_name, 
           type(r) AS mechanism,
           r.description AS impact_details
    """
    
    params = {}
    if parameter_id: params["parameter_id"] = parameter_id
    if equipment_id: params["equipment_id"] = equipment_id
        
    return _execute_cypher(cypher_query, params)

def search_equipment_manuals(query: str) -> List[Dict[str, Any]]:
    """
    Searches the vector store to retrieve technical operational guides, failure mode 
    symptoms, and physical maintenance workarounds.
    """
    try:
        store = get_vector_store()
    except Exception as e:
        return [{"error": f"Vector store initialization failed: {str(e)}"}]

    category_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchValue(value="Testing and Equipment")
            )
        ]
    )
    
    docs = store.similarity_search(query, k=3, filter=category_filter)
    
    return [
        {
            "manual_excerpt": doc.page_content,
            "tags": doc.metadata.get("tags")
        } for doc in docs
    ]

# ==========================================
# 4. MAINTENANCE AGENT TOOLS
# ==========================================

def search_maintenance_procedures(procedure_type: str, environmental_factor: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Queries the vector store to extract chronological checklists, safety precautions, 
    and tools required for standard upkeep tasks.
    """
    try:
        store = get_vector_store()
    except Exception as e:
        return [{"error": f"Vector store initialization failed: {str(e)}"}]

    # Build a richer semantic query
    search_query = procedure_type
    if environmental_factor:
        search_query += f" dealing with {environmental_factor}"
        
    category_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=["Maintenance", "Water Source"])
            )
        ]
    )
    
    docs = store.similarity_search(search_query, k=3, filter=category_filter)
    
    return [
        {
            "checklist_and_safety": doc.page_content,
            "source": doc.metadata.get("source")
        } for doc in docs
    ]
def query_maintenance_dependencies(node_id: str) -> List[Dict[str, Any]]:
    """
    Inspects the graph database for parameters or preparation tasks that must be 
    stabilized before or after a maintenance routine is completed.
    """
    cypher_query = """
    MATCH (c)-[r:BALANCE_BEFORE|BUFFERS]->(target {id: $node_id})
    RETURN c.id AS prerequisite_id,
           c.name AS prerequisite_name, 
           type(r) AS dependency_type,
           r.description AS dependency_details
    """
    return _execute_cypher(cypher_query, {"node_id": node_id})


# ==========================================
# 5. PREDICTIVE / FORECASTING TOOLS (New)
# ==========================================


PoolSurface = Literal[
    "plaster",
    "vinyl",
    "fiberglass",
    "pebble"
]

SanitizerType = Literal[
    "chlorine",
    "saltwater",
    "bromine"
]


# =========================================================
# LAYER 1 — PURE CHEMISTRY ENGINE
# =========================================================

def calculate_lsi(
    ph: float,
    ta: float,
    ch: float,
    temp_c: float,
    cya: Optional[float] = 0.0,
    tds: Optional[float] = 1000.0,
) -> Dict[str, Any]:
    """
    Deterministic LSI chemistry calculation.

    Formula:
        LSI = pH + TF + CF + AF - TDSF - 12.1

    Notes:
    - Uses adjusted alkalinity.
    - Includes temperature factor interpolation.
    - Includes calcium and alkalinity logarithmic factors.
    """

    adjusted_ta = max(ta - (cya / 3), 1)

    temp_f = (temp_c * 9 / 5) + 32

    temperature_factor = _temperature_factor(temp_f)

    calcium_factor = log10(max(ch, 1)) - 0.4

    alkalinity_factor = log10(adjusted_ta)

    tds_factor = _tds_factor(tds)

    lsi = (
        ph
        + temperature_factor
        + calcium_factor
        + alkalinity_factor
        - tds_factor
        - 12.1
    )

    return {
        "lsi": round(lsi, 2),

        "water_balance": {
            "ph": ph,
            "ta": ta,
            "adjusted_ta": round(adjusted_ta, 1),
            "calcium_hardness": ch,
            "temperature_c": temp_c,
            "temperature_f": round(temp_f, 1),
            "cya": cya,
            "tds": tds
        },

        "factors": {
            "temperature_factor": round(temperature_factor, 2),
            "calcium_factor": round(calcium_factor, 2),
            "alkalinity_factor": round(alkalinity_factor, 2),
            "tds_factor": round(tds_factor, 2)
        }
    }


def _temperature_factor(temp_f: float) -> float:
    """
    Approximate industry-standard temperature factor table.
    """

    if temp_f < 37:
        return 0.0
    elif temp_f < 46:
        return 0.1
    elif temp_f < 53:
        return 0.2
    elif temp_f < 60:
        return 0.3
    elif temp_f < 66:
        return 0.4
    elif temp_f < 76:
        return 0.5
    elif temp_f < 84:
        return 0.6
    elif temp_f < 94:
        return 0.7
    elif temp_f < 105:
        return 0.8

    return 0.9


def _tds_factor(tds: float) -> float:
    """
    Simplified TDS correction factor.
    """

    if tds <= 1000:
        return 0.0

    return (log10(tds) - 3) * 0.1


# =========================================================
# LAYER 2 — INTERPRETATION ENGINE
# =========================================================

def interpret_lsi(
    lsi: float,
    pool_surface: PoolSurface,
    sanitizer_type: SanitizerType,
    heater_present: bool = True
) -> Dict[str, Any]:
    """
    Converts raw LSI into operational intelligence.
    """

    state, severity = _classify_lsi(lsi)

    predicted_risks: List[Dict[str, Any]] = []

    # -----------------------------
    # CORROSION RISKS
    # -----------------------------

    if lsi < -0.3:

        if heater_present:
            predicted_risks.append({
                "equipment": "gas_heater",
                "risk": "copper_heat_exchanger_corrosion",
                "severity": "high" if lsi < -0.6 else "medium",
                "timeline": "1-6 months"
            })

        if pool_surface in ["plaster", "pebble"]:
            predicted_risks.append({
                "equipment": "pool_surface",
                "risk": "surface_etching",
                "severity": "high" if lsi < -0.6 else "medium"
            })

        if sanitizer_type == "saltwater":
            predicted_risks.append({
                "equipment": "salt_cell",
                "risk": "premature_plate_degradation",
                "severity": "medium"
            })

    # -----------------------------
    # SCALING RISKS
    # -----------------------------

    if lsi > 0.3:

        predicted_risks.append({
            "equipment": "plumbing",
            "risk": "calcium_scale_buildup",
            "severity": "high" if lsi > 0.6 else "medium"
        })

        if heater_present:
            predicted_risks.append({
                "equipment": "heater",
                "risk": "heat_exchanger_scaling",
                "severity": "high" if lsi > 0.6 else "medium"
            })

        if sanitizer_type == "saltwater":
            predicted_risks.append({
                "equipment": "salt_cell",
                "risk": "calcium_scaling_on_plates",
                "severity": "high"
            })

    return {
        "classification": {
            "state": state,
            "severity": severity
        },

        "predicted_risks": predicted_risks
    }


def _classify_lsi(lsi: float):

    if lsi < -0.6:
        return "corrosive", "high"

    elif lsi < -0.3:
        return "corrosive", "moderate"

    elif lsi <= 0.3:
        return "balanced", "ideal"

    elif lsi <= 0.6:
        return "scale_forming", "moderate"

    return "scale_forming", "high"


# =========================================================
# LAYER 3 — RECOMMENDATION ENGINE
# =========================================================

def recommend_lsi_correction(
    lsi: float,
    ph: float,
    ta: float,
    ch: float
) -> Dict[str, Any]:
    """
    Provides deterministic correction priorities.
    """

    priority_actions = []

    chemical_strategy = {
        "increase": [],
        "decrease": [],
        "avoid": []
    }

    # =====================================================
    # CORROSIVE WATER
    # =====================================================

    if lsi < -0.3:

        if ph < 7.4:
            priority_actions.append({
                "priority": 1,
                "parameter": "pH",
                "current": ph,
                "target": 7.5,
                "reason": "Fastest and safest way to raise LSI"
            })

            chemical_strategy["increase"].append("pH")

        elif ta < 80:
            priority_actions.append({
                "priority": 2,
                "parameter": "TA",
                "current": ta,
                "target": 90,
                "reason": "Improve pH stability and raise saturation"
            })

            chemical_strategy["increase"].append("TA")

        elif ch < 250:
            priority_actions.append({
                "priority": 3,
                "parameter": "CH",
                "current": ch,
                "target": 300,
                "reason": "Reduce corrosive tendency"
            })

            chemical_strategy["increase"].append("CH")

    # =====================================================
    # SCALE FORMING WATER
    # =====================================================

    elif lsi > 0.3:

        if ph > 7.8:
            priority_actions.append({
                "priority": 1,
                "parameter": "pH",
                "current": ph,
                "target": 7.4,
                "reason": "Most effective scale reduction path"
            })

            chemical_strategy["decrease"].append("pH")

        elif ta > 120:
            priority_actions.append({
                "priority": 2,
                "parameter": "TA",
                "current": ta,
                "target": 90,
                "reason": "Reduce carbonate saturation"
            })

            chemical_strategy["decrease"].append("TA")

        elif ch > 450:
            priority_actions.append({
                "priority": 3,
                "parameter": "CH",
                "current": ch,
                "target": 350,
                "reason": "Long-term scale prevention"
            })

            chemical_strategy["decrease"].append("CH")

    else:

        priority_actions.append({
            "priority": 0,
            "parameter": "none",
            "reason": "Water is balanced"
        })

    # Important operational safeguards

    if lsi < -0.5:
        chemical_strategy["avoid"].append(
            "Aggressive acid dosing"
        )

    if lsi > 0.6:
        chemical_strategy["avoid"].append(
            "Calcium increaser"
        )

    return {
        "priority_actions": priority_actions,
        "chemical_strategy": chemical_strategy
    }


# =========================================================
# ORCHESTRATOR TOOL
# =========================================================

def analyze_pool_lsi(
    ph: float,
    ta: float,
    ch: float,
    temp_c: float,
    cya: float = 0,
    tds: float = 1000,
    pool_surface: PoolSurface = "plaster",
    sanitizer_type: SanitizerType = "chlorine",
    heater_present: bool = True
) -> Dict[str, Any]:
    """
    Master orchestration tool for the agent.
    """

    chemistry = calculate_lsi(
        ph=ph,
        ta=ta,
        ch=ch,
        temp_c=temp_c,
        cya=cya,
        tds=tds
    )

    interpretation = interpret_lsi(
        lsi=chemistry["lsi"],
        pool_surface=pool_surface,
        sanitizer_type=sanitizer_type,
        heater_present=heater_present
    )

    recommendations = recommend_lsi_correction(
        lsi=chemistry["lsi"],
        ph=ph,
        ta=ta,
        ch=ch
    )

    return {
        **chemistry,
        **interpretation,
        **recommendations
    }