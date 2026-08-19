# tools.py
from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from dotenv import load_dotenv
from langchain_core.tools import tool
from neo4j import GraphDatabase, Driver
from streamlit import secrets

from src.qdrant_vector_store import cargar_vector_store

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

HIGH_VALUE_LABELS = [
    "Chemical", "Procedure", "Hazard", "Problem", "Symptom",
    "Equipment", "Parameter", "WaterParameter", "Product",
    "System", "Component", "Process", "Task", "Check", "Checklist",
    "Requirement", "Regulation", "Standard", "Organism", "Pest",
    "Strategy", "Practice", "Prevention", "Concept", "Condition",
    "Cause", "Factor", "Assessment", "Test", "Formula", "Rule",
]

# Labels that get a small score boost
TOP_PRIORITY_LABELS = {
    "Chemical", "Procedure", "Hazard", "Problem",
    "Equipment", "WaterParameter",
}

DEFAULT_PREFERRED_RELS = [
    "TREATS", "USES", "REQUIRES", "HAS_RISK", "CAUSES",
    "PART_OF", "CONTAINS", "APPLIES_TO", "RELATED_TO",
    "PREVENTS", "INDICATES", "MEASURED_BY", "PROCEDURE_FOR",
]

# Intent → preferred relationship boost
INTENT_BOOSTS: dict[str, list[str]] = {
    "risk":     ["HAS_RISK", "CAUSES"],
    "treat":    ["TREATS", "USES", "REQUIRES"],
    "procedure":["PROCEDURE_FOR", "REQUIRES", "USES"],
    "part":     ["PART_OF", "CONTAINS"],
}

# Truncation limits (keep LLM context under control)
MAX_CHUNK_CHARS = 900
MAX_DESC_CHARS  = 380
MAX_NODE_DESC   = 280
MAX_VECTOR_CTX  = 900

# ---------------------------------------------------------------------------
# Secrets & clients (simple singletons)
# ---------------------------------------------------------------------------

_neo4j_driver: Optional[Driver] = None
_vector_store = None


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Prefer env vars, fall back to Streamlit secrets."""
    return os.getenv(key) or secrets.get(key) or default


def _reset_cached_clients() -> None:
    """Useful when secrets change or for testing."""
    global _neo4j_driver, _vector_store
    if _neo4j_driver is not None:
        try:
            _neo4j_driver.close()
        except Exception:
            pass
    _neo4j_driver = None
    _vector_store = None


def get_neo4j_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        uri = _get_secret("NEO4J_URI")
        user = _get_secret("NEO4J_USER", "neo4j")
        password = _get_secret("NEO4J_PASSWORD")

        if not uri:
            raise ValueError(
                "NEO4J_URI is not configured. "
                "Add it to .streamlit/secrets.toml or Streamlit Cloud secrets."
            )
        if not password:
            raise ValueError("NEO4J_PASSWORD is not configured.")

        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        # Optional: verify connectivity on first use
        # _neo4j_driver.verify_connectivity()
    return _neo4j_driver


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = cargar_vector_store()
    return _vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _normalize_seed_ids(seed_node_ids: list[str] | str) -> list[str]:
    if isinstance(seed_node_ids, str):
        return [s.strip() for s in re.split(r"[,;\s]+", seed_node_ids) if s.strip()]
    return [str(s).strip() for s in seed_node_ids if str(s).strip()]


def _detect_intent_boosts(query: str) -> list[str]:
    q = query.lower()
    if any(w in q for w in ("riesgo", "peligro", "toxic", "hazard", "risk")):
        return INTENT_BOOSTS["risk"]
    if any(w in q for w in ("tratar", "tratamiento", "treat", "chemical", "producto")):
        return INTENT_BOOSTS["treat"]
    if any(w in q for w in ("procedimiento", "cómo", "how to", "steps", "limpiar", "clean")):
        return INTENT_BOOSTS["procedure"]
    if any(w in q for w in ("parte", "componente", "part of", "contains")):
        return INTENT_BOOSTS["part"]
    return []

def _as_list(value, delimiters: tuple[str, ...] = (";", ",")) -> list[str]:
    """Coerce a Neo4j property that may be LIST<STRING> or a delimited STRING into list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for d in delimiters:
            if d in text:
                return [p.strip() for p in text.split(d) if p.strip()]
        return [text]
    return [str(value)]

# ---------------------------------------------------------------------------
# 1. Vector store tool
# ---------------------------------------------------------------------------

@tool
def vector_search(query: str, k: int = 6) -> str:
    """
    Search the vector store for the most semantically relevant text chunks
    about aquatic facilities, pools, spas, water chemistry, cleaning,
    risks and procedures.

    Use this tool FIRST on almost every user question.
    It returns the top matching document chunks with source and category.
    """
    try:
        store = get_vector_store()
        results = store.similarity_search_with_score(query, k=k)

        if not results:
            return "No relevant chunks found in the vector store."

        parts: list[str] = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata or {}
            content = _truncate(doc.page_content, MAX_CHUNK_CHARS)
            parts.append(
                f"--- Chunk {i} (score: {score:.4f}) ---\n"
                f"Source: {meta.get('source', 'unknown')}\n"
                f"Category: {meta.get('category', 'unknown')}\n"
                f"Chunk ID: {meta.get('chunk_id', '')}\n"
                f"Content:\n{content}\n"
            )
        return "\n".join(parts)

    except Exception as e:
        return f"Error while searching vector store: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 2. Seed-node retrieval (Neo4j)
# ---------------------------------------------------------------------------

@tool
def search_seed_nodes(
    query: str,
    vector_chunks: str = "",
    top_k: int = 10,
    min_score: float = 0.25,
) -> str:
    """
    Identify and retrieve the most relevant seed nodes from the Neo4j graph
    based on the user question (and optionally on the chunks from the vector store).

    Searches primarily in: name, canonical_name, aliases, keywords and id.
    Uses description as a secondary signal.
    Prioritizes high-value semantic labels of the domain.

    Returns the candidate nodes with their description and initial score.
    """
    try:
        driver = get_neo4j_driver()

        # Enriched search text (now actually used)
        search_text = query.strip()
        if vector_chunks and len(vector_chunks) > 80:
            search_text = f"{query}\n\n{_truncate(vector_chunks, MAX_VECTOR_CTX)}"

        # Prefer a full-text index if you have one.
        # Create it once with:
        #   CREATE FULLTEXT INDEX node_search IF NOT EXISTS
        #   FOR (n:Chemical|Procedure|Hazard|...) ON EACH [n.name, n.aliases, n.keywords, n.id, n.description]
        # Then the query becomes much simpler and faster.
        #
        # Fallback below works without an index (still usable for moderate graphs).

        cypher = """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN $high_value_labels)
          AND (
               n.name IS NOT NULL
            OR n.aliases IS NOT NULL
            OR n.keywords IS NOT NULL
            OR n.id IS NOT NULL
            OR n.description IS NOT NULL
          )

        WITH n,
             toLower($search_text) AS q,
             toLower(coalesce(n.name, '')) AS name_l,
             toLower(coalesce(n.id, '')) AS id_l,
             [a IN coalesce(n.aliases, []) | toLower(a)] AS aliases_l,
             [k IN coalesce(n.keywords, []) | toLower(k)] AS keywords_l,
             toLower(coalesce(n.description, '')) AS desc_l

        WITH n, q, name_l, id_l, aliases_l, keywords_l, desc_l,
             CASE
                WHEN name_l = q OR id_l = q THEN 1.15
                WHEN any(a IN aliases_l WHERE a = q) THEN 1.05
                WHEN name_l CONTAINS q THEN 0.92
                WHEN any(a IN aliases_l WHERE a CONTAINS q) THEN 0.88
                WHEN any(k IN keywords_l WHERE k CONTAINS q) THEN 0.78
                WHEN desc_l CONTAINS q THEN 0.55
                WHEN any(word IN split(q, ' ')
                         WHERE size(word) > 3
                           AND (name_l CONTAINS word
                                OR any(a IN aliases_l WHERE a CONTAINS word)
                                OR any(k IN keywords_l WHERE k CONTAINS word)))
                     THEN 0.65
                ELSE 0.0
             END AS score

        WHERE score >= $min_score

        WITH n, score,
             CASE
                WHEN any(lbl IN labels(n) WHERE lbl IN $top_priority_labels)
                THEN score + 0.08
                ELSE score
             END AS final_score

        RETURN n AS node, final_score AS score
        ORDER BY final_score DESC
        LIMIT $top_k
        """

        with driver.session() as session:
            result = session.run(
                cypher,
                search_text=search_text,          # ← now correctly passed
                high_value_labels=HIGH_VALUE_LABELS,
                top_priority_labels=list(TOP_PRIORITY_LABELS),
                top_k=top_k,
                min_score=min_score,
            )
            rows = list(result)

        if not rows:
            return "No se encontraron nodos semilla relevantes en el grafo."

        seen: set[str] = set()
        parts: list[str] = []
        for i, record in enumerate(rows, 1):
            node = record["node"]
            score = record["score"]

            node_id = node.get("id") or node.element_id
            if node_id in seen:
                continue
            seen.add(node_id)

            name = node.get("name") or node.get("id") or "Sin nombre"
            labels = list(node.labels)
            description = _truncate(
                node.get("description") or node.get("summary") or "", MAX_DESC_CHARS
            )
            aliases = _as_list(node.get("aliases"))
            keywords = _as_list(node.get("keywords"))

            parts.append(
                f"--- Seed {i} (score: {score:.3f}) ---\n"
                f"ID: {node_id}\n"
                f"Name: {name}\n"
                f"Label(s): {', '.join(labels)}\n"
                f"Aliases: {', '.join(aliases) if aliases else '—'}\n"
                f"Keywords: {', '.join(keywords) if keywords else '—'}\n"
                f"Description: {description or '—'}\n"
            )

        return "\n".join(parts)

    except Exception as e:
        return f"Error en search_seed_nodes: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 3. Subgraph expansion
# ---------------------------------------------------------------------------

@tool
def expand_subgraph(
    seed_node_ids: list[str] | str,
    query: str = "",
    max_hops: int = 2,
    max_nodes: int = 25,
    max_edges: int = 40,
    preferred_rels: Sequence[str] | None = None,
) -> str:
    """
    Expand the Neo4j graph starting from one or more seed nodes (1–2 hops).
    Prioritizes useful relationship types according to the question intent
    (TREATS, USES, REQUIRES, HAS_RISK, CAUSES, PART_OF, RELATED_TO, etc.).

    Returns a ranked and limited subgraph (nodes + relationships + descriptions)
    ready to be injected into the LLM context.
    """
    try:
        driver = get_neo4j_driver()
        seed_ids = _normalize_seed_ids(seed_node_ids)

        if not seed_ids:
            return "No seed nodes provided."

        # Build preferred relationship list (intent boost first)
        base_rels = list(preferred_rels) if preferred_rels is not None else list(DEFAULT_PREFERRED_RELS)
        boosts = _detect_intent_boosts(query)
        preferred = boosts + [r for r in base_rels if r not in boosts]

        # Safer expansion: collect paths, then limit nodes/edges in application layer
        # (or use APOC if available). This version stays pure Cypher.
        cypher = """
        UNWIND $seed_ids AS seed_id
        MATCH (seed)
        WHERE seed.id = seed_id
           OR elementId(seed) = seed_id
           OR seed.name = seed_id
        WITH collect(DISTINCT seed) AS seeds

        UNWIND seeds AS s
        CALL {
            WITH s
            MATCH path = (s)-[r*1..$max_hops]-(n)
            WHERE ALL(rel IN relationships(path) WHERE type(rel) IN $preferred_rels)
               OR size($preferred_rels) = 0
            RETURN path
            LIMIT 40
        }
        WITH seeds, collect(DISTINCT path) AS paths

        UNWIND paths AS p
        UNWIND nodes(p) AS node
        WITH seeds, paths, collect(DISTINCT node) AS all_nodes

        UNWIND paths AS p
        UNWIND relationships(p) AS rel
        WITH seeds, all_nodes, collect(DISTINCT rel) AS all_rels

        WITH seeds,
             all_nodes[0..$max_nodes] AS limited_nodes,
             all_rels[0..$max_edges] AS limited_rels

        RETURN
            [n IN limited_nodes | {
                id: coalesce(n.id, elementId(n)),
                name: coalesce(n.name, n.id, 'Unnamed'),
                label: head(labels(n)),
                description: coalesce(n.description, n.summary, '')
            }] AS nodes,
            [r IN limited_rels | {
                type: type(r),
                source: coalesce(startNode(r).id, elementId(startNode(r))),
                target: coalesce(endNode(r).id, elementId(endNode(r))),
                properties: properties(r)
            }] AS relationships,
            [s IN seeds | coalesce(s.id, elementId(s))] AS seed_ids_used
        """

        with driver.session() as session:
            result = session.run(
                cypher,
                seed_ids=seed_ids,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                preferred_rels=preferred,
            )
            record = result.single()

        if not record or not record["nodes"]:
            return "No subgraph could be expanded from the provided seed nodes."

        nodes = record["nodes"]
        relationships = record["relationships"]
        seeds_used = record["seed_ids_used"]

        parts: list[str] = [
            f"=== Subgraph expanded from seeds: {', '.join(map(str, seeds_used))} ===\n",
            f"Nodes ({len(nodes)}):\n",
        ]

        for i, n in enumerate(nodes, 1):
            desc = _truncate(n.get("description") or "", MAX_NODE_DESC)
            parts.append(
                f"  {i}. [{n['label']}] {n['name']} (id: {n['id']})\n"
                f"     {desc}\n"
            )

        parts.append(f"\nRelationships ({len(relationships)}):\n")
        for r in relationships:
            props = r.get("properties") or {}
            interesting = {
                k: v for k, v in props.items()
                if k in ("weight", "confidence", "note", "severity")
            }
            prop_str = f" | {interesting}" if interesting else ""
            parts.append(
                f"  ({r['source']}) -[{r['type']}]-> ({r['target']}){prop_str}"
            )

        return "\n".join(parts)

    except Exception as e:
        return f"Error in expand_subgraph: {type(e).__name__}: {e}"