# tools.py ok
from __future__ import annotations

import os
import re
from functools import lru_cache
import threading
from typing import Optional, Sequence, Any, Literal

from dotenv import load_dotenv
from langchain_core.tools import tool, ToolException
from neo4j import GraphDatabase, Driver
from streamlit import secrets

from ..qdrant_vector_store import cargar_vector_store, VectorStoreConfigError

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
 
FULLTEXT_INDEX = "node_search"
 
# Blocklist estructural (invierte la antigua whitelist HIGH_VALUE_LABELS:
# ningún nodo de dominio queda invisible por no estar en una lista de 30)
STRUCTURAL_LABELS = ["Chapter", "Section", "Document", "Source", "Appendix", "Table"]
 
TOP_PRIORITY_LABELS = [
    "Chemical", "Procedure", "Hazard", "Problem", "Equipment", "WaterParameter",
]
 
DEFAULT_PREFERRED_RELS = [
    "TREATS", "USES", "REQUIRES", "HAS_RISK", "CAUSES",
    "PART_OF", "CONTAINS", "APPLIES_TO",
    "PREVENTS", "INDICATES", "MEASURED_BY", "PROCEDURE_FOR",
]
INTENT_LABELS: dict[str, tuple[str, ...]] = {
    "normative":   ("Requirement", "WaterParameter", "Threshold", "Standard", "Code"),
    "procedural":  ("Procedure", "Operation", "Task", "Role"),
    "diagnostic":  ("Hazard", "Symptom", "Cause", "Risk", "Chemical"),
    "descriptive": ("Concept", "Chemical", "Equipment", "Venue"),
    "any":         TOP_PRIORITY_LABELS,
}

# Un seed cuya etiqueta no puede responder la pregunta se degrada, no se elimina:
# puede seguir siendo útil como contexto, pero no debe encabezar la lista.
OFF_INTENT_PENALTY = 0.45
ABSOLUTE_FLOOR = 0.35          # por debajo de esto, ningún seed es fiable
NEIGHBOR_LIMIT = 8
# Relaciones genéricas: se permiten a 1 hop pero explotan por hubs a 2+.
# Se bloquean en expansión multi-hop, no en la lista de preferidas.
GENERIC_RELS = ["RELATED_TO", "MENTIONED_IN", "SEE_ALSO"]
 
INTENT_BOOSTS: dict[str, list[str]] = {
    "risk":      ["HAS_RISK", "CAUSES"],
    "treat":     ["TREATS", "USES", "REQUIRES"],
    "procedure": ["PROCEDURE_FOR", "REQUIRES", "USES"],
    "part":      ["PART_OF", "CONTAINS"],
}
 
# --- Pesos de scoring ------------------------------------------------------
W_NAME, W_ALIAS, W_KEYWORD, W_DESC = 1.00, 0.90, 0.60, 0.25
W_BIGRAM, BIGRAM_CAP = 0.25, 0.35
W_NUMERIC = 0.15
W_LABEL = 0.05
 
RELATIVE_GATE = 0.60      # descarta lo que no llegue al 60% del top
MIN_SCORE_MULTI = 0.35    # gate absoluto con >= 2 términos
MIN_SCORE_SINGLE = 0.55   # gate absoluto con 1 término (evita match solo-descripción)
CANDIDATE_LIMIT = 50      # recall del fulltext antes del re-ranking
 
MAX_DESC_CHARS = 400
MAX_NODE_DESC = 240
MAX_VECTOR_CTX = 1200
 
# --- Tokenización ----------------------------------------------------------
 
FUNCTION_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "which",
    "does", "how", "are", "can", "should", "must", "have", "has", "was", "were",
    "you", "your", "its", "any", "all", "not", "but", "per", "into", "than",
    "is", "an", "of", "in", "on", "to", "at", "be", "it", "or", "as", "my",
}
 
# Genéricos de dominio: matchean casi todo el grafo. Se filtran en primera
# pasada, pero el fallback los recupera si no queda nada más.
DOMAIN_STOPWORDS = {
    "pool", "pools", "water", "spa", "spas", "level", "levels", "chemical",
    "facility", "aquatic", "operator",
}
 
STOPWORDS = FUNCTION_WORDS | DOMAIN_STOPWORDS
 
# Tokens cortos que SÍ son vocabulario core (el viejo size(word) > 3 los borraba)
DOMAIN_SHORT_TOKENS = {
    "ph", "cya", "fc", "tc", "cc", "ta", "ch", "lsi", "orp", "dpd", "ppm",
    "uv", "do", "rwi", "cdc", "mahc", "epa", "vgb", "gpm", "psi", "aoi",
}
 
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9%/\.\-]*")
_NUMERIC_RE = re.compile(r"[0-9%/]")
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
 
 
# ---------------------------------------------------------------------------
# Estado del driver
# ---------------------------------------------------------------------------
 
_neo4j_driver: Driver | None = None
_driver_lock = threading.Lock()
_fulltext_available: bool | None = None
 


# Truncation limits (keep LLM context under control)
MAX_CHUNK_CHARS = 1200
DEFAULT_K = 4
MAX_DESC_CHARS  = 380
MAX_NODE_DESC   = 280
MAX_VECTOR_CTX  = 900

# ---------------------------------------------------------------------------
# Secrets & clients (simple singletons)
# ---------------------------------------------------------------------------

_neo4j_driver: Optional[Driver] = None
_vector_store = None
_store_error: str | None = None  # cachea el fallo: no reintentar 6 veces por turno
 

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
    """Thread-safe. En Streamlit, envolver con @st.cache_resource."""
    global _neo4j_driver
    if _neo4j_driver is None:
        with _driver_lock:
            if _neo4j_driver is None:  # doble check
                uri = _get_secret("NEO4J_URI")
                user = _get_secret("NEO4J_USER", "neo4j")
                password = _get_secret("NEO4J_PASSWORD")
                if not uri:
                    raise ValueError("NEO4J_URI no está configurado.")
                if not password:
                    raise ValueError("NEO4J_PASSWORD no está configurado.")
                _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver
 
 
def _has_fulltext_index(driver: Driver) -> bool:
    """Detecta el índice una sola vez por proceso."""
    global _fulltext_available
    if _fulltext_available is None:
        try:
            with driver.session() as s:
                rows = s.run(
                    "SHOW INDEXES YIELD name, type, state "
                    "WHERE name = $n AND type = 'FULLTEXT' AND state = 'ONLINE' "
                    "RETURN count(*) AS c",
                    n=FULLTEXT_INDEX,
                ).single()
            _fulltext_available = bool(rows and rows["c"] > 0)
        except Exception:
            _fulltext_available = False
    return _fulltext_available
 


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = cargar_vector_store()
    return _vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
 
 
def _as_list(value: Any) -> list[str]:
    """Defensivo: tolera listas, strings delimitados por ';' o ',' y None."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value)
    sep = ";" if ";" in raw else ","
    return [p.strip() for p in raw.split(sep) if p.strip()]
 
 
def _tokenize(query: str, extra_text: str = "") -> dict[str, Any]:
    """
    Devuelve tres artefactos distintos:
      phrase  -> solo si la query es corta (<=4 tokens); alimenta el tier exacto
      terms   -> tokens de contenido, dedup, cap 12
      bigrams -> pares adyacentes, cap 6 (lo que discrimina de verdad)
    `extra_text` (chunks del vector store) SOLO aporta términos extra;
    nunca entra en `phrase`.
    """
    q = (query or "").lower().strip()
    raw = _TOKEN_RE.findall(q)
 
    def _keep(tok: str) -> bool:
        if tok in DOMAIN_SHORT_TOKENS:
            return True
        if tok in STOPWORDS:
            return False
        if _NUMERIC_RE.search(tok):     # 65%, 742/2013, 0.71 -> siempre valiosos
            return True
        return len(tok) > 3
 
    terms = [t for t in raw if _keep(t)]
 
    # Fallback: si el filtro de dominio lo dejó vacío ("pool water levels"),
    # recuperar los genéricos de dominio antes de rendirse
    if not terms:
        terms = [
            t for t in raw
            if t not in FUNCTION_WORDS and (len(t) > 2 or t in DOMAIN_SHORT_TOKENS)
        ]
 
    terms = list(dict.fromkeys(terms))[:12]
 
    # Bigramas sobre tokens ADYACENTES en la query original: quitar stopwords
    # antes crearía adyacencias falsas ("cya an" desde "pH and CYA for an indoor")
    bigrams = list(dict.fromkeys(
        f"{a} {b}"
        for a, b in zip(raw, raw[1:])
        if a not in FUNCTION_WORDS and b not in FUNCTION_WORDS
    ))[:6]
 
    numeric_terms = [t for t in terms if _NUMERIC_RE.search(t)]
 
    phrase = q if 0 < len(raw) <= 4 else ""
 
    if extra_text and len(extra_text) > 80:
        extra = _TOKEN_RE.findall(_truncate(extra_text, MAX_VECTOR_CTX).lower())
        extra = [t for t in extra if _keep(t) and t not in terms]
        # los términos del vector store pesan igual pero van al final del cap
        terms = (terms + list(dict.fromkeys(extra)))[:12]
 
    return {
        "phrase": phrase,
        "terms": terms,
        "bigrams": bigrams,
        "numeric_terms": numeric_terms,
    }
 
 
def _lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)
 
 
def _lucene_query(terms: Sequence[str], bigrams: Sequence[str]) -> str:
    parts = ['"' + _lucene_escape(b) + '"^2' for b in bigrams]
    parts += [_lucene_escape(t) for t in terms]
    return " OR ".join(p for p in parts if p)
 
 
def _detect_intent_boosts(query: str) -> list[str]:
    q = (query or "").lower()
    boosts: list[str] = []
    for key, rels in INTENT_BOOSTS.items():
        if key in q:
            boosts += [r for r in rels if r not in boosts]
    return boosts
 
 
def _normalize_seed_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        raw = str(value).strip()
        if raw.startswith("["):        # el LLM a veces manda un JSON string
            raw = raw.strip("[]").replace('"', "").replace("'", "")
        items = raw.split(",")
    return [i.strip() for i in items if i.strip()]


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

 
_SCORING_BODY = """
WITH n, lucene,
     toLower(coalesce(n.name, '')) AS name_l,
     toLower(coalesce(n.id, ''))   AS id_l,
     [a IN (CASE
              WHEN n.aliases IS NULL THEN []
              WHEN valueType(n.aliases) STARTS WITH 'LIST' THEN n.aliases
              ELSE split(toString(n.aliases), ';')
            END) | toLower(trim(toString(a)))] AS aliases_l,
     [k IN (CASE
              WHEN n.keywords IS NULL THEN []
              WHEN valueType(n.keywords) STARTS WITH 'LIST' THEN n.keywords
              ELSE split(toString(n.keywords), ';')
            END) | toLower(trim(toString(k)))] AS keywords_l,
     toLower(coalesce(n.description, n.summary, '')) AS desc_l
 
// --- cobertura por término: max sobre campos, no suma sobre campos ---
WITH n, lucene, name_l, id_l, aliases_l, keywords_l, desc_l,
     reduce(acc = 0.0, t IN $terms |
        acc + CASE
            WHEN name_l CONTAINS t                              THEN $w_name
            WHEN any(a IN aliases_l  WHERE a CONTAINS t)        THEN $w_alias
            WHEN any(k IN keywords_l WHERE k CONTAINS t)        THEN $w_keyword
            WHEN desc_l CONTAINS t                              THEN $w_desc
            ELSE 0.0
        END
     ) AS term_sum,
     reduce(acc = 0.0, b IN $bigrams |
        acc + CASE
            WHEN name_l CONTAINS b OR any(a IN aliases_l WHERE a CONTAINS b)
            THEN $w_bigram ELSE 0.0
        END
     ) AS bigram_raw,
     reduce(acc = 0.0, t IN $numeric_terms |
        acc + CASE
            WHEN name_l CONTAINS t
              OR any(a IN aliases_l  WHERE a CONTAINS t)
              OR any(k IN keywords_l WHERE k CONTAINS t)
              OR desc_l CONTAINS t
            THEN $w_numeric ELSE 0.0
        END
     ) AS numeric_bonus
 
WITH n, lucene, aliases_l, name_l, id_l,
     // normalizado por nº de términos -> mide cobertura de la pregunta,
     // no superficie del nodo (mata el sesgo de listas largas de keywords)
     CASE WHEN size($terms) = 0 THEN 0.0
          ELSE term_sum / toFloat(size($terms)) END AS base,
     CASE WHEN bigram_raw > $bigram_cap THEN $bigram_cap ELSE bigram_raw END AS bigram_bonus,
     numeric_bonus
 
WITH n, lucene, base, bigram_bonus, numeric_bonus,
     CASE
        WHEN $phrase <> '' AND (name_l = $phrase OR id_l = $phrase) THEN 1.0
        WHEN $phrase <> '' AND any(a IN aliases_l WHERE a = $phrase) THEN 0.95
        ELSE 0.0
     END AS exact
 
WITH n, lucene,
     CASE WHEN exact > 0.0
          THEN exact
          ELSE base + bigram_bonus + numeric_bonus
     END AS pre_label
 
WITH n, lucene,
     pre_label + CASE
        WHEN any(l IN labels(n) WHERE l IN $top_priority_labels) THEN $w_label
        ELSE 0.0
     END AS raw_score
 
WITH n, lucene, CASE WHEN raw_score > 1.0 THEN 1.0 ELSE raw_score END AS score
WHERE score >= $min_score
RETURN n AS node, score, lucene
ORDER BY score DESC, lucene DESC, n.id ASC
LIMIT $top_k
"""
 
_FULLTEXT_HEAD = """
CALL db.index.fulltext.queryNodes($ft_index, $ft_query, {limit: $candidate_limit})
YIELD node AS n, score AS lucene
WHERE NOT any(l IN labels(n) WHERE l IN $structural_labels)
"""
 
_SCAN_HEAD = """
MATCH (n)
WHERE NOT any(l IN labels(n) WHERE l IN $structural_labels)
  AND (n.name IS NOT NULL OR n.id IS NOT NULL OR n.keywords IS NOT NULL)
WITH n, 0.0 AS lucene
"""

# ---------------------------------------------------------------------------
# 1. Vector store tool
# ---------------------------------------------------------------------------

def get_vector_store():
    global _vector_store, _store_error
 
    if _store_error is not None:
        raise VectorStoreConfigError(_store_error)
 
    if _vector_store is None:
        try:
            _vector_store = cargar_vector_store()
        except Exception as e:
            _store_error = f"{type(e).__name__}: {e}"
            raise
 
    return _vector_store
 
 
def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " […]"
 
 
@tool
def vector_search(query: str, k: int = DEFAULT_K, regulatory_only: bool = False) -> str:
    """
    Search the pool manual for the most relevant text chunks about aquatic
    facilities, water chemistry, equipment, procedures, risks and safety.
 
    The corpus is built from CDC MAHC, OSHA and EPA public-domain sources.
    It covers US practice and general operations. It does NOT contain the
    national regulations of other countries.
 
    Args:
        query: Search terms in English.
        k: Number of chunks to return (default 4).
        regulatory_only: If True, return only chunks flagged as regulatory.
            Use for compliance questions. If this returns nothing, the corpus
            has no regulatory basis for the question — say so instead of
            answering from general knowledge.
 
    Returns the top matching chunks with chapter, section path and a
    REGULATORY flag where applicable.
    """
    try:
        store = get_vector_store()
    except Exception as e:
        # Fallo de infra: reformular la query no ayuda. Decirselo explicito.
        return (
            f"TOOL UNAVAILABLE (configuration error): {type(e).__name__}: {e}\n"
            "This is not a search failure. Do NOT retry this tool or rephrase "
            "the query. Continue using other tools, and state clearly that "
            "document retrieval was unavailable."
        )
 
    try:
        filtro = None
        if regulatory_only:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
 
            filtro = Filter(
                must=[
                    FieldCondition(
                        key="metadata.is_regulatory", match=MatchValue(value=True)
                    )
                ]
            )
 
        results = store.similarity_search_with_score(query, k=k, filter=filtro)
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"
 
    if not results:
        alcance = "regulatory chunks" if regulatory_only else "chunks"
        return (
            f"No matching {alcance} found for: {query!r}\n"
            "The manual may not cover this topic. Do not substitute general "
            "knowledge for a citation from the manual."
        )
 
    partes: list[str] = []
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata or {}
        flag = " | REGULATORY" if m.get("is_regulatory") else ""
        partes.append(
            f"--- Chunk {i} (score: {score:.4f}){flag} ---\n"
            f"Chapter: {m.get('chapter', '?')} | "
            f"Path: {m.get('hierarchy_path', 'unknown')}\n"
            f"Type: {m.get('fragment_type', '?')} | ID: {m.get('chunk_id', '')}\n"
            f"Content:\n{_truncate(doc.page_content, MAX_CHUNK_CHARS)}\n"
        )
 
    n_reg = sum(bool((d.metadata or {}).get("is_regulatory")) for d, _ in results)
    encabezado = (
        f"[{len(results)} chunks, {n_reg} regulatory | "
        f"corpus: MAHC/OSHA/EPA, US-focused]\n\n"
    )
    return encabezado + "\n".join(partes)


# ---------------------------------------------------------------------------
# TOOL 1 — search_seed_nodes
# ---------------------------------------------------------------------------
 

@lru_cache(maxsize=1)
def _fulltext_available() -> bool:
    # _has_fulltext_index hacía un round-trip a Neo4j en cada invocación.
    return _has_fulltext_index(get_neo4j_driver())


_NEIGHBOR_CYPHER = """
MATCH (s) WHERE s.id IN $seed_ids
MATCH (s)-[r]-(n)
WHERE any(l IN labels(n) WHERE l IN $intent_labels)
  AND NOT n.id IN $seed_ids
RETURN DISTINCT n.id AS id,
       labels(n) AS labels,
       n.name AS name,
       type(r) AS rel,
       coalesce(n.description, n.summary, '') AS description
LIMIT $limit
"""


@tool
def search_seed_nodes(
    query: str,
    intent: Literal["normative", "procedural", "diagnostic", "descriptive", "any"] = "any",
    vector_chunks: str = "",
    top_k: int = 5,
) -> str:
    """
    Find the most relevant seed nodes in the Neo4j graph for a question, plus
    the normative/procedural nodes one hop away from them.

    Pass `intent` so that node labels capable of answering the question are
    ranked first. Use "normative" for any question about a threshold, range,
    limit, requirement or code provision — otherwise an Equipment or Concept
    node may outrank the Requirement node that holds the actual answer.

    The first line of the result is a machine-readable STATUS:
      OK                 usable seeds found; related nodes listed
      WEAK               seeds found but confidence is low; treat as unconfirmed
      NO_GRAPH_COVERAGE  nothing relevant. STOP retrieving on this topic.

    When related nodes are returned they are already answer candidates: if one
    of them states the value asked for, call expand_subgraph on it once, or
    answer directly. Do not run further vector searches to confirm it.
    """
    driver = get_neo4j_driver()
    tok = _tokenize(query, vector_chunks)
    terms = tok["terms"]

    if not terms:
        return (
            "STATUS: NO_GRAPH_COVERAGE\n"
            "La consulta no contiene términos buscables. No hay seeds.\n"
            "No inventes ids de nodo. Reporta insufficient_evidence y detente."
        )

    intent_labels = INTENT_LABELS.get(intent, INTENT_LABELS["any"])
    min_score = MIN_SCORE_MULTI if len(terms) >= 2 else MIN_SCORE_SINGLE

    params: dict[str, Any] = {
        "terms": terms,
        "bigrams": tok["bigrams"],
        "numeric_terms": tok["numeric_terms"],
        "phrase": tok["phrase"],
        "structural_labels": STRUCTURAL_LABELS,
        "top_priority_labels": list(intent_labels),   # ← antes: constante global
        "w_name": W_NAME, "w_alias": W_ALIAS,
        "w_keyword": W_KEYWORD, "w_desc": W_DESC,
        "w_bigram": W_BIGRAM, "bigram_cap": BIGRAM_CAP,
        "w_numeric": W_NUMERIC, "w_label": W_LABEL,
        "min_score": min_score,
        # se sobre-recupera para que el re-rank por intención tenga con qué trabajar
        "top_k": max(1, min(int(top_k), 10)) * 3,
    }

    use_ft = _fulltext_available()
    if use_ft:
        cypher = _FULLTEXT_HEAD + _SCORING_BODY
        params["ft_index"] = FULLTEXT_INDEX
        params["ft_query"] = _lucene_query(terms, tok["bigrams"])
        params["candidate_limit"] = CANDIDATE_LIMIT
    else:
        cypher = _SCAN_HEAD + _SCORING_BODY

    try:
        with driver.session() as session:
            rows = list(session.run(cypher, **params))
    except Exception as e:
        raise ToolException(f"search_seed_nodes falló: {type(e).__name__}: {e}") from e

    if not rows:
        return (
            "STATUS: NO_GRAPH_COVERAGE\n"
            f"Ningún nodo del grafo cubre esta consulta. Términos: {', '.join(terms)}\n"
            "No inventes ids de nodo ni llames a expand_subgraph.\n"
            "No reformules esta búsqueda. Reporta insufficient_evidence y detente."
        )

    # Re-rank por intención ANTES del gate relativo. Sin esto el gate se ancla
    # al top_score de un nodo con la etiqueta equivocada y arrastra a todos.
    scored = []
    for r in rows:
        node = r["node"]
        on_intent = bool(set(node.labels) & set(intent_labels))
        adj = r["score"] if (on_intent or intent == "any") else r["score"] * OFF_INTENT_PENALTY
        scored.append((adj, r["score"], node, on_intent))
    scored.sort(key=lambda x: x[0], reverse=True)

    limit = max(1, min(int(top_k), 10))
    top_adj = scored[0][0]
    kept = [s for s in scored if s[0] >= RELATIVE_GATE * top_adj][:limit]

    status = "OK"
    if top_adj < ABSOLUTE_FLOOR or not any(s[3] for s in kept):
        status = "WEAK"

    parts: list[str] = [
        f"STATUS: {status}",
        f"=== {len(kept)} seed(s) | intent: {intent} | términos: {', '.join(terms)} "
        f"| modo: {'fulltext' if use_ft else 'scan'} ===\n",
    ]
    if status == "WEAK":
        parts.append(
            f"AVISO: ningún seed con etiqueta capaz de responder una pregunta "
            f"'{intent}'. El grafo probablemente no modela esto. Una búsqueda "
            f"más como máximo, después reporta insufficient_evidence.\n"
        )

    seed_ids: list[str] = []
    for i, (adj, raw, node, on_intent) in enumerate(kept, 1):
        node_id = node.get("id") or node.element_id
        seed_ids.append(node_id)
        aliases = _as_list(node.get("aliases"))
        keywords = _as_list(node.get("keywords"))
        flag = "" if on_intent else "  [off-intent: contexto, no respuesta]"
        parts.append(
            f"--- Seed {i} (score: {adj:.3f}{'' if adj == raw else f' / raw {raw:.3f}'}){flag} ---\n"
            f"ID: {node_id}\n"
            f"Name: {node.get('name') or node_id}\n"
            f"Label(s): {', '.join(node.labels)}\n"
            f"Aliases: {', '.join(aliases[:12]) if aliases else '—'}\n"
            f"Keywords: {', '.join(keywords[:15]) if keywords else '—'}\n"
            f"Description: "
            f"{_truncate(node.get('description') or node.get('summary') or '', MAX_DESC_CHARS) or '—'}\n"
        )

    # Un salto hacia las etiquetas que responden la pregunta. Esto es lo que
    # evita el viaje extra a expand_subgraph — y lo que hace innecesario que el
    # agente adivine slugs.
    try:
        with driver.session() as session:
            neighbors = list(session.run(
                _NEIGHBOR_CYPHER,
                seed_ids=seed_ids,
                intent_labels=list(intent_labels),
                limit=NEIGHBOR_LIMIT,
            ))
    except Exception:
        neighbors = []   # degradación silenciosa: los seeds ya son útiles

    if neighbors:
        parts.append(f"\n=== {len(neighbors)} nodo(s) {intent} a 1 salto ===\n")
        for n in neighbors:
            parts.append(
                f"  [{'/'.join(n['labels'])}] {n['name'] or n['id']} (id: {n['id']}) "
                f"vía {n['rel']}\n"
                f"    {_truncate(n['description'], MAX_DESC_CHARS)}\n"
            )
        parts.append(
            "\nEstos nodos son candidatos a respuesta. Si uno de ellos contiene "
            "el valor pedido, ya tienes la respuesta: cítalo y detente. No "
            "busques confirmación en prosa.\n"
        )

    parts.append(
        "Usa EXACTAMENTE los IDs de arriba en expand_subgraph. "
        "No construyas ids que no aparezcan en esta lista."
    )
    return "\n".join(parts)
 
 
# ---------------------------------------------------------------------------
# TOOL 2 — expand_subgraph
# ---------------------------------------------------------------------------
 
_RESOLVE_CYPHER = """
MATCH (s)
WHERE s.id IN $seed_ids OR elementId(s) IN $seed_ids OR s.name IN $seed_ids
RETURN elementId(s) AS eid, coalesce(s.id, elementId(s)) AS sid, s.name AS name
"""
 
# {H} se interpola: Neo4j NO admite parámetros en los bounds de longitud variable
_EXPAND_CYPHER = """
MATCH (s) WHERE elementId(s) IN $seed_eids
CALL {{
    WITH s
    MATCH path = (s)-[*1..{H}]-(m)
    WHERE elementId(m) <> elementId(s)
      AND NOT any(l IN labels(m) WHERE l IN $structural_labels)
      AND none(rel IN relationships(path)
               WHERE length(path) > 1 AND type(rel) IN $generic_rels)
    RETURN m,
           length(path) AS dist,
           reduce(p = 0, rel IN relationships(path) |
                  p + CASE WHEN type(rel) IN $preferred_rels THEN 1 ELSE 0 END) AS pref
    ORDER BY dist ASC, pref DESC
    LIMIT $per_seed_limit
}}
WITH m, min(dist) AS dist, max(pref) AS pref
ORDER BY dist ASC, pref DESC, m.id ASC
LIMIT $max_nodes
RETURN collect(elementId(m)) AS neighbor_eids
"""
 
# Subgrafo inducido: solo aristas cuyos DOS extremos están en el set final.
# Elimina las aristas colgantes que el slice independiente producía.
_FETCH_CYPHER = """
MATCH (n) WHERE elementId(n) IN $eids
WITH collect(n) AS ns
CALL {
    WITH ns
    UNWIND ns AS a
    MATCH (a)-[r]->(b) WHERE b IN ns
    RETURN collect(DISTINCT r) AS rels
}
RETURN
    [n IN ns | {
        id: coalesce(n.id, elementId(n)),
        name: coalesce(n.name, n.id, 'Unnamed'),
        label: head(labels(n)),
        description: coalesce(n.description, n.summary, '')
    }] AS nodes,
    [r IN rels | {
        type: type(r),
        source: coalesce(startNode(r).id, elementId(startNode(r))),
        target: coalesce(endNode(r).id,   elementId(endNode(r))),
        properties: properties(r)
    }] AS relationships
"""
 
 
@tool
def expand_subgraph(
    seed_node_ids: str,
    query: str = "",
    max_hops: int = 2,
    max_nodes: int = 25,
    max_edges: int = 40,
) -> str:
    """
    Expand the Neo4j graph from one or more seed node ids (1-2 hops).
 
    seed_node_ids: comma-separated node ids, exactly as returned by
    search_seed_nodes. Ids that do not exist in the graph are reported back
    explicitly rather than silently ignored.
 
    Returns an induced subgraph: nodes ranked by distance to the seeds and by
    how many preferred relationship types were traversed, plus only those
    relationships whose endpoints are both present in the returned node set.
    """
    driver = get_neo4j_driver()
    seed_ids = _normalize_seed_ids(seed_node_ids)
    if not seed_ids:
        return "No seed nodes provided."
 
    # clamp ANTES de interpolar en el Cypher
    hops = max(1, min(int(max_hops), 3))
    max_nodes = max(1, min(int(max_nodes), 60))
    max_edges = max(1, min(int(max_edges), 120))
 
    boosts = _detect_intent_boosts(query)
    preferred = boosts + [r for r in DEFAULT_PREFERRED_RELS if r not in boosts]
 
    try:
        with driver.session() as session:
            resolved = list(session.run(_RESOLVE_CYPHER, seed_ids=seed_ids))
 
            if not resolved:
                # Señal de alucinación: distinta de "existe pero no expandió"
                return (
                    "SEEDS_NOT_FOUND: ninguno de estos ids existe en el grafo: "
                    f"{', '.join(seed_ids)}.\n"
                    "Llama primero a search_seed_nodes y usa los IDs que devuelva "
                    "tal cual. No construyas ids."
                )
 
            seed_eids = [r["eid"] for r in resolved]
            found_ids = {r["sid"] for r in resolved} | {r["name"] for r in resolved}
            missing = [s for s in seed_ids if s not in found_ids]
 
            neighbors = session.run(
                _EXPAND_CYPHER.format(H=hops),
                seed_eids=seed_eids,
                structural_labels=STRUCTURAL_LABELS,
                generic_rels=GENERIC_RELS,
                preferred_rels=preferred,
                per_seed_limit=max_nodes * 3,
                max_nodes=max_nodes,
            ).single()
 
            neighbor_eids = (neighbors["neighbor_eids"] if neighbors else []) or []
 
            # Los seeds van primero y nunca se caen por el corte
            final_eids = seed_eids + [e for e in neighbor_eids if e not in seed_eids]
            final_eids = final_eids[:max_nodes]
 
            record = session.run(_FETCH_CYPHER, eids=final_eids).single()
    except Exception as e:
        raise ToolException(
            f"expand_subgraph falló: {type(e).__name__}: {e}"
        ) from e
 
    if not record or not record["nodes"]:
        return f"SEEDS_ISOLATED: los seeds existen pero no tienen expansión a {hops} hop(s)."
 
    nodes = record["nodes"]
    relationships = record["relationships"]
 
    # Ordenar aristas por prioridad antes de cortar
    rank = {t: i for i, t in enumerate(preferred)}
    relationships.sort(key=lambda r: rank.get(r["type"], len(rank)))
    relationships = relationships[:max_edges]
 
    header = f"=== Subgrafo desde seeds: {', '.join(r['sid'] for r in resolved)} ==="
    if missing:
        header += f"\n[AVISO] ids inexistentes ignorados: {', '.join(missing)}"
 
    parts: list[str] = [header, f"\nNodos ({len(nodes)}):"]
    for i, n in enumerate(nodes, 1):
        marker = " *seed*" if i <= len(seed_eids) else ""
        parts.append(
            f"  {i}. [{n['label']}] {n['name']} (id: {n['id']}){marker}\n"
            f"     {_truncate(n.get('description') or '', MAX_NODE_DESC)}"
        )
 
    parts.append(f"\nRelaciones ({len(relationships)}):")
    for r in relationships:
        props = {
            k: v for k, v in (r.get("properties") or {}).items()
            if k in ("weight", "confidence", "note", "severity")
        }
        suffix = f" | {props}" if props else ""
        parts.append(f"  ({r['source']}) -[{r['type']}]-> ({r['target']}){suffix}")
 
    return "\n".join(parts)