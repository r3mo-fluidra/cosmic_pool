"""
Vector store para el pool manual (Qdrant local + LangChain).

Modelo denso : BAAI/bge-small-en-v1.5 (384 dims, local, sin cuota API)
Modelo sparse: Qdrant/bm25 (matcheo exacto de terminos tipo "Ley 1209")
Modo         : HYBRID (denso + sparse via RRF)

El corpus del manual esta en ingles y las queries que emite el agente
tambien, por eso bge-small-en alcanza y no hace falta un modelo multilingue.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import uuid

import pandas as pd
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

# ========================= CONFIG =========================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "src" / "data"

QDRANT_PATH = DATA_DIR / "qdrant_pool_db"
DEFAULT_CSV_PATH = (
    DATA_DIR / "documents" / "semantic_search" / "pool_manual_chunks_clean.csv"
)

COLLECTION_NAME = "pool_manual_vectors"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Excluir encabezados puros (is_stub). Ponelo en False si el smoke test
# muestra que no contaminan el top-k.
EXCLUDE_STUBS = True

# Columnas que la ingestion necesita. Si falta alguna, falla ruidosamente
# en vez de embeber "None" en silencio (el bug del esquema v1).
REQUIRED_COLUMNS = {
    "chunk_id",
    "content",
    "title",
    "hierarchy_path",
    "chapter",
    "level",
    "fragment_type",
    "is_regulatory",
    "is_stub",
    "token_count",
}

# Campos con indice de payload, para poder filtrar server-side
INDEXED_PAYLOAD_FIELDS = {
    "metadata.is_regulatory": PayloadSchemaType.BOOL,
    "metadata.chapter": PayloadSchemaType.INTEGER,
    "metadata.fragment_type": PayloadSchemaType.KEYWORD,
    "metadata.chunk_id": PayloadSchemaType.KEYWORD,
}

POINT_ID_NAMESPACE = uuid.UUID("6f1c0e5a-3b7d-4a2e-9c88-1f5d2b7a4e10")


def _point_id(chunk_id: str) -> str:
    """UUIDv5 estable a partir del chunk_id.

    Qdrant exige UUID o entero sin signo como point ID. UUIDv5 es un hash
    determinístico, así que re-ingerir sobrescribe en lugar de duplicar.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))

class VectorStoreConfigError(RuntimeError):
    """Fallo de configuracion/infra, NO un fallo de busqueda.

    El agente no puede arreglar esto reformulando la query.
    """


# ========================= EMBEDDINGS =========================
def _dense_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=DENSE_MODEL)


def _sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=SPARSE_MODEL)


def _build_store(client: QdrantClient) -> QdrantVectorStore:
    """Constructor UNICO del store. Nada mas debe instanciar QdrantVectorStore."""
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=_dense_embeddings(),
        sparse_embedding=_sparse_embeddings(),
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )


# ========================= INGESTION =========================
def _validar_esquema(df: pd.DataFrame) -> None:
    faltantes = REQUIRED_COLUMNS - set(df.columns)
    if faltantes:
        raise VectorStoreConfigError(
            f"Columnas faltantes en el CSV: {sorted(faltantes)}\n"
            f"Presentes: {sorted(df.columns)}\n"
            "Usa pool_manual_chunks_clean.csv, no el original."
        )


def _crear_coleccion(client: QdrantClient, force_recreate: bool) -> None:
    existe = client.collection_exists(COLLECTION_NAME)

    if existe and not force_recreate:
        print(f"ℹ️  Coleccion '{COLLECTION_NAME}' ya existe.")
        return

    if existe:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Coleccion '{COLLECTION_NAME}' eliminada.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(
        f"✅ Coleccion creada: dense='{DENSE_VECTOR_NAME}' ({DENSE_DIM}d, cosine), "
        f"sparse='{SPARSE_VECTOR_NAME}' (bm25)"
    )

    for field, schema in INDEXED_PAYLOAD_FIELDS.items():
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
    print(f"✅ Indices de payload: {list(INDEXED_PAYLOAD_FIELDS)}")


def _fila_a_documento(row) -> Document:
    """Construye el texto embebido y la metadata.

    El texto NO incluye `keywords`: la extraccion del chunker no filtra
    stopwords ('this', 'how', 'use'), asi que meterlas degradaria el BM25.
    Quedan solo en metadata.
    """
    content = str(row.content).strip()
    hierarchy = str(row.hierarchy_path).strip()
    title = str(row.title).strip()

    # hierarchy_path da contexto jerarquico ("Ch9/9.4 Breakpoint Chlorination")
    # que el content solo no tiene.
    encabezado = hierarchy or title
    page_content = f"Section: {encabezado}\n\n{content}" if encabezado else content

    try:
        keywords = json.loads(row.keywords) if pd.notna(row.keywords) else []
    except (json.JSONDecodeError, TypeError):
        keywords = []

    return Document(
        page_content=page_content,
        metadata={
            "chunk_id": str(row.chunk_id),
            "chapter": int(row.chapter),
            "level": int(row.level),
            "fragment_type": str(row.fragment_type),
            "hierarchy_path": hierarchy,
            "title": title,
            "parent_chunk_id": None
            if pd.isna(row.parent_chunk_id)
            else str(row.parent_chunk_id),
            "is_regulatory": bool(row.is_regulatory),
            "is_stub": bool(row.is_stub),
            "token_count": int(row.token_count),
            "keywords": keywords,
        },
    )


def inicializar_vector_store(
    path_csv: str | Path | None = None,
    force_recreate: bool = True,
    exclude_stubs: bool = EXCLUDE_STUBS,
) -> QdrantVectorStore:
    archivo_csv = Path(path_csv) if path_csv else DEFAULT_CSV_PATH
    if not archivo_csv.exists():
        raise FileNotFoundError(f"CSV no encontrado: {archivo_csv.resolve()}")
    
    df = pd.read_csv(archivo_csv)
    documents = [_fila_a_documento(r) for r in df.itertuples()]
    ids = [_point_id(d.metadata["chunk_id"]) for d in documents]
    _validar_esquema(df)

    if df.empty:
        raise VectorStoreConfigError(f"CSV vacio: {archivo_csv}")
    if not df.chunk_id.is_unique:
        n = int(df.chunk_id.duplicated().sum())
        raise VectorStoreConfigError(
            f"{n} chunk_id duplicados. Corre clean_chunks.py primero."
        )

    print(f"📊 {len(df)} filas en {archivo_csv.name}")

    if exclude_stubs:
        n_stub = int(df.is_stub.sum())
        df = df[~df.is_stub].copy()
        print(f"🔻 {n_stub} stubs excluidos -> {len(df)} indexables")

    
    

    n_reg = sum(d.metadata["is_regulatory"] for d in documents)
    print(f"📄 {len(documents)} documentos ({n_reg} regulatorios)")

    # El lock de archivo de Qdrant es exclusivo: cerra Streamlit antes.
    if force_recreate and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
        print("🗑️  Directorio anterior eliminado.")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))
    _crear_coleccion(client, force_recreate)

    store = _build_store(client)

    print(f"📥 Indexando (dense + bm25)...")
    store.add_documents(documents, ids=ids)

    n_final = client.count(COLLECTION_NAME).count
    print(f"✅ {n_final} puntos en {QDRANT_PATH}")
    if n_final != len(documents):
        print(f"⚠️  Esperados {len(documents)}, escritos {n_final}")

    return store


# ========================= CARGA (produccion) =========================
def cargar_vector_store() -> QdrantVectorStore:
    """Carga sin recrear. Lo que debe usar el tool vector_search."""
    if not QDRANT_PATH.exists():
        raise VectorStoreConfigError(
            f"No existe la BD en {QDRANT_PATH}. "
            "Corre `python -m src.qdrant_vector_store` primero."
        )

    client = QdrantClient(path=str(QDRANT_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        raise VectorStoreConfigError(
            f"Coleccion '{COLLECTION_NAME}' no existe en {QDRANT_PATH}."
        )

    # Validar que la coleccion en disco coincide con lo que el codigo espera.
    # Sin esto, el desalineamiento se manifiesta como un error opaco en runtime.
    cfg = client.get_collection(COLLECTION_NAME).config.params
    vectors = cfg.vectors

    if vectors is None or hasattr(vectors, "size"):
        raise VectorStoreConfigError(
            f"La coleccion tiene un vector denso SIN NOMBRE; se esperaba "
            f"'{DENSE_VECTOR_NAME}'. Fue creada por otro pipeline: re-ingesta."
        )
    if DENSE_VECTOR_NAME not in vectors:
        raise VectorStoreConfigError(
            f"Vector '{DENSE_VECTOR_NAME}' ausente. Presentes: {list(vectors)}"
        )
    if vectors[DENSE_VECTOR_NAME].size != DENSE_DIM:
        raise VectorStoreConfigError(
            f"Dimension {vectors[DENSE_VECTOR_NAME].size} != {DENSE_DIM} "
            f"esperada por {DENSE_MODEL}. Re-ingesta."
        )
    if SPARSE_VECTOR_NAME not in (cfg.sparse_vectors or {}):
        raise VectorStoreConfigError(
            f"Vector sparse '{SPARSE_VECTOR_NAME}' ausente; HYBRID no puede "
            "funcionar. Re-ingesta."
        )

    return _build_store(client)


if __name__ == "__main__":
    inicializar_vector_store(force_recreate=True)
    print("\nAhora corre: python -m src.qdrant_smoke_test")