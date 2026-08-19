# src/qdrant_vector_store.py
import os
import shutil
from pathlib import Path
import pandas as pd
from streamlit import secrets
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)
from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIG =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or secrets.get("GEMINI_API_KEY")  #

DEFAULT_CSV_PATH = Path(r".data\documentos\semantic_search\pool_manual_chunks.csv")
COLLECTION_NAME  = "pool_manual_vectors"

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
QDRANT_PATH  = PROJECT_ROOT / "src" / "data" / "qdrant_pool_db"

# Nombres de vectores: los definimos nosotros para que coincidan
# entre la creación de la colección y el constructor de QdrantVectorStore.
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Dimensión de gemini-embedding-001 (valor fijo del modelo)
GEMINI_EMBEDDING_DIM = 3072


def _crear_coleccion(client: QdrantClient, force_recreate: bool) -> None:
    """Crea (o recrea) la colección con soporte para búsqueda híbrida."""
    existe = client.collection_exists(COLLECTION_NAME)

    if existe and not force_recreate:
        print(f"ℹ️  Colección '{COLLECTION_NAME}' ya existe. Usando la existente.")
        return

    if existe and force_recreate:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Colección '{COLLECTION_NAME}' eliminada.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=GEMINI_EMBEDDING_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"✅ Colección '{COLLECTION_NAME}' creada.")


def inicializar_vector_store(path_csv=None, force_recreate: bool = True):
    if not GEMINI_API_KEY:
        raise ValueError("❌ GEMINI_API_KEY no encontrado.")

    archivo_csv = Path(path_csv) if path_csv else DEFAULT_CSV_PATH
    if not archivo_csv.exists():
        raise FileNotFoundError(f"CSV no encontrado: {archivo_csv.resolve()}")

    # Eliminar directorio anterior si se requiere
    if force_recreate and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
        print("🗑️  Directorio anterior eliminado.")

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(archivo_csv)
    print(f"📊 Cargando {len(df)} chunks...")

    # Embeddings densos (Gemini)
    dense_embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    # Embeddings dispersos (BM25)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Documentos
    documents = []
    for _, row in df.iterrows():
        texto = (
            f"Section: {row.get('section_title', '')}\n"
            f"Category: {row.get('category', '')}\n\n"
            f"{row.get('chunk_content', '')}"
        )
        doc = Document(
            page_content=texto,
            metadata={
                "chunk_id": str(row.get("chunk_id")),
                "source":   str(row.get("document_name")),
                "category": str(row.get("category")),
                "tags":     str(row.get("metadata_tags", "")),
            },
        )
        documents.append(doc)

    print("🚀 Conectando con Qdrant local...")

    # ✅ QdrantClient(path=...) es la única forma confiable en Windows.
    #    Nunca usar `location=` con rutas absolutas: Qdrant las parsea
    #    como URL y falla con "Unknown scheme" o errores IDNA.
    client = QdrantClient(path=str(QDRANT_PATH))

    # Crear la colección con los vectores correctos ANTES de add_documents
    _crear_coleccion(client, force_recreate)

    # Constructor con nombres de vectores explícitos (deben coincidir
    # con los usados en create_collection arriba)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    print(f"📥 Indexando {len(documents)} documentos...")
    vector_store.add_documents(documents)

    print(f"✅ ¡Éxito! Qdrant en:\n   {QDRANT_PATH}")
    print(f"   Documentos indexados: {len(documents)}")
    return vector_store


def cargar_vector_store() -> QdrantVectorStore:
    """Carga una BD existente sin recrearla (para uso en producción)."""
    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"No existe la BD en {QDRANT_PATH}. "
            "Ejecuta inicializar_vector_store() primero."
        )

    dense_embeddings  = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    client            = QdrantClient(path=str(QDRANT_PATH))

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )


if __name__ == "__main__":
    try:
        store = inicializar_vector_store(force_recreate=True)

        print("\n" + "=" * 70)
        resultados = store.similarity_search(
            "¿Cómo afecta el exceso de ácido al calentador de gas?",
            k=5,
        )

        for i, doc in enumerate(resultados):
            print(f"\n--- Resultado {i+1} ---")
            print(f"Fuente:    {doc.metadata.get('source')}")
            print(f"Categoría: {doc.metadata.get('category')}")
            print("-" * 50)
            contenido = doc.page_content
            print(contenido[:700] + "..." if len(contenido) > 700 else contenido)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise