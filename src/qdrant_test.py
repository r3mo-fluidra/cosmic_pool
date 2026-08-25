# scripts/diag_qdrant.py
from pathlib import Path
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
QDRANT_PATH = PROJECT_ROOT / "src" / "data" / "qdrant_pool_db"
COLLECTION = "pool_manual_vectors"

print(f"path: {QDRANT_PATH}  exists={QDRANT_PATH.exists()}")

client = QdrantClient(path=str(QDRANT_PATH))
print("colecciones:", [c.name for c in client.get_collections().collections])

info = client.get_collection(COLLECTION)
cfg = info.config.params

print("\n--- DENSE ---")
if cfg.vectors is None:
    print("  (ninguno)")
elif hasattr(cfg.vectors, "size"):
    print(f"  SIN NOMBRE  size={cfg.vectors.size}  dist={cfg.vectors.distance}")
else:
    for name, p in cfg.vectors.items():
        print(f"  nombre={name!r}  size={p.size}  dist={p.distance}")

print("--- SPARSE ---")
print(f"  {list((cfg.sparse_vectors or {}).keys()) or '(ninguno)'}")

print(f"\npuntos: {client.count(COLLECTION).count}")