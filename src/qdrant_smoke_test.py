"""
Smoke test del vector store. Re-ejecutable sin re-ingerir.

Valida dos cosas:
  1. DISCRIMINACION: los scores varian entre queries. Si el spread es ~0,
     el retrieval no diferencia y el resto no importa.
  2. CALIBRACION: las queries jurisdiccionales (Colombia, Espana) deben dar
     scores BAJOS y cero chunks regulatorios. El corpus es MAHC/OSHA/EPA;
     no tiene normativa LATAM ni EU. Un score alto ahi seria un falso positivo,
     que es justo lo que llevo al agente a inventar "Ley 1209, Articulo 11".
"""

from __future__ import annotations

from qdrant_vector_store import cargar_vector_store

# (query, espera_hits_utiles)
QUERIES: list[tuple[str, bool]] = [
    # --- deberian dar scores altos: el manual cubre esto ---
    ("free chlorine target range", True),
    ("how does excess acid affect the gas heater", True),
    ("breakpoint chlorination chloramine", True),
    ("pool barrier fence height requirement", True),
    ("total alkalinity why it matters", True),
    # --- matcheo exacto: aca el BM25 hace la diferencia vs denso puro ---
    ("MAHC", True),
    ("Langelier Saturation Index LSI", True),
    # --- deberian dar scores bajos: fuera del corpus ---
    ("Colombia Ley 1209 de 2008 pool safety barrier fence regulations", False),
    ("Spain Real Decreto 742/2013 free chlorine community pool", False),
]

K = 4


def main(k: int = K) -> None:
    store = cargar_vector_store()
    print(f"✅ Store cargado (k={k})\n")

    resumen: list[tuple[str, float, float, int, bool]] = []

    for query, espera_utiles in QUERIES:
        etiqueta = "ESPERA HITS" if espera_utiles else "ESPERA NADA"
        print(f"\n{'=' * 78}\n[{etiqueta}] {query}")

        hits = store.similarity_search_with_score(query, k=k)
        if not hits:
            print("  (sin resultados)")
            resumen.append((query, 0.0, 0.0, 0, espera_utiles))
            continue

        n_reg = 0
        for i, (doc, score) in enumerate(hits, 1):
            m = doc.metadata or {}
            reg = " [REGULATORY]" if m.get("is_regulatory") else ""
            n_reg += bool(m.get("is_regulatory"))
            cuerpo = doc.page_content.replace("\n", " ")[:95].strip()
            print(
                f"  {i}. [{score:.4f}]{reg} Ch{m.get('chapter')} "
                f"{m.get('fragment_type')} | {m.get('hierarchy_path')}"
            )
            print(f"     {cuerpo}...")

        top = hits[0][1]
        spread = top - hits[-1][1]
        print(f"  → top={top:.4f}  spread={spread:.4f}  regulatorios={n_reg}/{len(hits)}")
        resumen.append((query, top, spread, n_reg, espera_utiles))

    # ---------------- veredicto ----------------
    print(f"\n\n{'=' * 78}\nRESUMEN\n{'=' * 78}")
    print(f"{'top':>8} {'spread':>8} {'reg':>4}  query")
    for q, top, spread, n_reg, _ in resumen:
        print(f"{top:8.4f} {spread:8.4f} {n_reg:4}  {q[:52]}")

    tops = [t for _, t, _, _, _ in resumen if t > 0]
    spreads = [s for _, _, s, _, _ in resumen if s > 0]

    print(f"\n{'=' * 78}")

    if not spreads or max(spreads) < 0.01:
        print("❌ FALLO: los scores no discriminan (spread ~0 en todas).")
        print("   El problema no era el naming del vector. Revisar el pipeline.")
        return

    rango_tops = max(tops) - min(tops)
    print(f"✅ Discriminacion OK — spread max {max(spreads):.4f}")
    print(f"   Rango de top-scores entre queries: {rango_tops:.4f}")

    dentro = [t for q, t, _, _, e in resumen if e]
    fuera = [t for q, t, _, _, e in resumen if not e]

    if dentro and fuera:
        print(f"\n   media top DENTRO del corpus:  {sum(dentro)/len(dentro):.4f}")
        print(f"   media top FUERA del corpus:   {sum(fuera)/len(fuera):.4f}")
        margen = sum(dentro) / len(dentro) - sum(fuera) / len(fuera)
        if margen > 0.05:
            umbral = (sum(dentro) / len(dentro) + sum(fuera) / len(fuera)) / 2
            print(f"   ✅ Separables (margen {margen:.4f})")
            print(f"   → umbral sugerido para insufficient_evidence: {umbral:.3f}")
        else:
            print(f"   ⚠️  Margen chico ({margen:.4f}): el score solo no alcanza")
            print("      para gatear. Usar tambien is_regulatory.")


if __name__ == "__main__":
    main()