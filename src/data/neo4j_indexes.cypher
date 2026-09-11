// ===========================================================================
// Índices de Neo4j para el retrieval del Pool Assistant
// ===========================================================================
//
// QUÉ ARREGLA
// -----------
// Hoy no existe ni un índice en la base (grepeado: no hay un solo CREATE INDEX
// en el repo). Las dos consultas que más se ejecutan hacen MATCH sin label:
//
//   src/agent/tools.py:754  _NEIGHBOR_CYPHER
//     MATCH (s) WHERE s.id IN $seed_ids
//
//   src/agent/tools.py:983  _RESOLVE_CYPHER
//     MATCH (s) WHERE s.id IN $seed_ids OR elementId(s) IN $seed_ids
//                  OR s.name IN $seed_ids
//
// Los índices de Neo4j están SCOPEADOS POR LABEL. Un patrón sin label no puede
// usar ninguno: es un scan de todos los nodos. Y se paga dos veces por turno
// (una en search_seed_nodes, otra en expand_subgraph). Con 10k nodos son
// ~10-40 ms; con 500k, 0.5-2 s por llamada.
//
// CÓMO USARLO
// -----------
//   cat src/data/neo4j_indexes.cypher | cypher-shell -a "$NEO4J_URI" \
//       -u "$NEO4J_USER" -p "$NEO4J_PASSWORD"
//
// o pegando los bloques en el navegador de Neo4j / la consola de Aura.
//
// Es idempotente: todo usa IF NOT EXISTS o MERGE. Correrlo dos veces no hace
// daño. El paso 2 puede tardar en una base grande (escribe un label en cada
// nodo); el resto es instantáneo.
//
// ORDEN DE APLICACIÓN — importa
// -----------------------------
// Los pasos 1 y 3 son seguros y dan ganancia HOY, sin tocar código.
// El paso 2 prepara el terreno para un cambio de código que todavía NO está
// hecho (ver la nota al final). Aplicarlo no rompe nada por sí solo.
// ===========================================================================


// ---------------------------------------------------------------------------
// PASO 1 — Índice fulltext `node_search`   [SEGURO, GANANCIA INMEDIATA]
// ---------------------------------------------------------------------------
// src/agent/tools.py:315 `_fulltext_available()` comprueba si este índice
// existe y está ONLINE. Si no, `search_seed_nodes` degrada a `_SCAN_HEAD`
// (tools.py:631), que es literalmente `MATCH (n)` sobre toda la base.
//
// Peor: ese chequeo está bajo @lru_cache y cachea también el False del
// except. Un blip de red en la primera consulta del proceso deja el retrieval
// en modo scan hasta que se reinicie.
//
// Los campos son los que puntúa _SCORING_BODY: name, description, summary,
// keywords, aliases.
CREATE FULLTEXT INDEX node_search IF NOT EXISTS
FOR (n:Chemical|Procedure|Hazard|Problem|Equipment|WaterParameter|
       Requirement|Threshold|Standard|Code|Operation|Task|Role|
       Symptom|Cause|Risk|Concept|Venue)
ON EACH [n.name, n.description, n.summary, n.keywords, n.aliases];


// ---------------------------------------------------------------------------
// PASO 2 — Label común `:Entity`   [PREPARATORIO — ver nota final]
// ---------------------------------------------------------------------------
// Da a todo nodo de dominio un label compartido, para que las consultas
// puedan anclarse a un índice. Excluye los STRUCTURAL_LABELS
// (tools.py:162), que el retrieval ya descarta: Chapter, Section, Document,
// Source, Appendix, Table.
//
// En CALL {} con IN TRANSACTIONS para no construir una transacción única
// gigante en bases grandes. Si tu versión de Neo4j es < 5.0, quita el
// `IN TRANSACTIONS OF 10000 ROWS` y córrelo entero.
MATCH (n)
WHERE NOT any(l IN labels(n)
              WHERE l IN ['Chapter','Section','Document','Source','Appendix','Table'])
  AND (n.id IS NOT NULL OR n.name IS NOT NULL)
  AND NOT n:Entity
CALL (n) {
    SET n:Entity
} IN TRANSACTIONS OF 10000 ROWS;


// ---------------------------------------------------------------------------
// PASO 3 — Índices de lookup   [SEGURO]
// ---------------------------------------------------------------------------
// `id` es la clave que usa todo el grafo. La constraint de unicidad crea su
// propio índice y además protege contra ingestas duplicadas — que es lo que
// produce los seeds fantasma que luego el LLM cita dos veces.
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

// `name` no es único (dos nodos pueden llamarse "pH" en contextos distintos),
// así que índice a secas. Lo usa la tercera rama del OR de _RESOLVE_CYPHER.
CREATE INDEX entity_name IF NOT EXISTS
FOR (n:Entity) ON (n.name);


// ---------------------------------------------------------------------------
// VERIFICACIÓN
// ---------------------------------------------------------------------------
// Los índices se pueblan en segundo plano. Esperá a que estén ONLINE:
SHOW INDEXES YIELD name, type, state, populationPercent
WHERE name IN ['node_search', 'entity_id_unique', 'entity_name']
RETURN name, type, state, populationPercent;

// Cuántos nodos quedaron sin el label (deberían ser solo estructurales):
MATCH (n) WHERE NOT n:Entity
RETURN labels(n) AS labels, count(*) AS n ORDER BY n DESC;


// ===========================================================================
// NOTA: el paso 2 por sí solo NO acelera nada
// ===========================================================================
// El label y sus índices no se usan hasta que las consultas lleven el label.
// El cambio que falta, en src/agent/tools.py:
//
//   _NEIGHBOR_CYPHER:  MATCH (s)  ->  MATCH (s:Entity)
//
//   _RESOLVE_CYPHER: partir el OR de tres propiedades en tres ramas UNION.
//   Un OR sobre propiedades distintas impide usar índice AUNQUE haya label:
//   el planner de Neo4j no puede combinar índices a través de un OR.
//
//       MATCH (s:Entity) WHERE s.id IN $seed_ids       RETURN s
//       UNION
//       MATCH (s:Entity) WHERE s.name IN $seed_ids     RETURN s
//       UNION
//       MATCH (s) WHERE elementId(s) IN $seed_ids      RETURN s
//
// Ese cambio NO está aplicado, a propósito: no tengo acceso a la base para
// comprobar que `:Entity` quedó en todos los nodos que el retrieval espera.
// Si se aplicara antes de correr el paso 2, TODAS las consultas devolverían
// cero resultados y el asistente dejaría de encontrar nada — un fallo mucho
// peor que la lentitud que arregla.
//
// Secuencia correcta:
//   1. Correr este script.
//   2. Comprobar con la query de verificación que solo quedan estructurales
//      sin :Entity.
//   3. Medir el antes/después con PROFILE sobre _NEIGHBOR_CYPHER.
//   4. Recién entonces aplicar el cambio de los Cypher en tools.py.
// ===========================================================================
