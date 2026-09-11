// ===========================================================================
// Relación proporcional cloro libre / ácido cianúrico — PRÁCTICA DE INDUSTRIA
// ===========================================================================
//
// POR QUÉ ESTE NODO
// -----------------
// Cuatro evaluaciones expertas seguidas señalaron el mismo hueco: el grafo
// codifica el escalón regulatorio (un mínimo mayor cuando hay isocianuratos)
// pero no la proporción, así que el agente no puede decir a qué cloro apuntar
// con un estabilizador concreto. Sin eso, `constraint_conflict` venía null en
// 7 de 7 parámetros y la guarda que impide entregar un objetivo insuficiente
// nunca podía dispararse.
//
// Consecuencia medida: con CYA 90 el sistema dio "sube por encima de 2.0 ppm".
// Un operador que lo cumple reabre con el agua efectivamente sin desinfectar.
//
// QUÉ NO ES
// ---------
// NO es normativo, y el nodo lo dice de todas las formas posibles: label
// Practice, evidence 'industry_practice', source_reference sin prefijo de
// capítulo, y una relación explícita al nodo del corpus que advierte de sus
// límites (fc_to_cya_ratio_heuristic, CH13-13.4).
//
// El corpus se niega deliberadamente a dar un porcentaje — "circulan varios",
// "expresamente no es un estándar regulatorio" — y esa postura se respeta:
// este nodo no la sustituye, la complementa, y arrastra su advertencia.
//
// LA REGLA QUE IMPORTA
// --------------------
// Cuando la proporción implica un cloro por encima del rango publicado, la
// conclusión NO es dosificar por encima del máximo: es bajar el estabilizador.
// Eso lo prescribe el corpus y va escrito en la descripción, porque es lo que
// convierte "diluye" en una conclusión en vez de una tarea.
//
// USO
// ---
//   cat src/data/kb_fc_cya_ratio.cypher | cypher-shell -a "$NEO4J_URI" \
//       -u "$NEO4J_USER" -p "$NEO4J_PASSWORD"
//
// Idempotente (MERGE por id). Reejecutable sin efectos.
// ===========================================================================

MERGE (r:Practice:DecisionRule {id: 'fc_cya_proportional_target'})
SET r.name           = 'Free Chlorine Target Proportional to Cyanuric Acid',
    r.canonical_name = 'Free Chlorine Target Proportional to Cyanuric Acid',
    r.description    =
        'INDUSTRY PRACTICE, NOT A CODE REQUIREMENT. Where cyanuric acid is '
      + 'present, the free chlorine concentration needed for equivalent '
      + 'sanitizing activity scales with the stabilizer level rather than '
      + 'being fixed. The most commonly cited figure is about 7.5 percent of '
      + 'the measured cyanuric acid, with published values ranging roughly '
      + 'from 5 to 10 percent depending on the source and the intended '
      + 'disinfection endpoint. Target free chlorine (ppm) = measured cyanuric '
      + 'acid (ppm) x 0.075. '
      + 'This figure does not appear in model code and must never be reported '
      + 'as a regulatory requirement, cited to a code section, or used to '
      + 'justify operating outside a published range. '
      + 'CRITICAL RESOLUTION: where the proportional calculation implies a '
      + 'free chlorine level above the published operating range, the correct '
      + 'response is to REDUCE THE CYANURIC ACID, not to exceed the range. '
      + 'The finding in that case is that adequate sanitation is unreachable '
      + 'at the current stabilizer level, and the corrective action belongs to '
      + 'the stabilizer. Reporting the in-range figure on its own is unsafe: '
      + 'it reads as sufficient and it is not.',
    r.keywords       = ['fc cya ratio', 'chlorine target stabilizer',
                        'proportional chlorine target', 'cya compensation',
                        'active chlorine equivalence', '7.5 percent',
                        'industry practice not code'],
    r.aliases        = ['FC/CYA ratio', 'chlorine to stabilizer ratio',
                        'proportional FC target'],
    r.evidence       = 'industry_practice',
    r.source_reference = 'INDUSTRY-PRACTICE-fc-cya-proportional-target',
    r.chapter        = '13',
    r.status         = 'new';

// Enlace a la advertencia del corpus. Esta relación es el motivo de que el
// nodo se pueda usar sin falsear la fuente: quien recupere uno alcanza el
// otro en un salto, y expand_subgraph los devuelve juntos.
MATCH (r:DecisionRule {id: 'fc_cya_proportional_target'})
MATCH (h {id: 'fc_to_cya_ratio_heuristic'})
MERGE (r)-[:QUALIFIED_BY]->(h);

// El escalón regulatorio, que sigue siendo el que manda para el status.
MATCH (r:DecisionRule {id: 'fc_cya_proportional_target'})
MATCH (m {id: 'higher_fc_minimum_with_cyanurates'})
MERGE (r)-[:DOES_NOT_REPLACE]->(m);

// Los dos parámetros que la regla relaciona.
MATCH (r:DecisionRule {id: 'fc_cya_proportional_target'})
MATCH (fc {id: 'free_chlorine'})
MATCH (cya {id: 'cyanuric_acid'})
MERGE (r)-[:DETERMINES]->(fc)
MERGE (r)-[:DEPENDS_ON]->(cya);


// ---------------------------------------------------------------------------
// VERIFICACIÓN
// ---------------------------------------------------------------------------
MATCH (r:DecisionRule {id: 'fc_cya_proportional_target'})
OPTIONAL MATCH (r)-[rel]->(m)
RETURN r.name AS regla, r.evidence AS evidencia,
       collect(type(rel) + ' -> ' + m.id) AS enlaces;
