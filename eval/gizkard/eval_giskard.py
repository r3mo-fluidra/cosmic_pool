import os
import sys
from datetime import datetime
from streamlit import secrets
from dotenv import load_dotenv
# ==========================================
# 0. CONFIGURAR LLM ANTES DE TODO
# ==========================================
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

import giskard
giskard.llm.set_llm_model("gemini-3.1-flash-lite")
giskard.llm.set_embedding_model("gemini/text-embedding-004")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import uuid
from langchain_core.messages import HumanMessage
from src.agent.graph import graph
from golden_set import DATA as golden_data

# ==========================================
# 1. ENVOLTORIO DEL MODELO PARA GISKARD
# ==========================================
def run_agent(df: pd.DataFrame) -> list:
    responses = []
    for prompt in df["question"]:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            responses.append(result["messages"][-1].content)
        except Exception as e:
            responses.append(f"ERROR: {str(e)}")
    return responses


giskard_model = giskard.Model(
    model=run_agent,
    model_type="text_generation",
    name="Pool Assistant Orchestrator",
    description="Agente de IA especializado en química y mantenimiento de piscinas.",
    feature_names=["question"]
)

# ==========================================
# 2. DATASET
# ==========================================
data = golden_data
df = pd.DataFrame(data)

giskard_dataset = giskard.Dataset(
    df=df,
    target="reference_answer",
    name="Pool Assistant Regression Dataset"
)


# ==========================================
# 3. GENERADOR DE REPORTE MARKDOWN
# ==========================================
def generate_markdown_report(
    scan_results,
    suite_results,
    agent_responses: list,
    output_path: str = "reporte_eval.md"
):
    """Genera un reporte de evaluación completo en formato Markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    questions = data["question"]
    references = data["reference_answer"]

    # --- Recolectar issues del scan ---
    issues = scan_results.issues if hasattr(scan_results, "issues") else []
    total_issues = len(issues)

    # --- Agrupar issues por detector ---
    issues_by_detector: dict = {}
    for issue in issues:
        detector = getattr(issue, "detector", "Unknown")
        detector_name = getattr(detector, "__class__", type(detector)).__name__
        issues_by_detector.setdefault(detector_name, []).append(issue)

    # --- Estado de la suite ---
    suite_passed = getattr(suite_results, "passed", False)
    suite_status = "✅ PASSED" if suite_passed else "❌ FAILED"

    lines = []

    # ── Encabezado ──────────────────────────────────────────
    lines += [
        "# 🏊 Pool Assistant — Reporte de Evaluación",
        "",
        f"**Fecha:** {now}  ",
        f"**Modelo evaluado:** Pool Assistant Orchestrator  ",
        f"**LLM evaluador:** gemini/gemini-2.0-flash  ",
        f"**Dataset:** {len(questions)} preguntas de prueba  ",
        "",
        "---",
        "",
    ]

    # ── Resumen ejecutivo ────────────────────────────────────
    lines += [
        "## 📋 Resumen Ejecutivo",
        "",
        f"| Métrica                        | Resultado        |",
        f"|-------------------------------|------------------|",
        f"| Vulnerabilidades detectadas    | {total_issues}   |",
        f"| Detectores ejecutados          | 9                |",
        f"| Pruebas de correctitud (RAG)   | {suite_status}   |",
        f"| Preguntas evaluadas            | {len(questions)} |",
        "",
        "---",
        "",
    ]

    # ── Respuestas del agente vs referencia ─────────────────
    lines += [
        "## 🤖 Respuestas del Agente vs Referencia",
        "",
    ]
    for i, (q, ref, resp) in enumerate(zip(questions, references, agent_responses), 1):
        category = _classify_question(q)
        lines += [
            f"### Pregunta {i} — {category}",
            "",
            f"**Pregunta:** {q}",
            "",
            f"**Referencia esperada:**",
            f"> {ref}",
            "",
            f"**Respuesta del agente:**",
            f"> {resp}",
            "",
        ]

    lines += ["---", ""]

    # ── Vulnerabilidades del scan ────────────────────────────
    lines += [
        "## 🔍 Vulnerabilidades Detectadas por el Scan",
        "",
    ]

    if total_issues == 0:
        lines += [
            "✅ No se detectaron vulnerabilidades en esta ejecución.",
            "",
        ]
    else:
        for detector_name, det_issues in issues_by_detector.items():
            lines += [
                f"### {detector_name} ({len(det_issues)} issue{'s' if len(det_issues) > 1 else ''})",
                "",
            ]
            for issue in det_issues:
                description = getattr(issue, "description", str(issue))
                level = getattr(issue, "level", "warning")
                level_icon = "🔴" if level == "major" else "🟡"
                lines += [
                    f"- {level_icon} **{level.upper()}:** {description}",
                ]
            lines += [""]

    lines += ["---", ""]

# ── Resultado de correctitud ─────────────────────────────
    lines += [
        "## ⚖️ Evaluación de Correctitud RAG",
        "",
        f"**Estado:** {suite_status}",
        f"**Umbral requerido:** 80%",
        "",
        "| Test | Estado | Métrica |",
        "|------|--------|---------|",
    ]

    # Giskard puede exponer los resultados de la suite
    # mediante diferentes atributos dependiendo de la versión.
    suite_tests = None

    for attr in ("results", "tests", "results_list"):
        value = getattr(suite_results, attr, None)
        if value is not None:
            suite_tests = value
            break

    if suite_tests is not None:

        if isinstance(suite_tests, dict):
            iterator = suite_tests.items()

        elif isinstance(suite_tests, (list, tuple)):
            iterator = []

            for idx, result in enumerate(suite_tests, 1):
                test_name = getattr(
                    result,
                    "name",
                    f"Test {idx}"
                )
                iterator.append((test_name, result))

        else:
            iterator = [
                (
                    getattr(suite_tests, "name", "RAG Correctness"),
                    suite_tests
                )
            ]

        for test_name, test_result in iterator:

            metric = getattr(test_result, "metric", None)
            passed = getattr(test_result, "passed", False)

            icon = "✅" if passed else "❌"

            if isinstance(metric, (float, int)):
                metric_str = f"{metric:.2%}"
            elif metric is None:
                metric_str = "N/A"
            else:
                metric_str = str(metric)

            lines.append(
                f"| {test_name} | {icon} | {metric_str} |"
            )

    else:
        # Fallback: la suite no expone los tests individualmente.
        lines.append(
            f"| RAG Correctness Suite | "
            f"{'✅' if suite_passed else '❌'} | N/A |"
        )

    lines.append("")


def _classify_question(question: str) -> str:
    """Clasifica la pregunta para el reporte."""
    q = question.lower()
    if "inject" in q or "ignore" in q or "poem" in q:
        return "🛡️ Prueba Adversaria"
    if "mix" in q or "danger" in q or "acid" in q:
        return "⚠️ Seguridad Química"
    if "winterize" in q or "step" in q:
        return "📋 Procedimiento"
    return "🧪 Química / Mantenimiento"


def _generate_recommendations(issues: list, suite_passed: bool) -> list:
    """Genera recomendaciones basadas en los resultados."""
    recs = []
    detector_names = [
        getattr(getattr(i, "detector", type(i)), "__class__", type(i)).__name__
        for i in issues
    ]

    if any("Injection" in d for d in detector_names):
        recs.append(
            "Reforzar guardrails contra Prompt Injection: validar que el agente "
            "rechace instrucciones fuera del dominio de piscinas."
        )
    if any("Hallucination" in d or "Faithfulness" in d for d in detector_names):
        recs.append(
            "Revisar el pipeline RAG: las alucinaciones sugieren que el contexto "
            "recuperado no está siendo utilizado correctamente."
        )
    if any("Harmful" in d for d in detector_names):
        recs.append(
            "Agregar filtros de contenido dañino en el nodo de salida del grafo."
        )
    if not suite_passed:
        recs.append(
            "Las respuestas del agente no alcanzan el 80% de correctitud. "
            "Revisar los prompts del sistema y la base de conocimiento RAG."
        )
    if not recs:
        recs.append(
            "El agente pasó todas las pruebas. Mantener el dataset de evaluación "
            "actualizado con nuevos casos de borde."
        )
    return recs


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    print("🚀 Iniciando Escaneo de Vulnerabilidades...")
    scan_results = giskard.scan(
        giskard_model,
        giskard_dataset,
        raise_exceptions=False
    )

    # Obtener respuestas del agente para incluirlas en el reporte
    print("\n🤖 Recolectando respuestas del agente...")
    agent_responses = run_agent(df[["question"]])

    print("\n⚖️ Evaluando Correctitud RAG...")
    from giskard.testing.tests.llm import test_llm_correctness

    test_suite = giskard.Suite(name="RAG Correctness Suite")
    test_suite.add_test(
        test_llm_correctness(
            model=giskard_model,
            dataset=giskard_dataset,
            threshold=0.8
        )
    )
    suite_results = test_suite.run()
    print("\n========== GISKARD SUITE RESULT ==========")
    print("TYPE:", type(suite_results))
    print("DIR:", [x for x in dir(suite_results) if not x.startswith("_")])
    print("DICT:", getattr(suite_results, "__dict__", None))
    print("PASSED:", getattr(suite_results, "passed", None))
    print("==========================================\n")

    print("\n📄 Generando reporte Markdown...")
    report = generate_markdown_report(
        scan_results=scan_results,
        suite_results=suite_results,
        agent_responses=agent_responses,
        output_path="reporte_eval.md"
    )
    print("✅ Reporte generado: 'reporte_eval.md'")
    print("\n" + "=" * 60)
    print(report[:800] + "\n[...] Ver reporte_eval.md para el contenido completo.")

    suite_passed = getattr(suite_results, "passed", False)

    exit(0 if suite_passed else 1)