"""
Una banda educativa no es un límite de código.

El paso 7 salió de una comprobación contra el grafo real: CERO nodos con
cotas numéricas estructuradas, y cada cifra en prosa envuelta en su propia
negativa a ser normativa ("typical educational targets", "your local code
controls"). El corpus no publica límites.

Lo que encaja pieza por pieza es el trace 6660e14f: pH con regulatory_limit
7.8, cianúrico con 90, cloro con 2. Los tres son extremos de la banda
educativa publicada. El especialista no inventaba las cifras — les cambiaba
la etiqueta. Y la que sí era inventada (un techo de dureza de calcio que
ninguna fuente respalda) pasaba con autoridad completa.

La tabla canónica no arregla esto trayendo límites: los quita.
"""

import pytest

from src.graph_context.response_contracts import (
    DetailSection, ReadingLine, SynthesizerOutput, get_contract)
from src.graph_context.response_validator import (
    ValidationReport,
    build_readings,
    coherent_status,
    enforce_contract,
    reading_note,
    reconcile_with_published_bands,
)
from src.graph_context.water_targets import (
    WATER_TARGETS, is_published_figure, target_band)


def _lectura(parametro, medido, status, limite=None, target=None):
    return {"parameter": parametro, "measured": medido, "status": status,
            "regulatory_limit": limite, "operating_target": target}


def _nota(lectura, lang="en"):
    rec = reconcile_with_published_bands([lectura])
    return build_readings(rec, lang, ReadingLine)[0].note


class TestLaTablaEsUnaTranscripcion:
    """
    Cada fila es copia de un nodo del grafo. Si la copia deja de coincidir, la
    procedencia es falsa — y una cifra con procedencia falsa se defiende sola
    en una revisión. `python -m src.graph_context.water_targets --check` es la
    comprobación contra el grafo vivo; esto es lo que se puede exigir offline.
    """

    @pytest.mark.parametrize("clave", sorted(WATER_TARGETS))
    def test_toda_fila_declara_de_donde_sale(self, clave):
        b = WATER_TARGETS[clave]
        assert b.node_id, f"{clave} no dice de qué nodo salió"
        assert b.verbatim, f"{clave} no guarda la frase original"

    @pytest.mark.parametrize("clave", sorted(WATER_TARGETS))
    def test_los_extremos_aparecen_en_la_frase_citada(self, clave):
        """Una cota que no está en el texto transcrito es una cota inventada."""
        b = WATER_TARGETS[clave]
        texto = b.verbatim.replace(",", "")
        for v in (b.low, b.high, b.preferred_low, b.preferred_high):
            if v is None:
                continue
            crudo = str(int(v)) if float(v).is_integer() else str(v)
            assert crudo in texto, f"{clave}: {crudo} no aparece en el verbatim"

    def test_ninguna_fila_se_llama_limite(self):
        """
        El tipo no tiene campo de límite normativo, y es a propósito: no hay
        dónde escribir uno aunque alguien quisiera.
        """
        b = WATER_TARGETS["ph"]
        assert not hasattr(b, "regulatory_limit")
        assert not hasattr(b, "code_maximum")


class TestReconocerUnaCifraPublicada:
    @pytest.mark.parametrize("param,valor", [
        ("ph", 7.8), ("pH", 7.2), ("cyanuric_acid", 90.0), ("cya", 50),
        ("free_chlorine", 2.0), ("fc", 1.0), ("total_alkalinity", 180),
        ("ta", 120), ("calcium_hardness", 400),
    ])
    def test_los_extremos_del_corpus_se_reconocen(self, param, valor):
        assert is_published_figure(param, valor) is True

    @pytest.mark.parametrize("param,valor", [
        ("calcium_hardness", 350.0),   # el inventado del trace 6660e14f
        ("ph", 8.0), ("cyanuric_acid", 100),
    ])
    def test_una_cifra_ajena_al_corpus_no(self, param, valor):
        assert is_published_figure(param, valor) is False

    def test_un_parametro_que_la_tabla_no_cubre_no_afirma_nada(self):
        assert target_band("turbidity") is None
        assert is_published_figure("turbidity", 1.0) is False


class TestDegradacion:
    def test_un_extremo_publicado_deja_de_ser_limite(self):
        rec = reconcile_with_published_bands([_lectura("ph", 7.9, "above_maximum", 7.8)])
        assert rec[0]["regulatory_limit"] is None
        assert rec[0]["educational_bound"] == 7.8

    def test_el_hallazgo_sobrevive_a_la_degradacion(self):
        """
        Degradar la etiqueta no puede costar el dato: 7.9 sigue estando por
        encima del rango publicado y eso es accionable.
        """
        nota = _nota(_lectura("ph", 7.9, "above_maximum", 7.8))
        assert "7.8" in nota
        assert "not a code limit" in nota
        assert "violation" not in nota

    def test_la_eleccion_del_especialista_se_conserva_si_es_del_corpus(self):
        """
        2 ppm es el mínimo del escalón por isocianuratos, no el general de 1.
        Esa elección lleva información y no se sustituye por el extremo bajo.
        """
        rec = reconcile_with_published_bands(
            [_lectura("free_chlorine", 0.8, "below_minimum", 2.0)])
        assert rec[0]["educational_bound"] == 2.0

    def test_un_limite_inventado_no_se_conserva(self):
        """El techo de dureza de 350 ppm del trace. Ninguna fuente lo respalda."""
        rec = reconcile_with_published_bands(
            [_lectura("calcium_hardness", 380, "above_maximum", 350.0)])
        assert rec[0]["regulatory_limit"] is None
        assert "educational_bound" not in rec[0]

    def test_y_no_se_sustituye_por_otra_infraccion_falsa(self):
        """
        380 está DENTRO de la banda 150–400. Reportarlo contra el 400 sería
        cambiar una infracción falsa por otra. Sale sin veredicto, con su
        objetivo, que es lo único que el dato sostiene.
        """
        nota = _nota(_lectura("calcium_hardness", 380, "above_maximum", 350.0))
        assert "violation" not in nota
        assert "ceiling" not in nota and "350" not in nota
        # El 400 sí aparece — como extremo del objetivo, no como cota excedida.
        assert nota == "reported; operating target 150–400 ppm"

    def test_pero_si_el_valor_sale_de_la_banda_se_reporta_contra_ella(self):
        nota = _nota(_lectura("calcium_hardness", 450, "above_maximum", 350.0))
        assert "above the typical 400 ppm ceiling" in nota

    def test_un_parametro_fuera_de_la_tabla_queda_intacto(self):
        """
        Sin fila no hay base para degradar NI para sostener. Degradar por
        desconocimiento borraría un límite que quizá sí venía citado.
        """
        rec = reconcile_with_published_bands(
            [_lectura("turbidity", 1.2, "above_maximum", 1.0)])
        assert rec[0]["regulatory_limit"] == 1.0
        assert coherent_status(rec[0]) == "above_maximum"

    def test_no_muta_los_dicts_del_caller(self):
        original = _lectura("ph", 7.9, "above_maximum", 7.8)
        reconcile_with_published_bands([original])
        assert original["regulatory_limit"] == 7.8


class TestRellenoDeObjetivos:
    def test_se_rellena_desde_la_banda_preferida(self):
        rec = reconcile_with_published_bands([_lectura("ph", 7.9, "above_maximum", 7.8)])
        assert rec[0]["operating_target"] == "7.4–7.6"

    def test_no_pisa_el_objetivo_del_especialista(self):
        rec = reconcile_with_published_bands(
            [_lectura("ph", 7.9, "above_maximum", 7.8, target=7.4)])
        assert rec[0]["operating_target"] == 7.4

    def test_el_cloro_libre_NO_se_rellena(self):
        """
        El objetivo de cloro escala con el cianúrico (~7.5% del medido). Con
        CYA 90 la banda diría 1–4 ppm y el operador reabriría con el agua
        efectivamente sin desinfectar. El propio nodo del grafo lo dice: dar
        la cifra en rango por su cuenta "reads as sufficient and it is not".
        """
        rec = reconcile_with_published_bands(
            [_lectura("free_chlorine", 0.8, "below_minimum", 2.0)])
        assert rec[0]["operating_target"] is None
        assert WATER_TARGETS["free chlorine"].fill_target is False

    def test_tds_tampoco(self):
        """El corpus juzga TDS contra una línea base, "not at any absolute figure"."""
        rec = reconcile_with_published_bands([_lectura("tds", 3200, "in_range")])
        assert rec[0]["operating_target"] is None


class TestFraseoEnLosDosIdiomas:
    @pytest.mark.parametrize("lang,marca", [
        ("en", "not a code limit"),
        ("es", "no límite de código"),
    ])
    def test_la_linea_dice_que_no_es_codigo(self, lang, marca):
        assert marca in _nota(_lectura("ph", 7.9, "above_maximum", 7.8), lang)

    def test_un_techo_degradado_no_afirma_cumplimiento(self):
        """
        "compliant, no margin at the ceiling" es una afirmación de código
        tanto como "in violation". Sin límite que la respalde, tampoco.
        """
        nota = _nota(_lectura("cyanuric_acid", 90, "at_ceiling", 90.0))
        assert "compliant" not in nota
        assert "at the top of the typical range" in nota


class TestCableadoEnElContrato:
    def test_el_panel_se_arma_sobre_las_lecturas_ya_degradadas(self):
        p = SynthesizerOutput(answer="Cerrá la pileta.", actions=[], safety=None, details=[])
        p, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"], detail_cls=DetailSection,
            readings=[_lectura("ph", 7.9, "above_maximum", 7.8)], language="en",
        )
        nota = p.readings[0].note
        assert "not a code limit" in nota and "violation" not in nota
        assert rep.limits_demoted == ["pH"]
        assert rep.targets_filled == ["pH"]

    def test_la_telemetria_distingue_degradado_de_rellenado(self):
        p = SynthesizerOutput(answer="x", actions=[], safety=None, details=[])
        _, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"], detail_cls=DetailSection,
            readings=[_lectura("turbidity", 1.2, "above_maximum", 1.0)], language="en",
        )
        assert rep.limits_demoted == [] and rep.targets_filled == []
