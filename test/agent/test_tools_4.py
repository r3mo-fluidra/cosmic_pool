"""
Unit tests for the deterministic LSI engine.

These tests DO NOT mock external services.

Only analyze_pool_lsi() uses mocks because it is an orchestrator.
"""

from unittest.mock import patch

import pytest

import src.agent.tools as tools


# ============================================================
# TEMPERATURE FACTOR
# ============================================================

class TestTemperatureFactor:

    @pytest.mark.parametrize(
        "temp_f,expected",
        [
            (35, 0.0),
            (40, 0.1),
            (50, 0.2),
            (58, 0.3),
            (64, 0.4),
            (72, 0.5),
            (80, 0.6),
            (90, 0.7),
            (100, 0.8),
            (110, 0.9),
        ],
    )
    def test_temperature_factor(self, temp_f, expected):
        assert tools._temperature_factor(temp_f) == expected


# ============================================================
# TDS FACTOR
# ============================================================

class TestTDSFactor:

    def test_tds_under_1000(self):
        assert tools._tds_factor(800) == 0.0


    def test_tds_equal_1000(self):
        assert tools._tds_factor(1000) == 0.0


    def test_tds_above_1000(self):
        assert tools._tds_factor(2000) > 0


# ============================================================
# CALCULATE LSI
# ============================================================

class TestCalculateLSI:

    def test_returns_expected_structure(self):

        result = tools.calculate_lsi(
            ph=7.5,
            ta=90,
            ch=300,
            temp_c=25,
            cya=30,
            tds=1000,
        )

        assert "lsi" in result
        assert "water_balance" in result
        assert "factors" in result


    def test_adjusted_ta_is_never_below_one(self):

        result = tools.calculate_lsi(
            ph=7.5,
            ta=10,
            ch=250,
            temp_c=25,
            cya=100,
        )

        assert result["water_balance"]["adjusted_ta"] >= 1


# ============================================================
# CLASSIFY LSI
# ============================================================

class TestClassifyLSI:

    @pytest.mark.parametrize(
        "lsi,state,severity",
        [
            (-0.8, "corrosive", "high"),
            (-0.4, "corrosive", "moderate"),
            (0.0, "balanced", "ideal"),
            (0.5, "scale_forming", "moderate"),
            (0.8, "scale_forming", "high"),
        ],
    )
    def test_classification(
        self,
        lsi,
        state,
        severity,
    ):
        assert tools._classify_lsi(lsi) == (
            state,
            severity,
        )


# ============================================================
# INTERPRET LSI
# ============================================================

class TestInterpretLSI:

    def test_corrosive_water_generates_risks(self):

        result = tools.interpret_lsi(
            lsi=-0.7,
            pool_surface="plaster",
            sanitizer_type="saltwater",
        )

        assert result["classification"]["state"] == "corrosive"

        assert len(result["predicted_risks"]) >= 2


    def test_balanced_water_generates_no_risks(self):

        result = tools.interpret_lsi(
            lsi=0.0,
            pool_surface="plaster",
            sanitizer_type="chlorine",
        )

        assert result["classification"]["state"] == "balanced"

        assert result["predicted_risks"] == []


    def test_scale_forming_generates_scaling_risks(self):

        result = tools.interpret_lsi(
            lsi=0.8,
            pool_surface="plaster",
            sanitizer_type="saltwater",
        )

        assert result["classification"]["state"] == "scale_forming"

        assert len(result["predicted_risks"]) >= 2


# ============================================================
# RECOMMENDATIONS
# ============================================================

class TestRecommendLSICorrection:

    def test_corrosive_recommendations(self):

        result = tools.recommend_lsi_correction(
            lsi=-0.7,
            ph=7.1,
            ta=70,
            ch=220,
        )

        assert result["priority_actions"]

        assert "pH" in result["chemical_strategy"]["increase"]


    def test_scale_recommendations(self):

        result = tools.recommend_lsi_correction(
            lsi=0.8,
            ph=8.0,
            ta=130,
            ch=500,
        )

        assert "pH" in result["chemical_strategy"]["decrease"]


    def test_balanced_water(self):

        result = tools.recommend_lsi_correction(
            lsi=0.0,
            ph=7.5,
            ta=90,
            ch=300,
        )

        assert result["priority_actions"][0]["parameter"] == "none"


# ============================================================
# ORCHESTRATOR
# ============================================================

class TestAnalyzePoolLSI:

    @patch("src.agent.tools.calculate_lsi")
    @patch("src.agent.tools.interpret_lsi")
    @patch("src.agent.tools.recommend_lsi_correction")
    def test_orchestrates_all_components(
        self,
        mock_recommend,
        mock_interpret,
        mock_calculate,
    ):

        mock_calculate.return_value = {
            "lsi": 0.2,
            "water_balance": {},
            "factors": {},
        }

        mock_interpret.return_value = {
            "classification": {},
            "predicted_risks": [],
        }

        mock_recommend.return_value = {
            "priority_actions": [],
            "chemical_strategy": {},
        }

        result = tools.analyze_pool_lsi(
            ph=7.5,
            ta=90,
            ch=300,
            temp_c=25,
        )

        mock_calculate.assert_called_once()

        mock_interpret.assert_called_once()

        mock_recommend.assert_called_once()

        assert "lsi" in result
        assert "classification" in result
        assert "priority_actions" in result