# src/tools_math/__init__.py
"""Deterministic math layer for the MATH agent."""

from .catalog import CatalogError, load_catalog
from .tools import (
    MATH_TOOL_NAMES,
    MATH_TOOLS,
    calculate,
    check_plausibility,
    convert_units,
    get_constant,
    lookup_product,
    resolve_formula,
)

__all__ = [
    "MATH_TOOLS", "MATH_TOOL_NAMES", "load_catalog", "CatalogError",
    "resolve_formula", "get_constant", "convert_units",
    "lookup_product", "calculate", "check_plausibility",
    "get_catalog",
]

# ------------------------------------------------------------------
# Carga única del catálogo
# ------------------------------------------------------------------
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Cargando catálogo matemático...")
    def get_catalog():
        """Devuelve el catálogo (cacheado por Streamlit + cache interno)."""
        return load_catalog()

except ImportError:
    # Si se usa fuera de Streamlit
    def get_catalog():
        return load_catalog()

# Carga eager al importar el paquete → queda en RAM desde el principio
_ = get_catalog()