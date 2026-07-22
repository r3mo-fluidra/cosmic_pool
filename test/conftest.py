"""
Este conftest se ejecuta ANTES de que pytest importe cualquier archivo de test
en este directorio (y subdirectorios). Es el único lugar donde tiene sentido
neutralizar `langfuse.observe`, porque nodes.py aplica `@observe(...)` sobre
`planner`, `orchestrator` y `synthesizer` en tiempo de IMPORT — no en tiempo
de ejecución. Si lo parcheas después de que `nodes` ya fue importado (p.ej.
con monkeypatch dentro de un test), la decoración real ya está aplicada y no
sirve de nada.

Al reemplazar `sys.modules["langfuse"]` con un stub aquí, cuando cualquier
test haga `from src.agent import nodes`, la línea `from langfuse import observe`
de nodes.py resolverá contra este stub no-op en vez de contra el SDK real
(que intenta reportar trazas a us.cloud.langfuse.com y explota en el entorno
de test por falta de config/credenciales válidas).
"""

import sys
import types


def _passthrough_observe(*decorator_args, **decorator_kwargs):
    """Imita la firma de langfuse.observe pero no hace absolutamente nada:
    ni crea cliente, ni abre spans, ni intenta conectar a ningún host.

    Soporta ambos usos reales de la librería:
        @observe                          -> decorator_args = (fn,)
        @observe(as_type=..., name=...)   -> decorator_args = (), decorator_kwargs = {...}
    """
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]

    def _decorator(fn):
        return fn

    return _decorator


_fake_langfuse_module = types.ModuleType("langfuse")
_fake_langfuse_module.observe = _passthrough_observe
sys.modules["langfuse"] = _fake_langfuse_module