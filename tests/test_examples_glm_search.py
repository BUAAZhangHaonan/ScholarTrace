from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_example_module():
    path = Path(__file__).resolve().parent.parent / "examples" / "glm_scholar_search.py"
    spec = importlib.util.spec_from_file_location("glm_scholar_search", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_requires_explicit_bigmodel_key(monkeypatch: pytest.MonkeyPatch):
    module = _load_example_module()
    monkeypatch.delenv("SCHOLARTRACE_BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SCHOLARTRACE_BIGMODEL_API_KEY"):
        module.get_bigmodel_api_key()
