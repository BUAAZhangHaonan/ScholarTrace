from __future__ import annotations

from pathlib import Path

import pytest


def test_load_declared_dependencies_includes_runtime_and_dev_packages():
    from scholartrace.environment_check import load_declared_dependencies

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    runtime_only = load_declared_dependencies(pyproject_path, include_dev=False)
    with_dev = load_declared_dependencies(pyproject_path, include_dev=True)

    assert "fastapi" in runtime_only
    assert "pytest" not in runtime_only
    assert "pytest" in with_dev
    assert "pytest-asyncio" in with_dev


def test_missing_imports_uses_import_name_mapping(monkeypatch: pytest.MonkeyPatch):
    from scholartrace.environment_check import missing_imports

    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        if name == "bs4":
            raise ImportError("missing bs4")
        return object()

    monkeypatch.setattr("scholartrace.environment_check.import_module", fake_import)

    missing = missing_imports(["beautifulsoup4", "httpx"])

    assert missing == [("beautifulsoup4", "bs4")]
    assert imported == ["bs4", "httpx"]


def test_build_install_command_uses_constraints_and_dev_extra():
    from scholartrace.environment_check import build_install_command

    command = build_install_command(include_dev=True)
    assert "constraints-dev.txt" in command
    assert '".[dev]"' in command


def test_dependency_to_import_name_maps_pytest_asyncio():
    from scholartrace.environment_check import dependency_to_import_name

    assert dependency_to_import_name("pytest-asyncio") == "pytest_asyncio"
