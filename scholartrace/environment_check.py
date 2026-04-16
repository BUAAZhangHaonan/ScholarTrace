from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
import tomllib

_IMPORT_NAME_MAP = {
    "beautifulsoup4": "bs4",
    "pydantic-settings": "pydantic_settings",
    "PyMuPDF": "fitz",
    "pytest-asyncio": "pytest_asyncio",
    "scikit-learn": "sklearn",
    "uvicorn[standard]": "uvicorn",
}


def _normalize_dependency_name(spec: str) -> str:
    name = re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]
    return name.strip()


def load_declared_dependencies(
    pyproject_path: Path,
    *,
    include_dev: bool,
) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)

    project = data.get("project", {})
    dependencies = [
        _normalize_dependency_name(spec)
        for spec in project.get("dependencies", [])
    ]

    if include_dev:
        dependencies.extend(
            _normalize_dependency_name(spec)
            for spec in project.get("optional-dependencies", {}).get("dev", [])
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for dependency in dependencies:
        if dependency and dependency not in seen:
            seen.add(dependency)
            ordered.append(dependency)
    return ordered


def dependency_to_import_name(dependency: str) -> str:
    return _IMPORT_NAME_MAP.get(dependency, dependency)


def missing_imports(dependencies: list[str]) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for dependency in dependencies:
        import_name = dependency_to_import_name(dependency)
        try:
            import_module(import_name)
        except ImportError:
            missing.append((dependency, import_name))
    return missing


def build_install_command(*, include_dev: bool) -> str:
    editable_target = '".[dev]"' if include_dev else "."
    return (
        f"python -m pip install -c constraints-dev.txt -e {editable_target}"
    )
