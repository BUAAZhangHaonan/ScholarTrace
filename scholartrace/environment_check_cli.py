from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scholartrace.environment_check import (
    build_install_command,
    load_declared_dependencies,
    missing_imports,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ScholarTrace runtime and test dependencies.",
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Check optional development and test dependencies too.",
    )
    parser.add_argument(
        "--pytest-collect",
        action="store_true",
        help="Also run pytest --collect-only after import checks pass.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"
    dependencies = load_declared_dependencies(
        pyproject_path,
        include_dev=args.include_dev,
    )
    missing = missing_imports(dependencies)

    if missing:
        print("Missing Python modules for declared dependencies:")
        for dependency, import_name in missing:
            print(f"  - {dependency} (import: {import_name})")
        print("\nRecommended install command:")
        print(f"  {build_install_command(include_dev=args.include_dev)}")
        return 1

    print(
        f"Import check passed for {len(dependencies)} declared dependencies "
        f"({'runtime+dev' if args.include_dev else 'runtime only'}).",
    )

    if args.pytest_collect:
        command = ["pytest", "--collect-only", "-q"]
        result = subprocess.run(command, cwd=project_root, check=False)
        if result.returncode != 0:
            print("\nPytest collection failed.")
            return result.returncode
        print("Pytest collection passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
