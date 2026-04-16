#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from scholartrace.config import get_settings
from scholartrace.services.storage import StorageService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair duplicate and orphaned ScholarTrace SQLite state.",
    )
    parser.add_argument(
        "--db-path",
        help="Path to the ScholarTrace SQLite database. Defaults to SCHOLARTRACE_DB_PATH.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the repair. Without this flag the script runs as a dry run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    storage = StorageService(args.db_path or settings.db_path)
    storage.init_db()
    report = storage.repair_existing_work_state(apply=args.apply)
    print(json.dumps({"apply": args.apply, "report": report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
