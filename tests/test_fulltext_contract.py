from __future__ import annotations

import asyncio
import json
import os
import tempfile

from fastapi.testclient import TestClient

from scholartrace.api.rest import app
from scholartrace.models.schemas import Work
from scholartrace.services import runtime_limits
from scholartrace.services.storage import StorageService


def test_rest_and_mcp_cached_fulltext_payloads_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = StorageService(db_path)
        storage.init_db()

        work = Work(title="Contract Paper", pdf_url="https://example.com/paper.pdf")
        storage.save_work(work)

        import scholartrace.api.rest as rest_module
        import scholartrace.api.mcp_server as mcp_module
        from scholartrace.config import Settings

        settings = Settings(data_dir=os.path.join(tmpdir, "data"), db_path=db_path)
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        rest_module._storage = storage
        rest_module._settings = settings
        mcp_module.set_storage(storage)
        asyncio.run(runtime_limits.budget_manager.reset())

        rest_payload = TestClient(app).get(f"/papers/{work.id}/fulltext").json()
        mcp_payload = json.loads(asyncio.run(mcp_module.get_paper_fulltext(work.id)))

        for key in (
            "paper_id",
            "fulltext_available",
            "access_status",
            "acquisition_state",
            "needs_acquisition",
        ):
            assert rest_payload[key] == mcp_payload[key]

        asyncio.run(runtime_limits.budget_manager.reset())
        storage.close()
