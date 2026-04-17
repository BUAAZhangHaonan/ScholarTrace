"""Tests for the full-text acquisition cascade service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from scholartrace.config import Settings
from scholartrace.models.schemas import (
    AccessStatus,
    AcquisitionState,
    ArtifactKind,
    Work,
)
from scholartrace.services.fulltext import (
    _extract_pdf_text,
    _parse_html_sections,
    acquire_fulltext,
    read_cached_fulltext,
)
from scholartrace.services.storage import StorageService


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
SAMPLE_HTML = """\
<html>
<body>
<h1>Introduction</h1>
<p>This is the introduction section.</p>
<h2>Methods</h2>
<p>We describe our methods here.</p>
<h3>Experiments</h3>
<p>Experimental details follow.</p>
</body>
</html>
"""


def _make_pdf_bytes(text: str = "Hello PDF world") -> bytes:
    """Create a minimal valid PDF containing *text* using PyMuPDF."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        svc = StorageService(db_path)
        svc.init_db()
        yield svc
        svc.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", db_path=tmp_path / "data" / "test.db")


# ------------------------------------------------------------------
# Helper: build a mock httpx.AsyncClient
# ------------------------------------------------------------------
class _FakeResponse:
    """A synchronous fake response that does not involve unittest.mock."""

    def __init__(
        self,
        content: bytes | str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        if isinstance(content, str):
            self._text = content
            self.content = content.encode("utf-8")
        else:
            self.content = content
            self._text = content.decode("utf-8", errors="replace")
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                str(self.status_code),
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(self.status_code),
            )


class _FakeClient:
    """Minimal async-client fake whose ``get`` method is async."""

    def __init__(
        self,
        responses: dict[str, bytes | str] | None = None,
        errors: set[str] | None = None,
    ):
        self._responses = responses or {}
        self._errors = errors or set()

    async def get(self, url: str, **kwargs):
        if url in self._errors or url not in self._responses:
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", url),
                response=httpx.Response(404),
            )
        response = self._responses[url]
        if isinstance(response, _FakeResponse):
            return response
        return _FakeResponse(response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _CountingClient(_FakeClient):
    def __init__(self, *args, delay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []
        self._delay = delay

    async def get(self, url: str, **kwargs):
        import asyncio

        self.calls.append(url)
        if self._delay:
            await asyncio.sleep(self._delay)
        return await super().get(url, **kwargs)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestParseHtmlSections:
    def test_extracts_sections(self):
        sections = _parse_html_sections(SAMPLE_HTML)
        titles = [t for t, _ in sections]
        assert "Introduction" in titles
        assert "Methods" in titles
        assert "Experiments" in titles

    def test_content_associated_with_headings(self):
        sections = _parse_html_sections(SAMPLE_HTML)
        section_map = {t: c for t, c in sections}
        assert "introduction section" in section_map["Introduction"].lower()
        assert "methods here" in section_map["Methods"].lower()

    def test_no_headings_returns_full_text(self):
        html = "<html><body><p>Just some text.</p></body></html>"
        sections = _parse_html_sections(html)
        assert len(sections) == 1
        assert sections[0][0] == "Full Text"
        assert "Just some text" in sections[0][1]


class TestExtractPdfText:
    def test_extracts_text(self):
        pdf_bytes = _make_pdf_bytes("Hello from PDF")
        text = _extract_pdf_text(pdf_bytes)
        assert "Hello from PDF" in text


class TestArxivHtmlPath:
    """Cascade step 1: arXiv HTML works end-to-end."""

    @pytest.mark.asyncio
    async def test_html_success(self, storage, settings):
        arxiv_id = "2301.00001"
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        work = Work(title="Test Paper", arxiv_id=arxiv_id)
        storage.save_work(work)

        fake = _FakeClient(responses={html_url: SAMPLE_HTML})

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        assert result.access_status == AccessStatus.AVAILABLE

        # Check artifacts saved.
        artifacts = storage.get_artifacts_by_work(work.id)
        kinds = {a.kind for a in artifacts}
        assert ArtifactKind.HTML in kinds
        assert ArtifactKind.PARSED_TEXT in kinds

        # Check sections saved.
        sections = storage.get_sections_by_work(work.id)
        assert len(sections) >= 2
        section_titles = [s.section_title for s in sections]
        assert "Introduction" in section_titles
        assert "Methods" in section_titles


class TestArxivPdfFallback:
    """Cascade step 2: HTML fails, PDF succeeds."""

    @pytest.mark.asyncio
    async def test_pdf_fallback(self, storage, settings):
        arxiv_id = "2301.00002"
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        work = Work(title="PDF Paper", arxiv_id=arxiv_id)
        storage.save_work(work)

        pdf_bytes = _make_pdf_bytes("Extracted PDF content here")
        fake = _FakeClient(responses={pdf_url: pdf_bytes}, errors={html_url})

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        assert result.access_status == AccessStatus.AVAILABLE

        artifacts = storage.get_artifacts_by_work(work.id)
        kinds = {a.kind for a in artifacts}
        assert ArtifactKind.PDF in kinds
        assert ArtifactKind.PARSED_TEXT in kinds


class TestOaUrlPath:
    """Cascade step 3: no arxiv_id, use pdf_url."""

    @pytest.mark.asyncio
    async def test_pdf_url_download(self, storage, settings):
        pdf_url = "https://publisher.org/paper.pdf"
        work = Work(title="OA Paper", pdf_url=pdf_url)
        storage.save_work(work)

        pdf_bytes = _make_pdf_bytes("Open access content")
        fake = _FakeClient(responses={pdf_url: pdf_bytes})

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        assert result.access_status == AccessStatus.AVAILABLE

        artifacts = storage.get_artifacts_by_work(work.id)
        assert any(a.kind == ArtifactKind.PDF for a in artifacts)

    @pytest.mark.asyncio
    async def test_oa_url_succeeds_before_deepxiv_fallback(self, storage, settings):
        settings.deepxiv_tokens = "configured-token"
        arxiv_id = "2401.11111"
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        oa_url = "https://publisher.org/open-access.html"
        work = Work(title="OA Before DeepXiv", arxiv_id=arxiv_id, oa_url=oa_url)
        storage.save_work(work)

        fake = _FakeClient(
            responses={oa_url: SAMPLE_HTML},
            errors={html_url, pdf_url},
        )

        class _ForbiddenDeepXivConnector:
            async def get_fulltext(self, requested_arxiv_id: str) -> str | None:
                raise AssertionError("DeepXiv fallback should not run when oa_url succeeds")

            async def close(self) -> None:
                return None

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            with patch(
                "scholartrace.services.fulltext.DeepXivConnector",
                return_value=_ForbiddenDeepXivConnector(),
            ):
                result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        payload = read_cached_fulltext(result, storage, settings)
        assert any(a["source_url"] == oa_url for a in payload["artifacts"])
        assert any(a["kind"] == "html" for a in payload["artifacts"])
        assert not any(a["kind"] == "markdown" for a in payload["artifacts"])


class TestDeepXivFallback:
    @pytest.mark.asyncio
    async def test_explicit_acquire_uses_deepxiv_markdown_after_public_misses(
        self, storage, settings
    ):
        settings.deepxiv_tokens = "configured-token"
        arxiv_id = "2401.12345"
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        work = Work(title="DeepXiv Fallback Paper", arxiv_id=arxiv_id)
        storage.save_work(work)

        fake = _FakeClient(errors={html_url, pdf_url})

        class _FakeDeepXivConnector:
            def __init__(self, settings=None):
                self.calls: list[str] = []

            async def get_fulltext(self, requested_arxiv_id: str) -> str | None:
                self.calls.append(requested_arxiv_id)
                return (
                    "# Introduction\n\nDeepXiv introduction.\n\n"
                    "## Method\n\nDeepXiv method details."
                )

            async def close(self) -> None:
                return None

        connector = _FakeDeepXivConnector()

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            with patch(
                "scholartrace.services.fulltext.DeepXivConnector",
                return_value=connector,
            ):
                result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        assert result.access_status == AccessStatus.AVAILABLE
        assert connector.calls == [arxiv_id]

        payload = read_cached_fulltext(result, storage, settings)
        assert payload["fulltext_available"] is True
        assert payload["acquisition_state"] == AcquisitionState.AVAILABLE.value
        assert payload["needs_acquisition"] is False
        assert any(a["kind"] == "markdown" for a in payload["artifacts"])
        assert [section["section_title"] for section in payload["sections"]] == [
            "Introduction",
            "Method",
        ]
        assert "DeepXiv method details." in (payload["parsed_text"] or "")

    @pytest.mark.asyncio
    async def test_deepxiv_fallback_is_not_used_when_public_html_succeeds(
        self, storage, settings
    ):
        settings.deepxiv_tokens = "configured-token"
        arxiv_id = "2401.54321"
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        work = Work(title="Public HTML First", arxiv_id=arxiv_id)
        storage.save_work(work)

        fake = _FakeClient(responses={html_url: SAMPLE_HTML})

        class _ForbiddenDeepXivConnector:
            async def get_fulltext(self, requested_arxiv_id: str) -> str | None:
                raise AssertionError("DeepXiv fallback should not run when HTML succeeds")

            async def close(self) -> None:
                return None

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            with patch(
                "scholartrace.services.fulltext.DeepXivConnector",
                return_value=_ForbiddenDeepXivConnector(),
            ):
                result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is True
        payload = read_cached_fulltext(result, storage, settings)
        assert any(a["kind"] == "html" for a in payload["artifacts"])
        assert not any(a["kind"] == "markdown" for a in payload["artifacts"])


class TestAbstractOnlyFallback:
    """Cascade step 4: nothing works -> abstract_only."""

    @pytest.mark.asyncio
    async def test_no_arxiv_no_pdf_url(self, storage, settings):
        work = Work(title="Paywalled Paper")
        storage.save_work(work)

        fake = _FakeClient()

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is False
        assert result.access_status == AccessStatus.ABSTRACT_ONLY
        state = storage.get_fulltext_state(work.id)
        assert state is not None
        assert state.acquisition_state == AcquisitionState.NEGATIVE_CACHED


class TestNetworkErrorHandling:
    """HTTP errors are caught and cascade continues gracefully."""

    @pytest.mark.asyncio
    async def test_network_error_falls_through(self, storage, settings):
        arxiv_id = "2301.00003"
        work = Work(title="Error Paper", arxiv_id=arxiv_id)
        storage.save_work(work)

        class _ErrorClient:
            async def get(self, url, **kw):
                raise httpx.ConnectError("connection refused")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=_ErrorClient()):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is False
        assert result.access_status == AccessStatus.ABSTRACT_ONLY


class TestEmptyPdfText:
    """PDF downloads but PyMuPDF extracts nothing -> abstract_only."""

    @pytest.mark.asyncio
    async def test_empty_pdf_text(self, storage, settings):
        pdf_url = "https://example.com/empty.pdf"
        work = Work(title="Empty PDF", pdf_url=pdf_url)
        storage.save_work(work)

        # Create a minimal PDF with no extractable text.
        import fitz
        doc = fitz.open()
        doc.new_page()  # blank page, no text
        pdf_bytes = doc.tobytes()
        doc.close()

        fake = _FakeClient(responses={pdf_url: pdf_bytes})

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is False
        assert result.access_status == AccessStatus.ABSTRACT_ONLY


class TestHardening:
    @pytest.mark.asyncio
    async def test_ssrf_blocked_and_logged(self, storage, settings, caplog):
        work = Work(title="SSRF Paper", pdf_url="http://127.0.0.1:8080/private.pdf")
        storage.save_work(work)

        class _NeverCalledClient:
            async def get(self, url, **kwargs):
                raise AssertionError("network fetch should not be attempted for blocked targets")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=_NeverCalledClient()):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is False
        state = storage.get_fulltext_state(work.id)
        assert state is not None
        assert state.acquisition_state == AcquisitionState.NEGATIVE_CACHED
        assert "Blocked unsafe full-text fetch" in caplog.text
        assert "127.0.0.1:8080" in caplog.text

    @pytest.mark.asyncio
    async def test_redirect_to_private_host_is_blocked(self, storage, settings, caplog):
        pdf_url = "https://example.com/paper.pdf"
        work = Work(title="Redirect Paper", pdf_url=pdf_url)
        storage.save_work(work)

        fake = _CountingClient(
            responses={
                pdf_url: _FakeResponse(
                    "",
                    status_code=302,
                    headers={"location": "http://127.0.0.1:9000/secret.pdf"},
                ),
            }
        )

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            result = await acquire_fulltext(work, storage, settings)

        assert result.fulltext_available is False
        assert fake.calls == [pdf_url]
        assert "Blocked unsafe full-text fetch" in caplog.text

    @pytest.mark.asyncio
    async def test_negative_cache_skips_repeat_fetch(self, storage, settings):
        pdf_url = "https://example.com/missing.pdf"
        work = Work(title="Negative Cache Paper", pdf_url=pdf_url)
        storage.save_work(work)

        fake = _CountingClient()

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            first = await acquire_fulltext(work, storage, settings)
            second = await acquire_fulltext(first, storage, settings)

        assert first.fulltext_available is False
        assert second.fulltext_available is False
        assert fake.calls == [pdf_url]
        state = storage.get_fulltext_state(work.id)
        assert state is not None
        assert state.acquisition_state == AcquisitionState.NEGATIVE_CACHED
        assert state.next_retry_at is not None

    @pytest.mark.asyncio
    async def test_concurrent_acquire_requests_collapse_to_one_fetch(self, storage, settings):
        pdf_url = "https://example.com/collapse.pdf"
        work = Work(title="Collapse Paper", pdf_url=pdf_url)
        storage.save_work(work)

        pdf_bytes = _make_pdf_bytes("Collapsed download")
        fake = _CountingClient(responses={pdf_url: pdf_bytes}, delay=0.05)

        with patch("scholartrace.services.fulltext.httpx.AsyncClient", return_value=fake):
            results = await __import__("asyncio").gather(
                acquire_fulltext(work, storage, settings),
                acquire_fulltext(work, storage, settings),
            )

        assert all(result.fulltext_available for result in results)
        assert fake.calls == [pdf_url]

    def test_read_cached_fulltext_reports_missing_without_network(self, storage, settings):
        work = Work(title="Cache Only")
        storage.save_work(work)

        payload = read_cached_fulltext(work, storage, settings)
        assert payload["fulltext_available"] is False
        assert payload["acquisition_state"] == AcquisitionState.MISSING.value
        assert payload["needs_acquisition"] is True
