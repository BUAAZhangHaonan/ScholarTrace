"""Tests for the full-text acquisition cascade service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scholartrace.config import Settings
from scholartrace.models.schemas import (
    AccessStatus,
    Artifact,
    ArtifactKind,
    Section,
    Work,
)
from scholartrace.services.fulltext import (
    _extract_pdf_text,
    _parse_html_sections,
    acquire_fulltext,
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

    def __init__(self, content: bytes | str, status_code: int = 200):
        if isinstance(content, str):
            self._text = content
            self.content = content.encode("utf-8")
        else:
            self.content = content
            self._text = content.decode("utf-8", errors="replace")
        self.status_code = status_code

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
        return _FakeResponse(self._responses[url])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


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
