"""Tests for all six source connectors using mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from scholartrace.config import Settings
from scholartrace.connectors.arxiv import ArxivConnector
from scholartrace.connectors.crossref import CrossrefConnector
from scholartrace.connectors.dblp import DblpConnector
from scholartrace.connectors.openalex import OpenAlexConnector, _reconstruct_abstract
from scholartrace.connectors.openreview import OpenReviewConnector
from scholartrace.connectors.semantic_scholar import SemanticScholarConnector
from scholartrace.models.schemas import SourceName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_response(json_body: dict | None = None, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response."""
    request = MagicMock()
    if json_body is not None:
        resp = httpx.Response(
            status_code=200,
            json=json_body,
            request=request,
        )
    else:
        resp = httpx.Response(
            status_code=200,
            text=text,
            request=request,
        )
    return resp


def _settings() -> Settings:
    return Settings(
        semantic_scholar_api_key="test-key",
        openalex_mailto="test@example.com",
        crossref_mailto="test@example.com",
    )


# ===========================================================================
# OpenAlex
# ===========================================================================
class TestOpenAlexConnector:
    @pytest.fixture()
    def connector(self):
        c = OpenAlexConnector(settings=_settings())
        yield c
        # close is async — but we don't need real cleanup in tests

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: OpenAlexConnector):
        body = {
            "meta": {"next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": "10.1234/test",
                    "title": "Test Paper",
                    "publication_year": 2024,
                    "cited_by_count": 42,
                    "authorships": [
                        {"author": {"display_name": "Alice"}},
                        {"author": {"display_name": "Bob"}},
                    ],
                    "abstract_inverted_index": {
                        "This": [0],
                        "is": [1],
                        "a": [2],
                        "test": [3],
                    },
                    "locations": [
                        {"source": {"display_name": "Nature"}}
                    ],
                    "open_access": {"oa_url": "https://oa.example.com/paper.pdf"},
                    "type": "article",
                },
            ],
        }
        mock_get = AsyncMock(return_value=_mock_response(json_body=body))
        connector._client.get = mock_get

        results = await connector.search("machine learning", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "Test Paper"
        assert r.authors == ["Alice", "Bob"]
        assert r.year == 2024
        assert r.citation_count == 42
        assert r.openalex_id == "W123"
        assert r.abstract == "This is a test"
        assert r.venue == "Nature"
        assert r.doi == "https://doi.org/10.1234/test"
        assert r.oa_url == "https://oa.example.com/paper.pdf"
        assert r.source == SourceName.OPENALEX

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: OpenAlexConnector):
        page1 = {
            "meta": {"next_cursor": "abc123"},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": None,
                    "title": "Paper 1",
                    "publication_year": 2023,
                    "cited_by_count": 0,
                    "authorships": [],
                    "abstract_inverted_index": None,
                    "locations": [],
                    "open_access": None,
                    "type": "article",
                },
            ],
        }
        page2 = {
            "meta": {"next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W2",
                    "doi": "10.5678/p2",
                    "title": "Paper 2",
                    "publication_year": 2024,
                    "cited_by_count": 5,
                    "authorships": [],
                    "abstract_inverted_index": None,
                    "locations": [],
                    "open_access": None,
                    "type": "article",
                },
            ],
        }
        mock_get = AsyncMock(
            side_effect=[
                _mock_response(json_body=page1),
                _mock_response(json_body=page2),
            ]
        )
        connector._client.get = mock_get

        results = await connector.search("test", max_results=200)
        assert len(results) == 2
        assert results[0].title == "Paper 1"
        assert results[1].title == "Paper 2"
        assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: OpenAlexConnector):
        body = {"meta": {"next_cursor": None}, "results": []}
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("obscure query")
        assert results == []

    def test_inverted_index_reconstruction(self):
        idx = {
            "Hello": [0],
            "world": [1],
            "of": [2, 4],
            "science": [3],
            "test": [5],
        }
        assert _reconstruct_abstract(idx) == "Hello world of science of test"

    def test_inverted_index_none(self):
        assert _reconstruct_abstract(None) is None

    @pytest.mark.asyncio
    async def test_missing_title_skipped(self, connector: OpenAlexConnector):
        body = {
            "meta": {"next_cursor": None},
            "results": [
                {"id": "W1", "title": None, "doi": None},
                {"id": "W2", "title": "Valid", "doi": None,
                 "publication_year": 2024, "cited_by_count": 0,
                 "authorships": [], "abstract_inverted_index": None,
                 "locations": [], "open_access": None, "type": "article"},
            ],
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert len(results) == 1
        assert results[0].title == "Valid"

    @pytest.mark.asyncio
    async def test_doi_without_prefix(self, connector: OpenAlexConnector):
        body = {
            "meta": {"next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "10.1234/test",
                    "title": "Paper",
                    "publication_year": 2024,
                    "cited_by_count": 0,
                    "authorships": [],
                    "abstract_inverted_index": None,
                    "locations": [],
                    "open_access": None,
                    "type": "article",
                },
            ],
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results[0].doi == "https://doi.org/10.1234/test"

    @pytest.mark.asyncio
    async def test_doi_already_has_prefix(self, connector: OpenAlexConnector):
        body = {
            "meta": {"next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1234/test",
                    "title": "Paper",
                    "publication_year": 2024,
                    "cited_by_count": 0,
                    "authorships": [],
                    "abstract_inverted_index": None,
                    "locations": [],
                    "open_access": None,
                    "type": "article",
                },
            ],
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results[0].doi == "https://doi.org/10.1234/test"


# ===========================================================================
# arXiv
# ===========================================================================
def _arxiv_xml(entries: list[str]) -> str:
    """Wrap entry XML fragments in a full Atom feed."""
    entries_text = "\n".join(entries)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"'
        ' xmlns:arxiv="http://arxiv.org/schemas/atom">'
        f"{entries_text}"
        "</feed>"
    )


def _arxiv_entry(
    title: str = "Test arXiv Paper",
    summary: str = "This is a test abstract.",
    arxiv_id: str = "2301.12345v1",
    published: str = "2023-01-15T00:00:00Z",
    authors: list[str] | None = None,
    doi: str | None = None,
    pdf_url: str | None = None,
) -> str:
    authors_xml = "\n".join(
        f"<author><name>{a}</name></author>" for a in (authors or ["Test Author"])
    )
    links = ""
    if pdf_url:
        links += f'<link title="pdf" href="{pdf_url}" type="application/pdf"/>'
    doi_el = ""
    if doi:
        doi_el = f"<arxiv:doi>{doi}</arxiv:doi>"
    return (
        "<entry>"
        f"<id>http://arxiv.org/abs/{arxiv_id}</id>"
        f"<title>{title}</title>"
        f"<summary>{summary}</summary>"
        f"<published>{published}</published>"
        f"{authors_xml}"
        f"{links}"
        f"{doi_el}"
        "</entry>"
    )


class TestArxivConnector:
    @pytest.fixture()
    def connector(self):
        return ArxivConnector(settings=_settings())

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: ArxivConnector):
        xml = _arxiv_xml([
            _arxiv_entry(
                title="Deep Learning",
                summary="Abstract here.",
                arxiv_id="2301.12345v1",
                published="2023-06-01T00:00:00Z",
                authors=["Alice", "Bob"],
                doi="10.1234/arxiv",
                pdf_url="https://arxiv.org/pdf/2301.12345v1",
            )
        ])
        connector._client.get = AsyncMock(
            return_value=_mock_response(text=xml))

        # Patch sleep to avoid delay in tests
        with patch("scholartrace.connectors.arxiv.asyncio.sleep", new_callable=AsyncMock):
            results = await connector.search("deep learning", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "Deep Learning"
        assert r.authors == ["Alice", "Bob"]
        assert r.year == 2023
        assert r.abstract == "Abstract here."
        assert r.arxiv_id == "2301.12345"
        assert r.doi == "10.1234/arxiv"
        assert r.pdf_url == "https://arxiv.org/pdf/2301.12345v1"
        assert r.source == SourceName.ARXIV

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: ArxivConnector):
        # Fill the first page so pagination continues (_PAGE_SIZE=200)
        from scholartrace.connectors.arxiv import _PAGE_SIZE
        page1_entries = [_arxiv_entry(
            title=f"P1-{i}", arxiv_id=f"2301.{i:05d}v1") for i in range(_PAGE_SIZE)]
        page1_xml = _arxiv_xml(page1_entries)
        page2_xml = _arxiv_xml(
            [_arxiv_entry(title="Paper 2", arxiv_id="2301.99999v1")])
        connector._client.get = AsyncMock(
            side_effect=[
                _mock_response(text=page1_xml),
                _mock_response(text=page2_xml),
            ]
        )
        with patch("scholartrace.connectors.arxiv.asyncio.sleep", new_callable=AsyncMock):
            results = await connector.search("test", max_results=_PAGE_SIZE + 10)

        assert len(results) == _PAGE_SIZE + 1
        assert results[0].title == "P1-0"
        assert results[-1].title == "Paper 2"

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: ArxivConnector):
        xml = _arxiv_xml([])
        connector._client.get = AsyncMock(
            return_value=_mock_response(text=xml))
        results = await connector.search("obscure")
        assert results == []

    @pytest.mark.asyncio
    async def test_arxiv_id_strips_version(self, connector: ArxivConnector):
        xml = _arxiv_xml([_arxiv_entry(arxiv_id="2403.00123v2")])
        connector._client.get = AsyncMock(
            return_value=_mock_response(text=xml))
        with patch("scholartrace.connectors.arxiv.asyncio.sleep", new_callable=AsyncMock):
            results = await connector.search("test", max_results=10)
        assert results[0].arxiv_id == "2403.00123"

    @pytest.mark.asyncio
    async def test_missing_title_skipped(self, connector: ArxivConnector):
        # Create an entry with empty title
        xml = _arxiv_xml([
            "<entry>"
            "<id>http://arxiv.org/abs/2301.99999v1</id>"
            "<title></title>"
            "<summary>Abstract</summary>"
            "<published>2023-01-01T00:00:00Z</published>"
            "<author><name>Test</name></author>"
            "</entry>"
        ])
        connector._client.get = AsyncMock(
            return_value=_mock_response(text=xml))
        results = await connector.search("test")
        assert results == []


# ===========================================================================
# Semantic Scholar
# ===========================================================================
class TestSemanticScholarConnector:
    @pytest.fixture()
    def connector(self):
        return SemanticScholarConnector(settings=_settings())

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: SemanticScholarConnector):
        body = {
            "total": 1,
            "next": None,
            "data": [
                {
                    "paperId": "abc123",
                    "title": "S2 Test Paper",
                    "year": 2024,
                    "abstract": "Test abstract for S2.",
                    "authors": [
                        {"name": "Carol"},
                        {"name": "Dave"},
                    ],
                    "citationCount": 10,
                    "venue": "ICML",
                    "externalIds": {
                        "DOI": "10.1234/s2test",
                        "ArXiv": "2301.99999",
                        "DBLP": "conf/icml/test2024",
                    },
                    "url": "https://semanticscholar.org/paper/abc123",
                    "openAccessPdf": {"url": "https://pdf.example.com/test.pdf"},
                    "fieldsOfStudy": ["Computer Science"],
                }
            ],
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("test query", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "S2 Test Paper"
        assert r.authors == ["Carol", "Dave"]
        assert r.year == 2024
        assert r.abstract == "Test abstract for S2."
        assert r.citation_count == 10
        assert r.venue == "ICML"
        assert r.doi == "https://doi.org/10.1234/s2test"
        assert r.arxiv_id == "2301.99999"
        assert r.s2_id == "abc123"
        assert r.dblp_key == "conf/icml/test2024"
        assert r.pdf_url == "https://pdf.example.com/test.pdf"
        assert r.source == SourceName.SEMANTIC_SCHOLAR

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: SemanticScholarConnector):
        page1 = {
            "total": 2,
            "next": 1,
            "data": [
                {"paperId": "p1", "title": "Page 1", "year": 2023,
                 "authors": [], "externalIds": {}},
            ],
        }
        page2 = {
            "total": 2,
            "next": None,
            "data": [
                {"paperId": "p2", "title": "Page 2", "year": 2024,
                 "authors": [], "externalIds": {}},
            ],
        }
        connector._client.get = AsyncMock(
            side_effect=[
                _mock_response(json_body=page1),
                _mock_response(json_body=page2),
            ]
        )
        results = await connector.search("test", max_results=200)
        assert len(results) == 2
        assert results[0].title == "Page 1"
        assert results[1].title == "Page 2"

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: SemanticScholarConnector):
        body = {"total": 0, "next": None, "data": []}
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("obscure")
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_fields(self, connector: SemanticScholarConnector):
        body = {
            "total": 1,
            "next": None,
            "data": [
                {"paperId": "p1", "title": "Sparse Paper",
                 "authors": [], "externalIds": {}},
            ],
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert len(results) == 1
        r = results[0]
        assert r.year is None
        assert r.abstract is None
        assert r.doi is None
        assert r.venue is None


# ===========================================================================
# DBLP
# ===========================================================================
class TestDblpConnector:
    @pytest.fixture()
    def connector(self):
        return DblpConnector(settings=_settings())

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: DblpConnector):
        body = {
            "result": {
                "hits": {
                    "@total": 1,
                    "@sent": 1,
                    "hit": [
                        {
                            "info": {
                                "title": "DBLP Test Paper",
                                "authors": {
                                    "author": [
                                        {"text": "Eve"},
                                        {"text": "Frank"},
                                    ]
                                },
                                "year": "2024",
                                "venue": "SIGMOD",
                                "doi": "10.1234/dblp",
                                "key": "conf/sigmod/test2024",
                                "url": "https://dblp.org/rec/conf/sigmod/test2024",
                                "type": "Conference",
                            }
                        }
                    ],
                }
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("database", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "DBLP Test Paper"
        assert r.authors == ["Eve", "Frank"]
        assert r.year == 2024
        assert r.venue == "SIGMOD"
        assert r.doi == "https://doi.org/10.1234/dblp"
        assert r.dblp_key == "conf/sigmod/test2024"
        assert r.html_url == "https://dblp.org/rec/conf/sigmod/test2024"
        assert r.source == SourceName.DBLP

    @pytest.mark.asyncio
    async def test_single_author_edge_case(self, connector: DblpConnector):
        """DBLP returns a single author as a dict, not a list."""
        body = {
            "result": {
                "hits": {
                    "@total": 1,
                    "@sent": 1,
                    "hit": [
                        {
                            "info": {
                                "title": "Solo Paper",
                                "authors": {"author": {"text": "Solo Author"}},
                                "year": "2023",
                                "key": "journals/test/solo2023",
                                "url": "https://dblp.org/rec/journals/test/solo2023",
                            }
                        }
                    ],
                }
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("solo")
        assert len(results) == 1
        assert results[0].authors == ["Solo Author"]

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: DblpConnector):
        page1 = {
            "result": {
                "hits": {
                    "@total": 2,
                    "@sent": 1,
                    "hit": [
                        {"info": {"title": "Page 1", "year": "2023",
                                  "authors": {"author": {"text": "A"}},
                                  "key": "k1"}},
                    ],
                }
            }
        }
        page2 = {
            "result": {
                "hits": {
                    "@total": 2,
                    "@sent": 1,
                    "hit": [
                        {"info": {"title": "Page 2", "year": "2024",
                                  "authors": {"author": {"text": "B"}},
                                  "key": "k2"}},
                    ],
                }
            }
        }
        connector._client.get = AsyncMock(
            side_effect=[
                _mock_response(json_body=page1),
                _mock_response(json_body=page2),
            ]
        )
        results = await connector.search("test", max_results=200)
        assert len(results) == 2
        assert results[0].title == "Page 1"
        assert results[1].title == "Page 2"

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: DblpConnector):
        body = {"result": {"hits": {"@total": 0, "@sent": 0, "hit": []}}}
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("obscure")
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_title_skipped(self, connector: DblpConnector):
        body = {
            "result": {
                "hits": {
                    "@total": 1,
                    "@sent": 1,
                    "hit": [
                        {"info": {"title": "", "year": "2024", "key": "k1"}},
                    ],
                }
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results == []


# ===========================================================================
# OpenReview
# ===========================================================================
class TestOpenReviewConnector:
    @pytest.fixture()
    def connector(self):
        return OpenReviewConnector(settings=_settings())

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: OpenReviewConnector):
        body = {
            "notes": [
                {
                    "id": "note1",
                    "forum": "forum123",
                    "content": {
                        "title": {"value": "OpenReview Paper"},
                        "abstract": {"value": "Test abstract for OR."},
                        "authors": {"value": ["Author A", "Author B"]},
                        "venue": {"value": "NeurIPS 2024"},
                    },
                }
            ]
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("test", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "OpenReview Paper"
        assert r.authors == ["Author A", "Author B"]
        assert r.abstract == "Test abstract for OR."
        assert r.venue == "NeurIPS 2024"
        assert r.openreview_id == "forum123"
        assert r.source == SourceName.OPENREVIEW

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: OpenReviewConnector):
        from scholartrace.connectors.openreview import _PAGE_SIZE
        page1_notes = [
            {"id": f"n1-{i}", "forum": f"f1-{i}",
             "content": {"title": {"value": f"Page 1 Note {i}"}}}
            for i in range(_PAGE_SIZE)
        ]
        page1 = {"notes": page1_notes}
        page2 = {
            "notes": [
                {
                    "id": "n2",
                    "forum": "f2",
                    "content": {"title": {"value": "Page 2"}},
                }
            ]
        }
        connector._client.get = AsyncMock(
            side_effect=[
                _mock_response(json_body=page1),
                _mock_response(json_body=page2),
            ]
        )
        results = await connector.search("test", max_results=_PAGE_SIZE + 10)
        assert len(results) == _PAGE_SIZE + 1
        assert results[0].title == "Page 1 Note 0"
        assert results[-1].title == "Page 2"

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: OpenReviewConnector):
        body = {"notes": []}
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("obscure")
        assert results == []

    @pytest.mark.asyncio
    async def test_authors_as_string(self, connector: OpenReviewConnector):
        body = {
            "notes": [
                {
                    "id": "n1",
                    "forum": "f1",
                    "content": {
                        "title": {"value": "String Authors"},
                        "authors": {"value": "Single Author"},
                    },
                }
            ]
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results[0].authors == ["Single Author"]

    @pytest.mark.asyncio
    async def test_missing_title_skipped(self, connector: OpenReviewConnector):
        body = {
            "notes": [
                {
                    "id": "n1",
                    "forum": "f1",
                    "content": {
                        "abstract": {"value": "No title here"},
                    },
                }
            ]
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_venue_fallback(self, connector: OpenReviewConnector):
        body = {
            "notes": [
                {
                    "id": "n1",
                    "forum": "f1",
                    "content": {
                        "title": {"value": "Paper"},
                        "venueid": {"value": "ICLR 2024"},
                    },
                }
            ]
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results[0].venue == "ICLR 2024"


# ===========================================================================
# Crossref
# ===========================================================================
class TestCrossrefConnector:
    @pytest.fixture()
    def connector(self):
        return CrossrefConnector(settings=_settings())

    @pytest.mark.asyncio
    async def test_search_single_page(self, connector: CrossrefConnector):
        body = {
            "message": {
                "next-cursor": None,
                "items": [
                    {
                        "DOI": "10.5678/crossref",
                        "title": ["Crossref Test Paper"],
                        "author": [
                            {"given": "Grace", "family": "Hopper"},
                        ],
                        "published-print": {"date-parts": [[2024, 3, 15]]},
                        "container-title": ["Journal of Tests"],
                        "is-referenced-by-count": 99,
                        "abstract": "<jats:p>This is a JATS abstract.</jats:p>",
                    }
                ],
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))

        results = await connector.search("test", max_results=10)

        assert len(results) == 1
        r = results[0]
        assert r.title == "Crossref Test Paper"
        assert r.authors == ["Grace Hopper"]
        assert r.year == 2024
        assert r.venue == "Journal of Tests"
        assert r.citation_count == 99
        assert r.abstract == "This is a JATS abstract."
        assert r.doi == "https://doi.org/10.5678/crossref"
        assert r.source == SourceName.CROSSREF

    @pytest.mark.asyncio
    async def test_search_pagination(self, connector: CrossrefConnector):
        page1 = {
            "message": {
                "next-cursor": "cursor_page2",
                "items": [
                    {
                        "DOI": "10.1/p1",
                        "title": ["Page 1"],
                        "author": [],
                        "container-title": [],
                    },
                ],
            }
        }
        page2 = {
            "message": {
                "next-cursor": None,
                "items": [
                    {
                        "DOI": "10.1/p2",
                        "title": ["Page 2"],
                        "author": [],
                        "container-title": [],
                    },
                ],
            }
        }
        connector._client.get = AsyncMock(
            side_effect=[
                _mock_response(json_body=page1),
                _mock_response(json_body=page2),
            ]
        )
        results = await connector.search("test", max_results=200)
        assert len(results) == 2
        assert results[0].title == "Page 1"
        assert results[1].title == "Page 2"

    @pytest.mark.asyncio
    async def test_empty_results(self, connector: CrossrefConnector):
        body = {"message": {"next-cursor": None, "items": []}}
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("obscure")
        assert results == []

    @pytest.mark.asyncio
    async def test_year_from_published_online(self, connector: CrossrefConnector):
        body = {
            "message": {
                "next-cursor": None,
                "items": [
                    {
                        "DOI": "10.1/online",
                        "title": ["Online Paper"],
                        "author": [],
                        "published-online": {"date-parts": [[2023, 1]]},
                        "container-title": [],
                    },
                ],
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results[0].year == 2023

    @pytest.mark.asyncio
    async def test_jats_stripping(self, connector: CrossrefConnector):
        body = {
            "message": {
                "next-cursor": None,
                "items": [
                    {
                        "DOI": "10.1/jats",
                        "title": ["JATS Paper"],
                        "author": [],
                        "container-title": [],
                        "abstract": "<jats:title>Abstract</jats:title><jats:p>First para.</jats:p><jats:p>Second para.</jats:p>",
                    },
                ],
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert "<jats:" not in results[0].abstract
        assert "Abstract" in results[0].abstract

    @pytest.mark.asyncio
    async def test_missing_title_skipped(self, connector: CrossrefConnector):
        body = {
            "message": {
                "next-cursor": None,
                "items": [
                    {"DOI": "10.1/notitle", "title": [], "author": []},
                ],
            }
        }
        connector._client.get = AsyncMock(
            return_value=_mock_response(json_body=body))
        results = await connector.search("test")
        assert results == []
