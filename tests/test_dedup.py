"""Tests for the deduplication service."""

from scholartrace.models.schemas import RawCandidate, SourceName
from scholartrace.services.dedup import deduplicate_candidates


def _make(
    title: str = "Test Paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    s2_id: str | None = None,
    openalex_id: str | None = None,
    dblp_key: str | None = None,
    openreview_id: str | None = None,
    source: SourceName = SourceName.OPENALEX,
    year: int | None = 2024,
    authors: list[str] | None = None,
    abstract: str | None = None,
    citation_count: int = 0,
    reference_count: int = 0,
    venue: str | None = None,
    fulltext_url: str | None = None,
    pdf_url: str | None = None,
    html_url: str | None = None,
    oa_url: str | None = None,
    license: str | None = None,
) -> RawCandidate:
    return RawCandidate(
        title=title,
        doi=doi,
        arxiv_id=arxiv_id,
        s2_id=s2_id,
        openalex_id=openalex_id,
        dblp_key=dblp_key,
        openreview_id=openreview_id,
        source=source,
        year=year,
        authors=authors or [],
        abstract=abstract,
        citation_count=citation_count,
        reference_count=reference_count,
        venue=venue,
        fulltext_url=fulltext_url,
        pdf_url=pdf_url,
        html_url=html_url,
        oa_url=oa_url,
        license=license,
    )


class TestSameDOIDifferentSources:
    """Test 1: Same DOI from different sources -> merged, provenance has both."""

    def test_merged(self):
        a = _make(title="Paper A", doi="10.1234/test", source=SourceName.OPENALEX)
        b = _make(
            title="Paper A", doi="10.1234/test", source=SourceName.SEMANTIC_SCHOLAR
        )
        result = deduplicate_candidates([a, b])
        assert len(result) == 1
        assert result[0].doi == "10.1234/test"

    def test_provenance_has_both_sources(self):
        a = _make(title="Paper A", doi="10.1234/test", source=SourceName.OPENALEX)
        b = _make(
            title="Paper A", doi="10.1234/test", source=SourceName.SEMANTIC_SCHOLAR
        )
        result = deduplicate_candidates([a, b])
        assert set(result[0].source_provenance) == {"openalex", "semantic_scholar"}


class TestSameArxivID:
    """Test 2: Same arXiv ID -> merged."""

    def test_merged(self):
        a = _make(title="Paper A", arxiv_id="2401.00001", source=SourceName.ARXIV)
        b = _make(
            title="Paper A",
            arxiv_id="2401.00001",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.00001"


class TestSameS2ID:
    """Test 3: Same S2 ID -> merged."""

    def test_merged(self):
        a = _make(
            title="Paper A",
            s2_id="abc123",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        b = _make(title="Paper A", s2_id="abc123", source=SourceName.OPENALEX)
        result = deduplicate_candidates([a, b])
        assert len(result) == 1
        assert result[0].s2_id == "abc123"


class TestFuzzyTitleMatchSameYear:
    """Test 4: Fuzzy title match + same year -> merged."""

    def test_merged(self):
        a = _make(
            title="Deep Learning for Natural Language Processing",
            source=SourceName.OPENALEX,
            year=2023,
        )
        b = _make(
            title="Deep Learning for Natural Language Processsing",
            source=SourceName.SEMANTIC_SCHOLAR,
            year=2023,
        )
        result = deduplicate_candidates([a, b])
        assert len(result) == 1


class TestFuzzyTitleMatchDifferentYear:
    """Test 5: Fuzzy title match but different year -> NOT merged."""

    def test_not_merged(self):
        a = _make(
            title="Deep Learning for Natural Language Processing",
            source=SourceName.OPENALEX,
            year=2023,
        )
        b = _make(
            title="Deep Learning for Natural Language Processsing",
            source=SourceName.SEMANTIC_SCHOLAR,
            year=2024,
        )
        result = deduplicate_candidates([a, b])
        assert len(result) == 2


class TestCompletelyDifferentPapers:
    """Test 6: Completely different papers -> stay separate."""

    def test_stay_separate(self):
        a = _make(
            title="Attention Is All You Need",
            year=2017,
            source=SourceName.OPENALEX,
        )
        b = _make(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            year=2019,
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert len(result) == 2


class TestBestMetadata:
    """Test 7: Merged candidate keeps best metadata."""

    def test_longest_title(self):
        a = _make(title="Short Title", doi="10.1234/x", source=SourceName.OPENALEX)
        b = _make(
            title="A Much Longer and More Descriptive Title",
            doi="10.1234/x",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert result[0].title == "A Much Longer and More Descriptive Title"

    def test_longest_abstract(self):
        a = _make(
            abstract="Short abstract.",
            doi="10.1234/x",
            source=SourceName.OPENALEX,
        )
        b = _make(
            abstract="This is a much longer and more detailed abstract "
            "that provides comprehensive information about the paper.",
            doi="10.1234/x",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert result[0].abstract and len(result[0].abstract) > 20

    def test_max_citations(self):
        a = _make(
            citation_count=10,
            doi="10.1234/x",
            source=SourceName.OPENALEX,
        )
        b = _make(
            citation_count=500,
            doi="10.1234/x",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert result[0].citation_count == 500

    def test_first_non_none_year(self):
        a = _make(year=None, doi="10.1234/x", source=SourceName.OPENALEX)
        b = _make(year=2023, doi="10.1234/x", source=SourceName.SEMANTIC_SCHOLAR)
        result = deduplicate_candidates([a, b])
        assert result[0].year == 2023

    def test_first_non_none_venue(self):
        a = _make(venue=None, doi="10.1234/x", source=SourceName.OPENALEX)
        b = _make(venue="NeurIPS", doi="10.1234/x", source=SourceName.SEMANTIC_SCHOLAR)
        result = deduplicate_candidates([a, b])
        assert result[0].venue == "NeurIPS"

    def test_longest_authors(self):
        a = _make(
            authors=["Alice"],
            doi="10.1234/x",
            source=SourceName.OPENALEX,
        )
        b = _make(
            authors=["Alice", "Bob", "Charlie"],
            doi="10.1234/x",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b])
        assert result[0].authors == ["Alice", "Bob", "Charlie"]


class TestMultiHopMerge:
    """Test 8: A matches B by DOI, B matches C by arXiv -> all three merged."""

    def test_transitive_merge(self):
        a = _make(
            title="Paper X",
            doi="10.1234/hop",
            source=SourceName.OPENALEX,
        )
        b = _make(
            title="Paper X",
            doi="10.1234/hop",
            arxiv_id="2401.99999",
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        c = _make(
            title="Paper X",
            arxiv_id="2401.99999",
            source=SourceName.ARXIV,
        )
        result = deduplicate_candidates([a, b, c])
        assert len(result) == 1
        assert result[0].doi == "10.1234/hop"
        assert result[0].arxiv_id == "2401.99999"
        assert set(result[0].source_provenance) == {"openalex", "semantic_scholar", "arxiv"}


class TestEmptyCandidates:
    """Test 9: Empty candidates list -> returns empty list."""

    def test_empty(self):
        result = deduplicate_candidates([])
        assert result == []


class TestNoIdentifiersNoFuzzy:
    """Test 10: No identifiers, no fuzzy match -> all stay separate."""

    def test_all_separate(self):
        a = _make(title="Paper About Cats", year=2023, source=SourceName.OPENALEX)
        b = _make(title="Paper About Dogs", year=2023, source=SourceName.ARXIV)
        c = _make(
            title="Paper About Birds",
            year=2023,
            source=SourceName.SEMANTIC_SCHOLAR,
        )
        result = deduplicate_candidates([a, b, c])
        assert len(result) == 3
