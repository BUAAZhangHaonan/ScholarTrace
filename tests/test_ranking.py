"""Tests for the ranking service."""

from __future__ import annotations

from scholartrace.models.schemas import Theme, Work
from scholartrace.services.ranking import rank_papers


# ── Helpers ──────────────────────────────────────────────────────────
def _work(
    *,
    title: str = "Test Paper",
    abstract: str | None = None,
    year: int | None = 2024,
    venue: str | None = None,
    citation_count: int = 0,
    fulltext_available: bool = False,
    source_provenance: list[str] | None = None,
) -> Work:
    return Work(
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        citation_count=citation_count,
        fulltext_available=fulltext_available,
        source_provenance=source_provenance or [],
    )


def _theme(queries: list[str] | None = None) -> Theme:
    return Theme(parsed_queries=queries or ["machine learning"])


# ── Tests ────────────────────────────────────────────────────────────
class TestScoreRanges:
    """All individual scores and composite must be in [0, 1]."""

    def test_scores_in_valid_range(self):
        works = [
            _work(
                title="Deep Learning for NLP",
                abstract="We propose a transformer model.",
                year=2023,
                venue="NeurIPS",
                citation_count=50,
                fulltext_available=True,
                source_provenance=["openalex", "semantic_solar"],
            ),
            _work(
                title="Computer Vision Survey",
                abstract="A survey of CNN architectures.",
                year=2018,
                venue="arXiv",
                citation_count=300,
            ),
        ]
        result = rank_papers(works, _theme(["deep learning transformers"]))
        for w in result:
            assert 0.0 <= w.relevance_score <= 1.0, f"relevance out of range: {w.relevance_score}"
            assert 0.0 <= w.recency_score <= 1.0, f"recency out of range: {w.recency_score}"
            assert 0.0 <= w.influence_score <= 1.0, f"influence out of range: {w.influence_score}"
            assert 0.0 <= w.venue_score <= 1.0, f"venue out of range: {w.venue_score}"
            assert 0.0 <= w.composite_score <= 1.0, f"composite out of range: {w.composite_score}"


class TestCompositeIsWeightedSum:
    """Composite score must equal the weighted sum of components."""

    def test_composite_math(self):
        weights = {
            "weight_relevance": 0.35,
            "weight_recency": 0.20,
            "weight_influence": 0.20,
            "weight_venue": 0.10,
            "weight_fulltext": 0.10,
            "weight_source_agreement": 0.05,
        }
        w = _work(
            title="irrelevant so relevance varies",
            abstract="irrelevant",
            year=2024,
            venue="NeurIPS",
            citation_count=100,
            fulltext_available=True,
            source_provenance=["a", "b", "c"],
        )
        # Need a second work so max_citations is 100
        works = [w, _work(title="other", citation_count=10)]
        result = rank_papers(works, _theme(["xyz unmatched"]), weights=weights)
        ranked_w = result[0] if result[0].title == w.title else result[1]

        expected_composite = (
            weights["weight_relevance"] * ranked_w.relevance_score
            + weights["weight_recency"] * ranked_w.recency_score
            + weights["weight_influence"] * ranked_w.influence_score
            + weights["weight_venue"] * ranked_w.venue_score
            + weights["weight_fulltext"] * 1.0  # fulltext_available
            + weights["weight_source_agreement"] * 1.0  # 3 sources / 3 = 1.0
        )
        assert abs(ranked_w.composite_score - expected_composite) < 1e-9


class TestRelevanceBoost:
    """A paper matching theme queries ranks higher than an irrelevant one."""

    def test_higher_relevance_ranks_higher(self):
        works = [
            _work(title="Unrelated Physics Paper", abstract="We study quantum entanglement."),
            _work(title="Deep Learning for NLP", abstract="We train transformers for language."),
        ]
        result = rank_papers(works, _theme(["deep learning language transformers"]))
        # The NLP paper should rank first
        assert result[0].title == "Deep Learning for NLP"
        assert result[0].relevance_score > result[1].relevance_score


class TestRecencyBoost:
    """Recent papers should get a recency boost over older ones."""

    def test_recent_paper_ranks_higher(self):
        # Use uniform weights so recency is the only differentiator
        weights = {
            "weight_relevance": 0.0,
            "weight_recency": 1.0,
            "weight_influence": 0.0,
            "weight_venue": 0.0,
            "weight_fulltext": 0.0,
            "weight_source_agreement": 0.0,
        }
        works = [
            _work(title="Old Paper", year=2015, citation_count=100),
            _work(title="New Paper", year=2024, citation_count=100),
        ]
        result = rank_papers(works, _theme(), weights=weights)
        assert result[0].title == "New Paper"
        assert result[0].recency_score > result[1].recency_score


class TestInfluenceBoost:
    """Highly cited papers should get an influence boost."""

    def test_highly_cited_ranks_higher(self):
        weights = {
            "weight_relevance": 0.0,
            "weight_recency": 0.0,
            "weight_influence": 1.0,
            "weight_venue": 0.0,
            "weight_fulltext": 0.0,
            "weight_source_agreement": 0.0,
        }
        works = [
            _work(title="Low Cite", citation_count=10),
            _work(title="High Cite", citation_count=1000),
        ]
        result = rank_papers(works, _theme(), weights=weights)
        assert result[0].title == "High Cite"
        assert result[0].influence_score > result[1].influence_score


class TestSourceAgreementBoost:
    """Papers from more sources should get an agreement boost."""

    def test_multi_source_ranks_higher(self):
        weights = {
            "weight_relevance": 0.0,
            "weight_recency": 0.0,
            "weight_influence": 0.0,
            "weight_venue": 0.0,
            "weight_fulltext": 0.0,
            "weight_source_agreement": 1.0,
        }
        works = [
            _work(title="Single Source", source_provenance=["openalex"]),
            _work(title="Triple Source", source_provenance=["openalex", "arxiv", "semantic_solar"]),
        ]
        result = rank_papers(works, _theme(), weights=weights)
        assert result[0].title == "Triple Source"


class TestVenueScoring:
    """NeurIPS paper should get higher venue score than unknown venue."""

    def test_top_tier_venue(self):
        weights = {
            "weight_relevance": 0.0,
            "weight_recency": 0.0,
            "weight_influence": 0.0,
            "weight_venue": 1.0,
            "weight_fulltext": 0.0,
            "weight_source_agreement": 0.0,
        }
        works = [
            _work(title="Unknown Venue Paper", venue="ObscureConf"),
            _work(title="NeurIPS Paper", venue="NeurIPS"),
        ]
        result = rank_papers(works, _theme(), weights=weights)
        assert result[0].title == "NeurIPS Paper"
        assert result[0].venue_score == 1.0
        assert result[1].venue_score == 0.5  # unknown

    def test_venue_none(self):
        w = _work(venue=None)
        rank_papers([w], _theme())
        assert w.venue_score == 0.3

    def test_good_venue_workshop(self):
        w = _work(venue="ICML Workshop on RL")
        rank_papers([w], _theme())
        assert w.venue_score == 0.7

    def test_arxiv_venue(self):
        w = _work(venue="arXiv")
        rank_papers([w], _theme())
        assert w.venue_score == 0.7


class TestEmptyInput:
    """Empty works list returns empty list."""

    def test_empty_list(self):
        result = rank_papers([], _theme())
        assert result == []


class TestMissingFields:
    """Works with missing fields should be handled gracefully."""

    def test_no_year_no_venue_no_abstract(self):
        w = _work(title="", abstract=None, year=None, venue=None, citation_count=0)
        result = rank_papers([w], _theme(["test query"]))
        assert len(result) == 1
        assert result[0].recency_score == 0.0
        assert result[0].venue_score == 0.3
        assert result[0].influence_score == 0.0
        assert result[0].composite_score >= 0.0

    def test_no_title_no_abstract_relevance_zero(self):
        w = _work(title="", abstract=None)
        result = rank_papers([w], _theme(["some query"]))
        assert result[0].relevance_score == 0.0
