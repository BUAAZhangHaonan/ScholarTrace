from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from scholartrace.models.schemas import (
    AccessStatus,
    Artifact,
    ArtifactKind,
    JobStatus,
    RetrievalJob,
    Section,
    Theme,
    Work,
)
from scholartrace.services.storage import StorageService


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        svc = StorageService(db_path)
        svc.init_db()
        yield svc
        svc.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestInitDb:
    def test_init_db_creates_tables(self, storage):
        conn = storage._get_conn()
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "works" in tables
        assert "artifacts" in tables
        assert "sections" in tables
        assert "themes" in tables
        assert "theme_works" in tables
        assert "jobs" in tables


class TestWorkCRUD:
    def test_save_and_get_work(self, storage):
        work = Work(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            doi="10.5555/3295222.3295349",
            arxiv_id="1706.03762",
            abstract="We propose a new network architecture, the Transformer.",
            venue="NeurIPS",
            citation_count=100000,
            source_provenance=["openalex", "semantic_scholar"],
        )
        saved = storage.save_work(work)
        assert saved.id == work.id

        fetched = storage.get_work(work.id)
        assert fetched is not None
        assert fetched.title == "Attention Is All You Need"
        assert fetched.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert fetched.year == 2017
        assert fetched.doi == "10.5555/3295222.3295349"
        assert fetched.arxiv_id == "1706.03762"
        assert fetched.citation_count == 100000
        assert fetched.source_provenance == ["openalex", "semantic_scholar"]

    def test_get_work_by_doi(self, storage):
        work = Work(
            title="BERT",
            doi="10.18653/v1/N19-1423",
            authors=["Jacob Devlin"],
        )
        storage.save_work(work)

        fetched = storage.get_work_by_doi("10.18653/v1/N19-1423")
        assert fetched is not None
        assert fetched.title == "BERT"

        assert storage.get_work_by_doi("nonexistent") is None

    def test_get_work_by_arxiv_id(self, storage):
        work = Work(title="GPT-3", arxiv_id="2005.14165")
        storage.save_work(work)

        fetched = storage.get_work_by_arxiv_id("2005.14165")
        assert fetched is not None
        assert fetched.title == "GPT-3"

        assert storage.get_work_by_arxiv_id("nonexistent") is None

    def test_list_works_by_theme(self, storage):
        w1 = Work(title="Paper A")
        w2 = Work(title="Paper B")
        w3 = Work(title="Paper C")
        storage.save_work(w1)
        storage.save_work(w2)
        storage.save_work(w3)

        theme = Theme(document_text="test theme")
        storage.save_theme(theme)

        storage.link_theme_work(theme.id, w1.id, 0)
        storage.link_theme_work(theme.id, w2.id, 1)
        storage.link_theme_work(theme.id, w3.id, 2)

        works = storage.list_works_by_theme(theme.id)
        assert len(works) == 3
        assert works[0].title == "Paper A"
        assert works[1].title == "Paper B"
        assert works[2].title == "Paper C"

        assert storage.count_works_by_theme(theme.id) == 3

        # Pagination
        page = storage.list_works_by_theme(theme.id, limit=2, offset=0)
        assert len(page) == 2
        assert page[0].title == "Paper A"
        assert page[1].title == "Paper B"

    def test_save_work_upsert(self, storage):
        work = Work(title="Original Title", doi="10.1234/test")
        storage.save_work(work)

        work.title = "Updated Title"
        storage.save_work(work)

        fetched = storage.get_work(work.id)
        assert fetched is not None
        assert fetched.title == "Updated Title"

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            ("doi", "10.1000/phase2-doi"),
            ("arxiv_id", "2401.12345"),
            ("openalex_id", "W-phase2"),
            ("s2_id", "S2-phase2"),
            ("dblp_key", "conf/test/Phase2"),
            ("openreview_id", "or-phase2"),
        ],
    )
    def test_reingestion_preserves_work_identity_by_external_identifier(
        self,
        storage,
        field_name: str,
        field_value: str,
    ):
        original = Work(title="Original", **{field_name: field_value})
        storage.save_work(original)

        incoming = Work(title="Updated", authors=["Alice"], **{field_name: field_value})
        saved = storage.save_work(incoming)

        assert saved.id == original.id
        assert storage.get_work(original.id) is not None
        assert storage.get_work(incoming.id) is None
        assert getattr(storage.get_work(original.id), field_name) == field_value

    def test_reingestion_preserves_theme_links_artifacts_and_sections(self, storage):
        theme = Theme(document_text="phase2 theme")
        storage.save_theme(theme)

        original = Work(title="Original", doi="10.1000/stable-link")
        storage.save_work(original)
        storage.link_theme_work(theme.id, original.id, 5)

        artifact = Artifact(
            work_id=original.id,
            kind=ArtifactKind.PDF,
            source_url="https://example.com/original.pdf",
            access_status=AccessStatus.AVAILABLE,
        )
        section = Section(
            work_id=original.id,
            section_title="Intro",
            section_order=0,
            text_content="hello",
        )
        storage.save_artifact(artifact)
        storage.save_section(section)

        incoming = Work(
            title="Updated",
            doi="10.1000/stable-link",
            arxiv_id="2402.00001",
        )
        saved = storage.save_work(incoming)

        assert saved.id == original.id
        linked = storage.list_works_by_theme(theme.id, limit=10)
        assert [work.id for work in linked] == [original.id]
        assert storage.get_artifacts_by_work(original.id)[0].work_id == original.id
        assert storage.get_sections_by_work(original.id)[0].work_id == original.id

    def test_repair_duplicate_logical_works_rewires_records_and_keeps_best_rank(self, storage):
        conn = storage._get_conn()
        conn.execute("DROP INDEX idx_works_doi")
        conn.commit()

        theme = Theme(document_text="repair theme")
        storage.save_theme(theme)

        first = Work(id="work-a", title="A", doi="10.1000/repair")
        second = Work(id="work-b", title="B", doi="10.1000/repair", arxiv_id="2403.00002")
        storage.save_work(first)
        with storage.transaction(immediate=True) as tx:
            storage._write_work_row(tx, second)
        storage.link_theme_work(theme.id, first.id, 5)
        storage.link_theme_work(theme.id, second.id, 2)
        storage.save_artifact(
            Artifact(
                id="artifact-b",
                work_id=second.id,
                kind=ArtifactKind.PDF,
                source_url="https://example.com/repair.pdf",
            )
        )
        storage.save_section(
            Section(
                id="section-a",
                work_id=first.id,
                section_title="Intro",
                section_order=0,
                text_content="repair",
            )
        )

        report = storage.repair_existing_work_state(apply=True)

        assert report["clusters_repaired"] == 1
        works = storage.list_works_by_theme(theme.id, limit=10)
        assert len(works) == 1
        canonical_id = works[0].id
        raw_conn = storage._get_conn()
        row = raw_conn.execute(
            "SELECT rank_order FROM theme_works WHERE theme_id = ? AND work_id = ?",
            (theme.id, canonical_id),
        ).fetchone()
        assert row is not None
        assert row["rank_order"] == 2
        assert raw_conn.execute("SELECT COUNT(*) AS c FROM works").fetchone()["c"] == 1
        assert storage.get_artifacts_by_work(canonical_id)[0].work_id == canonical_id
        assert storage.get_sections_by_work(canonical_id)[0].work_id == canonical_id

    def test_concurrent_reingestion_keeps_single_work_row(self, storage):
        def _save(idx: int) -> str:
            work = Work(
                title=f"Concurrent {idx}",
                doi="10.1000/concurrent",
                authors=[f"Author {idx}"],
            )
            return storage.save_work(work).id

        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = list(executor.map(_save, range(4)))

        assert len(set(ids)) == 1
        conn = storage._get_conn()
        count = conn.execute("SELECT COUNT(*) AS c FROM works").fetchone()["c"]
        assert count == 1


class TestArtifactCRUD:
    def test_save_and_get_artifacts(self, storage):
        work = Work(title="Test Paper")
        storage.save_work(work)

        art = Artifact(
            work_id=work.id,
            kind=ArtifactKind.PDF,
            source_url="https://example.com/paper.pdf",
            access_status=AccessStatus.AVAILABLE,
        )
        storage.save_artifact(art)

        artifacts = storage.get_artifacts_by_work(work.id)
        assert len(artifacts) == 1
        assert artifacts[0].kind == ArtifactKind.PDF
        assert artifacts[0].source_url == "https://example.com/paper.pdf"


class TestSectionCRUD:
    def test_save_and_get_sections(self, storage):
        work = Work(title="Test Paper")
        storage.save_work(work)

        s1 = Section(
            work_id=work.id,
            section_title="Introduction",
            section_order=0,
            text_content="This is the intro.",
        )
        s2 = Section(
            work_id=work.id,
            section_title="Methods",
            section_order=1,
            text_content="We used deep learning.",
        )
        storage.save_section(s1)
        storage.save_section(s2)

        sections = storage.get_sections_by_work(work.id)
        assert len(sections) == 2
        assert sections[0].section_title == "Introduction"
        assert sections[1].section_title == "Methods"


class TestThemeCRUD:
    def test_save_and_get_theme(self, storage):
        theme = Theme(
            document_text="Find papers on transformer architectures.",
            parsed_topics=["transformers", "attention"],
            parsed_methods=["self-attention", "positional encoding"],
            parsed_datasets=["WMT14", "WMT17"],
            parsed_queries=["transformer architecture", "attention mechanism"],
        )
        storage.save_theme(theme)

        fetched = storage.get_theme(theme.id)
        assert fetched is not None
        assert fetched.document_text == "Find papers on transformer architectures."
        assert fetched.parsed_topics == ["transformers", "attention"]
        assert fetched.parsed_methods == [
            "self-attention", "positional encoding"]
        assert fetched.parsed_datasets == ["WMT14", "WMT17"]
        assert fetched.parsed_queries == [
            "transformer architecture",
            "attention mechanism",
        ]

        assert storage.get_theme("nonexistent") is None


class TestJobCRUD:
    def test_save_and_get_job(self, storage):
        theme = Theme(document_text="test")
        storage.save_theme(theme)

        job = RetrievalJob(
            theme_id=theme.id,
            status=JobStatus.PENDING,
            query_count=5,
        )
        storage.save_job(job)

        fetched = storage.get_job(job.id)
        assert fetched is not None
        assert fetched.theme_id == theme.id
        assert fetched.status == JobStatus.PENDING
        assert fetched.query_count == 5

        assert storage.get_job("nonexistent") is None

    def test_update_job_status(self, storage):
        theme = Theme(document_text="test")
        storage.save_theme(theme)

        job = RetrievalJob(theme_id=theme.id)
        storage.save_job(job)

        storage.update_job_status(
            job.id,
            JobStatus.COMPLETED.value,
            candidate_count=100,
            result_count=50,
        )

        fetched = storage.get_job(job.id)
        assert fetched is not None
        assert fetched.status == JobStatus.COMPLETED
        assert fetched.candidate_count == 100
        assert fetched.result_count == 50

    def test_get_active_job_by_theme(self, storage):
        theme = Theme(document_text="job theme")
        storage.save_theme(theme)
        first = RetrievalJob(theme_id=theme.id, status=JobStatus.PENDING)
        storage.save_job(first)

        active = storage.get_active_job_by_theme(theme.id)
        assert active is not None
        assert active.id == first.id

        storage.update_job_status(first.id, JobStatus.COMPLETED.value)
        assert storage.get_active_job_by_theme(theme.id) is None

    def test_concurrent_job_creation_reuses_one_active_job(self, storage):
        theme = Theme(document_text="job theme")
        storage.save_theme(theme)

        from scholartrace.jobs.manager import JobManager

        manager = JobManager(storage)

        def _create() -> str:
            return manager.create_or_get_active_job(theme.id).id

        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = list(executor.map(lambda _: _create(), range(4)))

        assert len(set(ids)) == 1
        conn = storage._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE theme_id = ?",
            (theme.id,),
        ).fetchone()["c"]
        assert count == 1
