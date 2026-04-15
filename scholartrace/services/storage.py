from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from scholartrace.models.schemas import (
    Artifact,
    ArtifactKind,
    AccessStatus,
    JobStatus,
    RetrievalJob,
    Section,
    Theme,
    Work,
)


class StorageService:
    """SQLite-backed storage for ScholarTrace entities."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS works (
                id TEXT PRIMARY KEY,
                doi TEXT,
                arxiv_id TEXT,
                openalex_id TEXT,
                s2_id TEXT,
                dblp_key TEXT,
                openreview_id TEXT,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                abstract TEXT,
                relevance_score REAL DEFAULT 0,
                recency_score REAL DEFAULT 0,
                influence_score REAL DEFAULT 0,
                venue_score REAL DEFAULT 0,
                composite_score REAL DEFAULT 0,
                fulltext_available INTEGER DEFAULT 0,
                access_status TEXT DEFAULT 'unknown',
                source_provenance TEXT,
                citation_count INTEGER DEFAULT 0,
                reference_count INTEGER DEFAULT 0,
                pdf_url TEXT,
                html_url TEXT,
                oa_url TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_doi
                ON works(doi) WHERE doi IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_arxiv
                ON works(arxiv_id) WHERE arxiv_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_openalex
                ON works(openalex_id) WHERE openalex_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_s2
                ON works(s2_id) WHERE s2_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_dblp
                ON works(dblp_key) WHERE dblp_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                kind TEXT,
                source_url TEXT,
                local_path TEXT,
                sha256 TEXT,
                license TEXT,
                access_status TEXT DEFAULT 'unknown',
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sections (
                id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                artifact_id TEXT,
                section_title TEXT,
                section_order INTEGER,
                text_content TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS themes (
                id TEXT PRIMARY KEY,
                document_text TEXT,
                parsed_topics TEXT,
                parsed_methods TEXT,
                parsed_datasets TEXT,
                parsed_queries TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS theme_works (
                theme_id TEXT NOT NULL,
                work_id TEXT NOT NULL,
                rank_order INTEGER,
                PRIMARY KEY (theme_id, work_id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                theme_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                query_count INTEGER DEFAULT 0,
                candidate_count INTEGER DEFAULT 0,
                result_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT,
                updated_at TEXT,
                completed_at TEXT
            );
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_to_str(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        return dt.isoformat()

    @staticmethod
    def _str_to_dt(s: str | None) -> datetime | None:
        if s is None:
            return None
        return datetime.fromisoformat(s)

    @staticmethod
    def _list_to_json(lst: list | None) -> str | None:
        if lst is None:
            return None
        return json.dumps(lst)

    @staticmethod
    def _json_to_list(s: str | None) -> list:
        if s is None:
            return []
        return json.loads(s)

    # ------------------------------------------------------------------
    # Work CRUD
    # ------------------------------------------------------------------

    def _row_to_work(self, row: sqlite3.Row) -> Work:
        return Work(
            id=row["id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            openalex_id=row["openalex_id"],
            s2_id=row["s2_id"],
            dblp_key=row["dblp_key"],
            openreview_id=row["openreview_id"],
            title=row["title"],
            authors=self._json_to_list(row["authors"]),
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            relevance_score=row["relevance_score"] or 0.0,
            recency_score=row["recency_score"] or 0.0,
            influence_score=row["influence_score"] or 0.0,
            venue_score=row["venue_score"] or 0.0,
            composite_score=row["composite_score"] or 0.0,
            fulltext_available=bool(row["fulltext_available"]),
            access_status=AccessStatus(row["access_status"] or "unknown"),
            source_provenance=self._json_to_list(row["source_provenance"]),
            citation_count=row["citation_count"] or 0,
            reference_count=row["reference_count"] or 0,
            pdf_url=row["pdf_url"] if "pdf_url" in row.keys() else None,
            html_url=row["html_url"] if "html_url" in row.keys() else None,
            oa_url=row["oa_url"] if "oa_url" in row.keys() else None,
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=self._str_to_dt(row["updated_at"]) or datetime.utcnow(),
        )

    def save_work(self, work: Work) -> Work:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO works (
                id, doi, arxiv_id, openalex_id, s2_id, dblp_key, openreview_id,
                title, authors, year, venue, abstract,
                relevance_score, recency_score, influence_score, venue_score, composite_score,
                fulltext_available, access_status, source_provenance,
                citation_count, reference_count, pdf_url, html_url, oa_url,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work.id,
                work.doi,
                work.arxiv_id,
                work.openalex_id,
                work.s2_id,
                work.dblp_key,
                work.openreview_id,
                work.title,
                self._list_to_json(work.authors),
                work.year,
                work.venue,
                work.abstract,
                work.relevance_score,
                work.recency_score,
                work.influence_score,
                work.venue_score,
                work.composite_score,
                int(work.fulltext_available),
                work.access_status.value,
                self._list_to_json(work.source_provenance),
                work.citation_count,
                work.reference_count,
                work.pdf_url,
                work.html_url,
                work.oa_url,
                self._dt_to_str(work.created_at),
                self._dt_to_str(work.updated_at),
            ),
        )
        conn.commit()
        return work

    def get_work(self, work_id: str) -> Work | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    def get_work_by_doi(self, doi: str) -> Work | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM works WHERE doi = ?", (doi,)).fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    def get_work_by_arxiv_id(self, arxiv_id: str) -> Work | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM works WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    def get_work_by_s2_id(self, s2_id: str) -> Work | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM works WHERE s2_id = ?", (s2_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    def get_work_by_openalex_id(self, openalex_id: str) -> Work | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM works WHERE openalex_id = ?", (openalex_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    def list_works_by_theme(
        self, theme_id: str, limit: int = 50, offset: int = 0
    ) -> list[Work]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT w.* FROM works w
            JOIN theme_works tw ON w.id = tw.work_id
            WHERE tw.theme_id = ?
            ORDER BY tw.rank_order
            LIMIT ? OFFSET ?
            """,
            (theme_id, limit, offset),
        ).fetchall()
        return [self._row_to_work(r) for r in rows]

    def count_works_by_theme(self, theme_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM theme_works WHERE theme_id = ?", (theme_id,)
        ).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Artifact CRUD
    # ------------------------------------------------------------------

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            work_id=row["work_id"],
            kind=ArtifactKind(row["kind"] or "pdf"),
            source_url=row["source_url"],
            local_path=row["local_path"],
            sha256=row["sha256"],
            license=row["license"],
            access_status=AccessStatus(row["access_status"] or "unknown"),
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
        )

    def save_artifact(self, artifact: Artifact) -> Artifact:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                id, work_id, kind, source_url, local_path, sha256,
                license, access_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.work_id,
                artifact.kind.value,
                artifact.source_url,
                artifact.local_path,
                artifact.sha256,
                artifact.license,
                artifact.access_status.value,
                self._dt_to_str(artifact.created_at),
            ),
        )
        conn.commit()
        return artifact

    def get_artifacts_by_work(self, work_id: str) -> list[Artifact]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE work_id = ?", (work_id,)
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    # ------------------------------------------------------------------
    # Section CRUD
    # ------------------------------------------------------------------

    def _row_to_section(self, row: sqlite3.Row) -> Section:
        return Section(
            id=row["id"],
            work_id=row["work_id"],
            artifact_id=row["artifact_id"] or "",
            section_title=row["section_title"] or "",
            section_order=row["section_order"] or 0,
            text_content=row["text_content"] or "",
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
        )

    def save_section(self, section: Section) -> Section:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO sections (
                id, work_id, artifact_id, section_title, section_order,
                text_content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                section.id,
                section.work_id,
                section.artifact_id,
                section.section_title,
                section.section_order,
                section.text_content,
                self._dt_to_str(section.created_at),
            ),
        )
        conn.commit()
        return section

    def get_sections_by_work(self, work_id: str) -> list[Section]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sections WHERE work_id = ? ORDER BY section_order",
            (work_id,),
        ).fetchall()
        return [self._row_to_section(r) for r in rows]

    # ------------------------------------------------------------------
    # Theme CRUD
    # ------------------------------------------------------------------

    def _row_to_theme(self, row: sqlite3.Row) -> Theme:
        return Theme(
            id=row["id"],
            document_text=row["document_text"] or "",
            parsed_topics=self._json_to_list(row["parsed_topics"]),
            parsed_methods=self._json_to_list(row["parsed_methods"]),
            parsed_datasets=self._json_to_list(row["parsed_datasets"]),
            parsed_queries=self._json_to_list(row["parsed_queries"]),
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
        )

    def save_theme(self, theme: Theme) -> Theme:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO themes (
                id, document_text, parsed_topics, parsed_methods,
                parsed_datasets, parsed_queries, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                theme.id,
                theme.document_text,
                self._list_to_json(theme.parsed_topics),
                self._list_to_json(theme.parsed_methods),
                self._list_to_json(theme.parsed_datasets),
                self._list_to_json(theme.parsed_queries),
                self._dt_to_str(theme.created_at),
            ),
        )
        conn.commit()
        return theme

    def get_theme(self, theme_id: str) -> Theme | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM themes WHERE id = ?", (theme_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_theme(row)

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def _row_to_job(self, row: sqlite3.Row) -> RetrievalJob:
        return RetrievalJob(
            id=row["id"],
            theme_id=row["theme_id"],
            status=JobStatus(row["status"] or "pending"),
            query_count=row["query_count"] or 0,
            candidate_count=row["candidate_count"] or 0,
            result_count=row["result_count"] or 0,
            error_message=row["error_message"],
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=self._str_to_dt(row["updated_at"]) or datetime.utcnow(),
            completed_at=self._str_to_dt(row["completed_at"]),
        )

    def save_job(self, job: RetrievalJob) -> RetrievalJob:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                id, theme_id, status, query_count, candidate_count,
                result_count, error_message, created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.theme_id,
                job.status.value,
                job.query_count,
                job.candidate_count,
                job.result_count,
                job.error_message,
                self._dt_to_str(job.created_at),
                self._dt_to_str(job.updated_at),
                self._dt_to_str(job.completed_at),
            ),
        )
        conn.commit()
        return job

    def get_job(self, job_id: str) -> RetrievalJob | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def update_job_status(self, job_id: str, status: str, **kwargs) -> None:
        conn = self._get_conn()
        sets: list[str] = ["status = ?", "updated_at = ?"]
        vals: list = [status, self._dt_to_str(datetime.utcnow())]

        for key, value in kwargs.items():
            if key == "completed_at" and isinstance(value, datetime):
                sets.append(f"{key} = ?")
                vals.append(self._dt_to_str(value))
            else:
                sets.append(f"{key} = ?")
                vals.append(value)

        vals.append(job_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Theme-Work link
    # ------------------------------------------------------------------

    def link_theme_work(
        self, theme_id: str, work_id: str, rank_order: int
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO theme_works (theme_id, work_id, rank_order)
            VALUES (?, ?, ?)
            """,
            (theme_id, work_id, rank_order),
        )
        conn.commit()
