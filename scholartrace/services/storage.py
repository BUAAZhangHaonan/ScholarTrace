from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from scholartrace.models.schemas import (
    AccessStatus,
    AcquisitionState,
    Artifact,
    ArtifactKind,
    FullTextState,
    JobStatus,
    RetrievalJob,
    Section,
    Theme,
    Work,
)

logger = logging.getLogger(__name__)

WORK_IDENTIFIER_FIELDS = (
    "doi",
    "arxiv_id",
    "openalex_id",
    "s2_id",
    "dblp_key",
    "openreview_id",
)

ACCESS_STATUS_PRIORITY = {
    AccessStatus.UNKNOWN: 0,
    AccessStatus.PAYWALL: 1,
    AccessStatus.ABSTRACT_ONLY: 2,
    AccessStatus.AVAILABLE: 3,
}


class StorageService:
    """SQLite-backed storage for ScholarTrace entities."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Compatibility helper for tests and low-level inspection."""
        return self._connect()

    def close(self) -> None:
        """Compatibility no-op: storage now uses short-lived connections."""
        return None

    @contextmanager
    def transaction(self, *, immediate: bool = False):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def init_db(self) -> None:
        with self.transaction() as conn:
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
                    agent_score REAL DEFAULT 0,
                    agent_rank INTEGER,
                    agent_rationale TEXT,
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

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    work_id TEXT NOT NULL,
                    kind TEXT,
                    source_url TEXT,
                    local_path TEXT,
                    sha256 TEXT,
                    license TEXT,
                    access_status TEXT DEFAULT 'unknown',
                    created_at TEXT,
                    FOREIGN KEY(work_id) REFERENCES works(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sections (
                    id TEXT PRIMARY KEY,
                    work_id TEXT NOT NULL,
                    artifact_id TEXT,
                    section_title TEXT,
                    section_order INTEGER,
                    text_content TEXT,
                    created_at TEXT,
                    FOREIGN KEY(work_id) REFERENCES works(id) ON DELETE CASCADE,
                    FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
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
                    PRIMARY KEY (theme_id, work_id),
                    FOREIGN KEY(theme_id) REFERENCES themes(id) ON DELETE CASCADE,
                    FOREIGN KEY(work_id) REFERENCES works(id) ON DELETE CASCADE
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
                    completed_at TEXT,
                    FOREIGN KEY(theme_id) REFERENCES themes(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS fulltext_states (
                    work_id TEXT PRIMARY KEY,
                    acquisition_state TEXT DEFAULT 'missing',
                    last_attempt_at TEXT,
                    next_retry_at TEXT,
                    error_message TEXT,
                    updated_at TEXT,
                    FOREIGN KEY(work_id) REFERENCES works(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_work_id
                    ON artifacts(work_id);
                CREATE INDEX IF NOT EXISTS idx_sections_work_id
                    ON sections(work_id);
                CREATE INDEX IF NOT EXISTS idx_theme_works_theme
                    ON theme_works(theme_id, rank_order);
                CREATE INDEX IF NOT EXISTS idx_jobs_theme
                    ON jobs(theme_id);
                CREATE INDEX IF NOT EXISTS idx_fulltext_state_retry
                    ON fulltext_states(next_retry_at);
                """
            )
            self._ensure_work_columns(conn)
            self._ensure_indexes(conn)

    def _ensure_work_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(works)").fetchall()
        }
        additions = [
            ("agent_score", "ALTER TABLE works ADD COLUMN agent_score REAL DEFAULT 0"),
            ("agent_rank", "ALTER TABLE works ADD COLUMN agent_rank INTEGER"),
            ("agent_rationale", "ALTER TABLE works ADD COLUMN agent_rationale TEXT"),
        ]
        for column, statement in additions:
            if column not in columns:
                conn.execute(statement)

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        statements = [
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_doi
                ON works(doi) WHERE doi IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_arxiv
                ON works(arxiv_id) WHERE arxiv_id IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_openalex
                ON works(openalex_id) WHERE openalex_id IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_s2
                ON works(s2_id) WHERE s2_id IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_dblp
                ON works(dblp_key) WHERE dblp_key IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_works_openreview
                ON works(openreview_id) WHERE openreview_id IS NOT NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_theme
                ON jobs(theme_id) WHERE status IN ('pending', 'running')
            """,
        ]
        for statement in statements:
            try:
                conn.execute(statement)
            except sqlite3.IntegrityError:
                logger.warning("Skipped index creation due to existing duplicate rows")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dt_to_str(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt is not None else None

    @staticmethod
    def _str_to_dt(s: str | None) -> datetime | None:
        return datetime.fromisoformat(s) if s else None

    @staticmethod
    def _list_to_json(lst: list | None) -> str | None:
        return json.dumps(lst) if lst is not None else None

    @staticmethod
    def _json_to_list(s: str | None) -> list:
        return json.loads(s) if s else []

    @staticmethod
    def _merge_lists(left: list[str], right: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*left, *right]:
            if item and item not in merged:
                merged.append(item)
        return merged

    @staticmethod
    def _prefer_text(current: str | None, incoming: str | None) -> str | None:
        if incoming and incoming.strip():
            return incoming
        return current

    @staticmethod
    def _best_access_status(current: AccessStatus, incoming: AccessStatus) -> AccessStatus:
        if ACCESS_STATUS_PRIORITY[incoming] > ACCESS_STATUS_PRIORITY[current]:
            return incoming
        return current

    # ------------------------------------------------------------------
    # Row conversions
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
            title=row["title"] or "",
            authors=self._json_to_list(row["authors"]),
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            relevance_score=row["relevance_score"] or 0.0,
            recency_score=row["recency_score"] or 0.0,
            influence_score=row["influence_score"] or 0.0,
            venue_score=row["venue_score"] or 0.0,
            composite_score=row["composite_score"] or 0.0,
            agent_score=row["agent_score"] or 0.0,
            agent_rank=row["agent_rank"],
            agent_rationale=row["agent_rationale"],
            fulltext_available=bool(row["fulltext_available"]),
            access_status=AccessStatus(row["access_status"] or AccessStatus.UNKNOWN.value),
            source_provenance=self._json_to_list(row["source_provenance"]),
            citation_count=row["citation_count"] or 0,
            reference_count=row["reference_count"] or 0,
            pdf_url=row["pdf_url"],
            html_url=row["html_url"],
            oa_url=row["oa_url"],
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=self._str_to_dt(row["updated_at"]) or datetime.utcnow(),
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            work_id=row["work_id"],
            kind=ArtifactKind(row["kind"] or ArtifactKind.PDF.value),
            source_url=row["source_url"],
            local_path=row["local_path"],
            sha256=row["sha256"],
            license=row["license"],
            access_status=AccessStatus(row["access_status"] or AccessStatus.UNKNOWN.value),
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
        )

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

    def _row_to_job(self, row: sqlite3.Row) -> RetrievalJob:
        return RetrievalJob(
            id=row["id"],
            theme_id=row["theme_id"],
            status=JobStatus(row["status"] or JobStatus.PENDING.value),
            query_count=row["query_count"] or 0,
            candidate_count=row["candidate_count"] or 0,
            result_count=row["result_count"] or 0,
            error_message=row["error_message"],
            created_at=self._str_to_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=self._str_to_dt(row["updated_at"]) or datetime.utcnow(),
            completed_at=self._str_to_dt(row["completed_at"]),
        )

    def _row_to_fulltext_state(self, row: sqlite3.Row) -> FullTextState:
        return FullTextState(
            work_id=row["work_id"],
            acquisition_state=AcquisitionState(
                row["acquisition_state"] or AcquisitionState.MISSING.value
            ),
            last_attempt_at=self._str_to_dt(row["last_attempt_at"]),
            next_retry_at=self._str_to_dt(row["next_retry_at"]),
            error_message=row["error_message"],
            updated_at=self._str_to_dt(row["updated_at"]) or datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Work identity helpers
    # ------------------------------------------------------------------
    def _work_identifier_count(self, work: Work) -> int:
        return sum(1 for field in WORK_IDENTIFIER_FIELDS if getattr(work, field))

    def _find_work_ids_by_identifiers(self, conn: sqlite3.Connection, work: Work) -> set[str]:
        work_ids: set[str] = set()
        for field in WORK_IDENTIFIER_FIELDS:
            value = getattr(work, field)
            if not value:
                continue
            rows = conn.execute(
                f"SELECT id FROM works WHERE {field} = ?",
                (value,),
            ).fetchall()
            work_ids.update(row["id"] for row in rows)
        row = conn.execute("SELECT id FROM works WHERE id = ?", (work.id,)).fetchone()
        if row is not None:
            work_ids.add(row["id"])
        return work_ids

    def _count_work_usage(self, conn: sqlite3.Connection, work_id: str) -> int:
        theme_links = conn.execute(
            "SELECT COUNT(*) AS c FROM theme_works WHERE work_id = ?",
            (work_id,),
        ).fetchone()["c"]
        artifacts = conn.execute(
            "SELECT COUNT(*) AS c FROM artifacts WHERE work_id = ?",
            (work_id,),
        ).fetchone()["c"]
        sections = conn.execute(
            "SELECT COUNT(*) AS c FROM sections WHERE work_id = ?",
            (work_id,),
        ).fetchone()["c"]
        return int(theme_links) + int(artifacts) + int(sections)

    def _canonical_sort_key(self, conn: sqlite3.Connection, work: Work) -> tuple:
        return (
            -self._count_work_usage(conn, work.id),
            -self._work_identifier_count(work),
            self._dt_to_str(work.created_at) or "",
            work.id,
        )

    def _choose_canonical_work(self, conn: sqlite3.Connection, works: list[Work]) -> Work:
        return min(works, key=lambda work: self._canonical_sort_key(conn, work))

    def _merge_work_models(self, current: Work, incoming: Work) -> Work:
        merged = current.model_copy(deep=True)
        for field in WORK_IDENTIFIER_FIELDS:
            setattr(merged, field, getattr(incoming, field) or getattr(merged, field))
        merged.title = self._prefer_text(merged.title, incoming.title) or ""
        merged.authors = self._merge_lists(merged.authors, incoming.authors)
        merged.year = incoming.year or merged.year
        merged.venue = self._prefer_text(merged.venue, incoming.venue)
        if incoming.abstract and (
            not merged.abstract or len(incoming.abstract) > len(merged.abstract)
        ):
            merged.abstract = incoming.abstract
        merged.relevance_score = incoming.relevance_score or merged.relevance_score
        merged.recency_score = incoming.recency_score or merged.recency_score
        merged.influence_score = incoming.influence_score or merged.influence_score
        merged.venue_score = incoming.venue_score or merged.venue_score
        merged.composite_score = incoming.composite_score or merged.composite_score
        merged.agent_score = incoming.agent_score or merged.agent_score
        merged.agent_rank = incoming.agent_rank if incoming.agent_rank is not None else merged.agent_rank
        merged.agent_rationale = incoming.agent_rationale or merged.agent_rationale
        merged.fulltext_available = merged.fulltext_available or incoming.fulltext_available
        merged.access_status = self._best_access_status(merged.access_status, incoming.access_status)
        merged.source_provenance = self._merge_lists(
            merged.source_provenance,
            incoming.source_provenance,
        )
        merged.citation_count = max(merged.citation_count, incoming.citation_count)
        merged.reference_count = max(merged.reference_count, incoming.reference_count)
        merged.pdf_url = incoming.pdf_url or merged.pdf_url
        merged.html_url = incoming.html_url or merged.html_url
        merged.oa_url = incoming.oa_url or merged.oa_url
        merged.created_at = current.created_at
        merged.updated_at = datetime.utcnow()
        return merged

    def _write_work_row(self, conn: sqlite3.Connection, work: Work) -> Work:
        conn.execute(
            """
            INSERT INTO works (
                id, doi, arxiv_id, openalex_id, s2_id, dblp_key, openreview_id,
                title, authors, year, venue, abstract,
                relevance_score, recency_score, influence_score, venue_score, composite_score,
                agent_score, agent_rank, agent_rationale,
                fulltext_available, access_status, source_provenance,
                citation_count, reference_count, pdf_url, html_url, oa_url,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                doi = excluded.doi,
                arxiv_id = excluded.arxiv_id,
                openalex_id = excluded.openalex_id,
                s2_id = excluded.s2_id,
                dblp_key = excluded.dblp_key,
                openreview_id = excluded.openreview_id,
                title = excluded.title,
                authors = excluded.authors,
                year = excluded.year,
                venue = excluded.venue,
                abstract = excluded.abstract,
                relevance_score = excluded.relevance_score,
                recency_score = excluded.recency_score,
                influence_score = excluded.influence_score,
                venue_score = excluded.venue_score,
                composite_score = excluded.composite_score,
                agent_score = excluded.agent_score,
                agent_rank = excluded.agent_rank,
                agent_rationale = excluded.agent_rationale,
                fulltext_available = excluded.fulltext_available,
                access_status = excluded.access_status,
                source_provenance = excluded.source_provenance,
                citation_count = excluded.citation_count,
                reference_count = excluded.reference_count,
                pdf_url = excluded.pdf_url,
                html_url = excluded.html_url,
                oa_url = excluded.oa_url,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
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
                work.agent_score,
                work.agent_rank,
                work.agent_rationale,
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
        return work

    def _move_work_associations(
        self,
        conn: sqlite3.Connection,
        from_work_id: str,
        to_work_id: str,
    ) -> dict[str, int]:
        moved = {
            "theme_links": 0,
            "artifacts": 0,
            "sections": 0,
        }
        theme_rows = conn.execute(
            "SELECT theme_id, rank_order FROM theme_works WHERE work_id = ?",
            (from_work_id,),
        ).fetchall()
        for row in theme_rows:
            self.link_theme_work(
                row["theme_id"],
                to_work_id,
                row["rank_order"],
                conn=conn,
            )
            conn.execute(
                "DELETE FROM theme_works WHERE theme_id = ? AND work_id = ?",
                (row["theme_id"], from_work_id),
            )
            moved["theme_links"] += 1

        artifacts = conn.execute(
            "UPDATE artifacts SET work_id = ? WHERE work_id = ?",
            (to_work_id, from_work_id),
        )
        moved["artifacts"] = artifacts.rowcount or 0

        sections = conn.execute(
            "UPDATE sections SET work_id = ? WHERE work_id = ?",
            (to_work_id, from_work_id),
        )
        moved["sections"] = sections.rowcount or 0
        return moved

    def _dedupe_artifacts_for_work(self, conn: sqlite3.Connection, work_id: str) -> int:
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE work_id = ?
            ORDER BY created_at, id
            """,
            (work_id,),
        ).fetchall()
        seen: dict[tuple, str] = {}
        removed = 0
        for row in rows:
            key = (
                row["kind"] or "",
                row["source_url"] or "",
                row["local_path"] or "",
                row["sha256"] or "",
            )
            if key not in seen:
                seen[key] = row["id"]
                continue
            canonical_id = seen[key]
            conn.execute(
                """
                UPDATE sections
                SET artifact_id = ?
                WHERE artifact_id = ?
                """,
                (canonical_id, row["id"]),
            )
            conn.execute("DELETE FROM artifacts WHERE id = ?", (row["id"],))
            removed += 1
        return removed

    def _dedupe_sections_for_work(self, conn: sqlite3.Connection, work_id: str) -> int:
        rows = conn.execute(
            """
            SELECT * FROM sections
            WHERE work_id = ?
            ORDER BY created_at, id
            """,
            (work_id,),
        ).fetchall()
        seen: set[tuple] = set()
        removed = 0
        for row in rows:
            key = (
                row["artifact_id"] or "",
                row["section_title"] or "",
                row["section_order"] or 0,
                row["text_content"] or "",
            )
            if key in seen:
                conn.execute("DELETE FROM sections WHERE id = ?", (row["id"],))
                removed += 1
                continue
            seen.add(key)
        return removed

    def _repair_work_cluster(
        self,
        conn: sqlite3.Connection,
        work_ids: list[str],
    ) -> dict[str, int]:
        rows = [
            conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone()
            for work_id in work_ids
        ]
        works = [self._row_to_work(row) for row in rows if row is not None]
        if len(works) <= 1:
            return {
                "works_removed": 0,
                "theme_links_rewired": 0,
                "artifacts_rewired": 0,
                "sections_rewired": 0,
                "artifacts_deduped": 0,
                "sections_deduped": 0,
            }

        canonical = self._choose_canonical_work(conn, works)
        merged = canonical.model_copy(deep=True)
        report = {
            "works_removed": 0,
            "theme_links_rewired": 0,
            "artifacts_rewired": 0,
            "sections_rewired": 0,
            "artifacts_deduped": 0,
            "sections_deduped": 0,
        }

        for work in works:
            merged = self._merge_work_models(merged, work)

        for work in works:
            if work.id == canonical.id:
                continue
            moved = self._move_work_associations(conn, work.id, canonical.id)
            report["theme_links_rewired"] += moved["theme_links"]
            report["artifacts_rewired"] += moved["artifacts"]
            report["sections_rewired"] += moved["sections"]
            conn.execute("DELETE FROM works WHERE id = ?", (work.id,))
            report["works_removed"] += 1

        self._write_work_row(conn, merged)
        report["artifacts_deduped"] += self._dedupe_artifacts_for_work(conn, canonical.id)
        report["sections_deduped"] += self._dedupe_sections_for_work(conn, canonical.id)
        return report

    def _find_duplicate_work_clusters(self, conn: sqlite3.Connection) -> list[list[str]]:
        rows = conn.execute(
            """
            SELECT id, doi, arxiv_id, openalex_id, s2_id, dblp_key, openreview_id
            FROM works
            """
        ).fetchall()
        parent = {row["id"]: row["id"] for row in rows}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        identifier_map: dict[tuple[str, str], str] = {}
        for row in rows:
            for field in WORK_IDENTIFIER_FIELDS:
                value = row[field]
                if not value:
                    continue
                key = (field, value)
                seen = identifier_map.get(key)
                if seen is None:
                    identifier_map[key] = row["id"]
                    continue
                union(seen, row["id"])

        groups: dict[str, list[str]] = {}
        for row in rows:
            root = find(row["id"])
            groups.setdefault(root, []).append(row["id"])
        return [ids for ids in groups.values() if len(ids) > 1]

    def _cleanup_orphan_rows(self, conn: sqlite3.Connection) -> int:
        total_removed = 0
        deletes = [
            """
            DELETE FROM theme_works
            WHERE theme_id NOT IN (SELECT id FROM themes)
               OR work_id NOT IN (SELECT id FROM works)
            """,
            """
            DELETE FROM artifacts
            WHERE work_id NOT IN (SELECT id FROM works)
            """,
            """
            DELETE FROM sections
            WHERE work_id NOT IN (SELECT id FROM works)
            """,
            """
            DELETE FROM jobs
            WHERE theme_id NOT IN (SELECT id FROM themes)
            """,
            """
            DELETE FROM fulltext_states
            WHERE work_id NOT IN (SELECT id FROM works)
            """,
        ]
        for statement in deletes:
            total_removed += conn.execute(statement).rowcount or 0
        conn.execute(
            """
            UPDATE sections
            SET artifact_id = ''
            WHERE artifact_id != ''
              AND artifact_id NOT IN (SELECT id FROM artifacts)
            """
        )
        return total_removed

    # ------------------------------------------------------------------
    # Work CRUD
    # ------------------------------------------------------------------
    def save_work(self, work: Work, conn: sqlite3.Connection | None = None) -> Work:
        if conn is None:
            with self.transaction(immediate=True) as tx:
                return self.save_work(work, conn=tx)

        existing_ids = self._find_work_ids_by_identifiers(conn, work)
        if not existing_ids:
            work.updated_at = datetime.utcnow()
            self._write_work_row(conn, work)
            return work

        existing_rows = [
            conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone()
            for work_id in existing_ids
        ]
        existing_works = [self._row_to_work(row) for row in existing_rows if row is not None]
        canonical = self._choose_canonical_work(conn, existing_works)
        merged = canonical.model_copy(deep=True)
        for existing in existing_works:
            merged = self._merge_work_models(merged, existing)
        merged = self._merge_work_models(merged, work)
        merged.id = canonical.id
        merged.created_at = canonical.created_at

        duplicates = [existing for existing in existing_works if existing.id != canonical.id]
        for duplicate in duplicates:
            self._move_work_associations(conn, duplicate.id, canonical.id)
            conn.execute("DELETE FROM works WHERE id = ?", (duplicate.id,))

        self._write_work_row(conn, merged)
        self._dedupe_artifacts_for_work(conn, canonical.id)
        self._dedupe_sections_for_work(conn, canonical.id)
        return merged

    def get_work(self, work_id: str) -> Work | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone()
            return self._row_to_work(row) if row is not None else None
        finally:
            conn.close()

    def _get_work_by_field(self, field: str, value: str) -> Work | None:
        conn = self._connect()
        try:
            row = conn.execute(f"SELECT * FROM works WHERE {field} = ?", (value,)).fetchone()
            return self._row_to_work(row) if row is not None else None
        finally:
            conn.close()

    def get_work_by_doi(self, doi: str) -> Work | None:
        return self._get_work_by_field("doi", doi)

    def get_work_by_arxiv_id(self, arxiv_id: str) -> Work | None:
        return self._get_work_by_field("arxiv_id", arxiv_id)

    def get_work_by_s2_id(self, s2_id: str) -> Work | None:
        return self._get_work_by_field("s2_id", s2_id)

    def get_work_by_openalex_id(self, openalex_id: str) -> Work | None:
        return self._get_work_by_field("openalex_id", openalex_id)

    def list_works_by_theme(self, theme_id: str, limit: int = 50, offset: int = 0) -> list[Work]:
        conn = self._connect()
        try:
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
            return [self._row_to_work(row) for row in rows]
        finally:
            conn.close()

    def count_works_by_theme(self, theme_id: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM theme_works WHERE theme_id = ?",
                (theme_id,),
            ).fetchone()
            return int(row["c"])
        finally:
            conn.close()

    def replace_theme_results(self, theme_id: str, works: list[Work]) -> list[Work]:
        with self.transaction(immediate=True) as conn:
            saved_works = [self.save_work(work, conn=conn) for work in works]
            conn.execute("DELETE FROM theme_works WHERE theme_id = ?", (theme_id,))
            for rank_order, work in enumerate(saved_works, start=1):
                self.link_theme_work(theme_id, work.id, rank_order, conn=conn)
            return saved_works

    # ------------------------------------------------------------------
    # Artifact CRUD
    # ------------------------------------------------------------------
    def save_artifact(self, artifact: Artifact, conn: sqlite3.Connection | None = None) -> Artifact:
        if conn is None:
            with self.transaction() as tx:
                return self.save_artifact(artifact, conn=tx)

        conn.execute(
            """
            INSERT INTO artifacts (
                id, work_id, kind, source_url, local_path, sha256,
                license, access_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                work_id = excluded.work_id,
                kind = excluded.kind,
                source_url = excluded.source_url,
                local_path = excluded.local_path,
                sha256 = excluded.sha256,
                license = excluded.license,
                access_status = excluded.access_status,
                created_at = excluded.created_at
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
        return artifact

    def get_artifacts_by_work(self, work_id: str) -> list[Artifact]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM artifacts WHERE work_id = ?", (work_id,)).fetchall()
            return [self._row_to_artifact(row) for row in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Section CRUD
    # ------------------------------------------------------------------
    def save_section(self, section: Section, conn: sqlite3.Connection | None = None) -> Section:
        if conn is None:
            with self.transaction() as tx:
                return self.save_section(section, conn=tx)

        conn.execute(
            """
            INSERT INTO sections (
                id, work_id, artifact_id, section_title, section_order,
                text_content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                work_id = excluded.work_id,
                artifact_id = excluded.artifact_id,
                section_title = excluded.section_title,
                section_order = excluded.section_order,
                text_content = excluded.text_content,
                created_at = excluded.created_at
            """,
            (
                section.id,
                section.work_id,
                section.artifact_id or None,
                section.section_title,
                section.section_order,
                section.text_content,
                self._dt_to_str(section.created_at),
            ),
        )
        return section

    def get_sections_by_work(self, work_id: str) -> list[Section]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM sections WHERE work_id = ? ORDER BY section_order",
                (work_id,),
            ).fetchall()
            return [self._row_to_section(row) for row in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Theme CRUD
    # ------------------------------------------------------------------
    def save_theme(self, theme: Theme, conn: sqlite3.Connection | None = None) -> Theme:
        if conn is None:
            with self.transaction() as tx:
                return self.save_theme(theme, conn=tx)

        conn.execute(
            """
            INSERT INTO themes (
                id, document_text, parsed_topics, parsed_methods,
                parsed_datasets, parsed_queries, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_text = excluded.document_text,
                parsed_topics = excluded.parsed_topics,
                parsed_methods = excluded.parsed_methods,
                parsed_datasets = excluded.parsed_datasets,
                parsed_queries = excluded.parsed_queries,
                created_at = excluded.created_at
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
        return theme

    def get_theme(self, theme_id: str) -> Theme | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM themes WHERE id = ?", (theme_id,)).fetchone()
            return self._row_to_theme(row) if row is not None else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------
    def save_job(self, job: RetrievalJob, conn: sqlite3.Connection | None = None) -> RetrievalJob:
        if conn is None:
            with self.transaction(immediate=True) as tx:
                return self.save_job(job, conn=tx)

        conn.execute(
            """
            INSERT INTO jobs (
                id, theme_id, status, query_count, candidate_count,
                result_count, error_message, created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                theme_id = excluded.theme_id,
                status = excluded.status,
                query_count = excluded.query_count,
                candidate_count = excluded.candidate_count,
                result_count = excluded.result_count,
                error_message = excluded.error_message,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                completed_at = excluded.completed_at
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
        return job

    def get_job(self, job_id: str) -> RetrievalJob | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row is not None else None
        finally:
            conn.close()

    def get_active_job_by_theme(self, theme_id: str) -> RetrievalJob | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE theme_id = ?
                  AND status IN (?, ?)
                ORDER BY created_at
                LIMIT 1
                """,
                (
                    theme_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            ).fetchone()
            return self._row_to_job(row) if row is not None else None
        finally:
            conn.close()

    def count_active_jobs(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM jobs
                WHERE status IN (?, ?)
                """,
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            ).fetchone()
            return int(row["c"])
        finally:
            conn.close()

    def update_job_status(self, job_id: str, status: JobStatus | str, **kwargs) -> None:
        normalized_status = status.value if isinstance(status, JobStatus) else status
        with self.transaction(immediate=True) as conn:
            sets: list[str] = ["status = ?", "updated_at = ?"]
            vals: list = [normalized_status, self._dt_to_str(datetime.utcnow())]
            for key, value in kwargs.items():
                sets.append(f"{key} = ?")
                if key == "completed_at" and isinstance(value, datetime):
                    vals.append(self._dt_to_str(value))
                else:
                    vals.append(value)
            vals.append(job_id)
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                vals,
            )

    # ------------------------------------------------------------------
    # Full-text state
    # ------------------------------------------------------------------
    def save_fulltext_state(
        self,
        state: FullTextState,
        conn: sqlite3.Connection | None = None,
    ) -> FullTextState:
        if conn is None:
            with self.transaction() as tx:
                return self.save_fulltext_state(state, conn=tx)

        conn.execute(
            """
            INSERT INTO fulltext_states (
                work_id, acquisition_state, last_attempt_at, next_retry_at,
                error_message, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(work_id) DO UPDATE SET
                acquisition_state = excluded.acquisition_state,
                last_attempt_at = excluded.last_attempt_at,
                next_retry_at = excluded.next_retry_at,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                state.work_id,
                state.acquisition_state.value,
                self._dt_to_str(state.last_attempt_at),
                self._dt_to_str(state.next_retry_at),
                state.error_message,
                self._dt_to_str(state.updated_at),
            ),
        )
        return state

    def get_fulltext_state(self, work_id: str) -> FullTextState | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM fulltext_states WHERE work_id = ?",
                (work_id,),
            ).fetchone()
            return self._row_to_fulltext_state(row) if row is not None else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Theme-Work link
    # ------------------------------------------------------------------
    def link_theme_work(
        self,
        theme_id: str,
        work_id: str,
        rank_order: int,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is None:
            with self.transaction() as tx:
                self.link_theme_work(theme_id, work_id, rank_order, conn=tx)
                return

        conn.execute(
            """
            INSERT INTO theme_works (theme_id, work_id, rank_order)
            VALUES (?, ?, ?)
            ON CONFLICT(theme_id, work_id) DO UPDATE SET
                rank_order = CASE
                    WHEN theme_works.rank_order IS NULL THEN excluded.rank_order
                    WHEN excluded.rank_order IS NULL THEN theme_works.rank_order
                    WHEN excluded.rank_order < theme_works.rank_order THEN excluded.rank_order
                    ELSE theme_works.rank_order
                END
            """,
            (theme_id, work_id, rank_order),
        )

    # ------------------------------------------------------------------
    # Repair / migration helpers
    # ------------------------------------------------------------------
    def repair_existing_work_state(self, *, apply: bool = False) -> dict[str, int]:
        conn = self._connect()
        report = {
            "clusters_repaired": 0,
            "works_removed": 0,
            "theme_links_rewired": 0,
            "artifacts_rewired": 0,
            "sections_rewired": 0,
            "artifacts_deduped": 0,
            "sections_deduped": 0,
            "orphan_rows_removed": 0,
        }
        try:
            conn.execute("BEGIN IMMEDIATE")
            for cluster in self._find_duplicate_work_clusters(conn):
                cluster_report = self._repair_work_cluster(conn, cluster)
                report["clusters_repaired"] += 1
                for key, value in cluster_report.items():
                    report[key] += value
            report["orphan_rows_removed"] += self._cleanup_orphan_rows(conn)
            self._ensure_indexes(conn)
            if apply:
                conn.commit()
                logger.info("Repaired existing ScholarTrace storage state: %s", report)
            else:
                conn.rollback()
            return report
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
