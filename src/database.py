"""SQLite storage for tracking courses and lectures."""

import os
import sqlite3
from datetime import datetime

from . import config


class Database:
    """SQLite database for course and lecture tracking."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS courses (
                    course_id TEXT PRIMARY KEY,
                    title TEXT,
                    teacher TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS lectures (
                    sub_id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    sub_title TEXT,
                    date TEXT,
                    transcript TEXT,
                    summary TEXT,
                    processed_at TEXT,
                    emailed_at TEXT,
                    FOREIGN KEY (course_id) REFERENCES courses(course_id)
                )
            """)
            # Migrate: add error tracking and summary_model columns
            existing = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(lectures)").fetchall()
            }
            for col, typedef in [
                ("error_msg", "TEXT"),
                ("error_count", "INTEGER DEFAULT 0"),
                ("error_stage", "TEXT"),
                ("summary_model", "TEXT"),
                ("summary_format_version", "INTEGER DEFAULT 0"),
            ]:
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE lectures ADD COLUMN {col} {typedef}")

            # Per-page OCR results (child table for resumability + concurrency).
            # ocr_status: 'pending' | 'done' | 'failed'
            # text is NULL until OCR completes; filtered pages stay 'done' but
            # the in-memory dedup/desktop filter excludes them at prompt-build time.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ppt_pages (
                    sub_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    created_sec INTEGER NOT NULL,
                    pptimgurl TEXT,
                    text TEXT,
                    ocr_status TEXT NOT NULL DEFAULT 'pending',
                    ocr_at TEXT,
                    PRIMARY KEY (sub_id, page_num),
                    FOREIGN KEY (sub_id) REFERENCES lectures(sub_id)
                )
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ppt_pages_sub_status "
                "ON ppt_pages(sub_id, ocr_status)"
            )

    def upsert_course(self, course_id: str, title: str, teacher: str):
        with self.conn:
            self.conn.execute(
                """INSERT INTO courses (course_id, title, teacher)
                   VALUES (?, ?, ?)
                   ON CONFLICT(course_id) DO UPDATE SET
                       title=excluded.title, teacher=excluded.teacher""",
                (course_id, title, teacher),
            )

    def insert_lecture(
        self, sub_id: str, course_id: str, sub_title: str, date: str
    ) -> bool:
        """Insert a new lecture. Returns True if inserted, False if already exists."""
        try:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO lectures (sub_id, course_id, sub_title, date)
                       VALUES (?, ?, ?, ?)""",
                    (sub_id, course_id, sub_title, date),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_processed_sub_ids(self, course_id: str) -> set[str]:
        """Return sub_ids that have been fully processed."""
        rows = self.conn.execute(
            "SELECT sub_id FROM lectures WHERE course_id = ? AND processed_at IS NOT NULL",
            (course_id,),
        ).fetchall()
        return {row["sub_id"] for row in rows}

    def get_unprocessed_lectures(self, course_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM lectures WHERE processed_at IS NULL"
        params = ()
        if course_id:
            query += " AND course_id = ?"
            params = (course_id,)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_transcript(self, sub_id: str, transcript: str):
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET transcript = ? WHERE sub_id = ?",
                (transcript, sub_id),
            )

    def update_summary(self, sub_id: str, summary: str):
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET summary = ? WHERE sub_id = ?",
                (summary, sub_id),
            )

    def mark_processed(self, sub_id: str):
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET processed_at = ? WHERE sub_id = ?",
                (datetime.now().isoformat(), sub_id),
            )

    def mark_emailed(self, sub_id: str):
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET emailed_at = ? WHERE sub_id = ?",
                (datetime.now().isoformat(), sub_id),
            )

    def mark_emailed_batch(self, sub_ids: list[str]):
        """Mark multiple lectures as emailed in a single transaction."""
        if not sub_ids:
            return
        now = datetime.now().isoformat()
        with self.conn:
            self.conn.executemany(
                "UPDATE lectures SET emailed_at = ? WHERE sub_id = ?",
                [(now, sid) for sid in sub_ids],
            )

    def update_error(self, sub_id: str, stage: str, error_msg: str):
        """Record a processing error for a lecture."""
        with self.conn:
            self.conn.execute(
                """UPDATE lectures
                   SET error_stage = ?, error_msg = ?,
                       error_count = COALESCE(error_count, 0) + 1
                   WHERE sub_id = ?""",
                (stage, error_msg, sub_id),
            )

    def clear_error(self, sub_id: str):
        """Clear error state after successful processing."""
        with self.conn:
            self.conn.execute(
                """UPDATE lectures
                   SET error_stage = NULL, error_msg = NULL, error_count = 0
                   WHERE sub_id = ?""",
                (sub_id,),
            )

    def update_summary_with_model(self, sub_id: str, summary: str, model: str):
        """Save summary and the model that produced it."""
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET summary = ?, summary_model = ? WHERE sub_id = ?",
                (summary, model, sub_id),
            )

    def update_ppt_page(self, sub_id: str, page_num: int,
                        text: str | None, status: str):
        """Mark a page's OCR result. status: 'done' | 'failed'."""
        with self.conn:
            self.conn.execute(
                """UPDATE ppt_pages
                   SET text = ?, ocr_status = ?, ocr_at = ?
                   WHERE sub_id = ? AND page_num = ?""",
                (text, status, datetime.now().isoformat(), sub_id, page_num),
            )

    def insert_ppt_pages_pending(self, sub_id: str, items: list[dict]) -> int:
        """Bulk-insert PPT page rows with status='pending'.

        items: list of {page_num, created_sec, pptimgurl}.
        Existing rows are left untouched (INSERT OR IGNORE), so this is safe
        to call repeatedly across reruns and across concurrent workers.
        Returns number of newly inserted rows.
        """
        if not items:
            return 0
        with self.conn:
            cur = self.conn.executemany(
                """INSERT OR IGNORE INTO ppt_pages
                       (sub_id, page_num, created_sec, pptimgurl, ocr_status)
                   VALUES (?, ?, ?, ?, 'pending')""",
                [
                    (sub_id, int(it["page_num"]), int(it["created_sec"]),
                     it.get("pptimgurl"))
                    for it in items
                ],
            )
            return cur.rowcount or 0

    def get_pending_ppt_pages(self, sub_id: str) -> list[dict]:
        """Pages still awaiting OCR. Workers claim via update_ppt_page."""
        rows = self.conn.execute(
            """SELECT page_num, created_sec, pptimgurl
               FROM ppt_pages
               WHERE sub_id = ? AND ocr_status = 'pending'
               ORDER BY created_sec""",
            (sub_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_done_ppt_pages(self, sub_id: str) -> list[dict]:
        """Successfully-OCR'd pages, sorted by time. Used by the bucketer."""
        rows = self.conn.execute(
            """SELECT page_num, created_sec, text
               FROM ppt_pages
               WHERE sub_id = ? AND ocr_status = 'done'
                 AND text IS NOT NULL AND text != ''
               ORDER BY created_sec""",
            (sub_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_pending_ppt_pages(self, sub_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM ppt_pages "
            "WHERE sub_id = ? AND ocr_status = 'pending'",
            (sub_id,),
        ).fetchone()[0]

    def count_total_ppt_pages(self, sub_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM ppt_pages WHERE sub_id = ?",
            (sub_id,),
        ).fetchone()[0]

    def update_summary_v2(self, sub_id: str, summary: str, model: str):
        """Save summary, model, AND mark format version = 1 (PPT-aware)."""
        with self.conn:
            self.conn.execute(
                """UPDATE lectures
                   SET summary = ?, summary_model = ?, summary_format_version = 1
                   WHERE sub_id = ?""",
                (summary, model, sub_id),
            )

    def reset_emailed(self, sub_id: str):
        """Clear emailed_at so a re-summarized lecture re-sends on next run."""
        with self.conn:
            self.conn.execute(
                "UPDATE lectures SET emailed_at = NULL WHERE sub_id = ?",
                (sub_id,),
            )

    def get_lectures_to_resummarize(self) -> list[dict]:
        """Old lectures with summary but missing v2 PPT-aware format.

        Triggers re-OCR + re-summarize (flat mode, since old transcripts
        have no in-memory segment timestamps).
        """
        rows = self.conn.execute(
            """SELECT l.*, c.title AS course_title, c.teacher
               FROM lectures l
               JOIN courses c ON l.course_id = c.course_id
               WHERE l.summary IS NOT NULL
                 AND COALESCE(l.summary_format_version, 0) = 0"""
        ).fetchall()
        return [dict(row) for row in rows]

    def get_lecture(self, sub_id: str) -> dict | None:
        """Get a single lecture row by sub_id."""
        row = self.conn.execute(
            "SELECT * FROM lectures WHERE sub_id = ?", (sub_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_unsent_lectures(self) -> list[dict]:
        """Find lectures that are processed but not yet emailed."""
        rows = self.conn.execute(
            """SELECT l.*, c.title AS course_title, c.teacher
               FROM lectures l
               JOIN courses c ON l.course_id = c.course_id
               WHERE l.processed_at IS NOT NULL
                 AND l.emailed_at IS NULL
                 AND l.summary IS NOT NULL""",
        ).fetchall()
        return [dict(row) for row in rows]
