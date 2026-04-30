#!/usr/bin/env python3
"""Merge local DB into remote DB (additive-only).

Used at deploy time to safely combine results from concurrent workflow runs.
For each lecture row, fields only progress forward (null -> non-null).
"""

import sqlite3
import sys


def _ensure_schema(conn: sqlite3.Connection):
    """Create tables and migration columns if missing in remote DB."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            course_id TEXT PRIMARY KEY, title TEXT, teacher TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lectures (
            sub_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            sub_title TEXT, date TEXT,
            transcript TEXT, summary TEXT,
            processed_at TEXT, emailed_at TEXT,
            FOREIGN KEY (course_id) REFERENCES courses(course_id)
        )
    """)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(lectures)")}
    for col, typedef in [
        ("error_msg", "TEXT"),
        ("error_count", "INTEGER DEFAULT 0"),
        ("error_stage", "TEXT"),
        ("summary_model", "TEXT"),
        ("summary_format_version", "INTEGER DEFAULT 0"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE lectures ADD COLUMN {col} {typedef}")

    conn.execute("""
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ppt_pages_sub_status "
        "ON ppt_pages(sub_id, ocr_status)"
    )


def merge(local_path: str, remote_path: str):
    """Merge local changes into remote DB.  Only adds/progresses, never deletes."""
    conn = sqlite3.connect(remote_path)
    _ensure_schema(conn)
    conn.execute("ATTACH DATABASE ? AS local", (local_path,))

    with conn:
        # 1) Courses: upsert
        conn.execute("""
            INSERT OR REPLACE INTO main.courses (course_id, title, teacher)
            SELECT course_id, title, teacher FROM local.courses
        """)

        # 2) Lectures: insert rows that only exist in local
        conn.execute("""
            INSERT OR IGNORE INTO main.lectures
                (sub_id, course_id, sub_title, date, transcript, summary,
                 processed_at, emailed_at, error_msg, error_count, error_stage,
                 summary_model, summary_format_version)
            SELECT sub_id, course_id, sub_title, date, transcript, summary,
                   processed_at, emailed_at, error_msg, error_count, error_stage,
                   summary_model, summary_format_version
            FROM local.lectures
        """)

        # 3) Lectures: merge existing rows (progress forward only)
        #    - Progress fields: COALESCE(local, remote) — prefer non-null
        #    - Error fields: clear if processed, otherwise keep the most info
        #    - summary_format_version: take MAX so v2 wins
        conn.execute("""
            UPDATE main.lectures SET
                transcript    = COALESCE(l.transcript,    main.lectures.transcript),
                summary       = COALESCE(l.summary,       main.lectures.summary),
                summary_model = COALESCE(l.summary_model, main.lectures.summary_model),
                summary_format_version = MAX(
                    COALESCE(l.summary_format_version, 0),
                    COALESCE(main.lectures.summary_format_version, 0)
                ),
                processed_at  = COALESCE(l.processed_at,  main.lectures.processed_at),
                emailed_at    = COALESCE(l.emailed_at,    main.lectures.emailed_at),
                error_msg = CASE
                    WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                    THEN NULL
                    ELSE COALESCE(l.error_msg, main.lectures.error_msg)
                END,
                error_count = CASE
                    WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                    THEN 0
                    ELSE MAX(COALESCE(l.error_count, 0), COALESCE(main.lectures.error_count, 0))
                END,
                error_stage = CASE
                    WHEN COALESCE(l.processed_at, main.lectures.processed_at) IS NOT NULL
                    THEN NULL
                    ELSE COALESCE(l.error_stage, main.lectures.error_stage)
                END
            FROM local.lectures l
            WHERE main.lectures.sub_id = l.sub_id
        """)

        # 4) PPT pages: insert local-only rows
        conn.execute("""
            INSERT OR IGNORE INTO main.ppt_pages
                (sub_id, page_num, created_sec, pptimgurl, text, ocr_status, ocr_at)
            SELECT sub_id, page_num, created_sec, pptimgurl, text, ocr_status, ocr_at
            FROM local.ppt_pages
        """)

        # 5) PPT pages: merge existing rows.
        #    Status priority: 'done' > 'failed' > 'pending'.  Text wins if non-null
        #    on either side (a 'done' row's text is preferred, but COALESCE handles
        #    the rare case of done-without-text gracefully).
        conn.execute("""
            UPDATE main.ppt_pages SET
                text = COALESCE(l.text, main.ppt_pages.text),
                ocr_status = CASE
                    WHEN l.ocr_status = 'done' OR main.ppt_pages.ocr_status = 'done'
                        THEN 'done'
                    WHEN l.ocr_status = 'failed' OR main.ppt_pages.ocr_status = 'failed'
                        THEN 'failed'
                    ELSE 'pending'
                END,
                ocr_at = COALESCE(l.ocr_at, main.ppt_pages.ocr_at),
                created_sec = COALESCE(l.created_sec, main.ppt_pages.created_sec),
                pptimgurl = COALESCE(l.pptimgurl, main.ppt_pages.pptimgurl)
            FROM local.ppt_pages l
            WHERE main.ppt_pages.sub_id = l.sub_id
              AND main.ppt_pages.page_num = l.page_num
        """)

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} LOCAL_DB REMOTE_DB")
        print("Merges LOCAL_DB into REMOTE_DB (additive-only).")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
    print("Merge complete.")
