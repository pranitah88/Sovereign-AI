"""
File metadata repository — tracks uploaded and generated files.
"""

import json
import logging

from backend.database.connection import get_connection, transaction

logger = logging.getLogger(__name__)


# ── Uploaded Files ────────────────────────────────────────────────────────

def ensure_uploaded_files_schema() -> None:
    """Ensure uploaded_files table supports all classification, metadata, status, and error columns."""
    try:
        with transaction() as conn:
            # Check existing columns
            cols = [row[1] for row in conn.execute("PRAGMA table_info(uploaded_files)").fetchall()]
            if not cols:
                return

            if "classification" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN classification TEXT NOT NULL DEFAULT 'INTERNAL'")
            if "department" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN department TEXT NOT NULL DEFAULT 'GENERAL'")
            if "owner" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN owner TEXT NOT NULL DEFAULT 'SYSTEM'")
            if "version" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN version TEXT NOT NULL DEFAULT '1.0'")
            if "doc_id" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN doc_id TEXT")
            if "error_message" not in cols:
                conn.execute("ALTER TABLE uploaded_files ADD COLUMN error_message TEXT")

            # Check if index_status CHECK constraint needs migration for 'completed'
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='uploaded_files'"
            ).fetchone()
            if row and row[0]:
                table_sql = row[0]
                if "'completed'" not in table_sql:
                    logger.info("Migrating uploaded_files table to support 'completed' status...")
                    conn.execute("PRAGMA foreign_keys = OFF")
                    conn.execute("ALTER TABLE uploaded_files RENAME TO _uploaded_files_old")
                    conn.execute(
                        """
                        CREATE TABLE uploaded_files (
                            id              INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            original_name   TEXT    NOT NULL,
                            stored_path     TEXT    NOT NULL,
                            file_size_bytes INTEGER NOT NULL,
                            mime_type       TEXT,
                            category        TEXT,
                            classification  TEXT    NOT NULL DEFAULT 'INTERNAL',
                            department      TEXT    NOT NULL DEFAULT 'GENERAL',
                            owner           TEXT    NOT NULL DEFAULT 'SYSTEM',
                            version         TEXT    NOT NULL DEFAULT '1.0',
                            doc_id          TEXT,
                            is_indexed      INTEGER NOT NULL DEFAULT 0,
                            index_status    TEXT    DEFAULT 'pending' CHECK(index_status IN ('pending', 'indexing', 'indexed', 'completed', 'failed')),
                            error_message   TEXT,
                            ocr_applied     INTEGER NOT NULL DEFAULT 0,
                            uploaded_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                        )
                        """
                    )
                    conn.execute(
                        """
                        INSERT INTO uploaded_files
                            (id, user_id, original_name, stored_path, file_size_bytes, mime_type, category,
                             classification, department, owner, version, doc_id, is_indexed, index_status, error_message, ocr_applied, uploaded_at)
                        SELECT
                            id, user_id, original_name, stored_path, file_size_bytes, mime_type, category,
                            COALESCE(classification, 'INTERNAL'), COALESCE(department, 'GENERAL'),
                            COALESCE(owner, 'SYSTEM'), COALESCE(version, '1.0'), doc_id,
                            is_indexed, index_status, error_message, ocr_applied, uploaded_at
                        FROM _uploaded_files_old
                        """
                    )
                    conn.execute("DROP TABLE _uploaded_files_old")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_uploaded_files_user ON uploaded_files(user_id)")
                    conn.execute("PRAGMA foreign_keys = ON")
                    logger.info("uploaded_files table successfully migrated.")
    except Exception as exc:
        logger.warning("Could not auto-migrate uploaded_files table: %s", exc)


def record_upload(
    user_id: int,
    original_name: str,
    stored_path: str,
    file_size_bytes: int,
    mime_type: str | None = None,
    category: str | None = None,
    classification: str = "INTERNAL",
    department: str = "GENERAL",
    owner: str = "SYSTEM",
    version: str = "1.0",
    doc_id: str | None = None,
) -> int:
    """Record an uploaded file with data classification metadata. Returns the file id."""
    ensure_uploaded_files_schema()
    import uuid
    actual_doc_id = doc_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO uploaded_files
                (user_id, original_name, stored_path, file_size_bytes, mime_type, category,
                 classification, department, owner, version, doc_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, original_name, stored_path, file_size_bytes, mime_type, category,
                classification.upper(), department.upper(), owner, version, actual_doc_id
            ),
        )
        return cursor.lastrowid


def get_uploaded_file(file_id: int) -> dict | None:
    """Return an uploaded file's metadata."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM uploaded_files WHERE id = ?", (file_id,)).fetchone()
    return dict(row) if row else None


def list_uploaded_files(user_id: int | None = None) -> list[dict]:
    """List uploaded files, optionally filtered by user."""
    conn = get_connection()
    if user_id is not None:
        rows = conn.execute(
            "SELECT * FROM uploaded_files WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM uploaded_files ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]



def update_index_status(
    file_id: int,
    status: str,
    ocr_applied: bool = False,
    error_message: str | None = None,
) -> bool:
    """Update the indexing status of an uploaded file."""
    ensure_uploaded_files_schema()
    is_indexed_val = 1 if status in ("completed", "indexed") else 0
    with transaction() as conn:
        cursor = conn.execute(
            """
            UPDATE uploaded_files
            SET index_status = ?, is_indexed = ?, ocr_applied = ?, error_message = ?
            WHERE id = ?
            """,
            (status, is_indexed_val, int(ocr_applied), error_message, file_id),
        )
        return cursor.rowcount > 0


def delete_uploaded_file(file_id: int) -> bool:
    """Delete an uploaded file record. Returns True if found."""
    with transaction() as conn:
        cursor = conn.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
        return cursor.rowcount > 0


# ── Generated Files ──────────────────────────────────────────────────────

def record_generated_file(
    user_id: int,
    file_type: str,
    original_name: str,
    stored_path: str,
    file_size_bytes: int | None = None,
    task_id: int | None = None,
) -> int:
    """Record a generated file. Returns the file id."""
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO generated_files
                (user_id, task_id, file_type, original_name, stored_path, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, task_id, file_type, original_name, stored_path, file_size_bytes),
        )
        return cursor.lastrowid


def list_generated_files(user_id: int | None = None) -> list[dict]:
    """List generated files, optionally filtered by user."""
    conn = get_connection()
    if user_id is not None:
        rows = conn.execute(
            "SELECT * FROM generated_files WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM generated_files ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_generated_file(file_id: int) -> dict | None:
    """Return a generated file's metadata."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM generated_files WHERE id = ?", (file_id,)).fetchone()
    return dict(row) if row else None
