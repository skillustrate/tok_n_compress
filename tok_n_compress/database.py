"""
Database Manager for Hybrid Memory Model Checkpoints.

Provides robust SQLite persistence with atomic transaction support,
safe JSON serialization, foreign key constraints, and connection safety.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


class DatabaseManager:
    """Manages SQLite database operations for the Hybrid Memory Model."""

    def __init__(self, db_path: str = "./checkpoints.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_db_exists()

    @contextmanager
    def _connection(self):
        """Context manager for SQLite connections ensuring closure and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_db_exists(self):
        """Create database tables and indexes if they don't exist."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                # Create checkpoints table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        parent_checkpoint_id INTEGER,
                        timestamp TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        files_metadata TEXT,       -- JSON object
                        key_prompts TEXT,          -- JSON list
                        raw_segment_refs TEXT,     -- JSON list of raw_segment IDs
                        FOREIGN KEY (parent_checkpoint_id) REFERENCES checkpoints(id) ON DELETE SET NULL
                    )
                """)

                # Create raw_segments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_segments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        checkpoint_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE
                    )
                """)

                # Indexes for retrieval speed
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_segments_checkpoint 
                    ON raw_segments(checkpoint_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_checkpoints_parent 
                    ON checkpoints(parent_checkpoint_id)
                """)

                conn.commit()

    @staticmethod
    def _safe_json_loads(data: Optional[str], default: Any) -> Any:
        """Safely deserialize JSON string with a fallback default."""
        if not data:
            return default
        try:
            return json.loads(data)
        except (ValueError, TypeError):
            return default

    # Atomic operations
    def save_checkpoint_atomic(
        self,
        parent_id: Optional[int],
        timestamp: str,
        summary: str,
        files_metadata: Any,
        key_prompts: Any,
        raw_segment_content: str
    ) -> Tuple[int, int]:
        """
        Atomically save both a raw conversation segment and its checkpoint in one transaction.

        Args:
            parent_id: ID of the previous checkpoint (for hierarchy), or None.
            timestamp: ISO format timestamp string.
            summary: Distilled conversation summary.
            files_metadata: List or Dict of file metadata.
            key_prompts: List of key prompts and outcomes.
            raw_segment_content: Full uncompressed text of the segment.

        Returns:
            Tuple of (checkpoint_id, raw_segment_id)
        """
        with self._lock:
            with self._connection() as conn:
                try:
                    cursor = conn.cursor()

                    # 1. Insert checkpoint placeholder
                    files_json = json.dumps(files_metadata if files_metadata is not None else [])
                    prompts_json = json.dumps(key_prompts if key_prompts is not None else [])
                    cursor.execute("""
                        INSERT INTO checkpoints (
                            parent_checkpoint_id, timestamp, summary,
                            files_metadata, key_prompts, raw_segment_refs
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (parent_id, timestamp, summary, files_json, prompts_json, json.dumps([])))
                    checkpoint_id = cursor.lastrowid
                    if checkpoint_id is None:
                        raise RuntimeError("Failed to obtain checkpoint ID during insert")

                    # 2. Insert raw segment referencing the new checkpoint ID
                    cursor.execute("""
                        INSERT INTO raw_segments (checkpoint_id, content, timestamp)
                        VALUES (?, ?, ?)
                    """, (checkpoint_id, raw_segment_content, timestamp))
                    raw_segment_id = cursor.lastrowid
                    if raw_segment_id is None:
                        raise RuntimeError("Failed to obtain raw segment ID during insert")

                    # 3. Update checkpoint with the raw_segment_ref
                    refs_json = json.dumps([raw_segment_id])
                    cursor.execute("""
                        UPDATE checkpoints SET raw_segment_refs = ? WHERE id = ?
                    """, (refs_json, checkpoint_id))

                    conn.commit()
                    return checkpoint_id, raw_segment_id
                except Exception:
                    conn.rollback()
                    raise

    # Individual Checkpoint operations
    def save_checkpoint(
        self,
        parent_id: Optional[int],
        timestamp: str,
        summary: str,
        files_metadata: Any,
        key_prompts: Any,
        raw_segment_refs: List[int]
    ) -> int:
        """Save a standalone checkpoint and return its ID."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                files_json = json.dumps(files_metadata if files_metadata is not None else [])
                prompts_json = json.dumps(key_prompts if key_prompts is not None else [])
                refs_json = json.dumps(raw_segment_refs if raw_segment_refs is not None else [])

                cursor.execute("""
                    INSERT INTO checkpoints (
                        parent_checkpoint_id, timestamp, summary,
                        files_metadata, key_prompts, raw_segment_refs
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (parent_id, timestamp, summary, files_json, prompts_json, refs_json))
                checkpoint_id = cursor.lastrowid
                conn.commit()
                return int(checkpoint_id)

    def get_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a checkpoint by ID with parsed JSON fields."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            if not row:
                return None

            checkpoint = dict(row)
            checkpoint["files_metadata"] = self._safe_json_loads(checkpoint.get("files_metadata"), [])
            checkpoint["key_prompts"] = self._safe_json_loads(checkpoint.get("key_prompts"), [])
            checkpoint["raw_segment_refs"] = self._safe_json_loads(checkpoint.get("raw_segment_refs"), [])
            return checkpoint

    def get_parent_checkpoint(self, checkpoint_id: int) -> Optional[int]:
        """Get the parent checkpoint ID for a given checkpoint."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT parent_checkpoint_id FROM checkpoints WHERE id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            return row["parent_checkpoint_id"] if row else None

    def get_child_checkpoints(self, checkpoint_id: int) -> List[int]:
        """Get list of IDs for checkpoints whose parent is checkpoint_id."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM checkpoints WHERE parent_checkpoint_id = ? ORDER BY id ASC", (checkpoint_id,))
            return [row["id"] for row in cursor.fetchall()]

    # Raw segment operations
    def save_raw_segment(self, checkpoint_id: int, content: str, timestamp: str) -> int:
        """Save a raw conversation segment linked to a checkpoint."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO raw_segments (checkpoint_id, content, timestamp) VALUES (?, ?, ?)",
                    (checkpoint_id, content, timestamp),
                )
                segment_id = cursor.lastrowid
                conn.commit()
                return int(segment_id)

    def get_raw_segment(self, segment_id: int) -> Optional[str]:
        """Retrieve a raw segment's text content by its ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM raw_segments WHERE id = ?", (segment_id,))
            row = cursor.fetchone()
            return row["content"] if row else None

    def get_raw_segments_by_checkpoint(self, checkpoint_id: int) -> List[str]:
        """Get all raw segments for a given checkpoint in creation order."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM raw_segments WHERE checkpoint_id = ? ORDER BY id ASC",
                (checkpoint_id,)
            )
            return [row["content"] for row in cursor.fetchall()]

    # Search & retrieval operations
    def search_raw_segments(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search raw segments by keyword, exact phrase, or constituent terms."""
        trimmed = query.strip()
        if not trimmed:
            return []

        import re
        terms = [t for t in re.findall(r"\b[a-zA-Z0-9_\-\./#]+\b", trimmed) if len(t) > 1]
        if not terms:
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            results_by_id: Dict[int, Dict[str, Any]] = {}

            # 1. Exact phrase match
            cursor.execute(
                """
                SELECT id, checkpoint_id, content, timestamp 
                FROM raw_segments 
                WHERE content LIKE ? 
                ORDER BY id DESC 
                LIMIT ?
                """,
                (f"%{trimmed}%", limit)
            )
            for row in cursor.fetchall():
                results_by_id[row["id"]] = dict(row)

            # 2. Term match if limit not satisfied
            content_terms = [t for t in terms if len(t) >= 3]
            if len(results_by_id) < limit and content_terms:
                clauses = " OR ".join(["content LIKE ?" for _ in content_terms])
                params = [f"%{t}%" for t in content_terms] + [limit * 2]
                cursor.execute(
                    f"""
                    SELECT id, checkpoint_id, content, timestamp 
                    FROM raw_segments 
                    WHERE {clauses}
                    ORDER BY id DESC 
                    LIMIT ?
                    """,
                    params
                )
                for row in cursor.fetchall():
                    if row["id"] not in results_by_id:
                        results_by_id[row["id"]] = dict(row)

            return list(results_by_id.values())[:limit]

    def get_raw_segment_by_checkpoint(self, checkpoint_id: int) -> Optional[str]:
        """Get the most recent raw segment for a checkpoint."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM raw_segments WHERE checkpoint_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                (checkpoint_id,),
            )
            row = cursor.fetchone()
            return row["content"] if row else None

    def get_all_checkpoints(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all checkpoints with full metadata and raw segment counts."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    id, parent_checkpoint_id, timestamp, summary,
                    files_metadata, key_prompts, raw_segment_refs,
                    (SELECT COUNT(*) FROM raw_segments WHERE checkpoint_id = checkpoints.id) as segment_count
                FROM checkpoints
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                cp = dict(row)
                cp["files_metadata"] = self._safe_json_loads(cp.get("files_metadata"), [])
                cp["key_prompts"] = self._safe_json_loads(cp.get("key_prompts"), [])
                cp["raw_segment_refs"] = self._safe_json_loads(cp.get("raw_segment_refs"), [])
                results.append(cp)

            return results

    def delete_checkpoint(self, checkpoint_id: int) -> bool:
        """Delete a checkpoint and its cascaded raw segments."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted

    def clear_all(self):
        """Clear all checkpoints and raw segments from the database (for testing)."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM raw_segments")
                cursor.execute("DELETE FROM checkpoints")
                conn.commit()

    def close(self):
        """Close hook (connections are closed per operation)."""
        pass


# Registry of DatabaseManager instances keyed by normalized db path
_db_managers: Dict[str, DatabaseManager] = {}
_db_lock = threading.Lock()


def get_database_manager(db_path: str = "./checkpoints.db") -> DatabaseManager:
    """Get or create the DatabaseManager instance for the given path."""
    resolved_key = str(Path(db_path).resolve())
    with _db_lock:
        if resolved_key not in _db_managers:
            _db_managers[resolved_key] = DatabaseManager(resolved_key)
        return _db_managers[resolved_key]
