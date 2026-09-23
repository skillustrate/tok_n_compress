"""
Database Manager for Agentic Hybrid Memory System (AHMS).

Provides SQLite persistence with:
- Single-file unified persistence for checkpoints, raw text, and vector embeddings.
- Full-text search (FTS5) for exact keyword/token matching.
- Embedded vector cosine distance calculation via registered SQLite function.
- Hybrid Reciprocal Rank Fusion (RRF) search.
- Pinned turn protection and state archiving flags.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .embeddings import FastLocalEmbedding, get_embedding_engine


class DatabaseManager:
    """Manages SQLite database operations with FTS5, vector search, and atomic transactions."""

    def __init__(self, db_path: str = "./checkpoints.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.embedder = get_embedding_engine()
        self._ensure_db_exists()

    @contextmanager
    def _connection(self):
        """Context manager for SQLite connections ensuring closure, foreign keys, and vector functions."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        # Register custom SQLite vector distance function
        def _calc_cosine_dist(b1: Optional[bytes], b2: Optional[bytes]) -> float:
            if not b1 or not b2:
                return 1.0
            v1 = FastLocalEmbedding.deserialize_vector(b1)
            v2 = FastLocalEmbedding.deserialize_vector(b2)
            return FastLocalEmbedding.cosine_distance(v1, v2)

        conn.create_function("cosine_dist", 2, _calc_cosine_dist)

        try:
            yield conn
        finally:
            conn.close()

    def _ensure_db_exists(self):
        """Create database tables, FTS5 virtual tables, and indexes if they don't exist."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()

                # 1. Checkpoints table (Layer 2)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        parent_checkpoint_id INTEGER,
                        timestamp TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        files_metadata TEXT,       -- JSON object
                        key_prompts TEXT,          -- JSON list
                        raw_segment_refs TEXT,     -- JSON list of raw_segment IDs
                        is_pinned INTEGER DEFAULT 0,
                        is_archived INTEGER DEFAULT 0,
                        FOREIGN KEY (parent_checkpoint_id) REFERENCES checkpoints(id) ON DELETE SET NULL
                    )
                """)

                # 2. Raw segments table (Layer 3) with embeddings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_segments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        checkpoint_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        embedding BLOB,            -- Float32 packed vector
                        is_pinned INTEGER DEFAULT 0,
                        is_archived INTEGER DEFAULT 0,
                        FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id) ON DELETE CASCADE
                    )
                """)

                # Migrations for existing databases
                self._migrate_columns_if_missing(cursor)

                # 3. FTS5 Virtual Table for full-text search
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS raw_segments_fts USING fts5(
                        segment_id UNINDEXED,
                        content,
                        tokenize = 'porter unicode61'
                    )
                """)

                # 4. Indexes for fast retrieval
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_segments_checkpoint 
                    ON raw_segments(checkpoint_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_checkpoints_parent 
                    ON checkpoints(parent_checkpoint_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_raw_segments_pinned
                    ON raw_segments(is_pinned)
                """)

                conn.commit()

    def _migrate_columns_if_missing(self, cursor: sqlite3.Cursor):
        """Safely add new columns to existing v1.0 databases."""
        # Check raw_segments columns
        cursor.execute("PRAGMA table_info(raw_segments)")
        raw_cols = {row["name"] for row in cursor.fetchall()}
        if "embedding" not in raw_cols:
            cursor.execute("ALTER TABLE raw_segments ADD COLUMN embedding BLOB")
        if "is_pinned" not in raw_cols:
            cursor.execute("ALTER TABLE raw_segments ADD COLUMN is_pinned INTEGER DEFAULT 0")
        if "is_archived" not in raw_cols:
            cursor.execute("ALTER TABLE raw_segments ADD COLUMN is_archived INTEGER DEFAULT 0")

        # Check checkpoints columns
        cursor.execute("PRAGMA table_info(checkpoints)")
        cp_cols = {row["name"] for row in cursor.fetchall()}
        if "is_pinned" not in cp_cols:
            cursor.execute("ALTER TABLE checkpoints ADD COLUMN is_pinned INTEGER DEFAULT 0")
        if "is_archived" not in cp_cols:
            cursor.execute("ALTER TABLE checkpoints ADD COLUMN is_archived INTEGER DEFAULT 0")

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
        raw_segment_content: str,
        embedding: Optional[List[float]] = None,
        is_pinned: bool = False
    ) -> Tuple[int, int]:
        """
        Atomically save both a raw conversation segment (with embeddings)
        and its checkpoint in one ACID transaction.
        """
        if embedding is None:
            embedding = self.embedder.embed_text(raw_segment_content)

        embedding_blob = FastLocalEmbedding.serialize_vector(embedding)
        pinned_int = 1 if is_pinned else 0

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
                            files_metadata, key_prompts, raw_segment_refs, is_pinned
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (parent_id, timestamp, summary, files_json, prompts_json, json.dumps([]), pinned_int))
                    checkpoint_id = cursor.lastrowid
                    if checkpoint_id is None:
                        raise RuntimeError("Failed to obtain checkpoint ID during insert")

                    # 2. Insert raw segment referencing the new checkpoint ID
                    cursor.execute("""
                        INSERT INTO raw_segments (checkpoint_id, content, timestamp, embedding, is_pinned)
                        VALUES (?, ?, ?, ?, ?)
                    """, (checkpoint_id, raw_segment_content, timestamp, embedding_blob, pinned_int))
                    raw_segment_id = cursor.lastrowid
                    if raw_segment_id is None:
                        raise RuntimeError("Failed to obtain raw segment ID during insert")

                    # 3. Synchronize with FTS5 index
                    cursor.execute("""
                        INSERT INTO raw_segments_fts (segment_id, content)
                        VALUES (?, ?)
                    """, (raw_segment_id, raw_segment_content))

                    # 4. Update checkpoint with the raw_segment_ref
                    refs_json = json.dumps([raw_segment_id])
                    cursor.execute("""
                        UPDATE checkpoints SET raw_segment_refs = ? WHERE id = ?
                    """, (refs_json, checkpoint_id))

                    conn.commit()
                    return checkpoint_id, raw_segment_id
                except Exception:
                    conn.rollback()
                    raise

    # Pinning & Archiving Operations
    def set_pin_status(self, segment_id: int, is_pinned: bool) -> bool:
        """Set pinned status on a raw segment and its parent checkpoint."""
        pinned_int = 1 if is_pinned else 0
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE raw_segments SET is_pinned = ? WHERE id = ?", (pinned_int, segment_id))
                updated = cursor.rowcount > 0
                conn.commit()
                return updated

    def archive_segment(self, segment_id: int) -> bool:
        """Mark a raw segment as archived (down-tiered)."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE raw_segments SET is_archived = 1 WHERE id = ?", (segment_id,))
                archived = cursor.rowcount > 0
                conn.commit()
                return archived

    # Checkpoint retrieval operations
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
    def save_raw_segment(
        self,
        checkpoint_id: int,
        content: str,
        timestamp: str,
        embedding: Optional[List[float]] = None
    ) -> int:
        """Save a raw conversation segment linked to a checkpoint."""
        if embedding is None:
            embedding = self.embedder.embed_text(content)
        embedding_blob = FastLocalEmbedding.serialize_vector(embedding)

        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO raw_segments (checkpoint_id, content, timestamp, embedding) VALUES (?, ?, ?, ?)",
                    (checkpoint_id, content, timestamp, embedding_blob),
                )
                segment_id = cursor.lastrowid
                if segment_id is not None:
                    cursor.execute(
                        "INSERT INTO raw_segments_fts (segment_id, content) VALUES (?, ?)",
                        (segment_id, content)
                    )
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

    # Search & Retrieval: FTS5, Vector & Hybrid RRF
    def search_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search raw segments using FTS5 full-text indexing with LIKE fallback."""
        trimmed = query.strip()
        if not trimmed:
            return []

        # Sanitize query for FTS5 (strip special syntax characters)
        import re
        words = re.findall(r"[a-zA-Z0-9_\-\./#]+", trimmed)
        if not words:
            return []

        fts_query = " OR ".join(f'"{w}"' for w in words)

        with self._connection() as conn:
            cursor = conn.cursor()
            results = []

            # 1. Attempt FTS5 query
            try:
                cursor.execute(
                    """
                    SELECT r.id, r.checkpoint_id, r.content, r.timestamp, r.is_pinned, r.is_archived
                    FROM raw_segments_fts f
                    JOIN raw_segments r ON f.segment_id = r.id
                    WHERE raw_segments_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit)
                )
                for row in cursor.fetchall():
                    item = dict(row)
                    item["segment_id"] = item["id"]
                    results.append(item)
            except sqlite3.OperationalError:
                pass  # Fallback to standard LIKE

            # 2. Fallback to LIKE if FTS yielded no results
            if not results:
                clauses = " OR ".join(["content LIKE ?" for _ in words[:5]])
                params = [f"%{w}%" for w in words[:5]] + [limit]
                cursor.execute(
                    f"""
                    SELECT id, checkpoint_id, content, timestamp, is_pinned, is_archived
                    FROM raw_segments
                    WHERE {clauses}
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    params
                )
                for row in cursor.fetchall():
                    item = dict(row)
                    item["segment_id"] = item["id"]
                    results.append(item)

            return results

    def search_vector(self, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
        """Search raw segments by semantic cosine distance inside SQLite."""
        if not query_vector:
            return []

        q_bytes = FastLocalEmbedding.serialize_vector(query_vector)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, checkpoint_id, content, timestamp, is_pinned, is_archived,
                       cosine_dist(embedding, ?) AS distance
                FROM raw_segments
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT ?
                """,
                (q_bytes, limit)
            )

            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["segment_id"] = d["id"]
                dist = d.get("distance", 1.0)
                # Convert cosine distance (0..2) to similarity score (0..1)
                d["relevance_score"] = round(max(0.0, 1.0 - (dist / 2.0)), 4)
                results.append(d)

            return results

    def search_hybrid_rrf(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        limit: int = 10,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search combining FTS5 keyword results with vector similarity
        using Reciprocal Rank Fusion (RRF).
        
        Score formula: RRF_score(d) = sum(1 / (k + rank_i))
        """
        trimmed = query.strip()
        if not trimmed:
            return []

        if query_vector is None:
            query_vector = self.embedder.embed_text(trimmed)

        # 1. Keyword search (FTS5)
        fts_matches = self.search_fts(trimmed, limit=limit * 2)

        # 2. Vector search (cosine distance)
        vec_matches = self.search_vector(query_vector, limit=limit * 2)

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[int, float] = {}
        items_by_id: Dict[int, Dict[str, Any]] = {}

        # Process FTS ranks
        for rank, item in enumerate(fts_matches):
            item_id = item["id"]
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            items_by_id[item_id] = item

        # Process Vector ranks
        for rank, item in enumerate(vec_matches):
            item_id = item["id"]
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if item_id not in items_by_id:
                items_by_id[item_id] = item

        # 4. Sort by blended RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for i in sorted_ids[:limit]:
            entry = dict(items_by_id[i])
            entry["segment_id"] = entry.get("id", i)
            entry["id"] = entry["segment_id"]
            entry["rrf_score"] = round(rrf_scores[i], 5)
            # Normalized score for presentation
            entry["relevance_score"] = round(min(rrf_scores[i] * rrf_k, 1.0), 4)
            results.append(entry)

        return results

    # Legacy keyword search fallback
    def search_raw_segments(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Backward-compatible search routing through FTS5."""
        return self.search_fts(query, limit=limit)

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
                    files_metadata, key_prompts, raw_segment_refs, is_pinned, is_archived,
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
        """Delete a checkpoint and its cascaded raw segments and FTS index."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                # Remove from FTS
                cursor.execute("""
                    DELETE FROM raw_segments_fts 
                    WHERE segment_id IN (SELECT id FROM raw_segments WHERE checkpoint_id = ?)
                """, (checkpoint_id,))
                cursor.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted

    def clear_all(self):
        """Clear all checkpoints, raw segments, and FTS indexes from the database (for testing)."""
        with self._lock:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM raw_segments_fts")
                cursor.execute("DELETE FROM raw_segments")
                cursor.execute("DELETE FROM checkpoints")
                conn.commit()

    def close(self):
        """Close hook."""
        pass


# Registry of DatabaseManager instances
_db_managers: Dict[str, DatabaseManager] = {}
_db_lock = threading.Lock()


def get_database_manager(db_path: str = "./checkpoints.db") -> DatabaseManager:
    """Get or create the DatabaseManager instance for the given path."""
    resolved_key = str(Path(db_path).resolve())
    with _db_lock:
        if resolved_key not in _db_managers:
            _db_managers[resolved_key] = DatabaseManager(resolved_key)
        return _db_managers[resolved_key]
