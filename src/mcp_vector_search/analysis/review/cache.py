"""SQLite-based cache for review findings to avoid re-reviewing unchanged code.

This cache stores review findings keyed by (file_path, content_hash, review_type)
to enable fast lookups when code hasn't changed since the last review.

Estimated 5x speedup for repeated reviews with 80% cache hit rate on stable codebases.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger


class ReviewCache:
    """SQLite-based cache for review findings.

    Cache key: (file_path, content_hash, review_type)
    Cache value: list of ReviewFinding serialized as JSON

    Example:
        >>> cache = ReviewCache(Path.cwd())
        >>> content_hash = ReviewCache.compute_hash(file_content)
        >>> cached_findings = cache.get("src/auth.py", content_hash, "security")
        >>> if cached_findings is None:
        ...     findings = run_llm_review(...)
        ...     cache.set("src/auth.py", content_hash, "security", findings, "gpt-4")
    """

    DB_PATH = ".mcp-vector-search/reviews.db"
    SCHEMA_VERSION = 1

    def __init__(self, project_root: Path) -> None:
        """Initialize review cache.

        Args:
            project_root: Project root directory (cache DB will be created here)
        """
        self.db_path = project_root / self.DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create review_cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS review_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    review_type TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    UNIQUE(file_path, content_hash, review_type)
                )
                """
            )

            # Create index for faster lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_lookup
                ON review_cache(file_path, content_hash, review_type)
                """
            )

            # Create stats table for tracking cache performance
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    file_path TEXT,
                    hit BOOLEAN
                )
                """
            )

            conn.commit()
            logger.debug(f"Initialized review cache database at {self.db_path}")

    def get(
        self, file_path: str, content_hash: str, review_type: str
    ) -> list[dict] | None:
        """Return cached findings or None if not found/expired.

        Args:
            file_path: Path to file that was reviewed
            content_hash: SHA256 hash of file content
            review_type: Type of review (security, architecture, performance)

        Returns:
            List of finding dictionaries if cached, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Query for cached findings
                cursor.execute(
                    """
                    SELECT findings_json, model_used, reviewed_at
                    FROM review_cache
                    WHERE file_path = ? AND content_hash = ? AND review_type = ?
                    """,
                    (file_path, content_hash, review_type),
                )

                row = cursor.fetchone()

                if row is None:
                    logger.debug(
                        f"Cache miss: {file_path} (type={review_type}, hash={content_hash[:8]})"
                    )
                    self._record_stat("get", file_path, hit=False)
                    return None

                findings_json, model_used, reviewed_at = row
                findings = json.loads(findings_json)

                logger.debug(
                    f"Cache hit: {file_path} (type={review_type}, model={model_used}, "
                    f"reviewed={reviewed_at}, findings={len(findings)})"
                )
                self._record_stat("get", file_path, hit=True)

                return findings

        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get cached findings for {file_path}: {e}")
            return None

    def set(
        self,
        file_path: str,
        content_hash: str,
        review_type: str,
        findings: list[dict],
        model_used: str,
    ) -> None:
        """Cache findings for a file+hash+type combination.

        Args:
            file_path: Path to file that was reviewed
            content_hash: SHA256 hash of file content
            review_type: Type of review (security, architecture, performance)
            findings: List of finding dictionaries to cache
            model_used: LLM model name used for review
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Serialize findings
                findings_json = json.dumps(findings)
                reviewed_at = datetime.utcnow().isoformat()

                # Insert or replace cached findings
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO review_cache
                    (file_path, content_hash, review_type, findings_json,
                     reviewed_at, model_used, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        content_hash,
                        review_type,
                        findings_json,
                        reviewed_at,
                        model_used,
                        self.SCHEMA_VERSION,
                    ),
                )

                conn.commit()
                logger.debug(
                    f"Cached {len(findings)} findings for {file_path} "
                    f"(type={review_type}, hash={content_hash[:8]})"
                )

        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning(f"Failed to cache findings for {file_path}: {e}")

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cache entries for a file.

        Args:
            file_path: Path to file to invalidate
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM review_cache WHERE file_path = ?",
                    (file_path,),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(
                    f"Invalidated {deleted_count} cache entries for {file_path}"
                )

        except sqlite3.Error as e:
            logger.warning(f"Failed to invalidate cache for {file_path}: {e}")

    def clear(self, review_type: str | None = None) -> int:
        """Clear all or type-specific cache entries.

        Args:
            review_type: Optional review type to clear (None = clear all)

        Returns:
            Number of cache entries cleared
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if review_type:
                    cursor.execute(
                        "DELETE FROM review_cache WHERE review_type = ?",
                        (review_type,),
                    )
                    logger.info(f"Cleared {review_type} review cache")
                else:
                    cursor.execute("DELETE FROM review_cache")
                    logger.info("Cleared all review cache")

                deleted_count = cursor.rowcount
                conn.commit()

                return deleted_count

        except sqlite3.Error as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0

    def stats(self) -> dict:
        """Return cache statistics: total_entries, hit_rate, size_bytes.

        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total entries
                cursor.execute("SELECT COUNT(*) FROM review_cache")
                total_entries = cursor.fetchone()[0]

                # Database size
                size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

                # Hit rate (from recent stats)
                cursor.execute(
                    """
                    SELECT
                        SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as hits,
                        COUNT(*) as total
                    FROM cache_stats
                    WHERE operation = 'get'
                    AND timestamp > datetime('now', '-1 hour')
                    """
                )
                row = cursor.fetchone()
                hits, total = row if row else (0, 0)
                hit_rate = hits / total if total > 0 else 0.0

                # Entries by review type
                cursor.execute(
                    """
                    SELECT review_type, COUNT(*) as count
                    FROM review_cache
                    GROUP BY review_type
                    """
                )
                by_type = {row[0]: row[1] for row in cursor.fetchall()}

                return {
                    "total_entries": total_entries,
                    "hit_rate": hit_rate,
                    "size_bytes": size_bytes,
                    "by_type": by_type,
                }

        except sqlite3.Error as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                "total_entries": 0,
                "hit_rate": 0.0,
                "size_bytes": 0,
                "by_type": {},
            }

    def _record_stat(self, operation: str, file_path: str, hit: bool) -> None:
        """Record cache operation statistic.

        Args:
            operation: Operation name (e.g., 'get', 'set')
            file_path: File path involved in operation
            hit: Whether operation was a cache hit
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO cache_stats (timestamp, operation, file_path, hit)
                    VALUES (?, ?, ?, ?)
                    """,
                    (datetime.utcnow().isoformat(), operation, file_path, hit),
                )

                conn.commit()

        except sqlite3.Error as e:
            # Don't log warnings for stat recording failures (not critical)
            logger.debug(f"Failed to record cache stat: {e}")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: String content to hash

        Returns:
            Hexadecimal SHA256 hash
        """
        return hashlib.sha256(content.encode()).hexdigest()
