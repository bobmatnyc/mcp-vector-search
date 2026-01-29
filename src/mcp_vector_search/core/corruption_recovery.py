"""Corruption detection and recovery for ChromaDB indices."""

import pickle  # nosec B403 # Trusted internal index files only
import shutil
import sqlite3
import time
from pathlib import Path

from loguru import logger

from .exceptions import IndexCorruptionError


class CorruptionRecovery:
    """Handles detection and recovery from ChromaDB index corruption.

    Implements multi-layer corruption detection:
    - Layer 1: SQLite database integrity checks (pre-initialization)
    - Layer 2: HNSW pickle file validation
    - Layer 3: Rust panic pattern detection during operations
    """

    def __init__(self, persist_directory: Path) -> None:
        """Initialize corruption recovery handler.

        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        self.persist_directory = persist_directory
        self.recovery_attempted = False

    async def detect_corruption(self) -> bool:
        """Detect index corruption proactively before initialization.

        Returns:
            True if corruption detected, False otherwise
        """
        corruption_detected = False

        # Layer 1: Check SQLite database integrity (if it exists)
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        if chroma_db_path.exists():
            if await self._check_sqlite_corruption(chroma_db_path):
                corruption_detected = True

        # Layer 2: Check HNSW index files (if they exist)
        # CRITICAL: Run this check even if SQLite doesn't exist, because
        # corrupted .bin files can cause bus errors before SQLite is accessed
        if await self._check_hnsw_corruption():
            corruption_detected = True

        return corruption_detected

    async def _check_sqlite_corruption(self, db_path: Path) -> bool:
        """Check SQLite database for corruption.

        Args:
            db_path: Path to SQLite database file

        Returns:
            True if corruption detected, False otherwise
        """
        try:
            logger.debug("Running SQLite integrity check...")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]
            conn.close()

            if result != "ok":
                logger.warning(f"SQLite database corruption detected: {result}")
                return True

            logger.debug("SQLite integrity check passed")
            return False

        except sqlite3.Error as e:
            logger.warning(f"SQLite database error during integrity check: {e}")
            return True

    async def _check_hnsw_corruption(self) -> bool:
        """Check HNSW index files for corruption.

        Returns:
            True if corruption detected, False otherwise
        """
        index_path = self.persist_directory / "index"
        if not index_path.exists():
            return False

        # Look for HNSW index files
        pickle_files = list(index_path.glob("**/*.pkl"))
        pickle_files.extend(list(index_path.glob("**/*.pickle")))
        bin_files = list(index_path.glob("**/*.bin"))

        logger.debug(
            f"Checking {len(pickle_files)} pickle files and {len(bin_files)} binary files for corruption..."
        )

        # Validate pickle files
        for pickle_file in pickle_files:
            try:
                # Check file size - suspiciously small files might be corrupted
                file_size = pickle_file.stat().st_size
                if file_size == 0:
                    logger.warning(
                        f"Empty HNSW pickle file detected: {pickle_file} (0 bytes)"
                    )
                    return True

                # Validate pickle file contents
                if await self._validate_pickle_file(pickle_file):
                    return True

            except (EOFError, pickle.UnpicklingError) as e:
                logger.warning(f"Pickle corruption detected in {pickle_file}: {e}")
                return True
            except Exception as e:
                # Check for Rust panic patterns
                error_msg = str(e).lower()
                if "range start index" in error_msg and "out of range" in error_msg:
                    logger.warning(f"Rust panic pattern detected in {pickle_file}: {e}")
                    return True

                logger.warning(f"Error reading HNSW pickle file {pickle_file}: {e}")
                # Continue checking other files before deciding to recover
                continue

        # Validate binary .bin files (CRITICAL: prevents bus errors)
        for bin_file in bin_files:
            try:
                if await self._validate_bin_file(bin_file):
                    return True
            except Exception as e:
                logger.warning(f"Error validating binary file {bin_file}: {e}")
                return True

        logger.debug("HNSW index files validation passed")
        return False

    async def _validate_pickle_file(self, pickle_file: Path) -> bool:
        """Validate a pickle file's contents.

        Args:
            pickle_file: Path to pickle file to validate

        Returns:
            True if corruption detected, False otherwise
        """
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)  # nosec B301 # Trusted internal index files only

            # Check if data structure is valid
            if data is None:
                logger.warning(f"HNSW index file contains None data: {pickle_file}")
                return True

            # Check for metadata consistency (if it's a dict)
            if isinstance(data, dict):
                # Look for known metadata keys that should exist
                if "space" in data and "dim" in data:
                    # Validate dimensions are reasonable
                    if data.get("dim", 0) <= 0:
                        logger.warning(
                            f"Invalid dimensions in HNSW index: {pickle_file} (dim={data.get('dim')})"
                        )
                        return True

        return False

    async def _validate_bin_file(self, bin_file: Path) -> bool:
        """Validate HNSW binary index file to prevent bus errors.

        This method reads the first few KB of .bin files to detect corruption
        BEFORE ChromaDB's Rust backend tries to access them. This prevents
        SIGBUS crashes that cannot be caught with try/except.

        Args:
            bin_file: Path to binary HNSW index file

        Returns:
            True if corruption detected, False otherwise
        """
        try:
            file_size = bin_file.stat().st_size

            # Zero-size files are definitely corrupted
            if file_size == 0:
                logger.warning(f"Empty binary HNSW file detected: {bin_file} (0 bytes)")
                return True

            # Suspiciously small files (< 100 bytes) are likely corrupted
            if file_size < 100:
                logger.warning(
                    f"Suspiciously small binary HNSW file: {bin_file} ({file_size} bytes)"
                )
                return True

            # Attempt to read first 4KB to verify file accessibility
            # This catches truncated files and I/O errors before Rust does
            chunk_size = min(4096, file_size)
            with open(bin_file, "rb") as f:
                header = f.read(chunk_size)

                # Verify we read the expected amount
                if len(header) < chunk_size:
                    logger.warning(
                        f"Truncated binary HNSW file: {bin_file} (expected {chunk_size} bytes, got {len(header)})"
                    )
                    return True

                # Check for all-zero files (corrupted/incomplete writes)
                if header == b"\x00" * len(header):
                    logger.warning(f"All-zero binary HNSW file detected: {bin_file}")
                    return True

            # File appears valid
            return False

        except OSError as e:
            logger.warning(f"I/O error reading binary HNSW file {bin_file}: {e}")
            return True

    def is_rust_panic_error(self, error: Exception) -> bool:
        """Check if an error matches Rust panic patterns.

        Args:
            error: Exception to check

        Returns:
            True if error appears to be a Rust panic
        """
        error_msg = str(error).lower()
        rust_panic_patterns = [
            "range start index",
            "out of range",
            "panic",
            "thread panicked",
            "slice of length",
            "index out of bounds",
        ]
        return any(pattern in error_msg for pattern in rust_panic_patterns)

    def is_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates database corruption.

        Args:
            error: Exception to check

        Returns:
            True if error indicates corruption
        """
        error_msg = str(error).lower()
        corruption_indicators = [
            "pickle",
            "unpickling",
            "eof",
            "ran out of input",
            "hnsw",
            "index",
            "deserialize",
            "corrupt",
            "file is not a database",
            "database error",
        ]
        return any(indicator in error_msg for indicator in corruption_indicators)

    async def recover(self) -> None:
        """Recover from index corruption by rebuilding the index.

        Creates a timestamped backup and clears the corrupted index.

        Raises:
            IndexCorruptionError: If recovery fails
        """
        logger.warning("=" * 80)
        logger.warning("INDEX CORRUPTION DETECTED - Initiating recovery...")
        logger.warning("=" * 80)

        # Create backup
        backup_path = await self._create_backup()

        # Clear corrupted index
        await self._clear_corrupted_index()

        # Recreate directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Index directory recreated")

        logger.warning("=" * 80)
        logger.warning("RECOVERY COMPLETE - Next steps:")
        logger.warning("  1. Run 'mcp-vector-search index' to rebuild the index")
        logger.warning(f"  2. Backup saved to: {backup_path}")
        logger.warning("=" * 80)

        # Mark recovery as attempted
        self.recovery_attempted = True

    async def _create_backup(self) -> Path:
        """Create a backup of the corrupted index.

        Returns:
            Path to backup directory
        """
        backup_dir = (
            self.persist_directory.parent / f"{self.persist_directory.name}_backup"
        )
        backup_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        backup_path = backup_dir / f"backup_{timestamp}"

        if self.persist_directory.exists():
            try:
                shutil.copytree(self.persist_directory, backup_path)
                logger.info(f"✓ Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"⚠ Could not create backup: {e}")

        return backup_path

    async def _clear_corrupted_index(self) -> None:
        """Clear the corrupted index directory.

        Raises:
            IndexCorruptionError: If clearing fails
        """
        if self.persist_directory.exists():
            try:
                # Log what we're about to delete
                total_size = sum(
                    f.stat().st_size
                    for f in self.persist_directory.rglob("*")
                    if f.is_file()
                )
                logger.info(
                    f"Clearing corrupted index ({total_size / 1024 / 1024:.2f} MB)..."
                )

                shutil.rmtree(self.persist_directory)
                logger.info(f"✓ Cleared corrupted index at {self.persist_directory}")
            except Exception as e:
                logger.error(f"✗ Failed to clear corrupted index: {e}")
                raise IndexCorruptionError(
                    f"Could not clear corrupted index: {e}. "
                    f"Please manually delete {self.persist_directory} and try again."
                ) from e
