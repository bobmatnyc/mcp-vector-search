"""Utility functions for cleaning up stale index artifacts.

These module-level helpers were extracted from indexer.py so they can be
imported and tested independently of the full SemanticIndexer class.

Public API (re-exported from indexer.py for backward compatibility):
    cleanup_stale_locks
    cleanup_stale_transactions
    cleanup_stale_progress

Private helper (not re-exported):
    _detect_filesystem_type
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from loguru import logger


def cleanup_stale_locks(project_dir: Path) -> None:
    """Remove stale SQLite journal files that indicate interrupted transactions.

    Journal files (-journal, -wal, -shm) can be left behind if indexing is
    interrupted or crashes, preventing future database access. This function
    safely removes stale lock files at index startup.

    Args:
        project_dir: Project root directory containing .mcp-vector-search/
    """
    mcp_dir = project_dir / ".mcp-vector-search"
    if not mcp_dir.exists():
        return

    # SQLite journal file extensions that indicate locks/transactions
    lock_extensions = ["-journal", "-wal", "-shm"]

    removed_count = 0
    for ext in lock_extensions:
        lock_path = mcp_dir / f"chroma.sqlite3{ext}"
        if lock_path.exists():
            try:
                lock_path.unlink()
                logger.warning(f"Removed stale database lock file: {lock_path.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove stale lock file {lock_path}: {e}")

    if removed_count > 0:
        logger.info(
            f"Cleaned up {removed_count} stale lock files (indexing can now proceed)"
        )


def cleanup_stale_transactions(index_path: Path, stale_age_seconds: int = 3600) -> None:
    """Remove stale LanceDB transaction files from crashed previous runs.

    LanceDB writes *.txn files inside {table}/_transactions/ directories while a
    write is in progress. If the process crashes mid-transaction these files are
    left behind and cause the next run to fail during table initialisation with
    an OS-level exit code (e.g. 120) that is indistinguishable from a signal.

    This helper removes transaction files that are older than *stale_age_seconds*
    (default: 1 hour). It is intentionally non-fatal — any OSError is swallowed
    so a failed cleanup never blocks indexing.

    Args:
        index_path: Root of the index directory (contains the ``lance/`` sub-dir).
        stale_age_seconds: Files older than this many seconds are considered stale
            and will be removed. Default is 3600 (one hour).
    """
    import glob

    lance_dir = index_path / "lance"
    if not lance_dir.exists():
        return

    txn_pattern = str(lance_dir / "**" / "_transactions" / "*.txn")
    stale_cutoff = time.time() - stale_age_seconds
    cleaned = 0
    for txn_file in glob.glob(txn_pattern, recursive=True):
        try:
            if os.path.getmtime(txn_file) < stale_cutoff:
                os.remove(txn_file)
                cleaned += 1
                logger.debug("Removed stale LanceDB transaction file: %s", txn_file)
        except OSError:
            pass  # Non-fatal: never block indexing over a cleanup failure

    if cleaned:
        logger.info(
            "Cleaned up %d stale LanceDB transaction file(s) from previous crashed run",
            cleaned,
        )


def cleanup_stale_progress(index_path: Path, stale_age_seconds: int = 3600) -> None:
    """Remove a stale progress.json left by a crashed indexing run.

    If progress.json reports ``phase: "chunking"`` with zero files processed and
    the file itself is older than *stale_age_seconds* it almost certainly belongs
    to a run that crashed at initialisation (before any real work was done).
    Leaving it around causes subsequent runs to display misleading progress.

    Args:
        index_path: Root of the index directory (contains ``.mcp-vector-search/``).
        stale_age_seconds: Files older than this many seconds are considered stale.
            Default is 3600 (one hour).
    """
    progress_file = index_path / ".mcp-vector-search" / "progress.json"
    if not progress_file.exists():
        return

    try:
        age = time.time() - os.path.getmtime(str(progress_file))
        if age < stale_age_seconds:
            return  # Recent file — leave it alone

        with open(progress_file) as fh:
            data = json.load(fh)

        phase = data.get("phase", "")
        chunking = data.get("chunking", {})
        processed_files = chunking.get("processed_files", -1)

        if phase == "chunking" and processed_files == 0:
            os.remove(str(progress_file))
            logger.info(
                "Removed stale progress.json (phase=chunking, 0 files processed, "
                "age=%.0f s) — likely left by a crashed previous run",
                age,
            )
    except Exception:
        pass  # Non-fatal: never block indexing over a cleanup failure


def _detect_filesystem_type(db_path: Path) -> str:
    """Detect filesystem type of the database path for I/O optimization.

    Returns: "nfs" for NFS/EFS/CIFS mounts, "nvme" for local NVMe, "default" otherwise.

    Non-fatal: any error causes fallback to "default".
    """
    try:
        # Linux: parse /proc/mounts to find the mount for db_path
        if Path("/proc/mounts").exists():
            db_str = str(db_path.resolve())
            best_mount = ""
            best_fstype = "default"
            best_source = ""

            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    source = parts[0]
                    mount_point = parts[1]
                    fstype = parts[2].lower()
                    # Find longest matching mount point for db_path
                    if db_str.startswith(mount_point) and len(mount_point) > len(
                        best_mount
                    ):
                        best_mount = mount_point
                        best_fstype = fstype
                        best_source = source

            if best_fstype in ("nfs", "nfs4", "cifs", "smbfs", "efs"):
                return "nfs"

            # Check for NVMe: source device name contains "nvme" (e.g. /dev/nvme0n1p1)
            # OR mount point path suggests a dedicated NVMe mount (e.g. /mnt/nvme)
            if "nvme" in best_source.lower() or "/nvme" in best_mount.lower():
                return "nvme"

        # macOS / fallback: use stat -f to detect network filesystems
        import subprocess

        result = subprocess.run(  # nosec B607
            ["stat", "-f", "-c", "%T", str(db_path.resolve())],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0:
            fstype = result.stdout.strip().lower()
            if "nfs" in fstype or "smbfs" in fstype or "cifs" in fstype:
                return "nfs"

        return "default"
    except Exception:
        return "default"
