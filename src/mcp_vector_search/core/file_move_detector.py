"""Detect file moves/renames by matching content hashes.

This module-level helper was extracted from indexer.py so it can be
imported and tested independently of the full SemanticIndexer class.

Public API (re-exported from indexer.py for backward compatibility):
    detect_file_moves
"""

from __future__ import annotations

import time as _time
from pathlib import Path

from loguru import logger

from .chunks_backend import compute_file_hash


def detect_file_moves(
    project_root: Path,
    current_files: list[Path],
    indexed_file_hashes: dict[str, str],
    progress_tracker: object | None = None,
) -> tuple[list[tuple[str, str, str]], set[str], dict[Path, str]]:
    """Detect file moves/renames by matching content hashes.

    Compares the set of currently indexed paths against on-disk paths.
    When an indexed path is gone but a new path has the same SHA-256
    content hash, we treat that as a move rather than a delete+add.

    Only unambiguous 1-to-1 moves are handled.  If multiple indexed
    paths share the same hash, they are matched by sorted name order —
    this covers batch directory-rename scenarios while staying safe.

    Also returns the file_path -> hash mapping computed during the scan
    so callers can reuse it for change detection without re-hashing.

    Args:
        project_root: Root directory of the project (used for rel_path computation).
        current_files: All currently-discoverable files on disk.
        indexed_file_hashes: Mapping of rel_path -> file_hash from the DB.
        progress_tracker: Optional progress tracker for displaying scan progress.
            Must have a ``progress_bar_with_eta(current, total, prefix, start_time)``
            method (duck-typed, may be None).

    Returns:
        Tuple of:
        - List of (old_path, new_path, file_hash) for each detected move
        - Set of old_path strings that were moved (exclude from normal
          delete/re-process flow after caller reloads hashes)
        - Dict mapping absolute Path -> hash for all scanned files
          (reusable by change detection to avoid double-hashing)
    """
    # Reverse index: hash -> set of paths currently in the DB
    hash_to_indexed: dict[str, set[str]] = {}
    for path, hash_val in indexed_file_hashes.items():
        hash_to_indexed.setdefault(hash_val, set()).add(path)

    # Build set of relative paths that exist on disk right now, and a
    # reverse map from hash -> set of on-disk relative paths.
    # Also cache file_path -> hash so callers skip double-hashing.
    current_rel_paths: set[str] = set()
    current_hash_to_paths: dict[str, set[str]] = {}
    file_hash_cache: dict[Path, str] = {}
    total = len(current_files)
    scan_start = _time.time()
    for idx, file_path in enumerate(current_files, start=1):
        try:
            rel_path = str(file_path.relative_to(project_root))
            current_rel_paths.add(rel_path)
            file_hash = compute_file_hash(file_path)
            current_hash_to_paths.setdefault(file_hash, set()).add(rel_path)
            file_hash_cache[file_path] = file_hash
        except Exception:  # nosec B112
            continue

        # Progress feedback during hashing
        if progress_tracker:
            progress_tracker.progress_bar_with_eta(  # type: ignore[union-attr]
                current=idx,
                total=total,
                prefix="Scanning files",
                start_time=scan_start,
            )
        elif idx % 500 == 0:
            logger.info(f"Scanning files: {idx:,}/{total:,} hashed")

    if total >= 500:
        logger.info(f"Scanned {total:,} files in {_time.time() - scan_start:.1f}s")

    # Find moves: indexed paths that are no longer on disk (orphaned) paired
    # with new on-disk paths that share the same hash and are not yet indexed
    moves: list[tuple[str, str, str]] = []
    moved_old_paths: set[str] = set()

    for hash_val, indexed_paths in hash_to_indexed.items():
        # Paths in DB that are no longer on disk at their original location
        orphaned = indexed_paths - current_rel_paths
        if not orphaned:
            continue

        # On-disk paths with the same hash that are not yet in the DB
        new_paths = current_hash_to_paths.get(hash_val, set()) - set(
            indexed_file_hashes.keys()
        )
        if not new_paths:
            continue

        if len(orphaned) == 1 and len(new_paths) == 1:
            # Unambiguous 1-to-1 move
            old_path = next(iter(orphaned))
            new_path = next(iter(new_paths))
            moves.append((old_path, new_path, hash_val))
            moved_old_paths.add(old_path)
        elif len(orphaned) == len(new_paths):
            # Same number of orphaned and new paths with matching hash
            # (e.g., directory rename).  Match by sorted name for stability.
            for old_p, new_p in zip(sorted(orphaned), sorted(new_paths), strict=True):
                moves.append((old_p, new_p, hash_val))
                moved_old_paths.add(old_p)

    return moves, moved_old_paths, file_hash_cache
