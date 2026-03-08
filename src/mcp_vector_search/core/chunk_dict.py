"""Standalone helpers for converting CodeChunk objects to storage dicts.

These functions are extracted from SemanticIndexer to keep dict-construction
logic in one place and make it testable without the full indexer stack.
"""

from __future__ import annotations

import json

from .models import CodeChunk


def build_hierarchy_path(chunk: CodeChunk) -> str:
    """Build dotted hierarchy path (e.g., MyClass.my_method).

    Args:
        chunk: The CodeChunk to build a hierarchy path for.

    Returns:
        A dotted string like "ClassName.method_name", or "" when neither
        class_name nor function_name is set.
    """
    parts = []
    if chunk.class_name:
        parts.append(chunk.class_name)
    if chunk.function_name:
        parts.append(chunk.function_name)
    return ".".join(parts) if parts else ""


def chunk_to_storage_dict(
    chunk: CodeChunk,
    rel_path: str,
    file_hash: str | None = None,
) -> dict:
    """Convert a CodeChunk to the dict format expected by ChunksBackend.

    The returned dict contains every column that ``chunks.lance`` tracks.
    Pass *file_hash* when you want the deduplication key pre-injected (e.g.
    for the two-pass / batched-write paths); leave it ``None`` if you will
    inject it later.

    Args:
        chunk: Parsed code chunk.
        rel_path: Relative file path stored in the index (e.g. "src/foo.py").
        file_hash: Optional SHA-based hash of the source file.

    Returns:
        A dict ready to be appended to a ``chunk_dicts`` list and passed to
        ``ChunksBackend.add_chunks()`` or ``add_chunks_batch()``.
    """
    d: dict = {
        "chunk_id": chunk.chunk_id,
        "file_path": rel_path,
        "content": chunk.content,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "start_char": 0,
        "end_char": 0,
        "chunk_type": chunk.chunk_type,
        "name": chunk.function_name or chunk.class_name or "",
        "parent_name": "",
        "hierarchy_path": build_hierarchy_path(chunk),
        "docstring": chunk.docstring or "",
        "signature": "",
        "complexity": int(chunk.complexity_score),
        "token_count": len(chunk.content.split()),
        "last_author": chunk.last_author or "",
        "last_modified": chunk.last_modified or "",
        "commit_hash": chunk.commit_hash or "",
        "calls": chunk.calls or [],
        "imports": [
            json.dumps(imp) if isinstance(imp, dict) else imp
            for imp in (chunk.imports or [])
        ],
        "inherits_from": chunk.inherits_from or [],
    }
    if file_hash is not None:
        d["file_hash"] = file_hash
    return d
