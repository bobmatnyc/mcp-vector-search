"""Build BM25 keyword index from chunked content.

This module-level helper was extracted from indexer.py so it can be
imported and tested independently of the full SemanticIndexer class.

Public API (re-exported from indexer.py for backward compatibility):
    build_bm25_index
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from loguru import logger

from .bm25_backend import BM25Backend
from .chunks_backend import ChunksBackend


async def build_bm25_index(
    chunks_backend: ChunksBackend,
    mcp_dir: Path,
    progress_tracker: object | None = None,
) -> None:
    """Build BM25 index from all chunks for keyword search.

    This is Phase 3 of indexing (after chunks are built).
    BM25 index enables hybrid search by combining keyword and semantic search.
    BM25 only needs text content, not embeddings, so reads from chunks.lance.

    Args:
        chunks_backend: Initialised ChunksBackend instance containing chunk data.
        mcp_dir: Path to the ``.mcp-vector-search`` directory where the BM25
            index file (``bm25_index.pkl``) will be written.
        progress_tracker: Optional progress tracker for displaying tokenisation
            progress.  Must have a ``progress_bar_with_eta(current, total,
            prefix, start_time)`` method (duck-typed, may be None).
    """
    try:
        # Get all chunks from chunks.lance table
        # ChunksBackend has all chunk data including content (vectors not needed for BM25)
        if chunks_backend._db is None or chunks_backend._table is None:
            logger.warning("Chunks backend not initialized, skipping BM25 index build")
            return

        logger.info("Phase 3: Building BM25 index for keyword search...")

        # Read only the columns BM25 needs (skip large/unused columns like
        # embeddings, metadata, etc.) to reduce memory and I/O.
        bm25_columns = ["chunk_id", "content", "name", "file_path", "chunk_type"]

        def _read_bm25_columns():
            """Read only required columns via Lance scanner for speed."""
            try:
                # Use lance scanner with column projection (faster than to_pandas)
                scanner = chunks_backend._table.to_lance().scanner(columns=bm25_columns)
                table = scanner.to_table()
                return table.to_pandas()
            except Exception:
                # Fallback: read all columns if scanner fails
                return chunks_backend._table.to_pandas()

        df = await asyncio.to_thread(_read_bm25_columns)

        if df.empty:
            logger.info("No chunks in chunks table, skipping BM25 index build")
            return

        # Show chunk count after reading from database
        chunk_count = len(df)
        if progress_tracker:
            sys.stderr.write(
                f"  Building keyword search index ({chunk_count:,} chunks)...\n"
            )
            sys.stderr.flush()

        # Vectorised conversion: use to_dict("records") instead of iterrows()
        # (~10-50x faster for 200K+ rows).
        # Fill missing columns with defaults before converting.
        if "name" not in df.columns:
            df["name"] = ""
        else:
            df["name"] = df["name"].fillna("")
        if "chunk_type" not in df.columns:
            df["chunk_type"] = "code"
        else:
            df["chunk_type"] = df["chunk_type"].fillna("code")

        chunks_for_bm25 = df[bm25_columns].to_dict("records")

        logger.info(
            f"Phase 3: Building BM25 index from {len(chunks_for_bm25):,} chunks..."
        )

        # Build BM25 index with tokenization progress reporting
        bm25_start = (
            time.time()
        )  # must use time.time() to match progress_bar_with_eta epoch

        def _bm25_progress(current: int, total: int) -> None:
            if progress_tracker:
                progress_tracker.progress_bar_with_eta(  # type: ignore[union-attr]
                    current=current,
                    total=total,
                    prefix="Tokenizing chunks",
                    start_time=bm25_start,
                )

        bm25_backend = BM25Backend()
        bm25_backend.build_index(chunks_for_bm25, progress_callback=_bm25_progress)

        # Save to disk.
        bm25_path = mcp_dir / "bm25_index.pkl"
        bm25_backend.save(bm25_path)

        stats = bm25_backend.get_stats()
        logger.info(
            f"Phase 3 complete: BM25 index built with {stats['chunk_count']} chunks "
            f"(avg doc length: {stats['avg_doc_length']:.1f} tokens)"
        )

    except Exception as e:
        # Non-fatal: BM25 failure shouldn't break indexing
        logger.warning(f"BM25 index building failed (non-fatal): {e}")
        logger.warning("Hybrid search will fall back to vector-only mode")
