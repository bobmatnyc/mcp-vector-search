#!/usr/bin/env python3
"""Migration script to add structural code metrics to existing ChromaDB chunks.

This script:
1. Connects to existing ChromaDB collection
2. For each chunk lacking metrics fields, adds default values
3. Logs progress and is idempotent (safe to run multiple times)

Usage:
    python scripts/migrate_chromadb_metrics.py
    python scripts/migrate_chromadb_metrics.py --persist-dir /path/to/chromadb
    python scripts/migrate_chromadb_metrics.py --dry-run  # Preview changes only

Default ChromaDB location: .mcp-vector-search/chromadb
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import chromadb
except ImportError:
    logger.error("ChromaDB not installed. Run: uv pip install chromadb")
    sys.exit(1)


# Default metrics for chunks without structural analysis
DEFAULT_METRICS: dict[str, Any] = {
    "cognitive_complexity": 0,
    "cyclomatic_complexity": 1,  # Base complexity
    "max_nesting_depth": 0,
    "parameter_count": 0,
    "lines_of_code": 0,
    "complexity_grade": "A",
    "code_smells": "[]",  # JSON string (ChromaDB doesn't support lists in metadata)
    "smell_count": 0,
}


def get_default_persist_dir() -> Path:
    """Get default ChromaDB persist directory."""
    return Path.cwd() / ".mcp-vector-search" / "chromadb"


async def migrate_metrics(
    persist_dir: Path,
    collection_name: str = "code_search",
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict[str, int]:
    """Migrate existing chunks to include structural metrics.

    Args:
        persist_dir: ChromaDB persist directory
        collection_name: Name of the collection to migrate
        dry_run: If True, preview changes without applying
        batch_size: Number of chunks to process per batch

    Returns:
        Dictionary with migration statistics
    """
    if not persist_dir.exists():
        logger.error(f"ChromaDB directory not found: {persist_dir}")
        return {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting migration...")
    logger.info(f"ChromaDB location: {persist_dir}")
    logger.info(f"Collection name: {collection_name}")

    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        logger.debug("Connected to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
        total_count = collection.count()
        logger.info(f"Collection contains {total_count} chunks")
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}")
        return {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

    if total_count == 0:
        logger.warning("Collection is empty, nothing to migrate")
        return {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

    # Process chunks in batches
    stats = {"total": total_count, "migrated": 0, "skipped": 0, "errors": 0}
    offset = 0

    while offset < total_count:
        # Fetch batch
        batch_limit = min(batch_size, total_count - offset)
        logger.debug(f"Processing batch: {offset}-{offset + batch_limit}")

        try:
            results = collection.get(
                include=["metadatas"],
                limit=batch_limit,
                offset=offset,
            )

            if not results or not results.get("ids"):
                logger.warning(f"No results returned for offset {offset}")
                break

            # Check which chunks need migration
            ids_to_update = []
            metadatas_to_update = []

            for chunk_id, metadata in zip(
                results["ids"], results["metadatas"], strict=False
            ):
                # Check if chunk already has metrics
                has_metrics = "cognitive_complexity" in metadata

                if has_metrics:
                    stats["skipped"] += 1
                    logger.debug(f"Chunk {chunk_id} already has metrics, skipping")
                    continue

                # Need to add metrics
                logger.debug(f"Chunk {chunk_id} needs metrics migration")

                # Create updated metadata with defaults
                updated_metadata = metadata.copy()
                updated_metadata.update(DEFAULT_METRICS)

                ids_to_update.append(chunk_id)
                metadatas_to_update.append(updated_metadata)

            # Update chunks (unless dry run)
            if ids_to_update:
                if dry_run:
                    logger.info(
                        f"[DRY RUN] Would update {len(ids_to_update)} chunks with default metrics"
                    )
                    stats["migrated"] += len(ids_to_update)
                else:
                    try:
                        collection.update(
                            ids=ids_to_update, metadatas=metadatas_to_update
                        )
                        stats["migrated"] += len(ids_to_update)
                        logger.info(
                            f"Migrated {len(ids_to_update)} chunks (total: {stats['migrated']}/{total_count})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to update batch: {e}")
                        stats["errors"] += len(ids_to_update)

            # Move to next batch
            offset += batch_limit

            # Yield to event loop
            await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            stats["errors"] += batch_limit
            offset += batch_limit

    # Log final statistics
    logger.info("Migration complete!")
    logger.info(f"Total chunks: {stats['total']}")
    logger.info(f"Migrated: {stats['migrated']}")
    logger.info(f"Skipped (already had metrics): {stats['skipped']}")
    logger.info(f"Errors: {stats['errors']}")

    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB chunks to include structural code metrics"
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=get_default_persist_dir(),
        help="ChromaDB persist directory (default: .mcp-vector-search/chromadb)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="code_search",
        help="Collection name (default: code_search)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to process per batch (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr, level=log_level, format="<level>{level: <8}</level> | {message}"
    )

    # Run migration
    try:
        stats = asyncio.run(
            migrate_metrics(
                persist_dir=args.persist_dir,
                collection_name=args.collection,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
        )

        # Return exit code based on results
        if stats["errors"] > 0:
            logger.error(f"Migration completed with {stats['errors']} errors")
            return 1
        elif stats["total"] == 0:
            logger.warning("No chunks found to migrate")
            return 0
        else:
            logger.success("Migration completed successfully!")
            return 0

    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
