"""Build knowledge graph from indexed code chunks.

This module extracts entities and relationships from code chunks
and populates the Kuzu knowledge graph.
"""

from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .knowledge_graph import CodeEntity, CodeRelationship, KnowledgeGraph
from .models import CodeChunk

console = Console()


class KGBuilder:
    """Build knowledge graph from code chunks.

    This class processes code chunks and extracts:
    - Code entities (functions, classes, modules)
    - Relationships (calls, imports, inheritance)
    """

    def __init__(self, kg: KnowledgeGraph, project_root: Path):
        """Initialize KG builder.

        Args:
            kg: KnowledgeGraph instance
            project_root: Project root directory
        """
        self.kg = kg
        self.project_root = project_root
        self._entity_map: dict[str, str] = {}  # name -> chunk_id mapping

    async def build_from_chunks(
        self, chunks: list[CodeChunk], show_progress: bool = True
    ) -> dict[str, int]:
        """Build graph from code chunks.

        Args:
            chunks: List of code chunks to process
            show_progress: Whether to show progress bar

        Returns:
            Statistics dictionary with counts
        """
        # Filter to relevant chunks only
        relevant_chunks = [
            c
            for c in chunks
            if c.chunk_type in ["function", "method", "class", "module"]
        ]

        logger.info(
            f"Building knowledge graph from {len(relevant_chunks)} chunks "
            f"({len(chunks)} total)..."
        )

        stats = {
            "entities": 0,
            "calls": 0,
            "imports": 0,
            "inherits": 0,
            "contains": 0,
        }

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Building knowledge graph...[/cyan]"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn(
                    "[green]{task.fields[entities]} entities, "
                    "{task.fields[relationships]} relationships"
                ),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "kg_build",
                    total=len(relevant_chunks),
                    entities=0,
                    relationships=0,
                )

                for chunk in relevant_chunks:
                    await self._process_chunk(chunk, stats)
                    progress.update(
                        task,
                        advance=1,
                        entities=stats["entities"],
                        relationships=sum(stats.values()) - stats["entities"],
                    )
        else:
            for chunk in relevant_chunks:
                await self._process_chunk(chunk, stats)

        logger.info(
            f"✓ Knowledge graph built: {stats['entities']} entities, "
            f"{sum(stats.values()) - stats['entities']} relationships"
        )

        return stats

    async def _process_chunk(self, chunk: CodeChunk, stats: dict[str, int]):
        """Extract entities and relationships from a chunk.

        Args:
            chunk: Code chunk to process
            stats: Statistics dictionary to update
        """
        # Create entity for this chunk
        chunk_id = chunk.chunk_id or chunk.id
        name = chunk.function_name or chunk.class_name or "module"

        entity = CodeEntity(
            id=chunk_id,
            name=name,
            entity_type=chunk.chunk_type,
            file_path=str(chunk.file_path),
        )

        try:
            await self.kg.add_entity(entity)
            stats["entities"] += 1

            # Track entity name mapping for relationship resolution
            self._entity_map[name] = chunk_id

            # Process function calls
            if hasattr(chunk, "calls") and chunk.calls:
                for called in chunk.calls:
                    # Try to resolve target entity
                    target_id = self._resolve_entity(called)

                    if target_id:
                        rel = CodeRelationship(
                            source_id=chunk_id,
                            target_id=target_id,
                            relationship_type="calls",
                        )
                        await self.kg.add_relationship(rel)
                        stats["calls"] += 1

            # Process inheritance
            if hasattr(chunk, "inherits_from") and chunk.inherits_from:
                for base in chunk.inherits_from:
                    target_id = self._resolve_entity(base)

                    if target_id:
                        rel = CodeRelationship(
                            source_id=chunk_id,
                            target_id=target_id,
                            relationship_type="inherits",
                        )
                        await self.kg.add_relationship(rel)
                        stats["inherits"] += 1

            # Process imports (create module entities if needed)
            if hasattr(chunk, "imports") and chunk.imports:
                for imp in chunk.imports:
                    module = (
                        imp.get("module", "") if isinstance(imp, dict) else str(imp)
                    )
                    if module:
                        # Create module entity if it doesn't exist
                        module_id = f"module:{module}"
                        module_entity = CodeEntity(
                            id=module_id,
                            name=module,
                            entity_type="module",
                            file_path="",  # External module
                        )
                        await self.kg.add_entity(module_entity)

                        # Create import relationship
                        rel = CodeRelationship(
                            source_id=chunk_id,
                            target_id=module_id,
                            relationship_type="imports",
                        )
                        await self.kg.add_relationship(rel)
                        stats["imports"] += 1

            # Process parent-child (contains) relationships
            if hasattr(chunk, "parent_chunk_id") and chunk.parent_chunk_id:
                rel = CodeRelationship(
                    source_id=chunk.parent_chunk_id,
                    target_id=chunk_id,
                    relationship_type="contains",
                )
                await self.kg.add_relationship(rel)
                stats["contains"] += 1

        except Exception as e:
            logger.debug(f"Failed to process chunk {chunk_id}: {e}")

    def _resolve_entity(self, name: str) -> str | None:
        """Resolve entity name to chunk ID.

        Args:
            name: Entity name to resolve

        Returns:
            Chunk ID if found, None otherwise
        """
        # Direct lookup
        if name in self._entity_map:
            return self._entity_map[name]

        # Try without module prefix (e.g., "module.function" -> "function")
        if "." in name:
            short_name = name.split(".")[-1]
            if short_name in self._entity_map:
                return self._entity_map[short_name]

        return None

    async def build_from_database(
        self, database, show_progress: bool = True, limit: int | None = None
    ) -> dict[str, int]:
        """Build graph from all chunks in database.

        Args:
            database: VectorDatabase instance
            show_progress: Whether to show progress bar
            limit: Optional limit on number of chunks to process (for testing)

        Returns:
            Statistics dictionary
        """
        # Get chunk count first for progress reporting
        total_chunks = database.get_chunk_count()
        logger.info(f"Database has {total_chunks} total chunks")

        # Load chunks with progress reporting
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Loading chunks from database...[/cyan]"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("loading", total=total_chunks)

                chunks = []
                for batch in database.iter_chunks_batched(batch_size=5000):
                    chunks.extend(batch)
                    progress.update(task, advance=len(batch))

                    # Apply limit if specified
                    if limit and len(chunks) >= limit:
                        chunks = chunks[:limit]
                        progress.update(task, completed=total_chunks)
                        break
        else:
            logger.info("Loading chunks from database...")
            chunks = []
            for batch in database.iter_chunks_batched(batch_size=5000):
                chunks.extend(batch)
                if limit and len(chunks) >= limit:
                    chunks = chunks[:limit]
                    break

        logger.info(f"Loaded {len(chunks)} chunks for processing")

        # Build graph
        return await self.build_from_chunks(chunks, show_progress=show_progress)
