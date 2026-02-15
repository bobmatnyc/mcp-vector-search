"""Build knowledge graph from indexed code chunks.

This module extracts entities and relationships from code chunks
and populates the Kuzu knowledge graph.
"""

import re
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

from .knowledge_graph import CodeEntity, CodeRelationship, DocSection, KnowledgeGraph
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
        # Separate code and text chunks
        code_chunks = [
            c
            for c in chunks
            if c.chunk_type in ["function", "method", "class", "module"]
        ]

        text_chunks = [
            c
            for c in chunks
            if c.language == "text" or str(c.file_path).endswith(".md")
        ]

        logger.info(
            f"Building knowledge graph from {len(code_chunks)} code chunks "
            f"and {len(text_chunks)} text chunks ({len(chunks)} total)..."
        )

        stats = {
            "entities": 0,
            "doc_sections": 0,
            "calls": 0,
            "imports": 0,
            "inherits": 0,
            "contains": 0,
            "references": 0,
            "documents": 0,
            "follows": 0,
        }

        total_chunks = len(code_chunks) + len(text_chunks)

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
                    total=total_chunks,
                    entities=0,
                    relationships=0,
                )

                # Process code chunks
                for chunk in code_chunks:
                    await self._process_chunk(chunk, stats)
                    progress.update(
                        task,
                        advance=1,
                        entities=stats["entities"] + stats["doc_sections"],
                        relationships=sum(stats.values())
                        - stats["entities"]
                        - stats["doc_sections"],
                    )

                # Process text chunks
                for chunk in text_chunks:
                    await self._process_text_chunk(chunk, stats)
                    progress.update(
                        task,
                        advance=1,
                        entities=stats["entities"] + stats["doc_sections"],
                        relationships=sum(stats.values())
                        - stats["entities"]
                        - stats["doc_sections"],
                    )
        else:
            for chunk in code_chunks:
                await self._process_chunk(chunk, stats)

            for chunk in text_chunks:
                await self._process_text_chunk(chunk, stats)

        logger.info(
            f"✓ Knowledge graph built: {stats['entities']} code entities, "
            f"{stats['doc_sections']} doc sections, "
            f"{sum(stats.values()) - stats['entities'] - stats['doc_sections']} relationships"
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

    async def _process_text_chunk(self, chunk: CodeChunk, stats: dict[str, int]):
        """Extract documentation sections and references from text chunk.

        Args:
            chunk: Text chunk to process
            stats: Statistics dictionary to update
        """
        # Extract markdown headers from content
        headers = self._extract_headers(chunk.content, chunk.start_line)

        if not headers:
            return

        # Create doc sections and FOLLOWS relationships
        prev_section_id = None
        for header in headers:
            # Create unique ID for this section
            section_id = f"doc:{chunk.chunk_id}:{header['line']}"

            doc_section = DocSection(
                id=section_id,
                name=header["title"],
                file_path=str(chunk.file_path),
                level=header["level"],
                line_start=header["line"],
                line_end=header.get("end_line", chunk.end_line),
                doc_type="section",
            )

            try:
                await self.kg.add_doc_section(doc_section)
                stats["doc_sections"] += 1

                # Create FOLLOWS relationship (reading order)
                if prev_section_id:
                    rel = CodeRelationship(
                        source_id=prev_section_id,
                        target_id=section_id,
                        relationship_type="follows",
                    )
                    await self.kg.add_relationship(rel)
                    stats["follows"] += 1

                # Extract code references from section content
                section_content = self._extract_section_content(
                    chunk.content, header["line"] - chunk.start_line
                )
                code_refs = self._extract_code_refs(section_content)

                # Use NLP-extracted code refs if available
                if chunk.nlp_code_refs:
                    code_refs.extend(chunk.nlp_code_refs)

                # Create REFERENCES relationships
                for ref in code_refs:
                    target_id = self._resolve_entity(ref)
                    if target_id:
                        rel = CodeRelationship(
                            source_id=section_id,
                            target_id=target_id,
                            relationship_type="references",
                        )
                        await self.kg.add_relationship(rel)
                        stats["references"] += 1

                prev_section_id = section_id

            except Exception as e:
                logger.debug(f"Failed to process doc section: {e}")

    def _extract_headers(self, content: str, start_line: int) -> list[dict[str, any]]:
        """Extract markdown headers from content.

        Args:
            content: Markdown content
            start_line: Starting line number in file

        Returns:
            List of header dictionaries with title, level, line
        """
        headers = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Calculate line number
            line_offset = content[: match.start()].count("\n")
            line = start_line + line_offset

            headers.append({"level": level, "title": title, "line": line})

        return headers

    def _extract_section_content(self, content: str, header_line_offset: int) -> str:
        """Extract content for a specific section.

        Args:
            content: Full markdown content
            header_line_offset: Line offset of header within content

        Returns:
            Section content (up to next header of same or higher level)
        """
        lines = content.split("\n")
        if header_line_offset >= len(lines):
            return ""

        # Get section lines (until next header)
        section_lines = []
        for i in range(header_line_offset + 1, len(lines)):
            line = lines[i]
            # Stop at next header of same or higher level
            if re.match(r"^#{1,6}\s+", line):
                break
            section_lines.append(line)

        return "\n".join(section_lines)

    def _extract_code_refs(self, content: str) -> list[str]:
        """Extract backtick code references from content.

        Args:
            content: Text content

        Returns:
            List of code reference strings
        """
        # Match `code_ref` but not ```code blocks```
        refs = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", content)

        # Filter out inline code that's not a reference
        # Keep: function_name, ClassName, module.function
        # Skip: strings with spaces, numbers only
        filtered_refs = []
        for ref in refs:
            ref = ref.strip()
            # Skip empty, whitespace-only, or pure number refs
            if not ref or " " in ref or ref.isdigit():
                continue
            # Keep valid identifiers
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_\.]*$", ref):
                # Remove () if present (function calls)
                if ref.endswith("()"):
                    ref = ref[:-2]
                filtered_refs.append(ref)

        return list(set(filtered_refs))  # Remove duplicates

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
