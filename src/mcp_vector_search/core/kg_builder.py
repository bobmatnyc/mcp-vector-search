"""Build knowledge graph from indexed code chunks.

This module extracts entities and relationships from code chunks
and populates the Kuzu knowledge graph.
"""

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .knowledge_graph import (
    Branch,
    CodeEntity,
    CodeRelationship,
    Commit,
    DocSection,
    KnowledgeGraph,
    Person,
    ProgrammingFramework,
    ProgrammingLanguage,
    Project,
    Repository,
)
from .models import CodeChunk
from .resource_manager import get_batch_size_for_memory, get_configured_workers

console = Console()

# Generic entity names to exclude from KG (too common to be useful)
GENERIC_ENTITY_NAMES = {
    # Python builtins/common
    "main",
    "run",
    "test",
    "get",
    "set",
    "init",
    "__init__",
    "__main__",
    "setup",
    "config",
    "name",
    "value",
    "data",
    "result",
    "results",
    "item",
    "items",
    "key",
    "keys",
    "args",
    "kwargs",
    "self",
    "cls",
    # Single letters
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "i",
    "j",
    "k",
    "n",
    "x",
    "y",
    "z",
    # Common short names
    "id",
    "db",
    "fn",
    "cb",
    "err",
    "msg",
    "req",
    "res",
    "ctx",
    "env",
    # Common verbs (too generic)
    "add",
    "delete",
    "remove",
    "update",
    "create",
    "read",
    "write",
    "load",
    "save",
    "parse",
    "process",
    "handle",
    "execute",
    # Common nouns
    "file",
    "path",
    "module",
    "class",
    "function",
    "method",
    "list",
    "dict",
    "string",
    "int",
    "bool",
    "none",
    # Test-related
    "tests",
    "fixture",
    "mock",
}


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
        self._metadata_path = (
            project_root / ".mcp-vector-search" / "knowledge_graph" / "kg_metadata.json"
        )

        # Auto-configure based on memory
        self._workers = get_configured_workers()
        self._batch_size = get_batch_size_for_memory(
            item_size_kb=5,
            target_batch_mb=50,  # ~5KB per entity  # 50MB batches
        )
        logger.debug(
            f"KGBuilder: {self._workers} workers, batch_size={self._batch_size}"
        )

    def _load_metadata(self) -> dict | None:
        """Load KG metadata from disk.

        Returns:
            Metadata dictionary or None if not found
        """
        if not self._metadata_path.exists():
            return None

        try:
            with open(self._metadata_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load KG metadata: {e}")
            return None

    def _save_metadata(
        self,
        source_chunk_count: int,
        source_chunk_ids: set[str],
        entities_created: int,
        relationships_created: int,
        build_duration_seconds: float,
    ) -> None:
        """Save KG metadata to disk.

        Args:
            source_chunk_count: Number of chunks processed
            source_chunk_ids: Set of chunk IDs processed
            entities_created: Number of entities created
            relationships_created: Number of relationships created
            build_duration_seconds: Build duration in seconds
        """
        # Ensure directory exists
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute hash of chunk IDs for efficient comparison
        chunk_id_hash = hashlib.sha256(
            json.dumps(sorted(source_chunk_ids), sort_keys=True).encode()
        ).hexdigest()[:16]

        metadata = {
            "last_build": datetime.now(UTC).isoformat(),
            "source_chunk_count": source_chunk_count,
            "source_chunk_id_hash": chunk_id_hash,
            "source_chunk_ids": list(source_chunk_ids),  # Store all IDs
            "entities_created": entities_created,
            "relationships_created": relationships_created,
            "build_duration_seconds": build_duration_seconds,
        }

        try:
            with open(self._metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved KG metadata to {self._metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save KG metadata: {e}")

    def _get_processed_chunk_ids(self) -> set[str]:
        """Get set of chunk IDs from metadata.

        Returns:
            Set of previously processed chunk IDs
        """
        metadata = self._load_metadata()
        if not metadata or "source_chunk_ids" not in metadata:
            return set()

        return set(metadata["source_chunk_ids"])

    def _is_generic_entity(self, name: str) -> bool:
        """Check if entity name is too generic to be useful.

        Args:
            name: Entity name to check

        Returns:
            True if entity should be filtered out, False otherwise
        """
        if not name:
            return True

        # Filter very short names (2 chars or less)
        if len(name) <= 2:
            return True

        # Filter exact matches (case-insensitive)
        if name.lower() in GENERIC_ENTITY_NAMES:
            return True

        # Filter names starting with single underscore (private/internal, but not dunder)
        if name.startswith("_") and not name.startswith("__"):
            return True

        return False

    async def build_from_chunks(
        self,
        chunks: list[CodeChunk],
        show_progress: bool = True,
        skip_documents: bool = False,
    ) -> dict[str, int]:
        """Build graph from code chunks using batch inserts.

        Args:
            chunks: List of code chunks to process
            show_progress: Whether to show progress bar
            skip_documents: Skip expensive DOCUMENTS relationship extraction (default False)

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

        if not show_progress:
            logger.info(
                f"Building knowledge graph from {len(code_chunks)} code chunks "
                f"and {len(text_chunks)} text chunks ({len(chunks)} total)..."
            )

        stats = {
            "entities": 0,
            "doc_sections": 0,
            "tags": 0,
            "calls": 0,
            "imports": 0,
            "inherits": 0,
            "contains": 0,
            "references": 0,
            "documents": 0,
            "follows": 0,
            "has_tag": 0,
            "demonstrates": 0,
            "links_to": 0,
            "persons": 0,
            "projects": 0,
            "authored": 0,
            "modified": 0,
            "part_of": 0,
        }

        if show_progress:
            # Use Rich progress bars for visual feedback
            # IMPORTANT: Disable auto-refresh to avoid thread safety issues with Kuzu
            # The background refresh thread (default 4Hz) causes segfaults when
            # accessing Kuzu connection from multiple threads
            with (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    console=console,
                    auto_refresh=False,  # CRITICAL: Disable background thread entirely
                ) as progress
            ):
                # Phase 1: Extract entities and relationships
                task1 = progress.add_task(
                    "[cyan]🔍 Scanning chunks...",
                    total=len(code_chunks) + len(text_chunks),
                )

                code_entities: list[CodeEntity] = []
                doc_sections: list[DocSection] = []
                tags: set[str] = set()
                relationships: dict[str, list[CodeRelationship]] = {
                    "CALLS": [],
                    "IMPORTS": [],
                    "INHERITS": [],
                    "CONTAINS": [],
                    "REFERENCES": [],
                    "DOCUMENTS": [],
                    "FOLLOWS": [],
                    "HAS_TAG": [],
                    "DEMONSTRATES": [],
                    "LINKS_TO": [],
                }

                # Extract from code chunks
                for chunk in code_chunks:
                    entity, rels = self._extract_code_entity(chunk)
                    if entity:
                        code_entities.append(entity)
                        for rel_type, rel_list in rels.items():
                            relationships[rel_type].extend(rel_list)
                    progress.update(task1, advance=1)
                    progress.refresh()  # Manual refresh (no background thread)

                # Extract from text chunks
                for chunk in text_chunks:
                    docs, chunk_tags, rels = self._extract_doc_sections(chunk)
                    doc_sections.extend(docs)
                    tags.update(chunk_tags)
                    for rel_type, rel_list in rels.items():
                        relationships[rel_type].extend(rel_list)
                    progress.update(task1, advance=1)
                    progress.refresh()  # Manual refresh (no background thread)

                progress.update(
                    task1,
                    description=f"[green]✓ Scanned {len(chunks)} chunks",
                    completed=len(chunks),
                )
                progress.refresh()

                # Phase 2: Insert entities
                total_entities = len(code_entities) + len(doc_sections) + len(tags)
                task2 = progress.add_task(
                    "[cyan]🏗️  Extracting entities...", total=total_entities
                )

                if code_entities:
                    stats["entities"] = await self.kg.add_entities_batch(code_entities)
                    progress.update(task2, advance=len(code_entities))
                    progress.refresh()

                if doc_sections:
                    stats["doc_sections"] = await self.kg.add_doc_sections_batch(
                        doc_sections
                    )
                    progress.update(task2, advance=len(doc_sections))
                    progress.refresh()

                if tags:
                    stats["tags"] = await self.kg.add_tags_batch(list(tags))
                    progress.update(task2, advance=len(tags))
                    progress.refresh()

                progress.update(
                    task2,
                    description=f"[green]✓ Extracted {total_entities} entities",
                    completed=total_entities,
                )
                progress.refresh()

                # Phase 3: Insert relationships
                total_rels = sum(len(r) for r in relationships.values())
                task3 = progress.add_task(
                    "[cyan]🔗 Building relations...", total=total_rels
                )

                for rel_type, rels in relationships.items():
                    if rels:
                        count = await self.kg.add_relationships_batch(rels)
                        stats[rel_type.lower()] = count
                        progress.update(task3, advance=len(rels))
                        progress.refresh()

                progress.update(
                    task3,
                    description=f"[green]✓ Built {total_rels} relations",
                    completed=total_rels,
                )
                progress.refresh()

                # Phase 4: Extract DOCUMENTS relationships (optional)
                if not skip_documents and text_chunks and code_chunks:
                    task4 = progress.add_task(
                        "[cyan]📄 Extracting DOCUMENTS...",
                        total=len(text_chunks) * len(code_chunks),
                    )
                    await self._extract_documents_relationships(
                        text_chunks, stats, progress_task=task4, progress_obj=progress
                    )

        else:
            # No progress bars - original implementation
            logger.info("Phase 1: Extracting entities and relationships from chunks...")
            code_entities: list[CodeEntity] = []
            doc_sections: list[DocSection] = []
            tags: set[str] = set()
            relationships: dict[str, list[CodeRelationship]] = {
                "CALLS": [],
                "IMPORTS": [],
                "INHERITS": [],
                "CONTAINS": [],
                "REFERENCES": [],
                "DOCUMENTS": [],
                "FOLLOWS": [],
                "HAS_TAG": [],
                "DEMONSTRATES": [],
                "LINKS_TO": [],
            }

            # Extract from code chunks
            for chunk in code_chunks:
                entity, rels = self._extract_code_entity(chunk)
                if entity:
                    code_entities.append(entity)
                    for rel_type, rel_list in rels.items():
                        relationships[rel_type].extend(rel_list)

            # Extract from text chunks
            for chunk in text_chunks:
                docs, chunk_tags, rels = self._extract_doc_sections(chunk)
                doc_sections.extend(docs)
                tags.update(chunk_tags)
                for rel_type, rel_list in rels.items():
                    relationships[rel_type].extend(rel_list)

            logger.info(
                f"Extracted {len(code_entities)} entities, {len(doc_sections)} doc sections, "
                f"{len(tags)} tags, {sum(len(r) for r in relationships.values())} relationships"
            )

            # Phase 2: Batch insert entities
            logger.info("Phase 2: Batch inserting entities...")
            if code_entities:
                stats["entities"] = await self.kg.add_entities_batch(code_entities)
                logger.info(f"✓ Inserted {stats['entities']} code entities")

            if doc_sections:
                stats["doc_sections"] = await self.kg.add_doc_sections_batch(
                    doc_sections
                )
                logger.info(f"✓ Inserted {stats['doc_sections']} doc sections")

            if tags:
                stats["tags"] = await self.kg.add_tags_batch(list(tags))
                logger.info(f"✓ Inserted {stats['tags']} tags")

            # Phase 3: Batch insert relationships
            logger.info("Phase 3: Batch inserting relationships...")
            for rel_type, rels in relationships.items():
                if rels:
                    count = await self.kg.add_relationships_batch(rels)
                    stats[rel_type.lower()] = count
                    if count > 0:
                        logger.info(f"✓ Inserted {count} {rel_type} relationships")

            # Phase 4: Extract DOCUMENTS relationships (optional, expensive)
            if not skip_documents and text_chunks and code_chunks:
                logger.info(
                    "Phase 4: Extracting DOCUMENTS relationships (this may take a while)..."
                )
                await self._extract_documents_relationships(text_chunks, stats)

            logger.info(
                f"✓ Knowledge graph built: {stats['entities']} code entities, "
                f"{stats['doc_sections']} doc sections, "
                f"{sum(stats.values()) - stats['entities'] - stats['doc_sections']} relationships"
            )

        return stats

    def _extract_code_entity(
        self, chunk: CodeChunk
    ) -> tuple[CodeEntity | None, dict[str, list[CodeRelationship]]]:
        """Extract entity and relationships from code chunk (no DB writes).

        Args:
            chunk: Code chunk to process

        Returns:
            Tuple of (entity, relationships_by_type)
        """
        chunk_id = chunk.chunk_id or chunk.id
        name = chunk.function_name or chunk.class_name or "module"

        # Skip generic entity names
        if self._is_generic_entity(name):
            logger.debug(f"Skipping generic entity: {name}")
            return None, {}

        entity = CodeEntity(
            id=chunk_id,
            name=name,
            entity_type=chunk.chunk_type,
            file_path=str(chunk.file_path),
        )

        # Track entity name mapping for relationship resolution
        self._entity_map[name] = chunk_id

        relationships: dict[str, list[CodeRelationship]] = {
            "CALLS": [],
            "IMPORTS": [],
            "INHERITS": [],
            "CONTAINS": [],
        }

        # Process function calls
        if hasattr(chunk, "calls") and chunk.calls:
            for called in chunk.calls:
                target_id = self._resolve_entity(called)
                if target_id:
                    relationships["CALLS"].append(
                        CodeRelationship(
                            source_id=chunk_id,
                            target_id=target_id,
                            relationship_type="calls",
                        )
                    )

        # Process inheritance
        if hasattr(chunk, "inherits_from") and chunk.inherits_from:
            for base in chunk.inherits_from:
                target_id = self._resolve_entity(base)
                if target_id:
                    relationships["INHERITS"].append(
                        CodeRelationship(
                            source_id=chunk_id,
                            target_id=target_id,
                            relationship_type="inherits",
                        )
                    )

        # Process imports (create module entities)
        if hasattr(chunk, "imports") and chunk.imports:
            for imp in chunk.imports:
                module = imp.get("module", "") if isinstance(imp, dict) else str(imp)
                if module:
                    module_id = f"module:{module}"
                    # Module entities will be added separately
                    relationships["IMPORTS"].append(
                        CodeRelationship(
                            source_id=chunk_id,
                            target_id=module_id,
                            relationship_type="imports",
                        )
                    )

        # Process parent-child (contains) relationships
        if hasattr(chunk, "parent_chunk_id") and chunk.parent_chunk_id:
            relationships["CONTAINS"].append(
                CodeRelationship(
                    source_id=chunk.parent_chunk_id,
                    target_id=chunk_id,
                    relationship_type="contains",
                )
            )

        return entity, relationships

    async def _process_chunk(self, chunk: CodeChunk, stats: dict[str, int]):
        """Extract entities and relationships from a chunk.

        DEPRECATED: Use _extract_code_entity for batch processing.
        This method is kept for backwards compatibility.

        Args:
            chunk: Code chunk to process
            stats: Statistics dictionary to update
        """
        # Create entity for this chunk
        chunk_id = chunk.chunk_id or chunk.id
        name = chunk.function_name or chunk.class_name or "module"

        # Skip generic entity names
        if self._is_generic_entity(name):
            logger.debug(f"Skipping generic entity: {name}")
            return

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

    def _extract_doc_sections(
        self, chunk: CodeChunk
    ) -> tuple[list[DocSection], set[str], dict[str, list[CodeRelationship]]]:
        """Extract doc sections, tags, and relationships from text chunk (no DB writes).

        Args:
            chunk: Text chunk to process

        Returns:
            Tuple of (doc_sections, tags, relationships_by_type)
        """
        doc_sections: list[DocSection] = []
        tags: set[str] = set()
        relationships: dict[str, list[CodeRelationship]] = {
            "FOLLOWS": [],
            "HAS_TAG": [],
            "DEMONSTRATES": [],
            "LINKS_TO": [],
            "REFERENCES": [],
        }

        # Extract frontmatter metadata
        frontmatter = self._extract_frontmatter(chunk.content)
        tags_from_frontmatter = []
        related_docs = []

        if frontmatter:
            tags_data = frontmatter.get("tags", [])
            if isinstance(tags_data, list):
                tags_from_frontmatter = tags_data
            elif isinstance(tags_data, str):
                tags_from_frontmatter = [tags_data]

            related = frontmatter.get("related", [])
            if isinstance(related, list):
                related_docs = related
            elif isinstance(related, str):
                related_docs = [related]

        # Extract code blocks for DEMONSTRATES relationships
        code_blocks = self._extract_code_blocks(chunk.content)
        languages = set()
        for block in code_blocks:
            lang = block["language"]
            if lang and lang != "text":
                languages.add(lang)

        # Extract markdown headers
        headers = self._extract_headers(chunk.content, chunk.start_line)

        if not headers:
            return doc_sections, tags, relationships

        # Create doc sections and relationships
        prev_section_id = None
        for header in headers:
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
            doc_sections.append(doc_section)

            # Add frontmatter tags
            for tag in tags_from_frontmatter:
                tags.add(tag)
                relationships["HAS_TAG"].append(
                    CodeRelationship(
                        source_id=section_id,
                        target_id=f"tag:{tag}",
                        relationship_type="has_tag",
                    )
                )

            # Add language tags for code blocks
            for lang in languages:
                tags.add(f"lang:{lang}")
                relationships["DEMONSTRATES"].append(
                    CodeRelationship(
                        source_id=section_id,
                        target_id=f"tag:lang:{lang}",
                        relationship_type="demonstrates",
                    )
                )

            # Add LINKS_TO relationships
            for rel_doc in related_docs:
                relationships["LINKS_TO"].append(
                    CodeRelationship(
                        source_id=section_id,
                        target_id=f"doc:{rel_doc}",
                        relationship_type="links_to",
                    )
                )

            # Add FOLLOWS relationship
            if prev_section_id:
                relationships["FOLLOWS"].append(
                    CodeRelationship(
                        source_id=prev_section_id,
                        target_id=section_id,
                        relationship_type="follows",
                    )
                )

            # Extract code references
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
                    relationships["REFERENCES"].append(
                        CodeRelationship(
                            source_id=section_id,
                            target_id=target_id,
                            relationship_type="references",
                        )
                    )

            prev_section_id = section_id

        return doc_sections, tags, relationships

    async def _process_text_chunk(self, chunk: CodeChunk, stats: dict[str, int]):
        """Extract documentation sections and references from text chunk.

        DEPRECATED: Use _extract_doc_sections for batch processing.
        This method is kept for backwards compatibility.

        Args:
            chunk: Text chunk to process
            stats: Statistics dictionary to update
        """
        # Extract frontmatter metadata
        frontmatter = self._extract_frontmatter(chunk.content)
        tags_from_frontmatter = []
        related_docs = []

        if frontmatter:
            # Extract tags for HAS_TAG relationships
            tags = frontmatter.get("tags", [])
            if isinstance(tags, list):
                tags_from_frontmatter = tags
            elif isinstance(tags, str):
                tags_from_frontmatter = [tags]

            # Extract related docs for LINKS_TO relationships
            related = frontmatter.get("related", [])
            if isinstance(related, list):
                related_docs = related
            elif isinstance(related, str):
                related_docs = [related]

        # Extract code blocks for DEMONSTRATES relationships
        code_blocks = self._extract_code_blocks(chunk.content)
        languages = set()
        for block in code_blocks:
            lang = block["language"]
            if lang and lang != "text":
                languages.add(lang)

        # Track unique tags for accurate counting
        seen_tags = set()
        seen_langs = set()

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

                # Create HAS_TAG relationships for frontmatter tags
                for tag in tags_from_frontmatter:
                    # Add tag entity (only count once per unique tag)
                    await self.kg.add_tag(tag)
                    if tag not in seen_tags:
                        stats["tags"] += 1
                        seen_tags.add(tag)

                    # Create HAS_TAG relationship
                    rel = CodeRelationship(
                        source_id=section_id,
                        target_id=f"tag:{tag}",
                        relationship_type="has_tag",
                    )
                    await self.kg.add_relationship(rel)
                    stats["has_tag"] += 1

                # Create DEMONSTRATES relationships for code blocks
                for lang in languages:
                    # Add language as a tag (only count once per unique language)
                    await self.kg.add_tag(f"lang:{lang}")
                    if lang not in seen_langs:
                        stats["tags"] += 1
                        seen_langs.add(lang)

                    rel = CodeRelationship(
                        source_id=section_id,
                        target_id=f"tag:lang:{lang}",
                        relationship_type="demonstrates",
                    )
                    await self.kg.add_relationship(rel)
                    stats["demonstrates"] += 1

                # Create LINKS_TO relationships for related docs
                for rel_doc in related_docs:
                    rel = CodeRelationship(
                        source_id=section_id,
                        target_id=f"doc:{rel_doc}",
                        relationship_type="links_to",
                    )
                    await self.kg.add_relationship(rel)
                    stats["links_to"] += 1

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

    def _extract_frontmatter(self, content: str) -> dict | None:
        """Extract YAML frontmatter from markdown content.

        Frontmatter format:
        ---
        title: "Document Title"
        tags: [api, rest]
        related: [other-doc.md]
        category: guides
        ---

        Args:
            content: Markdown content

        Returns:
            Dictionary of frontmatter data or None if no frontmatter
        """
        if not content.startswith("---"):
            return None

        # Find closing ---
        end_match = re.search(r"\n---\n", content[3:])
        if not end_match:
            return None

        frontmatter_str = content[3 : end_match.start() + 3]
        try:
            return yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.debug(f"Failed to parse frontmatter: {e}")
            return None

    def _extract_code_blocks(self, content: str) -> list[dict]:
        """Extract fenced code blocks with language info.

        Args:
            content: Markdown content

        Returns:
            List of dicts with language, content, and start position
            Example: [{'language': 'python', 'content': '...', 'start': 10}, ...]
        """
        blocks = []
        pattern = r"```(\w*)\n(.*?)```"
        for match in re.finditer(pattern, content, re.DOTALL):
            blocks.append(
                {
                    "language": match.group(1) or "text",
                    "content": match.group(2),
                    "start": match.start(),
                }
            )
        return blocks

    def _compute_documents_score(
        self,
        doc_name: str,
        doc_content: str,
        doc_file_path: str,
        entity_name: str,
        entity_type: str,
        entity_file_path: str,
    ) -> float:
        """Score doc-entity semantic relevance (0.0-1.0).

        Scoring:
        - 0.4: Entity name in doc title
        - 0.2: Entity mentioned 2+ times in content
        - 0.3: README.md in same directory as entity
        - 0.1: Contextual keywords match entity type

        Args:
            doc_name: Documentation section title
            doc_content: Documentation section content
            doc_file_path: Documentation file path
            entity_name: Code entity name
            entity_type: Code entity type (function, class, module)
            entity_file_path: Code entity file path

        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        entity_lower = entity_name.lower()
        doc_name_lower = doc_name.lower()
        doc_content_lower = doc_content.lower()

        # Entity name in doc title (strong signal)
        if entity_lower in doc_name_lower:
            score += 0.4

        # Multiple mentions in content
        mentions = doc_content_lower.count(entity_lower)
        if mentions >= 2:
            score += 0.2
        elif mentions == 1:
            score += 0.1

        # File proximity (README.md for module)
        if self._is_readme_for_directory(doc_file_path, entity_file_path):
            score += 0.3

        # Contextual keywords
        type_keywords = {
            "function": ["function", "method", "returns", "parameters", "args"],
            "method": ["function", "method", "returns", "parameters", "args"],
            "class": ["class", "instance", "object", "inherits", "extends"],
            "module": ["module", "package", "import", "library"],
        }
        keywords = type_keywords.get(entity_type, [])
        if any(kw in doc_content_lower for kw in keywords):
            score += 0.1

        return min(score, 1.0)

    def _is_readme_for_directory(self, doc_path: str, code_path: str) -> bool:
        """Check if doc is README for code's directory.

        Args:
            doc_path: Documentation file path
            code_path: Code file path

        Returns:
            True if doc is README in same or parent directory of code
        """
        doc_p = Path(doc_path)
        code_p = Path(code_path)

        # README.md in same or parent directory of code
        if doc_p.name.lower() in ("readme.md", "readme.rst", "readme.txt"):
            code_dir = code_p.parent
            doc_dir = doc_p.parent
            return doc_dir == code_dir or doc_dir == code_dir.parent

        return False

    async def _extract_documents_relationships(
        self,
        doc_chunks: list[CodeChunk],
        stats: dict[str, int],
        progress_task=None,
        progress_obj=None,
    ) -> None:
        """Create DOCUMENTS edges between doc sections and code entities.

        Args:
            doc_chunks: Text/documentation chunks
            stats: Statistics dict to update
            progress_task: Optional Rich progress task ID
            progress_obj: Optional Rich progress object
        """
        threshold = 0.5
        documents_count = 0

        # Get all code entity info from KG
        entity_info = []
        try:
            result = self.kg.conn.execute(
                "MATCH (e:CodeEntity) RETURN e.id, e.name, e.entity_type, e.file_path"
            )
            while result.has_next():
                row = result.get_next()
                entity_info.append((row[0], row[1], row[2], row[3]))
        except Exception as e:
            logger.debug(f"Failed to fetch code entities: {e}")
            return

        if not progress_obj:
            logger.info(
                f"Matching {len(doc_chunks)} doc sections against {len(entity_info)} code entities..."
            )

        # Extract headers from doc chunks and create doc section IDs
        doc_sections = []
        for chunk in doc_chunks:
            headers = self._extract_headers(chunk.content, chunk.start_line)
            for header in headers:
                section_id = f"doc:{chunk.chunk_id}:{header['line']}"
                section_content = self._extract_section_content(
                    chunk.content, header["line"] - chunk.start_line
                )
                doc_sections.append(
                    {
                        "id": section_id,
                        "name": header["title"],
                        "content": section_content,
                        "file_path": str(chunk.file_path),
                    }
                )

        # Update progress total to actual doc sections count
        if progress_obj and progress_task:
            progress_obj.update(
                progress_task, total=len(doc_sections) * len(entity_info)
            )
            progress_obj.refresh()

        # Match doc sections against code entities
        processed = 0
        for doc_section in doc_sections:
            doc_id = doc_section["id"]
            doc_name = doc_section["name"]
            doc_content = doc_section["content"]
            doc_file = doc_section["file_path"]

            for entity_id, entity_name, entity_type, entity_file in entity_info:
                # Skip generic entity names using centralized filter
                if self._is_generic_entity(entity_name):
                    processed += 1
                    if progress_obj and progress_task:
                        progress_obj.update(progress_task, advance=1)
                        progress_obj.refresh()
                    continue

                score = self._compute_documents_score(
                    doc_name,
                    doc_content,
                    doc_file,
                    entity_name,
                    entity_type,
                    entity_file or "",
                )

                if score >= threshold:
                    rel = CodeRelationship(
                        source_id=doc_id,
                        target_id=entity_id,
                        relationship_type="documents",
                        weight=score,
                    )
                    try:
                        await self.kg.add_relationship(rel)
                        documents_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to add DOCUMENTS relationship: {e}")

                processed += 1
                if progress_obj and progress_task:
                    progress_obj.update(progress_task, advance=1)
                    progress_obj.refresh()

        stats["documents"] = documents_count

        if progress_obj and progress_task:
            progress_obj.update(
                progress_task,
                description=f"[green]✓ Extracted {documents_count} DOCUMENTS",
                completed=processed,
            )
            progress_obj.refresh()
        else:
            logger.info(f"✓ Created {documents_count} DOCUMENTS relationships")

    def _hash_email(self, email: str) -> str:
        """Hash email for privacy.

        Args:
            email: Email address to hash

        Returns:
            First 16 characters of SHA256 hash
        """
        return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

    async def _extract_git_authors(self, stats: dict) -> dict[str, Person]:
        """Extract Person entities from git log.

        Args:
            stats: Statistics dictionary to update

        Returns:
            Dictionary mapping person_id to Person
        """
        persons = {}

        try:
            # Get all commits with author info
            result = subprocess.run(
                ["git", "log", "--format=%H|%an|%ae|%aI", "--all"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    continue

                commit_sha, name, email, timestamp = parts[:4]
                email_hash = self._hash_email(email)
                person_id = f"person:{email_hash}"

                if person_id not in persons:
                    persons[person_id] = Person(
                        id=person_id,
                        name=name,
                        email_hash=email_hash,
                        commits_count=1,
                        first_commit=timestamp,
                        last_commit=timestamp,
                    )
                else:
                    persons[person_id].commits_count += 1
                    if timestamp > persons[person_id].last_commit:
                        persons[person_id].last_commit = timestamp
                    if timestamp < persons[person_id].first_commit:
                        persons[person_id].first_commit = timestamp

            # Add persons to KG
            for person in persons.values():
                await self.kg.add_person(person)

            stats["persons"] = len(persons)
            logger.info(f"✓ Extracted {len(persons)} person entities from git")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git log failed: {e}")
        except subprocess.TimeoutExpired:
            logger.warning("Git log timed out after 30 seconds")
        except FileNotFoundError:
            logger.debug("Git not found, skipping git author extraction")
        except Exception as e:
            logger.debug(f"Failed to extract git authors: {e}")

        return persons

    async def _extract_project_info(self, stats: dict) -> Project | None:
        """Extract Project entity from repo metadata.

        Args:
            stats: Statistics dictionary to update

        Returns:
            Project entity or None if extraction fails
        """
        project_name = self.project_root.name
        description = ""
        repo_url = ""

        # Try to get repo URL
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                repo_url = result.stdout.strip()
        except Exception:
            pass

        # Try to get description from pyproject.toml
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib

                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                if "project" in data:
                    project_name = data["project"].get("name", project_name)
                    description = data["project"].get("description", "")
            except Exception as e:
                logger.debug(f"Failed to parse pyproject.toml: {e}")

        project = Project(
            id=f"project:{project_name}",
            name=project_name,
            description=description,
            repo_url=repo_url,
        )

        await self.kg.add_project(project)
        stats["projects"] = 1
        logger.info(f"✓ Extracted project: {project_name}")

        return project

    def _detect_language_from_extension(self, file_path: str) -> str | None:
        """Detect programming language from file extension.

        Args:
            file_path: File path to detect language from

        Returns:
            Language name or None if not recognized
        """
        ext_map = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "objective-c",
            ".lua": "lua",
            ".pl": "perl",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
        }

        path = Path(file_path)
        suffix = path.suffix.lower()
        return ext_map.get(suffix)

    async def _extract_languages_and_frameworks(
        self, project: Project | None, stats: dict
    ) -> dict[str, ProgrammingLanguage]:
        """Extract programming languages and frameworks from project files.

        Args:
            project: Project entity (for USES_FRAMEWORK relationships)
            stats: Statistics dictionary to update

        Returns:
            Dictionary mapping language ID to ProgrammingLanguage entity
        """
        languages: dict[str, ProgrammingLanguage] = {}
        frameworks: list[ProgrammingFramework] = []

        # Detect languages from file extensions in CodeEntity nodes
        try:
            result = self.kg.conn.execute(
                "MATCH (e:CodeEntity) RETURN DISTINCT e.file_path"
            )
            file_paths = []
            while result.has_next():
                file_paths.append(result.get_next()[0])

            # Count language usage
            language_counts: dict[str, int] = {}
            for file_path in file_paths:
                lang = self._detect_language_from_extension(file_path)
                if lang:
                    language_counts[lang] = language_counts.get(lang, 0) + 1

            # Create ProgrammingLanguage nodes
            for lang_name, _count in language_counts.items():
                lang_id = f"lang:{lang_name}"
                extensions = self._get_extensions_for_language(lang_name)

                language = ProgrammingLanguage(
                    id=lang_id,
                    name=lang_name,
                    version="",  # Could be detected from config files
                    file_extensions=",".join(extensions),
                )
                await self.kg.add_programming_language(language)
                languages[lang_id] = language

            stats["languages"] = len(languages)
            if len(languages) > 0:
                logger.info(f"✓ Detected {len(languages)} programming languages")

        except Exception as e:
            logger.debug(f"Failed to extract languages: {e}")

        # Detect frameworks from config files
        try:
            # Python: pyproject.toml, requirements.txt
            python_frameworks = await self._detect_python_frameworks()
            frameworks.extend(python_frameworks)

            # JavaScript/TypeScript: package.json
            js_frameworks = await self._detect_javascript_frameworks()
            frameworks.extend(js_frameworks)

            # Rust: Cargo.toml
            rust_frameworks = await self._detect_rust_frameworks()
            frameworks.extend(rust_frameworks)

            # Go: go.mod
            go_frameworks = await self._detect_go_frameworks()
            frameworks.extend(go_frameworks)

            # Java: pom.xml, build.gradle
            java_frameworks = await self._detect_java_frameworks()
            frameworks.extend(java_frameworks)

            # Ruby: Gemfile
            ruby_frameworks = await self._detect_ruby_frameworks()
            frameworks.extend(ruby_frameworks)

            # PHP: composer.json
            php_frameworks = await self._detect_php_frameworks()
            frameworks.extend(php_frameworks)

            # C#/.NET: *.csproj
            csharp_frameworks = await self._detect_csharp_frameworks()
            frameworks.extend(csharp_frameworks)

            # Swift: Package.swift
            swift_frameworks = await self._detect_swift_frameworks()
            frameworks.extend(swift_frameworks)

            # Kotlin: build.gradle.kts
            kotlin_frameworks = await self._detect_kotlin_frameworks()
            frameworks.extend(kotlin_frameworks)

            # Add framework nodes
            for framework in frameworks:
                await self.kg.add_programming_framework(framework)

                # Add FRAMEWORK_FOR relationship
                if framework.language_id in languages:
                    await self.kg.add_framework_for_relationship(
                        framework.id, framework.language_id
                    )

                # Add USES_FRAMEWORK relationship
                if project:
                    await self.kg.add_uses_framework_relationship(
                        project.id, framework.id
                    )

            stats["frameworks"] = len(frameworks)
            if len(frameworks) > 0:
                logger.info(f"✓ Detected {len(frameworks)} frameworks")

        except Exception as e:
            logger.debug(f"Failed to extract frameworks: {e}")

        return languages

    def _get_extensions_for_language(self, language: str) -> list[str]:
        """Get file extensions for a programming language.

        Args:
            language: Language name

        Returns:
            List of file extensions (e.g., [".py", ".pyi"])
        """
        ext_map = {
            "python": [".py", ".pyi"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "rust": [".rs"],
            "go": [".go"],
            "java": [".java"],
            "ruby": [".rb"],
            "php": [".php"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
            "csharp": [".cs"],
            "swift": [".swift"],
            "kotlin": [".kt"],
            "scala": [".scala"],
            "r": [".r"],
            "objective-c": [".m"],
            "lua": [".lua"],
            "perl": [".pl"],
            "shell": [".sh", ".bash", ".zsh"],
        }
        return ext_map.get(language, [])

    async def _detect_python_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Python frameworks from pyproject.toml and requirements.txt.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        # Framework detection patterns
        framework_patterns = {
            "fastapi": ("web", "FastAPI"),
            "django": ("web", "Django"),
            "flask": ("web", "Flask"),
            "pytest": ("testing", "pytest"),
            "unittest": ("testing", "unittest"),
            "sqlalchemy": ("orm", "SQLAlchemy"),
            "pydantic": ("validation", "Pydantic"),
            "numpy": ("scientific", "NumPy"),
            "pandas": ("data", "Pandas"),
            "requests": ("http", "Requests"),
            "aiohttp": ("http", "aiohttp"),
            "click": ("cli", "Click"),
            "typer": ("cli", "Typer"),
        }

        # Check pyproject.toml
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib

                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)

                # Check dependencies
                deps = []
                if "project" in data and "dependencies" in data["project"]:
                    deps.extend(data["project"]["dependencies"])
                if "tool" in data and "poetry" in data["tool"]:
                    poetry_deps = data["tool"]["poetry"].get("dependencies", {})
                    deps.extend(poetry_deps.keys())

                for dep in deps:
                    # Parse dependency (e.g., "fastapi>=0.100.0" -> "fastapi")
                    package_name = (
                        dep.split("[")[0].split(">=")[0].split("==")[0].strip().lower()
                    )

                    if package_name in framework_patterns:
                        category, display_name = framework_patterns[package_name]
                        frameworks.append(
                            ProgrammingFramework(
                                id=f"framework:{package_name}",
                                name=display_name,
                                version="",  # Could parse from dependency spec
                                language_id="lang:python",
                                category=category,
                            )
                        )

            except Exception as e:
                logger.debug(f"Failed to parse pyproject.toml: {e}")

        # Check requirements.txt
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            try:
                with open(requirements) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        # Parse package name
                        package_name = (
                            line.split("[")[0]
                            .split(">=")[0]
                            .split("==")[0]
                            .strip()
                            .lower()
                        )

                        if package_name in framework_patterns:
                            category, display_name = framework_patterns[package_name]
                            framework_id = f"framework:{package_name}"

                            # Avoid duplicates
                            if not any(f.id == framework_id for f in frameworks):
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=framework_id,
                                        name=display_name,
                                        version="",
                                        language_id="lang:python",
                                        category=category,
                                    )
                                )

            except Exception as e:
                logger.debug(f"Failed to parse requirements.txt: {e}")

        return frameworks

    async def _detect_javascript_frameworks(self) -> list[ProgrammingFramework]:
        """Detect JavaScript/TypeScript frameworks from package.json.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "react": ("web", "React"),
            "vue": ("web", "Vue"),
            "angular": ("web", "Angular"),
            "express": ("web", "Express"),
            "next": ("web", "Next.js"),
            "nuxt": ("web", "Nuxt.js"),
            "jest": ("testing", "Jest"),
            "mocha": ("testing", "Mocha"),
            "vitest": ("testing", "Vitest"),
            "axios": ("http", "Axios"),
        }

        package_json = self.project_root / "package.json"
        if package_json.exists():
            try:
                import json

                with open(package_json) as f:
                    data = json.load(f)

                deps = []
                deps.extend(data.get("dependencies", {}).keys())
                deps.extend(data.get("devDependencies", {}).keys())

                for dep in deps:
                    dep_lower = dep.lower()
                    if dep_lower in framework_patterns:
                        category, display_name = framework_patterns[dep_lower]
                        lang_id = (
                            "lang:typescript"
                            if (self.project_root / "tsconfig.json").exists()
                            else "lang:javascript"
                        )

                        frameworks.append(
                            ProgrammingFramework(
                                id=f"framework:{dep_lower}",
                                name=display_name,
                                version=data.get("dependencies", {}).get(dep, "")
                                or data.get("devDependencies", {}).get(dep, ""),
                                language_id=lang_id,
                                category=category,
                            )
                        )

            except Exception as e:
                logger.debug(f"Failed to parse package.json: {e}")

        return frameworks

    async def _detect_rust_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Rust frameworks from Cargo.toml.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "actix-web": ("web", "Actix Web"),
            "rocket": ("web", "Rocket"),
            "tokio": ("async", "Tokio"),
            "serde": ("serialization", "Serde"),
        }

        cargo_toml = self.project_root / "Cargo.toml"
        if cargo_toml.exists():
            try:
                import tomllib

                with open(cargo_toml, "rb") as f:
                    data = tomllib.load(f)

                deps = data.get("dependencies", {}).keys()

                for dep in deps:
                    if dep in framework_patterns:
                        category, display_name = framework_patterns[dep]
                        frameworks.append(
                            ProgrammingFramework(
                                id=f"framework:{dep}",
                                name=display_name,
                                version="",
                                language_id="lang:rust",
                                category=category,
                            )
                        )

            except Exception as e:
                logger.debug(f"Failed to parse Cargo.toml: {e}")

        return frameworks

    async def _detect_go_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Go frameworks from go.mod.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "gin": ("web", "Gin"),
            "echo": ("web", "Echo"),
            "fiber": ("web", "Fiber"),
            "gorm": ("orm", "GORM"),
        }

        go_mod = self.project_root / "go.mod"
        if go_mod.exists():
            try:
                with open(go_mod) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("require"):
                            continue

                        for pattern, (
                            category,
                            display_name,
                        ) in framework_patterns.items():
                            if pattern in line.lower():
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=f"framework:{pattern}",
                                        name=display_name,
                                        version="",
                                        language_id="lang:go",
                                        category=category,
                                    )
                                )
                                break

            except Exception as e:
                logger.debug(f"Failed to parse go.mod: {e}")

        return frameworks

    async def _detect_java_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Java frameworks from pom.xml and build.gradle.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "spring-boot": ("web", "Spring Boot"),
            "spring": ("web", "Spring Framework"),
            "hibernate": ("orm", "Hibernate"),
            "junit": ("testing", "JUnit"),
            "testng": ("testing", "TestNG"),
            "log4j": ("logging", "Log4j"),
            "slf4j": ("logging", "SLF4J"),
            "jackson": ("serialization", "Jackson"),
            "gson": ("serialization", "Gson"),
            "mockito": ("testing", "Mockito"),
        }

        # Parse pom.xml (Maven)
        pom_xml = self.project_root / "pom.xml"
        if pom_xml.exists():
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(pom_xml)
                root = tree.getroot()

                # Extract namespace if present
                namespace = ""
                if root.tag.startswith("{"):
                    namespace = root.tag[: root.tag.index("}") + 1]

                # Check dependencies
                for dep in root.findall(f".//{namespace}dependency"):
                    artifact_id = dep.find(f"{namespace}artifactId")
                    if artifact_id is not None and artifact_id.text:
                        artifact_lower = artifact_id.text.lower()

                        # Check for pattern matches
                        for pattern, (
                            category,
                            display_name,
                        ) in framework_patterns.items():
                            if pattern in artifact_lower:
                                framework_id = f"framework:{pattern}"

                                # Avoid duplicates
                                if not any(f.id == framework_id for f in frameworks):
                                    frameworks.append(
                                        ProgrammingFramework(
                                            id=framework_id,
                                            name=display_name,
                                            version="",
                                            language_id="lang:java",
                                            category=category,
                                        )
                                    )
                                break

            except Exception as e:
                logger.debug(f"Failed to parse pom.xml: {e}")

        # Parse build.gradle (Gradle)
        build_gradle = self.project_root / "build.gradle"
        build_gradle_kts = self.project_root / "build.gradle.kts"

        gradle_file = build_gradle if build_gradle.exists() else build_gradle_kts
        if gradle_file.exists():
            try:
                with open(gradle_file) as f:
                    content = f.read()

                    for pattern, (category, display_name) in framework_patterns.items():
                        # Match dependencies like: implementation 'org.springframework.boot:spring-boot-starter-web'
                        if pattern in content.lower():
                            framework_id = f"framework:{pattern}"

                            # Avoid duplicates
                            if not any(f.id == framework_id for f in frameworks):
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=framework_id,
                                        name=display_name,
                                        version="",
                                        language_id="lang:java",
                                        category=category,
                                    )
                                )

            except Exception as e:
                logger.debug(f"Failed to parse {gradle_file.name}: {e}")

        return frameworks

    async def _detect_ruby_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Ruby frameworks from Gemfile.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "rails": ("web", "Ruby on Rails"),
            "sinatra": ("web", "Sinatra"),
            "rspec": ("testing", "RSpec"),
            "minitest": ("testing", "Minitest"),
            "sidekiq": ("background", "Sidekiq"),
            "activerecord": ("orm", "ActiveRecord"),
            "devise": ("auth", "Devise"),
            "pundit": ("authorization", "Pundit"),
            "factory_bot": ("testing", "FactoryBot"),
        }

        gemfile = self.project_root / "Gemfile"
        if gemfile.exists():
            try:
                with open(gemfile) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        # Match gem declarations: gem 'rails', '~> 7.0'
                        if line.startswith("gem"):
                            for pattern, (
                                category,
                                display_name,
                            ) in framework_patterns.items():
                                if pattern in line.lower():
                                    framework_id = f"framework:{pattern}"

                                    # Avoid duplicates
                                    if not any(
                                        f.id == framework_id for f in frameworks
                                    ):
                                        frameworks.append(
                                            ProgrammingFramework(
                                                id=framework_id,
                                                name=display_name,
                                                version="",
                                                language_id="lang:ruby",
                                                category=category,
                                            )
                                        )
                                    break

            except Exception as e:
                logger.debug(f"Failed to parse Gemfile: {e}")

        return frameworks

    async def _detect_php_frameworks(self) -> list[ProgrammingFramework]:
        """Detect PHP frameworks from composer.json.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "laravel": ("web", "Laravel"),
            "symfony": ("web", "Symfony"),
            "phpunit": ("testing", "PHPUnit"),
            "doctrine": ("orm", "Doctrine"),
            "guzzle": ("http", "Guzzle"),
            "monolog": ("logging", "Monolog"),
            "twig": ("templating", "Twig"),
            "pest": ("testing", "Pest"),
        }

        composer_json = self.project_root / "composer.json"
        if composer_json.exists():
            try:
                with open(composer_json) as f:
                    data = json.load(f)

                deps = []
                deps.extend(data.get("require", {}).keys())
                deps.extend(data.get("require-dev", {}).keys())

                for dep in deps:
                    dep_lower = dep.lower()

                    for pattern, (category, display_name) in framework_patterns.items():
                        if pattern in dep_lower:
                            framework_id = f"framework:{pattern}"

                            # Avoid duplicates
                            if not any(f.id == framework_id for f in frameworks):
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=framework_id,
                                        name=display_name,
                                        version="",
                                        language_id="lang:php",
                                        category=category,
                                    )
                                )
                            break

            except Exception as e:
                logger.debug(f"Failed to parse composer.json: {e}")

        return frameworks

    async def _detect_csharp_frameworks(self) -> list[ProgrammingFramework]:
        """Detect C#/.NET frameworks from *.csproj files.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "microsoft.aspnetcore": ("web", "ASP.NET Core"),
            "entityframework": ("orm", "Entity Framework"),
            "xunit": ("testing", "xUnit"),
            "nunit": ("testing", "NUnit"),
            "serilog": ("logging", "Serilog"),
            "automapper": ("mapping", "AutoMapper"),
            "newtonsoft.json": ("serialization", "Json.NET"),
            "fluentvalidation": ("validation", "FluentValidation"),
        }

        # Find all .csproj files
        csproj_files = list(self.project_root.glob("**/*.csproj"))

        for csproj_file in csproj_files:
            try:
                import xml.etree.ElementTree as ET

                tree = ET.parse(csproj_file)
                root = tree.getroot()

                # Check PackageReference elements
                for package_ref in root.findall(".//PackageReference"):
                    include = package_ref.get("Include")
                    if include:
                        include_lower = include.lower()

                        for pattern, (
                            category,
                            display_name,
                        ) in framework_patterns.items():
                            if pattern in include_lower:
                                framework_id = f"framework:{pattern.split('.')[0]}"

                                # Avoid duplicates
                                if not any(f.id == framework_id for f in frameworks):
                                    frameworks.append(
                                        ProgrammingFramework(
                                            id=framework_id,
                                            name=display_name,
                                            version="",
                                            language_id="lang:csharp",
                                            category=category,
                                        )
                                    )
                                break

            except Exception as e:
                logger.debug(f"Failed to parse {csproj_file.name}: {e}")

        return frameworks

    async def _detect_swift_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Swift frameworks from Package.swift.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "vapor": ("web", "Vapor"),
            "swiftui": ("ui", "SwiftUI"),
            "alamofire": ("http", "Alamofire"),
            "combine": ("reactive", "Combine"),
            "swiftnio": ("async", "SwiftNIO"),
        }

        package_swift = self.project_root / "Package.swift"
        if package_swift.exists():
            try:
                with open(package_swift) as f:
                    content = f.read().lower()

                    for pattern, (category, display_name) in framework_patterns.items():
                        if pattern in content:
                            framework_id = f"framework:{pattern}"

                            # Avoid duplicates
                            if not any(f.id == framework_id for f in frameworks):
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=framework_id,
                                        name=display_name,
                                        version="",
                                        language_id="lang:swift",
                                        category=category,
                                    )
                                )

            except Exception as e:
                logger.debug(f"Failed to parse Package.swift: {e}")

        return frameworks

    async def _detect_kotlin_frameworks(self) -> list[ProgrammingFramework]:
        """Detect Kotlin frameworks from build.gradle.kts.

        Returns:
            List of detected ProgrammingFramework entities
        """
        frameworks = []

        framework_patterns = {
            "ktor": ("web", "Ktor"),
            "spring": ("web", "Spring"),
            "exposed": ("orm", "Exposed"),
            "koin": ("di", "Koin"),
            "coroutines": ("async", "Kotlin Coroutines"),
            "kotest": ("testing", "Kotest"),
        }

        build_gradle_kts = self.project_root / "build.gradle.kts"
        if build_gradle_kts.exists():
            try:
                with open(build_gradle_kts) as f:
                    content = f.read().lower()

                    for pattern, (category, display_name) in framework_patterns.items():
                        if pattern in content:
                            framework_id = f"framework:{pattern}"

                            # Avoid duplicates
                            if not any(f.id == framework_id for f in frameworks):
                                frameworks.append(
                                    ProgrammingFramework(
                                        id=framework_id,
                                        name=display_name,
                                        version="",
                                        language_id="lang:kotlin",
                                        category=category,
                                    )
                                )

            except Exception as e:
                logger.debug(f"Failed to parse build.gradle.kts: {e}")

        return frameworks

    async def _extract_git_history(
        self, stats: dict
    ) -> tuple[Repository | None, dict[str, Branch], dict[str, Commit]]:
        """Extract Repository, Branch, and Commit entities from git history.

        Args:
            stats: Statistics dictionary to update

        Returns:
            Tuple of (repository, branches_dict, commits_dict)
        """
        repository = None
        branches = {}
        commits = {}

        try:
            # Get repository info
            repo_name = self.project_root.name
            repo_url = ""
            default_branch = "main"

            # Get remote URL
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    repo_url = result.stdout.strip()
            except Exception:
                pass

            # Get default branch
            try:
                result = subprocess.run(
                    ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Extract branch name from refs/remotes/origin/main
                    default_branch = result.stdout.strip().split("/")[-1]
            except Exception:
                pass

            # Get first commit timestamp
            first_commit_time = None
            try:
                result = subprocess.run(
                    ["git", "log", "--reverse", "--format=%aI", "--max-count=1"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    first_commit_time = result.stdout.strip()
            except Exception:
                pass

            # Create repository entity
            repository = Repository(
                id=f"repo:{repo_name}",
                name=repo_name,
                url=repo_url,
                default_branch=default_branch,
                created_at=first_commit_time,
            )
            await self.kg.add_repository(repository)
            stats["repositories"] = 1
            logger.info(f"✓ Extracted repository: {repo_name}")

            # Get all branches
            try:
                result = subprocess.run(
                    [
                        "git",
                        "for-each-ref",
                        "--format=%(refname:short)|%(creatordate:iso8601)",
                        "refs/heads/",
                    ],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if not line:
                            continue
                        parts = line.split("|")
                        if len(parts) < 2:
                            continue

                        branch_name = parts[0]
                        created_at = parts[1]
                        is_default = branch_name == default_branch

                        branch = Branch(
                            id=f"branch:{repo_name}:{branch_name}",
                            name=branch_name,
                            repository_id=repository.id,
                            is_default=is_default,
                            created_at=created_at,
                        )
                        await self.kg.add_branch(branch)
                        branches[branch_name] = branch

                        # Create BELONGS_TO relationship
                        await self.kg.add_belongs_to_relationship(
                            branch.id, repository.id
                        )

                    stats["branches"] = len(branches)
                    logger.info(f"✓ Extracted {len(branches)} branches")
            except Exception as e:
                logger.debug(f"Failed to extract branches: {e}")

            # Get recent commits (limit to 500 for performance)
            try:
                result = subprocess.run(
                    ["git", "log", "--format=%H|%s|%an|%ae|%aI", "--all", "-n", "500"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if not line:
                            continue
                        parts = line.split("|")
                        if len(parts) < 5:
                            continue

                        sha = parts[0]
                        message = parts[1]
                        author_name = parts[2]
                        author_email = parts[3]
                        timestamp = parts[4]

                        commit = Commit(
                            id=f"commit:{sha[:12]}",  # Use short SHA for ID
                            sha=sha,
                            message=message,
                            author_name=author_name,
                            author_email=author_email,
                            timestamp=timestamp,
                        )
                        await self.kg.add_commit(commit)
                        commits[sha] = commit

                    stats["commits"] = len(commits)
                    logger.info(f"✓ Extracted {len(commits)} commits")
            except Exception as e:
                logger.debug(f"Failed to extract commits: {e}")

            # Create COMMITTED_TO relationships (commit -> branch)
            # Map commits to branches they belong to
            if commits and branches:
                try:
                    for branch_name, branch in branches.items():
                        # Get commits for this branch (limit to 100 per branch)
                        result = subprocess.run(
                            ["git", "log", "--format=%H", branch_name, "-n", "100"],
                            cwd=self.project_root,
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if result.returncode == 0:
                            commit_count = 0
                            for sha in result.stdout.strip().split("\n"):
                                if sha and sha in commits:
                                    await self.kg.add_committed_to_relationship(
                                        commits[sha].id, branch.id
                                    )
                                    commit_count += 1
                            logger.debug(
                                f"Created {commit_count} COMMITTED_TO relationships for branch {branch_name}"
                            )
                except Exception as e:
                    logger.debug(f"Failed to create COMMITTED_TO relationships: {e}")

        except FileNotFoundError:
            logger.debug("Git not found, skipping git history extraction")
        except Exception as e:
            logger.debug(f"Failed to extract git history: {e}")

        return repository, branches, commits

    async def _extract_modifies_relationships(
        self, commits: dict[str, Commit], stats: dict
    ) -> None:
        """Extract MODIFIES relationships between commits and code entities.

        Args:
            commits: Dictionary of commit SHA to Commit entity
            stats: Statistics dictionary to update
        """
        if not commits:
            return

        modifies_count = 0

        try:
            # Get file paths for all code entities
            entity_files: dict[str, list[str]] = {}  # file_path -> [entity_ids]
            try:
                result = self.kg.conn.execute(
                    "MATCH (e:CodeEntity) RETURN e.id, e.file_path"
                )
                while result.has_next():
                    row = result.get_next()
                    entity_id, file_path = row[0], row[1]
                    if file_path:
                        # Normalize to relative path
                        try:
                            rel_path = str(
                                Path(file_path).relative_to(self.project_root)
                            )
                        except ValueError:
                            rel_path = str(file_path)

                        if rel_path not in entity_files:
                            entity_files[rel_path] = []
                        entity_files[rel_path].append(entity_id)
            except Exception as e:
                logger.debug(f"Failed to fetch entity file paths: {e}")
                return

            # For each commit, find modified files using git diff
            for sha, commit in list(commits.items())[:100]:  # Limit to 100 commits
                try:
                    result = subprocess.run(
                        ["git", "diff", "--numstat", f"{sha}^..{sha}"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        continue

                    # Parse diff output: "lines_added lines_deleted file_path"
                    for line in result.stdout.strip().split("\n"):
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) < 3:
                            continue

                        added = parts[0]
                        deleted = parts[1]
                        file_path = parts[2]

                        # Convert to int (handle binary files marked as "-")
                        try:
                            lines_added = int(added) if added != "-" else 0
                            lines_deleted = int(deleted) if deleted != "-" else 0
                        except ValueError:
                            continue

                        # Find entities for this file
                        if file_path in entity_files:
                            for entity_id in entity_files[file_path]:
                                try:
                                    await self.kg.add_modifies_relationship(
                                        commit.id,
                                        entity_id,
                                        lines_added,
                                        lines_deleted,
                                    )
                                    modifies_count += 1
                                except Exception:
                                    pass

                except subprocess.TimeoutExpired:
                    logger.debug(f"Git diff timed out for commit {sha}")
                except Exception as e:
                    logger.debug(f"Failed to process commit {sha}: {e}")

            stats["modifies"] = modifies_count
            logger.info(f"✓ Created {modifies_count} MODIFIES relationships")

        except Exception as e:
            logger.debug(f"Failed to extract MODIFIES relationships: {e}")

    async def _extract_authorship_fast(
        self,
        code_entities: list[tuple[str, str]],  # (entity_id, file_path)
        persons: dict[str, Person],
        stats: dict,
    ) -> None:
        """Extract AUTHORED relationships using git log (fast).

        Uses git log --name-only instead of git blame per file.
        Maps most recent commit author to each file.

        Args:
            code_entities: List of (entity_id, file_path) tuples
            persons: Dictionary of person entities
            stats: Statistics dictionary to update
        """
        authored_count = 0

        # Build file -> entity mapping
        file_to_entities: dict[str, list[str]] = {}
        for entity_id, file_path in code_entities:
            if not file_path:
                continue
            # Normalize to relative path
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
            except ValueError:
                rel_path = Path(file_path)
            rel_str = str(rel_path)

            if rel_str not in file_to_entities:
                file_to_entities[rel_str] = []
            file_to_entities[rel_str].append(entity_id)

        # Get file -> author mapping from git log (single command, very fast)
        # Maps file path -> (person_id, timestamp, sha)
        file_author_map: dict[str, tuple[str, str, str]] = {}

        try:
            # Get commits with files in one command
            result = subprocess.run(
                ["git", "log", "--format=%H|%an|%ae|%aI", "--name-only", "-n", "500"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning("git log failed")
                return

            current_commit = None
            current_email = None
            current_time = None

            for line in result.stdout.split("\n"):
                if "|" in line and line.count("|") == 3:
                    # Commit line: sha|name|email|time
                    parts = line.split("|")
                    current_commit = parts[0]
                    _ = parts[1]  # author name (unused, we use email)
                    current_email = parts[2]
                    current_time = parts[3]
                elif line.strip() and current_email:
                    # File line
                    file_path = line.strip()
                    if file_path not in file_author_map:
                        # First (most recent) author for this file
                        email_hash = self._hash_email(current_email)
                        person_id = f"person:{email_hash}"
                        file_author_map[file_path] = (
                            person_id,
                            current_time,
                            current_commit,
                        )

            logger.info(f"Mapped {len(file_author_map)} files to authors from git log")

        except subprocess.TimeoutExpired:
            logger.warning("git log timed out")
            return
        except Exception as e:
            logger.warning(f"git log failed: {e}")
            return

        # Create AUTHORED relationships
        for file_path, entity_ids in file_to_entities.items():
            if file_path in file_author_map:
                person_id, timestamp, commit_sha = file_author_map[file_path]

                if person_id in persons:
                    for entity_id in entity_ids[:5]:  # Limit per file
                        try:
                            await self.kg.add_authored_relationship(
                                person_id, entity_id, timestamp, commit_sha, 0
                            )
                            authored_count += 1
                        except Exception:
                            pass

        stats["authored"] = authored_count
        logger.info(f"✓ Created {authored_count} AUTHORED relationships")

    async def _extract_authorship_from_blame_detailed(
        self,
        code_entities: list[tuple[str, str]],  # (entity_id, file_path)
        persons: dict[str, Person],
        stats: dict,
    ) -> None:
        """Extract AUTHORED relationships from git blame (detailed, per-line).

        This is the slower, more accurate version that uses git blame per file.
        Kept for reference - use _extract_authorship_fast for production.

        Args:
            code_entities: List of (entity_id, file_path) tuples
            persons: Dictionary of person entities
            stats: Statistics dictionary to update
        """
        authored_count = 0

        # Group entities by file
        entities_by_file: dict[str, list[str]] = {}
        for entity_id, file_path in code_entities:
            if file_path not in entities_by_file:
                entities_by_file[file_path] = []
            entities_by_file[file_path].append(entity_id)

        logger.info(
            f"Extracting authorship from {min(100, len(entities_by_file))} files..."
        )

        # Process each file (limit to avoid timeout)
        for file_path, entity_ids in list(entities_by_file.items())[:100]:
            try:
                result = subprocess.run(
                    ["git", "blame", "-e", "--line-porcelain", file_path],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    continue

                # Parse blame output for author info
                current_email = None
                current_time = None

                for line in result.stdout.split("\n"):
                    if line.startswith("author-mail "):
                        current_email = line[12:].strip("<>")
                    elif line.startswith("author-time "):
                        timestamp = int(line[12:])
                        from datetime import datetime

                        current_time = datetime.fromtimestamp(timestamp).isoformat()

                # Create AUTHORED for first entity in file (simplified)
                if current_email and entity_ids:
                    email_hash = self._hash_email(current_email)
                    person_id = f"person:{email_hash}"

                    if person_id in persons:
                        for entity_id in entity_ids[:5]:  # Limit entities per file
                            try:
                                await self.kg.add_authored_relationship(
                                    person_id, entity_id, current_time or "", "", 0
                                )
                                authored_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to add AUTHORED: {e}")

            except subprocess.TimeoutExpired:
                logger.debug(f"Git blame timed out for {file_path}")
            except Exception as e:
                logger.debug(f"Blame failed for {file_path}: {e}")

        stats["authored"] = authored_count
        logger.info(f"✓ Created {authored_count} AUTHORED relationships")

    async def build_from_database(
        self,
        database,
        show_progress: bool = True,
        limit: int | None = None,
        skip_documents: bool = False,
        incremental: bool = False,
    ) -> dict[str, int]:
        """Build graph from all chunks in database.

        Args:
            database: VectorDatabase instance
            show_progress: Whether to show progress bar
            limit: Optional limit on number of chunks to process (for testing)
            skip_documents: Skip expensive DOCUMENTS relationship extraction (default False)
            incremental: Only process new chunks not in metadata (default False)

        Returns:
            Statistics dictionary
        """
        import time

        start_time = time.time()
        # Get chunk count first for progress reporting
        total_chunks = database.get_chunk_count()
        logger.info(f"Database has {total_chunks} total chunks")

        # Handle incremental mode
        processed_chunk_ids = set()
        if incremental:
            processed_chunk_ids = self._get_processed_chunk_ids()
            logger.info(
                f"Incremental mode: {len(processed_chunk_ids)} chunks already processed"
            )

        # Load chunks with progress reporting
        if show_progress:
            # IMPORTANT: Disable auto-refresh to avoid thread safety issues with Kuzu
            with (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[cyan]Loading chunks from database...[/cyan]"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    console=console,
                    auto_refresh=False,  # CRITICAL: Disable background thread entirely
                ) as progress
            ):
                task = progress.add_task("loading", total=total_chunks)

                chunks = []
                for batch in database.iter_chunks_batched(batch_size=5000):
                    # Filter out already processed chunks in incremental mode
                    if incremental:
                        batch = [
                            c
                            for c in batch
                            if (c.chunk_id or c.id) not in processed_chunk_ids
                        ]

                    chunks.extend(batch)
                    progress.update(task, advance=len(batch))
                    progress.refresh()  # Manual refresh (no background thread)

                    # Apply limit if specified
                    if limit and len(chunks) >= limit:
                        chunks = chunks[:limit]
                        progress.update(task, completed=total_chunks)
                        progress.refresh()
                        break
        else:
            logger.info("Loading chunks from database...")
            chunks = []
            for batch in database.iter_chunks_batched(batch_size=5000):
                # Filter out already processed chunks in incremental mode
                if incremental:
                    batch = [
                        c
                        for c in batch
                        if (c.chunk_id or c.id) not in processed_chunk_ids
                    ]

                chunks.extend(batch)
                if limit and len(chunks) >= limit:
                    chunks = chunks[:limit]
                    break

        if incremental and len(chunks) == 0:
            logger.info("No new chunks to process in incremental mode")
            return {}

        logger.info(f"Loaded {len(chunks)} chunks for processing")

        # Build graph from chunks
        stats = await self.build_from_chunks(
            chunks, show_progress=show_progress, skip_documents=skip_documents
        )

        # Extract work entities from git (if available)
        logger.info("Extracting work entities from git...")
        persons = await self._extract_git_authors(stats)
        project = await self._extract_project_info(stats)

        # Extract programming languages and frameworks
        logger.info("Extracting programming languages and frameworks...")
        languages = await self._extract_languages_and_frameworks(project, stats)

        # Create WRITTEN_IN relationships for all code entities
        if languages:
            logger.info("Creating WRITTEN_IN relationships...")
            try:
                result = self.kg.conn.execute(
                    "MATCH (e:CodeEntity) RETURN e.id, e.file_path"
                )
                written_in_count = 0
                while result.has_next():
                    row = result.get_next()
                    entity_id, file_path = row[0], row[1]
                    if file_path:
                        lang = self._detect_language_from_extension(file_path)
                        if lang:
                            lang_id = f"lang:{lang}"
                            if lang_id in languages:
                                await self.kg.add_written_in_relationship(
                                    entity_id, lang_id
                                )
                                written_in_count += 1

                stats["written_in"] = written_in_count
                logger.info(f"✓ Created {written_in_count} WRITTEN_IN relationships")
            except Exception as e:
                logger.debug(f"Failed to create WRITTEN_IN relationships: {e}")

        # Extract git history (repository, branches, commits)
        logger.info("Extracting git history (repository, branches, commits)...")
        repository, branches, commits = await self._extract_git_history(stats)

        # Extract MODIFIES relationships (commits -> code entities)
        if commits:
            logger.info("Extracting MODIFIES relationships...")
            await self._extract_modifies_relationships(commits, stats)

        # Extract authorship relationships (simplified - first author per file)
        if persons:
            entity_files = []
            try:
                result = self.kg.conn.execute(
                    "MATCH (e:CodeEntity) RETURN e.id, e.file_path"
                )
                while result.has_next():
                    row = result.get_next()
                    if row[1]:  # Has file path
                        entity_files.append((row[0], row[1]))
            except Exception as e:
                logger.debug(f"Failed to fetch entity file paths: {e}")

            if entity_files:
                await self._extract_authorship_fast(entity_files, persons, stats)

            # Add PART_OF relationships for all code entities (batch)
            if project:
                logger.info("Creating PART_OF relationships...")
                try:
                    result = self.kg.conn.execute("MATCH (e:CodeEntity) RETURN e.id")
                    entity_ids = []
                    while result.has_next():
                        entity_ids.append(result.get_next()[0])

                    # Use batch insert for PART_OF
                    part_of_count = await self.kg.add_part_of_batch(
                        entity_ids, project.id
                    )
                    stats["part_of"] = part_of_count
                    logger.info(f"✓ Created {part_of_count} PART_OF relationships")
                except Exception as e:
                    logger.debug(f"Failed to create PART_OF relationships: {e}")

        # Save metadata after successful build
        build_duration = time.time() - start_time

        # Collect all processed chunk IDs (existing + new)
        all_chunk_ids = processed_chunk_ids if incremental else set()
        for chunk in chunks:
            all_chunk_ids.add(chunk.chunk_id or chunk.id)

        # Calculate total entities and relationships
        total_entities = (
            stats.get("entities", 0)
            + stats.get("doc_sections", 0)
            + stats.get("tags", 0)
        )
        total_relationships = sum(
            stats.get(key, 0)
            for key in [
                "calls",
                "imports",
                "inherits",
                "contains",
                "references",
                "documents",
                "follows",
                "has_tag",
                "demonstrates",
                "links_to",
                "authored",
                "modified",
                "part_of",
            ]
        )

        self._save_metadata(
            source_chunk_count=len(all_chunk_ids),
            source_chunk_ids=all_chunk_ids,
            entities_created=total_entities,
            relationships_created=total_relationships,
            build_duration_seconds=build_duration,
        )

        logger.info(
            f"Build completed in {build_duration:.1f}s "
            f"({len(all_chunk_ids)} total chunks tracked)"
        )

        return stats
