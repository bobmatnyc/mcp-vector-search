"""Build knowledge graph from indexed code chunks.

This module extracts entities and relationships from code chunks
and populates the Kuzu knowledge graph.
"""

import hashlib
import re
import subprocess
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
    CodeEntity,
    CodeRelationship,
    DocSection,
    KnowledgeGraph,
    Person,
    Project,
)
from .models import CodeChunk

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

                # Extract DOCUMENTS relationships after all entities are processed
                if text_chunks and code_chunks:
                    progress.update(
                        task,
                        description="[cyan]Extracting DOCUMENTS relationships...[/cyan]",
                    )
                    await self._extract_documents_relationships(text_chunks, stats)
                    progress.update(
                        task,
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

            # Extract DOCUMENTS relationships after all entities are processed
            if text_chunks and code_chunks:
                logger.info("Extracting DOCUMENTS relationships...")
                await self._extract_documents_relationships(text_chunks, stats)

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

    async def _process_text_chunk(self, chunk: CodeChunk, stats: dict[str, int]):
        """Extract documentation sections and references from text chunk.

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
    ) -> None:
        """Create DOCUMENTS edges between doc sections and code entities.

        Args:
            doc_chunks: Text/documentation chunks
            stats: Statistics dict to update
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

        # Match doc sections against code entities
        for doc_section in doc_sections:
            doc_id = doc_section["id"]
            doc_name = doc_section["name"]
            doc_content = doc_section["content"]
            doc_file = doc_section["file_path"]

            for entity_id, entity_name, entity_type, entity_file in entity_info:
                # Skip generic entity names using centralized filter
                if self._is_generic_entity(entity_name):
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

        stats["documents"] = documents_count
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

    async def _extract_authorship_from_blame(
        self,
        code_entities: list[tuple[str, str]],  # (entity_id, file_path)
        persons: dict[str, Person],
        stats: dict,
    ) -> None:
        """Extract AUTHORED relationships from git blame.

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

        # Build graph from chunks
        stats = await self.build_from_chunks(chunks, show_progress=show_progress)

        # Extract work entities from git (if available)
        logger.info("Extracting work entities from git...")
        persons = await self._extract_git_authors(stats)
        project = await self._extract_project_info(stats)

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
                await self._extract_authorship_from_blame(entity_files, persons, stats)

            # Add PART_OF relationships for all code entities
            if project:
                logger.info("Creating PART_OF relationships...")
                part_of_count = 0
                try:
                    result = self.kg.conn.execute("MATCH (e:CodeEntity) RETURN e.id")
                    entity_ids = []
                    while result.has_next():
                        entity_ids.append(result.get_next()[0])

                    for entity_id in entity_ids:
                        try:
                            await self.kg.add_part_of_relationship(
                                entity_id, project.id
                            )
                            part_of_count += 1
                        except Exception:
                            pass

                    stats["part_of"] = part_of_count
                    logger.info(f"✓ Created {part_of_count} PART_OF relationships")
                except Exception as e:
                    logger.debug(f"Failed to create PART_OF relationships: {e}")

        return stats
