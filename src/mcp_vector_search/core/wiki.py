"""Wiki generation for codebase ontology and concept extraction.

This module generates a hierarchical wiki/ontology of concepts found in a codebase,
organized by semantic similarity using LLM-based categorization.

Features:
- Automatic concept extraction from code chunks (functions, classes, etc.)
- LLM-powered semantic grouping into top-level categories
- Caching with configurable TTL
- JSON ontology export for programmatic access
- HTML wiki generation with embedded CSS
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .chunks_backend import ChunksBackend
from .database import VectorDatabase
from .llm_client import LLMClient


@dataclass
class WikiConcept:
    """A single concept extracted from the codebase.

    Concepts are hierarchical (parents/children) and reference actual code chunks.
    """

    id: str  # Hash of name + parent
    name: str  # Display name (e.g., "Authentication", "database_connection")
    description: str  # LLM-generated description
    parent_id: str | None  # Parent concept ID
    children: list[str] = field(default_factory=list)  # Child concept IDs
    related_chunks: list[str] = field(default_factory=list)  # Associated chunk IDs
    frequency: int = 0  # Chunk reference count
    depth: int = 0  # Tree depth (0 = root)


@dataclass
class WikiOntology:
    """Complete ontology of codebase concepts."""

    version: str = "1.0"
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    ttl_hours: int = 24
    project_root: str = ""
    total_concepts: int = 0
    total_chunks: int = 0
    concepts: dict[str, WikiConcept] = field(default_factory=dict)
    root_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert ontology to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "ttl_hours": self.ttl_hours,
            "project_root": self.project_root,
            "total_concepts": self.total_concepts,
            "total_chunks": self.total_chunks,
            "concepts": {
                cid: asdict(concept) for cid, concept in self.concepts.items()
            },
            "root_categories": self.root_categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WikiOntology":
        """Reconstruct ontology from dictionary."""
        concepts = {
            cid: WikiConcept(**concept_data)
            for cid, concept_data in data.get("concepts", {}).items()
        }
        return cls(
            version=data.get("version", "1.0"),
            generated_at=data.get("generated_at", ""),
            ttl_hours=data.get("ttl_hours", 24),
            project_root=data.get("project_root", ""),
            total_concepts=data.get("total_concepts", 0),
            total_chunks=data.get("total_chunks", 0),
            concepts=concepts,
            root_categories=data.get("root_categories", []),
        )


class WikiCache:
    """File-based cache for wiki ontology with TTL."""

    def __init__(self, cache_path: Path, ttl_hours: int = 24):
        """Initialize wiki cache.

        Args:
            cache_path: Path to cache file (.mcp-vector-search/wiki_cache.json)
            ttl_hours: Time-to-live in hours (default: 24)
        """
        self.cache_path = cache_path
        self.ttl_hours = ttl_hours

    def get(self) -> WikiOntology | None:
        """Get cached ontology if valid.

        Returns:
            Cached ontology or None if expired/missing
        """
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path) as f:
                data = json.load(f)

            # Check TTL
            generated_at = datetime.fromisoformat(data["generated_at"])
            age_hours = (datetime.now(UTC) - generated_at).total_seconds() / 3600

            if age_hours > self.ttl_hours:
                logger.info(
                    f"Wiki cache expired ({age_hours:.1f}h > {self.ttl_hours}h)"
                )
                return None

            logger.info(f"Loaded wiki from cache (age: {age_hours:.1f}h)")
            return WikiOntology.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to load wiki cache: {e}")
            return None

    def set(self, ontology: WikiOntology) -> None:
        """Save ontology to cache.

        Args:
            ontology: Ontology to cache
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(ontology.to_dict(), f, indent=2)
            logger.info(f"Saved wiki cache to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save wiki cache: {e}")

    def invalidate(self) -> None:
        """Delete cache file."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info("Invalidated wiki cache")


class WikiGenerator:
    """Generate codebase wiki/ontology from indexed chunks.

    Two-phase approach:
    1. Extract concepts from chunks (no LLM)
    2. LLM semantic grouping into categories (with descriptions)
    """

    def __init__(
        self,
        project_root: Path,
        chunks_backend: ChunksBackend | None = None,
        llm_client: LLMClient | None = None,
        vector_database: VectorDatabase | None = None,
    ):
        """Initialize wiki generator.

        Args:
            project_root: Project root directory
            chunks_backend: Chunks backend for accessing indexed code (deprecated)
            llm_client: LLM client for semantic grouping (optional)
            vector_database: Main vector database for accessing indexed chunks
        """
        self.project_root = project_root
        self.chunks_backend = chunks_backend
        self.llm_client = llm_client
        self.vector_database = vector_database
        self.cache = WikiCache(project_root / ".mcp-vector-search" / "wiki_cache.json")

    async def generate(self, force: bool = False, use_llm: bool = True) -> WikiOntology:
        """Generate wiki ontology from indexed chunks.

        Args:
            force: Force regeneration, ignoring cache
            use_llm: Use LLM for semantic grouping (requires LLM client)

        Returns:
            Complete wiki ontology

        Raises:
            ValueError: If use_llm=True but no LLM client provided
        """
        # Check cache first
        if not force:
            cached = self.cache.get()
            if cached:
                return cached

        logger.info("Generating wiki ontology...")
        start_time = time.time()

        # Phase 1: Extract concepts from chunks
        concepts, chunk_count = await self._extract_concepts()
        logger.info(
            f"Extracted {len(concepts)} concepts from {chunk_count} chunks "
            f"in {time.time() - start_time:.1f}s"
        )

        # Phase 2: LLM semantic grouping (optional)
        if use_llm:
            if not self.llm_client:
                raise ValueError("LLM client required for semantic grouping")

            ontology = await self._llm_semantic_grouping(concepts, chunk_count)
        else:
            # Create simple ontology without LLM
            ontology = self._create_flat_ontology(concepts, chunk_count)

        # Cache result
        ontology.project_root = str(self.project_root)
        self.cache.set(ontology)

        elapsed = time.time() - start_time
        logger.info(
            f"Generated wiki with {ontology.total_concepts} concepts in {elapsed:.1f}s"
        )

        return ontology

    async def _extract_concepts(self) -> tuple[dict[str, WikiConcept], int]:
        """Extract concepts from chunks (Phase 1 - no LLM).

        Returns:
            Tuple of (concept_map, total_chunk_count)
        """
        concept_map: dict[str, WikiConcept] = {}
        chunk_count = 0

        # Iterate through chunks in batches
        offset = 0
        batch_size = 1000

        while True:
            # Get pending or complete chunks (all indexed chunks)
            chunks = await self._get_all_chunks(batch_size, offset)

            if not chunks:
                break

            for chunk in chunks:
                chunk_count += 1

                # Extract concepts from chunk metadata
                concepts_from_chunk = self._extract_concepts_from_chunk(chunk)

                for concept_name in concepts_from_chunk:
                    # Generate concept ID
                    concept_id = self._generate_concept_id(concept_name)

                    if concept_id not in concept_map:
                        concept_map[concept_id] = WikiConcept(
                            id=concept_id,
                            name=concept_name,
                            description="",  # LLM will generate
                            parent_id=None,
                            children=[],
                            related_chunks=[],
                            frequency=0,
                            depth=0,
                        )

                    # Track chunk reference
                    concept = concept_map[concept_id]
                    if chunk["chunk_id"] not in concept.related_chunks:
                        concept.related_chunks.append(chunk["chunk_id"])
                        concept.frequency += 1

            offset += batch_size

            if len(chunks) < batch_size:
                break

        # Filter low-frequency concepts (noise reduction)
        filtered_concepts = {
            cid: concept
            for cid, concept in concept_map.items()
            if concept.frequency >= 2  # Require at least 2 references
        }

        logger.debug(
            f"Filtered {len(concept_map)} concepts to {len(filtered_concepts)} "
            f"(frequency >= 2)"
        )

        return filtered_concepts, chunk_count

    async def _get_all_chunks(self, batch_size: int, offset: int) -> list[dict]:
        """Get all indexed chunks (complete or pending).

        Args:
            batch_size: Batch size for iteration
            offset: Offset for pagination

        Returns:
            List of chunk dictionaries
        """
        # Try vector database first (main index)
        if self.vector_database is not None and offset == 0:
            try:
                # Get all chunks from vector database
                chunks = await self.vector_database.get_all_chunks()
                if chunks:
                    logger.debug(f"Retrieved {len(chunks)} chunks from vector database")
                    # Convert CodeChunk objects to dictionaries
                    chunk_dicts = []
                    for chunk in chunks:
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id,
                            "file_path": str(chunk.file_path),
                            "content": chunk.content,
                            "function_name": chunk.function_name,
                            "class_name": chunk.class_name,
                            "language": chunk.language,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                        }
                        chunk_dicts.append(chunk_dict)
                    return chunk_dicts
            except Exception as e:
                logger.warning(f"Failed to get chunks from vector database: {e}")

        # Fallback to chunks_backend (two-phase indexing)
        if self.chunks_backend is not None:
            pending = await self.chunks_backend.get_pending_chunks(batch_size, offset)

            # If no pending chunks, try to get all chunks via table query
            if not pending and offset == 0:
                try:
                    if self.chunks_backend._table is not None:
                        df = self.chunks_backend._table.to_pandas()
                        return df.to_dict("records")
                except Exception as e:
                    logger.warning(f"Failed to get chunks from chunks_backend: {e}")

            return pending

        return []

    def _extract_concepts_from_chunk(self, chunk: dict[str, Any]) -> set[str]:
        """Extract concept names from a single chunk.

        Concepts extracted:
        - Function name
        - Class name
        - Docstring keywords (first sentence)
        - File path components

        Args:
            chunk: Chunk dictionary

        Returns:
            Set of concept names
        """
        concepts = set()

        # Function/class names
        name = chunk.get("name", "")
        if name and name != "":
            concepts.add(name)

        # Parent class/function
        parent_name = chunk.get("parent_name", "")
        if parent_name and parent_name != "":
            concepts.add(parent_name)

        # File path components (directories as concepts)
        file_path = chunk.get("file_path", "")
        if file_path:
            path = Path(file_path)
            # Add directory names as concepts
            for part in path.parts:
                if part and part not in {".", "..", "src", "lib", "tests", "test"}:
                    concepts.add(part)

        # Docstring keywords (first sentence only, avoid noise)
        docstring = chunk.get("docstring", "")
        if docstring:
            # Extract first sentence
            first_sentence = docstring.split(".")[0].strip()
            # Extract significant words (>3 chars, alphanumeric)
            words = [
                word.strip().lower()
                for word in first_sentence.split()
                if len(word) > 3 and word.isalnum()
            ]
            concepts.update(words[:3])  # Limit to first 3 significant words

        return concepts

    def _generate_concept_id(self, name: str, parent_id: str | None = None) -> str:
        """Generate unique concept ID.

        Args:
            name: Concept name
            parent_id: Parent concept ID (optional)

        Returns:
            SHA-256 hash of name + parent_id
        """
        id_source = f"{name}:{parent_id or ''}"
        return hashlib.sha256(id_source.encode()).hexdigest()[:12]

    async def _llm_semantic_grouping(
        self, concepts: dict[str, WikiConcept], chunk_count: int
    ) -> WikiOntology:
        """Use LLM to organize concepts into semantic categories (Phase 2).

        Args:
            concepts: Flat concept map
            chunk_count: Total chunk count

        Returns:
            Hierarchical ontology with LLM-generated categories
        """
        # Sort concepts by frequency (most common first)
        sorted_concepts = sorted(
            concepts.values(), key=lambda c: c.frequency, reverse=True
        )

        # Take top 500 concepts (avoid token limits)
        top_concepts = sorted_concepts[:500]

        # Format for LLM
        concept_list = "\n".join(
            [f"- {c.name} ({c.frequency} references)" for c in top_concepts]
        )

        system_prompt = """You are a software architecture expert organizing codebase concepts into a hierarchical ontology.

Your task: Organize the given concepts into 5-10 top-level categories with brief descriptions.

Rules:
1. Create meaningful semantic categories (e.g., "Authentication", "Database", "API")
2. Each category should have 3-7 word description
3. Assign concepts to categories (one concept can appear in multiple categories)
4. Return JSON format only, no explanations

JSON format:
{
  "categories": [
    {
      "name": "Authentication",
      "description": "User authentication and authorization logic",
      "concepts": ["login", "auth_middleware", "jwt_token"]
    },
    ...
  ]
}
"""

        user_prompt = f"""Organize these code concepts into 5-10 semantic categories:

{concept_list}

Return JSON with categories, descriptions, and concept assignments."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.llm_client._chat_completion(messages)
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # Parse JSON from response
            parsed = self._parse_llm_json_response(content)

            # Build ontology from LLM response
            return self._build_ontology_from_llm_response(parsed, concepts, chunk_count)

        except Exception as e:
            logger.error(f"LLM semantic grouping failed: {e}")
            # Fallback to flat ontology
            return self._create_flat_ontology(concepts, chunk_count)

    def _parse_llm_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            content: LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        # Remove markdown code blocks if present
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        return json.loads(content)

    def _build_ontology_from_llm_response(
        self,
        llm_response: dict[str, Any],
        concepts: dict[str, WikiConcept],
        chunk_count: int,
    ) -> WikiOntology:
        """Build hierarchical ontology from LLM response.

        Args:
            llm_response: Parsed LLM JSON response
            concepts: Original concept map
            chunk_count: Total chunk count

        Returns:
            Complete ontology with hierarchy
        """
        ontology = WikiOntology(
            total_concepts=0,
            total_chunks=chunk_count,
            concepts={},
            root_categories=[],
        )

        categories = llm_response.get("categories", [])

        for category_data in categories:
            category_name = category_data.get("name", "Unnamed")
            category_desc = category_data.get("description", "")
            category_concepts = category_data.get("concepts", [])

            # Create category concept
            category_id = self._generate_concept_id(category_name)
            category_concept = WikiConcept(
                id=category_id,
                name=category_name,
                description=category_desc,
                parent_id=None,
                children=[],
                related_chunks=[],
                frequency=0,
                depth=0,
            )

            ontology.concepts[category_id] = category_concept
            ontology.root_categories.append(category_id)

            # Assign concepts to category
            for concept_name in category_concepts:
                # Find matching concept (case-insensitive)
                matching_concepts = [
                    c
                    for c in concepts.values()
                    if c.name.lower() == concept_name.lower()
                ]

                if matching_concepts:
                    child_concept = matching_concepts[0]
                    child_id = child_concept.id

                    # Update parent/child relationships
                    child_concept.parent_id = category_id
                    child_concept.depth = 1
                    category_concept.children.append(child_id)
                    category_concept.frequency += child_concept.frequency

                    # Add child to ontology
                    ontology.concepts[child_id] = child_concept

        # Add orphan concepts to "Other" category
        assigned_ids = set(ontology.concepts.keys())
        orphan_concepts = [c for c in concepts.values() if c.id not in assigned_ids]

        if orphan_concepts:
            other_id = self._generate_concept_id("Other")
            other_concept = WikiConcept(
                id=other_id,
                name="Other",
                description="Uncategorized concepts",
                parent_id=None,
                children=[],
                related_chunks=[],
                frequency=0,
                depth=0,
            )

            for orphan in orphan_concepts[:50]:  # Limit orphans to avoid noise
                orphan.parent_id = other_id
                orphan.depth = 1
                other_concept.children.append(orphan.id)
                other_concept.frequency += orphan.frequency
                ontology.concepts[orphan.id] = orphan

            ontology.concepts[other_id] = other_concept
            ontology.root_categories.append(other_id)

        ontology.total_concepts = len(ontology.concepts)
        return ontology

    def _create_flat_ontology(
        self, concepts: dict[str, WikiConcept], chunk_count: int
    ) -> WikiOntology:
        """Create simple flat ontology without LLM grouping.

        Args:
            concepts: Concept map
            chunk_count: Total chunk count

        Returns:
            Flat ontology with all concepts at root level
        """
        ontology = WikiOntology(
            total_concepts=len(concepts),
            total_chunks=chunk_count,
            concepts=concepts,
            root_categories=list(concepts.keys()),
        )

        return ontology
