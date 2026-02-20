"""Semantic analyzer that enriches story data using vector search.

This module uses the vector database to:
- Identify code themes and semantic clusters
- Detect technologies and architectural patterns
- Analyze evolution phases based on code changes
- Extract language distribution from chunks
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.database import VectorDatabase
from ..core.search import SemanticSearchEngine
from .models import (
    ArchitecturalPattern,
    ConfidenceLevel,
    EvolutionPhase,
    SemanticCluster,
    StoryAnalysis,
    StoryExtraction,
    TechStackItem,
)

logger = logging.getLogger(__name__)


class StoryAnalyzer:
    """Semantic analyzer that enriches story data using vector search.

    Uses the existing vector database to identify code themes, patterns,
    and architectural decisions. Gracefully degrades when no index exists.
    """

    def __init__(self, project_root: Path) -> None:
        """Initialize analyzer with project root.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self._db: VectorDatabase | None = None
        self._search: SemanticSearchEngine | None = None
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        """Initialize vector DB connection. Returns False if no index exists.

        Returns:
            True if initialized successfully, False if no index exists
        """
        if self._initialized:
            return self._db is not None

        try:
            # Try to open the vector database
            # Standard path is .mcp-vector-search/lance
            index_path = self.project_root / ".mcp-vector-search" / "lance"

            if not index_path.exists():
                logger.warning(
                    f"Vector index not found at {index_path}. "
                    "Story analysis will be minimal without semantic search."
                )
                self._initialized = True
                return False

            # Import the LanceDB backend
            from ..core.lancedb_backend import LanceDBBackend

            # Initialize database
            self._db = LanceDBBackend(persist_directory=index_path)
            await self._db.initialize()

            # Initialize search engine
            self._search = SemanticSearchEngine(
                database=self._db,
                project_root=self.project_root,
                similarity_threshold=0.5,
                enable_kg=False,  # Disable KG for story analysis
            )

            logger.info("Vector database initialized for story analysis")
            self._initialized = True
            return True

        except Exception as e:
            logger.warning(
                f"Failed to initialize vector database: {e}. "
                "Continuing without semantic analysis."
            )
            self._initialized = True
            return False

    async def analyze(self, extraction: StoryExtraction) -> StoryAnalysis:
        """Run full semantic analysis. Works without vector index (returns minimal analysis).

        Args:
            extraction: Raw extracted data from git/GitHub

        Returns:
            StoryAnalysis with semantic insights (empty if no index)
        """
        try:
            # Try to initialize
            has_db = await self._ensure_initialized()

            if not has_db:
                # Return minimal analysis without vector search
                logger.info("Generating minimal analysis without vector search")
                return StoryAnalysis(
                    clusters=[],
                    tech_stack=await self._detect_tech_stack_minimal(),
                    architectural_patterns=[],
                    evolution_phases=await self._trace_evolution_minimal(extraction),
                    language_distribution={},
                )

            # Full analysis with vector search
            logger.info("Running full semantic analysis with vector search")

            clusters = await self.analyze_code_themes()
            tech_stack = await self.detect_tech_stack()
            patterns = await self.identify_architectural_patterns()
            phases = await self.trace_evolution(extraction)
            language_dist = await self.find_language_distribution()

            return StoryAnalysis(
                clusters=clusters,
                tech_stack=tech_stack,
                architectural_patterns=patterns,
                evolution_phases=phases,
                language_distribution=language_dist,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            # Return empty analysis on error
            return StoryAnalysis()

    async def analyze_code_themes(self) -> list[SemanticCluster]:
        """Search vectors for common code theme queries.

        Returns:
            List of semantic clusters representing code themes
        """
        if not self._search:
            return []

        # Theme queries covering common code patterns
        theme_queries = [
            "error handling and validation",
            "data models and schemas",
            "API endpoints and routing",
            "database operations and queries",
            "authentication and authorization",
            "testing and test utilities",
            "configuration and settings",
            "logging and monitoring",
            "file I/O and data processing",
            "CLI and user interface",
        ]

        clusters: list[SemanticCluster] = []

        try:
            for query in theme_queries:
                try:
                    # Search with lower threshold for broader coverage
                    results = await self._search.search(
                        query=query,
                        limit=10,
                        similarity_threshold=0.4,
                        include_context=False,
                    )

                    if not results:
                        continue

                    # Group results by file
                    files_seen = set()
                    chunk_ids = []
                    code_snippets = []

                    for result in results[:5]:  # Top 5 results
                        file_path = str(result.file_path)
                        if file_path not in files_seen:
                            files_seen.add(file_path)
                            # Add short snippet (first 100 chars)
                            snippet = result.content[:100].replace("\n", " ")
                            code_snippets.append(
                                f"{result.file_path.name}: {snippet}..."
                            )

                        # Store chunk IDs for reference
                        if hasattr(result, "chunk_id"):
                            chunk_ids.append(result.chunk_id)

                    # Determine confidence based on result quality
                    avg_score = sum(r.similarity_score for r in results) / len(results)
                    if avg_score > 0.7:
                        confidence = ConfidenceLevel.HIGH
                    elif avg_score > 0.5:
                        confidence = ConfidenceLevel.MEDIUM
                    else:
                        confidence = ConfidenceLevel.LOW

                    cluster = SemanticCluster(
                        name=query.title(),
                        description=f"Code related to {query}",
                        query=query,
                        files=sorted(files_seen),
                        chunk_ids=chunk_ids,
                        confidence=confidence,
                        code_snippets=code_snippets,
                    )
                    clusters.append(cluster)

                except Exception as e:
                    logger.debug(f"Failed to analyze theme '{query}': {e}")
                    continue

            logger.info(f"Identified {len(clusters)} code theme clusters")
            return clusters

        except Exception as e:
            logger.warning(f"Code theme analysis failed: {e}")
            return []

    async def detect_tech_stack(self) -> list[TechStackItem]:
        """Detect technologies from chunks table language distribution.

        Returns:
            List of detected technologies
        """
        if not self._db:
            return await self._detect_tech_stack_minimal()

        try:
            # Get language distribution from database stats
            stats = await self._db.get_stats()
            languages = stats.languages

            tech_stack: list[TechStackItem] = []

            # Map languages to tech stack items
            for language, count in languages.items():
                category = self._categorize_language(language)

                tech_stack.append(
                    TechStackItem(
                        name=language.title(),
                        category=category,
                        evidence=[f"{count} code chunks"],
                        version=None,
                    )
                )

            logger.info(f"Detected {len(tech_stack)} technologies")
            return tech_stack

        except Exception as e:
            logger.warning(f"Tech stack detection failed: {e}")
            return await self._detect_tech_stack_minimal()

    async def _detect_tech_stack_minimal(self) -> list[TechStackItem]:
        """Detect tech stack without vector database (from file extensions).

        Returns:
            List of basic tech stack items
        """
        try:
            # Scan project for common file extensions
            extensions_seen: dict[str, int] = defaultdict(int)

            for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".md"]:
                count = len(list(self.project_root.rglob(f"*{ext}")))
                if count > 0:
                    extensions_seen[ext] = count

            tech_stack: list[TechStackItem] = []
            ext_to_lang = {
                ".py": "Python",
                ".js": "JavaScript",
                ".ts": "TypeScript",
                ".java": "Java",
                ".go": "Go",
                ".rs": "Rust",
                ".md": "Markdown",
            }

            for ext, count in extensions_seen.items():
                if ext in ext_to_lang:
                    lang = ext_to_lang[ext]
                    category = self._categorize_language(lang.lower())

                    tech_stack.append(
                        TechStackItem(
                            name=lang,
                            category=category,
                            evidence=[f"{count} files"],
                            version=None,
                        )
                    )

            return tech_stack

        except Exception as e:
            logger.warning(f"Minimal tech stack detection failed: {e}")
            return []

    def _categorize_language(self, language: str) -> str:
        """Categorize language into tech category.

        Args:
            language: Language name

        Returns:
            Category string
        """
        language_lower = language.lower()

        if language_lower in [
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
        ]:
            return "language"
        elif language_lower in ["markdown", "html", "css"]:
            return "markup"
        elif language_lower in ["json", "yaml", "toml", "xml"]:
            return "config"
        elif language_lower in ["sql", "graphql"]:
            return "query"
        else:
            return "other"

    async def identify_architectural_patterns(self) -> list[ArchitecturalPattern]:
        """Search for MVC, repository, factory, etc. patterns.

        Returns:
            List of detected architectural patterns
        """
        if not self._search:
            return []

        # Pattern queries
        pattern_queries = [
            ("Repository Pattern", "repository pattern class CRUD operations"),
            ("MVC Architecture", "model view controller separation"),
            ("Factory Pattern", "factory pattern create instance"),
            ("Dependency Injection", "dependency injection container"),
            ("Event-Driven", "event listener subscriber publish"),
            ("Service Layer", "service layer business logic"),
        ]

        patterns: list[ArchitecturalPattern] = []

        try:
            for pattern_name, query in pattern_queries:
                try:
                    results = await self._search.search(
                        query=query,
                        limit=5,
                        similarity_threshold=0.5,
                        include_context=False,
                    )

                    if not results or len(results) < 2:
                        continue  # Need at least 2 results for confidence

                    files = [str(r.file_path) for r in results]
                    evidence = [
                        f"{r.file_path.name} (score: {r.similarity_score:.2f})"
                        for r in results[:3]
                    ]

                    # Determine confidence
                    avg_score = sum(r.similarity_score for r in results) / len(results)
                    if avg_score > 0.7 and len(results) >= 3:
                        confidence = ConfidenceLevel.HIGH
                    elif avg_score > 0.5:
                        confidence = ConfidenceLevel.MEDIUM
                    else:
                        confidence = ConfidenceLevel.LOW

                    pattern = ArchitecturalPattern(
                        name=pattern_name,
                        description=f"Evidence of {pattern_name} in codebase",
                        confidence=confidence,
                        evidence=evidence,
                        files=files[:5],
                    )
                    patterns.append(pattern)

                except Exception as e:
                    logger.debug(f"Failed to detect pattern '{pattern_name}': {e}")
                    continue

            logger.info(f"Identified {len(patterns)} architectural patterns")
            return patterns

        except Exception as e:
            logger.warning(f"Pattern identification failed: {e}")
            return []

    async def trace_evolution(
        self, extraction: StoryExtraction
    ) -> list[EvolutionPhase]:
        """Correlate commit dates with code area changes to identify evolution phases.

        Args:
            extraction: Raw extracted data with commits

        Returns:
            List of evolution phases
        """
        if not extraction.commits:
            return []

        try:
            # Sort commits by date
            sorted_commits = sorted(extraction.commits, key=lambda c: c.date)

            if not sorted_commits:
                return []

            # Group commits into phases (monthly or by significant gaps)
            phases: list[EvolutionPhase] = []

            # Simple strategy: split into 3-5 phases based on time distribution
            total_span = (sorted_commits[-1].date - sorted_commits[0].date).days

            if total_span < 30:
                # Single phase for projects < 1 month
                phase = self._create_phase(
                    "Initial Development",
                    sorted_commits,
                    sorted_commits[0].date,
                    sorted_commits[-1].date,
                )
                return [phase] if phase else []

            elif total_span < 180:
                # 2-3 phases for projects < 6 months
                num_phases = 2
            else:
                # 3-5 phases for longer projects
                num_phases = min(5, max(3, total_span // 90))

            # Divide commits into equal phases
            commits_per_phase = len(sorted_commits) // num_phases

            for i in range(num_phases):
                start_idx = i * commits_per_phase
                end_idx = (
                    (i + 1) * commits_per_phase
                    if i < num_phases - 1
                    else len(sorted_commits)
                )

                phase_commits = sorted_commits[start_idx:end_idx]
                if not phase_commits:
                    continue

                phase_name = self._get_phase_name(i, num_phases)

                phase = self._create_phase(
                    phase_name,
                    phase_commits,
                    phase_commits[0].date,
                    phase_commits[-1].date,
                )

                if phase:
                    phases.append(phase)

            logger.info(f"Traced {len(phases)} evolution phases")
            return phases

        except Exception as e:
            logger.warning(f"Evolution tracing failed: {e}")
            return await self._trace_evolution_minimal(extraction)

    async def _trace_evolution_minimal(
        self, extraction: StoryExtraction
    ) -> list[EvolutionPhase]:
        """Minimal evolution tracing without vector search.

        Args:
            extraction: Raw extracted data

        Returns:
            List of basic evolution phases
        """
        if not extraction.commits:
            return []

        try:
            sorted_commits = sorted(extraction.commits, key=lambda c: c.date)

            if not sorted_commits:
                return []

            # Single phase covering entire project
            phase = self._create_phase(
                "Development",
                sorted_commits,
                sorted_commits[0].date,
                sorted_commits[-1].date,
            )

            return [phase] if phase else []

        except Exception as e:
            logger.warning(f"Minimal evolution tracing failed: {e}")
            return []

    def _create_phase(
        self, name: str, commits: list[Any], start_date: datetime, end_date: datetime
    ) -> EvolutionPhase | None:
        """Create evolution phase from commits.

        Args:
            name: Phase name
            commits: Commits in this phase
            start_date: Phase start date
            end_date: Phase end date

        Returns:
            EvolutionPhase or None if invalid
        """
        if not commits:
            return None

        # Identify dominant areas (files most frequently changed)
        file_counts: dict[str, int] = defaultdict(int)

        for commit in commits:
            for file in commit.files:
                file_counts[file] += 1

        # Top 5 most active files/areas
        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        dominant_areas = [file for file, _ in top_files]

        # Key commits (those with most changes)
        key_commits = [
            c.short_hash
            for c in sorted(commits, key=lambda c: c.files_changed, reverse=True)[:5]
        ]

        description = f"Phase with {len(commits)} commits"
        if dominant_areas:
            description += f", focused on {dominant_areas[0]}"

        return EvolutionPhase(
            name=name,
            date_start=start_date,
            date_end=end_date,
            description=description,
            key_commits=key_commits,
            dominant_areas=dominant_areas,
            commit_count=len(commits),
        )

    def _get_phase_name(self, phase_num: int, total_phases: int) -> str:
        """Get descriptive name for phase.

        Args:
            phase_num: Zero-indexed phase number
            total_phases: Total number of phases

        Returns:
            Phase name
        """
        if total_phases == 1:
            return "Development"
        elif phase_num == 0:
            return "Initial Development"
        elif phase_num == total_phases - 1:
            return "Current State"
        else:
            return f"Phase {phase_num + 1}"

    async def find_language_distribution(self) -> dict[str, int]:
        """Get language distribution from chunks.

        Returns:
            Dictionary mapping language to line count
        """
        if not self._db:
            return {}

        try:
            stats = await self._db.get_stats()
            return dict(stats.languages)

        except Exception as e:
            logger.warning(f"Language distribution failed: {e}")
            return {}

    async def close(self) -> None:
        """Close database connections."""
        if self._db:
            try:
                await self._db.close()
            except Exception as e:
                logger.debug(f"Error closing database: {e}")

        self._db = None
        self._search = None
        self._initialized = False
