"""Review orchestration engine for AI-powered code review system."""

import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import SearchError
from ...core.knowledge_graph import KnowledgeGraph
from ...core.llm_client import LLMClient
from ...core.search import SemanticSearchEngine
from .models import ReviewFinding, ReviewResult, ReviewType, Severity
from .prompts import get_review_prompt

# Targeted search queries per review type
REVIEW_SEARCH_QUERIES = {
    ReviewType.SECURITY: [
        "authentication login password credential",
        "sql query database execute",
        "user input validation sanitize",
        "file upload path traversal",
        "encryption hash token secret key",
        "permission authorization access control",
    ],
    ReviewType.ARCHITECTURE: [
        "class definition inheritance",
        "import module dependency",
        "configuration settings",
        "error handling exception",
        "interface abstract base class",
    ],
    ReviewType.PERFORMANCE: [
        "database query loop iteration",
        "cache memory buffer pool",
        "async await concurrent parallel",
        "file read write io operation",
        "sort search algorithm complexity",
    ],
}


class ReviewEngine:
    """Orchestrates AI-powered code review using vector search, KG, and LLM.

    Pipeline:
    1. Vector search for relevant code chunks (targeted queries per review type)
    2. Knowledge graph queries for entity relationships
    3. Context formatting (code + relationships)
    4. LLM analysis with specialized review prompts
    5. Structured finding extraction and validation
    """

    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        knowledge_graph: KnowledgeGraph | None,
        llm_client: LLMClient,
        project_root: Path,
    ) -> None:
        """Initialize review engine.

        Args:
            search_engine: Semantic search engine for finding relevant code
            knowledge_graph: Optional knowledge graph for relationships (graceful degradation)
            llm_client: LLM client for code analysis
            project_root: Project root directory for path resolution
        """
        self.search_engine = search_engine
        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client
        self.project_root = project_root

    async def run_review(
        self,
        review_type: ReviewType,
        scope: str | None = None,
        max_chunks: int = 30,
        file_filter: list[str] | None = None,
    ) -> ReviewResult:
        """Run a code review using vector search, KG, and LLM analysis.

        Args:
            review_type: Type of review (security, architecture, performance)
            scope: Optional path filter (e.g., "src/auth" to review only auth module)
            max_chunks: Maximum code chunks to analyze (default: 30)
            file_filter: Optional list of file paths to restrict review to (for --changed-only)

        Returns:
            ReviewResult with findings and metadata
        """
        start_time = time.time()
        logger.info(
            f"Starting {review_type.value} review (scope: {scope or 'entire project'})"
        )

        # Step 1: Vector search for relevant code chunks
        search_results = await self._gather_code_chunks(
            review_type, scope, max_chunks, file_filter
        )

        if not search_results:
            logger.warning(f"No code chunks found for {review_type.value} review")
            return ReviewResult(
                review_type=review_type,
                findings=[],
                summary=f"No code chunks found matching {review_type.value} review queries",
                scope=scope or "entire project",
                context_chunks_used=0,
                kg_relationships_used=0,
                model_used=self.llm_client.model,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(search_results)} relevant code chunks")

        # Step 2: Gather knowledge graph relationships (graceful degradation)
        kg_relationships = await self._gather_kg_relationships(search_results)
        logger.info(f"Gathered {len(kg_relationships)} knowledge graph relationships")

        # Step 3: Format context for LLM
        code_context = self._format_code_context(search_results)
        kg_context = self._format_kg_context(kg_relationships)

        # Step 4: Call LLM with specialized review prompt
        findings = await self._analyze_with_llm(review_type, code_context, kg_context)

        logger.info(f"Review completed with {len(findings)} findings")

        duration = time.time() - start_time

        # Generate summary
        summary = self._generate_summary(findings, review_type)

        return ReviewResult(
            review_type=review_type,
            findings=findings,
            summary=summary,
            scope=scope or "entire project",
            context_chunks_used=len(search_results),
            kg_relationships_used=len(kg_relationships),
            model_used=self.llm_client.model,
            duration_seconds=duration,
        )

    async def _gather_code_chunks(
        self,
        review_type: ReviewType,
        scope: str | None,
        max_chunks: int,
        file_filter: list[str] | None = None,
    ) -> list[Any]:
        """Gather relevant code chunks using targeted search queries.

        Args:
            review_type: Type of review to perform
            scope: Optional path filter
            max_chunks: Maximum chunks to return
            file_filter: Optional list of file paths to filter results

        Returns:
            List of SearchResult objects
        """
        search_queries = REVIEW_SEARCH_QUERIES[review_type]
        all_results = {}  # chunk_id -> SearchResult (deduplicate)

        # Build filters if scope specified
        filters = {}
        if scope:
            # Support path prefix filtering
            filters["file_path"] = scope

        # Execute all search queries and merge results
        for query in search_queries:
            try:
                results = await self.search_engine.search(
                    query=query,
                    limit=max_chunks // len(search_queries)
                    + 5,  # Over-retrieve per query
                    filters=filters if filters else None,
                    similarity_threshold=0.5,  # Moderate threshold for review
                    include_context=True,
                )

                # Merge results (keep highest score per chunk)
                for result in results:
                    chunk_id = self._get_chunk_id(result)
                    if (
                        chunk_id not in all_results
                        or result.similarity_score
                        > all_results[chunk_id].similarity_score
                    ):
                        all_results[chunk_id] = result

            except SearchError as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Apply file filter if provided (for --changed-only)
        if file_filter:
            logger.info(f"Filtering to {len(file_filter)} changed files")
            filtered_results = {}
            for chunk_id, result in all_results.items():
                # Normalize paths for comparison
                result_path = str(Path(result.file_path).resolve())
                for filter_path in file_filter:
                    filter_path_resolved = str(Path(filter_path).resolve())
                    if result_path == filter_path_resolved:
                        filtered_results[chunk_id] = result
                        break
            all_results = filtered_results
            logger.info(f"After filtering: {len(all_results)} chunks remain")

        # Sort by score and limit to max_chunks
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.similarity_score,
            reverse=True,
        )

        return sorted_results[:max_chunks]

    async def _gather_kg_relationships(
        self, search_results: list[Any]
    ) -> list[dict[str, Any]]:
        """Gather knowledge graph relationships for search results.

        Args:
            search_results: List of SearchResult objects

        Returns:
            List of relationship dictionaries (gracefully handles missing KG)
        """
        if not self.knowledge_graph:
            logger.debug(
                "Knowledge graph not available, skipping relationship gathering"
            )
            return []

        relationships = []

        for result in search_results:
            # Extract entity names from result
            entity_names = []
            if result.function_name:
                entity_names.append(result.function_name)
            if hasattr(result, "class_name") and result.class_name:
                entity_names.append(result.class_name)

            # Query KG for each entity
            for entity_name in entity_names:
                try:
                    # Find entity
                    entity = await self.knowledge_graph.find_entity_by_name(entity_name)
                    if not entity:
                        continue

                    # Get related entities (1-2 hops for context)
                    related = await self.knowledge_graph.find_related(
                        entity_name, max_hops=2
                    )

                    # Add to relationships
                    relationships.append(
                        {
                            "entity": entity_name,
                            "type": entity.get("type"),
                            "file": str(result.file_path),
                            "related": related,
                        }
                    )

                except Exception as e:
                    logger.debug(f"Failed to get KG data for {entity_name}: {e}")
                    continue

        return relationships

    def _format_code_context(self, search_results: list[Any]) -> str:
        """Format code chunks for LLM context.

        Args:
            search_results: List of SearchResult objects

        Returns:
            Formatted code context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            context_parts.append(f"### Code Chunk {i}")
            context_parts.append(f"**File**: `{result.file_path}`")
            context_parts.append(f"**Lines**: {result.start_line}-{result.end_line}")

            if result.function_name:
                context_parts.append(f"**Function**: `{result.function_name}`")
            if hasattr(result, "class_name") and result.class_name:
                context_parts.append(f"**Class**: `{result.class_name}`")

            context_parts.append(f"**Similarity**: {result.similarity_score:.3f}")
            context_parts.append("")
            context_parts.append("```" + (result.language or "python"))
            context_parts.append(result.content)
            context_parts.append("```")
            context_parts.append("")

        return "\n".join(context_parts)

    def _format_kg_context(self, relationships: list[dict[str, Any]]) -> str:
        """Format knowledge graph relationships for LLM context.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            Formatted KG context string
        """
        if not relationships:
            return "No knowledge graph relationships available."

        context_parts = []

        for rel in relationships:
            context_parts.append(f"**Entity**: `{rel['entity']}` ({rel['type']})")
            context_parts.append(f"**File**: `{rel['file']}`")

            if rel["related"]:
                context_parts.append("**Related Entities**:")
                for related in rel["related"][:5]:  # Limit to top 5
                    rel_type = related.get("relationship", "related_to")
                    rel_name = related.get("name", "unknown")
                    context_parts.append(f"  - {rel_type}: `{rel_name}`")

            context_parts.append("")

        return "\n".join(context_parts)

    async def _analyze_with_llm(
        self, review_type: ReviewType, code_context: str, kg_context: str
    ) -> list[ReviewFinding]:
        """Analyze code with LLM and extract structured findings.

        Args:
            review_type: Type of review
            code_context: Formatted code chunks
            kg_context: Formatted knowledge graph relationships

        Returns:
            List of ReviewFinding objects
        """
        # Get specialized prompt for review type
        prompt_template = get_review_prompt(review_type)

        # Fill in context
        prompt = prompt_template.format(
            code_context=code_context,
            kg_relationships=kg_context,
        )

        # Call LLM
        try:
            messages = [
                {"role": "system", "content": prompt},
            ]

            response = await self.llm_client._chat_completion(messages)

            # Extract content from response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # Parse JSON findings
            findings = self._parse_findings_json(content)

            logger.info(f"LLM returned {len(findings)} findings")

            return findings

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise SearchError(f"Code review analysis failed: {e}") from e

    def _parse_findings_json(self, llm_response: str) -> list[ReviewFinding]:
        """Parse JSON findings from LLM response.

        Handles cases where JSON is wrapped in markdown code blocks or has extra text.

        Args:
            llm_response: Raw LLM response text

        Returns:
            List of ReviewFinding objects
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```json\s*\n(.*?)\n```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try without markdown
            json_match = re.search(r"```\s*\n(.*?)\n```", llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume entire response is JSON
                json_str = llm_response.strip()

        try:
            findings_data = json.loads(json_str)

            if not isinstance(findings_data, list):
                logger.warning("LLM returned non-list JSON, wrapping in list")
                findings_data = [findings_data]

            findings = []
            for item in findings_data:
                try:
                    # Validate and create ReviewFinding
                    finding = ReviewFinding(
                        title=item["title"],
                        description=item["description"],
                        severity=Severity(item["severity"].lower()),
                        file_path=item["file_path"],
                        start_line=int(item["start_line"]),
                        end_line=int(item["end_line"]),
                        category=item["category"],
                        recommendation=item["recommendation"],
                        confidence=float(item["confidence"]),
                        cwe_id=item.get("cwe_id"),
                        code_snippet=item.get("code_snippet"),
                        related_files=item.get("related_files", []),
                    )
                    findings.append(finding)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse finding: {e}, item: {item}")
                    continue

            return findings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"LLM response: {llm_response[:500]}")
            return []

    def _generate_summary(
        self, findings: list[ReviewFinding], review_type: ReviewType
    ) -> str:
        """Generate review summary from findings.

        Args:
            findings: List of findings
            review_type: Type of review

        Returns:
            Summary string
        """
        if not findings:
            return f"No {review_type.value} issues found."

        # Count by severity
        severity_counts = {}
        for finding in findings:
            severity_counts[finding.severity] = (
                severity_counts.get(finding.severity, 0) + 1
            )

        # Build summary
        parts = [f"Found {len(findings)} {review_type.value} issue(s):"]

        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                parts.append(f"  - {severity.value.upper()}: {count}")

        return "\n".join(parts)

    @staticmethod
    def _get_chunk_id(result: Any) -> str:
        """Get chunk ID from SearchResult for deduplication.

        Args:
            result: SearchResult object

        Returns:
            Unique chunk identifier
        """
        if hasattr(result, "chunk_id") and result.chunk_id:
            return result.chunk_id

        # Fallback: construct from file path and line range
        return f"{result.file_path}:{result.start_line}-{result.end_line}"
