"""Search operations handler for ChromaDB vector database."""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .exceptions import SearchError
from .models import SearchResult


class SearchHandler:
    """Handles search operations and result processing for ChromaDB.

    Responsible for:
    - Executing vector similarity searches
    - Converting distances to similarity scores
    - Parsing metadata and quality metrics
    - Filtering results by similarity threshold
    """

    @staticmethod
    def execute_search(
        collection: Any,
        query: str,
        limit: int = 10,
        where_clause: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a search query on the collection.

        Args:
            collection: ChromaDB collection instance
            query: Search query text
            limit: Maximum number of results
            where_clause: Optional filter conditions

        Returns:
            Raw search results from ChromaDB

        Raises:
            SearchError: If search fails
        """
        try:
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}") from e

    @staticmethod
    def process_results(
        results: dict[str, Any],
        query: str,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Process raw search results into SearchResult objects.

        Args:
            results: Raw search results from ChromaDB
            query: Original query string (for logging)
            similarity_threshold: Minimum similarity score

        Returns:
            List of processed search results
        """
        search_results = []

        if not results.get("documents") or not results["documents"][0]:
            logger.debug(f"No results found for query: {query}")
            return search_results

        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=False,
            )
        ):
            # Convert distance to similarity
            similarity = SearchHandler._distance_to_similarity(distance)

            if similarity >= similarity_threshold:
                result = SearchHandler._create_search_result(
                    doc, metadata, similarity, i + 1
                )
                search_results.append(result)

        logger.debug(f"Found {len(search_results)} results for query: {query}")
        return search_results

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert ChromaDB cosine distance to similarity score.

        ChromaDB returns cosine distances. This converts them to a
        0-1 similarity score where lower distances = higher similarity.

        Args:
            distance: Cosine distance from ChromaDB

        Returns:
            Similarity score between 0 and 1
        """
        # For cosine distance, use permissive conversion that handles distances > 1.0
        # Convert to 0-1 similarity score where lower distances = higher similarity
        return max(0.0, 1.0 / (1.0 + distance))

    @staticmethod
    def _create_search_result(
        doc: str,
        metadata: dict[str, Any],
        similarity: float,
        rank: int,
    ) -> SearchResult:
        """Create a SearchResult object from raw data.

        Args:
            doc: Document content
            metadata: Document metadata
            similarity: Similarity score
            rank: Result rank (1-indexed)

        Returns:
            SearchResult object with parsed quality metrics
        """
        # Parse code smells from JSON if present
        code_smells = []
        if "code_smells" in metadata:
            try:
                code_smells = json.loads(metadata["code_smells"])
            except (json.JSONDecodeError, TypeError):
                code_smells = []

        # Calculate quality score from metrics (0-100 scale)
        quality_score = SearchHandler._calculate_quality_score(metadata)

        return SearchResult(
            content=doc,
            file_path=Path(metadata["file_path"]),
            start_line=metadata["start_line"],
            end_line=metadata["end_line"],
            language=metadata["language"],
            similarity_score=similarity,
            rank=rank,
            chunk_type=metadata.get("chunk_type", "code"),
            function_name=metadata.get("function_name") or None,
            class_name=metadata.get("class_name") or None,
            # Quality metrics from structural analysis
            cognitive_complexity=metadata.get("cognitive_complexity"),
            cyclomatic_complexity=metadata.get("cyclomatic_complexity"),
            max_nesting_depth=metadata.get("max_nesting_depth"),
            parameter_count=metadata.get("parameter_count"),
            lines_of_code=metadata.get("lines_of_code"),
            complexity_grade=metadata.get("complexity_grade"),
            code_smells=code_smells,
            smell_count=metadata.get("smell_count"),
            quality_score=quality_score,
        )

    @staticmethod
    def _calculate_quality_score(metadata: dict[str, Any]) -> int | None:
        """Calculate code quality score from metrics.

        Simple quality score: start with 100, penalize complexity and smells.

        Args:
            metadata: Metadata dictionary with quality metrics

        Returns:
            Quality score (0-100) or None if metrics not available
        """
        if "cognitive_complexity" not in metadata or "smell_count" not in metadata:
            return None

        complexity = metadata["cognitive_complexity"]
        smells = metadata["smell_count"]

        # Start with 100, penalize for complexity and smells
        score = 100
        # Complexity penalty: -2 points per complexity unit (max 50 points)
        score -= min(50, complexity * 2)
        # Smell penalty: -10 points per smell (max 30 points)
        score -= min(30, smells * 10)

        return max(0, score)
