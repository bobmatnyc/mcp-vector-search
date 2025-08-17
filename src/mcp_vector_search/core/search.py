"""Semantic search engine for MCP Vector Search."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .database import VectorDatabase
from .exceptions import SearchError
from .models import SearchResult


class SemanticSearchEngine:
    """Semantic search engine for code search."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.7,
    ) -> None:
        """Initialize semantic search engine.
        
        Args:
            database: Vector database instance
            project_root: Project root directory
            similarity_threshold: Default similarity threshold
        """
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        include_context: bool = True,
    ) -> List[SearchResult]:
        """Perform semantic search for code.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (language, file_path, etc.)
            similarity_threshold: Minimum similarity score
            include_context: Whether to include context lines
            
        Returns:
            List of search results
        """
        if not query.strip():
            return []

        threshold = similarity_threshold or self.similarity_threshold

        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Perform vector search
            results = await self.database.search(
                query=processed_query,
                limit=limit,
                filters=filters,
                similarity_threshold=threshold,
            )

            # Post-process results
            enhanced_results = []
            for result in results:
                enhanced_result = await self._enhance_result(result, include_context)
                enhanced_results.append(enhanced_result)

            # Apply additional ranking if needed
            ranked_results = self._rerank_results(enhanced_results, query)

            logger.debug(f"Search for '{query}' returned {len(ranked_results)} results")
            return ranked_results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise SearchError(f"Search failed: {e}") from e

    async def search_similar(
        self,
        file_path: Path,
        function_name: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Find code similar to a specific function or file.
        
        Args:
            file_path: Path to the reference file
            function_name: Specific function name (optional)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar code results
        """
        try:
            # Read the reference file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # If function name is specified, try to extract just that function
            if function_name:
                function_content = self._extract_function_content(content, function_name)
                if function_content:
                    content = function_content

            # Use the content as the search query
            return await self.search(
                query=content,
                limit=limit,
                similarity_threshold=similarity_threshold,
                include_context=True,
            )

        except Exception as e:
            logger.error(f"Similar search failed for {file_path}: {e}")
            raise SearchError(f"Similar search failed: {e}") from e

    async def search_by_context(
        self,
        context_description: str,
        focus_areas: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search for code based on contextual description.
        
        Args:
            context_description: Description of what you're looking for
            focus_areas: Areas to focus on (e.g., ["security", "authentication"])
            limit: Maximum number of results
            
        Returns:
            List of contextually relevant results
        """
        # Build enhanced query with focus areas
        query_parts = [context_description]
        
        if focus_areas:
            query_parts.extend(focus_areas)
        
        enhanced_query = " ".join(query_parts)
        
        return await self.search(
            query=enhanced_query,
            limit=limit,
            include_context=True,
        )

    def _preprocess_query(self, query: str) -> str:
        """Preprocess search query for better results.
        
        Args:
            query: Raw search query
            
        Returns:
            Processed query
        """
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query.strip())
        
        # Expand common abbreviations
        expansions = {
            "auth": "authentication",
            "db": "database",
            "api": "application programming interface",
            "ui": "user interface",
            "util": "utility",
            "config": "configuration",
        }
        
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in expansions:
                expanded_words.extend([word, expansions[word]])
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)

    async def _enhance_result(
        self, result: SearchResult, include_context: bool
    ) -> SearchResult:
        """Enhance search result with additional information.
        
        Args:
            result: Original search result
            include_context: Whether to include context lines
            
        Returns:
            Enhanced search result
        """
        if not include_context:
            return result

        try:
            # Read the source file to get context
            with open(result.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Get context lines before and after
            context_size = 3
            start_idx = max(0, result.start_line - 1 - context_size)
            end_idx = min(len(lines), result.end_line + context_size)

            context_before = [
                line.rstrip() for line in lines[start_idx : result.start_line - 1]
            ]
            context_after = [
                line.rstrip() for line in lines[result.end_line : end_idx]
            ]

            # Update result with context
            result.context_before = context_before
            result.context_after = context_after

        except Exception as e:
            logger.warning(f"Failed to get context for {result.file_path}: {e}")

        return result

    def _rerank_results(
        self, results: List[SearchResult], query: str
    ) -> List[SearchResult]:
        """Apply additional ranking to search results.
        
        Args:
            results: Original search results
            query: Original search query
            
        Returns:
            Reranked search results
        """
        # Simple reranking based on additional factors
        query_lower = query.lower()
        
        for result in results:
            # Boost score for exact matches in function/class names
            boost = 0.0
            
            if result.function_name and query_lower in result.function_name.lower():
                boost += 0.1
            
            if result.class_name and query_lower in result.class_name.lower():
                boost += 0.1
            
            # Boost score for matches in file name
            if query_lower in result.file_path.name.lower():
                boost += 0.05
            
            # Apply boost
            result.similarity_score = min(1.0, result.similarity_score + boost)
        
        # Re-sort by similarity score
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results

    def _extract_function_content(self, content: str, function_name: str) -> Optional[str]:
        """Extract content of a specific function from code.
        
        Args:
            content: Full file content
            function_name: Name of function to extract
            
        Returns:
            Function content if found, None otherwise
        """
        # Simple regex-based extraction (could be improved with AST)
        pattern = rf"^\s*def\s+{re.escape(function_name)}\s*\("
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                # Found function start, now find the end
                start_line = i
                indent_level = len(line) - len(line.lstrip())
                
                # Find end of function
                end_line = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():  # Skip empty lines
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent <= indent_level:
                            end_line = j
                            break
                
                return "\n".join(lines[start_line:end_line])
        
        return None

    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics.
        
        Returns:
            Dictionary with search statistics
        """
        try:
            db_stats = await self.database.get_stats()
            
            return {
                "total_chunks": db_stats.total_chunks,
                "languages": db_stats.languages,
                "similarity_threshold": self.similarity_threshold,
                "project_root": str(self.project_root),
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}
