"""Semantic search engine for MCP Vector Search."""

import re
from pathlib import Path
from typing import Any

from loguru import logger

from .auto_indexer import AutoIndexer, SearchTriggeredIndexer
from .database import VectorDatabase
from .exceptions import SearchError
from .models import SearchResult


class SemanticSearchEngine:
    """Semantic search engine for code search."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.3,
        auto_indexer: AutoIndexer | None = None,
        enable_auto_reindex: bool = True,
    ) -> None:
        """Initialize semantic search engine.

        Args:
            database: Vector database instance
            project_root: Project root directory
            similarity_threshold: Default similarity threshold
            auto_indexer: Optional auto-indexer for semi-automatic reindexing
            enable_auto_reindex: Whether to enable automatic reindexing
        """
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold
        self.auto_indexer = auto_indexer
        self.enable_auto_reindex = enable_auto_reindex

        # Initialize search-triggered indexer if auto-indexer is provided
        self.search_triggered_indexer = None
        if auto_indexer and enable_auto_reindex:
            self.search_triggered_indexer = SearchTriggeredIndexer(auto_indexer)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        include_context: bool = True,
    ) -> list[SearchResult]:
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

        # Health check before search
        try:
            if hasattr(self.database, 'health_check'):
                is_healthy = await self.database.health_check()
                if not is_healthy:
                    logger.warning("Database health check failed - attempting recovery")
                    # Health check already attempts recovery, so we can proceed
        except Exception as e:
            logger.warning(f"Health check failed: {e}")

        # Auto-reindex check before search
        if self.search_triggered_indexer:
            try:
                await self.search_triggered_indexer.pre_search_hook()
            except Exception as e:
                logger.warning(f"Auto-reindex check failed: {e}")

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._get_adaptive_threshold(query)
        )

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

            logger.debug(
                f"Search for '{query}' with threshold {threshold:.3f} returned {len(ranked_results)} results"
            )
            return ranked_results

        except Exception as e:
            error_msg = str(e).lower()
            # Check for corruption indicators
            if any(indicator in error_msg for indicator in [
                "pickle", "unpickling", "eof", "ran out of input",
                "hnsw", "index", "deserialize", "corrupt"
            ]):
                logger.error(f"Index corruption detected during search: {e}")
                logger.info("The index appears to be corrupted. Please run 'mcp-vector-search reset' to clear the index and then 'mcp-vector-search index' to rebuild it.")
                raise SearchError(
                    "Index corruption detected. Please run 'mcp-vector-search reset' followed by 'mcp-vector-search index' to rebuild."
                ) from e
            else:
                logger.error(f"Search failed for query '{query}': {e}")
                raise SearchError(f"Search failed: {e}") from e

    async def search_similar(
        self,
        file_path: Path,
        function_name: str | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
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
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # If function name is specified, try to extract just that function
            if function_name:
                function_content = self._extract_function_content(
                    content, function_name
                )
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
        focus_areas: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
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

        # Expand common programming abbreviations and synonyms
        expansions = {
            "auth": "authentication authorize login",
            "db": "database data storage",
            "api": "application programming interface endpoint",
            "ui": "user interface frontend view",
            "util": "utility helper function",
            "config": "configuration settings options",
            "async": "asynchronous await promise",
            "sync": "synchronous blocking",
            "func": "function method",
            "var": "variable",
            "param": "parameter argument",
            "init": "initialize setup create",
            "parse": "parsing parser analyze",
            "validate": "validation check verify",
            "handle": "handler process manage",
            "error": "exception failure bug",
            "test": "testing unittest spec",
            "mock": "mocking stub fake",
            "log": "logging logger debug",
        }

        # Add programming language keywords and concepts
        programming_concepts = {
            "class": "class object type",
            "method": "method function procedure",
            "property": "property attribute field",
            "import": "import require include",
            "export": "export module public",
            "return": "return yield output",
            "loop": "loop iterate for while",
            "condition": "condition if else branch",
            "array": "array list collection",
            "string": "string text character",
            "number": "number integer float",
            "boolean": "boolean true false",
        }

        # Merge all expansions
        all_expansions = {**expansions, **programming_concepts}

        words = query.lower().split()
        expanded_words = []

        for word in words:
            # Add original word
            expanded_words.append(word)

            # Add expansions if available
            if word in all_expansions:
                expanded_words.extend(all_expansions[word].split())

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in expanded_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        return " ".join(unique_words)

    def _get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive similarity threshold based on query characteristics.

        Args:
            query: Search query

        Returns:
            Adaptive similarity threshold
        """
        base_threshold = self.similarity_threshold
        query_lower = query.lower()
        words = query.split()

        # Adjust threshold based on query characteristics

        # 1. Single word queries - lower threshold for broader results
        if len(words) == 1:
            return max(0.01, base_threshold - 0.29)

        # 2. Very specific technical terms - lower threshold
        technical_terms = [
            "javascript",
            "typescript",
            "python",
            "java",
            "cpp",
            "rust",
            "go",
            "function",
            "class",
            "method",
            "variable",
            "import",
            "export",
            "async",
            "await",
            "promise",
            "callback",
            "api",
            "database",
            "parser",
            "compiler",
            "interpreter",
            "syntax",
            "semantic",
            "mcp",
            "gateway",
            "server",
            "client",
            "protocol",
        ]

        if any(term in query_lower for term in technical_terms):
            return max(0.01, base_threshold - 0.29)

        # 3. Short queries (2-3 words) - slightly lower threshold
        if len(words) <= 3:
            return max(0.1, base_threshold - 0.1)

        # 4. Long queries (>6 words) - higher threshold for precision
        if len(words) > 6:
            return min(0.8, base_threshold + 0.1)

        # 5. Queries with exact identifiers (CamelCase, snake_case)
        if re.search(r"\b[A-Z][a-zA-Z]*\b", query) or "_" in query:
            return max(0.05, base_threshold - 0.25)

        # 6. Common programming patterns
        if any(pattern in query for pattern in ["()", ".", "->", "=>", "::"]):
            return max(0.25, base_threshold - 0.1)

        return base_threshold

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
            with open(result.file_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Get context lines before and after
            context_size = 3
            start_idx = max(0, result.start_line - 1 - context_size)
            end_idx = min(len(lines), result.end_line + context_size)

            context_before = [
                line.rstrip() for line in lines[start_idx : result.start_line - 1]
            ]
            context_after = [line.rstrip() for line in lines[result.end_line : end_idx]]

            # Update result with context
            result.context_before = context_before
            result.context_after = context_after

        except Exception as e:
            logger.warning(f"Failed to get context for {result.file_path}: {e}")

        return result

    def _rerank_results(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """Apply advanced ranking to search results using multiple factors.

        Args:
            results: Original search results
            query: Original search query

        Returns:
            Reranked search results
        """
        if not results:
            return results

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for result in results:
            # Start with base similarity score
            score = result.similarity_score

            # Factor 1: Exact matches in identifiers (high boost)
            if result.function_name:
                func_name_lower = result.function_name.lower()
                if query_lower in func_name_lower:
                    score += 0.15  # Strong boost for function name match
                # Partial word matches
                for word in query_words:
                    if word in func_name_lower:
                        score += 0.05

            if result.class_name:
                class_name_lower = result.class_name.lower()
                if query_lower in class_name_lower:
                    score += 0.15  # Strong boost for class name match
                # Partial word matches
                for word in query_words:
                    if word in class_name_lower:
                        score += 0.05

            # Factor 2: File name relevance
            file_name_lower = result.file_path.name.lower()
            if query_lower in file_name_lower:
                score += 0.08
            for word in query_words:
                if word in file_name_lower:
                    score += 0.03

            # Factor 3: Content density (how many query words appear)
            content_lower = result.content.lower()
            word_matches = sum(1 for word in query_words if word in content_lower)
            if word_matches > 0:
                density_boost = (word_matches / len(query_words)) * 0.1
                score += density_boost

            # Factor 4: Code structure preferences
            # Boost functions over general code blocks
            if result.chunk_type == "function":
                score += 0.05
            elif result.chunk_type == "class":
                score += 0.03

            # Factor 5: File type preferences (prefer source files over tests/docs)
            file_ext = result.file_path.suffix.lower()
            if file_ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]:
                score += 0.02
            elif "test" in result.file_path.name.lower():
                score -= 0.02  # Slightly penalize test files unless specifically searching for tests

            # Factor 6: Recency bias (prefer shorter file paths - often more core files)
            path_depth = len(result.file_path.parts)
            if path_depth <= 3:
                score += 0.02
            elif path_depth > 5:
                score -= 0.01

            # Ensure score doesn't exceed 1.0
            result.similarity_score = min(1.0, score)

        # Sort by enhanced similarity score
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze search query and provide suggestions for improvement.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with analysis results and suggestions
        """
        analysis = {
            "original_query": query,
            "processed_query": self._preprocess_query(query),
            "query_type": "general",
            "suggestions": [],
            "confidence": "medium",
        }

        query_lower = query.lower()

        # Detect query type
        if any(word in query_lower for word in ["function", "method", "def", "func"]):
            analysis["query_type"] = "function_search"
            analysis["suggestions"].append(
                "Try searching for specific function names or patterns"
            )
        elif any(word in query_lower for word in ["class", "object", "type"]):
            analysis["query_type"] = "class_search"
            analysis["suggestions"].append(
                "Include class inheritance or interface information"
            )
        elif any(word in query_lower for word in ["error", "exception", "bug", "fix"]):
            analysis["query_type"] = "error_handling"
            analysis["suggestions"].append("Include error types or exception names")
        elif any(word in query_lower for word in ["test", "spec", "mock"]):
            analysis["query_type"] = "testing"
            analysis["suggestions"].append("Specify test framework or testing patterns")
        elif any(word in query_lower for word in ["config", "setting", "option"]):
            analysis["query_type"] = "configuration"
            analysis["suggestions"].append(
                "Include configuration file types or setting names"
            )

        # Analyze query complexity
        words = query.split()
        if len(words) == 1:
            analysis["confidence"] = "low"
            analysis["suggestions"].append(
                "Try adding more descriptive words for better results"
            )
        elif len(words) > 10:
            analysis["confidence"] = "low"
            analysis["suggestions"].append(
                "Consider simplifying your query for better matching"
            )
        else:
            analysis["confidence"] = "high"

        # Check for common programming patterns
        if re.search(r"\b\w+\(\)", query):
            analysis["suggestions"].append(
                "Function call detected - searching for function definitions"
            )
        if re.search(r"\b[A-Z][a-zA-Z]*\b", query):
            analysis["suggestions"].append(
                "CamelCase detected - searching for class or type names"
            )
        if re.search(r"\b\w+\.\w+", query):
            analysis["suggestions"].append(
                "Dot notation detected - searching for method calls or properties"
            )

        return analysis

    def suggest_related_queries(
        self, query: str, results: list[SearchResult]
    ) -> list[str]:
        """Suggest related queries based on search results.

        Args:
            query: Original search query
            results: Search results

        Returns:
            List of suggested related queries
        """
        suggestions = []

        if not results:
            # No results - suggest broader queries
            words = query.lower().split()
            if len(words) > 1:
                # Try individual words
                suggestions.extend(words[:3])  # Top 3 words

            # Suggest common related terms
            related_terms = {
                "auth": ["login", "user", "session", "token"],
                "database": ["query", "model", "schema", "connection"],
                "api": ["endpoint", "request", "response", "handler"],
                "test": ["mock", "assert", "spec", "unit"],
                "error": ["exception", "handle", "catch", "debug"],
            }

            for word in words:
                if word in related_terms:
                    suggestions.extend(related_terms[word][:2])
        else:
            # Extract common patterns from results
            function_names = [r.function_name for r in results if r.function_name]
            class_names = [r.class_name for r in results if r.class_name]

            # Suggest function names
            if function_names:
                unique_functions = list(set(function_names))[:3]
                suggestions.extend(unique_functions)

            # Suggest class names
            if class_names:
                unique_classes = list(set(class_names))[:3]
                suggestions.extend(unique_classes)

            # Suggest file-based queries
            file_patterns = set()
            for result in results[:5]:  # Top 5 results
                file_name = result.file_path.stem
                if "_" in file_name:
                    file_patterns.update(file_name.split("_"))
                elif file_name not in suggestions:
                    file_patterns.add(file_name)

            suggestions.extend(list(file_patterns)[:3])

        # Remove duplicates and original query words
        query_words = set(query.lower().split())
        unique_suggestions = []
        for suggestion in suggestions:
            if (
                suggestion
                and suggestion.lower() not in query_words
                and suggestion not in unique_suggestions
            ):
                unique_suggestions.append(suggestion)

        return unique_suggestions[:5]  # Return top 5 suggestions

    async def search_with_context(
        self,
        query: str,
        context_files: list[Path] | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Enhanced search with contextual analysis and suggestions.

        Args:
            query: Search query
            context_files: Optional list of files to provide context
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            Dictionary with results, analysis, and suggestions
        """
        # Analyze the query
        query_analysis = self.analyze_query(query)

        # Perform the search
        results = await self.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            include_context=True,
        )

        # Get related query suggestions
        suggestions = self.suggest_related_queries(query, results)

        # Enhance results with additional context if context files provided
        if context_files:
            results = await self._enhance_with_file_context(results, context_files)

        # Calculate result quality metrics
        quality_metrics = self._calculate_result_quality(results, query)

        return {
            "query": query,
            "analysis": query_analysis,
            "results": results,
            "suggestions": suggestions,
            "metrics": quality_metrics,
            "total_results": len(results),
        }

    async def _enhance_with_file_context(
        self, results: list[SearchResult], context_files: list[Path]
    ) -> list[SearchResult]:
        """Enhance results by considering context from specific files.

        Args:
            results: Original search results
            context_files: Files to use for context

        Returns:
            Enhanced search results
        """
        # Read context from files
        context_content = []
        for file_path in context_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    context_content.append(content)
            except Exception as e:
                logger.warning(f"Failed to read context file {file_path}: {e}")

        if not context_content:
            return results

        # Boost results that are related to context files
        context_text = " ".join(context_content).lower()

        for result in results:
            # Check if result is from one of the context files
            if result.file_path in context_files:
                result.similarity_score = min(1.0, result.similarity_score + 0.1)

            # Check if result content relates to context
            result.content.lower()
            if result.function_name:
                func_name_lower = result.function_name.lower()
                if func_name_lower in context_text:
                    result.similarity_score = min(1.0, result.similarity_score + 0.05)

            if result.class_name:
                class_name_lower = result.class_name.lower()
                if class_name_lower in context_text:
                    result.similarity_score = min(1.0, result.similarity_score + 0.05)

        # Re-sort by updated scores
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _calculate_result_quality(
        self, results: list[SearchResult], query: str
    ) -> dict[str, Any]:
        """Calculate quality metrics for search results.

        Args:
            results: Search results
            query: Original query

        Returns:
            Dictionary with quality metrics
        """
        if not results:
            return {
                "average_score": 0.0,
                "score_distribution": {},
                "diversity": 0.0,
                "coverage": 0.0,
            }

        # Calculate average similarity score
        scores = [r.similarity_score for r in results]
        avg_score = sum(scores) / len(scores)

        # Score distribution
        high_quality = sum(1 for s in scores if s >= 0.8)
        medium_quality = sum(1 for s in scores if 0.6 <= s < 0.8)
        low_quality = sum(1 for s in scores if s < 0.6)

        # Diversity (unique files)
        unique_files = len({r.file_path for r in results})
        diversity = unique_files / len(results) if results else 0.0

        # Coverage (how many query words are covered)
        query_words = set(query.lower().split())
        covered_words = set()
        for result in results:
            content_words = set(result.content.lower().split())
            covered_words.update(query_words.intersection(content_words))

        coverage = len(covered_words) / len(query_words) if query_words else 0.0

        return {
            "average_score": round(avg_score, 3),
            "score_distribution": {
                "high_quality": high_quality,
                "medium_quality": medium_quality,
                "low_quality": low_quality,
            },
            "diversity": round(diversity, 3),
            "coverage": round(coverage, 3),
        }

    def _extract_function_content(self, content: str, function_name: str) -> str | None:
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

    async def get_search_stats(self) -> dict[str, Any]:
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
