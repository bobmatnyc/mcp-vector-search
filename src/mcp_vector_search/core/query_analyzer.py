"""Query analysis and suggestion generation for semantic search."""

import re
from typing import Any

from .models import SearchResult


class QueryAnalyzer:
    """Handles query analysis and suggestion generation."""

    def __init__(self, query_processor) -> None:
        """Initialize query analyzer.

        Args:
            query_processor: QueryProcessor instance for preprocessing
        """
        self.query_processor = query_processor

    def analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze search query and provide suggestions for improvement.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with analysis results and suggestions
        """
        analysis = {
            "original_query": query,
            "processed_query": self.query_processor.preprocess_query(query),
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

    def calculate_result_quality(
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
