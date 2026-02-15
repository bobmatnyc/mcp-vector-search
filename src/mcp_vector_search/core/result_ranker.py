"""Result ranking and scoring for semantic search."""

from .boilerplate import BoilerplateFilter
from .models import SearchResult


class ResultRanker:
    """Handles reranking and scoring of search results."""

    # Reranking boost constants (class-level for performance)
    _BOOST_EXACT_IDENTIFIER = 0.15
    _BOOST_PARTIAL_IDENTIFIER = 0.05
    _BOOST_FILE_NAME_EXACT = 0.08
    _BOOST_FILE_NAME_PARTIAL = 0.03
    _BOOST_FUNCTION_CHUNK = 0.05
    _BOOST_CLASS_CHUNK = 0.03
    _BOOST_SOURCE_FILE = 0.02
    _BOOST_SHALLOW_PATH = 0.02
    _PENALTY_TEST_FILE = -0.02
    _PENALTY_DEEP_PATH = -0.01
    _PENALTY_BOILERPLATE = -0.15
    # NLP entity match boosts
    _BOOST_NLP_KEYWORD = 0.02  # Per keyword match
    _BOOST_NLP_CODE_REF = 0.03  # Per code reference match
    _BOOST_NLP_TECHNICAL_TERM = 0.02  # Per technical term match

    def __init__(self) -> None:
        """Initialize result ranker."""
        self._boilerplate_filter = BoilerplateFilter()

    def rerank_results(
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

        # Pre-compute lowercased strings once (avoid repeated .lower() calls)
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Pre-compute file extensions for source files
        source_exts = frozenset(
            [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]
        )

        for result in results:
            # Start with base similarity score
            score = result.similarity_score

            # Factor 1: Exact matches in identifiers (high boost)
            if result.function_name:
                func_name_lower = result.function_name.lower()
                if query_lower in func_name_lower:
                    score += self._BOOST_EXACT_IDENTIFIER
                # Partial word matches
                score += sum(
                    self._BOOST_PARTIAL_IDENTIFIER
                    for word in query_words
                    if word in func_name_lower
                )

            if result.class_name:
                class_name_lower = result.class_name.lower()
                if query_lower in class_name_lower:
                    score += self._BOOST_EXACT_IDENTIFIER
                # Partial word matches
                score += sum(
                    self._BOOST_PARTIAL_IDENTIFIER
                    for word in query_words
                    if word in class_name_lower
                )

            # Factor 2: File name relevance
            file_name_lower = result.file_path.name.lower()
            if query_lower in file_name_lower:
                score += self._BOOST_FILE_NAME_EXACT
            score += sum(
                self._BOOST_FILE_NAME_PARTIAL
                for word in query_words
                if word in file_name_lower
            )

            # Factor 3: Content density (how many query words appear)
            content_lower = result.content.lower()
            word_matches = sum(1 for word in query_words if word in content_lower)
            if word_matches > 0:
                score += (word_matches / len(query_words)) * 0.1

            # Factor 4: Code structure preferences (combined conditions)
            if result.chunk_type == "function":
                score += self._BOOST_FUNCTION_CHUNK
            elif result.chunk_type == "class":
                score += self._BOOST_CLASS_CHUNK

            # Factor 5: File type preferences (prefer source files over tests)
            file_ext = result.file_path.suffix.lower()
            if file_ext in source_exts:
                score += self._BOOST_SOURCE_FILE
            if "test" in file_name_lower:  # Already computed
                score += self._PENALTY_TEST_FILE

            # Factor 6: Path depth preference
            path_depth = len(result.file_path.parts)
            if path_depth <= 3:
                score += self._BOOST_SHALLOW_PATH
            elif path_depth > 5:
                score += self._PENALTY_DEEP_PATH

            # Factor 7: Boilerplate penalty (penalize common boilerplate patterns)
            # Apply penalty to function names (constructors, lifecycle methods, etc.)
            if result.function_name:
                boilerplate_penalty = self._boilerplate_filter.get_penalty(
                    name=result.function_name,
                    language=result.language,
                    query=query,
                    penalty=self._PENALTY_BOILERPLATE,
                )
                score += boilerplate_penalty

            # Factor 8: NLP entity matches (boost results with matching NLP-extracted terms)
            score += self._calculate_nlp_boost(result, query_words)

            # Ensure score doesn't exceed 1.0
            result.similarity_score = min(1.0, score)

        # Sort by enhanced similarity score
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _calculate_nlp_boost(
        self, result: SearchResult, query_words: set[str]
    ) -> float:
        """Calculate boost based on NLP entity matches.

        Args:
            result: Search result to boost
            query_words: Set of query words (lowercased)

        Returns:
            Boost score to add to similarity
        """
        boost = 0.0

        # Check if result has NLP attributes (from metadata)
        if not hasattr(result, "content"):
            return boost

        # Parse NLP fields from metadata (stored as comma-separated strings in LanceDB)
        # These are passed through from the database as metadata attributes
        nlp_keywords = []
        nlp_code_refs = []
        nlp_technical_terms = []

        # Try to get from metadata dict if present
        if hasattr(result, "__dict__"):
            metadata = result.__dict__
            if "nlp_keywords" in metadata and metadata["nlp_keywords"]:
                nlp_keywords = (
                    metadata["nlp_keywords"].split(",")
                    if isinstance(metadata["nlp_keywords"], str)
                    else metadata["nlp_keywords"]
                )
            if "nlp_code_refs" in metadata and metadata["nlp_code_refs"]:
                nlp_code_refs = (
                    metadata["nlp_code_refs"].split(",")
                    if isinstance(metadata["nlp_code_refs"], str)
                    else metadata["nlp_code_refs"]
                )
            if "nlp_technical_terms" in metadata and metadata["nlp_technical_terms"]:
                nlp_technical_terms = (
                    metadata["nlp_technical_terms"].split(",")
                    if isinstance(metadata["nlp_technical_terms"], str)
                    else metadata["nlp_technical_terms"]
                )

        # Boost for keyword matches
        for keyword in nlp_keywords:
            keyword_lower = keyword.lower()
            if any(word in keyword_lower for word in query_words):
                boost += self._BOOST_NLP_KEYWORD

        # Boost for code reference matches (stronger signal)
        for ref in nlp_code_refs:
            ref_lower = ref.lower()
            if any(word in ref_lower for word in query_words):
                boost += self._BOOST_NLP_CODE_REF

        # Boost for technical term matches
        for term in nlp_technical_terms:
            term_lower = term.lower()
            if any(word in term_lower for word in query_words):
                boost += self._BOOST_NLP_TECHNICAL_TERM

        return boost
