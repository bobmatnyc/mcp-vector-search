"""Boilerplate filtering for semantic search results.

This module provides language-specific filtering to penalize common boilerplate
code patterns (constructors, lifecycle methods, etc.) in search results while
still preserving them when explicitly queried.
"""

from typing import Final

# Language-specific boilerplate function/method names
# Using frozensets for O(1) lookup performance
_PYTHON_BOILERPLATE: Final[frozenset[str]] = frozenset(
    {
        "__init__",
        "__str__",
        "__repr__",
        "__eq__",
        "__hash__",
        "__len__",
        "__iter__",
        "__next__",
        "__enter__",
        "__exit__",
        "main",
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
    }
)

_JAVASCRIPT_TYPESCRIPT_BOILERPLATE: Final[frozenset[str]] = frozenset(
    {
        "constructor",
        "render",
        "componentDidMount",
        "componentWillUnmount",
        "componentDidUpdate",
        "useState",
        "useEffect",
        "index",
        "main",
        "default",
    }
)

_DART_BOILERPLATE: Final[frozenset[str]] = frozenset(
    {
        "build",
        "dispose",
        "initState",
        "didChangeDependencies",
        "main",
        "createState",
    }
)

_PHP_BOILERPLATE: Final[frozenset[str]] = frozenset(
    {
        "__construct",
        "__destruct",
        "__toString",
        "__get",
        "__set",
        "__call",
        "__callStatic",
        "index",
        "main",
    }
)

_RUBY_BOILERPLATE: Final[frozenset[str]] = frozenset(
    {
        "initialize",
        "to_s",
        "to_h",
        "to_a",
        "inspect",
        "main",
        "setup",
        "teardown",
    }
)


class BoilerplateFilter:
    """Filter for identifying and penalizing boilerplate code patterns.

    This filter applies language-specific penalties to common boilerplate
    patterns (constructors, lifecycle methods, etc.) to improve search
    result relevance. It avoids filtering when the user explicitly searches
    for a boilerplate pattern.

    Example:
        filter = BoilerplateFilter()

        # Returns penalty for __init__ in Python
        penalty = filter.get_penalty("__init__", "python", "search classes")

        # Returns 0.0 when explicitly searching for __init__
        penalty = filter.get_penalty("__init__", "python", "find __init__ methods")
    """

    # Default penalty for boilerplate patterns
    DEFAULT_PENALTY: Final[float] = -0.15

    # Mapping of language identifiers to boilerplate sets
    _LANGUAGE_BOILERPLATE: Final[dict[str, frozenset[str]]] = {
        "python": _PYTHON_BOILERPLATE,
        "javascript": _JAVASCRIPT_TYPESCRIPT_BOILERPLATE,
        "typescript": _JAVASCRIPT_TYPESCRIPT_BOILERPLATE,
        "jsx": _JAVASCRIPT_TYPESCRIPT_BOILERPLATE,
        "tsx": _JAVASCRIPT_TYPESCRIPT_BOILERPLATE,
        "dart": _DART_BOILERPLATE,
        "php": _PHP_BOILERPLATE,
        "ruby": _RUBY_BOILERPLATE,
    }

    def is_boilerplate(self, name: str, language: str, query: str) -> bool:
        """Check if a function/method name is considered boilerplate.

        Args:
            name: Function or method name to check
            language: Programming language (e.g., "python", "javascript")
            query: Original search query

        Returns:
            True if the name is boilerplate AND not explicitly in the query

        Example:
            >>> filter = BoilerplateFilter()
            >>> filter.is_boilerplate("__init__", "python", "find classes")
            True
            >>> filter.is_boilerplate("__init__", "python", "show __init__ methods")
            False
        """
        if not name:
            return False

        # Don't filter if user explicitly searched for this boilerplate
        query_lower = query.lower()
        name_lower = name.lower()
        if name_lower in query_lower:
            return False

        # Check if name is in language's boilerplate set
        language_lower = language.lower()
        boilerplate_set = self._LANGUAGE_BOILERPLATE.get(language_lower)

        if boilerplate_set is None:
            # Unknown language - no filtering
            return False

        # Exact match (case-sensitive for languages like Python)
        return name in boilerplate_set

    def get_penalty(
        self,
        name: str,
        language: str,
        query: str = "",
        penalty: float | None = None,
    ) -> float:
        """Calculate penalty for a potentially boilerplate name.

        Args:
            name: Function or method name to check
            language: Programming language (e.g., "python", "javascript")
            query: Original search query (default: "")
            penalty: Custom penalty value (default: uses DEFAULT_PENALTY)

        Returns:
            Penalty value (negative float) if boilerplate, 0.0 otherwise

        Example:
            >>> filter = BoilerplateFilter()
            >>> filter.get_penalty("__init__", "python", "search classes")
            -0.15
            >>> filter.get_penalty("__init__", "python", "find __init__")
            0.0
            >>> filter.get_penalty("custom_method", "python", "search")
            0.0
        """
        if self.is_boilerplate(name, language, query):
            return penalty if penalty is not None else self.DEFAULT_PENALTY
        return 0.0
