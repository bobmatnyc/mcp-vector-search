"""Unit tests for boilerplate filtering."""

import pytest

from mcp_vector_search.core.boilerplate import BoilerplateFilter


class TestBoilerplateFilter:
    """Test suite for BoilerplateFilter class."""

    @pytest.fixture
    def filter(self) -> BoilerplateFilter:
        """Create a BoilerplateFilter instance for testing."""
        return BoilerplateFilter()

    # Python boilerplate tests
    def test_python_init_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that __init__ is detected as Python boilerplate."""
        assert filter.is_boilerplate("__init__", "python", "search classes")

    def test_python_str_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that __str__ is detected as Python boilerplate."""
        assert filter.is_boilerplate("__str__", "python", "find methods")

    def test_python_main_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that main is detected as Python boilerplate."""
        assert filter.is_boilerplate("main", "python", "search functions")

    def test_python_custom_method_not_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that custom methods are not boilerplate."""
        assert not filter.is_boilerplate("process_data", "python", "search")
        assert not filter.is_boilerplate("calculate_total", "python", "find")

    # JavaScript/TypeScript boilerplate tests
    def test_javascript_constructor_is_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that constructor is detected as JavaScript boilerplate."""
        assert filter.is_boilerplate("constructor", "javascript", "search classes")
        assert filter.is_boilerplate("constructor", "typescript", "search classes")

    def test_javascript_render_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that render is detected as JavaScript boilerplate."""
        assert filter.is_boilerplate("render", "javascript", "find components")
        assert filter.is_boilerplate("render", "jsx", "find components")

    def test_typescript_use_effect_is_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that useEffect is detected as TypeScript/React boilerplate."""
        assert filter.is_boilerplate("useEffect", "typescript", "search hooks")
        assert filter.is_boilerplate("useEffect", "tsx", "search hooks")

    def test_javascript_custom_method_not_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that custom methods are not boilerplate."""
        assert not filter.is_boilerplate("handleClick", "javascript", "search")
        assert not filter.is_boilerplate("fetchData", "typescript", "find")

    # Dart boilerplate tests
    def test_dart_build_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that build is detected as Dart boilerplate."""
        assert filter.is_boilerplate("build", "dart", "search widgets")

    def test_dart_init_state_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that initState is detected as Dart boilerplate."""
        assert filter.is_boilerplate("initState", "dart", "find lifecycle")

    def test_dart_custom_method_not_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that custom methods are not boilerplate."""
        assert not filter.is_boilerplate("loadData", "dart", "search")

    # PHP boilerplate tests
    def test_php_construct_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that __construct is detected as PHP boilerplate."""
        assert filter.is_boilerplate("__construct", "php", "search classes")

    def test_php_tostring_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that __toString is detected as PHP boilerplate."""
        assert filter.is_boilerplate("__toString", "php", "find methods")

    def test_php_custom_method_not_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that custom methods are not boilerplate."""
        assert not filter.is_boilerplate("processOrder", "php", "search")

    # Ruby boilerplate tests
    def test_ruby_initialize_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that initialize is detected as Ruby boilerplate."""
        assert filter.is_boilerplate("initialize", "ruby", "search classes")

    def test_ruby_to_s_is_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that to_s is detected as Ruby boilerplate."""
        assert filter.is_boilerplate("to_s", "ruby", "find methods")

    def test_ruby_custom_method_not_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that custom methods are not boilerplate."""
        assert not filter.is_boilerplate("process_data", "ruby", "search")

    # Explicit query bypass tests
    def test_explicit_query_bypasses_filtering_python(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that explicit queries for __init__ bypass filtering."""
        # Searching for __init__ should NOT mark it as boilerplate
        assert not filter.is_boilerplate("__init__", "python", "find __init__ methods")
        assert not filter.is_boilerplate("__init__", "python", "show all __init__")
        assert not filter.is_boilerplate("__init__", "python", "search __INIT__")

    def test_explicit_query_bypasses_filtering_javascript(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that explicit queries for constructor bypass filtering."""
        assert not filter.is_boilerplate(
            "constructor", "javascript", "find constructor methods"
        )
        assert not filter.is_boilerplate(
            "constructor", "javascript", "show CONSTRUCTOR"
        )

    def test_explicit_query_bypasses_filtering_dart(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that explicit queries for build bypass filtering."""
        assert not filter.is_boilerplate("build", "dart", "find build methods")
        assert not filter.is_boilerplate("build", "dart", "show BUILD")

    # Penalty calculation tests
    def test_get_penalty_returns_default_for_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that get_penalty returns default penalty for boilerplate."""
        penalty = filter.get_penalty("__init__", "python", "search classes")
        assert penalty == BoilerplateFilter.DEFAULT_PENALTY
        assert penalty == -0.15

    def test_get_penalty_returns_zero_for_non_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that get_penalty returns 0.0 for non-boilerplate."""
        penalty = filter.get_penalty("custom_method", "python", "search")
        assert penalty == 0.0

    def test_get_penalty_custom_penalty_value(self, filter: BoilerplateFilter) -> None:
        """Test that custom penalty values are respected."""
        penalty = filter.get_penalty(
            "__init__", "python", "search classes", penalty=-0.25
        )
        assert penalty == -0.25

    def test_get_penalty_explicit_query_returns_zero(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that explicit queries return 0 penalty."""
        penalty = filter.get_penalty("__init__", "python", "find __init__ methods")
        assert penalty == 0.0

    # Unknown language tests
    def test_unknown_language_returns_no_penalty(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that unknown languages don't trigger penalties."""
        penalty = filter.get_penalty("__init__", "unknown", "search")
        assert penalty == 0.0

        penalty = filter.get_penalty("constructor", "cobol", "find")
        assert penalty == 0.0

    # Edge cases
    def test_empty_name_not_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that empty names are not considered boilerplate."""
        assert not filter.is_boilerplate("", "python", "search")
        assert filter.get_penalty("", "python", "search") == 0.0

    def test_none_name_not_boilerplate(self, filter: BoilerplateFilter) -> None:
        """Test that None names are not considered boilerplate."""
        assert not filter.is_boilerplate(None, "python", "search")
        assert filter.get_penalty(None, "python", "search") == 0.0

    def test_case_sensitivity(self, filter: BoilerplateFilter) -> None:
        """Test that matching is case-sensitive for the name."""
        # Python: __init__ is boilerplate, __INIT__ is not
        assert filter.is_boilerplate("__init__", "python", "search")
        assert not filter.is_boilerplate("__INIT__", "python", "search")

        # But query matching is case-insensitive
        assert not filter.is_boilerplate("__init__", "python", "find __INIT__")

    def test_language_case_insensitivity(self, filter: BoilerplateFilter) -> None:
        """Test that language matching is case-insensitive."""
        assert filter.is_boilerplate("__init__", "Python", "search")
        assert filter.is_boilerplate("__init__", "PYTHON", "search")
        assert filter.is_boilerplate("constructor", "JavaScript", "search")
        assert filter.is_boilerplate("constructor", "TypeScript", "search")

    def test_multiple_languages_share_boilerplate(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test that JavaScript and TypeScript share boilerplate sets."""
        js_languages = ["javascript", "typescript", "jsx", "tsx"]

        for lang in js_languages:
            assert filter.is_boilerplate("constructor", lang, "search")
            assert filter.is_boilerplate("render", lang, "search")
            assert filter.is_boilerplate("useEffect", lang, "search")

    # Integration tests with realistic queries
    def test_realistic_python_search_queries(self, filter: BoilerplateFilter) -> None:
        """Test realistic Python search queries."""
        # Generic search - should penalize __init__
        assert filter.get_penalty("__init__", "python", "database connection") == -0.15

        # Explicit search - should not penalize
        assert filter.get_penalty("__init__", "python", "how to write __init__") == 0.0

        # Custom method - no penalty
        assert filter.get_penalty("connect_db", "python", "database connection") == 0.0

    def test_realistic_javascript_search_queries(
        self, filter: BoilerplateFilter
    ) -> None:
        """Test realistic JavaScript search queries."""
        # Generic search - should penalize constructor
        assert (
            filter.get_penalty("constructor", "javascript", "user authentication")
            == -0.15
        )

        # Explicit search - should not penalize
        assert (
            filter.get_penalty("constructor", "javascript", "constructor examples")
            == 0.0
        )

        # Custom method - no penalty
        assert (
            filter.get_penalty("authenticate", "javascript", "user authentication")
            == 0.0
        )

    def test_empty_query_still_applies_penalty(self, filter: BoilerplateFilter) -> None:
        """Test that empty queries still apply boilerplate penalties."""
        # Empty query should still penalize boilerplate
        assert filter.get_penalty("__init__", "python", "") == -0.15
        assert filter.get_penalty("constructor", "javascript", "") == -0.15
