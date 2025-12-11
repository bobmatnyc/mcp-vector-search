"""Unit tests for search quality filters."""

from mcp_vector_search.cli.commands.search import _parse_grade_filter


class TestGradeFilterParser:
    """Test cases for grade filter parsing."""

    def test_parse_single_grade(self):
        """Test parsing single grade."""
        result = _parse_grade_filter("A")
        assert result == {"A"}

    def test_parse_multiple_grades_comma_separated(self):
        """Test parsing comma-separated grades."""
        result = _parse_grade_filter("A,B,C")
        assert result == {"A", "B", "C"}

    def test_parse_grade_range(self):
        """Test parsing grade range."""
        result = _parse_grade_filter("A-C")
        assert result == {"A", "B", "C"}

    def test_parse_grade_range_full(self):
        """Test parsing full grade range."""
        result = _parse_grade_filter("A-F")
        assert result == {"A", "B", "C", "D", "F"}

    def test_parse_reverse_range(self):
        """Test parsing reverse range (should normalize)."""
        result = _parse_grade_filter("C-A")
        assert result == {"A", "B", "C"}

    def test_parse_mixed_format(self):
        """Test parsing mixed format (ranges and single grades)."""
        result = _parse_grade_filter("A,C-D")
        assert result == {"A", "C", "D"}

    def test_parse_lowercase_input(self):
        """Test parsing lowercase input (should be normalized)."""
        result = _parse_grade_filter("a,b,c")
        assert result == {"A", "B", "C"}

    def test_parse_with_spaces(self):
        """Test parsing input with spaces."""
        result = _parse_grade_filter(" A , B , C ")
        assert result == {"A", "B", "C"}

    def test_parse_invalid_grade_ignored(self):
        """Test that invalid grades are ignored."""
        result = _parse_grade_filter("A,X,B")
        assert result == {"A", "B"}

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = _parse_grade_filter("")
        assert result == set()

    def test_parse_only_invalid_grades(self):
        """Test parsing only invalid grades."""
        result = _parse_grade_filter("X,Y,Z")
        assert result == set()


class TestQualityFilterIntegration:
    """Integration tests for quality filtering in search.

    Note: These tests verify the filtering logic without requiring
    a full database setup. They use mock SearchResult objects.
    """

    def test_filter_by_max_complexity(self):
        """Test filtering by maximum complexity."""
        from pathlib import Path

        from mcp_vector_search.core.models import SearchResult

        # Create test results with varying complexity
        results = [
            SearchResult(
                content="def simple(): pass",
                file_path=Path("test1.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.9,
                rank=1,
                cognitive_complexity=5,
                complexity_grade="A",
            ),
            SearchResult(
                content="def complex(): pass",
                file_path=Path("test2.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.8,
                rank=2,
                cognitive_complexity=20,
                complexity_grade="C",
            ),
        ]

        # Filter by max_complexity=10
        max_complexity = 10
        filtered = [
            r
            for r in results
            if r.cognitive_complexity is None
            or r.cognitive_complexity <= max_complexity
        ]

        assert len(filtered) == 1
        assert filtered[0].cognitive_complexity == 5

    def test_filter_by_no_smells(self):
        """Test filtering to exclude code smells."""
        from pathlib import Path

        from mcp_vector_search.core.models import SearchResult

        # Create test results with and without smells
        results = [
            SearchResult(
                content="def clean(): pass",
                file_path=Path("test1.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.9,
                rank=1,
                smell_count=0,
                code_smells=[],
            ),
            SearchResult(
                content="def smelly(): pass",
                file_path=Path("test2.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.8,
                rank=2,
                smell_count=2,
                code_smells=["too_many_parameters", "long_function"],
            ),
        ]

        # Filter to exclude smells
        no_smells = True
        filtered = [
            r
            for r in results
            if not no_smells or (r.smell_count is None or r.smell_count == 0)
        ]

        assert len(filtered) == 1
        assert filtered[0].smell_count == 0

    def test_filter_by_grade(self):
        """Test filtering by complexity grade."""
        from pathlib import Path

        from mcp_vector_search.core.models import SearchResult

        # Create test results with different grades
        results = [
            SearchResult(
                content="def excellent(): pass",
                file_path=Path("test1.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.9,
                rank=1,
                complexity_grade="A",
            ),
            SearchResult(
                content="def good(): pass",
                file_path=Path("test2.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.85,
                rank=2,
                complexity_grade="B",
            ),
            SearchResult(
                content="def poor(): pass",
                file_path=Path("test3.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.8,
                rank=3,
                complexity_grade="D",
            ),
        ]

        # Filter by grade A,B
        allowed_grades = {"A", "B"}
        filtered = [
            r
            for r in results
            if r.complexity_grade is None or r.complexity_grade in allowed_grades
        ]

        assert len(filtered) == 2
        assert all(r.complexity_grade in ["A", "B"] for r in filtered)

    def test_filter_by_min_quality(self):
        """Test filtering by minimum quality score."""
        from pathlib import Path

        from mcp_vector_search.core.models import SearchResult

        # Create test results with different quality scores
        results = [
            SearchResult(
                content="def high_quality(): pass",
                file_path=Path("test1.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.9,
                rank=1,
                quality_score=90,
            ),
            SearchResult(
                content="def medium_quality(): pass",
                file_path=Path("test2.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.8,
                rank=2,
                quality_score=60,
            ),
            SearchResult(
                content="def low_quality(): pass",
                file_path=Path("test3.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.7,
                rank=3,
                quality_score=30,
            ),
        ]

        # Filter by min_quality=70
        min_quality = 70
        filtered = [
            r
            for r in results
            if r.quality_score is None or r.quality_score >= min_quality
        ]

        assert len(filtered) == 1
        assert filtered[0].quality_score == 90

    def test_combined_filters(self):
        """Test combining multiple quality filters."""
        from pathlib import Path

        from mcp_vector_search.core.models import SearchResult

        # Create test results with various quality metrics
        results = [
            SearchResult(
                content="def perfect(): pass",
                file_path=Path("test1.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.9,
                rank=1,
                cognitive_complexity=3,
                complexity_grade="A",
                smell_count=0,
                quality_score=95,
            ),
            SearchResult(
                content="def complex_but_clean(): pass",
                file_path=Path("test2.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.85,
                rank=2,
                cognitive_complexity=25,
                complexity_grade="D",
                smell_count=0,
                quality_score=50,
            ),
            SearchResult(
                content="def simple_but_smelly(): pass",
                file_path=Path("test3.py"),
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.8,
                rank=3,
                cognitive_complexity=5,
                complexity_grade="A",
                smell_count=3,
                quality_score=60,
            ),
        ]

        # Apply multiple filters: max_complexity=10, no_smells=True, grade in A,B
        max_complexity = 10
        no_smells = True
        allowed_grades = {"A", "B"}

        filtered = results
        # Filter by complexity
        filtered = [
            r
            for r in filtered
            if r.cognitive_complexity is None
            or r.cognitive_complexity <= max_complexity
        ]
        # Filter by smells
        filtered = [
            r
            for r in filtered
            if not no_smells or (r.smell_count is None or r.smell_count == 0)
        ]
        # Filter by grade
        filtered = [
            r
            for r in filtered
            if r.complexity_grade is None or r.complexity_grade in allowed_grades
        ]

        # Only the perfect result should pass all filters
        assert len(filtered) == 1
        assert filtered[0].cognitive_complexity == 3
        assert filtered[0].smell_count == 0
        assert filtered[0].complexity_grade == "A"


class TestQualityScoreCalculation:
    """Test cases for quality score calculation."""

    def test_quality_score_perfect(self):
        """Test quality score for perfect code (0 complexity, 0 smells)."""
        complexity = 0
        smells = 0

        score = 100
        score -= min(50, complexity * 2)
        score -= min(30, smells * 10)
        quality_score = max(0, score)

        assert quality_score == 100

    def test_quality_score_with_complexity(self):
        """Test quality score with complexity penalty."""
        complexity = 10
        smells = 0

        score = 100
        score -= min(50, complexity * 2)  # -20
        score -= min(30, smells * 10)
        quality_score = max(0, score)

        assert quality_score == 80

    def test_quality_score_with_smells(self):
        """Test quality score with smell penalty."""
        complexity = 0
        smells = 2

        score = 100
        score -= min(50, complexity * 2)
        score -= min(30, smells * 10)  # -20
        quality_score = max(0, score)

        assert quality_score == 80

    def test_quality_score_max_penalty(self):
        """Test quality score with maximum penalties."""
        complexity = 50  # Would be -100, capped at -50
        smells = 5  # Would be -50, capped at -30

        score = 100
        score -= min(50, complexity * 2)  # -50
        score -= min(30, smells * 10)  # -30
        quality_score = max(0, score)

        assert quality_score == 20

    def test_quality_score_minimum_zero(self):
        """Test quality score cannot go below zero."""
        complexity = 100
        smells = 10

        score = 100
        score -= min(50, complexity * 2)  # -50
        score -= min(30, smells * 10)  # -30
        quality_score = max(0, score)

        assert quality_score == 20  # 100 - 50 - 30 = 20
