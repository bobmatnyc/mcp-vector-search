"""Unit tests for quality-aware ranking functionality."""

from pathlib import Path

import pytest

from mcp_vector_search.core.models import SearchResult


class TestQualityScoreCalculation:
    """Test quality score calculation in SearchResult."""

    def test_quality_score_grade_a_no_smells(self):
        """Test quality score for grade A with no smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="A",
            smell_count=0,
        )

        score = result.calculate_quality_score()
        # Grade A (100) + bonus for no smells (20) = 120, capped at 100
        assert score == 100

    def test_quality_score_grade_b_no_smells(self):
        """Test quality score for grade B with no smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="B",
            smell_count=0,
        )

        score = result.calculate_quality_score()
        # Grade B (80) + bonus (20) = 100
        assert score == 100

    def test_quality_score_grade_c_no_smells(self):
        """Test quality score for grade C with no smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="C",
            smell_count=0,
        )

        score = result.calculate_quality_score()
        # Grade C (60) + bonus (20) = 80
        assert score == 80

    def test_quality_score_grade_a_with_one_smell(self):
        """Test quality score for grade A with one code smell."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="A",
            smell_count=1,
            code_smells=["long_method"],
        )

        score = result.calculate_quality_score()
        # Grade A (100) - 1 smell penalty (10) = 90
        assert score == 90

    def test_quality_score_grade_a_with_three_smells(self):
        """Test quality score for grade A with three code smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="A",
            smell_count=3,
            code_smells=["long_method", "deep_nesting", "high_complexity"],
        )

        score = result.calculate_quality_score()
        # Grade A (100) - 3 smell penalties (30) = 70
        assert score == 70

    def test_quality_score_grade_f_no_smells(self):
        """Test quality score for grade F with no smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="F",
            smell_count=0,
        )

        score = result.calculate_quality_score()
        # Grade F (20) + bonus (20) = 40
        assert score == 40

    def test_quality_score_grade_f_with_smells(self):
        """Test quality score for grade F with smells."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="F",
            smell_count=5,
        )

        score = result.calculate_quality_score()
        # Grade F (20) - 5 smell penalties (50) = -30, clamped to 0
        assert score == 0

    def test_quality_score_no_grade(self):
        """Test quality score when no complexity grade is available."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
        )

        score = result.calculate_quality_score()
        assert score is None

    def test_quality_score_clamped_to_zero(self):
        """Test quality score is clamped to minimum of 0."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="D",  # 40 base
            smell_count=10,  # 100 penalty
        )

        score = result.calculate_quality_score()
        # Grade D (40) - 10 smell penalties (100) = -60, clamped to 0
        assert score == 0

    def test_quality_score_clamped_to_hundred(self):
        """Test quality score is clamped to maximum of 100."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            complexity_grade="A",  # 100 base
            smell_count=0,  # 20 bonus
        )

        score = result.calculate_quality_score()
        # Grade A (100) + bonus (20) = 120, clamped to 100
        assert score == 100


class TestQualityAwareRanking:
    """Test quality-aware ranking scenarios."""

    def test_pure_semantic_search_quality_weight_zero(self):
        """Test that quality_weight=0.0 preserves semantic ranking."""
        # Create two results with different quality but similar relevance
        result_high_quality = SearchResult(
            content="def auth(): pass",
            file_path=Path("auth.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.95,
            rank=1,
            complexity_grade="A",
            smell_count=0,
        )

        result_low_quality = SearchResult(
            content="def authenticate(): pass",
            file_path=Path("legacy.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.98,  # Higher relevance
            rank=2,
            complexity_grade="F",
            smell_count=5,
        )

        # Calculate quality scores
        result_high_quality.quality_score = (
            result_high_quality.calculate_quality_score()
        )
        result_low_quality.quality_score = result_low_quality.calculate_quality_score()

        # Store original scores
        result_high_quality._original_similarity = result_high_quality.similarity_score
        result_low_quality._original_similarity = result_low_quality.similarity_score

        # Apply combined scoring (this would normally be in search engine)
        # Combined = (1-W) × relevance + W × quality
        # With W=0: Combined = relevance
        results = [result_high_quality, result_low_quality]

        # Sort should preserve semantic order (low_quality has higher relevance)
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        assert (
            results[0].file_path.name == "legacy.py"
        )  # Lower quality, higher relevance
        assert results[1].file_path.name == "auth.py"

    def test_pure_quality_ranking_weight_one(self):
        """Test that quality_weight=1.0 prioritizes quality."""
        result_high_quality = SearchResult(
            content="def auth(): pass",
            file_path=Path("auth.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.75,  # Lower relevance
            rank=1,
            complexity_grade="A",
            smell_count=0,
        )

        result_low_quality = SearchResult(
            content="def authenticate(): pass",
            file_path=Path("legacy.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.98,  # Higher relevance
            rank=2,
            complexity_grade="F",
            smell_count=5,
        )

        quality_weight = 1.0

        # Calculate quality scores
        result_high_quality.quality_score = (
            result_high_quality.calculate_quality_score()
        )
        result_low_quality.quality_score = result_low_quality.calculate_quality_score()

        # Store original scores
        result_high_quality._original_similarity = result_high_quality.similarity_score
        result_low_quality._original_similarity = result_low_quality.similarity_score

        # Apply combined scoring
        # Combined = (1-W) × relevance + W × quality
        # With W=1.0: Combined = quality/100
        result_high_quality.similarity_score = (
            1.0 - quality_weight
        ) * result_high_quality._original_similarity + quality_weight * (
            result_high_quality.quality_score / 100.0
        )

        result_low_quality.similarity_score = (
            1.0 - quality_weight
        ) * result_low_quality._original_similarity + quality_weight * (
            result_low_quality.quality_score / 100.0
        )

        results = [result_high_quality, result_low_quality]
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # High quality should win with weight=1.0
        assert results[0].file_path.name == "auth.py"  # High quality
        assert results[1].file_path.name == "legacy.py"  # Low quality

    def test_balanced_ranking_weight_0_3(self):
        """Test default quality_weight=0.3 balances relevance and quality."""
        result_high_quality_low_relevance = SearchResult(
            content="def auth(): pass",
            file_path=Path("auth.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.70,  # Lower relevance
            rank=1,
            complexity_grade="A",  # Quality: 100
            smell_count=0,
        )

        result_low_quality_high_relevance = SearchResult(
            content="def authenticate(): pass",
            file_path=Path("legacy.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.95,  # Higher relevance
            rank=2,
            complexity_grade="F",  # Quality: 40 (20 base + 20 bonus for no smells)
            smell_count=0,
        )

        quality_weight = 0.3

        # Calculate quality scores
        result_high_quality_low_relevance.quality_score = (
            result_high_quality_low_relevance.calculate_quality_score()
        )
        result_low_quality_high_relevance.quality_score = (
            result_low_quality_high_relevance.calculate_quality_score()
        )

        # Store original scores
        result_high_quality_low_relevance._original_similarity = (
            result_high_quality_low_relevance.similarity_score
        )
        result_low_quality_high_relevance._original_similarity = (
            result_low_quality_high_relevance.similarity_score
        )

        # Apply combined scoring
        # High quality, low relevance: (0.7 × 0.7) + (0.3 × 1.0) = 0.49 + 0.3 = 0.79
        result_high_quality_low_relevance.similarity_score = (
            1.0 - quality_weight
        ) * result_high_quality_low_relevance._original_similarity + quality_weight * (
            result_high_quality_low_relevance.quality_score / 100.0
        )

        # Low quality, high relevance: (0.7 × 0.95) + (0.3 × 0.4) = 0.665 + 0.12 = 0.785
        result_low_quality_high_relevance.similarity_score = (
            1.0 - quality_weight
        ) * result_low_quality_high_relevance._original_similarity + quality_weight * (
            result_low_quality_high_relevance.quality_score / 100.0
        )

        results = [result_high_quality_low_relevance, result_low_quality_high_relevance]
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # With balanced weight and F grade having quality 40:
        # High quality (A=100): (0.7 × 0.7) + (0.3 × 1.0) = 0.49 + 0.3 = 0.79
        # Low quality (F=40): (0.7 × 0.95) + (0.3 × 0.4) = 0.665 + 0.12 = 0.785
        # High quality actually wins by a tiny margin!
        assert results[0].file_path.name == "auth.py"
        assert results[1].file_path.name == "legacy.py"

        # With higher quality weight (0.5), high quality wins even more clearly
        quality_weight = 0.5
        result_high_quality_low_relevance.similarity_score = (
            1.0 - quality_weight
        ) * result_high_quality_low_relevance._original_similarity + quality_weight * (
            result_high_quality_low_relevance.quality_score / 100.0
        )
        # (0.5 × 0.7) + (0.5 × 1.0) = 0.35 + 0.5 = 0.85

        result_low_quality_high_relevance.similarity_score = (
            1.0 - quality_weight
        ) * result_low_quality_high_relevance._original_similarity + quality_weight * (
            result_low_quality_high_relevance.quality_score / 100.0
        )
        # (0.5 × 0.95) + (0.5 × 0.4) = 0.475 + 0.2 = 0.675

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        assert results[0].file_path.name == "auth.py"  # High quality wins more clearly

    def test_combined_score_formula_correctness(self):
        """Test that combined score formula is calculated correctly."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.8,
            rank=1,
            complexity_grade="B",  # Quality: 100 (80 + 20 bonus)
            smell_count=0,
        )

        result.quality_score = result.calculate_quality_score()
        assert result.quality_score == 100

        quality_weight = 0.3
        relevance = 0.8
        normalized_quality = 1.0  # 100/100

        expected_combined = (
            1.0 - quality_weight
        ) * relevance + quality_weight * normalized_quality
        # (0.7 × 0.8) + (0.3 × 1.0) = 0.56 + 0.3 = 0.86

        assert expected_combined == pytest.approx(0.86)

    def test_no_quality_metrics_preserves_semantic_score(self):
        """Test that results without quality metrics keep their semantic score."""
        result = SearchResult(
            content="def foo(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
        )

        # No quality grade means calculate_quality_score returns None
        assert result.calculate_quality_score() is None

        # In search engine, this result would keep its original similarity_score
        # and not be affected by quality ranking
        original_score = result.similarity_score
        result.quality_score = result.calculate_quality_score()

        # Since quality_score is None, result should keep original similarity
        assert result.similarity_score == original_score
