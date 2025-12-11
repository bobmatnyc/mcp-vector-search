"""Unit tests for code smell detection."""

from src.mcp_vector_search.analysis.collectors.smells import (
    CodeSmell,
    SmellDetector,
    SmellSeverity,
)
from src.mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics
from src.mcp_vector_search.config.thresholds import ThresholdConfig


class TestSmellSeverity:
    """Test SmellSeverity enum."""

    def test_severity_values(self):
        """Test that severity enum has expected values."""
        assert SmellSeverity.INFO.value == "info"
        assert SmellSeverity.WARNING.value == "warning"
        assert SmellSeverity.ERROR.value == "error"

    def test_severity_string_conversion(self):
        """Test string conversion of severity."""
        assert str(SmellSeverity.INFO) == "info"
        assert str(SmellSeverity.WARNING) == "warning"
        assert str(SmellSeverity.ERROR) == "error"


class TestCodeSmell:
    """Test CodeSmell dataclass."""

    def test_code_smell_creation(self):
        """Test creating a CodeSmell instance."""
        smell = CodeSmell(
            name="Test Smell",
            description="This is a test smell",
            severity=SmellSeverity.WARNING,
            location="test.py:10",
            metric_value=100.0,
            threshold=50.0,
            suggestion="Fix this issue",
        )

        assert smell.name == "Test Smell"
        assert smell.description == "This is a test smell"
        assert smell.severity == SmellSeverity.WARNING
        assert smell.location == "test.py:10"
        assert smell.metric_value == 100.0
        assert smell.threshold == 50.0
        assert smell.suggestion == "Fix this issue"

    def test_code_smell_string_representation(self):
        """Test string representation of CodeSmell."""
        smell = CodeSmell(
            name="Long Method",
            description="Method is too long",
            severity=SmellSeverity.WARNING,
            location="test.py:10",
            metric_value=100.0,
            threshold=50.0,
        )

        smell_str = str(smell)
        assert "WARNING" in smell_str
        assert "Long Method" in smell_str
        assert "test.py:10" in smell_str


class TestSmellDetector:
    """Test SmellDetector class."""

    def test_detector_initialization_default(self):
        """Test detector initialization with default thresholds."""
        detector = SmellDetector()
        assert detector.thresholds is not None
        assert detector.thresholds.smells.long_method_lines == 50

    def test_detector_initialization_custom(self):
        """Test detector initialization with custom thresholds."""
        custom_config = ThresholdConfig()
        custom_config.smells.long_method_lines = 100

        detector = SmellDetector(thresholds=custom_config)
        assert detector.thresholds.smells.long_method_lines == 100

    def test_detect_long_method_by_lines(self):
        """Test detection of long method by line count."""
        detector = SmellDetector()

        # Create metrics that exceed long method threshold (50 lines)
        metrics = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=2,
            parameter_count=2,
            lines_of_code=60,  # Exceeds threshold of 50
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 1
        assert smells[0].name == "Long Method"
        assert smells[0].severity == SmellSeverity.WARNING
        assert "60 lines" in smells[0].description
        assert smells[0].metric_value == 60.0

    def test_detect_long_method_by_complexity(self):
        """Test detection of long method by cognitive complexity."""
        detector = SmellDetector()

        # Create metrics with high cognitive complexity (>15)
        metrics = ChunkMetrics(
            cognitive_complexity=20,  # Exceeds threshold of 15
            cyclomatic_complexity=8,
            max_nesting_depth=2,
            parameter_count=2,
            lines_of_code=30,  # Below line threshold
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 1
        assert smells[0].name == "Long Method"
        assert smells[0].severity == SmellSeverity.WARNING
        assert "cognitive complexity 20" in smells[0].description

    def test_detect_deep_nesting(self):
        """Test detection of deep nesting."""
        detector = SmellDetector()

        # Create metrics with deep nesting (>4)
        metrics = ChunkMetrics(
            cognitive_complexity=8,
            cyclomatic_complexity=5,
            max_nesting_depth=6,  # Exceeds threshold of 4
            parameter_count=2,
            lines_of_code=30,
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 1
        assert smells[0].name == "Deep Nesting"
        assert smells[0].severity == SmellSeverity.WARNING
        assert "6 levels" in smells[0].description
        assert smells[0].metric_value == 6.0

    def test_detect_long_parameter_list(self):
        """Test detection of long parameter list."""
        detector = SmellDetector()

        # Create metrics with too many parameters (>5)
        metrics = ChunkMetrics(
            cognitive_complexity=3,
            cyclomatic_complexity=2,
            max_nesting_depth=1,
            parameter_count=8,  # Exceeds threshold of 5
            lines_of_code=20,
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 1
        assert smells[0].name == "Long Parameter List"
        assert smells[0].severity == SmellSeverity.WARNING
        assert "8 parameters" in smells[0].description
        assert smells[0].metric_value == 8.0

    def test_detect_complex_method(self):
        """Test detection of complex method by cyclomatic complexity."""
        detector = SmellDetector()

        # Create metrics with high cyclomatic complexity (>10)
        metrics = ChunkMetrics(
            cognitive_complexity=8,
            cyclomatic_complexity=15,  # Exceeds threshold of 10
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=40,
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 1
        assert smells[0].name == "Complex Method"
        assert smells[0].severity == SmellSeverity.WARNING
        assert "cyclomatic complexity: 15" in smells[0].description

    def test_detect_multiple_smells(self):
        """Test detection of multiple smells in one chunk."""
        detector = SmellDetector()

        # Create metrics that violate multiple thresholds
        metrics = ChunkMetrics(
            cognitive_complexity=20,  # Long method
            cyclomatic_complexity=12,  # Complex method
            max_nesting_depth=5,  # Deep nesting
            parameter_count=7,  # Long parameter list
            lines_of_code=60,  # Long method
        )

        smells = detector.detect(metrics, "test.py", 10)

        # Should detect: Long Method, Deep Nesting, Long Parameter List, Complex Method
        assert len(smells) == 4

        smell_names = {smell.name for smell in smells}
        assert "Long Method" in smell_names
        assert "Deep Nesting" in smell_names
        assert "Long Parameter List" in smell_names
        assert "Complex Method" in smell_names

    def test_detect_no_smells(self):
        """Test that clean code produces no smells."""
        detector = SmellDetector()

        # Create metrics that are all below thresholds
        metrics = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=30,
        )

        smells = detector.detect(metrics, "test.py", 10)

        assert len(smells) == 0

    def test_detect_god_class(self):
        """Test detection of God Class smell."""
        detector = SmellDetector()

        # Create file metrics that indicate God Class
        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=600,  # Exceeds 500 line threshold
            method_count=25,  # Exceeds 20 method threshold
        )

        smells = detector.detect_god_class(file_metrics, "test.py")

        assert len(smells) == 1
        assert smells[0].name == "God Class"
        assert smells[0].severity == SmellSeverity.ERROR
        assert "25 methods" in smells[0].description
        assert "600 lines" in smells[0].description

    def test_detect_god_class_only_methods(self):
        """Test that God Class requires both methods AND lines."""
        detector = SmellDetector()

        # High method count but low line count - NOT a God Class
        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=200,  # Below 500 line threshold
            method_count=25,  # Exceeds 20 method threshold
        )

        smells = detector.detect_god_class(file_metrics, "test.py")
        assert len(smells) == 0

    def test_detect_god_class_only_lines(self):
        """Test that God Class requires both methods AND lines."""
        detector = SmellDetector()

        # High line count but low method count - NOT a God Class
        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=600,  # Exceeds 500 line threshold
            method_count=10,  # Below 20 method threshold
        )

        smells = detector.detect_god_class(file_metrics, "test.py")
        assert len(smells) == 0

    def test_detect_all_smells(self):
        """Test detect_all method across entire file."""
        detector = SmellDetector()

        # Create file with multiple chunks and God Class characteristics
        chunk1 = ChunkMetrics(
            cognitive_complexity=20,  # Long method
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=60,  # Long method
        )

        chunk2 = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            max_nesting_depth=5,  # Deep nesting
            parameter_count=7,  # Long parameter list
            lines_of_code=30,
        )

        file_metrics = FileMetrics(
            file_path="test.py",
            total_lines=600,  # God Class
            method_count=25,  # God Class
            chunks=[chunk1, chunk2],
        )

        smells = detector.detect_all(file_metrics, "test.py")

        # Should detect:
        # - Chunk 1: Long Method
        # - Chunk 2: Deep Nesting, Long Parameter List
        # - File: God Class
        assert len(smells) >= 4

        smell_names = {smell.name for smell in smells}
        assert "Long Method" in smell_names
        assert "Deep Nesting" in smell_names
        assert "Long Parameter List" in smell_names
        assert "God Class" in smell_names

    def test_get_smell_summary(self):
        """Test smell summary statistics."""
        detector = SmellDetector()

        smells = [
            CodeSmell(
                name="Long Method",
                description="Test",
                severity=SmellSeverity.WARNING,
                location="test.py:10",
                metric_value=100.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Deep Nesting",
                description="Test",
                severity=SmellSeverity.WARNING,
                location="test.py:20",
                metric_value=6.0,
                threshold=4.0,
            ),
            CodeSmell(
                name="God Class",
                description="Test",
                severity=SmellSeverity.ERROR,
                location="test.py",
                metric_value=25.0,
                threshold=20.0,
            ),
        ]

        summary = detector.get_smell_summary(smells)

        assert summary["total"] == 3
        assert summary["error"] == 1
        assert summary["warning"] == 2
        assert summary["info"] == 0
        assert summary["by_type"]["Long Method"] == 1
        assert summary["by_type"]["Deep Nesting"] == 1
        assert summary["by_type"]["God Class"] == 1

    def test_custom_thresholds(self):
        """Test detection with custom threshold configuration."""
        # Create custom config with stricter thresholds
        custom_config = ThresholdConfig()
        custom_config.smells.long_method_lines = 30
        custom_config.smells.high_complexity = 10

        detector = SmellDetector(thresholds=custom_config)

        # Create metrics that would be OK with defaults but violate custom
        metrics = ChunkMetrics(
            cognitive_complexity=12,  # Would be OK (default: 15), now violates (10)
            cyclomatic_complexity=5,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=40,  # Would be OK (default: 50), now violates (30)
        )

        smells = detector.detect(metrics, "test.py", 10)

        # Should detect Long Method due to stricter thresholds
        assert len(smells) == 1
        assert smells[0].name == "Long Method"

    def test_edge_case_exact_threshold(self):
        """Test that exact threshold values do NOT trigger smells."""
        detector = SmellDetector()

        # Set metrics exactly at thresholds (should NOT trigger)
        metrics = ChunkMetrics(
            cognitive_complexity=15,  # Exactly at threshold
            cyclomatic_complexity=10,  # Exactly at threshold
            max_nesting_depth=4,  # Exactly at threshold
            parameter_count=5,  # Exactly at threshold
            lines_of_code=50,  # Exactly at threshold
        )

        smells = detector.detect(metrics, "test.py", 10)

        # Exact threshold should NOT trigger smells (> not >=)
        assert len(smells) == 0

    def test_edge_case_one_above_threshold(self):
        """Test that values one above threshold DO trigger smells."""
        detector = SmellDetector()

        # Set metrics one above thresholds (should trigger)
        metrics = ChunkMetrics(
            cognitive_complexity=16,  # One above threshold
            cyclomatic_complexity=11,  # One above threshold
            max_nesting_depth=5,  # One above threshold
            parameter_count=6,  # One above threshold
            lines_of_code=51,  # One above threshold
        )

        smells = detector.detect(metrics, "test.py", 10)

        # Should detect all smells
        assert len(smells) == 4
