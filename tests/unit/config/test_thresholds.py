"""Unit tests for threshold configuration."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mcp_vector_search.config.thresholds import (
    ComplexityThresholds,
    SmellThresholds,
    ThresholdConfig,
)


class TestComplexityThresholds:
    """Test ComplexityThresholds dataclass."""

    def test_default_values(self):
        """Test that default threshold values are set correctly."""
        thresholds = ComplexityThresholds()

        # Cognitive complexity thresholds
        assert thresholds.cognitive_a == 5
        assert thresholds.cognitive_b == 10
        assert thresholds.cognitive_c == 20
        assert thresholds.cognitive_d == 30

        # Cyclomatic complexity thresholds
        assert thresholds.cyclomatic_low == 4
        assert thresholds.cyclomatic_moderate == 10
        assert thresholds.cyclomatic_high == 20

        # Nesting depth thresholds
        assert thresholds.nesting_warning == 3
        assert thresholds.nesting_error == 5

        # Parameter count thresholds
        assert thresholds.parameters_warning == 4
        assert thresholds.parameters_error == 7

        # Method count thresholds
        assert thresholds.methods_warning == 10
        assert thresholds.methods_error == 20

    def test_custom_values(self):
        """Test creating ComplexityThresholds with custom values."""
        thresholds = ComplexityThresholds(
            cognitive_a=3,
            cognitive_b=8,
            cyclomatic_low=3,
            nesting_warning=2,
        )

        assert thresholds.cognitive_a == 3
        assert thresholds.cognitive_b == 8
        assert thresholds.cyclomatic_low == 3
        assert thresholds.nesting_warning == 2

        # Other values should be defaults
        assert thresholds.cognitive_c == 20
        assert thresholds.cyclomatic_moderate == 10


class TestSmellThresholds:
    """Test SmellThresholds dataclass."""

    def test_default_values(self):
        """Test that default smell threshold values are set correctly."""
        thresholds = SmellThresholds()

        assert thresholds.long_method_lines == 50
        assert thresholds.too_many_parameters == 5
        assert thresholds.deep_nesting_depth == 4
        assert thresholds.high_complexity == 15
        assert thresholds.god_class_methods == 20
        assert thresholds.feature_envy_external_calls == 5

    def test_custom_values(self):
        """Test creating SmellThresholds with custom values."""
        thresholds = SmellThresholds(
            long_method_lines=100,
            too_many_parameters=7,
            god_class_methods=25,
        )

        assert thresholds.long_method_lines == 100
        assert thresholds.too_many_parameters == 7
        assert thresholds.god_class_methods == 25

        # Other values should be defaults
        assert thresholds.deep_nesting_depth == 4
        assert thresholds.high_complexity == 15


class TestThresholdConfig:
    """Test ThresholdConfig dataclass."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        config = ThresholdConfig()

        # Check nested defaults
        assert isinstance(config.complexity, ComplexityThresholds)
        assert isinstance(config.smells, SmellThresholds)

        # Check quality gate defaults
        assert config.fail_on_f_grade is True
        assert config.fail_on_smell_count == 10
        assert config.warn_on_d_grade is True

    def test_custom_values(self):
        """Test creating ThresholdConfig with custom values."""
        custom_complexity = ComplexityThresholds(cognitive_a=3)
        custom_smells = SmellThresholds(long_method_lines=100)

        config = ThresholdConfig(
            complexity=custom_complexity,
            smells=custom_smells,
            fail_on_f_grade=False,
            fail_on_smell_count=20,
            warn_on_d_grade=False,
        )

        assert config.complexity.cognitive_a == 3
        assert config.smells.long_method_lines == 100
        assert config.fail_on_f_grade is False
        assert config.fail_on_smell_count == 20
        assert config.warn_on_d_grade is False

    def test_get_grade_default_thresholds(self):
        """Test grade calculation with default thresholds."""
        config = ThresholdConfig()

        # Test each grade boundary
        assert config.get_grade(0) == "A"
        assert config.get_grade(5) == "A"
        assert config.get_grade(6) == "B"
        assert config.get_grade(10) == "B"
        assert config.get_grade(11) == "C"
        assert config.get_grade(20) == "C"
        assert config.get_grade(21) == "D"
        assert config.get_grade(30) == "D"
        assert config.get_grade(31) == "F"
        assert config.get_grade(100) == "F"

    def test_get_grade_custom_thresholds(self):
        """Test grade calculation with custom thresholds."""
        custom_complexity = ComplexityThresholds(
            cognitive_a=3,
            cognitive_b=8,
            cognitive_c=15,
            cognitive_d=25,
        )
        config = ThresholdConfig(complexity=custom_complexity)

        # Test with custom boundaries
        assert config.get_grade(0) == "A"
        assert config.get_grade(3) == "A"
        assert config.get_grade(4) == "B"
        assert config.get_grade(8) == "B"
        assert config.get_grade(9) == "C"
        assert config.get_grade(15) == "C"
        assert config.get_grade(16) == "D"
        assert config.get_grade(25) == "D"
        assert config.get_grade(26) == "F"


class TestThresholdConfigSerialization:
    """Test ThresholdConfig serialization and deserialization."""

    def test_to_dict(self):
        """Test converting ThresholdConfig to dictionary."""
        config = ThresholdConfig()
        data = config.to_dict()

        # Check structure
        assert "complexity" in data
        assert "smells" in data
        assert "fail_on_f_grade" in data
        assert "fail_on_smell_count" in data
        assert "warn_on_d_grade" in data

        # Check complexity values
        complexity = data["complexity"]
        assert complexity["cognitive_a"] == 5
        assert complexity["cognitive_b"] == 10
        assert complexity["cyclomatic_low"] == 4

        # Check smell values
        smells = data["smells"]
        assert smells["long_method_lines"] == 50
        assert smells["too_many_parameters"] == 5

    def test_from_dict(self):
        """Test creating ThresholdConfig from dictionary."""
        data = {
            "complexity": {
                "cognitive_a": 3,
                "cognitive_b": 8,
                "cyclomatic_low": 3,
            },
            "smells": {
                "long_method_lines": 100,
                "too_many_parameters": 7,
            },
            "fail_on_f_grade": False,
            "fail_on_smell_count": 20,
            "warn_on_d_grade": False,
        }

        config = ThresholdConfig.from_dict(data)

        # Check values were loaded correctly
        assert config.complexity.cognitive_a == 3
        assert config.complexity.cognitive_b == 8
        assert config.complexity.cyclomatic_low == 3

        assert config.smells.long_method_lines == 100
        assert config.smells.too_many_parameters == 7

        assert config.fail_on_f_grade is False
        assert config.fail_on_smell_count == 20
        assert config.warn_on_d_grade is False

        # Check defaults for missing values
        assert config.complexity.cognitive_c == 20  # Default
        assert config.smells.deep_nesting_depth == 4  # Default

    def test_from_dict_empty(self):
        """Test creating ThresholdConfig from empty dictionary."""
        config = ThresholdConfig.from_dict({})

        # Should create default config
        assert config.complexity.cognitive_a == 5
        assert config.smells.long_method_lines == 50
        assert config.fail_on_f_grade is True

    def test_from_dict_partial(self):
        """Test creating ThresholdConfig from partial dictionary."""
        data = {
            "complexity": {
                "cognitive_a": 3,
            },
            "fail_on_f_grade": False,
        }

        config = ThresholdConfig.from_dict(data)

        # Check partial values
        assert config.complexity.cognitive_a == 3
        assert config.fail_on_f_grade is False

        # Check defaults for missing values
        assert config.complexity.cognitive_b == 10
        assert config.smells.long_method_lines == 50
        assert config.fail_on_smell_count == 10

    def test_round_trip_to_dict_from_dict(self):
        """Test that to_dict and from_dict are inverses."""
        original = ThresholdConfig(
            complexity=ComplexityThresholds(cognitive_a=3, cognitive_b=8),
            smells=SmellThresholds(long_method_lines=100),
            fail_on_f_grade=False,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ThresholdConfig.from_dict(data)

        # Verify all values match
        assert restored.complexity.cognitive_a == original.complexity.cognitive_a
        assert restored.complexity.cognitive_b == original.complexity.cognitive_b
        assert restored.smells.long_method_lines == original.smells.long_method_lines
        assert restored.fail_on_f_grade == original.fail_on_f_grade


class TestThresholdConfigYAML:
    """Test ThresholdConfig YAML file operations."""

    def test_save_and_load(self):
        """Test saving and loading configuration from YAML file."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "thresholds.yaml"

            # Create and save config
            original = ThresholdConfig(
                complexity=ComplexityThresholds(cognitive_a=3, cognitive_b=8),
                smells=SmellThresholds(long_method_lines=100),
                fail_on_f_grade=False,
            )
            original.save(config_path)

            # Verify file was created
            assert config_path.exists()

            # Load config
            loaded = ThresholdConfig.load(config_path)

            # Verify values match
            assert loaded.complexity.cognitive_a == 3
            assert loaded.complexity.cognitive_b == 8
            assert loaded.smells.long_method_lines == 100
            assert loaded.fail_on_f_grade is False

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns default config."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"

            config = ThresholdConfig.load(config_path)

            # Should return default config
            assert config.complexity.cognitive_a == 5
            assert config.smells.long_method_lines == 50
            assert config.fail_on_f_grade is True

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if they don't exist."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dir" / "thresholds.yaml"

            config = ThresholdConfig()
            config.save(config_path)

            # Verify file and parent directories were created
            assert config_path.exists()
            assert config_path.parent.exists()

    def test_yaml_format(self):
        """Test that saved YAML file has correct format."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "thresholds.yaml"

            config = ThresholdConfig()
            config.save(config_path)

            # Read the YAML file
            content = config_path.read_text()

            # Verify key sections exist
            assert "complexity:" in content
            assert "smells:" in content
            assert "fail_on_f_grade:" in content
            assert "cognitive_a:" in content
            assert "long_method_lines:" in content

    def test_load_default_thresholds_yaml(self):
        """Test loading the default_thresholds.yaml file from package."""
        # Get path to default thresholds file
        from pathlib import Path

        config_dir = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "mcp_vector_search"
            / "config"
        )
        default_config_path = config_dir / "default_thresholds.yaml"

        # Load default config
        config = ThresholdConfig.load(default_config_path)

        # Verify default values are loaded correctly
        assert config.complexity.cognitive_a == 5
        assert config.complexity.cognitive_b == 10
        assert config.complexity.cognitive_c == 20
        assert config.complexity.cognitive_d == 30
        assert config.smells.long_method_lines == 50
        assert config.fail_on_f_grade is True
        assert config.fail_on_smell_count == 10
        assert config.warn_on_d_grade is True


class TestThresholdConfigIntegration:
    """Integration tests for ThresholdConfig with ChunkMetrics."""

    def test_integration_with_chunk_metrics(self):
        """Test that ThresholdConfig integrates with ChunkMetrics._compute_grade()."""
        from mcp_vector_search.analysis.metrics import ChunkMetrics

        # Create custom thresholds
        custom_thresholds = ThresholdConfig(
            complexity=ComplexityThresholds(
                cognitive_a=10,  # More lenient
                cognitive_b=20,
                cognitive_c=30,
                cognitive_d=40,
            )
        )

        # Create chunk with complexity 15 (would be C with defaults, B with custom)
        chunk = ChunkMetrics(cognitive_complexity=15)

        # With default thresholds
        assert chunk._compute_grade() == "C"

        # With custom thresholds
        assert chunk._compute_grade(custom_thresholds) == "B"

    def test_multiple_chunks_with_custom_thresholds(self):
        """Test multiple chunks with custom threshold configuration."""
        from mcp_vector_search.analysis.metrics import ChunkMetrics

        custom_thresholds = ThresholdConfig(
            complexity=ComplexityThresholds(
                cognitive_a=3,
                cognitive_b=8,
                cognitive_c=15,
                cognitive_d=25,
            )
        )

        chunks = [
            ChunkMetrics(cognitive_complexity=2),  # A
            ChunkMetrics(cognitive_complexity=5),  # B
            ChunkMetrics(cognitive_complexity=12),  # C
            ChunkMetrics(cognitive_complexity=30),  # F
        ]

        grades = [chunk._compute_grade(custom_thresholds) for chunk in chunks]

        assert grades == ["A", "B", "C", "F"]
