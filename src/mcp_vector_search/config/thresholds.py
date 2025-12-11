"""Threshold configuration for code quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ComplexityThresholds:
    """Thresholds for complexity metrics."""

    # Cognitive complexity thresholds for grades
    cognitive_a: int = 5  # A grade: 0-5
    cognitive_b: int = 10  # B grade: 6-10
    cognitive_c: int = 20  # C grade: 11-20
    cognitive_d: int = 30  # D grade: 21-30
    # F grade: 31+

    # Cyclomatic complexity thresholds
    cyclomatic_low: int = 4  # Low complexity
    cyclomatic_moderate: int = 10  # Moderate
    cyclomatic_high: int = 20  # High (needs attention)
    # Very high: 21+

    # Nesting depth thresholds
    nesting_warning: int = 3  # Warning level
    nesting_error: int = 5  # Error level

    # Parameter count thresholds
    parameters_warning: int = 4  # Warning level
    parameters_error: int = 7  # Error level

    # Method count thresholds
    methods_warning: int = 10  # Warning level
    methods_error: int = 20  # Error level


@dataclass
class SmellThresholds:
    """Thresholds for code smell detection."""

    # Long method threshold (lines of code)
    long_method_lines: int = 50

    # Too many parameters
    too_many_parameters: int = 5

    # Deep nesting
    deep_nesting_depth: int = 4

    # High complexity
    high_complexity: int = 15

    # God class (too many methods and lines)
    god_class_methods: int = 20
    god_class_lines: int = 500

    # Feature envy (placeholder for future)
    feature_envy_external_calls: int = 5


@dataclass
class ThresholdConfig:
    """Complete threshold configuration."""

    complexity: ComplexityThresholds = field(default_factory=ComplexityThresholds)
    smells: SmellThresholds = field(default_factory=SmellThresholds)

    # Quality gate settings
    fail_on_f_grade: bool = True
    fail_on_smell_count: int = 10  # Fail if more than N smells
    warn_on_d_grade: bool = True

    @classmethod
    def load(cls, path: Path) -> ThresholdConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ThresholdConfig instance
        """
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            ThresholdConfig instance
        """
        complexity_data = data.get("complexity", {})
        smells_data = data.get("smells", {})

        return cls(
            complexity=(
                ComplexityThresholds(**complexity_data)
                if complexity_data
                else ComplexityThresholds()
            ),
            smells=(
                SmellThresholds(**smells_data) if smells_data else SmellThresholds()
            ),
            fail_on_f_grade=data.get("fail_on_f_grade", True),
            fail_on_smell_count=data.get("fail_on_smell_count", 10),
            warn_on_d_grade=data.get("warn_on_d_grade", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "complexity": {
                "cognitive_a": self.complexity.cognitive_a,
                "cognitive_b": self.complexity.cognitive_b,
                "cognitive_c": self.complexity.cognitive_c,
                "cognitive_d": self.complexity.cognitive_d,
                "cyclomatic_low": self.complexity.cyclomatic_low,
                "cyclomatic_moderate": self.complexity.cyclomatic_moderate,
                "cyclomatic_high": self.complexity.cyclomatic_high,
                "nesting_warning": self.complexity.nesting_warning,
                "nesting_error": self.complexity.nesting_error,
                "parameters_warning": self.complexity.parameters_warning,
                "parameters_error": self.complexity.parameters_error,
                "methods_warning": self.complexity.methods_warning,
                "methods_error": self.complexity.methods_error,
            },
            "smells": {
                "long_method_lines": self.smells.long_method_lines,
                "too_many_parameters": self.smells.too_many_parameters,
                "deep_nesting_depth": self.smells.deep_nesting_depth,
                "high_complexity": self.smells.high_complexity,
                "god_class_methods": self.smells.god_class_methods,
                "god_class_lines": self.smells.god_class_lines,
                "feature_envy_external_calls": self.smells.feature_envy_external_calls,
            },
            "fail_on_f_grade": self.fail_on_f_grade,
            "fail_on_smell_count": self.fail_on_smell_count,
            "warn_on_d_grade": self.warn_on_d_grade,
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_grade(self, cognitive_complexity: int) -> str:
        """Get complexity grade based on cognitive complexity.

        Args:
            cognitive_complexity: Cognitive complexity value

        Returns:
            Grade from A to F
        """
        if cognitive_complexity <= self.complexity.cognitive_a:
            return "A"
        elif cognitive_complexity <= self.complexity.cognitive_b:
            return "B"
        elif cognitive_complexity <= self.complexity.cognitive_c:
            return "C"
        elif cognitive_complexity <= self.complexity.cognitive_d:
            return "D"
        else:
            return "F"
