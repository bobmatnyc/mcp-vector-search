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
class CouplingThresholds:
    """Thresholds for coupling and instability metrics."""

    # Efferent coupling (Ce) thresholds
    efferent_low: int = 3  # Low coupling (0-3 dependencies)
    efferent_moderate: int = 7  # Moderate coupling (4-7)
    efferent_high: int = 12  # High coupling (8-12)
    # Very high: 13+

    # Afferent coupling (Ca) thresholds
    afferent_low: int = 2  # Low coupling (0-2 dependents)
    afferent_moderate: int = 5  # Moderate coupling (3-5)
    afferent_high: int = 10  # High coupling (6-10)
    # Very high: 11+

    # Instability (I) thresholds for grades
    instability_a: float = 0.2  # A grade: very stable (0.0-0.2)
    instability_b: float = 0.4  # B grade: stable (0.2-0.4)
    instability_c: float = 0.6  # C grade: balanced (0.4-0.6)
    instability_d: float = 0.8  # D grade: unstable (0.6-0.8)
    # F grade: very unstable (0.8-1.0)

    # Category thresholds
    stable_max: float = 0.3  # Stable category (0.0-0.3)
    balanced_max: float = 0.7  # Balanced category (0.3-0.7)
    # Unstable category: 0.7-1.0


@dataclass
class ThresholdConfig:
    """Complete threshold configuration."""

    complexity: ComplexityThresholds = field(default_factory=ComplexityThresholds)
    smells: SmellThresholds = field(default_factory=SmellThresholds)
    coupling: CouplingThresholds = field(default_factory=CouplingThresholds)

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
        coupling_data = data.get("coupling", {})

        return cls(
            complexity=(
                ComplexityThresholds(**complexity_data)
                if complexity_data
                else ComplexityThresholds()
            ),
            smells=(
                SmellThresholds(**smells_data) if smells_data else SmellThresholds()
            ),
            coupling=(
                CouplingThresholds(**coupling_data)
                if coupling_data
                else CouplingThresholds()
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
            "coupling": {
                "efferent_low": self.coupling.efferent_low,
                "efferent_moderate": self.coupling.efferent_moderate,
                "efferent_high": self.coupling.efferent_high,
                "afferent_low": self.coupling.afferent_low,
                "afferent_moderate": self.coupling.afferent_moderate,
                "afferent_high": self.coupling.afferent_high,
                "instability_a": self.coupling.instability_a,
                "instability_b": self.coupling.instability_b,
                "instability_c": self.coupling.instability_c,
                "instability_d": self.coupling.instability_d,
                "stable_max": self.coupling.stable_max,
                "balanced_max": self.coupling.balanced_max,
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

    def get_instability_grade(self, instability: float) -> str:
        """Get instability grade based on instability value.

        Args:
            instability: Instability value (0.0-1.0)

        Returns:
            Grade from A to F
        """
        if instability <= self.coupling.instability_a:
            return "A"
        elif instability <= self.coupling.instability_b:
            return "B"
        elif instability <= self.coupling.instability_c:
            return "C"
        elif instability <= self.coupling.instability_d:
            return "D"
        else:
            return "F"

    def get_stability_category(self, instability: float) -> str:
        """Get stability category based on instability value.

        Args:
            instability: Instability value (0.0-1.0)

        Returns:
            Category: "Stable", "Balanced", or "Unstable"
        """
        if instability <= self.coupling.stable_max:
            return "Stable"
        elif instability <= self.coupling.balanced_max:
            return "Balanced"
        else:
            return "Unstable"
