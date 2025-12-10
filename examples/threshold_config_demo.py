#!/usr/bin/env python
"""Demonstration of threshold configuration system.

This script shows how to:
1. Load default thresholds from YAML
2. Create custom thresholds
3. Use thresholds with ChunkMetrics
4. Save and load custom configurations
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from mcp_vector_search.analysis.metrics import ChunkMetrics
from mcp_vector_search.config.thresholds import (
    ComplexityThresholds,
    ThresholdConfig,
)


def demo_default_thresholds():
    """Demonstrate loading and using default thresholds."""
    print("=" * 60)
    print("1. DEFAULT THRESHOLDS")
    print("=" * 60)

    # Load default configuration from package
    config_dir = Path(__file__).parent.parent / "src" / "mcp_vector_search" / "config"
    default_config_path = config_dir / "default_thresholds.yaml"

    config = ThresholdConfig.load(default_config_path)

    print("\nCognitive complexity thresholds:")
    print(f"  A grade: 0-{config.complexity.cognitive_a}")
    print(
        f"  B grade: {config.complexity.cognitive_a + 1}-{config.complexity.cognitive_b}"
    )
    print(
        f"  C grade: {config.complexity.cognitive_b + 1}-{config.complexity.cognitive_c}"
    )
    print(
        f"  D grade: {config.complexity.cognitive_c + 1}-{config.complexity.cognitive_d}"
    )
    print(f"  F grade: {config.complexity.cognitive_d + 1}+")

    print("\nCode smell thresholds:")
    print(f"  Long method: {config.smells.long_method_lines} lines")
    print(f"  Too many parameters: {config.smells.too_many_parameters}")
    print(f"  Deep nesting: {config.smells.deep_nesting_depth}")
    print(f"  High complexity: {config.smells.high_complexity}")

    # Test grade calculation
    print("\nGrade examples with default thresholds:")
    test_values = [3, 7, 15, 25, 35]
    for value in test_values:
        grade = config.get_grade(value)
        print(f"  Complexity {value:2d} -> Grade {grade}")


def demo_custom_thresholds():
    """Demonstrate creating and using custom thresholds."""
    print("\n" + "=" * 60)
    print("2. CUSTOM THRESHOLDS")
    print("=" * 60)

    # Create stricter thresholds (lower values)
    strict_config = ThresholdConfig(
        complexity=ComplexityThresholds(
            cognitive_a=3,  # More strict (default: 5)
            cognitive_b=8,  # More strict (default: 10)
            cognitive_c=15,  # More strict (default: 20)
            cognitive_d=25,  # More strict (default: 30)
        )
    )

    print("\nStrict cognitive complexity thresholds:")
    print(f"  A grade: 0-{strict_config.complexity.cognitive_a}")
    print(
        f"  B grade: {strict_config.complexity.cognitive_a + 1}-{strict_config.complexity.cognitive_b}"
    )
    print(
        f"  C grade: {strict_config.complexity.cognitive_b + 1}-{strict_config.complexity.cognitive_c}"
    )
    print(
        f"  D grade: {strict_config.complexity.cognitive_c + 1}-{strict_config.complexity.cognitive_d}"
    )
    print(f"  F grade: {strict_config.complexity.cognitive_d + 1}+")

    # Compare grades with default vs strict
    print("\nGrade comparison (default vs strict):")
    test_values = [3, 7, 15, 25, 35]
    default_config = ThresholdConfig()

    for value in test_values:
        default_grade = default_config.get_grade(value)
        strict_grade = strict_config.get_grade(value)
        print(
            f"  Complexity {value:2d}: {default_grade} (default) vs {strict_grade} (strict)"
        )


def demo_integration_with_metrics():
    """Demonstrate integration with ChunkMetrics."""
    print("\n" + "=" * 60)
    print("3. INTEGRATION WITH CHUNK METRICS")
    print("=" * 60)

    # Create custom thresholds
    custom_config = ThresholdConfig(
        complexity=ComplexityThresholds(
            cognitive_a=10,  # More lenient (default: 5)
            cognitive_b=20,  # More lenient (default: 10)
            cognitive_c=30,  # More lenient (default: 20)
            cognitive_d=40,  # More lenient (default: 30)
        )
    )

    # Create chunks with different complexity levels
    chunks = [
        ChunkMetrics(cognitive_complexity=5, cyclomatic_complexity=3),
        ChunkMetrics(cognitive_complexity=15, cyclomatic_complexity=8),
        ChunkMetrics(cognitive_complexity=25, cyclomatic_complexity=12),
        ChunkMetrics(cognitive_complexity=35, cyclomatic_complexity=18),
    ]

    print("\nChunk grading with default vs custom thresholds:")
    print(f"{'Complexity':<12} {'Default':<10} {'Custom (Lenient)':<20}")
    print("-" * 45)

    for chunk in chunks:
        default_grade = chunk._compute_grade()
        custom_grade = chunk._compute_grade(custom_config)
        print(
            f"{chunk.cognitive_complexity:<12} {default_grade:<10} {custom_grade:<20}"
        )


def demo_save_and_load():
    """Demonstrate saving and loading custom configurations."""
    print("\n" + "=" * 60)
    print("4. SAVING AND LOADING CONFIGURATIONS")
    print("=" * 60)

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "my_thresholds.yaml"

        # Create custom configuration
        custom_config = ThresholdConfig(
            complexity=ComplexityThresholds(
                cognitive_a=4,
                cognitive_b=9,
                cyclomatic_low=3,
            ),
            fail_on_f_grade=True,
            fail_on_smell_count=5,
            warn_on_d_grade=True,
        )

        # Save configuration
        custom_config.save(config_path)
        print(f"\nSaved custom configuration to: {config_path}")

        # Load configuration
        loaded_config = ThresholdConfig.load(config_path)
        print("Loaded configuration successfully")

        # Verify values
        print("\nVerifying loaded values:")
        print(
            f"  Cognitive A threshold: {loaded_config.complexity.cognitive_a} (expected: 4)"
        )
        print(
            f"  Cognitive B threshold: {loaded_config.complexity.cognitive_b} (expected: 9)"
        )
        print(
            f"  Cyclomatic low: {loaded_config.complexity.cyclomatic_low} (expected: 3)"
        )
        print(f"  Fail on F grade: {loaded_config.fail_on_f_grade} (expected: True)")

        # Show YAML content
        print("\nYAML file content:")
        print("-" * 60)
        print(config_path.read_text())


def main():
    """Run all demonstrations."""
    demo_default_thresholds()
    demo_custom_thresholds()
    demo_integration_with_metrics()
    demo_save_and_load()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print(
        "  1. ThresholdConfig provides flexible configuration for code quality metrics"
    )
    print("  2. Default thresholds can be loaded from YAML file")
    print("  3. Custom thresholds can be created for stricter or more lenient grading")
    print("  4. Thresholds integrate seamlessly with ChunkMetrics grade calculation")
    print("  5. Configurations can be saved/loaded for project-specific standards")


if __name__ == "__main__":
    main()
