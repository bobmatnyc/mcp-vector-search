#!/usr/bin/env python3
"""Example: Export structural code analysis to JSON.

This example demonstrates how to use the JSONExporter to export
analysis results to the standardized JSON format for visualization
and external tool integration.
"""

from datetime import datetime
from pathlib import Path

from mcp_vector_search.analysis.metrics import (
    ChunkMetrics,
    CouplingMetrics,
    FileMetrics,
    ProjectMetrics,
)
from mcp_vector_search.analysis.visualizer import JSONExporter


def create_sample_project_metrics() -> ProjectMetrics:
    """Create sample project metrics for demonstration."""
    # Create sample chunk metrics (function metrics)
    chunk1 = ChunkMetrics(
        cognitive_complexity=8,
        cyclomatic_complexity=5,
        max_nesting_depth=3,
        parameter_count=2,
        lines_of_code=25,
        smells=["deep_nesting"],
        halstead_volume=150.5,
        halstead_difficulty=10.2,
    )

    chunk2 = ChunkMetrics(
        cognitive_complexity=15,
        cyclomatic_complexity=10,
        max_nesting_depth=4,
        parameter_count=5,
        lines_of_code=50,
        smells=["long_method", "too_many_parameters"],
    )

    # Create coupling metrics
    coupling = CouplingMetrics(
        efferent_coupling=3,
        afferent_coupling=2,
        imports=["module_a", "module_b", "utils"],
        internal_imports=["utils"],
        external_imports=["module_a", "module_b"],
    )

    # Create file metrics
    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=75,
        comment_lines=15,
        blank_lines=10,
        function_count=2,
        class_count=0,
        chunks=[chunk1, chunk2],
        coupling=coupling,
    )
    file_metrics.compute_aggregates()

    # Create project metrics
    project = ProjectMetrics(
        project_root=str(Path.cwd()),
        analyzed_at=datetime.now(),
    )
    project.files["src/example.py"] = file_metrics
    project.compute_aggregates()

    return project


def main():
    """Main execution function."""
    # Create sample metrics
    print("Creating sample project metrics...")
    project_metrics = create_sample_project_metrics()

    # Initialize exporter
    print("Initializing JSON exporter...")
    exporter = JSONExporter(project_root=Path.cwd())

    # Export to JSON object
    print("\nExporting to JSON schema...")
    export = exporter.export(
        project_metrics,
        include_trends=False,  # No trend data in this example
        include_dependencies=True,
    )

    # Display summary
    print("\n=== Export Summary ===")
    print(f"Schema Version: {export.metadata.version}")
    print(f"Tool Version: {export.metadata.tool_version}")
    print(f"Generated At: {export.metadata.generated_at}")
    print(f"\nTotal Files: {export.summary.total_files}")
    print(f"Total Functions: {export.summary.total_functions}")
    print(f"Total Lines: {export.summary.total_lines}")
    print(f"Average Complexity: {export.summary.avg_complexity:.2f}")
    print(f"Total Smells: {export.summary.total_smells}")

    # Export to file
    output_path = Path("analysis-export.json")
    print(f"\nExporting to file: {output_path}")
    result_path = exporter.export_to_file(
        project_metrics, output_path, indent=2, include_dependencies=True
    )

    print("\n✓ Export completed successfully!")
    print(f"  File: {result_path}")
    print(f"  Size: {result_path.stat().st_size} bytes")

    # Show a sample of the JSON
    json_str = export.model_dump_json(indent=2)
    print("\n=== Sample JSON Output (first 500 chars) ===")
    print(json_str[:500])
    print("...\n")


if __name__ == "__main__":
    main()
