#!/usr/bin/env python3
"""Example script demonstrating HTML report generation.

This script shows how to:
1. Collect code metrics from a project
2. Export to the JSON schema format
3. Generate a standalone HTML report
4. View the report in a browser

Usage:
    # Analyze current directory
    python examples/generate_html_report.py

    # Analyze specific project
    python examples/generate_html_report.py --project /path/to/project

    # Custom output location
    python examples/generate_html_report.py --output reports/my-report.html

    # Open in browser after generation
    python examples/generate_html_report.py --open
"""

import argparse
import webbrowser
from pathlib import Path

from loguru import logger

# Import analysis components
from mcp_vector_search.analysis import ProjectMetrics
from mcp_vector_search.analysis.visualizer import HTMLReportGenerator, JSONExporter


def collect_metrics(project_root: Path) -> ProjectMetrics:
    """Collect metrics for a project.

    Args:
        project_root: Root directory of project to analyze

    Returns:
        Complete project metrics
    """
    logger.info(f"Analyzing project: {project_root}")

    # Note: In production, use the full analysis pipeline with collectors:
    # complexity_collector = CognitiveComplexityCollector()
    # coupling_collector = CouplingCollector()
    # smell_detector = SmellDetector()
    # This example uses simplified mock data for demonstration.

    # Create project metrics container
    project_metrics = ProjectMetrics()

    # Analyze Python files
    python_files = list(project_root.rglob("*.py"))
    logger.info(f"Found {len(python_files)} Python files")

    for file_path in python_files:
        # Skip virtual environments and build artifacts
        if any(
            part in file_path.parts
            for part in [".venv", "venv", "env", ".tox", "build", "dist", "__pycache__"]
        ):
            continue

        try:
            # Collect metrics for file
            logger.debug(f"Analyzing {file_path}")

            # Note: This is a simplified example - real usage would integrate with
            # the full analysis pipeline. For now, we create minimal FileMetrics.
            # In production, use:
            # file_metrics = complexity_collector.analyze_file(file_path)

            logger.debug(f"Collected metrics for {file_path}")

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")

    logger.info(f"Analysis complete: {len(project_metrics.files)} files analyzed")
    return project_metrics


def generate_report(
    project_root: Path, output_path: Path, open_browser: bool = False
) -> Path:
    """Generate HTML report for a project.

    Args:
        project_root: Root directory of project
        output_path: Where to write the HTML report
        open_browser: Whether to open report in browser

    Returns:
        Path to generated HTML report
    """
    # Collect metrics
    project_metrics = collect_metrics(project_root)

    # Export to JSON schema format
    logger.info("Exporting to JSON schema format...")
    exporter = JSONExporter(project_root=project_root)
    export = exporter.export(project_metrics)

    # Generate HTML report
    logger.info("Generating HTML report...")
    html_generator = HTMLReportGenerator(title=f"{project_root.name} - Code Analysis")
    report_path = html_generator.generate_to_file(export, output_path)

    logger.success(f"HTML report generated: {report_path}")

    # Open in browser if requested
    if open_browser:
        logger.info("Opening report in browser...")
        webbrowser.open(f"file://{report_path.absolute()}")

    return report_path


def main():
    """Main entry point for example script."""
    parser = argparse.ArgumentParser(
        description="Generate HTML code analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root directory to analyze (default: current directory)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "code-analysis-report.html",
        help="Output path for HTML report (default: ./code-analysis-report.html)",
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open report in browser after generation",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    else:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Validate project directory
    if not args.project.exists():
        logger.error(f"Project directory not found: {args.project}")
        return 1

    if not args.project.is_dir():
        logger.error(f"Project path is not a directory: {args.project}")
        return 1

    # Generate report
    try:
        generate_report(
            project_root=args.project,
            output_path=args.output,
            open_browser=args.open,
        )
        return 0

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        if args.verbose:
            logger.exception(e)
        return 1


if __name__ == "__main__":
    exit(main())
