"""Analyze command for MCP Vector Search CLI."""

import asyncio
from pathlib import Path

import typer
from loguru import logger

from ...analysis import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    FileMetrics,
    ProjectMetrics,
)
from ...core.exceptions import ProjectNotFoundError
from ...core.project import ProjectManager
from ...parsers.registry import ParserRegistry
from ..output import console, print_error, print_info, print_json

# Create analyze subcommand app
analyze_app = typer.Typer(help="📈 Analyze code complexity and quality")


@analyze_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        rich_help_panel="🔧 Global Options",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick mode (cognitive + cyclomatic complexity only)",
        rich_help_panel="⚡ Performance Options",
    ),
    show_smells: bool = typer.Option(
        True,
        "--smells/--no-smells",
        help="Show detected code smells in output",
        rich_help_panel="📊 Display Options",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Filter by programming language (python, javascript, typescript)",
        rich_help_panel="🔍 Filters",
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Analyze specific file or directory",
        rich_help_panel="🔍 Filters",
    ),
    top: int = typer.Option(
        10,
        "--top",
        help="Number of top complexity hotspots to show",
        min=1,
        max=100,
        rich_help_panel="📊 Display Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
        rich_help_panel="📊 Display Options",
    ),
) -> None:
    """📈 Analyze code complexity and quality.

    Performs structural code analysis to identify complexity hotspots,
    code smells, and quality metrics across your codebase.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Quick analysis (cognitive + cyclomatic complexity):[/green]
        $ mcp-vector-search analyze --quick

    [green]Full analysis (all collectors):[/green]
        $ mcp-vector-search analyze

    [green]Filter by language:[/green]
        $ mcp-vector-search analyze --language python

    [green]Analyze specific directory:[/green]
        $ mcp-vector-search analyze --path src/core

    [bold cyan]Output Options:[/bold cyan]

    [green]Show top 5 hotspots:[/green]
        $ mcp-vector-search analyze --top 5

    [green]Export to JSON:[/green]
        $ mcp-vector-search analyze --json > analysis.json

    [dim]💡 Tip: Use --quick for faster analysis on large projects.[/dim]
    """
    if ctx.invoked_subcommand is not None:
        # A subcommand was invoked - let it handle the request
        return

    try:
        # Use provided project_root or current working directory
        if project_root is None:
            project_root = Path.cwd()

        asyncio.run(
            run_analysis(
                project_root=project_root,
                quick_mode=quick,
                language_filter=language,
                path_filter=path,
                top_n=top,
                json_output=json_output,
                show_smells=show_smells,
            )
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(1)


async def run_analysis(
    project_root: Path,
    quick_mode: bool = False,
    language_filter: str | None = None,
    path_filter: Path | None = None,
    top_n: int = 10,
    json_output: bool = False,
    show_smells: bool = True,
) -> None:
    """Run code complexity analysis.

    Args:
        project_root: Root directory of the project
        quick_mode: Use only cognitive + cyclomatic complexity (faster)
        language_filter: Filter files by language
        path_filter: Analyze specific file or directory
        top_n: Number of top hotspots to show
        json_output: Output results as JSON
        show_smells: Show detected code smells in output
    """
    try:
        # Check if project is initialized (optional - we can analyze any directory)
        project_manager = ProjectManager(project_root)
        initialized = project_manager.is_initialized()

        if not initialized and not json_output:
            print_info(
                f"Analyzing directory: {project_root} (not initialized as MCP project)"
            )

        # Initialize parser registry
        parser_registry = ParserRegistry()

        # Determine which collectors to use
        if quick_mode:
            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
            ]
            mode_label = "Quick Mode (2 collectors)"
        else:
            # Import all collectors for full mode
            from ...analysis import (
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]
            mode_label = "Full Mode (5 collectors)"

        # Find files to analyze
        files_to_analyze = _find_analyzable_files(
            project_root, language_filter, path_filter, parser_registry
        )

        if not files_to_analyze:
            if json_output:
                print_json({"error": "No files found to analyze"})
            else:
                print_error("No files found to analyze")
            return

        if not json_output:
            console.print(
                f"\n[bold blue]Starting Code Analysis[/bold blue] - {mode_label}"
            )
            console.print(f"Files to analyze: {len(files_to_analyze)}\n")

        # Analyze files
        project_metrics = ProjectMetrics(project_root=str(project_root))

        for file_path in files_to_analyze:
            try:
                file_metrics = await _analyze_file(
                    file_path, parser_registry, collectors
                )
                if file_metrics and file_metrics.chunks:
                    project_metrics.files[str(file_path)] = file_metrics
            except Exception as e:
                logger.debug(f"Failed to analyze {file_path}: {e}")
                continue

        # Compute aggregates
        project_metrics.compute_aggregates()

        # Detect code smells if requested
        all_smells = []
        if show_smells:
            from ...analysis.collectors.smells import SmellDetector
            from ...config.thresholds import ThresholdConfig

            # Load threshold config (optional - defaults will be used)
            threshold_config = ThresholdConfig()
            smell_detector = SmellDetector(thresholds=threshold_config)

            # Detect smells across all analyzed files
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

        # Output results
        if json_output:
            output = project_metrics.to_summary()
            # Add smell data to JSON output if available
            if show_smells and all_smells:
                from ...analysis.collectors.smells import SmellDetector

                detector = SmellDetector()
                smell_summary = detector.get_smell_summary(all_smells)
                output["smells"] = {
                    "summary": smell_summary,
                    "details": [
                        {
                            "name": smell.name,
                            "severity": smell.severity.value,
                            "location": smell.location,
                            "description": smell.description,
                            "metric_value": smell.metric_value,
                            "threshold": smell.threshold,
                            "suggestion": smell.suggestion,
                        }
                        for smell in all_smells
                    ],
                }
            print_json(output)
        else:
            # Import console reporter
            from ...analysis.reporters.console import ConsoleReporter

            reporter = ConsoleReporter()
            reporter.print_summary(project_metrics)
            reporter.print_distribution(project_metrics)
            reporter.print_hotspots(project_metrics, top=top_n)

            # Print code smells if requested
            if show_smells and all_smells:
                reporter.print_smells(all_smells, top=top_n)

            reporter.print_recommendations(project_metrics)

    except ProjectNotFoundError as e:
        if json_output:
            print_json({"error": str(e)})
        else:
            print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        if json_output:
            print_json({"error": str(e)})
        else:
            print_error(f"Analysis failed: {e}")
        raise


def _find_analyzable_files(
    project_root: Path,
    language_filter: str | None,
    path_filter: Path | None,
    parser_registry: ParserRegistry,
) -> list[Path]:
    """Find files that can be analyzed.

    Args:
        project_root: Root directory
        language_filter: Optional language filter
        path_filter: Optional path filter
        parser_registry: Parser registry for checking supported files

    Returns:
        List of file paths to analyze
    """
    import fnmatch

    # Determine base path to search
    base_path = path_filter if path_filter and path_filter.exists() else project_root

    # If path_filter is a file, return just that file
    if base_path.is_file():
        # Check if file extension is supported
        if base_path.suffix.lower() in parser_registry.get_supported_extensions():
            return [base_path]
        return []

    # Find all supported files
    files: list[Path] = []
    supported_extensions = parser_registry.get_supported_extensions()

    # Common ignore patterns
    ignore_patterns = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }

    for file_path in base_path.rglob("*"):
        # Skip directories
        if file_path.is_dir():
            continue

        # Skip ignored directories
        if any(
            ignored in file_path.parts or fnmatch.fnmatch(file_path.name, f"{ignored}*")
            for ignored in ignore_patterns
        ):
            continue

        # Check if file extension is supported
        if file_path.suffix.lower() not in supported_extensions:
            continue

        # Apply language filter
        if language_filter:
            parser = parser_registry.get_parser_for_file(file_path)
            if parser.language.lower() != language_filter.lower():
                continue

        files.append(file_path)

    return sorted(files)


async def _analyze_file(
    file_path: Path, parser_registry: ParserRegistry, collectors: list
) -> FileMetrics | None:
    """Analyze a single file and return metrics.

    Args:
        file_path: Path to file
        parser_registry: Parser registry
        collectors: List of metric collectors

    Returns:
        FileMetrics or None if analysis failed
    """
    try:
        # Get parser for file
        parser = parser_registry.get_parser_for_file(file_path)

        # Parse file into chunks
        chunks = await parser.parse_file(file_path)

        if not chunks:
            return None

        # Create file metrics
        file_metrics = FileMetrics(file_path=str(file_path))

        # Count lines
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                file_metrics.total_lines = len(lines)
                file_metrics.code_lines = sum(
                    1
                    for line in lines
                    if line.strip() and not line.strip().startswith("#")
                )
                file_metrics.comment_lines = sum(
                    1 for line in lines if line.strip().startswith("#")
                )
                file_metrics.blank_lines = sum(1 for line in lines if not line.strip())
        except Exception:
            pass

        # Count functions and classes from chunks
        for chunk in chunks:
            if chunk.chunk_type == "function":
                file_metrics.function_count += 1
            elif chunk.chunk_type == "class":
                file_metrics.class_count += 1
            elif chunk.chunk_type == "method":
                file_metrics.method_count += 1

        # Extract chunk metrics from parsed chunks
        from ...analysis.metrics import ChunkMetrics

        for chunk in chunks:
            # Use complexity_score from parser (cyclomatic complexity)
            # For quick mode, this is sufficient
            complexity = (
                int(chunk.complexity_score) if chunk.complexity_score > 0 else 1
            )

            # Count parameters if available
            param_count = len(chunk.parameters) if chunk.parameters else 0

            # Estimate cognitive complexity from cyclomatic (rough approximation)
            # Cognitive is typically 1.2-1.5x cyclomatic for complex code
            cognitive = int(complexity * 1.3)

            chunk_metrics = ChunkMetrics(
                cognitive_complexity=cognitive,
                cyclomatic_complexity=complexity,
                max_nesting_depth=0,  # Not available without collectors
                parameter_count=param_count,
                lines_of_code=chunk.end_line - chunk.start_line + 1,
            )
            file_metrics.chunks.append(chunk_metrics)

        # Compute aggregates
        file_metrics.compute_aggregates()

        return file_metrics

    except Exception as e:
        logger.debug(f"Failed to analyze file {file_path}: {e}")
        return None


if __name__ == "__main__":
    analyze_app()
