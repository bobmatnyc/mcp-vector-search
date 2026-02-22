"""Analyze command for MCP Vector Search CLI."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from ...analysis import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    FileMetrics,
    ProjectMetrics,
)
from ...analysis.baseline import (
    BaselineComparator,
    BaselineExistsError,
    BaselineManager,
    BaselineNotFoundError,
)
from ...analysis.storage.metrics_store import MetricsStore, MetricsStoreError
from ...analysis.storage.trend_tracker import TrendData, TrendDirection, TrendTracker
from ...core.exceptions import ProjectNotFoundError
from ...core.git import GitError, GitManager, GitNotAvailableError, GitNotRepoError
from ...core.project import ProjectManager
from ...parsers.registry import ParserRegistry
from ..output import console, print_error, print_info, print_json

# Create analyze subcommand app
analyze_app = typer.Typer(help="📈 Analyze code complexity and quality")


# Main callback - runs both analyses when no subcommand specified
@analyze_app.callback(invoke_without_command=True)
def analyze_callback(
    ctx: typer.Context,
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick mode (cognitive + cyclomatic complexity only, skip dead-code)",
        rich_help_panel="⚡ Performance Options",
    ),
) -> None:
    """Analyze code complexity and quality.

    When called without a subcommand, runs both complexity and dead-code analysis.

    Available commands:
      complexity - Analyze code complexity (cyclomatic, cognitive, smells)
      dead-code  - Detect dead/unreachable code
    """
    if ctx.invoked_subcommand is None:
        # No subcommand - run both analyses
        from ..output import console

        mode = "quick" if quick else "full"
        console.print(f"[bold blue]Running {mode} analysis[/bold blue]\n")

        # Run complexity analysis directly using asyncio.run instead of ctx.invoke
        # to avoid Typer's option handling issues with boolean defaults
        project_root = Path.cwd()
        asyncio.run(
            run_analysis(
                project_root=project_root,
                quick_mode=quick,
                language_filter=None,
                path_filter=None,
                top_n=10,
                json_output=False,
                show_smells=True,
                output_format="console",
                output_file=None,
                fail_on_smell=False,
                severity_threshold="error",
                changed_only=False,
                baseline=None,
                save_baseline=None,
                compare_baseline=None,
                force_baseline=False,
                baseline_manager=BaselineManager(),
                include_context=False,
            )
        )

        # Skip dead code analysis in quick mode (it's slow)
        if not quick:
            console.print("\n" + "─" * 60 + "\n")

            # Run dead code analysis directly using asyncio.run
            asyncio.run(
                run_dead_code_analysis(
                    project_root=project_root,
                    custom_entry_points=[],
                    include_public=False,
                    min_confidence="low",
                    exclude_patterns=None,
                    output_format="console",
                    output_file=None,
                    fail_on_dead=False,
                )
            )


@analyze_app.command(name="complexity")
def complexity_analysis(
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
    include_context: bool = typer.Option(
        False,
        "--include-context",
        help="Include LLM-consumable context in JSON output (enhanced interpretation)",
        rich_help_panel="📊 Display Options",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: console, json, sarif, markdown",
        rich_help_panel="📊 Display Options",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (required for sarif format)",
        rich_help_panel="📊 Display Options",
    ),
    fail_on_smell: bool = typer.Option(
        False,
        "--fail-on-smell",
        help="Exit with code 1 if code smells are detected",
        rich_help_panel="🚦 Quality Gates",
    ),
    severity_threshold: str = typer.Option(
        "error",
        "--severity-threshold",
        help="Minimum severity to trigger failure: info, warning, error, none",
        rich_help_panel="🚦 Quality Gates",
    ),
    changed_only: bool = typer.Option(
        False,
        "--changed-only/--no-changed-only",
        help="Analyze only uncommitted changes (staged + unstaged + untracked)",
        rich_help_panel="🔍 Filters",
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Compare against baseline branch (e.g., main, master, develop)",
        rich_help_panel="🔍 Filters",
    ),
    save_baseline: str | None = typer.Option(
        None,
        "--save-baseline",
        help="Save current analysis as named baseline",
        rich_help_panel="📊 Baseline Management",
    ),
    compare_baseline: str | None = typer.Option(
        None,
        "--compare-baseline",
        help="Compare current analysis against named baseline",
        rich_help_panel="📊 Baseline Management",
    ),
    list_baselines: bool = typer.Option(
        False,
        "--list-baselines",
        help="List all available baselines (standalone action)",
        rich_help_panel="📊 Baseline Management",
    ),
    delete_baseline: str | None = typer.Option(
        None,
        "--delete-baseline",
        help="Delete a named baseline",
        rich_help_panel="📊 Baseline Management",
    ),
    force_baseline: bool = typer.Option(
        False,
        "--force",
        help="Force overwrite when saving baseline that already exists",
        rich_help_panel="📊 Baseline Management",
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

    [green]Analyze only uncommitted changes:[/green]
        $ mcp-vector-search analyze --changed-only

    [green]Compare against baseline branch:[/green]
        $ mcp-vector-search analyze --baseline main

    [bold cyan]Output Options:[/bold cyan]

    [green]Show top 5 hotspots:[/green]
        $ mcp-vector-search analyze --top 5

    [green]Export to JSON:[/green]
        $ mcp-vector-search analyze --json > analysis.json

    [green]Export to SARIF format:[/green]
        $ mcp-vector-search analyze --format sarif --output report.sarif

    [green]Export to Markdown format:[/green]
        $ mcp-vector-search analyze --format markdown --output .

    [bold cyan]CI/CD Quality Gates:[/bold cyan]

    [green]Fail on ERROR-level smells (default):[/green]
        $ mcp-vector-search analyze --fail-on-smell

    [green]Fail on WARNING or ERROR smells:[/green]
        $ mcp-vector-search analyze --fail-on-smell --severity-threshold warning

    [green]CI/CD workflow with SARIF:[/green]
        $ mcp-vector-search analyze --fail-on-smell --format sarif --output report.sarif

    [dim]💡 Tip: Use --quick for faster analysis on large projects.[/dim]
    """
    # Handle standalone baseline operations first
    baseline_manager = BaselineManager()

    # List baselines (standalone action)
    if list_baselines:
        baselines = baseline_manager.list_baselines()
        if not baselines:
            console.print("[yellow]No baselines found[/yellow]")
            console.print(
                f"\nBaselines are stored in: {baseline_manager.storage_dir}\n"
            )
            console.print(
                "Create a baseline with: [cyan]mcp-vector-search analyze --save-baseline <name>[/cyan]"
            )
        else:
            console.print(f"\n[bold]Available Baselines[/bold] ({len(baselines)})")
            console.print("━" * 80)
            for baseline in baselines:
                console.print(f"\n[cyan]• {baseline.baseline_name}[/cyan]")
                console.print(f"  Created: {baseline.created_at}")
                console.print(f"  Project: {baseline.project_path}")
                console.print(
                    f"  Files: {baseline.file_count} | Functions: {baseline.function_count}"
                )
                console.print(f"  Tool Version: {baseline.tool_version}")
                if baseline.git_info.commit:
                    console.print(
                        f"  Git: {baseline.git_info.branch or 'detached'} @ {baseline.git_info.commit[:8]}"
                    )
            console.print()
        raise typer.Exit(0)

    # Delete baseline (standalone action)
    if delete_baseline:
        try:
            baseline_manager.delete_baseline(delete_baseline)
            console.print(
                f"[green]✓[/green] Deleted baseline: [cyan]{delete_baseline}[/cyan]"
            )
            raise typer.Exit(0)
        except BaselineNotFoundError as e:
            print_error(str(e))
            console.print("\nAvailable baselines:")
            baselines = baseline_manager.list_baselines()
            for baseline in baselines[:5]:
                console.print(f"  • {baseline.baseline_name}")
            raise typer.Exit(1)

    try:
        # Validate format and output options
        valid_formats = ["console", "json", "sarif", "markdown"]
        format_lower = format.lower()

        if format_lower not in valid_formats:
            print_error(
                f"Invalid format: {format}. Must be one of: {', '.join(valid_formats)}"
            )
            raise typer.Exit(1)

        # SARIF and markdown formats should have output path (defaults to current dir)
        if format_lower == "sarif" and output is None:
            print_error("--output is required when using --format sarif")
            raise typer.Exit(1)

        # JSON flag overrides format for backward compatibility
        if json_output:
            format_lower = "json"

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
                json_output=(format_lower == "json"),
                show_smells=show_smells,
                output_format=format_lower,
                output_file=output,
                fail_on_smell=fail_on_smell,
                severity_threshold=severity_threshold,
                changed_only=changed_only,
                baseline=baseline,
                save_baseline=save_baseline,
                compare_baseline=compare_baseline,
                force_baseline=force_baseline,
                baseline_manager=baseline_manager,
                include_context=include_context,
            )
        )

    except typer.Exit:
        # Re-raise typer.Exit to preserve exit codes from run_analysis
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(2)  # Exit code 2 for analysis errors


def filter_smells_by_severity(smells: list, severity_threshold: str) -> list:
    """Filter smells by minimum severity threshold.

    Args:
        smells: List of CodeSmell objects to filter
        severity_threshold: Minimum severity level - "info", "warning", "error", or "none"

    Returns:
        Filtered list of smells matching or exceeding the severity threshold
    """
    from ...analysis.collectors.smells import SmellSeverity

    if severity_threshold.lower() == "none":
        return []

    severity_levels = {
        "info": [SmellSeverity.INFO, SmellSeverity.WARNING, SmellSeverity.ERROR],
        "warning": [SmellSeverity.WARNING, SmellSeverity.ERROR],
        "error": [SmellSeverity.ERROR],
    }

    allowed = severity_levels.get(severity_threshold.lower(), [SmellSeverity.ERROR])
    return [s for s in smells if s.severity in allowed]


async def run_analysis(
    project_root: Path,
    quick_mode: bool = False,
    language_filter: str | None = None,
    path_filter: Path | None = None,
    top_n: int = 10,
    json_output: bool = False,
    show_smells: bool = True,
    output_format: str = "console",
    output_file: Path | None = None,
    fail_on_smell: bool = False,
    severity_threshold: str = "error",
    changed_only: bool = False,
    baseline: str | None = None,
    save_baseline: str | None = None,
    compare_baseline: str | None = None,
    force_baseline: bool = False,
    baseline_manager: BaselineManager | None = None,
    include_context: bool = False,
) -> None:
    """Run code complexity analysis.

    Args:
        project_root: Root directory of the project
        quick_mode: Use only cognitive + cyclomatic complexity (faster)
        language_filter: Filter files by language
        path_filter: Analyze specific file or directory
        top_n: Number of top hotspots to show
        json_output: Output results as JSON (deprecated, use output_format)
        show_smells: Show detected code smells in output
        output_format: Output format (console, json, sarif)
        output_file: Output file path (for sarif format)
        fail_on_smell: Exit with code 1 if smells are detected
        severity_threshold: Minimum severity to trigger failure
        changed_only: Analyze only uncommitted changes
        baseline: Compare against baseline branch
        save_baseline: Save analysis as named baseline
        compare_baseline: Compare against named baseline
        force_baseline: Force overwrite existing baseline
        baseline_manager: BaselineManager instance
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

        # Initialize git manager if needed for changed/baseline filtering
        git_manager = None
        git_changed_files = None

        if changed_only or baseline:
            try:
                git_manager = GitManager(project_root)

                # Get changed files based on mode
                if changed_only:
                    git_changed_files = git_manager.get_changed_files(
                        include_untracked=True
                    )
                    if not git_changed_files:
                        if json_output:
                            print_json(
                                {"error": "No changed files found. Nothing to analyze."}
                            )
                        else:
                            print_info("No changed files found. Nothing to analyze.")
                        return
                elif baseline:
                    git_changed_files = git_manager.get_diff_files(baseline)
                    if not git_changed_files:
                        if json_output:
                            print_json(
                                {"error": f"No files changed vs baseline '{baseline}'."}
                            )
                        else:
                            print_info(f"No files changed vs baseline '{baseline}'.")
                        return

            except GitNotAvailableError as e:
                if json_output:
                    print_json({"warning": str(e), "fallback": "full analysis"})
                else:
                    console.print(f"[yellow]⚠️  {e}[/yellow]")
                    print_info("Proceeding with full codebase analysis...")
                git_manager = None
                git_changed_files = None

            except GitNotRepoError as e:
                if json_output:
                    print_json({"warning": str(e), "fallback": "full analysis"})
                else:
                    console.print(f"[yellow]⚠️  {e}[/yellow]")
                    print_info("Proceeding with full codebase analysis...")
                git_manager = None
                git_changed_files = None

            except GitError as e:
                if json_output:
                    print_json(
                        {"warning": f"Git error: {e}", "fallback": "full analysis"}
                    )
                else:
                    console.print(f"[yellow]⚠️  Git error: {e}[/yellow]")
                    print_info("Proceeding with full codebase analysis...")
                git_manager = None
                git_changed_files = None

        # Find files to analyze
        files_to_analyze = _find_analyzable_files(
            project_root,
            language_filter,
            path_filter,
            parser_registry,
            git_changed_files,
        )

        if not files_to_analyze:
            if json_output:
                print_json({"error": "No files found to analyze"})
            else:
                print_error("No files found to analyze")
            return

        # Display analysis info
        if not json_output:
            console.print(
                f"\n[bold blue]Starting Code Analysis[/bold blue] - {mode_label}"
            )

            # Show file count information with git filtering context
            if git_changed_files is not None:
                # Get total files for context
                total_files = len(
                    _find_analyzable_files(
                        project_root,
                        language_filter,
                        path_filter,
                        parser_registry,
                        None,
                    )
                )
                filter_type = "changed" if changed_only else f"vs {baseline}"
                console.print(
                    f"Analyzing {len(files_to_analyze)} {filter_type} files "
                    f"({total_files} total in project)\n"
                )
            else:
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

        # Save snapshot to metrics store for historical tracking
        trend_data: TrendData | None = None
        try:
            metrics_db_path = project_root / ".mcp-vector-search" / "metrics.db"
            metrics_store = MetricsStore(metrics_db_path)
            snapshot_id = metrics_store.save_project_snapshot(project_metrics)
            logger.debug(f"Saved metrics snapshot {snapshot_id}")

            # Check for historical data and compute trends if available
            trend_tracker = TrendTracker(metrics_store)
            trend_data = trend_tracker.get_trends(project_root, days=30)

            # Only show trends if we have at least 2 snapshots
            if len(trend_data.snapshots) >= 2 and not json_output:
                _print_trends(trend_data)

        except MetricsStoreError as e:
            logger.debug(f"Could not save metrics snapshot: {e}")
        except Exception as e:
            logger.debug(f"Trend tracking unavailable: {e}")

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

        # Output results based on format
        if output_format == "markdown":
            # Markdown format - write two files
            from ...analysis.reporters.markdown import MarkdownReporter

            reporter = MarkdownReporter()

            # Generate full analysis report
            analysis_file = reporter.generate_analysis_report(
                project_metrics, all_smells, output_file
            )
            console.print(
                f"[green]✓[/green] Analysis report written to: {analysis_file}"
            )

            # Generate fixes report if smells were detected
            if all_smells:
                fixes_file = reporter.generate_fixes_report(
                    project_metrics, all_smells, output_file
                )
                console.print(f"[green]✓[/green] Fixes report written to: {fixes_file}")

        elif output_format == "sarif":
            # SARIF format - write to file
            from ...analysis.reporters.sarif import SARIFReporter

            if not all_smells:
                print_error(
                    "No code smells detected - SARIF report requires smells to report"
                )
                return

            reporter = SARIFReporter()
            reporter.write_sarif(all_smells, output_file, base_path=project_root)
            console.print(f"[green]✓[/green] SARIF report written to: {output_file}")

        elif json_output or output_format == "json":
            # JSON format - with optional LLM context
            if include_context:
                # Enhanced JSON export with LLM-consumable context
                from ...analysis.interpretation import EnhancedJSONExporter
                from ...config.thresholds import ThresholdConfig

                threshold_config = ThresholdConfig()
                exporter = EnhancedJSONExporter(
                    project_root=project_root, threshold_config=threshold_config
                )
                enhanced_export = exporter.export_with_context(
                    project_metrics, include_smells=show_smells
                )
                # Output as JSON
                import json

                print_json(json.loads(enhanced_export.model_dump_json()))
            else:
                # Standard JSON format
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
            # Console format (default)
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

        # Handle baseline operations after analysis
        if baseline_manager:
            # Save baseline if requested
            if save_baseline:
                try:
                    baseline_path = baseline_manager.save_baseline(
                        baseline_name=save_baseline,
                        metrics=project_metrics,
                        overwrite=force_baseline,
                    )
                    if not json_output:
                        console.print(
                            f"\n[green]✓[/green] Saved baseline: [cyan]{save_baseline}[/cyan]"
                        )
                        console.print(f"  Location: {baseline_path}")
                except BaselineExistsError as e:
                    if json_output:
                        print_json({"error": str(e)})
                    else:
                        print_error(str(e))
                        console.print(
                            "\nUse [cyan]--force[/cyan] to overwrite the existing baseline"
                        )
                    raise typer.Exit(1)

            # Compare against baseline if requested
            if compare_baseline:
                try:
                    baseline_metrics = baseline_manager.load_baseline(compare_baseline)
                    comparator = BaselineComparator()
                    comparison_result = comparator.compare(
                        current=project_metrics,
                        baseline=baseline_metrics,
                        baseline_name=compare_baseline,
                    )

                    # Print comparison results (console only)
                    if not json_output and output_format == "console":
                        from ...analysis.reporters.console import ConsoleReporter

                        reporter = ConsoleReporter()
                        reporter.print_baseline_comparison(comparison_result)

                except BaselineNotFoundError as e:
                    if json_output:
                        print_json({"error": str(e)})
                    else:
                        print_error(str(e))
                        console.print("\nAvailable baselines:")
                        baselines = baseline_manager.list_baselines()
                        for baseline_meta in baselines[:5]:
                            console.print(f"  • {baseline_meta.baseline_name}")
                    raise typer.Exit(1)

        # Quality gate: check if we should fail on smells
        if fail_on_smell and all_smells:
            failing_smells = filter_smells_by_severity(all_smells, severity_threshold)
            if failing_smells:
                console.print(
                    f"\n[red]❌ Quality gate failed: {len(failing_smells)} "
                    f"{severity_threshold}+ severity smell(s) detected[/red]"
                )
                raise typer.Exit(1)

    except ProjectNotFoundError as e:
        if json_output:
            print_json({"error": str(e)})
        else:
            print_error(str(e))
        raise typer.Exit(1)
    except typer.Exit:
        # Let typer.Exit propagate for quality gate failures
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        if json_output:
            print_json({"error": str(e)})
        else:
            print_error(f"Analysis failed: {e}")
        raise typer.Exit(2)  # Exit code 2 for analysis errors


def _find_analyzable_files(
    project_root: Path,
    language_filter: str | None,
    path_filter: Path | None,
    parser_registry: ParserRegistry,
    git_changed_files: list[Path] | None = None,
) -> list[Path]:
    """Find files that can be analyzed.

    Args:
        project_root: Root directory
        language_filter: Optional language filter
        path_filter: Optional path filter
        parser_registry: Parser registry for checking supported files
        git_changed_files: Optional list of git changed files to filter by

    Returns:
        List of file paths to analyze
    """

    # If git_changed_files is provided, use it as the primary filter
    if git_changed_files is not None:
        # Filter based on supported extensions and language
        files: list[Path] = []
        supported_extensions = parser_registry.get_supported_extensions()

        for file_path in git_changed_files:
            # Check if file extension is supported
            if file_path.suffix.lower() not in supported_extensions:
                logger.debug(f"Skipping unsupported file type: {file_path}")
                continue

            # Apply language filter
            if language_filter:
                try:
                    parser = parser_registry.get_parser_for_file(file_path)
                    if parser.language.lower() != language_filter.lower():
                        logger.debug(
                            f"Skipping file (language mismatch): {file_path} "
                            f"({parser.language} != {language_filter})"
                        )
                        continue
                except Exception as e:
                    logger.debug(f"Skipping file (parser error): {file_path}: {e}")
                    continue

            # Apply path filter if specified
            if path_filter:
                path_filter_resolved = path_filter.resolve()
                file_path_resolved = file_path.resolve()

                # Check if file is within path_filter scope
                try:
                    # If path_filter is a file, only include that specific file
                    if path_filter_resolved.is_file():
                        if file_path_resolved != path_filter_resolved:
                            continue
                    # If path_filter is a directory, check if file is within it
                    elif path_filter_resolved.is_dir():
                        file_path_resolved.relative_to(path_filter_resolved)
                except ValueError:
                    # File is not within path_filter scope
                    logger.debug(f"Skipping file (outside path filter): {file_path}")
                    continue

            files.append(file_path)

        return sorted(files)

    # No git filtering - fall back to standard directory traversal
    # Determine base path to search
    base_path = path_filter if path_filter and path_filter.exists() else project_root

    # If path_filter is a file, return just that file
    if base_path.is_file():
        # Check if file extension is supported
        if base_path.suffix.lower() in parser_registry.get_supported_extensions():
            return [base_path]
        return []

    # Find all supported files
    files = []
    supported_extensions = parser_registry.get_supported_extensions()

    # Common ignore patterns - exact matches for directory names
    ignore_dirs = {
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
        "vendor",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        "site-packages",
        ".nox",
        "env",
        ".env",
        "virtualenv",
        ".cache",
        ".uv",
    }

    # Prefix patterns - match any directory starting with these
    ignore_prefixes = {".venv", "venv", ".env", "env"}

    for file_path in base_path.rglob("*"):
        # Skip symlinks to prevent traversing outside project
        if file_path.is_symlink():
            continue

        # Skip directories
        if file_path.is_dir():
            continue

        # Skip files in ignored directories (exact match or prefix match)
        should_skip = False
        for part in file_path.parts:
            # Exact match
            if part in ignore_dirs:
                should_skip = True
                break
            # Prefix match (e.g., .venv-mcp, venv-test, .env-local)
            for prefix in ignore_prefixes:
                if part.startswith(prefix):
                    should_skip = True
                    break
            if should_skip:
                break

        if should_skip:
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
            with open(file_path, encoding="utf-8", errors="replace") as f:
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
        except Exception as e:
            logger.debug("Failed to read file metrics for %s: %s", file_path, e)

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


def _print_trends(trend_data: TrendData) -> None:
    """Print trend analysis to console.

    Args:
        trend_data: TrendData from TrendTracker
    """
    from rich.panel import Panel
    from rich.table import Table

    # Build trend display
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Direction")
    table.add_column("Change")

    def trend_icon(direction: TrendDirection) -> str:
        """Get icon for trend direction."""
        if direction == TrendDirection.IMPROVING:
            return "[green]↓ improving[/green]"
        elif direction == TrendDirection.WORSENING:
            return "[red]↑ worsening[/red]"
        else:
            return "[dim]→ stable[/dim]"

    def format_change(change: float, invert: bool = False) -> str:
        """Format percentage change with color."""
        if abs(change) < 0.1:
            return "[dim]—[/dim]"
        # For complexity/smells, negative is good; for health, positive is good
        is_good = (change < 0) if not invert else (change > 0)
        color = "green" if is_good else "red"
        sign = "+" if change > 0 else ""
        return f"[{color}]{sign}{change:.1f}%[/{color}]"

    # Complexity trend
    table.add_row(
        "Complexity",
        trend_icon(trend_data.complexity_direction),
        format_change(trend_data.avg_complexity_change),
    )

    # Smell trend
    table.add_row(
        "Code Smells",
        trend_icon(trend_data.smell_direction),
        format_change(trend_data.smell_count_change),
    )

    # Health trend
    table.add_row(
        "Health Score",
        trend_icon(trend_data.health_direction),
        format_change(
            (
                trend_data.health_trend[-1][1] - trend_data.health_trend[0][1]
                if len(trend_data.health_trend) >= 2
                else 0
            ),
            invert=True,
        ),
    )

    # Show panel with snapshot count
    snapshot_count = len(trend_data.snapshots)
    panel = Panel(
        table,
        title=f"[bold cyan]Trends[/bold cyan] (last 30 days, {snapshot_count} snapshots)",
        border_style="cyan",
        padding=(0, 1),
    )
    console.print(panel)

    # Show critical regressions if any
    if trend_data.critical_regressions:
        console.print("\n[bold red]⚠ Regressions Detected:[/bold red]")
        for regression in trend_data.critical_regressions[:3]:
            console.print(
                f"  • [red]{regression.file_path}[/red]: "
                f"complexity {regression.change_percentage:+.1f}%"
            )

    # Show significant improvements if any
    if trend_data.significant_improvements:
        console.print("\n[bold green]✓ Improvements:[/bold green]")
        for improvement in trend_data.significant_improvements[:3]:
            console.print(
                f"  • [green]{improvement.file_path}[/green]: "
                f"complexity {improvement.change_percentage:+.1f}%"
            )


@analyze_app.command(name="dead-code")
def dead_code(
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
    entry_point: list[str] = typer.Option(
        [],
        "--entry-point",
        "-e",
        help="Custom entry point in format 'file.py:function_name'",
        rich_help_panel="🔍 Entry Point Options",
    ),
    include_public: bool = typer.Option(
        False,
        "--include-public",
        help="Treat all public functions as entry points",
        rich_help_panel="🔍 Entry Point Options",
    ),
    min_confidence: str = typer.Option(
        "low",
        "--min-confidence",
        help="Minimum confidence level: high, medium, low",
        rich_help_panel="🔍 Filters",
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        help="Exclude file patterns (e.g., '**/tests/**', '**/_*.py')",
        rich_help_panel="🔍 Filters",
    ),
    output_format: str = typer.Option(
        "console",
        "--output",
        "-o",
        help="Output format: console, json, sarif, markdown",
        rich_help_panel="📊 Output Options",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        "-f",
        help="Output file path (stdout if not specified)",
        rich_help_panel="📊 Output Options",
    ),
    fail_on_dead: bool = typer.Option(
        False,
        "--fail-on-dead",
        help="Exit with code 1 if dead code is detected (for CI/CD)",
        rich_help_panel="🚦 Quality Gates",
    ),
) -> None:
    """🧹 Detect dead/unreachable code in your project.

    Analyzes your codebase to identify functions that are never called from
    any entry point. Entry points include main blocks, CLI commands, HTTP routes,
    tests, and module exports.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Analyze project for dead code:[/green]
        $ mcp-vector-search analyze dead-code

    [green]Include public functions as entry points:[/green]
        $ mcp-vector-search analyze dead-code --include-public

    [green]Filter by confidence level:[/green]
        $ mcp-vector-search analyze dead-code --min-confidence high

    [green]Custom entry point:[/green]
        $ mcp-vector-search analyze dead-code --entry-point "main.py:run"

    [bold cyan]Output Formats:[/bold cyan]

    [green]Export to JSON:[/green]
        $ mcp-vector-search analyze dead-code --output json

    [green]Export to SARIF for GitHub:[/green]
        $ mcp-vector-search analyze dead-code --output sarif --output-file report.sarif

    [green]Export to Markdown report:[/green]
        $ mcp-vector-search analyze dead-code --output markdown --output-file report.md

    [bold cyan]CI/CD Integration:[/bold cyan]

    [green]Fail build if dead code found:[/green]
        $ mcp-vector-search analyze dead-code --fail-on-dead

    [green]Exclude test files:[/green]
        $ mcp-vector-search analyze dead-code --exclude "**/tests/**"

    [dim]💡 Tip: Use --include-public if you're building a library with public API.[/dim]
    """
    try:
        # Use provided project_root or current working directory
        if project_root is None:
            project_root = Path.cwd()

        # Run dead code analysis
        asyncio.run(
            run_dead_code_analysis(
                project_root=project_root,
                custom_entry_points=entry_point,
                include_public=include_public,
                min_confidence=min_confidence,
                exclude_patterns=exclude,
                output_format=output_format,
                output_file=output_file,
                fail_on_dead=fail_on_dead,
            )
        )

    except typer.Exit:
        # Re-raise typer.Exit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"Dead code analysis failed: {e}")
        print_error(f"Dead code analysis failed: {e}")
        raise typer.Exit(2)


async def run_dead_code_analysis(
    project_root: Path,
    custom_entry_points: list[str],
    include_public: bool = False,
    min_confidence: str = "low",
    exclude_patterns: list[str] | None = None,
    output_format: str = "console",
    output_file: Path | None = None,
    fail_on_dead: bool = False,
) -> None:
    """Run dead code analysis workflow.

    Args:
        project_root: Root directory of the project
        custom_entry_points: List of custom entry points
        include_public: Treat public functions as entry points
        min_confidence: Minimum confidence level (high, medium, low)
        exclude_patterns: File path patterns to exclude
        output_format: Output format (console, json, sarif, markdown)
        output_file: Output file path (None for stdout)
        fail_on_dead: Exit with code 1 if dead code found
    """
    from ...analysis.dead_code import Confidence, DeadCodeAnalyzer
    from ...analysis.dead_code_formatters import get_formatter
    from ...core.embeddings import create_embedding_function
    from ...core.factory import create_database

    try:
        # Check if project is initialized
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )
            raise typer.Exit(1)

        # Map string confidence to enum
        confidence_map = {
            "high": Confidence.HIGH,
            "medium": Confidence.MEDIUM,
            "low": Confidence.LOW,
        }
        min_confidence_enum = confidence_map.get(min_confidence.lower())
        if not min_confidence_enum:
            print_error(
                f"Invalid confidence level: {min_confidence}. Must be: high, medium, low"
            )
            raise typer.Exit(1)

        # Initialize database to get chunks
        print_info("Loading indexed code chunks...")

        config = project_manager.load_config()
        db_path = Path(config.index_path)
        embedding_function, _ = create_embedding_function(config.embedding_model)

        async with create_database(
            persist_directory=db_path,
            embedding_function=embedding_function,
        ) as db:
            # Get all chunks from database
            chunks = await db.get_all_chunks()

            if not chunks:
                print_error(
                    "No code chunks found. Run 'mcp-vector-search index' first."
                )
                raise typer.Exit(1)

            print_info(f"Analyzing {len(chunks)} code chunks...")

            # Convert CodeChunk objects to dict format expected by analyzer
            chunks_dict = []
            for chunk in chunks:
                chunks_dict.append(
                    {
                        "type": chunk.chunk_type,
                        "content": chunk.content,
                        "function_name": chunk.function_name,
                        "class_name": chunk.class_name,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "decorators": chunk.decorators or [],
                    }
                )

            # Parse custom entry points
            custom_entry_point_names = []
            for ep_str in custom_entry_points:
                if ":" in ep_str:
                    # Format: file.py:function_name
                    _, func_name = ep_str.split(":", 1)
                    custom_entry_point_names.append(func_name)
                else:
                    # Just function name
                    custom_entry_point_names.append(ep_str)

            # Create analyzer
            analyzer = DeadCodeAnalyzer(
                include_public_entry_points=include_public,
                custom_entry_points=custom_entry_point_names,
                exclude_patterns=exclude_patterns or [],
                min_confidence=min_confidence_enum,
            )

            # Run analysis
            report = analyzer.analyze(project_root, chunks_dict)

            # Format output
            formatter = get_formatter(output_format)

            if output_file:
                # Write to file
                with open(output_file, "w", encoding="utf-8") as f:
                    formatter.format(report, f)
                console.print(f"[green]✓[/green] Report written to: {output_file}")
            else:
                # Write to stdout
                formatter.format(report, sys.stdout)

            # Quality gate check
            if fail_on_dead and report.unreachable_count > 0:
                console.print(
                    f"\n[red]❌ Quality gate failed: {report.unreachable_count} "
                    f"unreachable function(s) detected[/red]"
                )
                raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Dead code analysis failed: {e}", exc_info=True)
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(2)


# Register engineers command directly (not as nested app)
from .analyze_engineers import analyze_engineers  # noqa: E402

analyze_app.command(name="engineers", help="👥 Profile engineers by code quality")(
    analyze_engineers
)


@analyze_app.command(name="review")
def review_code(
    review_type: str = typer.Argument(
        ...,
        help="Type of review: security, architecture, or performance",
    ),
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
    format: str = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: console, json, sarif, markdown",
        rich_help_panel="📊 Display Options",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (stdout for console, auto-named for others)",
        rich_help_panel="📊 Display Options",
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Scope review to specific path/directory",
        rich_help_panel="🔍 Filters",
    ),
    max_chunks: int = typer.Option(
        30,
        "--max-chunks",
        help="Maximum code chunks to analyze",
        min=10,
        max=100,
        rich_help_panel="⚡ Performance Options",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
        rich_help_panel="📊 Display Options",
    ),
) -> None:
    """🔍 AI-powered code review using vector search and LLM analysis.

    Performs specialized code reviews with targeted search queries and LLM analysis
    to identify security vulnerabilities, architectural issues, or performance problems.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Security review:[/green]
        $ mcp-vector-search analyze review security

    [green]Architecture review:[/green]
        $ mcp-vector-search analyze review architecture

    [green]Performance review:[/green]
        $ mcp-vector-search analyze review performance

    [bold cyan]Scoped Reviews:[/bold cyan]

    [green]Review specific module:[/green]
        $ mcp-vector-search analyze review security --path src/auth

    [green]Review with more context:[/green]
        $ mcp-vector-search analyze review architecture --max-chunks 50

    [bold cyan]Output Formats:[/bold cyan]

    [green]Export to JSON:[/green]
        $ mcp-vector-search analyze review security --format json --output findings.json

    [green]Export to SARIF (for GitHub):[/green]
        $ mcp-vector-search analyze review security --format sarif --output findings.sarif

    [green]Export to Markdown report:[/green]
        $ mcp-vector-search analyze review architecture --format markdown --output review.md

    [dim]💡 Tip: Use --verbose to see search queries and LLM analysis progress.[/dim]
    """
    try:
        # Use provided project_root or current working directory
        if project_root is None:
            project_root = Path.cwd()

        # Validate review type
        from ...analysis.review import ReviewType

        valid_types = [rt.value for rt in ReviewType]
        if review_type.lower() not in valid_types:
            print_error(
                f"Invalid review type: {review_type}. Must be one of: {', '.join(valid_types)}"
            )
            raise typer.Exit(1)

        # Validate format
        valid_formats = ["console", "json", "sarif", "markdown"]
        format_lower = format.lower()
        if format_lower not in valid_formats:
            print_error(
                f"Invalid format: {format}. Must be one of: {', '.join(valid_formats)}"
            )
            raise typer.Exit(1)

        # Run review
        asyncio.run(
            run_review_analysis(
                project_root=project_root,
                review_type=review_type.lower(),
                output_format=format_lower,
                output_file=output,
                scope_path=path,
                max_chunks=max_chunks,
                verbose=verbose,
            )
        )

    except typer.Exit:
        # Re-raise typer.Exit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"Review analysis failed: {e}")
        print_error(f"Review analysis failed: {e}")
        raise typer.Exit(2)


async def run_review_analysis(
    project_root: Path,
    review_type: str,
    output_format: str = "console",
    output_file: Path | None = None,
    scope_path: Path | None = None,
    max_chunks: int = 30,
    verbose: bool = False,
) -> None:
    """Run AI-powered code review workflow.

    Args:
        project_root: Root directory of the project
        review_type: Type of review (security, architecture, performance)
        output_format: Output format (console, json, sarif, markdown)
        output_file: Output file path (None for stdout)
        scope_path: Optional path to scope review
        max_chunks: Maximum code chunks to analyze
        verbose: Show detailed progress
    """
    from rich.console import Console

    from ...analysis.review import ReviewEngine, ReviewType
    from ...core.embeddings import create_embedding_function
    from ...core.factory import create_database
    from ...core.llm_client import LLMClient
    from ...core.search import SemanticSearchEngine

    console = Console()

    try:
        # Check if project is initialized
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )
            raise typer.Exit(1)

        config = project_manager.load_config()

        # Initialize search engine
        if verbose:
            console.print("[bold blue]Initializing search engine...[/bold blue]")

        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = create_database(
            persist_directory=config.index_path / "lance",
            embedding_function=embedding_function,
            collection_name="vectors",
        )
        await database.initialize()

        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

        # Initialize LLM client
        if verbose:
            console.print("[bold blue]Initializing LLM client...[/bold blue]")

        llm_client = LLMClient()

        # Try to initialize knowledge graph (optional)
        knowledge_graph = None
        try:
            from ...core.knowledge_graph import KnowledgeGraph

            kg_path = config.index_path / "kg.db"
            if kg_path.exists():
                knowledge_graph = KnowledgeGraph(kg_path)
                if verbose:
                    console.print("[bold blue]Knowledge graph loaded[/bold blue]")
        except Exception as e:
            logger.debug(f"Knowledge graph not available: {e}")
            if verbose:
                console.print(
                    "[yellow]Knowledge graph not available (optional)[/yellow]"
                )

        # Create review engine
        review_engine = ReviewEngine(
            search_engine=search_engine,
            knowledge_graph=knowledge_graph,
            llm_client=llm_client,
            project_root=project_root,
        )

        # Convert review type string to enum
        review_type_enum = ReviewType(review_type)

        # Prepare scope string
        scope_str = str(scope_path.relative_to(project_root)) if scope_path else None

        # Run review
        if verbose:
            console.print(f"\n[bold blue]Running {review_type} review...[/bold blue]")

        async with database:
            result = await review_engine.run_review(
                review_type=review_type_enum,
                scope=scope_str,
                max_chunks=max_chunks,
            )

        # Format output based on format
        if output_format == "console":
            _print_review_console(result, console)
        elif output_format == "json":
            _export_review_json(result, output_file)
        elif output_format == "sarif":
            _export_review_sarif(result, output_file, project_root)
        elif output_format == "markdown":
            _export_review_markdown(result, output_file)

    except ProjectNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Review analysis failed: {e}", exc_info=True)
        print_error(f"Review analysis failed: {e}")
        raise typer.Exit(2)


def _print_review_console(result, console) -> None:
    """Print review results to console in rich format."""

    # Header
    console.print(
        f"\n[bold blue]🔍 {result.review_type.value.title()} Review[/bold blue] — {result.scope}"
    )
    console.print("━" * 80)

    # Summary
    severity_counts = {}
    for finding in result.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    console.print(
        f"\nFound [bold]{len(result.findings)}[/bold] findings "
        f"([red]{severity_counts.get('critical', 0)} critical[/red], "
        f"[yellow]{severity_counts.get('high', 0)} high[/yellow], "
        f"[blue]{severity_counts.get('medium', 0)} medium[/blue])\n"
    )

    # Findings table
    for finding in result.findings:
        # Color-code by severity
        severity_colors = {
            "critical": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "green",
            "info": "dim",
        }
        color = severity_colors.get(finding.severity.value, "white")

        # Print finding
        console.print(
            f"[{color}]{'🔴' if finding.severity.value == 'critical' else '🟠' if finding.severity.value == 'high' else '🟡'} "
            f"{finding.severity.value.upper()}: {finding.title}[/{color}]"
        )
        console.print(f"   {finding.file_path}:{finding.start_line}-{finding.end_line}")
        if finding.cwe_id:
            console.print(f"   {finding.cwe_id} | Confidence: {finding.confidence:.0%}")
        else:
            console.print(f"   Confidence: {finding.confidence:.0%}")
        console.print(f"   → {finding.recommendation}\n")

    # Metadata
    console.print(
        f"\n[dim]Summary: Analyzed {result.context_chunks_used} code chunks, "
        f"{result.kg_relationships_used} KG relationships[/dim]"
    )
    console.print(
        f"[dim]Review completed in {result.duration_seconds:.1f}s using {result.model_used}[/dim]\n"
    )


def _export_review_json(result, output_file: Path | None) -> None:
    """Export review results to JSON format."""
    import json

    output_data = {
        "review_type": result.review_type.value,
        "scope": result.scope,
        "summary": result.summary,
        "findings": [
            {
                "title": f.title,
                "description": f.description,
                "severity": f.severity.value,
                "file_path": f.file_path,
                "start_line": f.start_line,
                "end_line": f.end_line,
                "category": f.category,
                "recommendation": f.recommendation,
                "confidence": f.confidence,
                "cwe_id": f.cwe_id,
                "code_snippet": f.code_snippet,
                "related_files": f.related_files,
            }
            for f in result.findings
        ],
        "metadata": {
            "context_chunks_used": result.context_chunks_used,
            "kg_relationships_used": result.kg_relationships_used,
            "model_used": result.model_used,
            "duration_seconds": result.duration_seconds,
        },
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        from ..output import console

        console.print(f"[green]✓[/green] JSON report written to: {output_file}")
    else:
        # Print to stdout
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


def _export_review_sarif(result, output_file: Path | None, base_path: Path) -> None:
    """Export review results to SARIF 2.1.0 format."""
    import json
    from datetime import UTC, datetime

    # Build SARIF document
    sarif_doc = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "MCP Vector Search - Code Review",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/bobmatnyc/mcp-vector-search",
                        "rules": _build_sarif_rules(result),
                    }
                },
                "results": [
                    _finding_to_sarif_result(f, base_path) for f in result.findings
                ],
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": datetime.now(UTC).isoformat(),
                    }
                ],
            }
        ],
    }

    # Write to file
    if not output_file:
        output_file = Path(f"review-{result.review_type.value}.sarif")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sarif_doc, f, indent=2, ensure_ascii=False)

    from ..output import console

    console.print(f"[green]✓[/green] SARIF report written to: {output_file}")


def _build_sarif_rules(result) -> list[dict]:
    """Build SARIF rules from review findings."""
    unique_categories = {}
    for finding in result.findings:
        if finding.category not in unique_categories:
            unique_categories[finding.category] = finding

    rules = []
    for category, finding in unique_categories.items():
        rule = {
            "id": category.lower().replace(" ", "-"),
            "shortDescription": {"text": category},
            "fullDescription": {"text": finding.description},
            "help": {"text": finding.recommendation},
            "defaultConfiguration": {
                "level": _severity_to_sarif_level(finding.severity.value)
            },
        }

        # Add CWE reference for security findings
        if finding.cwe_id:
            rule["properties"] = {"cwe": finding.cwe_id}

        rules.append(rule)

    return rules


def _severity_to_sarif_level(severity: str) -> str:
    """Map review severity to SARIF level."""
    mapping = {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "warning",
        "info": "note",
    }
    return mapping.get(severity, "warning")


def _finding_to_sarif_result(finding, base_path: Path) -> dict:
    """Convert ReviewFinding to SARIF result."""
    from pathlib import Path

    # Make path relative
    file_path = finding.file_path
    try:
        file_path_obj = Path(file_path)
        if file_path_obj.is_absolute():
            file_path = str(file_path_obj.relative_to(base_path))
    except (ValueError, OSError):
        pass

    result = {
        "ruleId": finding.category.lower().replace(" ", "-"),
        "level": _severity_to_sarif_level(finding.severity.value),
        "message": {"text": f"{finding.title}: {finding.description}"},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": file_path},
                    "region": {
                        "startLine": finding.start_line,
                        "endLine": finding.end_line,
                    },
                }
            }
        ],
        "properties": {
            "confidence": finding.confidence,
            "recommendation": finding.recommendation,
        },
    }

    if finding.cwe_id:
        result["properties"]["cwe"] = finding.cwe_id

    return result


def _export_review_markdown(result, output_file: Path | None) -> None:
    """Export review results to Markdown format."""
    lines = [
        f"# {result.review_type.value.title()} Review Report",
        "",
        f"**Scope:** {result.scope}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        result.summary,
        "",
        "## Findings",
        "",
    ]

    # Group findings by severity
    by_severity = {}
    for finding in result.findings:
        severity = finding.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(finding)

    # Output by severity (critical -> info)
    severity_order = ["critical", "high", "medium", "low", "info"]
    for severity in severity_order:
        if severity not in by_severity:
            continue

        findings = by_severity[severity]
        severity_icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "ℹ️",
        }
        icon = severity_icons.get(severity, "•")

        lines.append(f"### {icon} {severity.upper()}")
        lines.append("")

        for finding in findings:
            lines.append(f"#### {finding.title}")
            lines.append("")
            lines.append(
                f"**Location:** `{finding.file_path}:{finding.start_line}-{finding.end_line}`"
            )
            lines.append(f"**Category:** {finding.category}")
            lines.append(f"**Confidence:** {finding.confidence:.0%}")
            if finding.cwe_id:
                lines.append(f"**CWE:** {finding.cwe_id}")
            lines.append("")
            lines.append(f"**Description:** {finding.description}")
            lines.append("")
            lines.append(f"**Recommendation:** {finding.recommendation}")
            lines.append("")
            if finding.code_snippet:
                lines.append("**Code Snippet:**")
                lines.append("```")
                lines.append(finding.code_snippet)
                lines.append("```")
                lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Code Chunks Analyzed:** {result.context_chunks_used}")
    lines.append(f"- **Knowledge Graph Relationships:** {result.kg_relationships_used}")
    lines.append(f"- **Model Used:** {result.model_used}")
    lines.append(f"- **Duration:** {result.duration_seconds:.1f}s")
    lines.append("")

    markdown_content = "\n".join(lines)

    # Write to file
    if not output_file:
        output_file = Path(f"review-{result.review_type.value}.md")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    from ..output import console

    console.print(f"[green]✓[/green] Markdown report written to: {output_file}")


if __name__ == "__main__":
    analyze_app()
