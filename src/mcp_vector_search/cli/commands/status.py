"""Status command for MCP Vector Search CLI."""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.panel import Panel
from rich.table import Table

from ... import __version__
from ...analysis.storage.metrics_store import MetricsStore, MetricsStoreError
from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.indexer import SemanticIndexer
from ...core.project import ProjectManager
from ..output import (
    console,
    print_dependency_status,
    print_error,
    print_info,
    print_json,
)

# Create status subcommand app
status_app = typer.Typer(help="Show project status and statistics")


@status_app.command()
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including paths and patterns",
        rich_help_panel="📊 Display Options",
    ),
    health_check: bool = typer.Option(
        False,
        "--health-check",
        help="Perform comprehensive health check of all components",
        rich_help_panel="🔍 Diagnostics",
    ),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Check Claude Code MCP integration status",
        rich_help_panel="🔍 Diagnostics",
    ),
    metrics: bool = typer.Option(
        False,
        "--metrics",
        "-m",
        help="Show project metrics summary from latest analysis",
        rich_help_panel="📊 Display Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output status in JSON format",
        rich_help_panel="📊 Display Options",
    ),
    skip_stats: bool = typer.Option(
        False,
        "--skip-stats",
        help="Skip detailed statistics collection (useful for large databases >500MB)",
        rich_help_panel="🔍 Diagnostics",
    ),
    force_stats: bool = typer.Option(
        False,
        "--force-stats",
        help="Force full statistics even for large databases (may crash on very large DBs)",
        rich_help_panel="🔍 Diagnostics",
    ),
) -> None:
    """📊 Show project status and indexing statistics.

    Displays comprehensive information about your project including configuration,
    indexing statistics, and system health. Use this to verify setup and monitor
    indexing progress.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Quick status check:[/green]
        $ mcp-vector-search status

    [green]Show code metrics summary:[/green]
        $ mcp-vector-search status --metrics

    [green]Detailed status with all information:[/green]
        $ mcp-vector-search status --verbose

    [green]Check MCP integration:[/green]
        $ mcp-vector-search status --mcp

    [bold cyan]Diagnostics:[/bold cyan]

    [green]Full health check:[/green]
        $ mcp-vector-search status --health-check

    [green]Export metrics to JSON:[/green]
        $ mcp-vector-search status --metrics --json > metrics.json

    [green]Combined diagnostics:[/green]
        $ mcp-vector-search status --verbose --health-check --mcp

    [dim]💡 Tip: Use --metrics to see code quality analysis from 'mcp-vector-search analyze'[/dim]
    """
    try:
        # Use provided project_root or current working directory
        if project_root is None:
            project_root = Path.cwd()

        async def run_status_with_timeout():
            """Run status command with timeout protection."""
            try:
                await asyncio.wait_for(
                    show_status(
                        project_root=project_root,
                        verbose=verbose,
                        health_check=health_check,
                        mcp=mcp,
                        metrics=metrics,
                        json_output=json_output,
                        skip_stats=skip_stats,
                        force_stats=force_stats,
                    ),
                    timeout=30.0,  # 30 second timeout
                )
            except TimeoutError:
                logger.error("Status check timed out after 30 seconds")
                print_error(
                    "Status check timed out after 30 seconds. "
                    "Try running with --verbose for more details."
                )
                raise typer.Exit(1)

        asyncio.run(run_status_with_timeout())

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


async def show_status(
    project_root: Path,
    verbose: bool = False,
    health_check: bool = False,
    mcp: bool = False,
    metrics: bool = False,
    json_output: bool = False,
    skip_stats: bool = False,
    force_stats: bool = False,
) -> None:
    """Show comprehensive project status.

    Args:
        project_root: Project root directory
        verbose: Show detailed information
        health_check: Perform health check
        mcp: Check MCP integration
        metrics: Show metrics summary
        json_output: Output as JSON
        skip_stats: Skip statistics collection
        force_stats: Force stats even for large databases
    """
    status_data = {}

    try:
        # If metrics flag is set, show metrics summary and return
        if metrics:
            await show_metrics_summary(
                project_root=project_root,
                json_output=json_output,
            )
            return
        # Check if project is initialized - use the specified project root
        project_manager = ProjectManager(project_root)

        if not project_manager.is_initialized():
            if json_output:
                status_data = {
                    "initialized": False,
                    "project_root": str(project_root),
                    "error": "Project not initialized",
                }
                print_json(status_data)
            else:
                print_error(f"Project not initialized at {project_root}")
                print_info("Run 'mcp-vector-search init' to initialize the project")
            return

        # Get configuration first
        config = project_manager.load_config()

        # Get indexing statistics from database (fast, no filesystem scan)
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_root,
            config=config,
        )

        # Get indexing stats (using database stats only, no filesystem scan)
        # Determine whether to skip stats based on flags and database size
        should_skip_stats = skip_stats and not force_stats

        async with database:
            db_stats = await database.get_stats(skip_stats=should_skip_stats)

            # Show warning if large database detected
            if isinstance(db_stats.total_chunks, str):
                if not json_output:
                    console.print(
                        f"\n[yellow]⚠️  Large database detected ({db_stats.index_size_mb:.1f} MB)[/yellow]"
                    )
                    console.print(
                        "    [yellow]Detailed statistics skipped to prevent potential crashes.[/yellow]"
                    )
                    if not force_stats:
                        console.print(
                            "    [dim]Use --force-stats to attempt full statistics (may crash).[/dim]\n"
                        )

            index_stats = await indexer.get_indexing_stats(db_stats=db_stats)

        # Get project information with pre-computed file count (avoids filesystem scan)
        project_info = project_manager.get_project_info(file_count=db_stats.total_files)

        # Get version information
        index_version = indexer.get_index_version()
        needs_reindex = indexer.needs_reindex_for_version()

        # Compile status data
        status_data = {
            "project": {
                "name": project_info.name,
                "root_path": str(project_info.root_path),
                "initialized": project_info.is_initialized,
                "languages": project_info.languages,
                "file_count": project_info.file_count,
            },
            "configuration": {
                "embedding_model": config.embedding_model,
                "similarity_threshold": config.similarity_threshold,
                "file_extensions": config.file_extensions,
                "max_chunk_size": config.max_chunk_size,
                "cache_embeddings": config.cache_embeddings,
                "watch_files": config.watch_files,
                "auto_reindex_on_upgrade": config.auto_reindex_on_upgrade,
            },
            "index": {
                "total_files": index_stats.get("total_indexable_files", 0),
                "indexed_files": index_stats.get("indexed_files", 0),
                "total_chunks": index_stats.get("total_chunks", 0),
                "languages": index_stats.get("languages", {}),
                "index_size_mb": db_stats.index_size_mb,
                "last_updated": db_stats.last_updated,
                "index_version": index_version,
                "current_version": __version__,
                "needs_reindex": needs_reindex,
            },
        }

        # Add health check if requested
        if health_check:
            health_status = await perform_health_check(project_root, config)
            status_data["health"] = health_status

        # Add MCP integration check if requested
        if mcp:
            mcp_status = await check_mcp_integration(project_root)
            status_data["mcp"] = mcp_status

        # Add verbose information
        if verbose:
            status_data["verbose"] = {
                "config_path": str(project_info.config_path),
                "index_path": str(project_info.index_path),
                "ignore_patterns": list(indexer.get_ignore_patterns()),
                "parser_info": index_stats.get("parser_info", {}),
            }

        # Output results
        if json_output:
            print_json(status_data)
        else:
            _display_status(status_data, verbose, mcp)

    except ProjectNotFoundError:
        if json_output:
            print_json({"initialized": False, "error": "Project not initialized"})
        else:
            print_error("Project not initialized")
            print_info("Run 'mcp-vector-search init' to initialize the project")
    except Exception as e:
        if json_output:
            print_json({"error": str(e)})
        else:
            print_error(f"Failed to get status: {e}")
        raise


def _display_status(
    status_data: dict[str, Any], verbose: bool, mcp: bool = False
) -> None:
    """Display status in human-readable format."""
    project_data = status_data["project"]
    config_data = status_data["configuration"]
    index_data = status_data["index"]

    # Project information
    console.print("[bold blue]Project Information[/bold blue]")
    console.print(f"  Name: {project_data['name']}")
    console.print(f"  Root: {project_data['root_path']}")
    console.print(
        f"  Languages: {', '.join(project_data['languages']) if project_data['languages'] else 'None detected'}"
    )
    console.print(f"  Indexable Files: {project_data['file_count']}")
    console.print()

    # Configuration
    console.print("[bold blue]Configuration[/bold blue]")
    console.print(f"  Embedding Model: {config_data['embedding_model']}")
    console.print(f"  Similarity Threshold: {config_data['similarity_threshold']}")
    console.print(f"  File Extensions: {', '.join(config_data['file_extensions'])}")
    console.print(
        f"  Cache Embeddings: {'✓' if config_data['cache_embeddings'] else '✗'}"
    )
    console.print()

    # Index statistics
    console.print("[bold blue]Index Statistics[/bold blue]")
    console.print(
        f"  Indexed Files: {index_data['indexed_files']}/{index_data['total_files']}"
    )

    # Handle both int and string values for total_chunks
    total_chunks = index_data["total_chunks"]
    if isinstance(total_chunks, str):
        console.print(f"  Total Chunks: [yellow]{total_chunks}[/yellow]")
    else:
        console.print(f"  Total Chunks: {total_chunks}")

    console.print(f"  Index Size: {index_data['index_size_mb']:.2f} MB")

    # Version information
    index_version = index_data.get("index_version")
    current_version = index_data.get("current_version", __version__)
    needs_reindex = index_data.get("needs_reindex", False)

    if index_version:
        if needs_reindex:
            console.print(
                f"  Version: [yellow]{index_version}[/yellow] (current: {current_version}) [yellow]⚠️  Reindex recommended[/yellow]"
            )
        else:
            console.print(f"  Version: [green]{index_version}[/green] (up to date)")
    else:
        console.print(
            f"  Version: [yellow]Not tracked[/yellow] (current: {current_version}) [yellow]⚠️  Reindex recommended[/yellow]"
        )

    if index_data["languages"]:
        console.print("  Language Distribution:")
        for lang, count in index_data["languages"].items():
            console.print(f"    {lang}: {count} chunks")
    console.print()

    # Show reindex recommendation if needed
    if needs_reindex:
        console.print(
            "[yellow]💡 Tip: Run 'mcp-vector-search index' to reindex with the latest improvements[/yellow]"
        )
        console.print()

    # Health check results
    if "health" in status_data:
        health_data = status_data["health"]
        console.print("[bold blue]Health Check[/bold blue]")

        overall_health = health_data.get("overall", "unknown")
        if overall_health == "healthy":
            console.print("[green]✓ System is healthy[/green]")
        elif overall_health == "warning":
            console.print("[yellow]⚠ System has warnings[/yellow]")
        else:
            console.print("[red]✗ System has issues[/red]")

        for component, status in health_data.get("components", {}).items():
            if status == "ok":
                console.print(f"  [green]✓[/green] {component}")
            elif status == "warning":
                console.print(f"  [yellow]⚠[/yellow] {component}")
            else:
                console.print(f"  [red]✗[/red] {component}")
        console.print()

    # MCP integration status
    if "mcp" in status_data:
        mcp_data = status_data["mcp"]
        console.print("[bold blue]MCP Integration[/bold blue]")

        if mcp_data.get("claude_available"):
            console.print("[green]✓[/green] Claude Code: Available")
        else:
            console.print("[red]✗[/red] Claude Code: Not available")

        server_status = mcp_data.get("server_status", "unknown")
        server_name = mcp_data.get("server_name", "mcp-vector-search")

        if server_status == "installed":
            console.print(f"[green]✓[/green] MCP Server '{server_name}': Installed")
        elif server_status == "not_installed":
            console.print(f"[red]✗[/red] MCP Server '{server_name}': Not installed")
        else:
            console.print(
                f"[yellow]⚠[/yellow] MCP Server '{server_name}': {server_status}"
            )

        if mcp_data.get("project_config"):
            console.print("[green]✓[/green] Project Configuration: Found")
        else:
            console.print("[red]✗[/red] Project Configuration: Missing")

        console.print()

    # Verbose information
    if verbose and "verbose" in status_data:
        verbose_data = status_data["verbose"]
        console.print("[bold blue]Detailed Information[/bold blue]")
        console.print(f"  Config Path: {verbose_data['config_path']}")
        console.print(f"  Index Path: {verbose_data['index_path']}")
        console.print(
            f"  Ignore Patterns: {', '.join(verbose_data['ignore_patterns'])}"
        )


async def perform_health_check(project_root: Path, config) -> dict[str, Any]:
    """Perform comprehensive health check."""
    health_status = {
        "overall": "healthy",
        "components": {},
        "issues": [],
    }

    try:
        # Check dependencies
        deps_ok = check_dependencies()
        health_status["components"]["dependencies"] = "ok" if deps_ok else "error"
        if not deps_ok:
            health_status["issues"].append("Missing dependencies")

        # Check configuration
        try:
            # Validate embedding model
            embedding_function, _ = create_embedding_function(config.embedding_model)
            health_status["components"]["embedding_model"] = "ok"
        except Exception as e:
            health_status["components"]["embedding_model"] = "error"
            health_status["issues"].append(f"Embedding model error: {e}")

        # Check database
        try:
            database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )
            async with database:
                # Skip stats for health check to avoid crashes on large DBs
                await database.get_stats(skip_stats=True)
            health_status["components"]["database"] = "ok"
        except Exception as e:
            health_status["components"]["database"] = "error"
            health_status["issues"].append(f"Database error: {e}")

        # Check file system permissions
        try:
            config.index_path.mkdir(parents=True, exist_ok=True)
            test_file = config.index_path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            health_status["components"]["file_permissions"] = "ok"
        except Exception as e:
            health_status["components"]["file_permissions"] = "error"
            health_status["issues"].append(f"File permission error: {e}")

        # Determine overall health
        if any(status == "error" for status in health_status["components"].values()):
            health_status["overall"] = "error"
        elif any(
            status == "warning" for status in health_status["components"].values()
        ):
            health_status["overall"] = "warning"

    except Exception as e:
        health_status["overall"] = "error"
        health_status["issues"].append(f"Health check failed: {e}")

    return health_status


async def check_mcp_integration(
    project_root: Path, server_name: str = "mcp-vector-search"
) -> dict[str, Any]:
    """Check MCP integration status."""
    mcp_status = {
        "claude_available": False,
        "server_status": "unknown",
        "server_name": server_name,
        "project_config": False,
        "issues": [],
    }

    try:
        # Import MCP functions from the mcp command module
        from .mcp import check_claude_code_available, get_claude_command

        # Check if Claude Code is available
        mcp_status["claude_available"] = check_claude_code_available()

        if mcp_status["claude_available"]:
            claude_cmd = get_claude_command()

            # Check if MCP server is installed
            try:
                result = subprocess.run(
                    [claude_cmd, "mcp", "get", server_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    mcp_status["server_status"] = "installed"
                else:
                    mcp_status["server_status"] = "not_installed"
                    mcp_status["issues"].append(
                        f"MCP server '{server_name}' not found in Claude Code"
                    )

            except subprocess.TimeoutExpired:
                mcp_status["server_status"] = "timeout"
                mcp_status["issues"].append("Timeout checking MCP server status")
            except Exception as e:
                mcp_status["server_status"] = "error"
                mcp_status["issues"].append(f"Error checking MCP server: {e}")
        else:
            mcp_status["issues"].append("Claude Code not available")

        # Check for project-level .claude.json configuration
        claude_json_path = project_root / ".claude.json"
        if claude_json_path.exists():
            try:
                with open(claude_json_path) as f:
                    config = json.load(f)
                if config.get("mcpServers", {}).get(server_name):
                    mcp_status["project_config"] = True
                else:
                    mcp_status["issues"].append(
                        f"MCP server '{server_name}' not found in project .claude.json"
                    )
            except Exception as e:
                mcp_status["issues"].append(f"Error reading project .claude.json: {e}")
        else:
            mcp_status["issues"].append("Project .claude.json not found")

    except Exception as e:
        mcp_status["issues"].append(f"MCP integration check failed: {e}")

    return mcp_status


async def show_metrics_summary(
    project_root: Path,
    json_output: bool = False,
) -> None:
    """Show code metrics summary from latest analysis.

    Args:
        project_root: Project root directory
        json_output: Output as JSON instead of formatted console

    Raises:
        typer.Exit: If no metrics found or error occurs
    """
    try:
        # Get metrics storage location
        storage_dir = project_root / ".mcp-vector-search"
        db_path = storage_dir / "metrics.db"

        # Check if metrics database exists
        if not db_path.exists():
            if json_output:
                print_json(
                    {
                        "status": "error",
                        "error": "No metrics found",
                        "message": "Run 'mcp-vector-search analyze' first",
                    }
                )
            else:
                console.print(
                    "[yellow]No metrics found. Run 'mcp-vector-search analyze' first.[/yellow]"
                )
            raise typer.Exit(1)

        # Load metrics store
        store = MetricsStore(db_path)

        # Get latest snapshot for this project
        snapshots = store.get_project_history(str(project_root), limit=1)

        if not snapshots:
            if json_output:
                print_json(
                    {
                        "status": "error",
                        "error": "No metrics found for this project",
                        "message": "Run 'mcp-vector-search analyze' first",
                    }
                )
            else:
                console.print(
                    "[yellow]No metrics found for this project. "
                    "Run 'mcp-vector-search analyze' first.[/yellow]"
                )
            raise typer.Exit(1)

        latest = snapshots[0]

        # Output JSON or formatted
        if json_output:
            _output_metrics_json(latest)
        else:
            _print_metrics_summary(latest)

    except MetricsStoreError as e:
        logger.error(f"Failed to load metrics: {e}")
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(f"Failed to load metrics: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading metrics: {e}")
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


def _output_metrics_json(snapshot) -> None:
    """Output metrics snapshot as JSON.

    Args:
        snapshot: ProjectSnapshot from metrics store
    """
    output = {
        "status": "success",
        "snapshot_id": snapshot.snapshot_id,
        "project_path": snapshot.project_path,
        "timestamp": snapshot.timestamp.isoformat(),
        "metrics": {
            "files": {
                "total": snapshot.total_files,
                "needing_attention": sum(
                    1
                    for grade, count in snapshot.grade_distribution.items()
                    if grade in ["D", "F"]
                    for _ in range(count)
                ),
            },
            "lines": {
                "total": snapshot.total_lines,
            },
            "functions": {
                "total": snapshot.total_functions,
            },
            "classes": {
                "total": snapshot.total_classes,
            },
            "complexity": {
                "average": round(snapshot.avg_complexity, 2),
                "maximum": snapshot.max_complexity,
                "total": snapshot.total_complexity,
                "grade_distribution": snapshot.grade_distribution,
            },
            "code_smells": {
                "total": snapshot.total_smells,
            },
            "health": {
                "average_score": round(snapshot.avg_health_score, 2),
            },
        },
        "metadata": {
            "git_commit": snapshot.git_commit,
            "git_branch": snapshot.git_branch,
            "tool_version": snapshot.tool_version,
        },
    }

    print_json(output)


def _print_metrics_summary(snapshot) -> None:
    """Print formatted metrics summary using Rich.

    Args:
        snapshot: ProjectSnapshot from metrics store
    """
    # Header panel with overall stats
    console.print(
        Panel.fit(
            f"[bold]Project Metrics Summary[/bold]\n"
            f"Files: {snapshot.total_files} | "
            f"Functions: {snapshot.total_functions} | "
            f"Classes: {snapshot.total_classes} | "
            f"Lines: {snapshot.total_lines:,}\n"
            f"Analyzed: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            title="📊 mcp-vector-search",
            border_style="blue",
        )
    )
    console.print()

    # Complexity metrics table
    complexity_table = Table(title="Complexity Metrics", show_header=True)
    complexity_table.add_column("Metric", style="cyan", no_wrap=True)
    complexity_table.add_column("Average", justify="right")
    complexity_table.add_column("Maximum", justify="right")
    complexity_table.add_column("Total", justify="right")
    complexity_table.add_column("Status", justify="center")

    complexity_table.add_row(
        "Cognitive Complexity",
        f"{snapshot.avg_complexity:.1f}",
        f"{snapshot.max_complexity}",
        f"{snapshot.total_complexity}",
        _status_indicator(snapshot.avg_complexity, 10, 20),
    )

    console.print(complexity_table)
    console.print()

    # Grade distribution table
    grade_table = Table(title="Complexity Grade Distribution", show_header=True)
    grade_table.add_column("Grade", style="cyan", no_wrap=True)
    grade_table.add_column("Count", justify="right")
    grade_table.add_column("Percentage", justify="right")
    grade_table.add_column("Description")

    total_chunks = sum(snapshot.grade_distribution.values())
    grade_descriptions = {
        "A": "Excellent (0-5)",
        "B": "Good (6-10)",
        "C": "Acceptable (11-20)",
        "D": "Needs Improvement (21-30)",
        "F": "Refactor Recommended (31+)",
    }

    for grade in ["A", "B", "C", "D", "F"]:
        count = snapshot.grade_distribution.get(grade, 0)
        percentage = (count / total_chunks * 100) if total_chunks > 0 else 0

        # Color code the grade
        grade_color = {
            "A": "green",
            "B": "blue",
            "C": "yellow",
            "D": "orange1",
            "F": "red",
        }.get(grade, "white")

        grade_table.add_row(
            f"[{grade_color}]{grade}[/{grade_color}]",
            str(count),
            f"{percentage:.1f}%",
            grade_descriptions[grade],
        )

    console.print(grade_table)
    console.print()

    # Code smells summary
    if snapshot.total_smells > 0:
        console.print(
            f"[yellow]Code Smells:[/yellow] {snapshot.total_smells} issues detected"
        )
        console.print()

    # Health score
    health_color = (
        "green"
        if snapshot.avg_health_score >= 0.8
        else "yellow"
        if snapshot.avg_health_score >= 0.6
        else "red"
    )
    console.print(
        f"[bold]Health Score:[/bold] [{health_color}]{snapshot.avg_health_score:.2f}[/{health_color}] / 1.00"
    )
    console.print()

    # Git metadata (if available)
    if snapshot.git_commit or snapshot.git_branch:
        metadata_parts = []
        if snapshot.git_branch:
            metadata_parts.append(f"Branch: {snapshot.git_branch}")
        if snapshot.git_commit:
            metadata_parts.append(f"Commit: {snapshot.git_commit[:8]}")
        if snapshot.tool_version:
            metadata_parts.append(f"Version: {snapshot.tool_version}")

        console.print(f"[dim]{' | '.join(metadata_parts)}[/dim]")
        console.print()

    # Files needing attention
    files_needing_attention = snapshot.grade_distribution.get(
        "D", 0
    ) + snapshot.grade_distribution.get("F", 0)
    if files_needing_attention > 0:
        console.print(
            f"[yellow]⚠️  {files_needing_attention} code chunks need attention (grades D or F)[/yellow]"
        )
        console.print()


def _status_indicator(
    value: float, warning_threshold: float, error_threshold: float
) -> str:
    """Return colored status indicator based on thresholds.

    Args:
        value: Value to check
        warning_threshold: Warning threshold (yellow)
        error_threshold: Error threshold (red)

    Returns:
        Colored status indicator (green, yellow, or red dot)
    """
    if value < warning_threshold:
        return "[green]●[/green]"
    elif value < error_threshold:
        return "[yellow]●[/yellow]"
    else:
        return "[red]●[/red]"


def check_dependencies() -> bool:
    """Check if all required dependencies are available.

    Returns:
        bool: True if all dependencies are available, False otherwise.
    """
    dependencies = [
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("tree_sitter", "Tree-sitter"),
        ("tree_sitter_language_pack", "Tree-sitter Languages"),
        ("typer", "Typer"),
        ("rich", "Rich"),
        ("pydantic", "Pydantic"),
        ("watchdog", "Watchdog"),
        ("loguru", "Loguru"),
    ]

    all_available = True
    missing_deps = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_dependency_status(display_name, True)
        except ImportError:
            print_dependency_status(display_name, False)
            all_available = False
            missing_deps.append(module_name)

    # Print helpful installation instructions if dependencies are missing
    if not all_available:
        console.print()
        console.print("[yellow]💡 Installation Help:[/yellow]")
        console.print()
        console.print("[bold]To install missing dependencies:[/bold]")
        console.print()
        console.print("  [cyan]# Using pip (recommended):[/cyan]")
        console.print("  pip install mcp-vector-search")
        console.print()
        console.print("  [cyan]# Using uv (faster):[/cyan]")
        console.print("  uv pip install mcp-vector-search")
        console.print()
        console.print("  [cyan]# Install from source:[/cyan]")
        console.print("  pip install -e .")
        console.print()

        if "tree_sitter_language_pack" in missing_deps:
            console.print(
                "[yellow]⚠️  Tree-sitter Language Pack Issue Detected[/yellow]"
            )
            console.print()
            console.print("If you installed via Homebrew, you may need to reinstall:")
            console.print("  brew reinstall mcp-vector-search")
            console.print()
            console.print("Or install the dependency manually:")
            console.print("  pip install tree-sitter-language-pack")
            console.print()

    return all_available


if __name__ == "__main__":
    status_app()
