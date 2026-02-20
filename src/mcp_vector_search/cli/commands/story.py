"""Story command for MCP Vector Search CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...story import StoryGenerator

console = Console()

# Create story subcommand app
story_app = typer.Typer(help="📖 Generate development narrative from git history")


@story_app.callback(invoke_without_command=True)
def story_main(
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
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (defaults to project root)",
        rich_help_panel="📊 Output Options",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Output format: markdown, json, html, or all",
        rich_help_panel="📊 Output Options",
    ),
    max_commits: int = typer.Option(
        200,
        "--max-commits",
        help="Maximum commits to analyze",
        min=1,
        max=5000,
        rich_help_panel="🔍 Filters",
    ),
    max_issues: int = typer.Option(
        100,
        "--max-issues",
        help="Maximum issues to fetch from GitHub",
        min=0,
        max=500,
        rich_help_panel="🔍 Filters",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM synthesis (extraction + analysis only)",
        rich_help_panel="⚡ Performance Options",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="LLM model to use (e.g., 'gpt-4', 'claude-opus-4.5')",
        rich_help_panel="⚡ Performance Options",
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Start local HTTP server for HTML visualization",
        rich_help_panel="🌐 Server Options",
    ),
    port: int = typer.Option(
        8502,
        "--port",
        help="Port for HTTP server (default: 8502)",
        min=1024,
        max=65535,
        rich_help_panel="🌐 Server Options",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
        rich_help_panel="🔧 Global Options",
    ),
) -> None:
    """📖 Generate a development narrative from git history and semantic analysis.

    Creates a narrative story of your project's evolution by analyzing git commits,
    GitHub issues/PRs, and semantic code patterns. Outputs interactive visualizations
    showing contributor collaboration, timeline events, and code evolution.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Generate full story with all formats:[/green]
        $ mcp-vector-search story

    [green]Markdown only (fast, no LLM):[/green]
        $ mcp-vector-search story --no-llm --format markdown

    [green]HTML with local server:[/green]
        $ mcp-vector-search story --format html --serve

    [green]Analyze more commits:[/green]
        $ mcp-vector-search story --max-commits 500

    [bold cyan]Output Formats:[/bold cyan]

    [green]JSON (for programmatic access):[/green]
        $ mcp-vector-search story --format json

    [green]Markdown (human-readable report):[/green]
        $ mcp-vector-search story --format markdown

    [green]HTML (interactive visualization):[/green]
        $ mcp-vector-search story --format html

    [bold cyan]Advanced Examples:[/bold cyan]

    [green]Custom output directory:[/green]
        $ mcp-vector-search story --output ./docs/story

    [green]Use specific LLM model:[/green]
        $ mcp-vector-search story --model "gpt-4"

    [green]Fast mode (skip LLM narrative):[/green]
        $ mcp-vector-search story --no-llm --max-commits 100

    [dim]💡 Tip: Use --no-llm for faster generation without LLM-powered narrative synthesis.[/dim]
    """
    if ctx.invoked_subcommand is not None:
        return

    # Setup logging
    if verbose:
        logger.enable("mcp_vector_search")
        logger.info("Verbose logging enabled")

    # Use provided project_root or current working directory
    if project_root is None:
        project_root = Path.cwd()

    # Validate format
    valid_formats = ["json", "markdown", "html", "all"]
    if format.lower() not in valid_formats:
        console.print(
            f"[red]Error:[/red] Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}"
        )
        raise typer.Exit(1)

    try:
        # Run async story generation
        asyncio.run(
            run_story_generation(
                project_root=project_root,
                output_dir=output,
                format=format.lower(),
                max_commits=max_commits,
                max_issues=max_issues,
                use_llm=not no_llm,
                model=model,
                serve=serve,
                port=port,
                verbose=verbose,
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Story generation cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Story generation failed: {e}", exc_info=verbose)
        console.print(f"\n[red]Error:[/red] Story generation failed: {e}")
        raise typer.Exit(1)


async def run_story_generation(
    project_root: Path,
    output_dir: Path | None,
    format: str,
    max_commits: int,
    max_issues: int,
    use_llm: bool,
    model: str | None,
    serve: bool,
    port: int,
    verbose: bool,
) -> None:
    """Run the story generation pipeline with progress display.

    Args:
        project_root: Project root directory
        output_dir: Output directory (None = project root)
        format: Output format ('json', 'markdown', 'html', 'all')
        max_commits: Maximum commits to analyze
        max_issues: Maximum issues to fetch
        use_llm: Whether to use LLM for narrative synthesis
        model: LLM model name (None = default)
        serve: Whether to start HTTP server
        port: Port for HTTP server
        verbose: Verbose output
    """
    console.print("\n[bold blue]📖 Generating Development Story[/bold blue]\n")

    # Show configuration
    console.print(f"[dim]Project:[/dim] {project_root}")
    console.print(f"[dim]Max Commits:[/dim] {max_commits}")
    console.print(f"[dim]Max Issues:[/dim] {max_issues}")
    console.print(f"[dim]LLM Synthesis:[/dim] {'enabled' if use_llm else 'disabled'}")
    if model:
        console.print(f"[dim]LLM Model:[/dim] {model}")
    console.print(f"[dim]Output Format:[/dim] {format}\n")

    # Create generator
    generator = StoryGenerator(
        project_root=project_root,
        max_commits=max_commits,
        max_issues=max_issues,
        use_llm=use_llm,
        model=model,
    )

    # Run generation with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Phase 1: Extract
        task_extract = progress.add_task(
            "[cyan]Extracting git history and GitHub data...", total=None
        )
        try:
            story = await generator.generate()
            progress.update(
                task_extract, completed=True, description="[green]✓ Extraction complete"
            )
        except Exception:
            progress.update(
                task_extract, completed=True, description="[red]✗ Extraction failed"
            )
            raise

    # Print summary
    console.print("\n[bold green]✓ Story Generation Complete[/bold green]\n")
    console.print("[bold]Summary:[/bold]")
    console.print(f"  • Commits analyzed: {story.metadata.total_commits}")
    console.print(f"  • Contributors: {story.metadata.total_contributors}")
    console.print(f"  • Files tracked: {story.metadata.total_files}")
    console.print(f"  • Generation time: {story.metadata.generation_time_seconds:.1f}s")

    if story.metadata.has_semantic_analysis:
        console.print(f"  • Semantic clusters: {len(story.analysis.clusters)}")
        console.print(f"  • Tech stack items: {len(story.analysis.tech_stack)}")

    if story.metadata.has_llm_narrative:
        console.print(f"  • Narrative acts: {len(story.narrative.acts)}")

    # Render output
    console.print(f"\n[bold]Rendering {format} output...[/bold]")
    output_paths = generator.render(story, format=format, output_dir=output_dir)

    console.print("\n[bold green]✓ Files Generated:[/bold green]")
    for path in output_paths:
        size_kb = path.stat().st_size / 1024
        console.print(f"  • {path} ({size_kb:.1f} KB)")

    # Start server if requested
    if serve:
        # Find HTML file
        html_file = None
        for path in output_paths:
            if path.suffix == ".html":
                html_file = path
                break

        if not html_file:
            console.print(
                "\n[yellow]Warning:[/yellow] No HTML file generated. Cannot start server."
            )
            console.print("Use --format html or --format all to generate HTML output.")
            return

        console.print("\n[bold cyan]🌐 Starting HTTP Server[/bold cyan]")

        # Find free port if requested port is taken
        from ..visualize.server import find_free_port

        try:
            actual_port = find_free_port(port, port + 100)
            if actual_port != port:
                console.print(
                    f"[yellow]Port {port} in use, using {actual_port} instead[/yellow]"
                )
        except OSError:
            console.print("[red]Error:[/red] No free ports available")
            return

        # Create FastAPI app
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import FileResponse

        app = FastAPI(title="Story Viewer")

        @app.get("/")
        async def serve_html():
            return FileResponse(html_file)

        # Print server info
        console.print(
            f"\n[bold green]✓ Server started at:[/bold green] http://localhost:{actual_port}"
        )
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]\n")

        # Run server with graceful shutdown handling
        try:
            config = uvicorn.Config(
                app,
                host="0.0.0.0",  # nosec B104 - intentional for local dev server
                port=actual_port,
                log_level="error" if not verbose else "info",
            )
            server = uvicorn.Server(config)
            await server.serve()
        except KeyboardInterrupt:
            console.print("\n[green]✓ Server stopped[/green]")


if __name__ == "__main__":
    story_app()
