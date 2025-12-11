"""Search command for MCP Vector Search CLI."""

import asyncio
import os
from fnmatch import fnmatch
from pathlib import Path

import typer
from loguru import logger

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.indexer import SemanticIndexer
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import (
    print_error,
    print_info,
    print_search_results,
    print_tip,
)

# Create search subcommand app with "did you mean" functionality
search_app = create_enhanced_typer(
    help="🔍 Search code semantically",
    invoke_without_command=True,
)


# Define search_main as the callback for the search command
# This makes `mcp-vector-search search "query"` work as main search
# and `mcp-vector-search search SUBCOMMAND` work for subcommands


@search_app.callback(invoke_without_command=True)
def search_main(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None, help="Search query or file path (for --similar)"
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
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of results to return",
        min=1,
        max=100,
        rich_help_panel="📊 Result Options",
    ),
    files: str | None = typer.Option(
        None,
        "--files",
        "-f",
        help="Filter by file glob patterns (e.g., '*.py', 'src/*.js', 'tests/*.ts'). Matches basename or relative path.",
        rich_help_panel="🔍 Filters",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Filter by programming language (python, javascript, typescript)",
        rich_help_panel="🔍 Filters",
    ),
    function_name: str | None = typer.Option(
        None,
        "--function",
        help="Filter by function name",
        rich_help_panel="🔍 Filters",
    ),
    class_name: str | None = typer.Option(
        None,
        "--class",
        help="Filter by class name",
        rich_help_panel="🔍 Filters",
    ),
    similarity_threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Minimum similarity threshold (0.0 to 1.0)",
        min=0.0,
        max=1.0,
        rich_help_panel="🎯 Search Options",
    ),
    similar: bool = typer.Option(
        False,
        "--similar",
        help="Find code similar to the query (treats query as file path)",
        rich_help_panel="🎯 Search Options",
    ),
    context: bool = typer.Option(
        False,
        "--context",
        help="Search for code based on contextual description",
        rich_help_panel="🎯 Search Options",
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        help="Focus areas for context search (comma-separated)",
        rich_help_panel="🎯 Search Options",
    ),
    no_content: bool = typer.Option(
        False,
        "--no-content",
        help="Don't show code content in results",
        rich_help_panel="📊 Result Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
        rich_help_panel="📊 Result Options",
    ),
    export_format: str | None = typer.Option(
        None,
        "--export",
        help="Export results to file (json, csv, markdown, summary)",
        rich_help_panel="💾 Export Options",
    ),
    export_path: Path | None = typer.Option(
        None,
        "--export-path",
        help="Custom export file path",
        rich_help_panel="💾 Export Options",
    ),
    max_complexity: int | None = typer.Option(
        None,
        "--max-complexity",
        help="Filter results with cognitive complexity greater than N",
        min=1,
        rich_help_panel="🎯 Quality Filters",
    ),
    no_smells: bool = typer.Option(
        False,
        "--no-smells",
        help="Exclude results with code smells",
        rich_help_panel="🎯 Quality Filters",
    ),
    grade: str | None = typer.Option(
        None,
        "--grade",
        help="Filter by complexity grade (e.g., 'A,B,C' or 'A-C')",
        rich_help_panel="🎯 Quality Filters",
    ),
    min_quality: int | None = typer.Option(
        None,
        "--min-quality",
        help="Filter by minimum quality score (0-100)",
        min=0,
        max=100,
        rich_help_panel="🎯 Quality Filters",
    ),
    quality_weight: float = typer.Option(
        0.3,
        "--quality-weight",
        help="Weight for quality ranking (0.0=pure relevance, 1.0=pure quality, default=0.3)",
        min=0.0,
        max=1.0,
        rich_help_panel="🎯 Quality Filters",
    ),
) -> None:
    """🔍 Search your codebase semantically.

    Performs vector similarity search across your indexed code to find relevant
    functions, classes, and patterns based on semantic meaning, not just keywords.

    [bold cyan]Basic Search Examples:[/bold cyan]

    [green]Simple semantic search:[/green]
        $ mcp-vector-search search "authentication middleware"

    [green]Search with language filter:[/green]
        $ mcp-vector-search search "database connection" --language python

    [green]Limit results:[/green]
        $ mcp-vector-search search "error handling" --limit 5

    [bold cyan]Advanced Search:[/bold cyan]

    [green]Filter by file pattern (glob):[/green]
        $ mcp-vector-search search "validation" --files "*.py"
        $ mcp-vector-search search "component" --files "src/*.tsx"
        $ mcp-vector-search search "test utils" --files "tests/*.ts"

    [green]Find similar code:[/green]
        $ mcp-vector-search search "src/auth.py" --similar

    [green]Context-based search:[/green]
        $ mcp-vector-search search "implement rate limiting" --context --focus security

    [bold cyan]Quality Filters:[/bold cyan]

    [green]Filter by complexity:[/green]
        $ mcp-vector-search search "authentication" --max-complexity 15

    [green]Exclude code smells:[/green]
        $ mcp-vector-search search "login" --no-smells

    [green]Filter by grade:[/green]
        $ mcp-vector-search search "api" --grade A,B

    [green]Minimum quality score:[/green]
        $ mcp-vector-search search "handler" --min-quality 80

    [green]Quality-aware ranking:[/green]
        $ mcp-vector-search search "auth" --quality-weight 0.5  # Balance relevance and quality
        $ mcp-vector-search search "api" --quality-weight 0.0   # Pure semantic search
        $ mcp-vector-search search "util" --quality-weight 1.0  # Pure quality ranking

    [bold cyan]Export Results:[/bold cyan]

    [green]Export to JSON:[/green]
        $ mcp-vector-search search "api endpoints" --export json

    [green]Export to markdown:[/green]
        $ mcp-vector-search search "utils" --export markdown

    [dim]💡 Tip: Use quotes for multi-word queries. Adjust --threshold for more/fewer results.[/dim]
    """
    # If no query provided and no subcommand invoked, exit (show help)
    if query is None:
        if ctx.invoked_subcommand is None:
            # No query and no subcommand - show help
            raise typer.Exit()
        else:
            # A subcommand was invoked - let it handle the request
            return

    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()

        # Validate mutually exclusive options
        if similar and context:
            print_error("Cannot use both --similar and --context flags together")
            raise typer.Exit(1)

        # Route to appropriate search function
        if similar:
            # Similar search - treat query as file path
            file_path = Path(query)
            if not file_path.exists():
                print_error(f"File not found: {query}")
                raise typer.Exit(1)

            asyncio.run(
                run_similar_search(
                    project_root=project_root,
                    file_path=file_path,
                    function_name=function_name,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    json_output=json_output,
                )
            )
        elif context:
            # Context search
            focus_areas = None
            if focus:
                focus_areas = [area.strip() for area in focus.split(",")]

            asyncio.run(
                run_context_search(
                    project_root=project_root,
                    description=query,
                    focus_areas=focus_areas,
                    limit=limit,
                    json_output=json_output,
                )
            )
        else:
            # Default semantic search
            asyncio.run(
                run_search(
                    project_root=project_root,
                    query=query,
                    limit=limit,
                    files=files,
                    language=language,
                    function_name=function_name,
                    class_name=class_name,
                    similarity_threshold=similarity_threshold,
                    show_content=not no_content,
                    json_output=json_output,
                    export_format=export_format,
                    export_path=export_path,
                    max_complexity=max_complexity,
                    no_smells=no_smells,
                    grade=grade,
                    min_quality=min_quality,
                    quality_weight=quality_weight,
                )
            )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


def _parse_grade_filter(grade_str: str) -> set[str]:
    """Parse grade filter string into set of allowed grades.

    Supports formats:
    - Comma-separated: "A,B,C"
    - Range: "A-C" (expands to A, B, C)
    - Mixed: "A,C-D" (expands to A, C, D)

    Args:
        grade_str: Grade filter string

    Returns:
        Set of allowed grade letters
    """
    allowed_grades = set()
    grade_order = ["A", "B", "C", "D", "F"]

    # Split by comma
    parts = [part.strip().upper() for part in grade_str.split(",")]

    for part in parts:
        if "-" in part:
            # Range format (e.g., "A-C")
            start, end = part.split("-", 1)
            start = start.strip()
            end = end.strip()

            if start in grade_order and end in grade_order:
                start_idx = grade_order.index(start)
                end_idx = grade_order.index(end)

                # Handle reverse ranges (C-A becomes A-C)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

                # Add all grades in range
                for grade in grade_order[start_idx : end_idx + 1]:
                    allowed_grades.add(grade)
        else:
            # Single grade
            if part in grade_order:
                allowed_grades.add(part)

    return allowed_grades


async def run_search(
    project_root: Path,
    query: str,
    limit: int = 10,
    files: str | None = None,
    language: str | None = None,
    function_name: str | None = None,
    class_name: str | None = None,
    similarity_threshold: float | None = None,
    show_content: bool = True,
    json_output: bool = False,
    export_format: str | None = None,
    export_path: Path | None = None,
    max_complexity: int | None = None,
    no_smells: bool = False,
    grade: str | None = None,
    min_quality: int | None = None,
    quality_weight: float = 0.3,
) -> None:
    """Run semantic search with optional quality filters and quality-aware ranking."""
    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Setup database and search engine
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Create indexer for version check
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
    )

    # Check if reindex is needed due to version upgrade
    if config.auto_reindex_on_upgrade and indexer.needs_reindex_for_version():
        from ..output import console

        index_version = indexer.get_index_version()
        from ... import __version__

        if index_version:
            console.print(
                f"[yellow]⚠️  Index created with version {index_version} (current: {__version__})[/yellow]"
            )
        else:
            console.print(
                "[yellow]⚠️  Index version not found (legacy format detected)[/yellow]"
            )

        console.print(
            "[yellow]   Reindexing to take advantage of improvements...[/yellow]"
        )

        # Auto-reindex with progress
        try:
            indexed_count = await indexer.index_project(
                force_reindex=True, show_progress=False
            )
            console.print(
                f"[green]✓ Index updated to version {__version__} ({indexed_count} files reindexed)[/green]\n"
            )
        except Exception as e:
            console.print(f"[red]✗ Reindexing failed: {e}[/red]")
            console.print(
                "[yellow]  Continuing with existing index (may have outdated patterns)[/yellow]\n"
            )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=similarity_threshold or config.similarity_threshold,
    )

    # Build filters (exclude file_path - will be handled with post-filtering)
    filters = {}
    if language:
        filters["language"] = language
    if function_name:
        filters["function_name"] = function_name
    if class_name:
        filters["class_name"] = class_name

    try:
        async with database:
            results = await search_engine.search(
                query=query,
                limit=limit,
                filters=filters if filters else None,
                similarity_threshold=similarity_threshold,
                include_context=show_content,
            )

            # Post-filter results by file pattern if specified
            if files and results:
                filtered_results = []
                for result in results:
                    # Get relative path from project root
                    try:
                        rel_path = str(result.file_path.relative_to(project_root))
                    except ValueError:
                        # If file is outside project root, use absolute path
                        rel_path = str(result.file_path)

                    # Match against glob pattern (both full path and basename)
                    if fnmatch(rel_path, files) or fnmatch(
                        os.path.basename(rel_path), files
                    ):
                        filtered_results.append(result)

                results = filtered_results
                logger.debug(
                    f"File pattern '{files}' filtered results to {len(results)} matches"
                )

            # Apply quality filters if specified
            if any([max_complexity, no_smells, grade, min_quality]) and results:
                filtered_results = []
                for result in results:
                    # Parse quality metrics from result metadata
                    cognitive_complexity = getattr(result, "cognitive_complexity", None)
                    complexity_grade = getattr(result, "complexity_grade", None)
                    smell_count = getattr(result, "smell_count", None)
                    quality_score = getattr(result, "quality_score", None)

                    # Filter by max complexity
                    if max_complexity is not None and cognitive_complexity is not None:
                        if cognitive_complexity > max_complexity:
                            continue

                    # Filter by code smells
                    if no_smells and smell_count is not None:
                        if smell_count > 0:
                            continue

                    # Filter by grade
                    if grade and complexity_grade:
                        allowed_grades = _parse_grade_filter(grade)
                        if complexity_grade not in allowed_grades:
                            continue

                    # Filter by minimum quality score
                    if min_quality is not None and quality_score is not None:
                        if quality_score < min_quality:
                            continue

                    filtered_results.append(result)

                initial_count = len(results)
                results = filtered_results
                logger.debug(
                    f"Quality filters reduced results from {initial_count} to {len(results)}"
                )

            # Apply quality-aware ranking if quality_weight > 0 and results have quality metrics
            if quality_weight > 0.0 and results:
                # Calculate quality scores for results that don't have them
                for result in results:
                    if result.quality_score is None:
                        # Calculate quality score using the formula
                        calculated_score = result.calculate_quality_score()
                        if calculated_score is not None:
                            result.quality_score = calculated_score

                # Re-rank results based on combined score
                # Store original similarity score for display
                for result in results:
                    # Store original relevance score
                    if not hasattr(result, "_original_similarity"):
                        result._original_similarity = result.similarity_score

                    # Calculate combined score
                    if result.quality_score is not None:
                        # Normalize quality score to 0-1 range (it's 0-100)
                        normalized_quality = result.quality_score / 100.0

                        # Combined score: (1-W) × relevance + W × quality
                        combined_score = (
                            (1.0 - quality_weight) * result.similarity_score
                            + quality_weight * normalized_quality
                        )

                        # Update similarity_score with combined score for sorting
                        result.similarity_score = combined_score
                    # If no quality score, keep original similarity_score

                # Re-sort by combined score
                results.sort(key=lambda r: r.similarity_score, reverse=True)

                # Update ranks
                for i, result in enumerate(results):
                    result.rank = i + 1

                logger.debug(
                    f"Quality-aware ranking applied with weight {quality_weight:.2f}"
                )

            # Handle export if requested
            if export_format:
                from ..export import SearchResultExporter, get_export_path

                exporter = SearchResultExporter()

                # Determine export path
                if export_path:
                    output_path = export_path
                else:
                    output_path = get_export_path(export_format, query, project_root)

                # Export based on format
                success = False
                if export_format == "json":
                    success = exporter.export_to_json(results, output_path, query)
                elif export_format == "csv":
                    success = exporter.export_to_csv(results, output_path, query)
                elif export_format == "markdown":
                    success = exporter.export_to_markdown(
                        results, output_path, query, show_content
                    )
                elif export_format == "summary":
                    success = exporter.export_summary_table(results, output_path, query)
                else:
                    from ..output import print_error

                    print_error(f"Unsupported export format: {export_format}")

                if not success:
                    return

            # Save to search history
            from ..history import SearchHistory

            history_manager = SearchHistory(project_root)
            history_manager.add_search(
                query=query,
                results_count=len(results),
                filters=filters if filters else None,
            )

            # Display results
            if json_output:
                from ..output import print_json

                results_data = [result.to_dict() for result in results]
                print_json(results_data, title="Search Results")
            else:
                print_search_results(
                    results=results,
                    query=query,
                    show_content=show_content,
                    quality_weight=quality_weight,
                )

                # Add contextual tips based on results
                if results:
                    if len(results) >= limit:
                        print_tip(
                            f"More results may be available. Use [cyan]--limit {limit * 2}[/cyan] to see more."
                        )
                    if not export_format:
                        print_tip(
                            "Export results with [cyan]--export json[/cyan] or [cyan]--export markdown[/cyan]"
                        )
                else:
                    # No results - provide helpful suggestions
                    print_info("\n[bold]No results found. Try:[/bold]")
                    print_info("  • Use more general terms in your query")
                    print_info(
                        "  • Lower the similarity threshold with [cyan]--threshold 0.3[/cyan]"
                    )
                    print_info(
                        "  • Check if files are indexed with [cyan]mcp-vector-search status[/cyan]"
                    )

    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        raise


def search_similar_cmd(
    ctx: typer.Context,
    file_path: Path = typer.Argument(
        ...,
        help="Reference file path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
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
    ),
    function_name: str | None = typer.Option(
        None,
        "--function",
        "-f",
        help="Specific function name to find similar code for",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of results",
        min=1,
        max=100,
    ),
    similarity_threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Minimum similarity threshold",
        min=0.0,
        max=1.0,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
    ),
) -> None:
    """Find code similar to a specific file or function.

    Examples:
        mcp-vector-search search similar src/auth.py
        mcp-vector-search search similar src/utils.py --function validate_email
    """
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()

        asyncio.run(
            run_similar_search(
                project_root=project_root,
                file_path=file_path,
                function_name=function_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
                json_output=json_output,
            )
        )

    except Exception as e:
        logger.error(f"Similar search failed: {e}")
        print_error(f"Similar search failed: {e}")
        raise typer.Exit(1)


async def run_similar_search(
    project_root: Path,
    file_path: Path,
    function_name: str | None = None,
    limit: int = 10,
    similarity_threshold: float | None = None,
    json_output: bool = False,
) -> None:
    """Run similar code search."""
    project_manager = ProjectManager(project_root)
    config = project_manager.load_config()

    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=similarity_threshold or config.similarity_threshold,
    )

    async with database:
        results = await search_engine.search_similar(
            file_path=file_path,
            function_name=function_name,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )

        if json_output:
            from ..output import print_json

            results_data = [result.to_dict() for result in results]
            print_json(results_data, title="Similar Code Results")
        else:
            query_desc = f"{file_path}"
            if function_name:
                query_desc += f" → {function_name}()"

            print_search_results(
                results=results,
                query=f"Similar to: {query_desc}",
                show_content=True,
            )


def search_context_cmd(
    ctx: typer.Context,
    description: str = typer.Argument(..., help="Context description"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        help="Comma-separated focus areas (e.g., 'security,authentication')",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of results",
        min=1,
        max=100,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
    ),
) -> None:
    """Search for code based on contextual description.

    Examples:
        mcp-vector-search search context "implement rate limiting"
        mcp-vector-search search context "user authentication" --focus security,middleware
    """
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()

        focus_areas = None
        if focus:
            focus_areas = [area.strip() for area in focus.split(",")]

        asyncio.run(
            run_context_search(
                project_root=project_root,
                description=description,
                focus_areas=focus_areas,
                limit=limit,
                json_output=json_output,
            )
        )

    except Exception as e:
        logger.error(f"Context search failed: {e}")
        print_error(f"Context search failed: {e}")
        raise typer.Exit(1)


async def run_context_search(
    project_root: Path,
    description: str,
    focus_areas: list[str] | None = None,
    limit: int = 10,
    json_output: bool = False,
) -> None:
    """Run contextual search."""
    project_manager = ProjectManager(project_root)
    config = project_manager.load_config()

    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=config.similarity_threshold,
    )

    async with database:
        results = await search_engine.search_by_context(
            context_description=description,
            focus_areas=focus_areas,
            limit=limit,
        )

        if json_output:
            from ..output import print_json

            results_data = [result.to_dict() for result in results]
            print_json(results_data, title="Context Search Results")
        else:
            query_desc = description
            if focus_areas:
                query_desc += f" (focus: {', '.join(focus_areas)})"

            print_search_results(
                results=results,
                query=query_desc,
                show_content=True,
            )


# ============================================================================
# SEARCH SUBCOMMANDS
# ============================================================================


@search_app.command("interactive")
def interactive_search(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """🎯 Start an interactive search session.

    Provides a rich terminal interface for searching your codebase with real-time
    filtering, query refinement, and result navigation.

    Examples:
        mcp-vector-search search interactive
        mcp-vector-search search interactive --project-root /path/to/project
    """
    import asyncio

    from ..interactive import start_interactive_search
    from ..output import console

    root = project_root or ctx.obj.get("project_root") or Path.cwd()

    try:
        asyncio.run(start_interactive_search(root))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive search cancelled[/yellow]")
    except Exception as e:
        print_error(f"Interactive search failed: {e}")
        raise typer.Exit(1)


@search_app.command("history")
def show_history(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-l", help="Number of entries to show"),
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """📜 Show search history.

    Displays your recent search queries with timestamps and result counts.
    Use this to revisit previous searches or track your search patterns.

    Examples:
        mcp-vector-search search history
        mcp-vector-search search history --limit 50
    """
    from ..history import show_search_history

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    show_search_history(root, limit)


@search_app.command("favorites")
def show_favorites_cmd(
    ctx: typer.Context,
    action: str | None = typer.Argument(None, help="Action: list, add, remove"),
    query: str | None = typer.Argument(None, help="Query to add/remove"),
    description: str | None = typer.Option(
        None, "--desc", help="Description for favorite"
    ),
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """⭐ Manage favorite queries.

    List, add, or remove favorite search queries for quick access.

    Examples:
        mcp-vector-search search favorites                # List all favorites
        mcp-vector-search search favorites list           # List all favorites
        mcp-vector-search search favorites add "auth"     # Add favorite
        mcp-vector-search search favorites remove "auth"  # Remove favorite
    """
    from ..history import SearchHistory, show_favorites

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    history_manager = SearchHistory(root)

    # Default to list if no action provided
    if not action or action == "list":
        show_favorites(root)
    elif action == "add":
        if not query:
            print_error("Query is required for 'add' action")
            raise typer.Exit(1)
        history_manager.add_favorite(query, description)
    elif action == "remove":
        if not query:
            print_error("Query is required for 'remove' action")
            raise typer.Exit(1)
        history_manager.remove_favorite(query)
    else:
        print_error(f"Unknown action: {action}. Use: list, add, or remove")
        raise typer.Exit(1)


# Add main command to search_app (allows: mcp-vector-search search main "query")
search_app.command("main")(search_main)


if __name__ == "__main__":
    search_app()
