"""Search command for MCP Vector Search CLI."""

import asyncio
from pathlib import Path

import typer
from loguru import logger

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import (
    print_error,
    print_search_results,
)

# Create search subcommand app with "did you mean" functionality (kept for backward compatibility)
search_app = create_enhanced_typer(help="Search code semantically")


def search_main(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query"),
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
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of results to return",
        min=1,
        max=100,
    ),
    files: str | None = typer.Option(
        None,
        "--files",
        "-f",
        help="Filter by file patterns (e.g., '*.py' or 'src/*.js')",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Filter by programming language",
    ),
    function_name: str | None = typer.Option(
        None,
        "--function",
        help="Filter by function name",
    ),
    class_name: str | None = typer.Option(
        None,
        "--class",
        help="Filter by class name",
    ),
    similarity_threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Minimum similarity threshold (0.0 to 1.0)",
        min=0.0,
        max=1.0,
    ),
    similar: bool = typer.Option(
        False,
        "--similar",
        help="Find code similar to the query (treats query as file path)",
    ),
    context: bool = typer.Option(
        False,
        "--context",
        help="Search for code based on contextual description",
    ),
    focus: str | None = typer.Option(
        None,
        "--focus",
        help="Focus areas for context search (comma-separated)",
    ),
    no_content: bool = typer.Option(
        False,
        "--no-content",
        help="Don't show code content in results",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
    ),
    export_format: str | None = typer.Option(
        None,
        "--export",
        help="Export results to file (json, csv, markdown, summary)",
    ),
    export_path: Path | None = typer.Option(
        None,
        "--export-path",
        help="Custom export file path",
    ),
) -> None:
    """Search your codebase semantically.

    This command performs semantic search across your indexed codebase,
    finding code that is conceptually similar to your query even if it
    doesn't contain the exact keywords.

    Examples:
        mcp-vector-search search "authentication middleware"
        mcp-vector-search search "database connection" --language python
        mcp-vector-search search "error handling" --files "*.js" --limit 5
        mcp-vector-search search "user validation" --function validate --json
        mcp-vector-search search "src/auth.py" --similar
        mcp-vector-search search "implement rate limiting" --context
        mcp-vector-search search "user authentication" --context --focus security,middleware
    """
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
                )
            )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)


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
) -> None:
    """Run semantic search."""
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

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=similarity_threshold or config.similarity_threshold,
    )

    # Build filters
    filters = {}
    if language:
        filters["language"] = language
    if function_name:
        filters["function_name"] = function_name
    if class_name:
        filters["class_name"] = class_name
    if files:
        # Simple file pattern matching (could be enhanced)
        filters["file_path"] = files

    try:
        async with database:
            results = await search_engine.search(
                query=query,
                limit=limit,
                filters=filters if filters else None,
                similarity_threshold=similarity_threshold,
                include_context=show_content,
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


# Add commands to search_app for backward compatibility
search_app.command("main")(search_main)
search_app.command("similar")(search_similar_cmd)
search_app.command("context")(search_context_cmd)


if __name__ == "__main__":
    search_app()
