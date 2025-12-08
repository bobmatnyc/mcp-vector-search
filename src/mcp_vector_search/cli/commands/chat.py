"""Chat command for LLM-powered intelligent code search."""

import asyncio
from pathlib import Path

import typer
from loguru import logger

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError, SearchError
from ...core.llm_client import LLMClient
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)

# Create chat subcommand app with "did you mean" functionality
chat_app = create_enhanced_typer(
    help="🤖 LLM-powered intelligent code search",
    invoke_without_command=True,
)


@chat_app.callback(invoke_without_command=True)
def chat_main(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Natural language query about your code",
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
        5,
        "--limit",
        "-l",
        help="Maximum number of results to return",
        min=1,
        max=20,
        rich_help_panel="📊 Result Options",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (defaults based on provider: gpt-4o-mini for OpenAI, claude-3-haiku for OpenRouter)",
        rich_help_panel="🤖 LLM Options",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="LLM provider to use: 'openai' or 'openrouter' (auto-detect if not specified)",
        rich_help_panel="🤖 LLM Options",
    ),
    timeout: float | None = typer.Option(
        30.0,
        "--timeout",
        help="API timeout in seconds",
        min=5.0,
        max=120.0,
        rich_help_panel="🤖 LLM Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
        rich_help_panel="📊 Result Options",
    ),
) -> None:
    """🤖 Ask questions about your code in natural language.

    Uses LLM (OpenAI or OpenRouter) to intelligently search your codebase and answer
    questions like "where is X defined?", "how does Y work?", etc.

    [bold cyan]Setup:[/bold cyan]

    [green]Option A - OpenAI (recommended):[/green]
        $ export OPENAI_API_KEY="your-key-here"
        Get a key at: [cyan]https://platform.openai.com/api-keys[/cyan]

    [green]Option B - OpenRouter:[/green]
        $ export OPENROUTER_API_KEY="your-key-here"
        Get a key at: [cyan]https://openrouter.ai/keys[/cyan]

    [dim]Provider is auto-detected. OpenAI is preferred if both keys are set.[/dim]

    [bold cyan]Examples:[/bold cyan]

    [green]Ask where a parameter is set:[/green]
        $ mcp-vector-search chat "where is similarity_threshold set?"

    [green]Ask how something works:[/green]
        $ mcp-vector-search chat "how does the indexing process work?"

    [green]Find implementation details:[/green]
        $ mcp-vector-search chat "show me the search ranking algorithm"

    [green]Force specific provider:[/green]
        $ mcp-vector-search chat "question" --provider openai
        $ mcp-vector-search chat "question" --provider openrouter

    [green]Use custom model:[/green]
        $ mcp-vector-search chat "question" --model gpt-4o
        $ mcp-vector-search chat "question" --model anthropic/claude-3.5-sonnet

    [bold cyan]Advanced:[/bold cyan]

    [green]Limit results:[/green]
        $ mcp-vector-search chat "find auth code" --limit 3

    [green]Custom timeout:[/green]
        $ mcp-vector-search chat "complex question" --timeout 60

    [dim]💡 Tip: More specific questions get better answers. The LLM generates multiple
    search queries and analyzes results to find the most relevant code.[/dim]
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

        # Validate provider if specified
        if provider and provider not in ("openai", "openrouter"):
            print_error(
                f"Invalid provider: {provider}. Must be 'openai' or 'openrouter'"
            )
            raise typer.Exit(1)

        # Run the chat search
        asyncio.run(
            run_chat_search(
                project_root=project_root,
                query=query,
                limit=limit,
                model=model,
                provider=provider,
                timeout=timeout,
                json_output=json_output,
            )
        )

    except Exception as e:
        logger.error(f"Chat search failed: {e}")
        print_error(f"Chat search failed: {e}")
        raise typer.Exit(1)


async def run_chat_search(
    project_root: Path,
    query: str,
    limit: int = 5,
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 30.0,
    json_output: bool = False,
) -> None:
    """Run LLM-powered chat search.

    Implementation Flow:
    1. Initialize LLM client and validate API key
    2. Generate 2-3 targeted search queries from natural language
    3. Execute each search query against vector database
    4. Have LLM analyze all results and select most relevant ones
    5. Display results with explanations

    Args:
        project_root: Project root directory
        query: Natural language query from user
        limit: Maximum number of results to return
        model: Model to use (optional, defaults based on provider)
        provider: LLM provider ('openai' or 'openrouter', auto-detect if None)
        timeout: API timeout in seconds
        json_output: Whether to output JSON format
    """
    # Check for API keys (environment variable or config file)
    from ...core.config_utils import (
        get_config_file_path,
        get_openai_api_key,
        get_openrouter_api_key,
        get_preferred_llm_provider,
    )

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Determine which provider to use
    if provider:
        # Explicit provider specified
        if provider == "openai" and not openai_key:
            print_error("OpenAI API key not found.")
            print_info("\n[bold]To use OpenAI:[/bold]")
            print_info(
                "1. Get an API key from [cyan]https://platform.openai.com/api-keys[/cyan]"
            )
            print_info("2. Set environment variable:")
            print_info("   [yellow]export OPENAI_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("Or run: [cyan]mcp-vector-search setup[/cyan]")
            raise typer.Exit(1)
        elif provider == "openrouter" and not openrouter_key:
            print_error("OpenRouter API key not found.")
            print_info("\n[bold]To use OpenRouter:[/bold]")
            print_info("1. Get an API key from [cyan]https://openrouter.ai/keys[/cyan]")
            print_info("2. Set environment variable:")
            print_info("   [yellow]export OPENROUTER_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("Or run: [cyan]mcp-vector-search setup[/cyan]")
            raise typer.Exit(1)
    else:
        # Auto-detect provider
        preferred_provider = get_preferred_llm_provider(config_dir)

        if preferred_provider == "openai" and openai_key:
            provider = "openai"
        elif preferred_provider == "openrouter" and openrouter_key:
            provider = "openrouter"
        elif openai_key:
            provider = "openai"
        elif openrouter_key:
            provider = "openrouter"
        else:
            print_error("No LLM API key found.")
            print_info("\n[bold]To use the chat command, set up an API key:[/bold]")
            print_info("")
            print_info("[cyan]Option A - OpenAI (recommended):[/cyan]")
            print_info(
                "1. Get a key from [cyan]https://platform.openai.com/api-keys[/cyan]"
            )
            print_info("2. [yellow]export OPENAI_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("[cyan]Option B - OpenRouter:[/cyan]")
            print_info("1. Get a key from [cyan]https://openrouter.ai/keys[/cyan]")
            print_info("2. [yellow]export OPENROUTER_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("Or run: [cyan]mcp-vector-search setup[/cyan]")
            config_path = get_config_file_path(config_dir)
            print_info(f"\n[dim]Config file location: {config_path}[/dim]\n")
            raise typer.Exit(1)

    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Initialize LLM client
    try:
        llm_client = LLMClient(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            model=model,
            provider=provider,
            timeout=timeout,
        )
        provider_display = llm_client.provider.capitalize()
        print_success(f"Connected to {provider_display}: {llm_client.model}")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Step 1: Generate search queries from natural language
    console.print(f"\n[cyan]💭 Analyzing query:[/cyan] [white]{query}[/white]")

    try:
        search_queries = await llm_client.generate_search_queries(query, limit=3)

        if not search_queries:
            print_error("Failed to generate search queries from your question.")
            raise typer.Exit(1)

        console.print(
            f"\n[cyan]🔍 Generated {len(search_queries)} search queries:[/cyan]"
        )
        for i, sq in enumerate(search_queries, 1):
            console.print(f"  {i}. [yellow]{sq}[/yellow]")

    except SearchError as e:
        print_error(f"Failed to generate queries: {e}")
        raise typer.Exit(1)

    # Step 2: Execute each search query
    console.print("\n[cyan]🔎 Searching codebase...[/cyan]")

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

    # Execute all searches
    search_results = {}
    total_results = 0

    try:
        async with database:
            for search_query in search_queries:
                results = await search_engine.search(
                    query=search_query,
                    limit=limit * 2,  # Get more results for LLM to analyze
                    similarity_threshold=config.similarity_threshold,
                    include_context=True,
                )
                search_results[search_query] = results
                total_results += len(results)

                console.print(
                    f"  • [yellow]{search_query}[/yellow]: {len(results)} results"
                )

    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)

    if total_results == 0:
        print_warning("\n⚠️  No results found for any search query.")
        print_info("\n[bold]Suggestions:[/bold]")
        print_info("  • Try rephrasing your question")
        print_info("  • Use more general terms")
        print_info(
            "  • Check if relevant files are indexed with [cyan]mcp-vector-search status[/cyan]"
        )
        raise typer.Exit(0)

    # Step 3: Have LLM analyze and rank results
    console.print(f"\n[cyan]🤖 Analyzing {total_results} results...[/cyan]")

    try:
        ranked_results = await llm_client.analyze_and_rank_results(
            original_query=query,
            search_results=search_results,
            top_n=limit,
        )

        if not ranked_results:
            print_warning("\n⚠️  LLM could not identify relevant results.")
            raise typer.Exit(0)

    except SearchError as e:
        print_error(f"Result analysis failed: {e}")
        # Fallback: show raw search results
        print_warning("\nShowing raw search results instead...")
        await _show_fallback_results(search_results, limit)
        raise typer.Exit(1)

    # Step 4: Display results with explanations
    if json_output:
        await _display_json_results(ranked_results)
    else:
        await _display_rich_results(ranked_results, query)


async def _display_rich_results(
    ranked_results: list[dict],
    original_query: str,
) -> None:
    """Display results in rich formatted output.

    Args:
        ranked_results: List of ranked results with explanations
        original_query: Original user query
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    console.print(
        f"\n[bold cyan]🎯 Top Results for:[/bold cyan] [white]{original_query}[/white]\n"
    )

    for i, item in enumerate(ranked_results, 1):
        result = item["result"]
        relevance = item["relevance"]
        explanation = item["explanation"]
        query = item["query"]

        # Determine relevance emoji and color
        if relevance == "High":
            relevance_emoji = "🟢"
            relevance_color = "green"
        elif relevance == "Medium":
            relevance_emoji = "🟡"
            relevance_color = "yellow"
        else:
            relevance_emoji = "🔴"
            relevance_color = "red"

        # Header with result number and file
        console.print(f"[bold]📍 Result {i} of {len(ranked_results)}[/bold]")
        console.print(
            f"[cyan]📂 {result.file_path.relative_to(result.file_path.parent.parent)}[/cyan]"
        )

        # Relevance and explanation
        console.print(
            f"\n{relevance_emoji} [bold {relevance_color}]Relevance: {relevance}[/bold {relevance_color}]"
        )
        console.print(f"[dim]Search query: {query}[/dim]")
        console.print(f"\n💡 [italic]{explanation}[/italic]\n")

        # Code snippet with syntax highlighting
        file_ext = result.file_path.suffix.lstrip(".")
        code_syntax = Syntax(
            result.content,
            lexer=file_ext or "python",
            theme="monokai",
            line_numbers=True,
            start_line=result.start_line,
        )

        panel = Panel(
            code_syntax,
            title=f"[bold]{result.function_name or result.class_name or 'Code'}[/bold]",
            border_style="cyan",
        )
        console.print(panel)

        # Metadata
        metadata = []
        if result.function_name:
            metadata.append(f"Function: [cyan]{result.function_name}[/cyan]")
        if result.class_name:
            metadata.append(f"Class: [cyan]{result.class_name}[/cyan]")
        metadata.append(f"Lines: [cyan]{result.start_line}-{result.end_line}[/cyan]")
        metadata.append(f"Similarity: [cyan]{result.similarity_score:.3f}[/cyan]")

        console.print("[dim]" + " | ".join(metadata) + "[/dim]")
        console.print()  # Blank line between results

    # Footer with tips
    console.print("[dim]─" * 80 + "[/dim]")
    console.print(
        "\n[dim]💡 Tip: Try different phrasings or add more specific terms for better results[/dim]"
    )


async def _display_json_results(ranked_results: list[dict]) -> None:
    """Display results in JSON format.

    Args:
        ranked_results: List of ranked results with explanations
    """
    from ..output import print_json

    json_data = []
    for item in ranked_results:
        result = item["result"]
        json_data.append(
            {
                "file": str(result.file_path),
                "start_line": result.start_line,
                "end_line": result.end_line,
                "function_name": result.function_name,
                "class_name": result.class_name,
                "content": result.content,
                "similarity_score": result.similarity_score,
                "relevance": item["relevance"],
                "explanation": item["explanation"],
                "search_query": item["query"],
            }
        )

    print_json(json_data, title="Chat Search Results")


async def _show_fallback_results(
    search_results: dict[str, list],
    limit: int,
) -> None:
    """Show fallback results when LLM analysis fails.

    Args:
        search_results: Dictionary of search queries to results
        limit: Number of results to show
    """
    from ..output import print_search_results

    # Flatten and deduplicate results
    all_results = []
    seen_files = set()

    for results in search_results.values():
        for result in results:
            file_key = (result.file_path, result.start_line)
            if file_key not in seen_files:
                all_results.append(result)
                seen_files.add(file_key)

    # Sort by similarity score
    all_results.sort(key=lambda r: r.similarity_score, reverse=True)

    # Show top N
    print_search_results(
        results=all_results[:limit],
        query="Combined search results",
        show_content=True,
    )


if __name__ == "__main__":
    chat_app()
