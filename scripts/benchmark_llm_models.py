#!/usr/bin/env python3
"""Benchmark LLM models for mcp-vector-search chat command.

This script tests various OpenRouter LLM models on the chat command to compare:
- Response quality
- Speed (latency)
- Token usage
- Cost

Usage:
    python scripts/benchmark_llm_models.py
    python scripts/benchmark_llm_models.py --models anthropic/claude-3-haiku openai/gpt-4o-mini
    python scripts/benchmark_llm_models.py --query "how does the indexer work?"
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.llm_client import LLMClient
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine

console = Console()

# OpenRouter pricing (per 1M tokens as of Dec 2024)
MODEL_PRICING = {
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "mistralai/mistral-large": {"input": 2.00, "output": 6.00},
}

# Default test queries
DEFAULT_TEST_QUERIES = [
    "where is similarity_threshold configured?",
    "how does the indexer handle TypeScript files?",
    "show me examples of error handling in the search module",
]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model: str
    query: str
    success: bool
    latency: float  # seconds
    input_tokens: int
    output_tokens: int
    cost: float  # USD
    num_queries_generated: int
    num_results_found: int
    num_ranked_results: int
    error: str | None = None


class EnhancedLLMClient(LLMClient):
    """LLM client that tracks token usage from API responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_usage: dict[str, int] = {}

    async def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make chat completion request and track usage."""
        response = await super()._chat_completion(messages)

        # Extract usage from response
        usage = response.get("usage", {})
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return response

    def get_last_usage(self) -> dict[str, int]:
        """Get token usage from last API call."""
        return self.last_usage


async def benchmark_model(
    model: str,
    query: str,
    project_root: Path,
    api_key: str,
    search_engine: SemanticSearchEngine,
    config: Any,
) -> BenchmarkResult:
    """Benchmark a single model on a single query.

    Args:
        model: Model identifier (e.g., "anthropic/claude-3-haiku")
        query: Natural language query to test
        project_root: Project root directory
        api_key: OpenRouter API key
        search_engine: Initialized search engine
        config: Project configuration

    Returns:
        BenchmarkResult with metrics
    """
    start_time = time.time()

    try:
        # Initialize enhanced LLM client
        llm_client = EnhancedLLMClient(api_key=api_key, model=model, timeout=30.0)

        # Step 1: Generate search queries
        search_queries = await llm_client.generate_search_queries(query, limit=3)
        query_gen_tokens = llm_client.get_last_usage()

        if not search_queries:
            return BenchmarkResult(
                model=model,
                query=query,
                success=False,
                latency=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                num_queries_generated=0,
                num_results_found=0,
                num_ranked_results=0,
                error="Failed to generate search queries",
            )

        # Step 2: Execute searches
        search_results = {}
        total_results = 0

        for search_query in search_queries:
            results = await search_engine.search(
                query=search_query,
                limit=10,
                similarity_threshold=config.similarity_threshold,
                include_context=True,
            )
            search_results[search_query] = results
            total_results += len(results)

        if total_results == 0:
            return BenchmarkResult(
                model=model,
                query=query,
                success=False,
                latency=time.time() - start_time,
                input_tokens=query_gen_tokens["prompt_tokens"],
                output_tokens=query_gen_tokens["completion_tokens"],
                cost=calculate_cost(
                    model,
                    query_gen_tokens["prompt_tokens"],
                    query_gen_tokens["completion_tokens"],
                ),
                num_queries_generated=len(search_queries),
                num_results_found=0,
                num_ranked_results=0,
                error="No results found",
            )

        # Step 3: Analyze and rank results
        ranked_results = await llm_client.analyze_and_rank_results(
            original_query=query,
            search_results=search_results,
            top_n=5,
        )
        analysis_tokens = llm_client.get_last_usage()

        # Calculate totals
        total_input_tokens = (
            query_gen_tokens["prompt_tokens"] + analysis_tokens["prompt_tokens"]
        )
        total_output_tokens = (
            query_gen_tokens["completion_tokens"] + analysis_tokens["completion_tokens"]
        )
        total_cost = calculate_cost(model, total_input_tokens, total_output_tokens)
        total_latency = time.time() - start_time

        return BenchmarkResult(
            model=model,
            query=query,
            success=True,
            latency=total_latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost=total_cost,
            num_queries_generated=len(search_queries),
            num_results_found=total_results,
            num_ranked_results=len(ranked_results),
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}"
        if e.response.status_code == 429:
            error_msg = "Rate limit exceeded"
        elif e.response.status_code == 401:
            error_msg = "Invalid API key"

        return BenchmarkResult(
            model=model,
            query=query,
            success=False,
            latency=time.time() - start_time,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            num_queries_generated=0,
            num_results_found=0,
            num_ranked_results=0,
            error=error_msg,
        )

    except Exception as e:
        logger.error(f"Benchmark failed for {model}: {e}")
        return BenchmarkResult(
            model=model,
            query=query,
            success=False,
            latency=time.time() - start_time,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            num_queries_generated=0,
            num_results_found=0,
            num_ranked_results=0,
            error=str(e),
        )


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on token usage.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        logger.warning(f"No pricing data for {model}, using $0")
        return 0.0

    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def rate_quality(result: BenchmarkResult) -> str:
    """Rate result quality with star rating.

    Args:
        result: Benchmark result

    Returns:
        Star rating string (e.g., "★★★★☆")
    """
    if not result.success:
        return "☆☆☆☆☆"

    # Quality factors:
    # - Did it return ranked results? (3 stars)
    # - Found reasonable number of results? (1 star)
    # - Generated multiple search queries? (1 star)

    stars = 0

    # Base: returned results
    if result.num_ranked_results > 0:
        stars += 3

    # Bonus: found good number of results
    if result.num_results_found >= 5:
        stars += 1

    # Bonus: generated multiple queries
    if result.num_queries_generated >= 2:
        stars += 1

    return "★" * stars + "☆" * (5 - stars)


def print_results_table(results: list[BenchmarkResult], query: str) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: List of benchmark results
        query: Query that was tested
    """
    table = Table(title=f'Query: "{query}"')

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Time(s)", justify="right", style="yellow")
    table.add_column("Input", justify="right", style="blue")
    table.add_column("Output", justify="right", style="blue")
    table.add_column("Cost($)", justify="right", style="green")
    table.add_column("Quality", justify="center")
    table.add_column("Status", style="dim")

    # Sort by latency (fastest first)
    sorted_results = sorted(results, key=lambda r: r.latency)

    for result in sorted_results:
        if result.success:
            status = (
                f"✓ {result.num_ranked_results} results"
                if result.num_ranked_results > 0
                else "✓ no results"
            )
            style = ""
        else:
            status = f"✗ {result.error}"
            style = "red"

        table.add_row(
            result.model.split("/")[-1],  # Show only model name
            f"{result.latency:.1f}s",
            str(result.input_tokens),
            str(result.output_tokens),
            f"${result.cost:.4f}",
            rate_quality(result),
            status,
            style=style,
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def print_summary(all_results: list[BenchmarkResult]) -> None:
    """Print benchmark summary with recommendations.

    Args:
        all_results: All benchmark results across queries
    """
    console.print("\n[bold cyan]═══ Benchmark Summary ═══[/bold cyan]\n")

    # Calculate aggregates
    successful_results = [r for r in all_results if r.success]

    if not successful_results:
        console.print("[red]✗ No successful results[/red]")
        return

    # Group by model
    by_model: dict[str, list[BenchmarkResult]] = {}
    for result in successful_results:
        if result.model not in by_model:
            by_model[result.model] = []
        by_model[result.model].append(result)

    # Calculate averages per model
    console.print("[bold]Performance by Model:[/bold]\n")

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Avg Time", justify="right", style="yellow")
    table.add_column("Avg Cost", justify="right", style="green")
    table.add_column("Avg Quality", justify="center")
    table.add_column("Success Rate", justify="right")

    for model, results in sorted(
        by_model.items(), key=lambda x: sum(r.latency for r in x[1]) / len(x[1])
    ):
        avg_latency = sum(r.latency for r in results) / len(results)
        avg_cost = sum(r.cost for r in results) / len(results)
        avg_quality = sum(
            len([c for c in rate_quality(r) if c == "★"]) for r in results
        ) / len(results)
        success_rate = (
            len([r for r in all_results if r.model == model and r.success])
            / len([r for r in all_results if r.model == model])
            * 100
        )

        table.add_row(
            model.split("/")[-1],
            f"{avg_latency:.1f}s",
            f"${avg_cost:.4f}",
            "★" * int(avg_quality) + "☆" * (5 - int(avg_quality)),
            f"{success_rate:.0f}%",
        )

    console.print(table)

    # Recommendations
    console.print("\n[bold]💡 Recommendations:[/bold]\n")

    # Fastest model
    fastest = min(by_model.items(), key=lambda x: sum(r.latency for r in x[1]))
    console.print(
        f"  🏃 [yellow]Fastest:[/yellow] {fastest[0]} "
        f"({sum(r.latency for r in fastest[1]) / len(fastest[1]):.1f}s avg)"
    )

    # Cheapest model
    cheapest = min(by_model.items(), key=lambda x: sum(r.cost for r in x[1]))
    console.print(
        f"  💰 [green]Cheapest:[/green] {cheapest[0]} "
        f"(${sum(r.cost for r in cheapest[1]) / len(cheapest[1]):.4f} avg)"
    )

    # Best quality model (most stars)
    best_quality = max(
        by_model.items(),
        key=lambda x: sum(len([c for c in rate_quality(r) if c == "★"]) for r in x[1]),
    )
    console.print(f"  ⭐ [cyan]Best Quality:[/cyan] {best_quality[0]}")

    # Overall recommendation
    console.print("\n[bold]🎯 Overall Recommendation:[/bold]")
    console.print(
        f"  For [yellow]speed[/yellow]: Use {fastest[0]} "
        f"(~{sum(r.latency for r in fastest[1]) / len(fastest[1]):.1f}s per query)"
    )
    console.print(
        f"  For [green]cost[/green]: Use {cheapest[0]} "
        f"(~${sum(r.cost for r in cheapest[1]) / len(cheapest[1]):.4f} per query)"
    )
    console.print(
        f"  For [cyan]quality[/cyan]: Use {best_quality[0]} (best result relevance)"
    )


async def run_benchmarks(
    models: list[str],
    queries: list[str],
    project_root: Path,
) -> list[BenchmarkResult]:
    """Run benchmarks for all model/query combinations.

    Args:
        models: List of model identifiers
        queries: List of test queries
        project_root: Project root directory

    Returns:
        List of all benchmark results
    """
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]✗ OPENROUTER_API_KEY not set[/red]", style="bold")
        console.print("\nSet your API key:")
        console.print("  export OPENROUTER_API_KEY='your-key-here'")
        raise typer.Exit(1)

    # Load project
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        console.print(
            f"[red]✗ Project not initialized at {project_root}[/red]",
            style="bold",
        )
        console.print("\nRun: mcp-vector-search init")
        raise typer.Exit(1)

    config = project_manager.load_config()

    # Initialize search engine
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

    all_results: list[BenchmarkResult] = []

    console.print(
        "\n[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    console.print(
        "[bold cyan]║           LLM Model Benchmark Results                        ║[/bold cyan]"
    )
    console.print(
        "[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]\n"
    )

    async with database:
        for query in queries:
            console.print(f"[bold]Testing query:[/bold] [white]{query}[/white]\n")

            query_results: list[BenchmarkResult] = []

            for model in models:
                console.print(f"  [dim]→ Testing {model}...[/dim]", end=" ")

                result = await benchmark_model(
                    model=model,
                    query=query,
                    project_root=project_root,
                    api_key=api_key,
                    search_engine=search_engine,
                    config=config,
                )

                query_results.append(result)
                all_results.append(result)

                if result.success:
                    console.print(
                        f"[green]✓[/green] {result.latency:.1f}s, ${result.cost:.4f}"
                    )
                else:
                    console.print(f"[red]✗[/red] {result.error}")

                # Rate limit protection: wait 1 second between requests
                await asyncio.sleep(1)

            # Print table for this query
            print_results_table(query_results, query)

    return all_results


def main(
    models: list[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Models to test (default: test all models)",
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        "-q",
        help="Single query to test (default: test all default queries)",
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        "-p",
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Benchmark LLM models for mcp-vector-search chat command.

    Tests various OpenRouter models on chat queries to compare:
    - Response quality (relevance of results)
    - Speed (latency in seconds)
    - Token usage (input/output tokens)
    - Cost (USD per query)

    Example:
        # Test all models on all queries
        python scripts/benchmark_llm_models.py

        # Test specific models
        python scripts/benchmark_llm_models.py --models anthropic/claude-3-haiku

        # Test single query
        python scripts/benchmark_llm_models.py --query "how does indexing work?"
    """
    # Default models
    if models is None:
        models = list(MODEL_PRICING.keys())

    # Default queries
    queries = [query] if query else DEFAULT_TEST_QUERIES

    # Run benchmarks
    try:
        all_results = asyncio.run(
            run_benchmarks(
                models=models,
                queries=queries,
                project_root=project_root,
            )
        )

        # Print summary
        print_summary(all_results)

        console.print("\n[dim]Benchmark completed![/dim]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
