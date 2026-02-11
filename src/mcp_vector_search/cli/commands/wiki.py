"""Wiki command for generating codebase ontology."""

import asyncio
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from ...core.chunks_backend import ChunksBackend
from ...core.embeddings import create_embedding_function
from ...core.factory import create_database
from ...core.llm_client import LLMClient
from ...core.project import ProjectManager
from ...core.wiki import WikiGenerator

console = Console()


def _load_env_files(project_root: Path) -> None:
    """Load environment variables from .env and .env.local files.

    Priority (later files override earlier):
    1. .env (base config)
    2. .env.local (local overrides, gitignored)

    Args:
        project_root: Project root directory to search for env files
    """
    env_files = [
        project_root / ".env",
        project_root / ".env.local",
    ]

    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=True)
            logger.debug(f"Loaded environment from {env_file}")


wiki_app = typer.Typer(
    help="📚 Generate wiki/ontology of codebase concepts",
    invoke_without_command=True,
)


@wiki_app.callback(invoke_without_command=True)
def wiki_main(
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
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON ontology to stdout",
    ),
    display: bool = typer.Option(
        False,
        "--display",
        help="Generate static HTML wiki for browser viewing",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for HTML wiki (default: ./docs/wiki.html)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force regeneration, ignoring cache",
    ),
    ttl: int | None = typer.Option(
        None,
        "--ttl",
        help="Cache TTL in hours (default: 24, env: MCP_WIKI_CACHE_TTL_HOURS)",
        min=1,
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM semantic grouping (flat ontology only)",
    ),
) -> None:
    """📚 Generate wiki/ontology of codebase concepts.

    Analyzes indexed code chunks to extract concepts (functions, classes, patterns)
    and organizes them into a hierarchical ontology using LLM semantic grouping.

    [bold cyan]Basic Usage:[/bold cyan]

    [green]Generate JSON ontology (cached):[/green]
        $ mcp-vector-search wiki

    [green]Output raw JSON:[/green]
        $ mcp-vector-search wiki --json

    [green]Generate HTML wiki:[/green]
        $ mcp-vector-search wiki --display --output ./docs/wiki.html

    [bold cyan]Cache Control:[/bold cyan]

    [green]Force regeneration:[/green]
        $ mcp-vector-search wiki --force

    [green]Custom TTL:[/green]
        $ mcp-vector-search wiki --ttl 48

    [green]Skip LLM grouping:[/green]
        $ mcp-vector-search wiki --no-llm

    [bold]Caching:[/bold]
    - Location: .mcp-vector-search/wiki_cache.json
    - Default TTL: 24 hours (configurable via --ttl or MCP_WIKI_CACHE_TTL_HOURS)
    - Invalidation: --force flag, TTL expiration, or chunk count mismatch

    [dim]💡 Tip: LLM grouping requires OPENAI_API_KEY or OPENROUTER_API_KEY[/dim]
    """
    try:
        # Resolve project root
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()

        # Run async wiki generation
        asyncio.run(
            run_wiki_generation(
                project_root=project_root,
                json_output=json_output,
                display=display,
                output=output,
                force=force,
                ttl=ttl,
                no_llm=no_llm,
            )
        )

    except Exception as e:
        logger.error(f"Wiki generation failed: {e}")
        console.print(f"[red]✗ Wiki generation failed: {e}[/red]")
        raise typer.Exit(1)


async def run_wiki_generation(
    project_root: Path,
    json_output: bool = False,
    display: bool = False,
    output: Path | None = None,
    force: bool = False,
    ttl: int | None = None,
    no_llm: bool = False,
) -> None:
    """Run wiki generation asynchronously.

    Args:
        project_root: Project root directory
        json_output: Output raw JSON to stdout
        display: Generate HTML wiki
        output: Output path for HTML wiki
        force: Force regeneration
        ttl: Cache TTL in hours
        no_llm: Skip LLM semantic grouping
    """
    # Load environment variables from .env and .env.local
    _load_env_files(project_root)

    # Load project config
    project_manager = ProjectManager(project_root)
    config = project_manager.load_config()

    # Initialize vector database (main index)
    embedding_function, _ = create_embedding_function(config.embedding_model)
    vector_database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )
    await vector_database.initialize()

    # Initialize chunks backend (for two-phase indexing fallback)
    chunks_backend = ChunksBackend(config.index_path)
    await chunks_backend.initialize()

    # Initialize LLM client (if not skipping)
    llm_client = None
    if not no_llm:
        try:
            llm_client = LLMClient()
            provider = llm_client.provider.capitalize()
            model = llm_client.model
            console.print(
                f"[dim]Using {provider} ({model}) for semantic grouping...[/dim]"
            )
        except ValueError:
            # Provide helpful error message with setup instructions
            console.print(
                "[yellow]⚠️  No LLM provider configured.[/yellow]\n"
                "\n"
                "[bold]To enable LLM semantic grouping, set one of:[/bold]\n"
                "\n"
                "  [cyan]AWS Bedrock (recommended):[/cyan]\n"
                "    export AWS_ACCESS_KEY_ID=your-key\n"
                "    export AWS_SECRET_ACCESS_KEY=your-secret\n"
                "    export AWS_REGION=us-east-1\n"
                "\n"
                "  [cyan]OpenRouter:[/cyan]\n"
                "    export OPENROUTER_API_KEY=your-key\n"
                "\n"
                "  [cyan]Or add to .env or .env.local in project root[/cyan]\n"
                "\n"
                "[dim]Falling back to flat ontology (--no-llm mode)[/dim]"
            )
            no_llm = True

    # Update cache TTL if specified
    if ttl is None:
        ttl = int(os.getenv("MCP_WIKI_CACHE_TTL_HOURS", "24"))

    # Initialize wiki generator
    generator = WikiGenerator(
        project_root=project_root,
        chunks_backend=chunks_backend,
        llm_client=llm_client,
        vector_database=vector_database,
    )
    generator.cache.ttl_hours = ttl

    # Generate ontology
    console.print("[cyan]Generating wiki ontology...[/cyan]")
    ontology = await generator.generate(force=force, use_llm=not no_llm)

    # Output based on flags
    if json_output:
        # Raw JSON to stdout
        print(json.dumps(ontology.to_dict(), indent=2))
    elif display:
        # Generate HTML wiki
        output_path = output or (project_root / "docs" / "wiki.html")
        generate_html_wiki(ontology, output_path)
        console.print(f"[green]✓ Generated HTML wiki: {output_path}[/green]")
    else:
        # Summary output (default)
        console.print("\n[bold]Wiki Ontology Summary[/bold]")
        console.print(f"  Total concepts: {ontology.total_concepts}")
        console.print(f"  Total chunks: {ontology.total_chunks}")
        console.print(f"  Root categories: {len(ontology.root_categories)}")
        console.print(f"  Generated: {ontology.generated_at}")
        console.print(f"  Cache TTL: {ontology.ttl_hours} hours")

        # List root categories
        console.print("\n[bold]Root Categories:[/bold]")
        for cat_id in ontology.root_categories[:10]:  # Limit to top 10
            concept = ontology.concepts[cat_id]
            console.print(
                f"  • {concept.name} ({len(concept.children)} concepts, "
                f"{concept.frequency} references)"
            )
            if concept.description:
                console.print(f"    [dim]{concept.description}[/dim]")

        console.print(
            "\n[dim]Run with --json for full ontology or --display for HTML wiki[/dim]"
        )

    # Cleanup
    await chunks_backend.close()
    await vector_database.close()


def generate_html_wiki(ontology, output_path: Path) -> None:
    """Generate static HTML wiki with embedded CSS.

    Args:
        ontology: WikiOntology to render
        output_path: Output file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load HTML template
    template_path = Path(__file__).parent.parent.parent / "templates" / "wiki.html"

    if template_path.exists():
        # Use external template
        with open(template_path) as f:
            template = f.read()
    else:
        # Fallback to inline template
        template = _get_inline_wiki_template()

    # Build HTML content from ontology
    categories_html = []

    for cat_id in ontology.root_categories:
        category = ontology.concepts[cat_id]

        # Build category section
        children_html = []
        for child_id in category.children[:100]:  # Limit per category
            if child_id in ontology.concepts:
                child = ontology.concepts[child_id]
                children_html.append(
                    f'<li class="concept-item">'
                    f'<span class="concept-name">{child.name}</span> '
                    f'<span class="concept-frequency">({child.frequency} refs)</span>'
                    f"</li>"
                )

        category_html = f"""
        <div class="category">
            <h2>{category.name}</h2>
            <p class="category-description">{category.description}</p>
            <p class="category-stats">{len(category.children)} concepts, {category.frequency} references</p>
            <ul class="concept-list">
                {"".join(children_html)}
            </ul>
        </div>
        """
        categories_html.append(category_html)

    # Replace placeholders
    html = template.replace("{{TITLE}}", f"Wiki: {ontology.project_root}")
    html = html.replace("{{GENERATED_AT}}", ontology.generated_at)
    html = html.replace("{{TOTAL_CONCEPTS}}", str(ontology.total_concepts))
    html = html.replace("{{TOTAL_CHUNKS}}", str(ontology.total_chunks))
    html = html.replace("{{CATEGORIES}}", "\n".join(categories_html))

    # Write output
    with open(output_path, "w") as f:
        f.write(html)


def _get_inline_wiki_template() -> str:
    """Get inline HTML template as fallback.

    Returns:
        HTML template string
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        header {
            border-bottom: 2px solid #3498db;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        h1 {
            color: #2c3e50;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .meta {
            color: #7f8c8d;
            font-size: 0.9rem;
        }

        .stats {
            display: flex;
            gap: 2rem;
            margin: 1rem 0;
            padding: 1rem;
            background: #ecf0f1;
            border-radius: 4px;
        }

        .stat {
            flex: 1;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }

        .category {
            margin: 2rem 0;
            padding: 1.5rem;
            background: #fff;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .category h2 {
            color: #2c3e50;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .category-description {
            color: #7f8c8d;
            margin-bottom: 0.5rem;
            font-style: italic;
        }

        .category-stats {
            color: #95a5a6;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .concept-list {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 0.5rem;
        }

        .concept-item {
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
            transition: background 0.2s;
        }

        .concept-item:hover {
            background: #e9ecef;
        }

        .concept-name {
            font-weight: 500;
            color: #2c3e50;
        }

        .concept-frequency {
            color: #95a5a6;
            font-size: 0.85rem;
        }

        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📚 Codebase Wiki</h1>
            <p class="meta">Generated: {{GENERATED_AT}}</p>
        </header>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Total Concepts</div>
                <div class="stat-value">{{TOTAL_CONCEPTS}}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total Chunks</div>
                <div class="stat-value">{{TOTAL_CHUNKS}}</div>
            </div>
        </div>

        <main>
            {{CATEGORIES}}
        </main>

        <footer>
            <p>Generated by mcp-vector-search wiki</p>
        </footer>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    wiki_app()
