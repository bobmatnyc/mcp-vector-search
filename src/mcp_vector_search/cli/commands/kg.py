"""Knowledge graph CLI commands."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.factory import ComponentFactory
from ...core.kg_builder import KGBuilder
from ...core.knowledge_graph import KnowledgeGraph

console = Console()
kg_app = typer.Typer(name="kg", help="📊 Knowledge graph operations")


@kg_app.command("build")
def build_kg(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
    force: bool = typer.Option(False, help="Force rebuild even if graph exists"),
    limit: int | None = typer.Option(
        None, help="Limit number of chunks to process (for testing)"
    ),
):
    """Build knowledge graph from indexed chunks.

    This command extracts entities and relationships from your indexed
    codebase and builds a queryable knowledge graph.

    Example:
        mcp-vector-search kg build
        mcp-vector-search kg build --force
        mcp-vector-search kg build --limit 100  # Test with 100 chunks
    """
    project_root = project_root.resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Building Knowledge Graph[/bold cyan]\nProject: {project_root}",
            border_style="cyan",
        )
    )

    async def _build():
        # Initialize components
        components = await ComponentFactory.create_standard_components(
            project_root, use_pooling=False
        )
        database = components.database

        # Use context manager to properly open database
        async with database:
            # Check if index exists (use chunk count instead of is_indexed)
            chunk_count = database.get_chunk_count()
            if chunk_count == 0:
                console.print(
                    "[red]✗[/red] No index found. Run 'mcp-vector-search index' first."
                )
                raise typer.Exit(1)

            # Initialize knowledge graph
            kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Check if graph already exists
            stats = await kg.get_stats()
            if stats["total_entities"] > 0 and not force:
                console.print(
                    f"[yellow]⚠[/yellow] Knowledge graph already exists "
                    f"({stats['total_entities']} entities). "
                    "Use --force to rebuild."
                )
                raise typer.Exit(0)

            # Build graph
            builder = KGBuilder(kg, project_root)
            build_stats = await builder.build_from_database(
                database, show_progress=True, limit=limit
            )

            # Show results
            table = Table(title="Knowledge Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green", justify="right")

            table.add_row("Entities", str(build_stats["entities"]))
            table.add_row("Calls", str(build_stats["calls"]))
            table.add_row("Imports", str(build_stats["imports"]))
            table.add_row("Inherits", str(build_stats["inherits"]))
            table.add_row("Contains", str(build_stats["contains"]))

            console.print(table)
            console.print("[green]✓[/green] Knowledge graph built successfully!")

        # Close connections
        await kg.close()
        await database.close()

    asyncio.run(_build())


@kg_app.command("query")
def query_kg(
    entity: str = typer.Argument(..., help="Entity name or ID to query"),
    hops: int = typer.Option(2, help="Maximum number of relationship hops"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Query relationships for an entity.

    Find entities related to the specified entity within N hops.

    Examples:
        mcp-vector-search kg query "SemanticSearchEngine"
        mcp-vector-search kg query "search" --hops 1
    """
    project_root = project_root.resolve()

    async def _query():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Check if graph exists
        stats = await kg.get_stats()
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            raise typer.Exit(1)

        # Try to find entity by name or ID
        # For simplicity, we'll search for entities matching the name
        related = await kg.find_related(entity, max_hops=hops)

        if not related:
            console.print(f"[yellow]No related entities found for '{entity}'[/yellow]")
            raise typer.Exit(0)

        # Display results
        table = Table(title=f"Entities related to '{entity}' (within {hops} hops)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("File", style="dim")

        for rel in related:
            table.add_row(rel["name"], rel["type"], rel["file_path"])

        console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_query())


@kg_app.command("stats")
def kg_stats(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show knowledge graph statistics.

    Display entity and relationship counts from the knowledge graph.

    Example:
        mcp-vector-search kg stats
    """
    project_root = project_root.resolve()

    async def _stats():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get statistics
        stats = await kg.get_stats()

        # Display results
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Entities", str(stats["total_entities"]))
        table.add_row("Database Path", stats["database_path"])

        # Add relationship counts
        if "relationships" in stats:
            for rel_type, count in stats["relationships"].items():
                table.add_row(f"  {rel_type.title()}", str(count))

        console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_stats())


@kg_app.command("calls")
def get_calls(
    function: str = typer.Argument(..., help="Function name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show call graph for a function.

    Display functions that call or are called by the specified function.

    Examples:
        mcp-vector-search kg calls "search"
        mcp-vector-search kg calls "SemanticSearchEngine.search"
    """
    project_root = project_root.resolve()

    async def _calls():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get call graph
        calls = await kg.get_call_graph(function)

        if not calls:
            console.print(
                f"[yellow]No call relationships found for '{function}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate calls vs called_by
        calls_out = [c for c in calls if c["direction"] == "calls"]
        calls_in = [c for c in calls if c["direction"] == "called_by"]

        # Display results
        if calls_out:
            table = Table(title=f"Functions called by '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_out:
                table.add_row(call["name"])

            console.print(table)

        if calls_in:
            table = Table(title=f"Functions that call '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_in:
                table.add_row(call["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_calls())


@kg_app.command("inherits")
def get_inheritance(
    class_name: str = typer.Argument(..., help="Class name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show inheritance tree for a class.

    Display parent and child classes in the inheritance hierarchy.

    Examples:
        mcp-vector-search kg inherits "BaseModel"
        mcp-vector-search kg inherits "SemanticSearchEngine"
    """
    project_root = project_root.resolve()

    async def _inherits():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get inheritance tree
        hierarchy = await kg.get_inheritance_tree(class_name)

        if not hierarchy:
            console.print(
                f"[yellow]No inheritance relationships found for '{class_name}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate parents vs children
        parents = [h for h in hierarchy if h["relation"] == "parent"]
        children = [h for h in hierarchy if h["relation"] == "child"]

        # Display results
        if parents:
            table = Table(title=f"Classes that '{class_name}' inherits from")
            table.add_column("Class", style="cyan")

            for parent in parents:
                table.add_row(parent["name"])

            console.print(table)

        if children:
            table = Table(title=f"Classes that inherit from '{class_name}'")
            table.add_column("Class", style="cyan")

            for child in children:
                table.add_row(child["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_inherits())
