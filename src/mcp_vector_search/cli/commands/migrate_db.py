"""Database migration commands for switching between backends."""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.factory import ComponentFactory
from ...core.lancedb_backend import LanceVectorDatabase

app = typer.Typer(help="Migrate database between backends (ChromaDB ↔ LanceDB)")
console = Console()


@app.command()
def chromadb_to_lancedb(
    project_path: Path = typer.Argument(
        ..., help="Project path containing .mcp-vector-search directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing LanceDB database"
    ),
) -> None:
    """Migrate from ChromaDB to LanceDB backend.

    This command:
    1. Reads all chunks from existing ChromaDB database
    2. Creates a new LanceDB database with the same data
    3. Preserves all metadata, embeddings, and relationships

    After migration, set MCP_VECTOR_SEARCH_BACKEND=lancedb to use the new backend.

    Example:
        mcp-vector-search migrate-db chromadb-to-lancedb /path/to/project
    """
    import asyncio

    asyncio.run(_migrate_chromadb_to_lancedb(project_path, force))


@app.command()
def lancedb_to_chromadb(
    project_path: Path = typer.Argument(
        ..., help="Project path containing .mcp-vector-search directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing ChromaDB database"
    ),
) -> None:
    """Migrate from LanceDB to ChromaDB backend.

    This command:
    1. Reads all chunks from existing LanceDB database
    2. Creates a new ChromaDB database with the same data
    3. Preserves all metadata and relationships

    After migration, set MCP_VECTOR_SEARCH_BACKEND=chromadb to use ChromaDB.

    Example:
        mcp-vector-search migrate-db lancedb-to-chromadb /path/to/project
    """
    import asyncio

    asyncio.run(_migrate_lancedb_to_chromadb(project_path, force))


async def _migrate_chromadb_to_lancedb(project_path: Path, force: bool) -> None:
    """Internal migration from ChromaDB to LanceDB."""
    try:
        # Load project config
        project_manager, config = ComponentFactory.load_config(project_path)

        # Check if ChromaDB exists
        chromadb_path = config.index_path / "chroma.sqlite3"
        if not chromadb_path.exists():
            console.print(
                "[red]Error:[/red] ChromaDB database not found. "
                "Please index your project first."
            )
            raise typer.Exit(1)

        # Check if LanceDB already exists
        lancedb_path = config.index_path / "lancedb"
        if lancedb_path.exists() and not force:
            console.print(
                "[yellow]Warning:[/yellow] LanceDB database already exists. "
                "Use --force to overwrite."
            )
            raise typer.Exit(1)

        console.print(
            "[bold blue]Starting migration from ChromaDB to LanceDB...[/bold blue]"
        )

        # Create embedding function
        embedding_function, _ = create_embedding_function(config.embedding_model)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Read from ChromaDB
            task1 = progress.add_task("Reading chunks from ChromaDB...", total=None)

            async with ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as chromadb:
                chunks = await chromadb.get_all_chunks()

            progress.update(task1, completed=True)
            console.print(f"[green]✓[/green] Read {len(chunks)} chunks from ChromaDB")

            if not chunks:
                console.print(
                    "[yellow]Warning:[/yellow] No chunks found in ChromaDB. "
                    "Nothing to migrate."
                )
                return

            # Step 2: Write to LanceDB
            task2 = progress.add_task("Writing chunks to LanceDB...", total=None)

            async with LanceVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as lancedb:
                # Add chunks in batches for better performance
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    await lancedb.add_chunks(batch)
                    progress.update(
                        task2,
                        description=f"Writing chunks to LanceDB ({i + len(batch)}/{len(chunks)})...",
                    )

            progress.update(task2, completed=True)
            console.print(f"[green]✓[/green] Wrote {len(chunks)} chunks to LanceDB")

            # Verify migration
            task3 = progress.add_task("Verifying migration...", total=None)

            async with LanceVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as lancedb:
                stats = await lancedb.get_stats()

            progress.update(task3, completed=True)

            if stats.total_chunks != len(chunks):
                console.print(
                    f"[red]Error:[/red] Migration verification failed. "
                    f"Expected {len(chunks)} chunks, found {stats.total_chunks}"
                )
                raise typer.Exit(1)

            console.print("[green]✓[/green] Migration verified successfully")

        console.print("\n[bold green]Migration completed successfully![/bold green]")
        console.print("\nTo use the new LanceDB backend, set the environment variable:")
        console.print("[cyan]export MCP_VECTOR_SEARCH_BACKEND=lancedb[/cyan]")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        console.print(f"[red]Error:[/red] Migration failed: {e}")
        raise typer.Exit(1)


async def _migrate_lancedb_to_chromadb(project_path: Path, force: bool) -> None:
    """Internal migration from LanceDB to ChromaDB."""
    try:
        # Load project config
        project_manager, config = ComponentFactory.load_config(project_path)

        # Check if LanceDB exists
        lancedb_path = config.index_path / "lancedb"
        if not lancedb_path.exists():
            console.print(
                "[red]Error:[/red] LanceDB database not found. "
                "Please index your project with LanceDB first."
            )
            raise typer.Exit(1)

        # Check if ChromaDB already exists
        chromadb_path = config.index_path / "chroma.sqlite3"
        if chromadb_path.exists() and not force:
            console.print(
                "[yellow]Warning:[/yellow] ChromaDB database already exists. "
                "Use --force to overwrite."
            )
            raise typer.Exit(1)

        console.print(
            "[bold blue]Starting migration from LanceDB to ChromaDB...[/bold blue]"
        )

        # Create embedding function
        embedding_function, _ = create_embedding_function(config.embedding_model)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Read from LanceDB
            task1 = progress.add_task("Reading chunks from LanceDB...", total=None)

            async with LanceVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as lancedb:
                chunks = await lancedb.get_all_chunks()

            progress.update(task1, completed=True)
            console.print(f"[green]✓[/green] Read {len(chunks)} chunks from LanceDB")

            if not chunks:
                console.print(
                    "[yellow]Warning:[/yellow] No chunks found in LanceDB. "
                    "Nothing to migrate."
                )
                return

            # Step 2: Write to ChromaDB
            task2 = progress.add_task("Writing chunks to ChromaDB...", total=None)

            # Remove existing ChromaDB if force is enabled
            if force and chromadb_path.exists():
                import shutil

                shutil.rmtree(config.index_path / "chroma", ignore_errors=True)
                chromadb_path.unlink(missing_ok=True)

            async with ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as chromadb:
                # Add chunks in batches
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    await chromadb.add_chunks(batch)
                    progress.update(
                        task2,
                        description=f"Writing chunks to ChromaDB ({i + len(batch)}/{len(chunks)})...",
                    )

            progress.update(task2, completed=True)
            console.print(f"[green]✓[/green] Wrote {len(chunks)} chunks to ChromaDB")

            # Verify migration
            task3 = progress.add_task("Verifying migration...", total=None)

            async with ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            ) as chromadb:
                stats = await chromadb.get_stats()

            progress.update(task3, completed=True)

            if stats.total_chunks != len(chunks):
                console.print(
                    f"[red]Error:[/red] Migration verification failed. "
                    f"Expected {len(chunks)} chunks, found {stats.total_chunks}"
                )
                raise typer.Exit(1)

            console.print("[green]✓[/green] Migration verified successfully")

        console.print("\n[bold green]Migration completed successfully![/bold green]")
        console.print("\nTo use the ChromaDB backend, set the environment variable:")
        console.print("[cyan]export MCP_VECTOR_SEARCH_BACKEND=chromadb[/cyan]")
        console.print("\n(LanceDB is the default in v2.x)")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        console.print(f"[red]Error:[/red] Migration failed: {e}")
        raise typer.Exit(1)
