"""Watch command for MCP Vector Search CLI."""

import asyncio
import signal
import sys
from pathlib import Path

import typer
from loguru import logger

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.indexer import SemanticIndexer
from ...core.project import ProjectManager
from ...core.watcher import FileWatcher
from ..output import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(name="watch", help="Watch for file changes and update index")


@app.command("main")
def watch_main(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory to watch",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file to use",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Watch for file changes and automatically update the search index.

    This command starts a file watcher that monitors your project directory
    for changes to code files. When files are created, modified, or deleted,
    the search index is automatically updated to reflect the changes.

    The watcher will:
    - Monitor all files with configured extensions
    - Debounce rapid changes to avoid excessive indexing
    - Update the index incrementally for better performance
    - Ignore common build/cache directories

    Press Ctrl+C to stop watching.

    Examples:
        mcp-vector-search watch
        mcp-vector-search watch /path/to/project --verbose
        mcp-vector-search watch --config custom-config.json
    """
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        asyncio.run(_watch_async(project_root, config))
    except KeyboardInterrupt:
        print_info("Watch stopped by user")
    except Exception as e:
        print_error(f"Watch failed: {e}")
        raise typer.Exit(1)


async def _watch_async(project_root: Path, config_path: Path | None) -> None:
    """Async implementation of watch command."""
    # Load project configuration
    try:
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(
                f"Project not initialized at {project_root}. "
                "Run 'mcp-vector-search init' first."
            )
            raise typer.Exit(1)

        config = project_manager.load_config()
        print_info(f"Loaded configuration from {project_root}")

    except ProjectNotFoundError:
        print_error(
            f"No MCP Vector Search project found at {project_root}. "
            "Run 'mcp-vector-search init' to initialize."
        )
        raise typer.Exit(1)

    # Setup database and indexer
    try:
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

        print_info(f"Initialized database at {config.index_path}")

    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        raise typer.Exit(1)

    # Start watching
    try:
        async with database:
            watcher = FileWatcher(
                project_root=project_root,
                config=config,
                indexer=indexer,
                database=database,
            )

            print_success("🔍 Starting file watcher...")
            print_info(f"📁 Watching: {project_root}")
            print_info(f"📄 Extensions: {', '.join(config.file_extensions)}")
            print_info("Press Ctrl+C to stop watching")

            async with watcher:
                # Set up signal handlers for graceful shutdown
                stop_event = asyncio.Event()

                def signal_handler():
                    print_info("\n⏹️  Stopping file watcher...")
                    stop_event.set()

                # Handle SIGINT (Ctrl+C) and SIGTERM
                if sys.platform != "win32":
                    loop = asyncio.get_running_loop()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.add_signal_handler(sig, signal_handler)

                try:
                    # Wait for stop signal
                    await stop_event.wait()
                except KeyboardInterrupt:
                    signal_handler()

                print_success("✅ File watcher stopped")

    except Exception as e:
        print_error(f"File watching failed: {e}")
        raise typer.Exit(1)


@app.command("status")
def watch_status(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Check if file watching is enabled for a project.

    This command checks the project configuration to see if file watching
    is enabled and provides information about the watch settings.
    """
    try:
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(f"Project not initialized at {project_root}")
            raise typer.Exit(1)

        config = project_manager.load_config()

        console.print("\n[bold]File Watch Status[/bold]")
        console.print(f"Project: {project_root}")
        console.print(f"Watch Files: {'✓' if config.watch_files else '✗'}")

        if config.watch_files:
            console.print(f"Extensions: {', '.join(config.file_extensions)}")
            print_info("File watching is enabled for this project")
        else:
            print_warning("File watching is disabled for this project")
            print_info("Enable with: mcp-vector-search config set watch_files true")

    except ProjectNotFoundError:
        print_error(f"No MCP Vector Search project found at {project_root}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to check watch status: {e}")
        raise typer.Exit(1)


@app.command("enable")
def watch_enable(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Enable file watching for a project.

    This command enables the watch_files setting in the project configuration.
    After enabling, you can use 'mcp-vector-search watch' to start monitoring.
    """
    try:
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(f"Project not initialized at {project_root}")
            raise typer.Exit(1)

        config = project_manager.load_config()
        config.watch_files = True
        project_manager.save_config(config)

        print_success("✅ File watching enabled")
        print_info("Start watching with: mcp-vector-search watch")

    except ProjectNotFoundError:
        print_error(f"No MCP Vector Search project found at {project_root}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to enable file watching: {e}")
        raise typer.Exit(1)


@app.command("disable")
def watch_disable(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Disable file watching for a project.

    This command disables the watch_files setting in the project configuration.
    """
    try:
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error(f"Project not initialized at {project_root}")
            raise typer.Exit(1)

        config = project_manager.load_config()
        config.watch_files = False
        project_manager.save_config(config)

        print_success("✅ File watching disabled")

    except ProjectNotFoundError:
        print_error(f"No MCP Vector Search project found at {project_root}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to disable file watching: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
