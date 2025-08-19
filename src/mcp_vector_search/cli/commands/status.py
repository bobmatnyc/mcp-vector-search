"""Status command for MCP Vector Search CLI."""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from loguru import logger

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
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
    health_check: bool = typer.Option(
        False,
        "--health-check",
        help="Perform health check of all components",
    ),
    mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Check Claude Code MCP integration status",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output status in JSON format",
    ),
) -> None:
    """Show project status and indexing statistics.

    This command displays comprehensive information about your MCP Vector Search
    project including configuration, indexing status, and system health.

    Examples:
        mcp-vector-search status
        mcp-vector-search status --verbose
        mcp-vector-search status --health-check --json
        mcp-vector-search status --mcp
    """
    try:
        # Use provided project_root or current working directory
        if project_root is None:
            project_root = Path.cwd()

        asyncio.run(
            show_status(
                project_root=project_root,
                verbose=verbose,
                health_check=health_check,
                mcp=mcp,
                json_output=json_output,
            )
        )

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


async def show_status(
    project_root: Path,
    verbose: bool = False,
    health_check: bool = False,
    mcp: bool = False,
    json_output: bool = False,
) -> None:
    """Show comprehensive project status."""
    status_data = {}

    try:
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

        # Get project information
        project_info = project_manager.get_project_info()
        config = project_manager.load_config()

        # Get indexing statistics
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_root,
            file_extensions=config.file_extensions,
        )

        async with database:
            index_stats = await indexer.get_indexing_stats()
            db_stats = await database.get_stats()

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
            },
            "index": {
                "total_files": index_stats.get("total_indexable_files", 0),
                "indexed_files": index_stats.get("indexed_files", 0),
                "total_chunks": index_stats.get("total_chunks", 0),
                "languages": index_stats.get("languages", {}),
                "index_size_mb": db_stats.index_size_mb,
                "last_updated": db_stats.last_updated,
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


def _display_status(status_data: dict[str, Any], verbose: bool, mcp: bool = False) -> None:
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
    console.print(f"  Total Chunks: {index_data['total_chunks']}")
    console.print(f"  Index Size: {index_data['index_size_mb']:.2f} MB")

    if index_data["languages"]:
        console.print("  Language Distribution:")
        for lang, count in index_data["languages"].items():
            console.print(f"    {lang}: {count} chunks")
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
            console.print(f"[yellow]⚠[/yellow] MCP Server '{server_name}': {server_status}")

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
                await database.get_stats()
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


async def check_mcp_integration(project_root: Path, server_name: str = "mcp-vector-search") -> dict[str, Any]:
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
                    timeout=10
                )

                if result.returncode == 0:
                    mcp_status["server_status"] = "installed"
                else:
                    mcp_status["server_status"] = "not_installed"
                    mcp_status["issues"].append(f"MCP server '{server_name}' not found in Claude Code")

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
                with open(claude_json_path, 'r') as f:
                    config = json.load(f)
                if config.get("mcpServers", {}).get(server_name):
                    mcp_status["project_config"] = True
                else:
                    mcp_status["issues"].append(f"MCP server '{server_name}' not found in project .claude.json")
            except Exception as e:
                mcp_status["issues"].append(f"Error reading project .claude.json: {e}")
        else:
            mcp_status["issues"].append("Project .claude.json not found")

    except Exception as e:
        mcp_status["issues"].append(f"MCP integration check failed: {e}")

    return mcp_status


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    dependencies = [
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("tree_sitter", "Tree-sitter"),
        ("tree_sitter_languages", "Tree-sitter Languages"),
        ("typer", "Typer"),
        ("rich", "Rich"),
        ("pydantic", "Pydantic"),
        ("watchdog", "Watchdog"),
        ("loguru", "Loguru"),
    ]

    all_available = True

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_dependency_status(display_name, True)
        except ImportError:
            print_dependency_status(display_name, False)
            all_available = False

    return all_available


if __name__ == "__main__":
    status_app()
