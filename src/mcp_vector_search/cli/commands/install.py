"""Install command for MCP Vector Search CLI."""

import asyncio
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from ...core.project import ProjectManager
from ..output import print_error, print_info, print_success, print_warning

# Create console for rich output
console = Console()

# Create install subcommand app
install_app = typer.Typer(help="Install mcp-vector-search in projects")


@install_app.command()
def main(
    ctx: typer.Context,
    target_directory: Path = typer.Argument(
        None,
        help="Target directory to install in (default: current directory)",
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
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated list of file extensions to index (e.g., '.py,.js,.ts,.txt,.md')",
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        "-m",
        help="Embedding model to use for semantic search",
    ),
    similarity_threshold: float = typer.Option(
        0.5,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold for search results (0.0 to 1.0)",
        min=0.0,
        max=1.0,
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP integration setup",
    ),
    no_auto_index: bool = typer.Option(
        False,
        "--no-auto-index",
        help="Skip automatic indexing after setup",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-installation if project is already initialized",
    ),
) -> None:
    """Install mcp-vector-search in a project directory.

    This command sets up mcp-vector-search in the specified directory (or current directory)
    with full initialization, indexing, and optional MCP integration.

    Examples:
        mcp-vector-search install                    # Install in current directory
        mcp-vector-search install ~/my-project       # Install in specific directory
        mcp-vector-search install --no-mcp           # Install without MCP integration
        mcp-vector-search install --force            # Force re-installation
        mcp-vector-search install --extensions .py,.js,.ts,.txt  # Custom file extensions
    """
    try:
        # Determine target directory - prioritize target_directory, then project_root, then context, then cwd
        target_dir = target_directory or project_root or ctx.obj.get("project_root") or Path.cwd()
        target_dir = target_dir.resolve()

        # Show installation header
        console.print(Panel.fit(
            f"[bold blue]🚀 MCP Vector Search - Project Installation[/bold blue]\n\n"
            f"📁 Target project: [cyan]{target_dir}[/cyan]\n"
            f"🔧 Installing with full setup and configuration",
            border_style="blue"
        ))

        # Check if target directory exists
        if not target_dir.exists():
            print_error(f"Target directory does not exist: {target_dir}")
            raise typer.Exit(1)

        # Check if already initialized
        project_manager = ProjectManager(target_dir)
        if project_manager.is_initialized() and not force:
            print_success("Project is already initialized and ready to use!")
            print_info("Your project has vector search capabilities enabled.")
            print_info(f"Use --force to re-initialize or run 'mcp-vector-search --project-root {target_dir} status main' to see current configuration")
            
            console.print("\n[bold green]🔍 Ready to use:[/bold green]")
            console.print(f"  • Search your code: [code]cd {target_dir} && mcp-vector-search search 'your query'[/code]")
            console.print(f"  • Check status: [code]cd {target_dir} && mcp-vector-search status main[/code]")
            console.print("  • Use [code]--force[/code] to re-initialize if needed")
            return

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [ext.strip() for ext in extensions.split(",")]
            # Ensure extensions start with dot
            file_extensions = [
                ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
            ]

        # Run initialization
        print_info("Initializing project...")
        
        # Import init functionality
        from .init import run_init_setup

        asyncio.run(
            run_init_setup(
                project_root=target_dir,
                file_extensions=file_extensions,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                mcp=not no_mcp,
                auto_index=not no_auto_index,
                force=force,
            )
        )

        # Show completion message
        console.print(Panel.fit(
            f"[bold green]🎉 Installation Complete![/bold green]\n\n"
            f"✅ Project initialized at [cyan]{target_dir}[/cyan]\n"
            f"✅ Vector database configured\n"
            f"✅ {'Codebase indexed' if not no_auto_index else 'Ready for indexing'}\n"
            f"✅ {'MCP integration enabled' if not no_mcp else 'MCP integration skipped'}\n\n"
            f"[bold blue]🔍 Next steps:[/bold blue]\n"
            f"  • [code]cd {target_dir}[/code]\n"
            f"  • [code]mcp-vector-search search 'your query'[/code]\n"
            f"  • [code]mcp-vector-search status main[/code]",
            border_style="green"
        ))

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)


@install_app.command("demo")
def demo(
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
) -> None:
    """Run installation demo with sample project."""
    try:
        import tempfile
        import shutil
        
        print_info("🎬 Running mcp-vector-search installation demo...")
        
        # Create temporary demo directory
        with tempfile.TemporaryDirectory(prefix="mcp-demo-") as temp_dir:
            demo_dir = Path(temp_dir) / "demo-project"
            demo_dir.mkdir()
            
            # Create sample files
            (demo_dir / "main.py").write_text("""
def main():
    '''Main entry point for the application.'''
    print("Hello, World!")
    user_service = UserService()
    user_service.create_user("Alice", "alice@example.com")

class UserService:
    '''Service for managing users.'''
    
    def create_user(self, name: str, email: str):
        '''Create a new user with the given name and email.'''
        print(f"Creating user: {name} ({email})")
        return {"name": name, "email": email}
    
    def authenticate_user(self, email: str, password: str):
        '''Authenticate user with email and password.'''
        # Simple authentication logic
        return email.endswith("@example.com")

if __name__ == "__main__":
    main()
""")
            
            (demo_dir / "utils.py").write_text("""
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    '''Load configuration from JSON file.'''
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_email(email: str) -> bool:
    '''Validate email address format.'''
    return "@" in email and "." in email.split("@")[1]

def hash_password(password: str) -> str:
    '''Hash password for secure storage.'''
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
""")
            
            console.print(f"\n[bold blue]📁 Created demo project at:[/bold blue] {demo_dir}")
            
            # Run installation
            print_info("Installing mcp-vector-search in demo project...")
            
            # Use subprocess to run the install command
            result = subprocess.run([
                sys.executable, "-m", "mcp_vector_search.cli.main",
                "--project-root", str(demo_dir),
                "install", str(demo_dir),
                "--extensions", ".py",
                "--no-mcp"  # Skip MCP for demo
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success("✅ Demo installation completed!")
                
                # Run a sample search
                print_info("Running sample search: 'user authentication'...")
                
                search_result = subprocess.run([
                    sys.executable, "-m", "mcp_vector_search.cli.main",
                    "--project-root", str(demo_dir),
                    "search", "user authentication",
                    "--limit", "3"
                ], capture_output=True, text=True)
                
                if search_result.returncode == 0:
                    console.print("\n[bold green]🔍 Sample search results:[/bold green]")
                    console.print(search_result.stdout)
                else:
                    print_warning("Search demo failed, but installation was successful")
                
                console.print(f"\n[bold blue]🎉 Demo completed![/bold blue]")
                console.print(f"Demo project was created at: [cyan]{demo_dir}[/cyan]")
                console.print("The temporary directory will be cleaned up automatically.")
                
            else:
                print_error(f"Demo installation failed: {result.stderr}")
                raise typer.Exit(1)
                
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print_error(f"Demo failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    install_app()
