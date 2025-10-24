"""Demo command for mcp-vector-search."""

import subprocess
import sys
import tempfile
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from ..output import print_error, print_info, print_success

console = Console()

demo_app = typer.Typer(
    name="demo",
    help="🎬 Run interactive demo with sample project",
    add_completion=False,
    rich_markup_mode="rich",
)


@demo_app.callback(invoke_without_command=True)
def demo(
    ctx: typer.Context,
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Skip search demo, only show installation",
    ),
    keep_files: bool = typer.Option(
        False,
        "--keep",
        "-k",
        help="Keep demo files (don't auto-cleanup)",
    ),
) -> None:
    """
    Run installation demo with sample project.

    [bold cyan]What this does:[/bold cyan]
      1. Creates a temporary project with sample Python files
      2. Initializes mcp-vector-search in the demo project
      3. Indexes the sample code
      4. Runs a sample semantic search
      5. Shows you the results

    [bold cyan]Examples:[/bold cyan]
      [green]mcp-vector-search demo[/green]
        Run full interactive demo

      [green]mcp-vector-search demo --quick[/green]
        Skip search demo, only show installation

      [green]mcp-vector-search demo --keep[/green]
        Keep demo files for inspection

    [dim]Perfect for first-time users to see how semantic search works![/dim]
    """
    # If subcommand was invoked, don't run the main demo
    if ctx.invoked_subcommand is not None:
        return

    try:
        print_info("🎬 Running mcp-vector-search installation demo...")
        console.print(
            "\n[bold]This demo will:[/bold]\n"
            "  1. Create a sample Python project\n"
            "  2. Initialize and index it\n"
            "  3. Run a semantic search\n"
            "  4. Show you the results\n"
        )

        # Create temporary demo directory
        temp_context = tempfile.TemporaryDirectory(prefix="mcp-demo-")
        temp_dir = temp_context.__enter__()

        try:
            demo_dir = Path(temp_dir) / "demo-project"
            demo_dir.mkdir()

            # Create sample files
            console.print("[bold blue]📝 Creating sample files...[/bold blue]")

            (demo_dir / "main.py").write_text(
                '''"""Main application module."""

def main():
    """Main entry point for the application."""
    print("Hello, World!")
    user_service = UserService()
    user_service.create_user("Alice", "alice@example.com")


class UserService:
    """Service for managing users."""

    def create_user(self, name: str, email: str):
        """Create a new user with the given name and email.

        Args:
            name: The user's full name
            email: The user's email address

        Returns:
            dict: User object with name and email
        """
        print(f"Creating user: {name} ({email})")
        return {"name": name, "email": email}

    def authenticate_user(self, email: str, password: str):
        """Authenticate user with email and password.

        Args:
            email: User's email address
            password: User's password

        Returns:
            bool: True if authentication successful
        """
        # Simple authentication logic
        return email.endswith("@example.com")

    def reset_password(self, email: str):
        """Send password reset email to user.

        Args:
            email: User's email address
        """
        print(f"Sending password reset to {email}")


if __name__ == "__main__":
    main()
'''
            )

            (demo_dir / "utils.py").write_text(
                '''"""Utility functions for the application."""

import hashlib
import json
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        return json.load(f)


def validate_email(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        bool: True if email format is valid
    """
    return "@" in email and "." in email.split("@")[1]


def hash_password(password: str) -> str:
    """Hash password for secure storage.

    Args:
        password: Plain text password

    Returns:
        str: Hexadecimal hash of password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def validate_password_strength(password: str) -> bool:
    """Check if password meets security requirements.

    Args:
        password: Password to validate

    Returns:
        bool: True if password is strong enough
    """
    return len(password) >= 8 and any(c.isdigit() for c in password)
'''
            )

            (demo_dir / "api.py").write_text(
                '''"""API endpoints for user management."""

from typing import Dict, List


class UserAPI:
    """REST API endpoints for user operations."""

    def get_user(self, user_id: int) -> Dict:
        """Get user by ID.

        Args:
            user_id: Unique user identifier

        Returns:
            dict: User object
        """
        return {"id": user_id, "name": "Example User"}

    def list_users(self, limit: int = 10) -> List[Dict]:
        """List all users with pagination.

        Args:
            limit: Maximum number of users to return

        Returns:
            list: List of user objects
        """
        return [{"id": i, "name": f"User {i}"} for i in range(limit)]

    def create_user_endpoint(self, name: str, email: str) -> Dict:
        """Create a new user via API.

        Args:
            name: User's full name
            email: User's email address

        Returns:
            dict: Created user object with ID
        """
        return {"id": 1, "name": name, "email": email}
'''
            )

            console.print(f"[green]✅ Created 3 sample files in:[/green] {demo_dir}\n")

            # Run initialization
            print_info("🔧 Initializing mcp-vector-search in demo project...")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mcp_vector_search.cli.main",
                    "--project-root",
                    str(demo_dir),
                    "init",
                    "--extensions",
                    ".py",
                    "--no-mcp",  # Skip MCP for demo
                    "--auto-index",  # Auto-index after init
                ],
                input="y\n",  # Auto-confirm the setup
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print_error(f"Demo initialization failed: {result.stderr}")
                raise typer.Exit(1)

            print_success("✅ Demo project initialized and indexed!")

            if not quick:
                # Give the index a moment to settle
                import time

                time.sleep(1)

                # Run sample searches to demonstrate different features
                searches = [
                    ("user authentication", "Finding authentication-related code"),
                    ("password", "Finding password-related functions"),
                    ("email validation", "Finding email validation logic"),
                ]

                for query, description in searches:
                    console.print(f"\n[bold blue]🔍 {description}:[/bold blue]")
                    console.print(f"[dim]Query: '{query}'[/dim]\n")

                    search_result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "mcp_vector_search.cli.main",
                            "--project-root",
                            str(demo_dir),
                            "search",
                            "--limit",
                            "2",
                            query,
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if search_result.returncode == 0 and search_result.stdout.strip():
                        console.print(search_result.stdout)
                    else:
                        # Show a friendly message if search didn't return results
                        console.print(
                            "[dim]Note: Search completed but may need more time "
                            "for the index to fully settle.[/dim]"
                        )

            # Show summary
            console.print("\n" + "=" * 70)
            console.print("[bold green]🎉 Demo completed successfully![/bold green]")
            console.print("=" * 70 + "\n")

            console.print("[bold]What you just saw:[/bold]")
            console.print("  ✅ Sample project creation")
            console.print("  ✅ Automatic code indexing")
            if not quick:
                console.print("  ✅ Semantic code search in action")
                console.print("  ✅ Finding code by meaning (not just keywords)\n")

            console.print("[bold cyan]Next steps to use in your project:[/bold cyan]")
            console.print("  1. [green]cd /your/project[/green]")
            console.print("  2. [green]mcp-vector-search init[/green]")
            console.print("  3. [green]mcp-vector-search search 'your query'[/green]\n")

            if keep_files:
                console.print(
                    f"[bold]Demo files saved at:[/bold] [cyan]{demo_dir}[/cyan]"
                )
                console.print(
                    "[yellow]Note: This is still in a temp directory, "
                    "copy files if you want to keep them![/yellow]\n"
                )
            else:
                console.print(
                    "[dim]Demo files will be cleaned up automatically.[/dim]\n"
                )

        finally:
            if not keep_files:
                temp_context.__exit__(None, None, None)
            else:
                # Keep the context manager open
                pass

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print_error(f"Demo failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    demo_app()
