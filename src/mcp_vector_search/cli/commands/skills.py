"""CLI commands for skill management."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ...skills import (
    get_available_skills,
    get_skill_content,
    get_skill_metadata,
    install_skill_to_claude,
)


app = typer.Typer(
    name="skills",
    help="🎯 Manage and install MCP Vector Search skills",
)

console = Console()


@app.command("list")
def list_skills() -> None:
    """📋 List available skills."""
    skills = get_available_skills()

    if not skills:
        console.print("[yellow]No skills available[/yellow]")
        return

    table = Table(title="Available MCP Vector Search Skills")
    table.add_column("Skill Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Title", style="white")

    for skill in skills:
        metadata = get_skill_metadata(skill)
        table.add_row(
            skill,
            metadata.get("version", "Unknown"),
            metadata.get("title", "No title"),
        )

    console.print(table)


@app.command("show")
def show_skill(
    skill_name: str = typer.Argument(
        ..., help="Name of the skill to display"
    ),
) -> None:
    """📖 Display skill content."""
    try:
        content = get_skill_content(skill_name)
        console.print(content)
    except Exception as e:
        console.print(f"[red]Error reading skill '{skill_name}': {e}[/red]")
        raise typer.Exit(1)


@app.command("install")
def install_skill(
    skill_name: str = typer.Argument(
        ..., help="Name of the skill to install"
    ),
    claude_dir: Optional[Path] = typer.Option(
        None,
        "--claude-dir",
        help="Claude configuration directory (auto-detected if not specified)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing skill if it exists",
    ),
) -> None:
    """🚀 Install skill to Claude's skills directory."""
    # Auto-detect Claude directory if not specified
    if claude_dir is None:
        possible_dirs = [
            Path.home() / ".config" / "claude" / "skills",
            Path.home() / ".claude" / "skills",
            Path.home() / "AppData" / "Roaming" / "Claude" / "skills",  # Windows
        ]

        for dir_path in possible_dirs:
            if dir_path.parent.exists():
                claude_dir = dir_path
                break

        if claude_dir is None:
            console.print("[red]Could not auto-detect Claude directory.[/red]")
            console.print("Please specify --claude-dir manually")
            raise typer.Exit(1)

    console.print(f"Installing skill '[cyan]{skill_name}[/cyan]' to {claude_dir}")

    # Check if skill exists
    available_skills = get_available_skills()
    if skill_name not in available_skills:
        console.print(f"[red]Skill '{skill_name}' not found.[/red]")
        console.print(f"Available skills: {', '.join(available_skills)}")
        raise typer.Exit(1)

    # Check if already installed
    target_dir = claude_dir / skill_name
    if target_dir.exists() and not force:
        console.print(f"[yellow]Skill already installed at {target_dir}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Install the skill
    success = install_skill_to_claude(skill_name, claude_dir)
    if success:
        console.print(f"[green]✅ Successfully installed skill to {target_dir}[/green]")
        console.print(f"[dim]Skill is now available as: /{skill_name}[/dim]")
    else:
        console.print("[red]❌ Failed to install skill[/red]")
        raise typer.Exit(1)


@app.command("install-all")
def install_all_skills(
    claude_dir: Optional[Path] = typer.Option(
        None,
        "--claude-dir",
        help="Claude configuration directory (auto-detected if not specified)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing skills if they exist",
    ),
) -> None:
    """🚀 Install all available skills to Claude."""
    skills = get_available_skills()
    if not skills:
        console.print("[yellow]No skills available to install[/yellow]")
        return

    console.print(f"Installing {len(skills)} skills...")

    success_count = 0
    for skill in skills:
        try:
            # Use the individual install command logic
            ctx = typer.Context(install_skill)
            ctx.invoke(install_skill, skill_name=skill, claude_dir=claude_dir, force=force)
            success_count += 1
        except typer.Exit:
            continue  # Skip failed installations

    console.print(f"[green]✅ Successfully installed {success_count}/{len(skills)} skills[/green]")


@app.command("uninstall")
def uninstall_skill(
    skill_name: str = typer.Argument(
        ..., help="Name of the skill to uninstall"
    ),
    claude_dir: Optional[Path] = typer.Option(
        None,
        "--claude-dir",
        help="Claude configuration directory (auto-detected if not specified)",
    ),
) -> None:
    """🗑️ Uninstall skill from Claude's skills directory."""
    # Auto-detect Claude directory logic (same as install)
    if claude_dir is None:
        possible_dirs = [
            Path.home() / ".config" / "claude" / "skills",
            Path.home() / ".claude" / "skills",
            Path.home() / "AppData" / "Roaming" / "Claude" / "skills",
        ]

        for dir_path in possible_dirs:
            if dir_path.exists():
                claude_dir = dir_path
                break

        if claude_dir is None:
            console.print("[red]Could not find Claude skills directory.[/red]")
            raise typer.Exit(1)

    target_dir = claude_dir / skill_name
    if not target_dir.exists():
        console.print(f"[yellow]Skill '{skill_name}' not found at {target_dir}[/yellow]")
        raise typer.Exit(1)

    # Remove the skill directory
    import shutil
    shutil.rmtree(target_dir)
    console.print(f"[green]✅ Successfully uninstalled skill from {target_dir}[/green]")