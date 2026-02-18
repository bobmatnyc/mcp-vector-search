"""Engineer analysis - profile developers based on code quality metrics."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...analysis.code_quality import (
    FunctionMetrics,
    analyze_python_file,
    get_commit_stats,
    get_function_author,
    get_total_lines_by_author,
)
from ..output import console

app = typer.Typer()


@dataclass
class EngineerProfile:
    """Profile of an engineer's code quality."""

    name: str
    email: str = ""
    total_lines: int = 0
    total_commits: int = 0

    # Quality metrics
    high_complexity_functions: int = 0  # cyclomatic > 10
    very_high_complexity: int = 0  # cyclomatic > 20
    long_functions: int = 0  # > 50 lines
    very_long_functions: int = 0  # > 100 lines

    # Code smells
    deeply_nested: int = 0  # nesting > 4
    long_parameter_lists: int = 0  # > 5 params

    # Track all functions for detailed analysis
    functions: list[FunctionMetrics] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Calculate quality score (0-100, higher is better).

        Score is based on issues per 1000 lines of code.
        - 0 issues = 100 score
        - More issues = lower score
        """
        if self.total_lines == 0:
            return 100.0

        # Weight different issues by severity
        issues = (
            self.high_complexity_functions * 2
            + self.very_high_complexity * 5
            + self.long_functions * 1
            + self.very_long_functions * 3
            + self.deeply_nested * 2
            + self.long_parameter_lists * 1
        )

        # Normalize to issues per 1000 lines
        issues_per_kloc = (issues / self.total_lines) * 1000

        # Convert to 0-100 scale (10 issues/kloc = 0 score)
        score = max(0, 100 - (issues_per_kloc * 10))
        return round(score, 1)

    @property
    def avg_complexity(self) -> float:
        """Average cyclomatic complexity of all functions."""
        if not self.functions:
            return 0.0
        return round(
            sum(f.cyclomatic_complexity for f in self.functions) / len(self.functions),
            1,
        )


@app.command("engineers")
def analyze_engineers(
    project_root: Path = typer.Option(
        Path.cwd(),
        "--path",
        "-p",
        help="Project root directory",
    ),
    min_commits: int = typer.Option(
        5,
        "--min-commits",
        help="Minimum commits to include engineer",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all engineers (not just summary)",
    ),
    top_n: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of engineers to show in summary mode",
    ),
) -> None:
    """👥 Analyze engineer code quality profiles.

    Examines code complexity, smells, and patterns correlated with
    git commit authors to build quality profiles.

    [bold cyan]Examples:[/bold cyan]

    [green]Basic analysis:[/green]
        $ mcp-vector-search analyze engineers

    [green]Show all engineers:[/green]
        $ mcp-vector-search analyze engineers --all

    [green]Filter by minimum commits:[/green]
        $ mcp-vector-search analyze engineers --min-commits 10

    [green]Show top 20 engineers:[/green]
        $ mcp-vector-search analyze engineers --top 20

    [dim]💡 Uses git blame to correlate code quality with authors.[/dim]
    """
    asyncio.run(
        run_engineer_analysis(
            project_root=project_root,
            min_commits=min_commits,
            show_all=show_all,
            top_n=top_n,
        )
    )


async def run_engineer_analysis(
    project_root: Path,
    min_commits: int,
    show_all: bool,
    top_n: int,
) -> None:
    """Run engineer analysis workflow.

    Args:
        project_root: Root directory of project
        min_commits: Minimum commits to include engineer
        show_all: Show all engineers instead of just top/bottom
        top_n: Number of engineers to show in summary mode
    """
    console.print("\n[bold blue]👥 Analyzing Engineer Profiles[/bold blue]\n")

    # Step 1: Get commit statistics
    console.print("[dim]Collecting git commit statistics...[/dim]")
    commit_stats = get_commit_stats(project_root)

    if not commit_stats:
        console.print("[red]❌ No git history found. Is this a git repository?[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Found {len(commit_stats)} contributors[/dim]")

    # Step 2: Get total lines by author
    console.print("[dim]Analyzing code ownership...[/dim]")
    author_lines = get_total_lines_by_author(project_root)

    # Step 3: Find all Python files
    console.print("[dim]Scanning Python files...[/dim]")
    python_files = list(project_root.rglob("*.py"))

    # Filter out common ignore directories
    ignore_dirs = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".tox",
        ".eggs",
        "vendor",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "htmlcov",
        "site-packages",
        ".nox",
        "env",
        ".env",
        ".cache",
        ".uv",
    }

    filtered_files = []
    for file_path in python_files:
        # Skip files in ignored directories
        if any(ignored in file_path.parts for ignored in ignore_dirs):
            continue
        filtered_files.append(file_path)

    console.print(f"[dim]Analyzing {len(filtered_files)} Python files...[/dim]")

    # Step 4: Analyze all functions and correlate with authors
    profiles: dict[str, EngineerProfile] = {}

    # Initialize profiles from commit stats
    for name, stats in commit_stats.items():
        if stats["commits"] >= min_commits:
            profiles[name] = EngineerProfile(
                name=name,
                email=stats["email"],
                total_commits=stats["commits"],
                total_lines=author_lines.get(name, 0),
            )

    # Analyze each file
    function_count = 0
    for file_path in filtered_files:
        try:
            metrics_list = analyze_python_file(file_path)
            function_count += len(metrics_list)

            # Get author for each function
            for metrics in metrics_list:
                author = get_function_author(
                    project_root, file_path, metrics.start_line
                )

                if author and author in profiles:
                    # Update profile with function metrics
                    profile = profiles[author]
                    profile.functions.append(metrics)

                    # Count quality issues
                    if metrics.is_complex:
                        profile.high_complexity_functions += 1
                    if metrics.is_very_complex:
                        profile.very_high_complexity += 1
                    if metrics.is_long:
                        profile.long_functions += 1
                    if metrics.is_very_long:
                        profile.very_long_functions += 1
                    if metrics.is_deeply_nested:
                        profile.deeply_nested += 1
                    if metrics.has_many_parameters:
                        profile.long_parameter_lists += 1

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to analyze {file_path}: {e}[/yellow]")
            continue

    console.print(f"[dim]Analyzed {function_count} functions[/dim]\n")

    # Step 5: Display results
    if not profiles:
        console.print("[yellow]⚠️  No engineers found matching criteria[/yellow]")
        console.print(
            f"[dim]Try lowering --min-commits (current: {min_commits})[/dim]"
        )
        raise typer.Exit(0)

    display_results(profiles, show_all, top_n)


def display_results(
    profiles: dict[str, EngineerProfile],
    show_all: bool,
    top_n: int,
) -> None:
    """Display engineer profiles in a rich table.

    Args:
        profiles: Dict of engineer profiles
        show_all: Show all engineers or just top/bottom
        top_n: Number of top/bottom engineers to show
    """
    # Convert to sorted list by quality score
    sorted_profiles = sorted(profiles.values(), key=lambda p: p.quality_score)

    # Filter profiles with at least one function
    active_profiles = [p for p in sorted_profiles if p.functions]

    if not active_profiles:
        console.print("[yellow]⚠️  No code attributed to engineers[/yellow]")
        console.print(
            "[dim]This might happen if git blame fails or files are too new[/dim]"
        )
        return

    # Determine which profiles to show
    if show_all:
        profiles_to_show = active_profiles
    else:
        # Show top N best and bottom N worst
        if len(active_profiles) <= top_n * 2:
            profiles_to_show = active_profiles
        else:
            worst = active_profiles[:top_n]
            best = active_profiles[-top_n:]
            profiles_to_show = worst + best

    # Create table
    table = Table(
        title=f"👥 Engineer Code Quality Profiles ({len(active_profiles)} total)",
        show_header=True,
    )
    table.add_column("Engineer", style="cyan", no_wrap=True)
    table.add_column("Commits", justify="right")
    table.add_column("Functions", justify="right")
    table.add_column("Avg Complexity", justify="right")
    table.add_column("Issues", justify="right", style="yellow")
    table.add_column("Quality Score", justify="right")
    table.add_column("Grade", justify="center")

    def get_grade(score: float) -> str:
        """Get letter grade with color."""
        if score >= 90:
            return "[green]A[/green]"
        if score >= 80:
            return "[green]B[/green]"
        if score >= 70:
            return "[yellow]C[/yellow]"
        if score >= 60:
            return "[yellow]D[/yellow]"
        return "[red]F[/red]"

    for profile in profiles_to_show:
        # Calculate total issues
        total_issues = (
            profile.high_complexity_functions
            + profile.long_functions
            + profile.deeply_nested
            + profile.long_parameter_lists
        )

        table.add_row(
            profile.name,
            str(profile.total_commits),
            str(len(profile.functions)),
            f"{profile.avg_complexity:.1f}",
            str(total_issues),
            f"{profile.quality_score:.1f}",
            get_grade(profile.quality_score),
        )

    console.print(table)

    # Show insights
    if active_profiles:
        worst = active_profiles[0]
        best = active_profiles[-1]

        # Calculate project averages
        avg_score = sum(p.quality_score for p in active_profiles) / len(active_profiles)
        avg_complexity = sum(p.avg_complexity for p in active_profiles) / len(
            active_profiles
        )

        console.print()
        console.print(
            Panel(
                f"[red]⚠️  Most Issues:[/red] {worst.name} "
                f"(score: {worst.quality_score}, {len(worst.functions)} functions)\n"
                f"[green]✨ Cleanest Code:[/green] {best.name} "
                f"(score: {best.quality_score}, {len(best.functions)} functions)\n\n"
                f"[cyan]📊 Project Average:[/cyan] Quality Score: {avg_score:.1f}, "
                f"Avg Complexity: {avg_complexity:.1f}",
                title="Insights",
            )
        )

        # Show detailed breakdown for worst performer
        if worst.quality_score < 70:
            console.print()
            console.print(
                f"[bold red]🔍 Issues for {worst.name}:[/bold red]",
            )

            issue_table = Table(show_header=True, box=None)
            issue_table.add_column("Issue Type", style="yellow")
            issue_table.add_column("Count", justify="right")

            if worst.high_complexity_functions > 0:
                issue_table.add_row(
                    "High Complexity (>10)", str(worst.high_complexity_functions)
                )
            if worst.very_high_complexity > 0:
                issue_table.add_row(
                    "Very High Complexity (>20)", str(worst.very_high_complexity)
                )
            if worst.long_functions > 0:
                issue_table.add_row("Long Functions (>50 lines)", str(worst.long_functions))
            if worst.very_long_functions > 0:
                issue_table.add_row(
                    "Very Long Functions (>100 lines)", str(worst.very_long_functions)
                )
            if worst.deeply_nested > 0:
                issue_table.add_row("Deeply Nested (>4 levels)", str(worst.deeply_nested))
            if worst.long_parameter_lists > 0:
                issue_table.add_row(
                    "Long Parameter Lists (>5)", str(worst.long_parameter_lists)
                )

            console.print(issue_table)


if __name__ == "__main__":
    app()
