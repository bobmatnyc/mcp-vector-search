"""Profile command for analyzing codebase characteristics."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ...core.codebase_profiler import CodebaseProfiler
from ..output import print_error, print_info, print_success

profile_app = typer.Typer(
    help="Profile codebase characteristics and optimization settings"
)

console = Console()


@profile_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    full_scan: bool = typer.Option(
        False,
        "--full",
        "-f",
        help="Perform full scan instead of sampling (slower but accurate)",
    ),
    show_preset: bool = typer.Option(
        True,
        "--show-preset/--no-preset",
        help="Show recommended optimization preset",
    ),
) -> None:
    """📊 Profile codebase to detect size, type, and optimization opportunities.

    Quickly analyzes your codebase to determine optimal indexing settings.
    By default, samples first 1000 files for speed (< 2 seconds).

    [bold cyan]Examples:[/bold cyan]

    [green]Quick profile (samples 1000 files):[/green]
        $ mcp-vector-search profile

    [green]Full scan (accurate but slower):[/green]
        $ mcp-vector-search profile --full

    [green]Profile without preset recommendation:[/green]
        $ mcp-vector-search profile --no-preset

    [dim]💡 Tip: Profile is automatically run during indexing with --auto-optimize.[/dim]
    """
    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

        print_info("Profiling codebase...\n")

        # Create profiler and run profile
        profiler = CodebaseProfiler(project_root)
        profile = profiler.profile(force_full_scan=full_scan)

        # Print profile summary
        console.print(profiler.format_profile_summary(profile))
        console.print()

        # Create detailed table
        table = Table(title="Language Distribution", show_header=True)
        table.add_column("Extension", style="cyan", width=15)
        table.add_column("Percentage", style="green", justify="right", width=12)
        table.add_column("Type", style="yellow", width=20)

        # Add top languages
        for ext, pct in profile.top_languages:
            lang_type = _classify_extension(ext)
            table.add_row(ext, f"{pct:.1f}%", lang_type)

        # Add "Other" row if there are more languages
        if len(profile.language_distribution) > len(profile.top_languages):
            other_pct = sum(
                pct
                for ext, pct in profile.language_distribution.items()
                if ext not in dict(profile.top_languages)
            )
            table.add_row("Other", f"{other_pct:.1f}%", "Mixed")

        console.print(table)
        console.print()

        # Show optimization preset if requested
        if show_preset:
            preset = profiler.get_optimization_preset(profile)
            console.print(
                profiler.format_optimization_summary(
                    profile, preset, previous_batch_size=32
                )
            )
            console.print()

            # Show what would change
            _print_preset_details(preset)

        print_success("✓ Profile complete")

    except Exception as e:
        print_error(f"Failed to profile codebase: {e}")
        raise typer.Exit(1)


def _classify_extension(ext: str) -> str:
    """Classify file extension into language category.

    Args:
        ext: File extension (e.g., ".py")

    Returns:
        Human-readable language category
    """
    ext = ext.lower()

    if ext == ".py":
        return "Python"
    elif ext in {".js", ".jsx"}:
        return "JavaScript"
    elif ext in {".ts", ".tsx"}:
        return "TypeScript"
    elif ext == ".java":
        return "Java"
    elif ext in {".go"}:
        return "Go"
    elif ext in {".rs"}:
        return "Rust"
    elif ext in {".c", ".h"}:
        return "C"
    elif ext in {".cpp", ".hpp", ".cc", ".cxx"}:
        return "C++"
    elif ext in {".cs"}:
        return "C#"
    elif ext in {".rb"}:
        return "Ruby"
    elif ext in {".php"}:
        return "PHP"
    elif ext in {".swift"}:
        return "Swift"
    elif ext in {".kt", ".kts"}:
        return "Kotlin"
    elif ext in {".scala"}:
        return "Scala"
    elif ext in {".md", ".txt", ".rst"}:
        return "Documentation"
    elif ext in {".json", ".yaml", ".yml", ".toml"}:
        return "Configuration"
    else:
        return "Other"


def _print_preset_details(preset) -> None:
    """Print detailed preset information.

    Args:
        preset: OptimizationPreset to display
    """
    details_table = Table(title="Optimization Settings", show_header=True)
    details_table.add_column("Setting", style="cyan", width=25)
    details_table.add_column("Value", style="green", width=40)

    details_table.add_row("Batch Size", str(preset.batch_size))
    details_table.add_row(
        "Parallel Embeddings", "Enabled" if preset.parallel_embeddings else "Disabled"
    )

    if preset.file_extensions:
        details_table.add_row(
            "File Filter", f"Code-only ({len(preset.file_extensions)} extensions)"
        )
        # Show sample extensions
        sample_exts = sorted(preset.file_extensions)[:10]
        sample_str = ", ".join(sample_exts)
        if len(preset.file_extensions) > 10:
            sample_str += f", ... (+{len(preset.file_extensions) - 10} more)"
        details_table.add_row("Extensions", sample_str)
    else:
        details_table.add_row("File Filter", "All files")

    details_table.add_row("Cache Size", f"{preset.max_cache_size:,} embeddings")

    console.print(details_table)
