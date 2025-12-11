"""Console reporter for code analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from ..metrics import ProjectMetrics

console = Console()


class ConsoleReporter:
    """Console reporter for displaying analysis results in terminal."""

    def print_summary(self, metrics: ProjectMetrics) -> None:
        """Print high-level project summary.

        Args:
            metrics: Project metrics to display
        """
        console.print("\n[bold blue]📈 Code Complexity Analysis[/bold blue]")
        console.print("━" * 60)
        console.print()

        console.print("[bold]Project Summary[/bold]")
        console.print(f"  Files Analyzed: {metrics.total_files}")
        console.print(f"  Total Lines: {metrics.total_lines:,}")
        console.print(f"  Functions: {metrics.total_functions}")
        console.print(f"  Classes: {metrics.total_classes}")
        console.print(f"  Avg File Complexity: {metrics.avg_file_complexity:.1f}")
        console.print()

    def print_distribution(self, metrics: ProjectMetrics) -> None:
        """Print complexity grade distribution.

        Args:
            metrics: Project metrics with grade distribution
        """
        console.print("[bold]Complexity Distribution[/bold]")

        # Get grade distribution
        distribution = metrics._compute_grade_distribution()
        total_chunks = sum(distribution.values())

        if total_chunks == 0:
            console.print("  No functions/methods analyzed")
            console.print()
            return

        # Define grade colors and descriptions
        grade_info = {
            "A": ("green", "Excellent (0-5)"),
            "B": ("blue", "Good (6-10)"),
            "C": ("yellow", "Acceptable (11-20)"),
            "D": ("orange1", "Needs Improvement (21-30)"),
            "F": ("red", "Refactor Required (31+)"),
        }

        # Print distribution table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Grade", style="bold", width=8)
        table.add_column("Description", width=25)
        table.add_column("Count", justify="right", width=8)
        table.add_column("Percentage", justify="right", width=10)
        table.add_column("Bar", width=20)

        for grade in ["A", "B", "C", "D", "F"]:
            count = distribution.get(grade, 0)
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            color, description = grade_info[grade]

            # Create visual bar
            bar_length = int(percentage / 5)  # Scale: 5% = 1 char
            bar = "█" * bar_length

            table.add_row(
                f"[{color}]{grade}[/{color}]",
                description,
                f"{count}",
                f"{percentage:.1f}%",
                f"[{color}]{bar}[/{color}]",
            )

        console.print(table)
        console.print()

    def print_hotspots(self, metrics: ProjectMetrics, top: int = 10) -> None:
        """Print complexity hotspots.

        Args:
            metrics: Project metrics
            top: Number of top hotspots to display
        """
        hotspot_files = metrics.get_hotspots(limit=top)

        if not hotspot_files:
            console.print("[bold]🔥 Complexity Hotspots[/bold]")
            console.print("  No hotspots found")
            console.print()
            return

        console.print(
            f"[bold]🔥 Top {min(top, len(hotspot_files))} Complexity Hotspots[/bold]"
        )

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Rank", justify="right", width=6)
        table.add_column("File", style="cyan", width=50)
        table.add_column("Avg Complexity", justify="right", width=16)
        table.add_column("Grade", justify="center", width=8)
        table.add_column("Functions", justify="right", width=10)

        for rank, file_metrics in enumerate(hotspot_files, 1):
            # Compute average grade
            if file_metrics.chunks:
                grades = [chunk.complexity_grade for chunk in file_metrics.chunks]
                avg_grade = max(set(grades), key=grades.count)  # Most common grade
            else:
                avg_grade = "N/A"

            # Color code grade
            grade_colors = {
                "A": "green",
                "B": "blue",
                "C": "yellow",
                "D": "orange1",
                "F": "red",
            }
            grade_color = grade_colors.get(avg_grade, "white")

            # Truncate file path if too long
            file_path = file_metrics.file_path
            if len(file_path) > 48:
                file_path = "..." + file_path[-45:]

            table.add_row(
                f"{rank}",
                file_path,
                f"{file_metrics.avg_complexity:.1f}",
                f"[{grade_color}]{avg_grade}[/{grade_color}]",
                f"{len(file_metrics.chunks)}",
            )

        console.print(table)
        console.print()

    def print_smells(self, smells: list, top: int = 10) -> None:
        """Print detected code smells.

        Args:
            smells: List of CodeSmell objects
            top: Maximum number of smells to display
        """
        from ..collectors.smells import SmellSeverity

        if not smells:
            console.print("[bold]🔍 Code Smells[/bold]")
            console.print("  No code smells detected!")
            console.print()
            return

        console.print(
            f"[bold]🔍 Code Smells Detected[/bold] - Found {len(smells)} issues"
        )

        # Group smells by severity
        error_smells = [s for s in smells if s.severity == SmellSeverity.ERROR]
        warning_smells = [s for s in smells if s.severity == SmellSeverity.WARNING]
        info_smells = [s for s in smells if s.severity == SmellSeverity.INFO]

        # Summary
        console.print(
            f"  [red]Errors: {len(error_smells)}[/red]  "
            f"[yellow]Warnings: {len(warning_smells)}[/yellow]  "
            f"[blue]Info: {len(info_smells)}[/blue]"
        )
        console.print()

        # Show top smells (prioritize errors first)
        smells_to_display = error_smells + warning_smells + info_smells
        smells_to_display = smells_to_display[:top]

        # Create table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Severity", width=10)
        table.add_column("Smell Type", width=20)
        table.add_column("Location", width=40)
        table.add_column("Details", width=30)

        for smell in smells_to_display:
            # Color code by severity
            if smell.severity == SmellSeverity.ERROR:
                severity_str = "[red]ERROR[/red]"
            elif smell.severity == SmellSeverity.WARNING:
                severity_str = "[yellow]WARNING[/yellow]"
            else:
                severity_str = "[blue]INFO[/blue]"

            # Truncate location if too long
            location = smell.location
            if len(location) > 38:
                location = "..." + location[-35:]

            # Format details (metric value vs threshold)
            details = f"{smell.metric_value} > {smell.threshold}"

            table.add_row(severity_str, smell.name, location, details)

        console.print(table)
        console.print()

        # Show suggestions for top smells
        if smells_to_display:
            console.print("[bold]💡 Top Suggestions[/bold]")
            shown_suggestions = set()
            suggestion_count = 0

            for smell in smells_to_display:
                if smell.suggestion and smell.suggestion not in shown_suggestions:
                    console.print(f"  • [dim]{smell.suggestion}[/dim]")
                    shown_suggestions.add(smell.suggestion)
                    suggestion_count += 1

                    # Limit to 5 unique suggestions
                    if suggestion_count >= 5:
                        break

        console.print()

    def print_recommendations(self, metrics: ProjectMetrics) -> None:
        """Print actionable recommendations.

        Args:
            metrics: Project metrics
        """
        console.print("[bold]💡 Recommendations[/bold]")

        recommendations: list[str] = []

        # Check for files needing attention
        files_needing_attention = metrics._count_files_needing_attention()
        if files_needing_attention > 0:
            recommendations.append(
                f"[yellow]•[/yellow] {files_needing_attention} files have health score below 0.7 - consider refactoring"
            )

        # Check for high complexity files
        hotspots = metrics.get_hotspots(limit=5)
        high_complexity_files = [f for f in hotspots if f.avg_complexity > 20]
        if high_complexity_files:
            recommendations.append(
                f"[yellow]•[/yellow] {len(high_complexity_files)} files have average complexity > 20 - prioritize these for refactoring"
            )

        # Check grade distribution
        distribution = metrics._compute_grade_distribution()
        total_chunks = sum(distribution.values())
        if total_chunks > 0:
            d_f_percentage = (
                (distribution.get("D", 0) + distribution.get("F", 0))
                / total_chunks
                * 100
            )
            if d_f_percentage > 20:
                recommendations.append(
                    f"[yellow]•[/yellow] {d_f_percentage:.1f}% of functions have D/F grades - aim to reduce this below 10%"
                )

        # Check overall health
        avg_health = metrics._compute_avg_health_score()
        if avg_health < 0.7:
            recommendations.append(
                f"[yellow]•[/yellow] Average health score is {avg_health:.2f} - target 0.8+ through refactoring"
            )
        elif avg_health >= 0.9:
            recommendations.append(
                "[green]✓[/green] Excellent code health! Keep up the good work."
            )

        if not recommendations:
            recommendations.append(
                "[green]✓[/green] Code quality looks good! No critical issues found."
            )

        for rec in recommendations:
            console.print(f"  {rec}")

        console.print()

        # Print tips
        console.print("[dim]💡 Tips:[/dim]")
        console.print(
            "[dim]  • Use [cyan]--top N[/cyan] to see more/fewer hotspots[/dim]"
        )
        console.print(
            "[dim]  • Use [cyan]--json[/cyan] to export results for further analysis[/dim]"
        )
        console.print(
            "[dim]  • Focus refactoring efforts on Grade D and F functions first[/dim]"
        )
        console.print()
