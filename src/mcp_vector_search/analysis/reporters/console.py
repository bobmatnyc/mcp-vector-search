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

    def print_instability(self, metrics: ProjectMetrics, top: int = 10) -> None:
        """Print instability metrics.

        Args:
            metrics: Project metrics
            top: Number of top files to display
        """
        console.print("[bold]🔗 Instability Metrics[/bold]")

        # Collect instability data from files with coupling metrics
        files_with_coupling = [
            (f.file_path, f.coupling)
            for f in metrics.files.values()
            if f.coupling.efferent_coupling + f.coupling.afferent_coupling > 0
        ]

        if not files_with_coupling:
            console.print("  No coupling data available")
            console.print()
            return

        # Compute instability distribution
        stable_count = sum(1 for _, c in files_with_coupling if c.instability <= 0.3)
        balanced_count = sum(
            1 for _, c in files_with_coupling if 0.3 < c.instability <= 0.7
        )
        unstable_count = sum(1 for _, c in files_with_coupling if c.instability > 0.7)
        total_files = len(files_with_coupling)

        console.print(f"  Total Files: {total_files}")
        console.print(
            f"  [green]Stable (I ≤ 0.3):[/green] {stable_count} "
            f"({stable_count / total_files * 100:.1f}%)"
        )
        console.print(
            f"  [yellow]Balanced (0.3 < I ≤ 0.7):[/yellow] {balanced_count} "
            f"({balanced_count / total_files * 100:.1f}%)"
        )
        console.print(
            f"  [red]Unstable (I > 0.7):[/red] {unstable_count} "
            f"({unstable_count / total_files * 100:.1f}%)"
        )
        console.print()

        # Show most stable files
        most_stable = sorted(files_with_coupling, key=lambda x: x[1].instability)[:top]
        if most_stable:
            console.print(
                f"[bold green]✓ Most Stable Files (Top {len(most_stable)})[/bold green]"
            )

            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("Rank", justify="right", width=6)
            table.add_column("File", style="cyan", width=45)
            table.add_column("Instability", justify="right", width=12)
            table.add_column("Category", justify="center", width=12)
            table.add_column("Ce/Ca", justify="right", width=10)

            for rank, (file_path, coupling) in enumerate(most_stable, 1):
                # Truncate file path if too long
                display_path = file_path
                if len(display_path) > 43:
                    display_path = "..." + display_path[-40:]

                # Determine category color
                if coupling.instability <= 0.3:
                    category = "[green]Stable[/green]"
                elif coupling.instability <= 0.7:
                    category = "[yellow]Balanced[/yellow]"
                else:
                    category = "[red]Unstable[/red]"

                table.add_row(
                    f"{rank}",
                    display_path,
                    f"{coupling.instability:.3f}",
                    category,
                    f"{coupling.efferent_coupling}/{coupling.afferent_coupling}",
                )

            console.print(table)
            console.print()

        # Show most unstable files
        most_unstable = sorted(
            files_with_coupling, key=lambda x: x[1].instability, reverse=True
        )[:top]
        if most_unstable:
            console.print(
                f"[bold red]⚠️  Most Unstable Files (Top {len(most_unstable)})[/bold red]"
            )

            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("Rank", justify="right", width=6)
            table.add_column("File", style="cyan", width=45)
            table.add_column("Instability", justify="right", width=12)
            table.add_column("Category", justify="center", width=12)
            table.add_column("Ce/Ca", justify="right", width=10)

            for rank, (file_path, coupling) in enumerate(most_unstable, 1):
                # Truncate file path if too long
                display_path = file_path
                if len(display_path) > 43:
                    display_path = "..." + display_path[-40:]

                # Determine category color
                if coupling.instability <= 0.3:
                    category = "[green]Stable[/green]"
                elif coupling.instability <= 0.7:
                    category = "[yellow]Balanced[/yellow]"
                else:
                    category = "[red]Unstable[/red]"

                table.add_row(
                    f"{rank}",
                    display_path,
                    f"{coupling.instability:.3f}",
                    category,
                    f"{coupling.efferent_coupling}/{coupling.afferent_coupling}",
                )

            console.print(table)
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

        # Check for highly unstable files (instability > 0.8)
        highly_unstable = [
            f
            for f in metrics.files.values()
            if f.coupling.instability > 0.8
            and (f.coupling.efferent_coupling + f.coupling.afferent_coupling) > 0
        ]
        if highly_unstable:
            recommendations.append(
                f"[yellow]•[/yellow] {len(highly_unstable)} files have instability > 0.8 - consider reducing dependencies"
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
        console.print(
            "[dim]  • Stable files (I ≤ 0.3) should contain abstractions and core logic[/dim]"
        )
        console.print(
            "[dim]  • Unstable files (I > 0.7) should contain concrete implementations[/dim]"
        )
        console.print()

    def print_baseline_comparison(self, comparison_result) -> None:
        """Print baseline comparison results.

        Args:
            comparison_result: ComparisonResult from BaselineComparator
        """
        console.print(
            f"\n[bold blue]📊 Baseline Comparison[/bold blue] - vs {comparison_result.baseline_name}"
        )
        console.print("━" * 80)
        console.print()

        # Summary statistics
        console.print("[bold]Summary[/bold]")
        summary = comparison_result.summary
        console.print(
            f"  Total Files Compared: {comparison_result.total_files_compared}"
        )
        console.print(
            f"  Files - Current: {summary.get('total_files_current', 0)} | "
            f"Baseline: {summary.get('total_files_baseline', 0)}"
        )
        console.print(
            f"  Functions - Current: {summary.get('total_functions_current', 0)} | "
            f"Baseline: {summary.get('total_functions_baseline', 0)}"
        )
        console.print()

        # Change summary
        console.print("[bold]Changes[/bold]")
        console.print(
            f"  [red]Regressions:[/red] {len(comparison_result.regressions)} files"
        )
        console.print(
            f"  [green]Improvements:[/green] {len(comparison_result.improvements)} files"
        )
        console.print(
            f"  [dim]Unchanged:[/dim] {len(comparison_result.unchanged)} files"
        )
        console.print(
            f"  [blue]New Files:[/blue] {len(comparison_result.new_files)} files"
        )
        console.print(
            f"  [yellow]Deleted Files:[/yellow] {len(comparison_result.deleted_files)} files"
        )
        console.print()

        # Complexity metrics comparison
        avg_cc_current = summary.get("avg_complexity_current", 0.0)
        avg_cc_baseline = summary.get("avg_complexity_baseline", 0.0)
        avg_cc_delta = avg_cc_current - avg_cc_baseline
        avg_cc_pct = (
            (avg_cc_delta / avg_cc_baseline * 100) if avg_cc_baseline > 0 else 0.0
        )

        max_cc_current = summary.get("max_complexity_current", 0)
        max_cc_baseline = summary.get("max_complexity_baseline", 0)
        max_cc_delta = max_cc_current - max_cc_baseline

        console.print("[bold]Complexity Metrics[/bold]")

        # Average complexity with color coding
        delta_color = (
            "red" if avg_cc_delta > 0 else "green" if avg_cc_delta < 0 else "dim"
        )
        delta_sign = "+" if avg_cc_delta > 0 else ""
        console.print(
            f"  Avg Complexity: {avg_cc_current:.2f} "
            f"(baseline: {avg_cc_baseline:.2f}, "
            f"[{delta_color}]{delta_sign}{avg_cc_delta:.2f} / {delta_sign}{avg_cc_pct:.1f}%[/{delta_color}])"
        )

        # Max complexity with color coding
        max_delta_color = (
            "red" if max_cc_delta > 0 else "green" if max_cc_delta < 0 else "dim"
        )
        max_delta_sign = "+" if max_cc_delta > 0 else ""
        console.print(
            f"  Max Complexity: {max_cc_current} "
            f"(baseline: {max_cc_baseline}, "
            f"[{max_delta_color}]{max_delta_sign}{max_cc_delta}[/{max_delta_color}])"
        )
        console.print()

        # Show regressions
        if comparison_result.regressions:
            console.print(
                f"[bold red]⚠️  Regressions ({len(comparison_result.regressions)} files)[/bold red]"
            )

            # Show top 10 regressions
            top_regressions = comparison_result.regressions[:10]

            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("File", style="cyan", width=45)
            table.add_column("Metric", width=20)
            table.add_column("Change", justify="right", width=15)

            for file_comp in top_regressions:
                # Truncate file path
                file_path = file_comp.file_path
                if len(file_path) > 43:
                    file_path = "..." + file_path[-40:]

                # Show worst regression metric for this file
                regression_changes = [
                    c for c in file_comp.metric_changes if c.is_regression
                ]
                if regression_changes:
                    worst_change = max(
                        regression_changes, key=lambda c: abs(c.percentage_delta)
                    )
                    table.add_row(
                        file_path,
                        worst_change.metric_name.replace("_", " ").title(),
                        f"+{worst_change.percentage_delta:.1f}%",
                    )

            console.print(table)

            if len(comparison_result.regressions) > 10:
                console.print(
                    f"  [dim]... and {len(comparison_result.regressions) - 10} more[/dim]"
                )
            console.print()

        # Show improvements
        if comparison_result.improvements:
            console.print(
                f"[bold green]✓ Improvements ({len(comparison_result.improvements)} files)[/bold green]"
            )

            # Show top 10 improvements
            top_improvements = comparison_result.improvements[:10]

            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("File", style="cyan", width=45)
            table.add_column("Metric", width=20)
            table.add_column("Change", justify="right", width=15)

            for file_comp in top_improvements:
                # Truncate file path
                file_path = file_comp.file_path
                if len(file_path) > 43:
                    file_path = "..." + file_path[-40:]

                # Show best improvement metric for this file
                improvement_changes = [
                    c for c in file_comp.metric_changes if c.is_improvement
                ]
                if improvement_changes:
                    best_change = max(
                        improvement_changes, key=lambda c: abs(c.percentage_delta)
                    )
                    table.add_row(
                        file_path,
                        best_change.metric_name.replace("_", " ").title(),
                        f"{best_change.percentage_delta:.1f}%",
                    )

            console.print(table)

            if len(comparison_result.improvements) > 10:
                console.print(
                    f"  [dim]... and {len(comparison_result.improvements) - 10} more[/dim]"
                )
            console.print()

        # Show new/deleted files summary
        if comparison_result.new_files:
            console.print(
                f"[bold blue]📄 New Files ({len(comparison_result.new_files)})[/bold blue]"
            )
            for file_comp in comparison_result.new_files[:5]:
                file_path = file_comp.file_path
                if len(file_path) > 70:
                    file_path = "..." + file_path[-67:]
                console.print(f"  • {file_path}")
            if len(comparison_result.new_files) > 5:
                console.print(
                    f"  [dim]... and {len(comparison_result.new_files) - 5} more[/dim]"
                )
            console.print()

        if comparison_result.deleted_files:
            console.print(
                f"[bold yellow]🗑  Deleted Files ({len(comparison_result.deleted_files)})[/bold yellow]"
            )
            for file_comp in comparison_result.deleted_files[:5]:
                file_path = file_comp.file_path
                if len(file_path) > 70:
                    file_path = "..." + file_path[-67:]
                console.print(f"  • {file_path}")
            if len(comparison_result.deleted_files) > 5:
                console.print(
                    f"  [dim]... and {len(comparison_result.deleted_files) - 5} more[/dim]"
                )
            console.print()
