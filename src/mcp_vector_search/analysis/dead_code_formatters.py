"""Output formatters for dead code analysis."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from typing import TextIO

from rich.console import Console
from rich.table import Table

from .dead_code import Confidence, DeadCodeFinding, DeadCodeReport
from .entry_points import EntryPoint, EntryPointType


class DeadCodeFormatter(ABC):
    """Base class for dead code report formatters."""

    @abstractmethod
    def format(self, report: DeadCodeReport, output: TextIO = sys.stdout) -> None:
        """Format and output the report.

        Args:
            report: Dead code analysis report
            output: Output stream (default: stdout)
        """
        pass


class ConsoleFormatter(DeadCodeFormatter):
    """Human-readable console output with colors using Rich."""

    def __init__(self) -> None:
        """Initialize console formatter."""
        self.console = Console()

    def format(self, report: DeadCodeReport, output: TextIO = sys.stdout) -> None:
        """Format report as colored console output.

        Args:
            report: Dead code analysis report
            output: Output stream (default: stdout)
        """
        # Override console output stream
        self.console = Console(file=output, force_terminal=True)

        # Header
        self.console.print("\n[bold cyan]Dead Code Analysis Report[/bold cyan]")
        self.console.print("═" * 80)

        # Entry points section
        self._print_entry_points(report.entry_points)

        # Findings section
        self._print_findings(report.findings)

        # Summary section
        self._print_summary(report)

        self.console.print()

    def _print_entry_points(self, entry_points: list[EntryPoint]) -> None:
        """Print entry points section.

        Args:
            entry_points: List of detected entry points
        """
        self.console.print(f"\n[bold]Entry Points Detected:[/bold] {len(entry_points)}")

        # Group by type
        by_type: dict[EntryPointType, list[EntryPoint]] = {}
        for ep in entry_points:
            by_type.setdefault(ep.type, []).append(ep)

        # Print grouped
        for ep_type, eps in sorted(by_type.items(), key=lambda x: x[0].value):
            for ep in eps[:5]:  # Show first 5 per type
                self.console.print(
                    f"  • [cyan]{ep.name}[/cyan] in {ep.file_path}:{ep.line_number} ({ep_type.value})"
                )
            if len(eps) > 5:
                self.console.print(
                    f"  ... and {len(eps) - 5} more {ep_type.value} entry points"
                )

    def _print_findings(self, findings: list[DeadCodeFinding]) -> None:
        """Print findings grouped by confidence.

        Args:
            findings: List of dead code findings
        """
        self.console.print(f"\n[bold]Dead Code Findings:[/bold] {len(findings)}\n")

        # Group by confidence
        by_confidence: dict[Confidence, list[DeadCodeFinding]] = {
            Confidence.HIGH: [],
            Confidence.MEDIUM: [],
            Confidence.LOW: [],
        }

        for finding in findings:
            by_confidence[finding.confidence].append(finding)

        # Print HIGH confidence
        if by_confidence[Confidence.HIGH]:
            self.console.print(
                f"[bold red][HIGH CONFIDENCE][/bold red] ({len(by_confidence[Confidence.HIGH])} findings)"
            )
            self.console.print("─" * 80)
            for finding in by_confidence[Confidence.HIGH][:10]:  # Show first 10
                self._print_finding(finding, "red")
            if len(by_confidence[Confidence.HIGH]) > 10:
                self.console.print(
                    f"\n  ... and {len(by_confidence[Confidence.HIGH]) - 10} more high confidence findings\n"
                )

        # Print MEDIUM confidence
        if by_confidence[Confidence.MEDIUM]:
            self.console.print(
                f"\n[bold yellow][MEDIUM CONFIDENCE][/bold yellow] ({len(by_confidence[Confidence.MEDIUM])} findings)"
            )
            self.console.print("─" * 80)
            for finding in by_confidence[Confidence.MEDIUM][:10]:
                self._print_finding(finding, "yellow")
            if len(by_confidence[Confidence.MEDIUM]) > 10:
                self.console.print(
                    f"\n  ... and {len(by_confidence[Confidence.MEDIUM]) - 10} more medium confidence findings\n"
                )

        # Print LOW confidence
        if by_confidence[Confidence.LOW]:
            self.console.print(
                f"\n[bold dim][LOW CONFIDENCE][/bold dim] ({len(by_confidence[Confidence.LOW])} findings)"
            )
            self.console.print("─" * 80)
            for finding in by_confidence[Confidence.LOW][:10]:
                self._print_finding(finding, "dim")
            if len(by_confidence[Confidence.LOW]) > 10:
                self.console.print(
                    f"\n  ... and {len(by_confidence[Confidence.LOW]) - 10} more low confidence findings\n"
                )

    def _print_finding(self, finding: DeadCodeFinding, color: str) -> None:
        """Print a single finding.

        Args:
            finding: Dead code finding
            color: Color style to use
        """
        symbol = "⚠️" if color == "red" else "?" if color == "yellow" else "ℹ"
        self.console.print(f"  {symbol}  [{color}]{finding.function_name}[/{color}]")
        self.console.print(
            f"      File: {finding.file_path}:{finding.start_line}-{finding.end_line}"
        )
        self.console.print(f"      Reason: {finding.reason}")

        if finding.caveats:
            self.console.print(f"      Caveats: {', '.join(finding.caveats)}")
        self.console.print()

    def _print_summary(self, report: DeadCodeReport) -> None:
        """Print summary statistics.

        Args:
            report: Dead code analysis report
        """
        self.console.print("\n[bold]Summary[/bold]")
        self.console.print("═" * 80)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total Functions", str(report.total_functions))
        table.add_row(
            "Reachable",
            f"{report.reachable_count} ({report.reachable_percentage:.1f}%)",
        )
        table.add_row(
            "Unreachable",
            f"{report.unreachable_count} ({100 - report.reachable_percentage:.1f}%)",
        )

        self.console.print(table)
        self.console.print(
            "\n[dim]💡 Tip: Use --min-confidence high to show only high-confidence findings.[/dim]"
        )
        self.console.print(
            "[dim]💡 Tip: Use --output json for machine-readable output.[/dim]"
        )


class JsonFormatter(DeadCodeFormatter):
    """JSON output for tooling and CI/CD integration."""

    def format(self, report: DeadCodeReport, output: TextIO = sys.stdout) -> None:
        """Format report as JSON.

        Args:
            report: Dead code analysis report
            output: Output stream (default: stdout)
        """
        data = {
            "entry_points": [
                {
                    "name": ep.name,
                    "file_path": ep.file_path,
                    "line_number": ep.line_number,
                    "type": ep.type.value,
                    "confidence": ep.confidence,
                }
                for ep in report.entry_points
            ],
            "findings": [
                {
                    "function_name": f.function_name,
                    "file_path": f.file_path,
                    "start_line": f.start_line,
                    "end_line": f.end_line,
                    "confidence": f.confidence.value,
                    "reason": f.reason,
                    "caveats": f.caveats,
                }
                for f in report.findings
            ],
            "summary": {
                "total_functions": report.total_functions,
                "reachable": report.reachable_count,
                "unreachable": report.unreachable_count,
                "reachable_percentage": report.reachable_percentage,
            },
        }

        json.dump(data, output, indent=2)
        output.write("\n")


class SarifFormatter(DeadCodeFormatter):
    """SARIF format for GitHub Code Scanning integration.

    Follows SARIF 2.1.0 specification:
    https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    """

    def __init__(self, tool_version: str = "1.1.28") -> None:
        """Initialize SARIF formatter.

        Args:
            tool_version: Version of mcp-vector-search tool
        """
        self.tool_version = tool_version

    def format(self, report: DeadCodeReport, output: TextIO = sys.stdout) -> None:
        """Format report as SARIF.

        Args:
            report: Dead code analysis report
            output: Output stream (default: stdout)
        """
        sarif = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "mcp-vector-search",
                            "version": self.tool_version,
                            "informationUri": "https://github.com/masa/mcp-vector-search",
                            "rules": self._generate_rules(),
                        }
                    },
                    "results": self._generate_results(report.findings),
                }
            ],
        }

        json.dump(sarif, output, indent=2)
        output.write("\n")

    def _generate_rules(self) -> list[dict]:
        """Generate SARIF rule definitions.

        Returns:
            List of SARIF rule objects
        """
        return [
            {
                "id": "dead-code-high",
                "name": "DeadCodeHigh",
                "shortDescription": {
                    "text": "Definitely unused code (high confidence)"
                },
                "fullDescription": {
                    "text": "This code appears to be unreachable from any entry point with high confidence. "
                    "Consider removing it to reduce maintenance burden."
                },
                "defaultConfiguration": {"level": "warning"},
                "help": {
                    "text": "Dead code is code that is never executed or called. "
                    "It increases maintenance burden and can confuse developers."
                },
            },
            {
                "id": "dead-code-medium",
                "name": "DeadCodeMedium",
                "shortDescription": {
                    "text": "Potentially unused code (medium confidence)"
                },
                "fullDescription": {
                    "text": "This code appears to be unreachable from any entry point, but might be part of a public API. "
                    "Verify that it's not used externally before removing."
                },
                "defaultConfiguration": {"level": "note"},
                "help": {
                    "text": "Public functions might be part of an external API. "
                    "Verify external usage before removing."
                },
            },
            {
                "id": "dead-code-low",
                "name": "DeadCodeLow",
                "shortDescription": {"text": "Possibly unused code (low confidence)"},
                "fullDescription": {
                    "text": "This code appears to be unreachable, but might be called dynamically or used as a callback. "
                    "Investigate before removing."
                },
                "defaultConfiguration": {"level": "note"},
                "help": {
                    "text": "Functions with decorators or callback patterns might be registered dynamically. "
                    "Check runtime behavior before removing."
                },
            },
        ]

    def _generate_results(self, findings: list[DeadCodeFinding]) -> list[dict]:
        """Generate SARIF results from findings.

        Args:
            findings: List of dead code findings

        Returns:
            List of SARIF result objects
        """
        results = []

        for finding in findings:
            # Map confidence to rule ID
            rule_id = f"dead-code-{finding.confidence.value.lower()}"

            result = {
                "ruleId": rule_id,
                "message": {
                    "text": f"Function '{finding.function_name}' appears to be dead code: {finding.reason}"
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.file_path},
                            "region": {
                                "startLine": finding.start_line,
                                "endLine": finding.end_line,
                            },
                        }
                    }
                ],
            }

            # Add caveats as notes
            if finding.caveats:
                result["notes"] = [
                    {"text": caveat, "level": "note"} for caveat in finding.caveats
                ]

            results.append(result)

        return results


class MarkdownFormatter(DeadCodeFormatter):
    """Markdown output for documentation and reports."""

    def format(self, report: DeadCodeReport, output: TextIO = sys.stdout) -> None:
        """Format report as Markdown.

        Args:
            report: Dead code analysis report
            output: Output stream (default: stdout)
        """
        output.write("# Dead Code Analysis Report\n\n")

        # Entry points section
        output.write(f"## Entry Points ({len(report.entry_points)})\n\n")
        output.write(
            "Entry points are functions that serve as starting points for code execution.\n\n"
        )

        # Group by type
        by_type: dict[EntryPointType, list[EntryPoint]] = {}
        for ep in report.entry_points:
            by_type.setdefault(ep.type, []).append(ep)

        for ep_type in sorted(by_type.keys(), key=lambda x: x.value):
            eps = by_type[ep_type]
            output.write(f"### {ep_type.value} ({len(eps)})\n\n")
            output.write("| Function | File | Line |\n")
            output.write("|----------|------|------|\n")
            for ep in eps[:10]:  # Show first 10
                output.write(f"| `{ep.name}` | {ep.file_path} | {ep.line_number} |\n")
            if len(eps) > 10:
                output.write(f"\n*... and {len(eps) - 10} more*\n")
            output.write("\n")

        # Findings section
        output.write(f"## Dead Code Findings ({len(report.findings)})\n\n")

        # Group by confidence
        by_confidence: dict[Confidence, list[DeadCodeFinding]] = {
            Confidence.HIGH: [],
            Confidence.MEDIUM: [],
            Confidence.LOW: [],
        }

        for finding in report.findings:
            by_confidence[finding.confidence].append(finding)

        # HIGH confidence
        if by_confidence[Confidence.HIGH]:
            output.write(
                f"### High Confidence ({len(by_confidence[Confidence.HIGH])})\n\n"
            )
            output.write("| Function | Location | Reason |\n")
            output.write("|----------|----------|--------|\n")
            for finding in by_confidence[Confidence.HIGH]:
                location = (
                    f"{finding.file_path}:{finding.start_line}-{finding.end_line}"
                )
                output.write(
                    f"| `{finding.function_name}` | {location} | {finding.reason} |\n"
                )
            output.write("\n")

        # MEDIUM confidence
        if by_confidence[Confidence.MEDIUM]:
            output.write(
                f"### Medium Confidence ({len(by_confidence[Confidence.MEDIUM])})\n\n"
            )
            output.write("| Function | Location | Reason |\n")
            output.write("|----------|----------|--------|\n")
            for finding in by_confidence[Confidence.MEDIUM]:
                location = (
                    f"{finding.file_path}:{finding.start_line}-{finding.end_line}"
                )
                output.write(
                    f"| `{finding.function_name}` | {location} | {finding.reason} |\n"
                )
            output.write("\n")

        # LOW confidence
        if by_confidence[Confidence.LOW]:
            output.write(
                f"### Low Confidence ({len(by_confidence[Confidence.LOW])})\n\n"
            )
            output.write("| Function | Location | Reason |\n")
            output.write("|----------|----------|--------|\n")
            for finding in by_confidence[Confidence.LOW]:
                location = (
                    f"{finding.file_path}:{finding.start_line}-{finding.end_line}"
                )
                output.write(
                    f"| `{finding.function_name}` | {location} | {finding.reason} |\n"
                )
            output.write("\n")

        # Summary section
        output.write("## Summary\n\n")
        output.write("| Metric | Value |\n")
        output.write("|--------|-------|\n")
        output.write(f"| Total Functions | {report.total_functions} |\n")
        output.write(
            f"| Reachable | {report.reachable_count} ({report.reachable_percentage:.1f}%) |\n"
        )
        output.write(
            f"| Unreachable | {report.unreachable_count} ({100 - report.reachable_percentage:.1f}%) |\n"
        )
        output.write("\n")

        output.write("---\n\n*Generated by mcp-vector-search dead code analyzer*\n")


def get_formatter(format_name: str) -> DeadCodeFormatter:
    """Factory function to get formatter by name.

    Args:
        format_name: Format name (console, json, sarif, markdown)

    Returns:
        Formatter instance

    Raises:
        ValueError: If format_name is invalid
    """
    formatters = {
        "console": ConsoleFormatter,
        "json": JsonFormatter,
        "sarif": SarifFormatter,
        "markdown": MarkdownFormatter,
    }

    formatter_class = formatters.get(format_name.lower())
    if not formatter_class:
        raise ValueError(
            f"Invalid format: {format_name}. Must be one of: {', '.join(formatters.keys())}"
        )

    return formatter_class()
