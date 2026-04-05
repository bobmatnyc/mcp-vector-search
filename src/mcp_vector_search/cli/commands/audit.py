"""Audit command for the privacy-policy vs code certification tool.

Task 1 + CLI wiring: Implements 'mvs audit run' which runs a full audit
and writes a Markdown certification report.

Exit codes:
  0 — CERTIFIED
  1 — FAILED
  2 — CERTIFIED_WITH_EXCEPTIONS
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

audit_app = typer.Typer(
    name="audit",
    help="Privacy-policy vs code certification",
    no_args_is_help=True,
)

console = Console()


@audit_app.callback(invoke_without_command=True)
def audit_callback(ctx: typer.Context) -> None:
    """Privacy-policy vs code certification.

    Audit a codebase against a privacy policy document and produce
    a Markdown certification report.
    """
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@audit_app.command("run")
def audit_run(
    target: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Path to the target repository to audit",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        rich_help_panel="Required Options",
    ),
    policy: Path = typer.Option(
        ...,
        "--policy",
        "-p",
        help="Path to the privacy policy document (Markdown or plain text)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        rich_help_panel="Required Options",
    ),
    output_dir: Path = typer.Option(
        Path("certifications"),
        "--output-dir",
        "-o",
        help="Directory to write certification output",
        rich_help_panel="Output Options",
    ),
    no_ignore: bool = typer.Option(
        False,
        "--no-ignore",
        help="Ignore the .audit-ignore.yml file in the target repo",
        rich_help_panel="Audit Options",
    ),
) -> None:
    """Run a privacy-policy audit against a codebase.

    Extracts testable claims from the policy, gathers evidence from the
    target repo's vector index and knowledge graph, judges each claim, and
    writes a Markdown certification report.

    Prerequisites:
    - Set MVS_AUDIT_ANTHROPIC_API_KEY environment variable.
    - Run 'mvs index' in the target repo to build the vector index.

    Examples:
        mvs audit run --target ~/my-project --policy ~/my-project/PRIVACY.md
        mvs audit run --target . --policy ./privacy-policy.txt --output-dir ./certs
    """
    try:
        asyncio.run(
            _run_audit_async(
                target_repo=target,
                policy_path=policy,
                output_dir=output_dir,
                no_ignore=no_ignore,
            )
        )
    except ImportError:
        err_console = Console(stderr=True)
        err_console.print(
            "[red]Error:[/red] 'anthropic' package required. "
            "Install with: pip install 'mcp-vector-search[auditor]'"
        )
        sys.exit(1)


async def _run_audit_async(
    target_repo: Path,
    policy_path: Path,
    output_dir: Path,
    no_ignore: bool,
) -> None:
    """Async implementation of the audit run command."""
    try:
        from ...auditor.certifier import write_certification
        from ...auditor.config import AuditorSettings
        from ...auditor.runner import run_audit
    except ImportError as exc:
        console.print(
            f"[red]Import error:[/red] {exc}\n"
            "Install auditor dependencies: pip install 'mcp-vector-search[auditor]'"
        )
        sys.exit(1)

    # Load settings from environment
    try:
        settings = AuditorSettings()
    except Exception as exc:
        console.print(f"[red]Failed to load audit settings:[/red] {exc}")
        sys.exit(1)

    console.print("[bold blue]Privacy Audit[/bold blue]")
    console.print(f"  Target repo : {target_repo}")
    console.print(f"  Policy file : {policy_path}")
    console.print(f"  Output dir  : {output_dir}")
    console.print(f"  Extractor   : {settings.extractor_model}")
    console.print(f"  Judge       : {settings.judge_model}")
    console.print(f"  Ignore file : {'disabled' if no_ignore else 'enabled'}")
    console.print("")

    # Read policy text for snapshot
    policy_text = policy_path.read_text(encoding="utf-8")

    # Temporarily patch ignore loading if --no-ignore
    if no_ignore:
        from unittest.mock import patch

        from ...auditor.ignore import IgnoreList

        with patch.object(IgnoreList, "load", return_value=IgnoreList([])):
            doc = await run_audit(target_repo, policy_path, settings)
    else:
        doc = await run_audit(target_repo, policy_path, settings)

    # Write certification output
    cert_path = write_certification(doc, policy_text, output_dir)

    # Print summary table to terminal
    table = Table(title="Audit Summary", show_header=True, header_style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")

    status_styles = {
        "PASS": "green",
        "FAIL": "red",
        "INSUFFICIENT_EVIDENCE": "yellow",
        "MANUAL_REVIEW": "yellow",
        "IGNORED": "dim",
        "TOTAL": "bold",
    }

    for key in (
        "PASS",
        "FAIL",
        "INSUFFICIENT_EVIDENCE",
        "MANUAL_REVIEW",
        "IGNORED",
        "TOTAL",
    ):
        count = doc.summary.get(key, 0)
        style = status_styles.get(key, "")
        label = key.replace("_", " ").title()
        table.add_row(label, str(count), style=style)

    console.print(table)
    console.print(f"\nCertification written to: [bold]{cert_path}[/bold]")

    # Exit code based on overall status
    exit_codes = {
        "CERTIFIED": 0,
        "FAILED": 1,
        "CERTIFIED_WITH_EXCEPTIONS": 2,
    }
    exit_code = exit_codes.get(doc.overall_status, 1)
    if exit_code != 0:
        sys.exit(exit_code)
