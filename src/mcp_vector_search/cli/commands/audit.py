"""Audit command for the privacy-policy vs code certification tool.

Task 1 + CLI wiring: Implements 'mvs audit run' which runs a full audit
and writes a Markdown certification report.

M2 additions:
  - 'mvs audit verify <cert-dir>' — verify manifest hashes and GPG signature
  - 'mvs audit list [--target]'   — list audits from index.json

Exit codes:
  0 — CERTIFIED (run) or verification passed
  1 — FAILED (run) or verification failed
  2 — CERTIFIED_WITH_EXCEPTIONS
"""

from __future__ import annotations

import asyncio
import json
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
    no_issues: bool = typer.Option(
        False,
        "--no-issues",
        help="Skip creating GitHub issues for FAIL/MANUAL_REVIEW/INSUFFICIENT_EVIDENCE verdicts",
        rich_help_panel="Audit Options",
    ),
) -> None:
    """Run a privacy-policy audit against a codebase.

    Extracts testable claims from the policy, gathers evidence from the
    target repo's vector index and knowledge graph, judges each claim, and
    writes a Markdown certification report alongside JSON sidecar, evidence
    files, manifest, and (optionally) a GPG signature.

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
                no_issues=no_issues,
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
    no_issues: bool = False,
) -> None:
    """Async implementation of the audit run command."""
    try:
        from ...auditor.certifier import write_certification
        from ...auditor.config import AuditorSettings
        from ...auditor.runner import finalize_audit, maybe_create_issues, run_audit
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

    # Apply --no-issues flag
    if no_issues:
        settings = settings.model_copy(update={"create_issues": False})

    console.print("[bold blue]Privacy Audit[/bold blue]")
    console.print(f"  Target repo : {target_repo}")
    console.print(f"  Policy file : {policy_path}")
    console.print(f"  Output dir  : {output_dir}")
    if settings.llm_backend == "openrouter":
        console.print("  Backend     : openrouter")
        console.print(f"  Extractor   : {settings.openrouter_extractor_model}")
        console.print(f"  Judge       : {settings.openrouter_judge_model}")
    else:
        console.print(f"  Extractor   : {settings.extractor_model}")
        console.print(f"  Judge       : {settings.judge_model}")
    console.print(f"  Ignore file : {'disabled' if no_ignore else 'enabled'}")
    if settings.gpg_key_id:
        console.print(f"  GPG Key     : {settings.gpg_key_id}")
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

    # Write all certification artifacts (M1 + M2)
    cert_path = write_certification(doc, policy_text, output_dir)
    cert_dir = cert_path.parent

    # Finalize: GPG sign, update index, append audit log, symlink
    signed = finalize_audit(doc, cert_dir, output_dir, settings)

    # M4: Create GitHub issues for verdicts requiring review
    issue_results = await maybe_create_issues(doc, settings, cert_path=cert_path)
    if issue_results:
        new_count = sum(1 for r in issue_results if r["status"] == "created")
        updated_count = sum(1 for r in issue_results if r["status"] == "updated")
        console.print(
            f"\nIssues created: [green]{new_count} new[/green], "
            f"[yellow]{updated_count} updated[/yellow]"
        )
        for r in issue_results:
            action_label = "updated" if r["status"] == "updated" else ""
            suffix = f" ({action_label})" if action_label else ""
            console.print(f"  #{r['issue_number']}: {r['url']}{suffix}")

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
    console.print(f"\nCertification written to: [bold]{cert_dir}[/bold]")
    if signed:
        console.print("[green]GPG signed:[/green] certification.json.sig")

    # Exit code based on overall status
    exit_codes = {
        "CERTIFIED": 0,
        "FAILED": 1,
        "CERTIFIED_WITH_EXCEPTIONS": 2,
    }
    exit_code = exit_codes.get(doc.overall_status, 1)
    if exit_code != 0:
        sys.exit(exit_code)


@audit_app.command("verify")
def audit_verify(
    cert_dir: Path = typer.Argument(
        ...,
        help="Path to a certification run directory (e.g. certifications/tripbot7/20260405-231955/)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Verify a certification directory's manifest hashes and GPG signature.

    Checks that all files recorded in manifest.json match their SHA-256
    hashes.  If certification.json.sig is present, also verifies the GPG
    detached signature against certification.json.

    Exit code 0 if valid, 1 if invalid.
    """
    try:
        from ...auditor.certifier import verify_certification
    except ImportError as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold]Verifying certification:[/bold] {cert_dir}")

    ok = verify_certification(cert_dir)

    sig_path = cert_dir / "certification.json.sig"
    if ok:
        if sig_path.exists():
            console.print(
                "[green]VALID[/green] — manifest hashes OK, GPG signature valid"
            )
        else:
            console.print("[green]VALID[/green] — manifest hashes OK (unsigned)")
        sys.exit(0)
    else:
        console.print("[red]INVALID[/red] — verification failed (see logs above)")
        sys.exit(1)


@audit_app.command("list")
def audit_list(
    output_dir: Path = typer.Option(
        Path("certifications"),
        "--output-dir",
        "-o",
        help="Certifications root directory",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Filter to a specific target slug",
    ),
) -> None:
    """List past audits from the certifications index.

    Reads certifications/index.json and displays a table of all recorded
    audits.  Use --target to filter to a specific repository slug.

    Example:
        mvs audit list
        mvs audit list --target tripbot7
    """
    index_path = output_dir / "index.json"

    if not index_path.exists():
        console.print(
            f"[yellow]No index found at {index_path}[/yellow]\n"
            "Run 'mvs audit run' first to create certifications."
        )
        sys.exit(0)

    try:
        index_data = json.loads(index_path.read_bytes())
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Failed to read index.json:[/red] {exc}")
        sys.exit(1)

    targets: dict = index_data.get("targets", {})

    if target:
        if target not in targets:
            console.print(f"[yellow]No audits found for target '{target}'[/yellow]")
            sys.exit(0)
        targets = {target: targets[target]}

    if not targets:
        console.print("[yellow]No audits recorded yet.[/yellow]")
        sys.exit(0)

    table = Table(
        title="Audit History",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Target", style="bold")
    table.add_column("Timestamp")
    table.add_column("Status")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Insuff.", justify="right", style="yellow")
    table.add_column("Manual", justify="right", style="yellow")
    table.add_column("Total", justify="right")
    table.add_column("Signed")

    status_styles = {
        "CERTIFIED": "green",
        "CERTIFIED_WITH_EXCEPTIONS": "yellow",
        "FAILED": "red",
    }

    for tgt_name, tgt_data in sorted(targets.items()):
        audits = tgt_data.get("audits", [])
        for entry in reversed(audits):  # newest first
            status = entry.get("overall_status", "?")
            summary = entry.get("summary", {})
            signed_icon = "yes" if entry.get("signed") else "no"
            status_style = status_styles.get(status, "")
            table.add_row(
                tgt_name,
                entry.get("timestamp", "?"),
                f"[{status_style}]{status}[/{status_style}]"
                if status_style
                else status,
                str(summary.get("PASS", 0)),
                str(summary.get("FAIL", 0)),
                str(summary.get("INSUFFICIENT_EVIDENCE", 0)),
                str(summary.get("MANUAL_REVIEW", 0)),
                str(summary.get("TOTAL", 0)),
                signed_icon,
            )

    console.print(table)
