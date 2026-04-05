"""Top-level audit orchestrator.

Task 8: Runs the full audit pipeline and returns a CertificationDocument.

Pipeline:
1. Read policy file, compute sha256
2. Get target repo HEAD commit SHA
3. Load .audit-ignore.yml from target repo
4. Extract claims (policy_extractor)
5. For each claim: apply ignore list, route to plans, collect evidence, judge
6. Aggregate verdicts and compute summary
7. Determine overall_status
8. Return CertificationDocument
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .. import __version__
from .claim_router import route
from .config import AuditorSettings
from .evidence_collector import collect
from .ignore import IgnoreList
from .judge import judge_claim
from .models import CertificationDocument, Verdict
from .policy_extractor import extract_claims

console = Console()


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _compute_content_hash(verdicts: list[Verdict]) -> str:
    """Compute a stable content hash over the verdicts for integrity checking."""
    import json

    raw = json.dumps(
        [v.model_dump(mode="json") for v in verdicts],
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _get_head_commit(repo_path: Path) -> str:
    """Get the HEAD commit SHA of a git repository.

    Args:
        repo_path: Path to the repository root.

    Returns:
        40-character commit SHA, or 'unknown' if git is unavailable.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607  # nosec B607
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("git rev-parse HEAD failed: %s", result.stderr.strip())
        return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Could not get HEAD commit: %s", exc)
        return "unknown"


def _determine_overall_status(
    verdicts: list[Verdict],
) -> str:
    """Determine overall certification status from individual verdicts.

    Logic (user decision #13):
    - Any FAIL → FAILED
    - Any MANUAL_REVIEW or INSUFFICIENT_EVIDENCE (but no FAIL) → CERTIFIED_WITH_EXCEPTIONS
    - All PASS → CERTIFIED

    Args:
        verdicts: List of all Verdict objects (excluding ignored).

    Returns:
        One of: "CERTIFIED", "CERTIFIED_WITH_EXCEPTIONS", "FAILED"
    """
    active = [v for v in verdicts if not v.ignored]

    if any(v.status == "FAIL" for v in active):
        return "FAILED"
    if any(v.status in ("MANUAL_REVIEW", "INSUFFICIENT_EVIDENCE") for v in active):
        return "CERTIFIED_WITH_EXCEPTIONS"
    return "CERTIFIED"


def _compute_summary(verdicts: list[Verdict]) -> dict[str, int]:
    """Compute summary counts by status."""
    summary: dict[str, int] = {
        "PASS": 0,
        "FAIL": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "MANUAL_REVIEW": 0,
        "IGNORED": 0,
        "TOTAL": len(verdicts),
    }
    for v in verdicts:
        if v.ignored:
            summary["IGNORED"] += 1
        else:
            summary[v.status] = summary.get(v.status, 0) + 1
    return summary


async def run_audit(
    target_repo: Path,
    policy_path: Path,
    settings: AuditorSettings,
) -> CertificationDocument:
    """Run a full privacy-policy audit against a target repository.

    Args:
        target_repo: Root directory of the repository to audit.
        policy_path: Path to the privacy policy document.
        settings: Auditor settings (models, thresholds, etc.).

    Returns:
        CertificationDocument with all verdicts and metadata.
    """
    generated_at = datetime.now(UTC)

    # Step 1: Read policy and compute hash
    policy_text = policy_path.read_text(encoding="utf-8")
    policy_sha256 = _compute_sha256(policy_path)
    logger.info("Policy: %s (sha256=%s...)", policy_path, policy_sha256[:16])

    # Step 2: Get HEAD commit SHA
    commit_sha = _get_head_commit(target_repo)
    logger.info(
        "Target repo: %s @ %s",
        target_repo,
        commit_sha[:12] if commit_sha != "unknown" else "unknown",
    )

    # Step 3: Load ignore list
    ignore_list = IgnoreList.load(target_repo)
    logger.info("Ignore list: %d entries", len(ignore_list))

    # Step 4: Extract claims
    console.print(
        f"[bold]Extracting claims from policy[/bold] using {settings.extractor_model}..."
    )
    claims = await extract_claims(policy_text, settings)
    console.print(f"[green]Extracted {len(claims)} claims[/green]")

    # Step 5-7: Process each claim with progress bar
    verdicts: list[Verdict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Auditing claims...", total=len(claims))

        for claim in claims:
            progress.update(
                task,
                description=f"[cyan]{claim.category}[/cyan] {claim.id}",
            )

            # Apply ignore list
            ignore_entry = ignore_list.matches(claim)
            if ignore_entry is not None:
                logger.info(
                    "Claim %s (%s) matched ignore entry by %s",
                    claim.id,
                    claim.category,
                    ignore_entry.approved_by,
                )
                verdicts.append(
                    Verdict(
                        claim_id=claim.id,
                        status="MANUAL_REVIEW",
                        confidence=1.0,
                        reasoning=f"Ignored: {ignore_entry.justification}",
                        evidence=[],
                        kg_path_present=False,
                        evidence_count=0,
                        ignored=True,
                        ignore_justification=ignore_entry.justification,
                    )
                )
                progress.advance(task)
                continue

            # Route claim to query plans
            plans = route(claim)
            logger.debug("Claim %s → %d query plans", claim.id, len(plans))

            # Collect evidence
            evidence = await collect(target_repo, plans)

            # Judge claim
            verdict = await judge_claim(claim, evidence, settings)
            verdicts.append(verdict)

            progress.advance(task)

    # Step 8: Aggregate
    summary = _compute_summary(verdicts)
    overall_status = _determine_overall_status(verdicts)
    content_hash = _compute_content_hash(verdicts)

    target_slug = target_repo.name
    policy_snapshot_path = f"certifications/{target_slug}/policy-snapshot.md"

    doc = CertificationDocument(
        target_repo=str(target_repo),
        target_commit_sha=commit_sha,
        policy_path=str(policy_path),
        policy_sha256=policy_sha256,
        policy_snapshot_path=policy_snapshot_path,
        generated_at=generated_at,
        generator_version=__version__,
        auditor_model=settings.judge_model,
        verdicts=verdicts,
        summary=summary,
        overall_status=overall_status,  # type: ignore[arg-type]
        content_hash=content_hash,
    )

    status_color = {
        "CERTIFIED": "green",
        "CERTIFIED_WITH_EXCEPTIONS": "yellow",
        "FAILED": "red",
    }.get(overall_status, "white")

    console.print(
        f"\n[bold]Overall status:[/bold] [{status_color}]{overall_status}[/{status_color}]"
    )
    console.print(
        f"PASS={summary['PASS']} FAIL={summary['FAIL']} "
        f"INSUFFICIENT={summary['INSUFFICIENT_EVIDENCE']} "
        f"MANUAL={summary['MANUAL_REVIEW']} IGNORED={summary['IGNORED']}"
    )

    return doc
