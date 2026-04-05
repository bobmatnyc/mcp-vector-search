"""Certification document renderer and writer.

Task 9: Renders CertificationDocument to Markdown and writes output files.
JSON signing is deferred to M2.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from .models import CertificationDocument, Verdict

# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_STATUS_EMOJI = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "INSUFFICIENT_EVIDENCE": "INSUFFICIENT",
    "MANUAL_REVIEW": "MANUAL",
}

_STATUS_MD = {
    "PASS": "[PASS]",
    "FAIL": "[FAIL]",
    "INSUFFICIENT_EVIDENCE": "[INSUFFICIENT EVIDENCE]",
    "MANUAL_REVIEW": "[MANUAL REVIEW]",
}

_OVERALL_STATUS_HEADER = {
    "CERTIFIED": "CERTIFIED",
    "CERTIFIED_WITH_EXCEPTIONS": "CERTIFIED WITH EXCEPTIONS",
    "FAILED": "FAILED",
}


def render_markdown(doc: CertificationDocument) -> str:
    """Render a CertificationDocument as a Markdown certification report.

    Args:
        doc: The CertificationDocument to render.

    Returns:
        Markdown string suitable for writing to certification.md.
    """
    lines: list[str] = []

    # --- Header ---
    overall_label = _OVERALL_STATUS_HEADER.get(doc.overall_status, doc.overall_status)
    lines.append(f"# Privacy Certification — {overall_label}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Target Repo | `{doc.target_repo}` |")
    lines.append(f"| Commit SHA | `{doc.target_commit_sha}` |")
    lines.append(f"| Policy Path | `{doc.policy_path}` |")
    lines.append(f"| Policy SHA-256 | `{doc.policy_sha256[:16]}...` |")
    lines.append(
        f"| Generated At | {doc.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} |"
    )
    lines.append(f"| Generator Version | `{doc.generator_version}` |")
    lines.append(f"| Auditor Model | `{doc.auditor_model}` |")
    lines.append(f"| Schema Version | `{doc.schema_version}` |")
    lines.append(f"| Content Hash | `{doc.content_hash}` |")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    for key in (
        "PASS",
        "FAIL",
        "INSUFFICIENT_EVIDENCE",
        "MANUAL_REVIEW",
        "IGNORED",
        "TOTAL",
    ):
        count = doc.summary.get(key, 0)
        label = key.replace("_", " ").title()
        lines.append(f"| {label} | {count} |")
    lines.append("")

    # --- Per-claim verdicts ---
    lines.append("## Claim Verdicts")
    lines.append("")

    # Group by category for readability
    from collections import defaultdict

    by_category: dict[str, list[tuple[Verdict, str]]] = defaultdict(list)

    # Build claim_id → category mapping from verdicts only
    # (We don't have claims in this function, so we show claim_id)
    for verdict in doc.verdicts:
        by_category["all"].append((verdict, verdict.claim_id))

    for verdict in doc.verdicts:
        status_md = _STATUS_MD.get(verdict.status, f"[{verdict.status}]")
        ignored_note = " *(ignored)*" if verdict.ignored else ""

        lines.append(f"### Claim `{verdict.claim_id}`{ignored_note}")
        lines.append("")
        lines.append(f"**Status:** {status_md}")
        lines.append(f"**Confidence:** {verdict.confidence:.0%}")
        lines.append(f"**Evidence Count:** {verdict.evidence_count}")
        lines.append(
            f"**KG Path Present:** {'Yes' if verdict.kg_path_present else 'No'}"
        )
        lines.append("")

        if verdict.ignored and verdict.ignore_justification:
            lines.append(f"> **Ignored:** {verdict.ignore_justification}")
            lines.append("")

        lines.append(f"**Reasoning:** {verdict.reasoning}")
        lines.append("")

        if verdict.evidence:
            lines.append("**Evidence:**")
            lines.append("")
            for ev in verdict.evidence[:5]:  # cap at 5 evidence items per claim
                lines.append(
                    f"- **[{ev.tool}]** `{ev.file_path}` lines {ev.start_line}-{ev.end_line} (score: {ev.score:.3f})"
                )
                if ev.kg_path:
                    lines.append(f"  - KG path: {' -> '.join(ev.kg_path)}")
                snippet_preview = ev.snippet[:150].replace("\n", " ")
                if snippet_preview:
                    lines.append(f"  - `{snippet_preview}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------


def write_certification(
    doc: CertificationDocument,
    policy_text: str,
    output_dir: Path,
) -> Path:
    """Write certification files to the output directory.

    Creates:
        <output_dir>/<target-slug>/<YYYYMMDD-HHMMSS>/certification.md
        <output_dir>/<target-slug>/<YYYYMMDD-HHMMSS>/policy-snapshot.md

    Args:
        doc: The CertificationDocument to write.
        policy_text: Verbatim text of the privacy policy.
        output_dir: Root output directory (e.g. Path("certifications")).

    Returns:
        Path to the created certification.md file.
    """
    target_slug = Path(doc.target_repo).name
    timestamp = doc.generated_at.strftime("%Y%m%d-%H%M%S")

    cert_dir = output_dir / target_slug / timestamp
    cert_dir.mkdir(parents=True, exist_ok=True)

    # Write certification.md
    cert_md_path = cert_dir / "certification.md"
    cert_md_path.write_text(render_markdown(doc), encoding="utf-8")
    logger.info("Wrote certification report to %s", cert_md_path)

    # Write policy-snapshot.md
    snapshot_path = cert_dir / "policy-snapshot.md"
    snapshot_content = f"# Policy Snapshot\n\n**Source:** `{doc.policy_path}`\n**SHA-256:** `{doc.policy_sha256}`\n\n---\n\n{policy_text}"
    snapshot_path.write_text(snapshot_content, encoding="utf-8")
    logger.info("Wrote policy snapshot to %s", snapshot_path)

    return cert_md_path
