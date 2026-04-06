"""Certification document renderer and writer.

Task 9 (M1): Renders CertificationDocument to Markdown and writes output files.
Task 10 (M2): JSON sidecar, per-claim evidence files, manifest, policy sha256.
Task 11 (M2): GPG signing and verification.
Task 12 (M2): Certification directory structure, index.json, audit-log.jsonl, latest symlink.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .models import CertificationDocument, Evidence

if TYPE_CHECKING:
    from .config import AuditorSettings

# Maximum evidence snippets displayed per claim.
_MAX_EVIDENCE_DISPLAY = 10

# Minimum non-zero-score snippets before we allow zero-score ones.
_MIN_NONZERO_BEFORE_ZEROS = 3

# Snippet character limit for display.
_SNIPPET_DISPLAY_CHARS = 200

# Regex to extract referenced filenames from judge reasoning text.
_FILE_REF_RE = re.compile(
    r"\b([\w.\-/]+\.(?:md|json|ts|tsx|js|jsx|py|yaml|yml|env|txt|toml|lock|config))\b"
)

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


def _select_display_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Select up to _MAX_EVIDENCE_DISPLAY evidence items for display.

    Preference order:
    1. Non-zero-score items first (sorted by score desc, already sorted by
       evidence_collector).
    2. If fewer than _MIN_NONZERO_BEFORE_ZEROS non-zero items exist, pad
       with zero-score items to reach that minimum.
    3. Hard cap at _MAX_EVIDENCE_DISPLAY total items.
    """
    nonzero = [ev for ev in evidence if ev.score > 0.0]
    zero = [ev for ev in evidence if ev.score == 0.0]

    if len(nonzero) >= _MIN_NONZERO_BEFORE_ZEROS:
        selected = nonzero[:_MAX_EVIDENCE_DISPLAY]
    else:
        needed_zeros = max(0, _MIN_NONZERO_BEFORE_ZEROS - len(nonzero))
        selected = nonzero + zero[:needed_zeros]
        selected = selected[:_MAX_EVIDENCE_DISPLAY]

    return selected


def _extract_key_files(reasoning: str) -> set[str]:
    """Extract filenames the judge referenced in its reasoning text."""
    matches = _FILE_REF_RE.findall(reasoning)
    return {m.rsplit("/", 1)[-1] for m in matches if m}


def render_markdown(doc: CertificationDocument) -> str:
    """Render a CertificationDocument as a Markdown certification report."""
    lines: list[str] = []

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

    lines.append("## Claim Verdicts")
    lines.append("")

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
            display_evidence = _select_display_evidence(verdict.evidence)
            key_files = _extract_key_files(verdict.reasoning)
            if key_files:
                lines.append("**Key Evidence Files Referenced by Judge:**")
                for fname in sorted(key_files):
                    lines.append(f"- `{fname}`")
                lines.append("")

            lines.append(
                f"**Evidence** (showing {len(display_evidence)} of {len(verdict.evidence)}, sorted by relevance score):"
            )
            lines.append("")
            for idx, ev in enumerate(display_evidence, 1):
                score_str = f"{ev.score:.3f}"
                line_range = (
                    f"lines {ev.start_line}–{ev.end_line}"
                    if ev.start_line or ev.end_line
                    else "lines n/a"
                )
                lines.append(
                    f"**Evidence #{idx}** — `{ev.file_path}` {line_range} "
                    f"| score: {score_str} | tool: {ev.tool}"
                )
                if ev.kg_path:
                    lines.append(f"  - KG path: {' -> '.join(ev.kg_path)}")
                snippet_preview = (
                    ev.snippet[:_SNIPPET_DISPLAY_CHARS].replace("\n", " ").strip()
                )
                if snippet_preview:
                    lines.append(f"  > {snippet_preview}")
                lines.append("")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _to_json_bytes(data: dict | list, *, pretty: bool = True) -> bytes:
    """Serialize data to JSON bytes.

    Uses orjson if available for speed; falls back to stdlib json.
    For the hash computation, pretty=False gives compact sorted output.
    """
    try:
        import orjson

        if pretty:
            return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        return orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
    except ImportError:
        kwargs: dict = {"sort_keys": True, "default": str}
        if pretty:
            kwargs["indent"] = 2
        return json.dumps(data, **kwargs).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _doc_to_dict(doc: CertificationDocument) -> dict:
    """Convert CertificationDocument to a JSON-serializable dict.

    Datetime fields are converted to ISO 8601 strings.
    All verdicts and evidence are included (no truncation).
    """
    raw = doc.model_dump(mode="json")
    # Ensure datetime is ISO 8601 string
    if isinstance(raw.get("generated_at"), str):
        pass  # pydantic json mode already stringified it
    return raw


# ---------------------------------------------------------------------------
# Task 10: JSON sidecar, evidence files, manifest, policy sha256
# ---------------------------------------------------------------------------


def _write_json_sidecar(doc: CertificationDocument, cert_dir: Path) -> Path:
    """Write certification.json with the full CertificationDocument.

    The content_hash field in the document is computed over the canonical
    (compact, sorted) JSON of the verdicts list, not the full doc, so it
    is already set before this function is called.

    Returns path to certification.json.
    """
    data = _doc_to_dict(doc)
    pretty_bytes = _to_json_bytes(data, pretty=True)

    json_path = cert_dir / "certification.json"
    json_path.write_bytes(pretty_bytes)
    logger.info("Wrote certification JSON to %s", json_path)
    return json_path


def _write_evidence_files(doc: CertificationDocument, cert_dir: Path) -> list[Path]:
    """Write per-claim evidence files to evidence/<claim_id>.json.

    Each file contains: {"claim_id": ..., "claim_text": ..., "evidence": [...]}
    claim_text is taken from the verdict reasoning as a proxy (we don't store
    the original PolicyClaim text in the document).

    Returns list of written paths.
    """
    evidence_dir = cert_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    written: list[Path] = []
    for verdict in doc.verdicts:
        ev_data = {
            "claim_id": verdict.claim_id,
            "claim_text": verdict.reasoning,
            "status": verdict.status,
            "confidence": verdict.confidence,
            "evidence": [e.model_dump(mode="json") for e in verdict.evidence],
        }
        ev_path = evidence_dir / f"{verdict.claim_id}.json"
        ev_path.write_bytes(_to_json_bytes(ev_data, pretty=True))
        written.append(ev_path)

    logger.info("Wrote %d evidence files to %s", len(written), evidence_dir)
    return written


def _write_policy_sha256(policy_snapshot_path: Path, cert_dir: Path) -> Path:
    """Write policy-snapshot.sha256 containing the SHA-256 of the snapshot file.

    Returns path to the .sha256 file.
    """
    digest = _sha256_file(policy_snapshot_path)
    sha256_path = cert_dir / "policy-snapshot.sha256"
    sha256_path.write_text(f"{digest}  policy-snapshot.md\n", encoding="utf-8")
    logger.info("Wrote policy snapshot SHA-256 to %s", sha256_path)
    return sha256_path


def _write_manifest(cert_dir: Path) -> Path:
    """Write manifest.json with SHA-256 hashes of all files in cert_dir.

    Recursively walks cert_dir (excluding manifest.json itself) and records
    each file's hash.  Returns path to manifest.json.
    """
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    files: dict[str, str] = {}

    for fpath in sorted(cert_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "manifest.json":
            rel = fpath.relative_to(cert_dir).as_posix()
            files[rel] = f"sha256:{_sha256_file(fpath)}"

    manifest_data = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "files": files,
    }
    manifest_path = cert_dir / "manifest.json"
    manifest_path.write_bytes(_to_json_bytes(manifest_data, pretty=True))
    logger.info("Wrote manifest with %d file hashes to %s", len(files), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Task 11: GPG signing and verification
# ---------------------------------------------------------------------------


def sign_certification(
    cert_dir: Path,
    settings: AuditorSettings,
) -> Path | None:
    """Sign certification.json with a GPG detached ASCII-armored signature.

    If settings.gpg_key_id is not set or python-gnupg is not available,
    returns None (silent skip).

    After signing, updates manifest.json to include the .sig file hash.

    Args:
        cert_dir: Directory containing the certification files.
        settings: AuditorSettings with optional gpg_key_id.

    Returns:
        Path to certification.json.sig, or None if signing was skipped.
    """
    if not settings.gpg_key_id:
        logger.debug("GPG signing skipped: no gpg_key_id configured")
        return None

    try:
        import gnupg  # type: ignore[import]
    except ImportError:
        logger.warning(
            "GPG signing skipped: python-gnupg not installed. "
            "Install with: pip install python-gnupg"
        )
        return None

    json_path = cert_dir / "certification.json"
    if not json_path.exists():
        logger.error("Cannot sign: certification.json not found in %s", cert_dir)
        return None

    sig_path = cert_dir / "certification.json.sig"

    try:
        gpg = gnupg.GPG()
        with json_path.open("rb") as fh:
            result = gpg.sign_file(
                fh,
                keyid=settings.gpg_key_id,
                detach=True,
                armor=True,
                output=str(sig_path),
            )

        if not result:
            logger.error("GPG signing failed: %s", result.stderr)
            return None

        logger.info("GPG signature written to %s", sig_path)

        # Update manifest.json to include the .sig file
        manifest_path = cert_dir / "manifest.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_bytes())
            rel = sig_path.relative_to(cert_dir).as_posix()
            manifest_data["files"][rel] = f"sha256:{_sha256_file(sig_path)}"
            manifest_path.write_bytes(_to_json_bytes(manifest_data, pretty=True))
            logger.debug("Updated manifest.json with .sig hash")

        return sig_path

    except Exception as exc:
        logger.error("GPG signing failed with exception: %s", exc)
        return None


def verify_certification(cert_dir: Path) -> bool:
    """Verify a certification directory's GPG signature and manifest hashes.

    Checks:
    1. All files listed in manifest.json match their recorded SHA-256 hashes.
    2. If certification.json.sig exists, verifies the GPG signature against
       certification.json.

    Args:
        cert_dir: Path to the certification run directory.

    Returns:
        True if all checks pass, False otherwise.
    """
    manifest_path = cert_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("Verification failed: manifest.json not found in %s", cert_dir)
        return False

    manifest_data = json.loads(manifest_path.read_bytes())
    files = manifest_data.get("files", {})

    # Verify all manifest hashes
    all_ok = True
    for rel, recorded_hash in files.items():
        fpath = cert_dir / rel
        if not fpath.exists():
            logger.error("Manifest file missing: %s", fpath)
            all_ok = False
            continue

        expected = recorded_hash.removeprefix("sha256:")
        actual = _sha256_file(fpath)
        if actual != expected:
            logger.error(
                "Hash mismatch for %s: expected %s, got %s",
                rel,
                expected[:16],
                actual[:16],
            )
            all_ok = False
        else:
            logger.debug("Hash OK: %s", rel)

    if not all_ok:
        return False

    # Optionally verify GPG signature
    sig_path = cert_dir / "certification.json.sig"
    json_path = cert_dir / "certification.json"

    if sig_path.exists() and json_path.exists():
        try:
            import gnupg  # type: ignore[import]

            gpg = gnupg.GPG()
            with json_path.open("rb") as fh:
                verified = gpg.verify_file(fh, sig_file=str(sig_path))

            if not verified:
                logger.error("GPG signature verification failed: %s", verified.status)
                return False

            logger.info("GPG signature valid (fingerprint: %s)", verified.fingerprint)
        except ImportError:
            logger.warning(
                "python-gnupg not installed; GPG signature not verified. "
                "Manifest hashes OK."
            )
    else:
        logger.info("No GPG signature found; manifest hashes verified OK.")

    return True


# ---------------------------------------------------------------------------
# Task 12: Index, audit-log, latest symlink
# ---------------------------------------------------------------------------


def update_index(
    cert_dir: Path,
    doc: CertificationDocument,
    output_dir: Path,
    signed: bool = False,
) -> None:
    """Update certifications/index.json with the new audit entry.

    Reads the existing index (or creates a new one), appends the new audit
    entry for the target, and writes the file back atomically.

    Args:
        cert_dir: Path to the specific run directory (e.g. certifications/tripbot7/20260405-231955/).
        doc: The CertificationDocument from the completed audit.
        output_dir: Root certifications directory.
        signed: Whether the certification was GPG-signed.
    """
    index_path = output_dir / "index.json"

    if index_path.exists():
        try:
            existing = json.loads(index_path.read_bytes())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read existing index.json (%s); starting fresh.", exc
            )
            existing = {"targets": {}}
    else:
        existing = {"targets": {}}

    targets: dict = existing.setdefault("targets", {})

    target_slug = Path(doc.target_repo).name
    timestamp = doc.generated_at.strftime("%Y%m%d-%H%M%S")

    audit_entry = {
        "timestamp": timestamp,
        "commit_sha": doc.target_commit_sha,
        "policy_sha256": doc.policy_sha256,
        "overall_status": doc.overall_status,
        "summary": {
            k: doc.summary.get(k, 0)
            for k in (
                "PASS",
                "FAIL",
                "INSUFFICIENT_EVIDENCE",
                "MANUAL_REVIEW",
                "IGNORED",
                "TOTAL",
            )
        },
        "signed": signed,
    }

    if target_slug not in targets:
        targets[target_slug] = {"latest": timestamp, "audits": [audit_entry]}
    else:
        targets[target_slug]["latest"] = timestamp
        targets[target_slug].setdefault("audits", []).append(audit_entry)

    index_path.write_bytes(_to_json_bytes(existing, pretty=True))
    logger.info(
        "Updated %s (target=%s, timestamp=%s)", index_path, target_slug, timestamp
    )


def append_audit_log(
    doc: CertificationDocument,
    output_dir: Path,
) -> None:
    """Append one JSON line to certifications/audit-log.jsonl.

    Each line captures high-level audit metadata for quick scanning.

    Args:
        doc: The completed CertificationDocument.
        output_dir: Root certifications directory.
    """
    log_path = output_dir / "audit-log.jsonl"

    summary = doc.summary
    line_data = {
        "timestamp": doc.generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": Path(doc.target_repo).name,
        "commit": doc.target_commit_sha,
        "policy_sha256": doc.policy_sha256[:16],
        "status": doc.overall_status,
        "claims": summary.get("TOTAL", 0),
        "pass": summary.get("PASS", 0),
        "fail": summary.get("FAIL", 0),
        "insufficient": summary.get("INSUFFICIENT_EVIDENCE", 0),
        "manual_review": summary.get("MANUAL_REVIEW", 0),
        "ignored": summary.get("IGNORED", 0),
    }

    line = json.dumps(line_data, sort_keys=True) + "\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)

    logger.info("Appended to audit log: %s", log_path)


def update_latest_symlink(
    cert_dir: Path, output_dir: Path, doc: CertificationDocument
) -> None:
    """Create or update the certifications/<target>/latest symlink.

    Points to the newest timestamp directory name (not a full path, just
    the directory name for portability).

    Args:
        cert_dir: Path to the current run directory.
        output_dir: Root certifications directory.
        doc: CertificationDocument with target_repo name.
    """
    target_slug = Path(doc.target_repo).name
    target_dir = output_dir / target_slug
    symlink_path = target_dir / "latest"
    timestamp_dir_name = cert_dir.name  # just YYYYMMDD-HHMMSS

    # Remove existing symlink or file
    if symlink_path.is_symlink():
        symlink_path.unlink()
    elif symlink_path.exists():
        logger.warning(
            "latest exists but is not a symlink at %s; skipping", symlink_path
        )
        return

    symlink_path.symlink_to(timestamp_dir_name)
    logger.info("Updated latest symlink: %s -> %s", symlink_path, timestamp_dir_name)


def _ensure_certifications_readme(output_dir: Path) -> None:
    """Write certifications/README.md if it does not already exist."""
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        return

    readme_path.write_text(
        """\
# Privacy Certifications

Audit artifacts from the mcp-vector-search privacy auditor.

## Structure

- `<target-slug>/<YYYYMMDD-HHMMSS>/` — one directory per audit run
- `certification.md` — human-readable report
- `certification.json` — machine-readable full data
- `certification.json.sig` — GPG detached signature (if signed)
- `policy-snapshot.md` — policy text at audit time
- `policy-snapshot.sha256` — SHA-256 of the policy snapshot
- `evidence/<claim_id>.json` — per-claim evidence
- `manifest.json` — SHA-256 hashes of all files

## Verification

```
mvs audit verify certifications/<target>/<timestamp>/
```
""",
        encoding="utf-8",
    )
    logger.info("Created certifications README at %s", readme_path)


# ---------------------------------------------------------------------------
# Main writer (orchestrates everything)
# ---------------------------------------------------------------------------


def write_certification(
    doc: CertificationDocument,
    policy_text: str,
    output_dir: Path,
) -> Path:
    """Write all certification artifacts to the output directory.

    Creates (M1):
        <output_dir>/<target-slug>/<YYYYMMDD-HHMMSS>/certification.md
        <output_dir>/<target-slug>/<YYYYMMDD-HHMMSS>/policy-snapshot.md

    Additionally creates (M2):
        certification.json          — full CertificationDocument as JSON
        evidence/<claim_id>.json    — per-claim evidence files
        policy-snapshot.sha256      — SHA-256 of policy-snapshot.md
        manifest.json               — SHA-256 hashes of all files

    Args:
        doc: The CertificationDocument to write.
        policy_text: Verbatim text of the privacy policy.
        output_dir: Root output directory (e.g. Path("certifications")).

    Returns:
        Path to the created certification.md file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_certifications_readme(output_dir)

    target_slug = Path(doc.target_repo).name
    timestamp = doc.generated_at.strftime("%Y%m%d-%H%M%S")

    cert_dir = output_dir / target_slug / timestamp
    cert_dir.mkdir(parents=True, exist_ok=True)

    # --- M1 files ---
    cert_md_path = cert_dir / "certification.md"
    cert_md_path.write_text(render_markdown(doc), encoding="utf-8")
    logger.info("Wrote certification report to %s", cert_md_path)

    snapshot_path = cert_dir / "policy-snapshot.md"
    snapshot_content = (
        f"# Policy Snapshot\n\n"
        f"**Source:** `{doc.policy_path}`\n"
        f"**SHA-256:** `{doc.policy_sha256}`\n\n"
        f"---\n\n{policy_text}"
    )
    snapshot_path.write_text(snapshot_content, encoding="utf-8")
    logger.info("Wrote policy snapshot to %s", snapshot_path)

    # --- M2: JSON sidecar ---
    _write_json_sidecar(doc, cert_dir)

    # --- M2: Per-claim evidence files ---
    _write_evidence_files(doc, cert_dir)

    # --- M2: policy-snapshot.sha256 ---
    _write_policy_sha256(snapshot_path, cert_dir)

    # --- M2: manifest.json ---
    _write_manifest(cert_dir)

    return cert_md_path
