"""Detect policy or code drift since last certification.

Task 17 (M5): Compares current policy + code state against the most recent
certification recorded in certifications/index.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .models import DriftReport


def _compute_policy_sha256(policy_path: Path) -> str:
    """Compute SHA-256 hex digest of the policy file contents."""
    h = hashlib.sha256()
    h.update(policy_path.read_bytes())
    return h.hexdigest()


def _get_head_commit(repo_path: Path) -> str | None:
    """Return the HEAD commit SHA for the given repo path.

    Returns None if git is unavailable or the directory is not a git repo.
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
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("Could not get HEAD commit: %s", exc)
        return None


def _target_slug(target_repo: Path) -> str:
    """Derive the target slug used as the key in index.json.

    Replicates the logic in certifier.update_index:
        target_slug = Path(doc.target_repo).name
    """
    return target_repo.name


def _load_latest_audit(
    slug: str,
    certifications_dir: Path,
) -> dict | None:
    """Load the most recent audit entry for *slug* from index.json.

    Returns None if the index does not exist or the slug has no entries.
    """
    index_path = certifications_dir / "index.json"
    if not index_path.exists():
        return None

    try:
        data = json.loads(index_path.read_bytes())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read index.json: %s", exc)
        return None

    targets: dict = data.get("targets", {})
    target_data = targets.get(slug)
    if not target_data:
        return None

    audits: list[dict] = target_data.get("audits", [])
    if not audits:
        return None

    # audits are appended in chronological order — last entry is most recent
    return audits[-1]


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse a YYYYMMDD-HHMMSS timestamp string to a UTC datetime.

    Returns None on parse failure.
    """
    try:
        return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _days_since(ts_str: str) -> int | None:
    """Return whole days elapsed since the given YYYYMMDD-HHMMSS timestamp.

    Returns None if parsing fails.
    """
    audit_dt = _parse_timestamp(ts_str)
    if audit_dt is None:
        return None
    now = datetime.now(UTC)
    delta = now - audit_dt
    return delta.days


def _build_details(
    *,
    policy_changed: bool,
    code_changed: bool,
    last_commit: str | None,
    current_commit: str | None,
    days: int | None,
) -> str:
    """Build a human-readable summary string for DriftReport.details."""
    if not policy_changed and not code_changed:
        days_str = (
            f"{days} day{'s' if days != 1 else ''}" if days is not None else "unknown"
        )
        return f"No drift detected. Last audit {days_str} ago."

    parts: list[str] = []

    if policy_changed and code_changed:
        short_last = last_commit[:12] if last_commit else "unknown"
        short_curr = current_commit[:12] if current_commit else "unknown"
        return (
            f"Both policy and code changed since last audit. Re-audit required. "
            f"(commit {short_last} -> {short_curr})"
        )

    if policy_changed:
        parts.append(
            "Policy changed since last audit (SHA mismatch). Re-audit recommended."
        )

    if code_changed:
        short_last = last_commit[:12] if last_commit else "unknown"
        short_curr = current_commit[:12] if current_commit else "unknown"
        parts.append(
            f"Code changed since last audit "
            f"(commit {short_last} -> {short_curr}). Re-audit recommended."
        )

    return " ".join(parts)


def check_drift(
    target_repo: Path,
    policy_path: Path,
    certifications_dir: Path = Path("audits"),
) -> DriftReport:
    """Compare current policy + code state against the most recent certification.

    Reads audits/index.json to find the latest audit for target_repo,
    then compares the current HEAD commit SHA and policy file SHA-256 against
    the recorded values.

    Args:
        target_repo: Root directory of the repository being checked.
        policy_path: Path to the privacy policy file.
        certifications_dir: Root certifications directory (default: "certifications").

    Returns:
        DriftReport indicating what (if anything) changed.
    """
    slug = _target_slug(target_repo)
    latest = _load_latest_audit(slug, certifications_dir)

    if latest is None:
        return DriftReport(
            target=slug,
            has_drift=True,
            policy_changed=False,
            code_changed=False,
            last_audit_timestamp=None,
            last_audit_commit=None,
            last_audit_policy_sha256=None,
            current_commit=None,
            current_policy_sha256=None,
            days_since_last_audit=None,
            details="No previous audit found",
        )

    last_timestamp: str = latest.get("timestamp", "")
    last_commit: str | None = latest.get("commit_sha")
    last_policy_sha: str | None = latest.get("policy_sha256")

    # Compute current state
    current_commit = _get_head_commit(target_repo)
    try:
        current_policy_sha = _compute_policy_sha256(policy_path)
    except OSError as exc:
        logger.warning("Could not read policy file %s: %s", policy_path, exc)
        current_policy_sha = None

    policy_changed = (
        current_policy_sha is not None
        and last_policy_sha is not None
        and current_policy_sha != last_policy_sha
    )
    code_changed = (
        current_commit is not None
        and last_commit is not None
        and current_commit != last_commit
    )
    has_drift = policy_changed or code_changed

    days = _days_since(last_timestamp) if last_timestamp else None

    details = _build_details(
        policy_changed=policy_changed,
        code_changed=code_changed,
        last_commit=last_commit,
        current_commit=current_commit,
        days=days,
    )

    return DriftReport(
        target=slug,
        has_drift=has_drift,
        policy_changed=policy_changed,
        code_changed=code_changed,
        last_audit_timestamp=last_timestamp or None,
        last_audit_commit=last_commit,
        last_audit_policy_sha256=last_policy_sha,
        current_commit=current_commit,
        current_policy_sha256=current_policy_sha,
        days_since_last_audit=days,
        details=details,
    )
