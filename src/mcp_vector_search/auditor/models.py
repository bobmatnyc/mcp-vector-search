"""Pydantic v2 models for the privacy-policy auditor.

Task 2: Core data models for claims, evidence, verdicts, and certification.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PolicyClaim(BaseModel):
    """An atomic, testable claim extracted from a privacy policy."""

    id: str  # sha1(category+normalized_text)[:12]
    category: Literal[
        "data_sharing",
        "encryption",
        "retention",
        "user_rights",
        "third_party",
        "logging_pii",
        "consent",
        "security",
        "access_control",
    ]
    text: str  # verbatim policy excerpt
    normalized: str  # LLM-rewritten testable form
    keywords: list[str]
    policy_section: str | None = None
    testable: bool  # false => auto MANUAL_REVIEW
    source_offsets: tuple[int, int]

    @classmethod
    def compute_id(cls, category: str, normalized_text: str) -> str:
        """Compute a stable ID from category and normalized claim text.

        Args:
            category: The claim category string.
            normalized_text: The normalized (LLM-rewritten) claim text.

        Returns:
            First 12 hex characters of SHA-1 hash of category+normalized_text.
        """
        raw = (category + normalized_text).encode("utf-8")
        return hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:12]  # noqa: S324


class Evidence(BaseModel):
    """A single piece of evidence supporting or refuting a policy claim."""

    tool: Literal[
        "search_code",
        "search_hybrid",
        "kg_query",
        "kg_callers_at_commit",
        "trace_execution_flow",
        "find_smells",
    ]
    query: str
    file_path: str
    start_line: int
    end_line: int
    snippet: str
    score: float
    kg_path: list[str] | None = None


class Verdict(BaseModel):
    """The final verdict for a single policy claim after evidence gathering and judging."""

    claim_id: str
    status: Literal["PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "MANUAL_REVIEW"]
    confidence: float
    reasoning: str
    evidence: list[Evidence]
    kg_path_present: bool
    evidence_count: int
    ignored: bool = False
    ignore_justification: str | None = None


class CertificationDocument(BaseModel):
    """Complete certification document produced by an audit run."""

    schema_version: Literal["1.0"] = "1.0"
    target_repo: str
    target_commit_sha: str
    policy_path: str
    policy_sha256: str
    policy_snapshot_path: str
    generated_at: datetime
    generator_version: str
    auditor_model: str
    verdicts: list[Verdict]
    summary: dict[str, int]
    overall_status: Literal["CERTIFIED", "CERTIFIED_WITH_EXCEPTIONS", "FAILED"]
    content_hash: str
    signature_path: str | None = None


class AuditIgnoreEntry(BaseModel):
    """A single entry in the .audit-ignore.yml ignore file."""

    claim_id: str | None = None
    category: str | None = None
    pattern: str | None = None
    justification: str = Field(min_length=20)
    expires: date | None = None
    approved_by: str


class DriftReport(BaseModel):
    """Report of policy/code drift since the last certification audit."""

    target: str
    has_drift: bool
    policy_changed: bool
    code_changed: bool
    last_audit_timestamp: str | None
    last_audit_commit: str | None
    last_audit_policy_sha256: str | None
    current_commit: str | None
    current_policy_sha256: str | None
    days_since_last_audit: int | None
    details: str  # human-readable summary
