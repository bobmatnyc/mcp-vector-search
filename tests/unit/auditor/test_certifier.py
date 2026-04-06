"""Unit tests for the certifier (Task 9).

Tests:
- render_markdown produces expected sections and content
- write_certification creates the correct directory structure
- Snapshot test for Markdown rendering (excluding timestamps)
"""

import re
from datetime import UTC, datetime

from mcp_vector_search.auditor.certifier import render_markdown, write_certification
from mcp_vector_search.auditor.models import (
    CertificationDocument,
    Evidence,
    Verdict,
)


def _make_evidence() -> Evidence:
    return Evidence(
        tool="search_code",
        query="TLS encryption",
        file_path="src/api/client.py",
        start_line=10,
        end_line=25,
        snippet="session.verify = True",
        score=0.85,
        kg_path=["main -> send_request"],
    )


def _make_verdict(status: str = "PASS", claim_id: str = "abc123def456") -> Verdict:
    return Verdict(
        claim_id=claim_id,
        status=status,
        confidence=0.9,
        reasoning=f"Evidence supports {status}.",
        evidence=[_make_evidence()],
        kg_path_present=True,
        evidence_count=1,
    )


def _make_doc(verdicts: list[Verdict] | None = None) -> CertificationDocument:
    if verdicts is None:
        verdicts = [_make_verdict("PASS")]

    summary = {
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

    all_pass = all(not v.ignored and v.status == "PASS" for v in verdicts)
    overall = "CERTIFIED" if all_pass else "CERTIFIED_WITH_EXCEPTIONS"

    return CertificationDocument(
        target_repo="/path/to/my-app",
        target_commit_sha="abc1234567890def",
        policy_path="/path/to/privacy-policy.md",
        policy_sha256="deadbeef" * 8,
        policy_snapshot_path="audits/my-app/policy-snapshot.md",
        generated_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        generator_version="3.0.76",
        auditor_model="claude-opus-4-6",
        verdicts=verdicts,
        summary=summary,
        overall_status=overall,
        content_hash="cafebabe12345678",
    )


# ---------------------------------------------------------------------------
# render_markdown tests
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_renders_header(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "# Privacy Certification" in md

    def test_renders_certified_status(self):
        doc = _make_doc([_make_verdict("PASS")])
        md = render_markdown(doc)
        assert "CERTIFIED" in md

    def test_renders_summary_table(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "## Summary" in md
        assert "| Pass |" in md or "| PASS |" in md or "Pass" in md

    def test_renders_per_claim_section(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "## Claim Verdicts" in md
        assert "abc123def456" in md

    def test_renders_evidence_file_path(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "src/api/client.py" in md

    def test_renders_kg_path(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "main -> send_request" in md

    def test_renders_confidence(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "90%" in md

    def test_renders_target_repo(self):
        doc = _make_doc()
        md = render_markdown(doc)
        assert "my-app" in md

    def test_renders_fail_status(self):
        doc = _make_doc([_make_verdict("FAIL")])
        doc.overall_status = "FAILED"
        md = render_markdown(doc)
        assert "[FAIL]" in md

    def test_renders_ignored_verdict(self):
        v = Verdict(
            claim_id="ignored001",
            status="MANUAL_REVIEW",
            confidence=1.0,
            reasoning="Ignored.",
            evidence=[],
            kg_path_present=False,
            evidence_count=0,
            ignored=True,
            ignore_justification="Manually reviewed and confirmed compliant by security team.",
        )
        doc = _make_doc([v])
        md = render_markdown(doc)
        assert "ignored" in md.lower()
        assert "Manually reviewed" in md

    def test_multiple_verdicts(self):
        verdicts = [
            _make_verdict("PASS", "claim001"),
            _make_verdict("FAIL", "claim002"),
            _make_verdict("MANUAL_REVIEW", "claim003"),
        ]
        doc = _make_doc(verdicts)
        doc.overall_status = "FAILED"
        md = render_markdown(doc)
        assert "claim001" in md
        assert "claim002" in md
        assert "claim003" in md

    def test_no_timestamps_in_snapshot_comparison(self):
        """Core content should be stable across renders (no random elements)."""
        doc = _make_doc()
        md1 = render_markdown(doc)
        md2 = render_markdown(doc)
        # Timestamp is fixed in our test doc so both renders should be identical
        assert md1 == md2


# ---------------------------------------------------------------------------
# write_certification tests
# ---------------------------------------------------------------------------


class TestWriteCertification:
    def test_creates_output_files(self, tmp_path):
        doc = _make_doc()
        cert_path = write_certification(doc, "## Privacy Policy\n\nWe care.", tmp_path)

        assert cert_path.exists()
        assert cert_path.name == "certification.md"

    def test_creates_policy_snapshot(self, tmp_path):
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text here.", tmp_path)

        snapshot_path = cert_path.parent / "policy-snapshot.md"
        assert snapshot_path.exists()
        assert "Policy text here." in snapshot_path.read_text()

    def test_directory_structure(self, tmp_path):
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)

        # Structure: <output_dir>/<target-slug>/<timestamp>/certification.md
        assert cert_path.parent.parent.name == "my-app"
        assert cert_path.parent.parent.parent == tmp_path

    def test_timestamp_in_dir_name(self, tmp_path):
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy.", tmp_path)

        # Timestamp dir should be YYYYMMDD-HHMMSS format
        timestamp_dir = cert_path.parent.name
        assert re.match(r"\d{8}-\d{6}", timestamp_dir), f"Bad dir name: {timestamp_dir}"

    def test_certification_md_contains_pass(self, tmp_path):
        doc = _make_doc([_make_verdict("PASS")])
        cert_path = write_certification(doc, "Policy.", tmp_path)

        content = cert_path.read_text()
        assert "[PASS]" in content

    def test_snapshot_contains_sha256(self, tmp_path):
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)

        snapshot = (cert_path.parent / "policy-snapshot.md").read_text()
        assert "SHA-256" in snapshot or doc.policy_sha256[:8] in snapshot
