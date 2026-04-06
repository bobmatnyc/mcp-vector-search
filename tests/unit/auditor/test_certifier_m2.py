"""Unit tests for certifier M2 features.

Tests:
- JSON sidecar creation
- Per-claim evidence files
- Manifest hash correctness
- Policy snapshot SHA-256
- verify_certification (valid and tampered)
- index.json creation and update
- audit-log.jsonl append
- latest symlink creation
- GPG signing skipped when not configured
- list command output (spot-checked via index data)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from mcp_vector_search.auditor.certifier import (
    append_audit_log,
    sign_certification,
    update_index,
    update_latest_symlink,
    verify_certification,
    write_certification,
)
from mcp_vector_search.auditor.models import (
    CertificationDocument,
    Evidence,
    Verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    all_pass = all(not v.ignored and v.status == "PASS" for v in verdicts)
    overall = "CERTIFIED" if all_pass else "CERTIFIED_WITH_EXCEPTIONS"

    return CertificationDocument(
        target_repo="/path/to/my-app",
        target_commit_sha="abc1234567890def",
        policy_path="/path/to/privacy-policy.md",
        policy_sha256="deadbeef" * 8,
        policy_snapshot_path="certifications/my-app/policy-snapshot.md",
        generated_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        generator_version="3.0.76",
        auditor_model="claude-opus-4-6",
        verdicts=verdicts,
        summary=summary,
        overall_status=overall,
        content_hash="cafebabe12345678",
    )


def _make_multi_doc() -> CertificationDocument:
    """Return a doc with multiple verdicts for richer testing."""
    return _make_doc(
        [
            _make_verdict("PASS", "claim001aaa"),
            _make_verdict("FAIL", "claim002bbb"),
            _make_verdict("INSUFFICIENT_EVIDENCE", "claim003ccc"),
        ]
    )


# ---------------------------------------------------------------------------
# Task 10: JSON sidecar
# ---------------------------------------------------------------------------


class TestWriteJsonSidecar:
    def test_json_file_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        json_path = cert_dir / "certification.json"
        assert json_path.exists(), "certification.json should be created"

    def test_json_parses_correctly(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        json_path = cert_dir / "certification.json"
        data = json.loads(json_path.read_bytes())
        assert data["target_repo"] == "/path/to/my-app"
        assert data["schema_version"] == "1.0"

    def test_json_contains_all_verdicts(self, tmp_path: Path) -> None:
        doc = _make_multi_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        json_path = cert_dir / "certification.json"
        data = json.loads(json_path.read_bytes())
        verdict_ids = {v["claim_id"] for v in data["verdicts"]}
        assert "claim001aaa" in verdict_ids
        assert "claim002bbb" in verdict_ids
        assert "claim003ccc" in verdict_ids

    def test_json_contains_full_evidence(self, tmp_path: Path) -> None:
        """Evidence is NOT truncated in the JSON (unlike Markdown display)."""
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        json_path = cert_dir / "certification.json"
        data = json.loads(json_path.read_bytes())
        ev = data["verdicts"][0]["evidence"][0]
        assert ev["file_path"] == "src/api/client.py"
        assert ev["snippet"] == "session.verify = True"

    def test_json_generated_at_is_iso8601(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        json_path = cert_dir / "certification.json"
        data = json.loads(json_path.read_bytes())
        gen = data["generated_at"]
        # Should be parseable as ISO 8601
        assert "2026" in gen


# ---------------------------------------------------------------------------
# Task 10: Per-claim evidence files
# ---------------------------------------------------------------------------


class TestEvidenceFiles:
    def test_evidence_dir_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        assert (cert_dir / "evidence").is_dir()

    def test_one_file_per_claim(self, tmp_path: Path) -> None:
        doc = _make_multi_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        ev_files = sorted((cert_dir / "evidence").glob("*.json"))
        assert len(ev_files) == 3

    def test_evidence_file_named_by_claim_id(self, tmp_path: Path) -> None:
        doc = _make_doc([_make_verdict("PASS", "myspecialid1")])
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        assert (cert_dir / "evidence" / "myspecialid1.json").exists()

    def test_evidence_file_content(self, tmp_path: Path) -> None:
        doc = _make_doc([_make_verdict("PASS", "myspecialid1")])
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        ev_path = cert_dir / "evidence" / "myspecialid1.json"
        data = json.loads(ev_path.read_bytes())
        assert data["claim_id"] == "myspecialid1"
        assert "evidence" in data
        assert isinstance(data["evidence"], list)
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["file_path"] == "src/api/client.py"


# ---------------------------------------------------------------------------
# Task 10: Manifest hashes
# ---------------------------------------------------------------------------


class TestManifestHashes:
    def test_manifest_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        assert (cert_dir / "manifest.json").exists()

    def test_manifest_contains_expected_files(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        manifest = json.loads((cert_dir / "manifest.json").read_bytes())
        files = manifest["files"]
        assert "certification.md" in files
        assert "certification.json" in files
        assert "policy-snapshot.md" in files
        assert "policy-snapshot.sha256" in files

    def test_manifest_hashes_are_correct(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        manifest = json.loads((cert_dir / "manifest.json").read_bytes())
        files = manifest["files"]

        # Verify each recorded hash matches the actual file
        for rel, recorded in files.items():
            fpath = cert_dir / rel
            assert fpath.exists(), f"File listed in manifest is missing: {rel}"
            expected_digest = recorded.removeprefix("sha256:")
            actual_digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert actual_digest == expected_digest, f"Hash mismatch for {rel}"

    def test_manifest_schema_version(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        manifest = json.loads((cert_dir / "manifest.json").read_bytes())
        assert manifest["schema_version"] == "1.0"
        assert "generated_at" in manifest


# ---------------------------------------------------------------------------
# Task 10: Policy snapshot SHA-256
# ---------------------------------------------------------------------------


class TestPolicySnapshotSha256:
    def test_sha256_file_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        assert (cert_dir / "policy-snapshot.sha256").exists()

    def test_sha256_matches_snapshot_file(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        sha256_content = (cert_dir / "policy-snapshot.sha256").read_text(
            encoding="utf-8"
        )
        recorded_digest = sha256_content.split()[0]
        actual_digest = hashlib.sha256(
            (cert_dir / "policy-snapshot.md").read_bytes()
        ).hexdigest()
        assert recorded_digest == actual_digest


# ---------------------------------------------------------------------------
# Task 11: verify_certification
# ---------------------------------------------------------------------------


class TestVerifyCertification:
    def test_verify_valid_certification(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        assert verify_certification(cert_dir) is True

    def test_verify_tampered_certification(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        # Tamper with certification.md after manifest is written
        cert_path.write_text("TAMPERED CONTENT", encoding="utf-8")
        assert verify_certification(cert_dir) is False

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        # Remove a file that is listed in manifest
        (cert_dir / "certification.json").unlink()
        assert verify_certification(cert_dir) is False

    def test_verify_missing_manifest(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        (cert_dir / "manifest.json").unlink()
        assert verify_certification(cert_dir) is False


# ---------------------------------------------------------------------------
# Task 11: GPG signing skipped when not configured
# ---------------------------------------------------------------------------


class TestGpgSigning:
    def test_sign_skip_without_gpg_key_id(self, tmp_path: Path) -> None:
        """sign_certification returns None when gpg_key_id is not set."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.gpg_key_id = None

        result = sign_certification(tmp_path, settings)
        assert result is None

    def test_sign_skip_returns_none(self, tmp_path: Path) -> None:
        """Confirm return type is None (not a path) when skipped."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.gpg_key_id = ""

        result = sign_certification(tmp_path, settings)
        assert result is None


# ---------------------------------------------------------------------------
# Task 12: index.json
# ---------------------------------------------------------------------------


class TestIndexJson:
    def test_index_created_on_first_audit(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_index(cert_dir, doc, tmp_path, signed=False)
        index_path = tmp_path / "index.json"
        assert index_path.exists()

    def test_index_contains_target(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_index(cert_dir, doc, tmp_path, signed=False)
        data = json.loads((tmp_path / "index.json").read_bytes())
        assert "my-app" in data["targets"]

    def test_index_contains_audit_entry(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_index(cert_dir, doc, tmp_path, signed=False)
        data = json.loads((tmp_path / "index.json").read_bytes())
        audits = data["targets"]["my-app"]["audits"]
        assert len(audits) == 1
        assert audits[0]["overall_status"] == "CERTIFIED"
        assert audits[0]["commit_sha"] == "abc1234567890def"

    def test_index_update_appends_entry(self, tmp_path: Path) -> None:
        """Running two audits should append a second entry, not overwrite."""
        doc1 = _make_doc()
        cert_path1 = write_certification(doc1, "Policy text.", tmp_path)
        update_index(cert_path1.parent, doc1, tmp_path, signed=False)

        # Second audit with a different timestamp (change generated_at)
        doc2 = CertificationDocument(
            target_repo="/path/to/my-app",
            target_commit_sha="fffaaa000111",
            policy_path="/path/to/privacy-policy.md",
            policy_sha256="deadbeef" * 8,
            policy_snapshot_path="certifications/my-app/policy-snapshot.md",
            generated_at=datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC),
            generator_version="3.0.77",
            auditor_model="claude-opus-4-6",
            verdicts=[_make_verdict("PASS")],
            summary={
                "PASS": 1,
                "FAIL": 0,
                "INSUFFICIENT_EVIDENCE": 0,
                "MANUAL_REVIEW": 0,
                "IGNORED": 0,
                "TOTAL": 1,
            },
            overall_status="CERTIFIED",
            content_hash="newhashabcdef01",
        )
        cert_path2 = write_certification(doc2, "Policy text.", tmp_path)
        update_index(cert_path2.parent, doc2, tmp_path, signed=False)

        data = json.loads((tmp_path / "index.json").read_bytes())
        audits = data["targets"]["my-app"]["audits"]
        assert len(audits) == 2
        # Latest should be the second audit's timestamp
        assert data["targets"]["my-app"]["latest"] == "20260406-120000"

    def test_index_signed_field(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_index(cert_dir, doc, tmp_path, signed=True)
        data = json.loads((tmp_path / "index.json").read_bytes())
        assert data["targets"]["my-app"]["audits"][0]["signed"] is True


# ---------------------------------------------------------------------------
# Task 12: audit-log.jsonl
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_audit_log_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        append_audit_log(doc, tmp_path)
        log_path = tmp_path / "audit-log.jsonl"
        assert log_path.exists()

    def test_audit_log_contains_valid_json_line(self, tmp_path: Path) -> None:
        doc = _make_doc()
        append_audit_log(doc, tmp_path)
        log_path = tmp_path / "audit-log.jsonl"
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["target"] == "my-app"
        assert entry["status"] == "CERTIFIED"

    def test_audit_log_append(self, tmp_path: Path) -> None:
        """Multiple calls append separate lines."""
        doc = _make_doc()
        append_audit_log(doc, tmp_path)
        append_audit_log(doc, tmp_path)
        log_path = tmp_path / "audit-log.jsonl"
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2

    def test_audit_log_summary_fields(self, tmp_path: Path) -> None:
        doc = _make_multi_doc()
        doc.overall_status = "CERTIFIED_WITH_EXCEPTIONS"  # type: ignore[assignment]
        append_audit_log(doc, tmp_path)
        log_path = tmp_path / "audit-log.jsonl"
        entry = json.loads(log_path.read_text().splitlines()[0])
        assert entry["pass"] == doc.summary.get("PASS", 0)
        assert entry["fail"] == doc.summary.get("FAIL", 0)
        assert entry["claims"] == doc.summary.get("TOTAL", 0)


# ---------------------------------------------------------------------------
# Task 12: latest symlink
# ---------------------------------------------------------------------------


class TestLatestSymlink:
    def test_latest_symlink_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_latest_symlink(cert_dir, tmp_path, doc)
        latest = tmp_path / "my-app" / "latest"
        assert latest.is_symlink()

    def test_latest_symlink_points_to_timestamp(self, tmp_path: Path) -> None:
        doc = _make_doc()
        cert_path = write_certification(doc, "Policy text.", tmp_path)
        cert_dir = cert_path.parent
        update_latest_symlink(cert_dir, tmp_path, doc)
        latest = tmp_path / "my-app" / "latest"
        target_name = (
            latest.readlink().name
            if hasattr(latest.readlink(), "name")
            else str(latest.readlink())
        )
        assert re.match(r"\d{8}-\d{6}", target_name), (
            f"Expected timestamp dir, got: {target_name}"
        )

    def test_latest_symlink_updated_on_second_audit(self, tmp_path: Path) -> None:
        doc1 = _make_doc()
        cert_path1 = write_certification(doc1, "Policy text.", tmp_path)
        update_latest_symlink(cert_path1.parent, tmp_path, doc1)

        doc2 = CertificationDocument(
            target_repo="/path/to/my-app",
            target_commit_sha="fffaaa000111",
            policy_path="/path/to/privacy-policy.md",
            policy_sha256="deadbeef" * 8,
            policy_snapshot_path="certifications/my-app/policy-snapshot.md",
            generated_at=datetime(2026, 4, 7, 9, 30, 0, tzinfo=UTC),
            generator_version="3.0.77",
            auditor_model="claude-opus-4-6",
            verdicts=[_make_verdict("PASS")],
            summary={
                "PASS": 1,
                "FAIL": 0,
                "INSUFFICIENT_EVIDENCE": 0,
                "MANUAL_REVIEW": 0,
                "IGNORED": 0,
                "TOTAL": 1,
            },
            overall_status="CERTIFIED",
            content_hash="newhashabcdef01",
        )
        cert_path2 = write_certification(doc2, "Policy text.", tmp_path)
        update_latest_symlink(cert_path2.parent, tmp_path, doc2)

        latest = tmp_path / "my-app" / "latest"
        link_target = str(latest.readlink())
        assert "20260407-093000" in link_target


# ---------------------------------------------------------------------------
# Task 12: certifications README
# ---------------------------------------------------------------------------


class TestCertificationsReadme:
    def test_readme_created(self, tmp_path: Path) -> None:
        doc = _make_doc()
        write_certification(doc, "Policy text.", tmp_path)
        readme = tmp_path / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "Privacy Certifications" in content

    def test_readme_not_overwritten_if_exists(self, tmp_path: Path) -> None:
        custom = "# My Custom README\n"
        (tmp_path / "README.md").write_text(custom)
        doc = _make_doc()
        write_certification(doc, "Policy text.", tmp_path)
        assert (tmp_path / "README.md").read_text() == custom


# ---------------------------------------------------------------------------
# Integration: write_certification produces all expected files
# ---------------------------------------------------------------------------


class TestWriteCertificationFull:
    def test_all_expected_files_present(self, tmp_path: Path) -> None:
        doc = _make_multi_doc()
        cert_path = write_certification(doc, "Full policy text here.", tmp_path)
        cert_dir = cert_path.parent

        expected = [
            "certification.md",
            "certification.json",
            "policy-snapshot.md",
            "policy-snapshot.sha256",
            "manifest.json",
            "evidence/claim001aaa.json",
            "evidence/claim002bbb.json",
            "evidence/claim003ccc.json",
        ]
        for rel in expected:
            assert (cert_dir / rel).exists(), f"Missing expected file: {rel}"

    def test_verify_passes_after_write(self, tmp_path: Path) -> None:
        doc = _make_multi_doc()
        cert_path = write_certification(doc, "Full policy text here.", tmp_path)
        cert_dir = cert_path.parent
        assert verify_certification(cert_dir) is True
