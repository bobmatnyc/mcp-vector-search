"""Unit tests for M3 CI artifacts (Task 13 + 14).

Tests:
- test_workflow_yaml_valid       — load and validate the YAML structure
- test_run_audit_script_exists  — verify script exists and is executable
- test_docs_exists              — verify docs/features/privacy-auditor.md exists
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import yaml

# Resolve repo root relative to this test file's location
# tests/unit/auditor/ -> ../../.. -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "privacy-audit.yml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "run_audit.sh"
DOCS_PATH = REPO_ROOT / "docs" / "features" / "privacy-auditor.md"


class TestWorkflowYaml:
    """Validate the GitHub Actions workflow YAML for privacy-audit."""

    def test_workflow_yaml_valid(self) -> None:
        """Workflow YAML must be parseable by PyYAML."""
        assert WORKFLOW_PATH.exists(), f"Workflow not found: {WORKFLOW_PATH}"
        content = WORKFLOW_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "Workflow YAML must parse to a dict"

    def test_workflow_name(self) -> None:
        """Workflow must be named 'Privacy Policy Audit'."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        assert data.get("name") == "Privacy Policy Audit"

    def test_workflow_has_workflow_dispatch(self) -> None:
        """Workflow must be triggered by workflow_dispatch."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        on = data.get("on") or data.get(
            True
        )  # YAML 'on' parses as True in some loaders
        assert on is not None, "Workflow must have an 'on' trigger"
        assert "workflow_dispatch" in on, (
            "Workflow must support workflow_dispatch trigger"
        )

    def test_workflow_dispatch_inputs(self) -> None:
        """workflow_dispatch must declare the required inputs."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        on = data.get("on") or data.get(True)
        dispatch = on["workflow_dispatch"]
        inputs = dispatch.get("inputs", {})
        required_inputs = {
            "target_repo",
            "policy_path",
            "require_kg_path",
            "llm_backend",
        }
        missing = required_inputs - set(inputs.keys())
        assert not missing, f"Missing workflow_dispatch inputs: {missing}"

    def test_workflow_target_repo_required(self) -> None:
        """target_repo input must be marked required."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        on = data.get("on") or data.get(True)
        inputs = on["workflow_dispatch"]["inputs"]
        assert inputs["target_repo"].get("required") is True, (
            "target_repo input must be required"
        )

    def test_workflow_policy_path_has_default(self) -> None:
        """policy_path input must have a default value of 'PRIVACY.md'."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        on = data.get("on") or data.get(True)
        inputs = on["workflow_dispatch"]["inputs"]
        assert inputs["policy_path"].get("default") == "PRIVACY.md", (
            "policy_path default must be 'PRIVACY.md'"
        )

    def test_workflow_llm_backend_choices(self) -> None:
        """llm_backend input must offer openrouter and anthropic as choices."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        on = data.get("on") or data.get(True)
        inputs = on["workflow_dispatch"]["inputs"]
        backend = inputs.get("llm_backend", {})
        options = backend.get("options", [])
        assert "openrouter" in options, "llm_backend must include 'openrouter' option"
        assert "anthropic" in options, "llm_backend must include 'anthropic' option"

    def test_workflow_permissions(self) -> None:
        """Workflow must declare contents:write and issues:write permissions."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        perms = data.get("permissions", {})
        assert perms.get("contents") == "write", (
            "Workflow needs contents: write permission"
        )
        assert perms.get("issues") == "write", "Workflow needs issues: write permission"

    def test_workflow_has_audit_job(self) -> None:
        """Workflow must define an 'audit' job."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        jobs = data.get("jobs", {})
        assert "audit" in jobs, "Workflow must have an 'audit' job"

    def test_workflow_audit_runs_on_ubuntu(self) -> None:
        """The audit job must run on ubuntu-latest."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        audit_job = data["jobs"]["audit"]
        assert audit_job.get("runs-on") == "ubuntu-latest"

    def test_workflow_uses_python_311(self) -> None:
        """Workflow must set up Python 3.11."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = data["jobs"]["audit"]["steps"]
        setup_steps = [
            s for s in steps if s.get("uses", "").startswith("actions/setup-python")
        ]
        assert setup_steps, "Workflow must use actions/setup-python"
        py_version = setup_steps[0].get("with", {}).get("python-version")
        assert py_version == "3.11", (
            f"Expected python-version '3.11', got {py_version!r}"
        )

    def test_workflow_installs_auditor_extra(self) -> None:
        """Install step must install mcp-vector-search with [auditor] extra."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = data["jobs"]["audit"]["steps"]
        install_steps = [s for s in steps if "Install" in s.get("name", "")]
        assert install_steps, "Workflow must have an Install dependencies step"
        install_run = install_steps[0].get("run", "")
        assert "[auditor]" in install_run, (
            "Install step must install mcp-vector-search[auditor]"
        )

    def test_workflow_uploads_artifact(self) -> None:
        """Workflow must upload a certification artifact."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = data["jobs"]["audit"]["steps"]
        upload_steps = [
            s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")
        ]
        assert upload_steps, "Workflow must use actions/upload-artifact"

    def test_workflow_artifact_retention_90_days(self) -> None:
        """Artifact retention must be set to 90 days."""
        data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = data["jobs"]["audit"]["steps"]
        upload_steps = [
            s for s in steps if s.get("uses", "").startswith("actions/upload-artifact")
        ]
        assert upload_steps
        retention = upload_steps[0].get("with", {}).get("retention-days")
        assert retention == 90, f"Expected retention-days 90, got {retention!r}"

    def test_workflow_posts_summary(self) -> None:
        """Workflow must include a step that writes to GITHUB_STEP_SUMMARY."""
        content = WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "GITHUB_STEP_SUMMARY" in content, (
            "Workflow must write to GITHUB_STEP_SUMMARY for the audit summary"
        )

    def test_workflow_commits_results(self) -> None:
        """Workflow must commit certification results back to the repo."""
        content = WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "git commit" in content, "Workflow must commit certification results"
        assert "git push" in content, (
            "Workflow must push committed certification results"
        )

    def test_workflow_passes_secrets_to_audit(self) -> None:
        """Run audit step must pass OPENROUTER_API_KEY and ANTHROPIC_API_KEY secrets."""
        content = WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY" in content
        assert "ANTHROPIC_API_KEY" in content


class TestRunAuditScript:
    """Validate the CI wrapper shell script."""

    def test_run_audit_script_exists(self) -> None:
        """scripts/ci/run_audit.sh must exist."""
        assert SCRIPT_PATH.exists(), f"CI script not found: {SCRIPT_PATH}"

    def test_run_audit_script_is_executable(self) -> None:
        """scripts/ci/run_audit.sh must be executable by the owner."""
        assert SCRIPT_PATH.exists(), f"CI script not found: {SCRIPT_PATH}"
        mode = os.stat(SCRIPT_PATH).st_mode
        assert mode & stat.S_IXUSR, (
            f"Script {SCRIPT_PATH} is not executable. Run: chmod +x {SCRIPT_PATH}"
        )

    def test_run_audit_script_has_shebang(self) -> None:
        """Script must start with the bash shebang."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash"), (
            "Script must start with '#!/usr/bin/env bash'"
        )

    def test_run_audit_script_set_euo_pipefail(self) -> None:
        """Script must use 'set -euo pipefail' for safe error handling."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "set -euo pipefail" in content, "Script must include 'set -euo pipefail'"

    def test_run_audit_script_calls_mvs_index(self) -> None:
        """Script must call 'mvs index' to index the target repository."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "mvs index" in content, "Script must call 'mvs index'"

    def test_run_audit_script_calls_mvs_audit_run(self) -> None:
        """Script must call 'mvs audit run' to execute the audit."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "mvs audit run" in content, "Script must call 'mvs audit run'"

    def test_run_audit_script_calls_mvs_audit_verify(self) -> None:
        """Script must call 'mvs audit verify' to validate the output."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "mvs audit verify" in content, "Script must call 'mvs audit verify'"

    def test_run_audit_script_documents_exit_codes(self) -> None:
        """Script must document exit codes (CERTIFIED, FAILED, WITH EXCEPTIONS)."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "CERTIFIED" in content
        assert "FAILED" in content

    def test_run_audit_script_documents_env_vars(self) -> None:
        """Script must document the required environment variables in comments."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY" in content or "ANTHROPIC_API_KEY" in content, (
            "Script must document required API key environment variables"
        )

    def test_run_audit_script_output_dir_defaults_to_certifications(self) -> None:
        """Script must default OUTPUT_DIR to 'certifications'."""
        content = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "certifications" in content, (
            "Script must use 'certifications' as the default output directory"
        )


class TestDocsExist:
    """Validate that the user-facing documentation exists."""

    def test_docs_exists(self) -> None:
        """docs/features/privacy-auditor.md must exist."""
        assert DOCS_PATH.exists(), f"Documentation not found: {DOCS_PATH}"

    def test_docs_has_quick_start_section(self) -> None:
        """Documentation must include a Quick Start section."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "Quick Start" in content, "Docs must have a 'Quick Start' section"

    def test_docs_covers_cli_usage(self) -> None:
        """Documentation must cover CLI usage with mvs audit run."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "mvs audit run" in content, "Docs must document 'mvs audit run'"

    def test_docs_covers_github_action(self) -> None:
        """Documentation must cover GitHub Action usage."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "GitHub Action" in content or "workflow_dispatch" in content, (
            "Docs must cover GitHub Actions usage"
        )

    def test_docs_lists_required_secrets(self) -> None:
        """Documentation must list required secrets."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY" in content or "ANTHROPIC_API_KEY" in content, (
            "Docs must list required API key secrets"
        )

    def test_docs_explains_verdict_types(self) -> None:
        """Documentation must explain the verdict types."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        for verdict in ("PASS", "FAIL", "INSUFFICIENT"):
            assert verdict in content, f"Docs must mention verdict type: {verdict}"

    def test_docs_explains_overall_status(self) -> None:
        """Documentation must explain the overall status and exit codes."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "CERTIFIED" in content
        assert "Exit Code" in content or "exit code" in content.lower()

    def test_docs_describes_certification_output(self) -> None:
        """Documentation must describe the certification output files."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "certification.md" in content
        assert "certification.json" in content

    def test_docs_covers_configuration(self) -> None:
        """Documentation must cover environment variable configuration."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert "MVS_AUDIT_" in content, (
            "Docs must document MVS_AUDIT_ environment variable prefix"
        )

    def test_docs_covers_audit_ignore(self) -> None:
        """Documentation must explain the .audit-ignore.yml mechanism."""
        content = DOCS_PATH.read_text(encoding="utf-8")
        assert ".audit-ignore.yml" in content, (
            "Docs must explain the .audit-ignore.yml suppress mechanism"
        )
