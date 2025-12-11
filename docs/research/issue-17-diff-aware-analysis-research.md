# Issue #17: Diff-Aware Analysis - Research Document

**Research Date**: December 11, 2025
**Researcher**: Claude (Research Agent)
**Issue**: [#17 - Diff-aware analysis](https://github.com/bobmatnyc/mcp-vector-search/issues/17)
**Milestone**: v0.18.0 - Quality Gates
**Phase**: 2 of 5 - Quality Gates
**Status**: 📋 Backlog (Ready after Issue #10 completes)

---

## Executive Summary

Issue #17 implements **diff-aware analysis** capability for the `mcp-vector-search analyze` command, enabling CI/CD pipelines to analyze only changed files in a git repository. This feature reduces analysis time and provides focused quality feedback on code changes, making it ideal for pull request checks and continuous integration workflows.

**Key Findings:**
- **Purpose**: Optimize analysis for CI/CD by analyzing only git-changed files
- **Dependencies**: Requires Issue #10 (Analyze CLI command) to be completed
- **Impact**: Significant performance improvement for incremental analysis in CI/CD
- **Complexity**: Moderate - requires git integration and baseline comparison
- **Timeline**: Part of Phase 2 (v0.18.0), due December 24-30, 2024

---

## 1. Requirements Specification

### 1.1 Functional Requirements

Based on the design document and project roadmap, Issue #17 must implement:

#### FR-1: Git Changed Files Detection
- **Description**: Detect files modified in the git working directory
- **Command Interface**: `mcp-vector-search analyze --changed-only`
- **Behavior**:
  - Analyze only files with uncommitted changes
  - Support both staged and unstaged changes
  - Respect `.gitignore` patterns

#### FR-2: Baseline Comparison
- **Description**: Compare current branch against a baseline branch
- **Command Interface**: `mcp-vector-search analyze --baseline <branch>`
- **Behavior**:
  - Compute diff between current branch and baseline branch
  - Analyze only files that differ between branches
  - Support common baseline targets: `main`, `master`, `develop`, `HEAD~1`

#### FR-3: Combined Diff Modes
- **Description**: Support combined usage of diff flags
- **Examples**:
  ```bash
  # Analyze uncommitted changes
  mcp-vector-search analyze --changed-only

  # Analyze changes vs main branch
  mcp-vector-search analyze --baseline main

  # Analyze specific path with baseline
  mcp-vector-search analyze --baseline main --path src/core
  ```

#### FR-4: CI/CD Integration
- **Description**: Exit codes and output optimized for CI/CD
- **Behavior**:
  - Exit code 0: No issues found in changed files
  - Exit code 1: Quality gate failures in changed files
  - Exit code 2: Analysis errors (git not available, branch not found)
  - SARIF output support for GitHub Code Scanning

#### FR-5: Performance Optimization
- **Description**: Skip analysis of unchanged files
- **Metrics**:
  - Only parse files in git diff result
  - Cache unchanged file metrics
  - Report time savings in output

### 1.2 Non-Functional Requirements

#### NFR-1: Git Availability
- **Requirement**: Gracefully handle missing git binary
- **Behavior**: Clear error message if git not in PATH
- **Fallback**: Suggest full analysis without `--changed-only`

#### NFR-2: Performance
- **Target**: 10x speedup for typical PR (5-10 changed files out of 100+ total)
- **Measurement**: Report "X files analyzed (Y skipped)" in output

#### NFR-3: Accuracy
- **Requirement**: Diff detection must match `git status` and `git diff`
- **Validation**: Support binary files, renamed files, deleted files

#### NFR-4: User Experience
- **Requirement**: Clear feedback on what's being analyzed
- **Output Example**:
  ```
  📊 Analyzing changed files (baseline: main)
  Files to analyze: 8 changed (127 total in project)
  ```

---

## 2. Technical Design

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ CLI Command: analyze --changed-only --baseline main     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Git Integration Layer (NEW)                             │
│ - GitManager class                                       │
│ - Changed file detection                                │
│ - Baseline comparison                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ File Filter (MODIFIED)                                   │
│ - Apply git diff filter to file list                    │
│ - Combine with existing filters (language, path)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Existing Analysis Pipeline                               │
│ - Parse filtered files                                   │
│ - Run collectors                                         │
│ - Generate reports                                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 New Component: GitManager

**File**: `src/mcp_vector_search/core/git.py`

```python
"""Git integration for diff-aware analysis."""

import subprocess
from pathlib import Path
from typing import List, Optional
from loguru import logger


class GitError(Exception):
    """Git operation failed."""
    pass


class GitManager:
    """Manage git operations for diff-aware analysis."""

    def __init__(self, project_root: Path):
        """Initialize git manager.

        Args:
            project_root: Root directory of the project

        Raises:
            GitError: If project_root is not a git repository
        """
        self.project_root = project_root
        self.git_dir = project_root / ".git"

        if not self.is_git_repo():
            raise GitError(f"Not a git repository: {project_root}")

    def is_git_repo(self) -> bool:
        """Check if project is a git repository."""
        return self.git_dir.exists()

    def is_git_available(self) -> bool:
        """Check if git command is available."""
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_changed_files(self, include_untracked: bool = True) -> List[Path]:
        """Get list of changed files in working directory.

        Args:
            include_untracked: Include untracked files

        Returns:
            List of changed file paths relative to project root
        """
        cmd = ["git", "status", "--porcelain"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            changed_files = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                # Parse git status porcelain format
                # Format: XY filename
                # X = index status, Y = working tree status
                status = line[:2]
                filename = line[3:].strip()

                # Handle renamed files: "R  old -> new"
                if " -> " in filename:
                    filename = filename.split(" -> ")[1]

                # Skip deleted files
                if "D" in status:
                    continue

                # Skip untracked if not requested
                if not include_untracked and status.startswith("??"):
                    continue

                file_path = self.project_root / filename
                if file_path.exists():
                    changed_files.append(file_path)

            return changed_files

        except subprocess.CalledProcessError as e:
            logger.error(f"Git status failed: {e.stderr}")
            raise GitError(f"Failed to get changed files: {e.stderr}")

    def get_diff_files(self, baseline: str = "main") -> List[Path]:
        """Get list of files that differ from baseline branch.

        Args:
            baseline: Baseline branch or commit (default: main)

        Returns:
            List of changed file paths relative to project root
        """
        # First, check if baseline exists
        if not self.ref_exists(baseline):
            # Try common alternatives
            for alt in ["master", "develop", "HEAD~1"]:
                if self.ref_exists(alt):
                    logger.warning(
                        f"Baseline '{baseline}' not found, using '{alt}' instead"
                    )
                    baseline = alt
                    break
            else:
                raise GitError(
                    f"Baseline '{baseline}' not found. "
                    "Try: main, master, develop, or HEAD~1"
                )

        # Get list of changed files
        cmd = ["git", "diff", "--name-only", baseline]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            changed_files = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                file_path = self.project_root / line.strip()
                if file_path.exists():
                    changed_files.append(file_path)

            return changed_files

        except subprocess.CalledProcessError as e:
            logger.error(f"Git diff failed: {e.stderr}")
            raise GitError(f"Failed to get diff files: {e.stderr}")

    def ref_exists(self, ref: str) -> bool:
        """Check if a git ref (branch, tag, commit) exists.

        Args:
            ref: Git reference to check

        Returns:
            True if ref exists
        """
        cmd = ["git", "rev-parse", "--verify", ref]

        try:
            subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_current_branch(self) -> Optional[str]:
        """Get name of current branch.

        Returns:
            Branch name or None if detached HEAD
        """
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )

            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None

        except subprocess.CalledProcessError:
            return None
```

### 2.3 CLI Integration

**Modifications to**: `src/mcp_vector_search/cli/commands/analyze.py`

```python
# Add new CLI options
@analyze_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # ... existing parameters ...

    # NEW: Diff-aware options
    changed_only: bool = typer.Option(
        False,
        "--changed-only",
        help="Analyze only git-changed files (uncommitted changes)",
        rich_help_panel="🔍 Filters",
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help="Compare against baseline branch (e.g., main, develop, HEAD~1)",
        rich_help_panel="🔍 Filters",
    ),
    # ... rest of parameters ...
) -> None:
    """Enhanced analyze command with diff-aware options."""

    # Validate git options
    if changed_only and baseline:
        print_error("Cannot use --changed-only and --baseline together")
        raise typer.Exit(1)

    # ... existing validation ...

    asyncio.run(
        run_analysis(
            project_root=project_root,
            # ... existing parameters ...
            changed_only=changed_only,
            baseline=baseline,
        )
    )


async def run_analysis(
    project_root: Path,
    # ... existing parameters ...
    changed_only: bool = False,
    baseline: str | None = None,
) -> None:
    """Enhanced run_analysis with diff-aware filtering."""

    # ... existing code ...

    # NEW: Apply git filtering
    git_filter: List[Path] | None = None

    if changed_only or baseline:
        try:
            from ...core.git import GitManager, GitError

            git_manager = GitManager(project_root)

            if changed_only:
                git_filter = git_manager.get_changed_files()
                filter_desc = "changed files (uncommitted)"
            elif baseline:
                git_filter = git_manager.get_diff_files(baseline)
                filter_desc = f"changed files (vs {baseline})"

            if not git_filter:
                print_error(f"No {filter_desc} found")
                return

            if not json_output:
                console.print(
                    f"[blue]ℹ️ Git filter:[/blue] {len(git_filter)} {filter_desc}"
                )

        except GitError as e:
            print_error(f"Git error: {e}")
            if not json_output:
                print_info("Tip: Run without --changed-only/--baseline for full analysis")
            raise typer.Exit(2)

    # Find files to analyze (with git filter applied)
    files_to_analyze = _find_analyzable_files(
        project_root,
        language_filter,
        path_filter,
        parser_registry,
        git_filter=git_filter  # NEW parameter
    )

    # ... rest of existing analysis logic ...


def _find_analyzable_files(
    project_root: Path,
    language_filter: str | None,
    path_filter: Path | None,
    parser_registry: ParserRegistry,
    git_filter: List[Path] | None = None,  # NEW parameter
) -> list[Path]:
    """Enhanced file finder with git filtering.

    Args:
        project_root: Root directory
        language_filter: Optional language filter
        path_filter: Optional path filter
        parser_registry: Parser registry for checking supported files
        git_filter: Optional list of git-changed files (NEW)

    Returns:
        List of file paths to analyze
    """
    # ... existing file finding logic ...

    # NEW: Apply git filter if provided
    if git_filter is not None:
        # Convert git_filter to set for O(1) lookup
        git_files = set(git_filter)

        # Filter files to only those in git_filter
        files = [f for f in files if f in git_files]

    return sorted(files)
```

### 2.4 Error Handling

```python
# Error scenarios and handling

class GitError(Exception):
    """Base exception for git-related errors."""
    pass


# Error Scenario 1: Git not available
try:
    git_manager = GitManager(project_root)
except GitError as e:
    print_error("Git not available. Install git or run without --changed-only")
    raise typer.Exit(2)


# Error Scenario 2: Not a git repository
try:
    git_manager = GitManager(project_root)
except GitError as e:
    print_error(f"Not a git repository: {project_root}")
    print_info("Initialize git: git init")
    raise typer.Exit(2)


# Error Scenario 3: Baseline branch not found
try:
    files = git_manager.get_diff_files(baseline="nonexistent")
except GitError as e:
    print_error(f"Baseline not found: {e}")
    print_info("Available branches: git branch -a")
    raise typer.Exit(2)


# Error Scenario 4: No changed files
files = git_manager.get_changed_files()
if not files:
    print_info("No changed files found. Nothing to analyze.")
    raise typer.Exit(0)  # Success, just nothing to do
```

---

## 3. Acceptance Criteria

Based on Phase 2 requirements, Issue #17 is **DONE** when:

### AC-1: Changed Files Detection
- [ ] `mcp-vector-search analyze --changed-only` analyzes only uncommitted changes
- [ ] Both staged and unstaged changes are included
- [ ] Untracked files are included by default
- [ ] Deleted files are excluded from analysis

### AC-2: Baseline Comparison
- [ ] `mcp-vector-search analyze --baseline main` analyzes files differing from main branch
- [ ] Supports common baselines: `main`, `master`, `develop`, `HEAD~1`
- [ ] Auto-fallback if baseline not found (main → master → develop)
- [ ] Clear error if no valid baseline exists

### AC-3: Performance
- [ ] Analysis time reduced by 10x for typical PR (5 changed files out of 100)
- [ ] Output reports: "X files analyzed (Y skipped)"
- [ ] Unchanged file metrics are not recomputed

### AC-4: CI/CD Integration
- [ ] Exit code 0 for clean analysis
- [ ] Exit code 1 for quality gate failures (with `--fail-on-smell`)
- [ ] Exit code 2 for git errors
- [ ] Works with `--format sarif` for GitHub Code Scanning

### AC-5: Error Handling
- [ ] Clear error message if git not installed
- [ ] Clear error message if not a git repository
- [ ] Graceful handling of detached HEAD state
- [ ] Informative error for missing baseline

### AC-6: User Experience
- [ ] Output shows "Analyzing changed files (baseline: main)"
- [ ] Output shows count: "8 changed (127 total)"
- [ ] `--changed-only` and `--baseline` are mutually exclusive
- [ ] Help text explains diff-aware options clearly

### AC-7: Testing
- [ ] Unit tests for GitManager class
- [ ] Integration tests with real git repository
- [ ] Tests for error scenarios (no git, not a repo, bad baseline)
- [ ] Tests for combined filters (git + language + path)

---

## 4. Implementation Approach

### 4.1 Recommended Implementation Steps

**Step 1: Create GitManager class**
- Location: `src/mcp_vector_search/core/git.py`
- Implement core git operations (status, diff, ref checking)
- Add error handling and logging
- Write unit tests

**Step 2: Extend CLI interface**
- Add `--changed-only` and `--baseline` options
- Add validation (mutually exclusive)
- Update help text and examples

**Step 3: Integrate git filtering**
- Modify `_find_analyzable_files()` to accept git filter
- Apply git filter to file list
- Add performance metrics to output

**Step 4: Testing**
- Create test repository fixture
- Test changed file detection
- Test baseline comparison
- Test error scenarios
- Test combined filters

**Step 5: Documentation**
- Update CLI help text
- Add examples to README
- Document CI/CD integration patterns
- Add troubleshooting guide

### 4.2 Testing Strategy

```python
# tests/test_git/test_git_manager.py

import pytest
from pathlib import Path
from mcp_vector_search.core.git import GitManager, GitError


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, check=True
    )

    # Create initial commit
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path, check=True
    )

    return tmp_path


def test_git_manager_init(git_repo):
    """Test GitManager initialization."""
    manager = GitManager(git_repo)
    assert manager.project_root == git_repo
    assert manager.is_git_repo()


def test_git_manager_not_a_repo(tmp_path):
    """Test GitManager with non-git directory."""
    with pytest.raises(GitError, match="Not a git repository"):
        GitManager(tmp_path)


def test_get_changed_files_uncommitted(git_repo):
    """Test detection of uncommitted changes."""
    manager = GitManager(git_repo)

    # Create a new file
    new_file = git_repo / "new.py"
    new_file.write_text("print('new')")

    # Modify existing file
    test_file = git_repo / "test.py"
    test_file.write_text("print('modified')")

    changed = manager.get_changed_files()
    assert len(changed) == 2
    assert new_file in changed
    assert test_file in changed


def test_get_diff_files_baseline(git_repo):
    """Test baseline comparison."""
    manager = GitManager(git_repo)

    # Create a new branch
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=git_repo, check=True
    )

    # Create new file in feature branch
    new_file = git_repo / "feature.py"
    new_file.write_text("print('feature')")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add feature"],
        cwd=git_repo, check=True
    )

    # Get diff vs main
    changed = manager.get_diff_files(baseline="main")
    assert len(changed) == 1
    assert new_file in changed


def test_baseline_not_found(git_repo):
    """Test error handling for missing baseline."""
    manager = GitManager(git_repo)

    with pytest.raises(GitError, match="Baseline .* not found"):
        manager.get_diff_files(baseline="nonexistent")


def test_changed_files_exclude_deleted(git_repo):
    """Test that deleted files are not included."""
    manager = GitManager(git_repo)

    # Delete existing file
    test_file = git_repo / "test.py"
    test_file.unlink()

    # Create new file
    new_file = git_repo / "new.py"
    new_file.write_text("print('new')")

    changed = manager.get_changed_files()
    assert test_file not in changed  # Deleted file excluded
    assert new_file in changed  # New file included
```

### 4.3 CI/CD Integration Examples

**GitHub Actions Workflow:**

```yaml
# .github/workflows/quality-gate.yml

name: Quality Gate

on:
  pull_request:
    branches: [main, develop]

jobs:
  analyze-changes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Fetch full history for baseline comparison

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install mcp-vector-search
        run: pip install mcp-vector-search

      - name: Analyze changed files
        run: |
          mcp-vector-search analyze \
            --baseline ${{ github.base_ref }} \
            --fail-on-smell \
            --severity-threshold warning \
            --format sarif \
            --output results.sarif

      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: results.sarif
```

**GitLab CI:**

```yaml
# .gitlab-ci.yml

code-quality:
  stage: test
  script:
    - pip install mcp-vector-search
    - |
      mcp-vector-search analyze \
        --baseline main \
        --fail-on-smell \
        --format json \
        --output quality-report.json
  artifacts:
    reports:
      codequality: quality-report.json
  only:
    - merge_requests
```

---

## 5. Dependencies & Blockers

### 5.1 Upstream Dependencies

**Issue #10: Create `analyze --quick` CLI**
- **Status**: 🎯 Ready (as of research date)
- **Why Required**: Issue #17 extends the `analyze` command with new options
- **Impact**: Cannot implement diff-aware analysis without base CLI command
- **Risk**: LOW - Issue #10 is on critical path and marked ready

### 5.2 Related Issues

**Issue #18: Baseline comparison**
- **Relationship**: Issue #17 implements `--baseline`, Issue #18 stores/compares metrics
- **Sequence**: Issue #17 provides the filtering, Issue #18 adds metric comparison
- **Dependencies**: Issue #18 depends on Issue #17

**Issue #15: SARIF output format**
- **Relationship**: Diff-aware analysis works well with SARIF for CI/CD
- **Sequence**: Can be developed in parallel
- **Integration**: Both issues target CI/CD workflows

---

## 6. Testing & Validation

### 6.1 Unit Tests

```python
# tests/test_git/test_git_integration.py

async def test_analyze_changed_only_cli(git_repo, cli_runner):
    """Test --changed-only CLI option."""
    # Modify a file
    test_file = git_repo / "src" / "test.py"
    test_file.write_text("print('modified')")

    # Run analysis
    result = cli_runner.invoke(
        analyze_app,
        ["--changed-only", "--json"],
        cwd=git_repo
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["total_files"] == 1
    assert "test.py" in str(output["files"])


async def test_analyze_baseline_cli(git_repo, cli_runner):
    """Test --baseline CLI option."""
    # Create feature branch with changes
    subprocess.run(
        ["git", "checkout", "-b", "feature"],
        cwd=git_repo, check=True
    )
    new_file = git_repo / "src" / "feature.py"
    new_file.write_text("print('feature')")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add feature"],
        cwd=git_repo, check=True
    )

    # Run analysis
    result = cli_runner.invoke(
        analyze_app,
        ["--baseline", "main", "--json"],
        cwd=git_repo
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["total_files"] == 1
    assert "feature.py" in str(output["files"])
```

### 6.2 Integration Tests

```python
# tests/integration/test_ci_workflow.py

@pytest.mark.slow
async def test_full_ci_workflow(sample_project_with_changes):
    """Test complete CI/CD workflow with diff-aware analysis."""
    # Simulate PR workflow
    project_root = sample_project_with_changes

    # Step 1: Analyze baseline
    baseline_result = await run_analysis(
        project_root=project_root,
        baseline="main",
        fail_on_smell=True,
        severity_threshold="warning"
    )

    # Step 2: Verify only changed files analyzed
    assert baseline_result.files_analyzed < baseline_result.total_files

    # Step 3: Check quality gate
    if baseline_result.smells_found > 0:
        assert baseline_result.exit_code == 1
    else:
        assert baseline_result.exit_code == 0
```

### 6.3 Performance Benchmarks

```python
# tests/benchmarks/test_diff_performance.py

@pytest.mark.benchmark
async def test_diff_analysis_speedup(large_project, benchmark):
    """Benchmark diff-aware vs full analysis."""

    # Modify 5 files out of 100
    for i in range(5):
        file = large_project / f"src/module{i}.py"
        file.write_text("# modified\n" + file.read_text())

    # Benchmark full analysis
    full_time = benchmark(
        lambda: run_analysis(large_project, quick_mode=True)
    )

    # Benchmark diff analysis
    diff_time = benchmark(
        lambda: run_analysis(
            large_project,
            quick_mode=True,
            changed_only=True
        )
    )

    # Assert 10x speedup
    assert diff_time < full_time / 10
```

---

## 7. Risks & Mitigations

### Risk 1: Git Binary Not Available
- **Probability**: Low
- **Impact**: High (feature unusable)
- **Mitigation**:
  - Clear error message with installation instructions
  - Graceful fallback to full analysis
  - Document git as optional dependency

### Risk 2: Complex Git States
- **Probability**: Medium
- **Impact**: Medium (incorrect diff detection)
- **Examples**: Detached HEAD, merge conflicts, submodules
- **Mitigation**:
  - Comprehensive error handling
  - Clear error messages for unsupported states
  - Integration tests with complex git scenarios

### Risk 3: Large Diffs
- **Probability**: Low
- **Impact**: Low (performance still better than full analysis)
- **Example**: 500 changed files in a refactoring PR
- **Mitigation**:
  - Add `--max-files` safety limit
  - Warn if diff exceeds threshold
  - Suggest full analysis for large changes

### Risk 4: False Positives/Negatives
- **Probability**: Low
- **Impact**: High (incorrect analysis results)
- **Mitigation**:
  - Match git's diff behavior exactly
  - Extensive testing with git porcelain format
  - Validate against `git status` output

---

## 8. Documentation Requirements

### 8.1 CLI Help Text

```bash
$ mcp-vector-search analyze --help

📈 Analyze code complexity and quality

Options:
  # ... existing options ...

  🔍 Filters:
    --changed-only              Analyze only git-changed files (uncommitted)
    --baseline TEXT             Compare against baseline branch
                                (e.g., main, master, develop, HEAD~1)
    --language TEXT             Filter by programming language
    --path PATH                 Analyze specific file or directory

Usage Examples:

  # Analyze uncommitted changes
  $ mcp-vector-search analyze --changed-only

  # Analyze changes vs main branch
  $ mcp-vector-search analyze --baseline main

  # CI/CD quality gate
  $ mcp-vector-search analyze --baseline main --fail-on-smell --format sarif

  # Combined filters
  $ mcp-vector-search analyze --baseline main --language python --path src/
```

### 8.2 README Section

```markdown
## Diff-Aware Analysis (v0.18.0+)

Analyze only changed files for faster CI/CD feedback.

### Quick Start

```bash
# Analyze uncommitted changes
mcp-vector-search analyze --changed-only

# Analyze changes vs main branch
mcp-vector-search analyze --baseline main

# CI/CD quality gate
mcp-vector-search analyze --baseline main --fail-on-smell
```

### Use Cases

**Pull Request Checks**
- Analyze only files changed in PR
- Fast feedback on code quality
- Integration with GitHub Code Scanning

**Pre-Commit Hooks**
- Check quality before committing
- Block commits with critical issues
- Incremental improvement workflow

**CI/CD Pipelines**
- 10x faster than full analysis
- Focus on changed code
- SARIF output for issue tracking
```

### 8.3 Troubleshooting Guide

```markdown
## Troubleshooting Diff-Aware Analysis

### Error: "Git not available"

Install git:
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git

# Windows
# Download from https://git-scm.com/
```

### Error: "Not a git repository"

Initialize git:
```bash
git init
git add .
git commit -m "Initial commit"
```

### Error: "Baseline 'main' not found"

Check available branches:
```bash
git branch -a
```

Use existing branch:
```bash
mcp-vector-search analyze --baseline master
```

### No changed files found

Check git status:
```bash
git status
```

If files are staged but not shown, use:
```bash
git reset HEAD  # Unstage files
mcp-vector-search analyze --changed-only
```
```

---

## 9. Estimated Implementation Effort

### Time Breakdown

| Task | Estimated Hours | Priority |
|------|-----------------|----------|
| GitManager class implementation | 4h | High |
| CLI integration | 2h | High |
| Unit tests (GitManager) | 3h | High |
| Integration tests (CLI) | 2h | Medium |
| Error handling & edge cases | 2h | High |
| Documentation | 2h | Medium |
| Performance testing | 1h | Low |
| Code review & refinement | 2h | Medium |
| **Total** | **18h** | - |

### Milestones

- **Day 1-2**: GitManager implementation + unit tests (7h)
- **Day 3**: CLI integration + testing (4h)
- **Day 4**: Error handling + edge cases (2h)
- **Day 5**: Documentation + polish (5h)

**Total Estimated Duration**: 4-5 days (assuming 3-4h/day)

---

## 10. Success Metrics

### Quantitative Metrics

1. **Performance Improvement**
   - Target: 10x speedup for 5% file change rate
   - Measurement: Analyze 100-file project with 5 changed files
   - Success: <5 seconds vs >50 seconds for full analysis

2. **Adoption Rate**
   - Target: 80% of CI/CD runs use diff-aware analysis
   - Measurement: GitHub Actions logs
   - Success: `--baseline` in 80%+ of CI workflow runs

3. **Error Rate**
   - Target: <1% of diff-aware analyses fail with git errors
   - Measurement: Exit code 2 occurrences
   - Success: 99%+ successful git operations

### Qualitative Metrics

1. **User Feedback**
   - Target: Positive feedback on CI/CD speed improvement
   - Measurement: GitHub issues, discussions
   - Success: No complaints about diff detection accuracy

2. **Integration Success**
   - Target: Seamless GitHub Actions integration
   - Measurement: Example workflows in repo
   - Success: Copy-paste workflow examples work without modification

---

## 11. Recommendations

### Priority 1: Implement Core Functionality
- Focus on `--changed-only` first (simpler, no baseline logic)
- Add comprehensive error handling from the start
- Write tests concurrently with implementation

### Priority 2: Optimize for CI/CD
- Ensure SARIF output works with diff-aware analysis
- Document GitHub Actions integration prominently
- Provide example workflows in repo

### Priority 3: User Experience
- Clear, actionable error messages
- Helpful suggestions when git errors occur
- Performance feedback in output ("8 files analyzed, 92 skipped")

### Priority 4: Edge Cases
- Handle detached HEAD gracefully
- Support renamed files correctly
- Deal with merge conflicts appropriately

### Recommended Implementation Order

1. **Implement GitManager** (core functionality)
2. **Add CLI options** (user interface)
3. **Write tests** (ensure correctness)
4. **Add error handling** (robustness)
5. **Document integration** (adoption)

---

## 12. References

### Internal Documents
- [Structural Code Analysis Design Document](./mcp-vector-search-structural-analysis-design.md)
- [Project Roadmap](../projects/structural-code-analysis.md)
- [PR Workflow Guide](../development/pr-workflow-guide.md)

### External References
- [Git Porcelain Format](https://git-scm.com/docs/git-status#_porcelain_format_version_1)
- [SARIF Format Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
- [GitHub Code Scanning](https://docs.github.com/en/code-security/code-scanning)

### Related Issues
- Issue #10: Create `analyze --quick` CLI (blocker)
- Issue #15: SARIF output format (complementary)
- Issue #18: Baseline comparison (depends on #17)

---

## Appendix A: Git Status Porcelain Format

```
Format: XY filename

X = index status (staged)
Y = working tree status (unstaged)

Status codes:
' ' = unmodified
M = modified
A = added
D = deleted
R = renamed
C = copied
U = updated but unmerged

Examples:
 M file.py         # Modified in working tree
M  file.py         # Modified in index (staged)
MM file.py         # Modified in both
A  new.py          # New file staged
?? untracked.py    # Untracked file
R  old.py -> new.py  # Renamed file
```

---

## Appendix B: Example Git Operations

```bash
# Get uncommitted changes (working tree + index)
git status --porcelain

# Get changes vs baseline
git diff --name-only main

# Get changes between two commits
git diff --name-only HEAD~1 HEAD

# Check if ref exists
git rev-parse --verify main

# Get current branch
git rev-parse --abbrev-ref HEAD

# Check if working tree is clean
git diff-index --quiet HEAD --
```

---

**Research Completed**: December 11, 2025
**Next Action**: Begin implementation after Issue #10 is merged
**Estimated Completion**: Phase 2 milestone (v0.18.0) - December 24-30, 2024
