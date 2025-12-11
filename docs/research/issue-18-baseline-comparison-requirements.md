# Issue #18: Baseline Comparison - Research Document

**Research Date**: December 11, 2025
**Researcher**: Research Agent
**Issue**: [#18 - Baseline comparison](https://github.com/bobmatnyc/mcp-vector-search/issues/18)
**Milestone**: v0.18.0 - Quality Gates
**Phase**: 2 of 5 - Quality Gates
**Status**: 📋 Backlog (Ready after Issue #17 completes)

---

## Executive Summary

Issue #18 implements **baseline comparison capability** for the `mcp-vector-search analyze` command, enabling developers and CI/CD pipelines to track metric changes over time by comparing current code quality metrics against a previous known-good state (baseline). This feature unblocks quality gate enforcement and technical debt tracking.

**Key Findings:**
- **Purpose**: Store baseline metrics and compare current analysis against them
- **Dependencies**: Requires Issue #17 (Diff-aware analysis) - blocks on `--baseline` functionality
- **Impact**: Enables CI/CD quality gates and technical debt trend tracking
- **Complexity**: High - requires baseline storage, comparison logic, and historical tracking
- **Timeline**: Part of Phase 2 (v0.18.0), due December 24-30, 2024
- **Estimated effort**: 1.5 days (based on Phase 2 timeline estimates)

---

## 1. Requirements Specification

### 1.1 Functional Requirements

Based on the design document, GitHub issues summary, and existing Phase 2 architecture, Issue #18 must implement:

#### FR-1: Baseline Storage and Persistence
- **Description**: Store metric snapshots for future comparison
- **Command Interface**: `mcp-vector-search analyze --save-baseline <name>`
- **Behavior**:
  - Capture all current metrics for the analyzed files
  - Store with human-readable identifier (e.g., "main-branch", "v1.2.0", "2025-12-11")
  - Persist for future retrieval and comparison
  - Support multiple baselines per project
  - Enable baseline listing and inspection

#### FR-2: Baseline Comparison
- **Description**: Compare current metrics against a stored baseline
- **Command Interface**: `mcp-vector-search analyze --compare-baseline <name>`
- **Behavior**:
  - Load stored baseline metrics
  - Compare current analysis results against baseline
  - Report metric changes (deltas)
  - Highlight regressions (metrics that worsened)
  - Highlight improvements (metrics that improved)
  - Generate diff report

#### FR-3: Automatic Baseline Creation
- **Description**: Create baselines from branch points and releases
- **Examples**:
  ```bash
  # Save baseline after successful PR merge
  mcp-vector-search analyze --save-baseline "main-branch"

  # Compare current branch against main
  mcp-vector-search analyze --compare-baseline "main-branch"

  # Create release baseline
  mcp-vector-search analyze --save-baseline "v1.2.0"

  # List available baselines
  mcp-vector-search analyze --list-baselines
  ```

#### FR-4: CI/CD Integration
- **Description**: Baseline comparison in pull request and continuous integration workflows
- **Behavior**:
  - Automatically compare PR branch against main-branch baseline
  - Exit code reflects comparison results
  - Generate diff report in console or SARIF format
  - Support threshold-based quality gates
  - Integration with GitHub Actions workflow

#### FR-5: Metric Comparison Output
- **Description**: Clear reporting of metric changes and trends
- **Output includes**:
  - Per-file metric changes
  - Per-function metric changes
  - Aggregate project-level changes
  - Change direction (↑ regression, ↓ improvement, → no change)
  - Percentage change where applicable
  - Severity classification

#### FR-6: Historical Tracking
- **Description**: Track metric trends over multiple baselines
- **Behavior**:
  - Support storing multiple timestamped baselines
  - Compare across baseline versions
  - Generate trend reports (improving/degrading metrics)
  - Identify when regressions were introduced
  - Visualize metric trajectory over time

### 1.2 Non-Functional Requirements

#### NFR-1: Storage Performance
- **Requirement**: Baseline storage and retrieval must be fast
- **Target**: <100ms to save baseline, <50ms to load baseline for comparison
- **Measurement**: Profile baseline operations with 500+ files, 1000+ metrics

#### NFR-2: Storage Efficiency
- **Requirement**: Minimal disk space overhead per baseline
- **Target**: <5MB per baseline for typical project (100+ files)
- **Strategy**: Use compressed JSON or SQLite with schema optimization

#### NFR-3: Data Integrity
- **Requirement**: Stored baselines must accurately represent state at capture time
- **Behavior**:
  - Include timestamp of baseline creation
  - Include git commit hash (if in git repo)
  - Include branch name
  - Validate baseline data on load
  - Handle corrupted baseline files gracefully

#### NFR-4: Backward Compatibility
- **Requirement**: Support changing metrics between versions
- **Behavior**:
  - Handle metrics added in later versions
  - Handle metrics renamed or removed
  - Graceful migration of baseline formats
  - Clear error messages for incompatible baselines

#### NFR-5: User Experience
- **Requirement**: Clear feedback on baseline operations
- **Output Examples**:
  ```
  Saved baseline: main-branch (42 files, 156 functions)
  Comparing against baseline: main-branch

  Project Changes:
    Cognitive Complexity: 245 → 248 (+3, +1.2%)
    Max Nesting Depth: 8 → 8 (no change)
    Functions > Grade C: 12 → 14 (+2, regression)

  Files with Regressions (3):
    src/core/analyzer.py: CC 18 → 22 (+4)
    src/cli/main.py: CC 15 → 18 (+3)
    src/mcp/server.py: CC 12 → 14 (+2)

  Files with Improvements (2):
    src/search/indexer.py: CC 20 → 15 (-5)
    src/utils/helpers.py: CC 11 → 8 (-3)
  ```

---

## 2. Storage Format Analysis

### 2.1 Storage Format Options

Based on project architecture and Phase 3 planning (SQLite metrics store in Issue #24), the following formats are viable:

#### Option A: JSON Files (Recommended for Phase 2)
**Pros**:
- Human-readable and inspectable
- No new dependencies (json stdlib)
- Fast for small baselines (typical projects)
- Easy to version control if desired
- Suitable for standalone usage
- Simple to migrate data between versions

**Cons**:
- Less efficient for large projects (1000+ files)
- No query capabilities
- Difficult to track trends across many baselines
- No built-in data validation

**File Structure**:
```json
{
  "version": "1.0",
  "baseline_name": "main-branch",
  "created_at": "2025-12-11T15:30:00Z",
  "git_info": {
    "commit": "0d40b6741f4b78164accdc36e6f776fd18b0684e",
    "branch": "main"
  },
  "project": {
    "path": "/Users/masa/Projects/mcp-vector-search",
    "language_count": 3,
    "file_count": 42
  },
  "aggregate_metrics": {
    "cognitive_complexity": {
      "sum": 245,
      "avg": 4.2,
      "max": 28,
      "functions_grade_a": 8,
      "functions_grade_b": 12,
      "functions_grade_c": 15,
      "functions_grade_d": 6,
      "functions_grade_f": 1
    },
    "cyclomatic_complexity": {
      "sum": 312,
      "avg": 5.3,
      "max": 35
    },
    "max_nesting_depth": {
      "max": 8,
      "avg": 3.2
    },
    "parameter_count": {
      "max": 9,
      "avg": 2.1
    },
    "method_count": {
      "max": 18,
      "avg": 6.5
    }
  },
  "files": {
    "src/mcp_vector_search/core/indexer.py": {
      "path": "src/mcp_vector_search/core/indexer.py",
      "language": "python",
      "file_metrics": {
        "lines_of_code": 412,
        "functions": 8,
        "classes": 2
      },
      "aggregate": {
        "cognitive_complexity": 18,
        "cyclomatic_complexity": 22,
        "max_nesting_depth": 5,
        "max_parameter_count": 6,
        "method_count": 8
      },
      "functions": {
        "index": {
          "start_line": 45,
          "end_line": 112,
          "cognitive_complexity": 18,
          "cyclomatic_complexity": 8,
          "nesting_depth": 5,
          "parameter_count": 4,
          "lines_of_code": 67
        }
      }
    }
  }
}
```

#### Option B: SQLite Database (Recommended for Phase 3+)
**Pros**:
- Efficient for large projects and many baselines
- Query capabilities for trend analysis
- Built-in data validation (schema)
- Suitable for cross-file analysis
- Recommended in design document for Phase 3

**Cons**:
- New dependency (sqlite3, but already in Phase 3 planning)
- Less human-readable
- Requires schema migration handling
- More complex implementation

**Note**: Issue #24 (Phase 3) is planned to implement SQLite metrics store for trend tracking. Baseline comparison (Issue #18) should use JSON in Phase 2 to remain independent, with migration to SQLite in Phase 3.

#### Option C: YAML Files
**Pros**:
- Human-readable
- Fewer dependencies than XML

**Cons**:
- YAML parsing complexity
- No meaningful advantage over JSON
- Not recommended

**Recommendation**: Use **JSON** for Phase 2. Plan migration to SQLite in Phase 3 (Issue #24) when formal metrics store is implemented. This allows Issue #18 to be completed without blocking on Issue #24.

### 2.2 Storage Location

**Primary**: `~/.mcp-vector-search/baselines/`
- Directory structure:
  ```
  ~/.mcp-vector-search/
  ├── baselines/
  │   ├── main-branch.json          (baseline snapshots)
  │   ├── v1.2.0.json
  │   └── pre-refactor.json
  ├── config/
  ├── database/
  │   └── chroma.sqlite3
  └── ...
  ```

**Project-specific** (Optional): `.mcp-vector-search/baselines/` in project root
- Allows baselines to be version-controlled with project
- Better for team collaboration
- Respects `.gitignore` if needed

**Default behavior**: Check project-local baselines first, then fall back to user home baselines.

---

## 3. Detailed Feature Specifications

### 3.1 Save Baseline Command

```bash
mcp-vector-search analyze --save-baseline <name> [options]
```

**Parameters**:
- `<name>`: Baseline identifier (alphanumeric + hyphens, no spaces)
- `--overwrite`: Overwrite existing baseline with same name (default: error)
- `--include-description`: Add custom description to baseline

**Behavior**:
1. Analyze current codebase (standard analyze)
2. Collect all metrics for all files
3. Aggregate metrics at project level
4. Capture metadata (timestamp, git info, project info)
5. Serialize to JSON
6. Save to `~/.mcp-vector-search/baselines/{name}.json`
7. Confirm save with file size and metrics count

**Example Output**:
```
Analyzing codebase...
Collecting metrics for 42 files...
Saving baseline: main-branch
✓ Baseline saved: ~/.mcp-vector-search/baselines/main-branch.json (156 KB)
  Files: 42
  Functions: 156
  Metrics collected: 780
  Git commit: 0d40b6741f4b (main branch)
```

**Error Handling**:
- Baseline already exists → show error, suggest `--overwrite`
- Directory permission denied → show clear error with suggested fix
- Invalid baseline name → show error with naming rules
- Analysis fails → don't save baseline, show analysis errors

### 3.2 Compare Baseline Command

```bash
mcp-vector-search analyze --compare-baseline <name> [options]
```

**Parameters**:
- `<name>`: Baseline name to compare against
- `--show-all`: Show all files, not just changed (default: only changes)
- `--threshold <percentage>`: Only show changes >= percentage (default: 0%)
- `--regression-only`: Only show regressions, not improvements

**Behavior**:
1. Analyze current codebase (standard analyze)
2. Load stored baseline
3. Compare current metrics against baseline
4. Calculate deltas and percentage changes
5. Classify changes as regression/improvement/neutral
6. Generate comparison report
7. Return exit code based on quality gate results

**Comparison Output**:
```
Analyzing current codebase...
Loading baseline: main-branch
Comparing metrics...

=== PROJECT OVERVIEW ===
Current vs Baseline (main-branch from 2025-12-11)

Total Files: 42 (no change)
Total Functions: 156 (no change)

=== AGGREGATE METRICS ===
                        Current  Baseline  Change
Cognitive Complexity:      248      245    +3 (+1.2%) ⚠️
  Grade A (≤5):            8        8      =
  Grade B (≤10):           12       12     =
  Grade C (≤15):           15       14     +1 ⚠️
  Grade D (≤25):           6        8      -2 ✓
  Grade F (>25):           1        3      -2 ✓

Max Nesting Depth:         8        8      =
Avg Parameter Count:       2.1      2.1    =
Max Method Count:          18       18     =

=== FILES WITH REGRESSIONS (3 files) ===
src/core/analyzer.py (CHANGED)
  Cognitive Complexity: 18 → 22 (+4, +22.2%) ⚠️ [was Grade B]
  Max Nesting Depth: 5 → 6 (+1) ⚠️

src/cli/main.py (CHANGED)
  Cognitive Complexity: 15 → 18 (+3, +20.0%) ⚠️ [now Grade D]

src/mcp/server.py (UNCHANGED)
  Cognitive Complexity: 12 → 14 (+2, +16.7%) ⚠️

=== FILES WITH IMPROVEMENTS (2 files) ===
src/search/indexer.py (CHANGED)
  Cognitive Complexity: 20 → 15 (-5, -25.0%) ✓ [improved to Grade C]

src/utils/helpers.py (CHANGED)
  Cognitive Complexity: 11 → 8 (-3, -27.3%) ✓ [improved to Grade B]

=== QUALITY GATE RESULTS ===
Current max cognitive complexity: 28 (threshold: 25)
Status: ⚠️ FAILS - Project has functions exceeding complexity threshold
```

### 3.3 List Baselines Command

```bash
mcp-vector-search analyze --list-baselines
```

**Output**:
```
Available Baselines:

1. main-branch (2025-12-11 15:30 UTC)
   Files: 42 | Functions: 156 | Max CC: 28
   Location: ~/.mcp-vector-search/baselines/main-branch.json

2. v1.2.0 (2025-12-09 10:15 UTC)
   Files: 40 | Functions: 150 | Max CC: 25
   Location: ~/.mcp-vector-search/baselines/v1.2.0.json

3. pre-refactor (2025-12-08 14:45 UTC)
   Files: 45 | Functions: 172 | Max CC: 32
   Location: ~/.mcp-vector-search/baselines/pre-refactor.json
```

### 3.4 Inspect Baseline Command

```bash
mcp-vector-search analyze --inspect-baseline <name>
```

**Output**: Full baseline metadata (git info, timestamp, aggregate metrics)

---

## 4. Integration with Diff-Aware Analysis (Issue #17)

### Combined Command Usage

Issue #18 is dependent on Issue #17 but adds an orthogonal capability:

**Issue #17 Capabilities**:
- `--changed-only`: Analyze uncommitted changes
- `--baseline <branch>`: Analyze changes vs git branch

**Issue #18 Capabilities**:
- `--save-baseline <name>`: Store snapshot of metrics
- `--compare-baseline <name>`: Compare against stored snapshot

**Combined Usage**:
```bash
# Analyze changed files vs main branch, compare against saved baseline
mcp-vector-search analyze --changed-only --compare-baseline main-branch

# Save current analysis as baseline after successful test
mcp-vector-search analyze --save-baseline "post-review"

# Create baseline from git branch analysis
mcp-vector-search analyze --baseline main --save-baseline "main-current"
```

### Distinction

- **Issue #17 (Diff-aware)**: Filters **which files** to analyze (git-based)
- **Issue #18 (Baseline)**: Compares **metrics** from current vs. previous state (snapshot-based)

They complement each other:
- Issue #17 optimizes **scope** (analyze fewer files)
- Issue #18 enables **comparison** (understand changes)

---

## 5. Error Handling and Edge Cases

### 5.1 Missing Baseline

```
Error: Baseline 'unknown-baseline' not found
Available baselines:
  - main-branch (2025-12-11)
  - v1.2.0 (2025-12-09)

Save a baseline with:
  mcp-vector-search analyze --save-baseline main-branch
```

### 5.2 Incompatible Baseline

If baseline was created with different metrics or tool version:
```
Warning: Baseline 'main-branch' uses different metric version
  Baseline version: 1.0 (created with mcp-vector-search v0.16.1)
  Current version: 1.1 (running mcp-vector-search v0.18.0)

Comparing available metrics only:
  Cognitive Complexity ✓
  Cyclomatic Complexity ✓
  Max Nesting Depth ✓
  (Parameter Count unavailable in baseline)

Recommendation: Re-create baseline with current version:
  mcp-vector-search analyze --save-baseline main-branch --overwrite
```

### 5.3 Files Added/Removed

- **New files**: Show in improvement section with baseline value "0 → X"
- **Deleted files**: Show in baseline but not in current, ignore in comparison
- **Renamed files**: Treat as new file and deleted file (no smart matching)

### 5.4 Metric Changes Between Versions

Baseline format includes version number for compatibility checking.

---

## 6. Phase 2 Dependencies and Blocking

### 6.1 Blocking Dependencies

- **Issue #17 (Diff-aware analysis)**: MUST complete before Issue #18
  - Rationale: Issue #18 requires understanding of `--baseline <branch>` concept
  - Issue #17 implements git branch comparison
  - Issue #18 implements metric snapshot comparison
  - They use similar terminology but solve different problems

### 6.2 Non-Blocking Dependencies

- **Issue #13 (Threshold configuration)**: Already complete (shipped in #8)
- **Issue #14 (Code smell detection)**: Can run in parallel with #18
- **Issue #15 (SARIF output)**: Can run in parallel with #18
- **Issue #16 (Exit codes)**: Can run in parallel with #18

---

## 7. Acceptance Criteria

### AC-1: Baseline Storage
- [ ] Can save baseline with human-readable name
- [ ] Baseline file contains complete metric snapshots
- [ ] Baseline includes metadata (timestamp, git info)
- [ ] Multiple baselines can coexist
- [ ] Baseline names are validated (alphanumeric + hyphens)
- [ ] Existing baseline error handling works

### AC-2: Baseline Comparison
- [ ] Can compare current analysis against saved baseline
- [ ] Comparison shows metric deltas (absolute and percentage)
- [ ] Comparison classifies changes (regression/improvement/neutral)
- [ ] Output is clear and actionable
- [ ] Comparison works with different file sets

### AC-3: Baseline Management
- [ ] List baselines command works
- [ ] Inspect baseline command shows metadata
- [ ] Delete baseline functionality exists
- [ ] Baseline location is documented and customizable

### AC-4: CI/CD Integration
- [ ] Baseline comparison works in CI/CD environment
- [ ] Exit codes reflect comparison results
- [ ] Works with GitHub Actions workflow
- [ ] Can compare PR branch against main-branch baseline

### AC-5: Error Handling
- [ ] Missing baseline shows helpful error
- [ ] Incompatible baseline handled gracefully
- [ ] File addition/deletion handled correctly
- [ ] Git unavailability doesn't crash baseline comparison

### AC-6: Testing
- [ ] Unit tests for baseline serialization
- [ ] Unit tests for metric comparison
- [ ] Integration tests with real codebases
- [ ] Error case tests
- [ ] Coverage ≥ 80%

---

## 8. Implementation Strategy

### 8.1 Recommended Approach

**Phase 2 (Issue #18) - Focus on JSON baseline**:
1. Create `BaselineManager` class for storage/retrieval
   - Save baseline to JSON
   - Load baseline from JSON
   - List available baselines
   - Delete baselines

2. Create `BaselineComparator` class for comparison logic
   - Compare metric dictionaries
   - Calculate deltas and percentages
   - Classify changes (regression/improvement)
   - Generate formatted report

3. Add CLI commands
   - `--save-baseline <name>`
   - `--compare-baseline <name>`
   - `--list-baselines`
   - `--inspect-baseline <name>`
   - `--delete-baseline <name>`

4. Add tests
   - Unit tests for JSON serialization
   - Unit tests for comparison logic
   - Integration tests with real codebase

**Phase 3 (Issue #24) - Migrate to SQLite**:
- Refactor storage layer to use SQLite
- Implement trend tracking queries
- Add historical analysis capabilities

### 8.2 Code Organization

**New files**:
- `src/mcp_vector_search/analysis/baseline/manager.py` - BaselineManager class
- `src/mcp_vector_search/analysis/baseline/comparator.py` - BaselineComparator class
- `src/mcp_vector_search/analysis/baseline/__init__.py`
- `tests/unit/analysis/baseline/test_manager.py`
- `tests/unit/analysis/baseline/test_comparator.py`

**Modified files**:
- `src/mcp_vector_search/cli/commands/analyze.py` - Add baseline flags
- `src/mcp_vector_search/analysis/reporters/console.py` - Add comparison output

---

## 9. Timeline and Effort

**Estimated Duration**: 1.5 days (12 hours)

**Breakdown**:
- Baseline storage (BaselineManager): 3 hours
- Comparison logic (BaselineComparator): 2 hours
- CLI integration: 2 hours
- Testing: 3 hours
- Documentation: 2 hours

**Critical Path**:
- Issue #17 (Diff-aware) must complete first
- Issue #18 can run in parallel with #14, #15, #16

**Release Target**: v0.18.0 (December 24-30, 2024)

---

## 10. Future Considerations (Phase 3+)

### 10.1 SQLite Migration (Issue #24)
- Migrate baseline storage to SQLite for better queryability
- Support trend analysis across multiple baselines
- Enable historical metric tracking

### 10.2 Trend Analysis
- Compare metrics across 3+ baselines
- Identify when regressions were introduced
- Visualize metric trajectories

### 10.3 Automated Baseline Creation
- Auto-save baseline after successful CI run
- Auto-save baseline on release tags
- Auto-create "main-branch" baseline from main branch

### 10.4 Baseline Sharing
- Export baseline for team collaboration
- Import baseline from teammates
- Cloud storage support

---

## 11. References

### Design Document
- **Location**: `docs/research/mcp-vector-search-structural-analysis-design.md`
- **Relevant Section**: Tier 4 metrics (cross-file analysis phase)

### Related Issues
- **Issue #17** (Diff-aware analysis): `--baseline <branch>` git comparison
- **Issue #14** (Code smell detection): Uses thresholds for quality gates
- **Issue #24** (SQLite metrics store): Phase 3 storage layer

### GitHub Issues
- **Epic #12** (Quality Gates): `https://github.com/bobmatnyc/mcp-vector-search/issues/12`
- **Issue #18**: `https://github.com/bobmatnyc/mcp-vector-search/issues/18`

### Project Milestones
- **v0.18.0 - Quality Gates**: https://github.com/bobmatnyc/mcp-vector-search/milestones
- **Phase 2 Timeline**: December 24-30, 2024

---

## Appendix: Example Baseline JSON Schema

```json
{
  "version": "1.0",
  "baseline_name": "main-branch",
  "created_at": "2025-12-11T15:30:00Z",
  "created_by": "user@example.com",
  "description": "Baseline from main branch after PR #43 merge",
  "tool_version": "v0.18.0",
  "git_info": {
    "commit": "0d40b6741f4b78164accdc36e6f776fd18b0684e",
    "branch": "main",
    "remote": "origin",
    "tag": null
  },
  "project": {
    "path": "/Users/masa/Projects/mcp-vector-search",
    "name": "mcp-vector-search",
    "language_count": 3,
    "file_count": 42,
    "function_count": 156,
    "class_count": 8
  },
  "aggregate_metrics": {
    "cognitive_complexity": {
      "sum": 245,
      "avg": 1.57,
      "min": 0,
      "max": 28,
      "median": 1,
      "grade_distribution": {
        "A": 8,
        "B": 12,
        "C": 15,
        "D": 6,
        "F": 1
      }
    },
    "cyclomatic_complexity": {
      "sum": 312,
      "avg": 2.0,
      "min": 1,
      "max": 35,
      "median": 1
    },
    "nesting_depth": {
      "max": 8,
      "avg": 3.2
    },
    "parameter_count": {
      "max": 9,
      "avg": 2.1
    },
    "method_count": {
      "max": 18,
      "avg": 6.5
    }
  },
  "files": {
    "src/mcp_vector_search/core/indexer.py": {
      "path": "src/mcp_vector_search/core/indexer.py",
      "language": "python",
      "lines_of_code": 412,
      "functions": 8,
      "classes": 2,
      "metrics": {
        "cognitive_complexity": 18,
        "cyclomatic_complexity": 22,
        "nesting_depth": 5,
        "parameter_count": 6,
        "method_count": 8
      },
      "functions": {
        "index": {
          "start_line": 45,
          "end_line": 112,
          "lines_of_code": 67,
          "metrics": {
            "cognitive_complexity": 18,
            "cyclomatic_complexity": 8,
            "nesting_depth": 5,
            "parameter_count": 4
          }
        }
      }
    }
  }
}
```

---

**Research completed**: 2025-12-11
**Report generated by**: Research Agent
**Next step**: Begin Issue #17 (Diff-aware analysis) implementation
**Follow-up**: After Issue #17 complete, start Issue #18 (Baseline comparison)

