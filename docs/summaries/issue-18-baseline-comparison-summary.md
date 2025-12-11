# Issue #18: Baseline Comparison - Summary

**Status**: Phase 2 - Quality Gates (Blocked by Issue #17)
**Effort**: 1.5 days
**Release**: v0.18.0 (December 24-30, 2024)

## Quick Overview

Issue #18 adds **baseline comparison** capability to track metric changes over time. Developers can save metric snapshots and later compare current code quality against them.

## Key Commands

```bash
# Save current metrics as baseline
mcp-vector-search analyze --save-baseline main-branch

# Compare current code against baseline
mcp-vector-search analyze --compare-baseline main-branch

# List available baselines
mcp-vector-search analyze --list-baselines

# Inspect baseline metadata
mcp-vector-search analyze --inspect-baseline main-branch
```

## Metrics to Store

All metrics collected during analysis:
- Cognitive Complexity (per function, file, project)
- Cyclomatic Complexity
- Nesting Depth
- Parameter Count
- Method Count per Class
- File-level aggregates
- Project-level aggregates

## Storage Format: JSON (Phase 2)

**Location**: `~/.mcp-vector-search/baselines/`

**File structure**:
```
{
  "version": "1.0",
  "baseline_name": "main-branch",
  "created_at": "2025-12-11T15:30:00Z",
  "git_info": { commit, branch, tag },
  "aggregate_metrics": { ... },
  "files": { ... }
}
```

**Size**: ~100-200 KB per typical project (100+ files)
**Format**: Plain JSON (no compression)

## Comparison Output

Example comparison output:
```
Project Changes:
  Cognitive Complexity: 245 → 248 (+3, +1.2%)
  Grade C Functions: 14 → 15 (+1, regression)
  Grade D Functions: 8 → 6 (-2, improvement)

Files with Regressions (3):
  src/core/analyzer.py: CC 18 → 22 (+4)
  src/cli/main.py: CC 15 → 18 (+3)
  src/mcp/server.py: CC 12 → 14 (+2)

Files with Improvements (2):
  src/search/indexer.py: CC 20 → 15 (-5)
  src/utils/helpers.py: CC 11 → 8 (-3)
```

## Acceptance Criteria

- [ ] Save baseline with metadata (timestamp, git info)
- [ ] Load and compare against baseline
- [ ] Show metric deltas (absolute and percentage)
- [ ] Classify changes (regression/improvement/neutral)
- [ ] List and manage baselines
- [ ] Handle missing/incompatible baselines gracefully
- [ ] CI/CD integration with exit codes
- [ ] 80% test coverage

## Recommended Storage

| Aspect | Recommendation |
|--------|-----------------|
| **Phase 2 Format** | JSON files |
| **Storage Location** | `~/.mcp-vector-search/baselines/` |
| **File Size** | ~100-200 KB per baseline |
| **Naming** | Alphanumeric + hyphens (e.g., `main-branch`, `v1.2.0`) |
| **Phase 3 Plan** | Migrate to SQLite (Issue #24) for trend analysis |

## Dependency

**Blocked by**: Issue #17 (Diff-aware analysis)
- Issue #17 implements `--baseline <branch>` (git-based)
- Issue #18 implements `--baseline <name>` (snapshot-based)
- Concepts are similar but distinct
- Issue #17 completes first

**Parallel work possible**: Issues #14, #15, #16

## CI/CD Integration

```bash
# Save baseline after successful PR merge
mcp-vector-search analyze --save-baseline "main-$(date +%Y-%m-%d)"

# Compare PR branch against main baseline
mcp-vector-search analyze --compare-baseline main-branch --fail-on-regression
```

## Implementation Breakdown

1. **BaselineManager** (3 hours)
   - Save baseline to JSON
   - Load baseline from JSON
   - List/delete/inspect baselines
   - Handle metadata (timestamp, git info)

2. **BaselineComparator** (2 hours)
   - Compare metric dictionaries
   - Calculate deltas and percentages
   - Classify changes (regression/improvement)
   - Generate formatted report

3. **CLI Integration** (2 hours)
   - Add baseline flags to analyze command
   - Wire up comparison output
   - Handle error cases

4. **Testing** (3 hours)
   - Unit tests for storage
   - Unit tests for comparison
   - Integration tests
   - Error case tests

## Files to Create/Modify

**New files**:
- `src/mcp_vector_search/analysis/baseline/manager.py`
- `src/mcp_vector_search/analysis/baseline/comparator.py`
- `tests/unit/analysis/baseline/test_manager.py`
- `tests/unit/analysis/baseline/test_comparator.py`

**Modified files**:
- `src/mcp_vector_search/cli/commands/analyze.py`
- `src/mcp_vector_search/analysis/reporters/console.py`

## Future (Phase 3+)

- **Issue #24**: Migrate to SQLite metrics store for better queries
- **Issue #25**: Trend tracking across multiple baselines
- **Auto-baseline**: Automatically save baselines on release/merge
- **Baseline sharing**: Export/import baselines for team collaboration

## Full Documentation

See: `/docs/research/issue-18-baseline-comparison-requirements.md`
