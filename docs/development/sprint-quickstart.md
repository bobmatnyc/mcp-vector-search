# Sprint Quickstart Guide

> Get started immediately with the right issues

## START HERE: Issue #2

**Title**: Create metric dataclasses
**Branch**: `feature/2-metric-dataclasses`
**Estimate**: 3 days
**Dependencies**: None
**Blocks**: 13 downstream issues

### Why Start Here?
- Only issue with zero dependencies
- On the critical path (#2 → #8 → #10 → #14 → #35 → #37)
- Blocks all collectors and integration work
- Simple, well-defined scope
- Foundation for entire project

### What to Build
Create Pydantic dataclasses for all metrics:

```python
# src/mcp_vector_search/analysis/metrics.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ComplexityMetrics(BaseModel):
    """Tier 1: Complexity metrics"""
    cognitive_complexity: int = Field(ge=0)
    cyclomatic_complexity: int = Field(ge=0)
    nesting_depth: int = Field(ge=0)

class SizeMetrics(BaseModel):
    """Tier 1: Size metrics"""
    parameter_count: int = Field(ge=0)
    method_count: int = Field(ge=0)
    lines_of_code: int = Field(ge=0)

class CouplingMetrics(BaseModel):
    """Tier 2: Coupling metrics"""
    efferent_coupling: int = Field(ge=0)  # Dependencies out
    afferent_coupling: int = Field(ge=0)  # Dependencies in
    instability: float = Field(ge=0.0, le=1.0)

class CohesionMetrics(BaseModel):
    """Tier 3: Cohesion metrics"""
    lcom4: int = Field(ge=1)  # Lack of Cohesion

class HalsteadMetrics(BaseModel):
    """Tier 3: Halstead complexity"""
    volume: float = Field(ge=0.0)
    difficulty: float = Field(ge=0.0)
    effort: float = Field(ge=0.0)

class FunctionMetrics(BaseModel):
    """Metrics for a single function"""
    name: str
    start_line: int
    end_line: int
    complexity: ComplexityMetrics
    size: SizeMetrics

class FileMetrics(BaseModel):
    """Metrics for a single file"""
    file_path: str
    language: str
    analyzed_at: datetime
    functions: list[FunctionMetrics]
    size: SizeMetrics
    coupling: Optional[CouplingMetrics] = None
    cohesion: Optional[CohesionMetrics] = None
```

### Acceptance Criteria (8 total)
- [ ] Dataclasses for all Tier 1-3 metrics
- [ ] Pydantic validation (non-negative integers, 0-1 floats)
- [ ] JSON serialization (`model_dump_json()`)
- [ ] Proper type hints (pass `mypy --strict`)
- [ ] Unit tests for all dataclasses (80%+ coverage)
- [ ] Test validation edge cases (negative values, invalid types)
- [ ] Documentation docstrings
- [ ] CHANGELOG.md entry

### Success Criteria
```bash
# All tests pass
uv run pytest tests/unit/analysis/test_metrics.py

# Type checking passes
uv run mypy src/mcp_vector_search/analysis/metrics.py

# Code coverage ≥80%
uv run pytest --cov=src/mcp_vector_search/analysis
```

---

## Next Up: Sprint 1 (Dec 10-23)

After #2, immediately start:

### Issue #13: Threshold Configuration
**Branch**: `feature/13-threshold-config`
**Can start**: In parallel with #2 (no dependencies)
**Estimate**: 2 days

```yaml
# .mcp-vector-search/thresholds.yaml

presets:
  strict:
    complexity:
      cognitive_max: 15
      cyclomatic_max: 10
      nesting_max: 3
    size:
      parameters_max: 4
      methods_max: 10
      lines_max: 100

  recommended:
    complexity:
      cognitive_max: 25
      cyclomatic_max: 15
      nesting_max: 4
    size:
      parameters_max: 6
      methods_max: 15
      lines_max: 200
```

---

## Sprint 2: Collectors (After #2 completes)

All 6 can be worked in parallel:

| Priority | Issue | Estimate | Complexity |
|----------|-------|----------|------------|
| 1 | #3 - Cognitive Complexity | 3d | High (complex algorithm) |
| 2 | #4 - Cyclomatic Complexity | 2d | Medium |
| 3 | #9 - ChromaDB schema | 2d | Medium |
| 4 | #5 - Nesting Depth | 1d | Low |
| 5 | #6 - Parameter Count | 1d | Low |
| 6 | #7 - Method Count | 1d | Low |

**Parallelization Strategy**:
- Developer 1: #3 (cognitive - hardest)
- Developer 2: #4, #9 (medium complexity)
- Developer 3: #5, #6, #7 (simple collectors)

---

## Critical Path Roadmap

**Week 1-2**: Foundation + Collectors
```
[#2] ──┬──> [#3, #4, #5, #6, #7] ──> [#8]
       └──> [#9] ─────────────────────┘
```

**Week 2**: Integration
```
[#8] + [#9] ──> [#10] ──> [#11]
                           └──> Ship v0.17.0
```

**Week 3**: Quality Gates
```
[#8] + [#13] ──> [#14] ──> [#15, #16]
                           └──> Ship v0.18.0
```

**Week 4**: Cross-File
```
[#2] + [#8] ──> [#20-26]
                └──> Ship v0.19.0
```

**Week 5-6**: Visualization
```
[#10] + [#14] + [#24] ──> [#28-33]
                          └──> Ship v0.20.0
```

**Week 7-8**: Search Integration
```
[#10] + [#14] ──> [#35] ──> [#37]
                            └──> Ship v0.21.0
                                 PROJECT COMPLETE
```

---

## Daily Workflow

### Morning
1. Check critical path status (#2 → #8 → #10 → #14 → #35 → #37)
2. Review blockers from previous day
3. Pull latest from `main`
4. Create/switch to feature branch

### During Development
1. Write tests first (TDD)
2. Implement feature
3. Run `make pre-publish` before committing
4. Update CHANGELOG.md

### Before Commit
```bash
# Quality gate (MUST pass)
make pre-publish

# This runs:
# - black (formatting)
# - ruff (linting)
# - mypy (type checking)
# - pytest (tests)
# - coverage check (≥80%)
```

### End of Day
1. Commit work (even if WIP)
2. Push to remote branch
3. Update sprint board status
4. Post standup update (if applicable)

---

## Branch Naming

```bash
# Feature branches
git checkout -b feature/2-metric-dataclasses
git checkout -b feature/13-threshold-config

# Format
feature/<issue-number>-<short-description>
```

---

## Commit Message Format

```bash
# Convention: <type>(<scope>): <description>

git commit -m "feat(metrics): add core metric dataclasses"
git commit -m "test(metrics): add validation tests for ComplexityMetrics"
git commit -m "docs(metrics): add docstrings to metric classes"
git commit -m "fix(metrics): handle negative complexity values"
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

---

## PR Checklist

Before creating PR:

- [ ] All tests pass (`uv run pytest`)
- [ ] Coverage ≥80% (`uv run pytest --cov`)
- [ ] Type checking passes (`uv run mypy src/`)
- [ ] Code formatted (`uv run black .`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] CHANGELOG.md updated
- [ ] Documentation updated
- [ ] No files in project root

---

## Milestone Targets

| Phase | Version | Target Date | Issues | Critical Path |
|-------|---------|-------------|--------|---------------|
| 1 | v0.17.0 | Dec 23, 2024 | #1-11 | #2, #8, #10 |
| 2 | v0.18.0 | Dec 30, 2024 | #12-18 | #14 |
| 3 | v0.19.0 | Jan 6, 2025 | #19-26 | None |
| 4 | v0.20.0 | Jan 13, 2025 | #27-33 | None |
| 5 | v0.21.0 | Feb 3, 2025 | #34-38 | #35, #37 |

**Total Project Duration**: 8 weeks (56 days)
**Critical Path Duration**: 23 days
**Buffer**: 33 days (58% slack)

---

## Common Questions

### Q: Can I work on #3 before #2 is done?
**A**: No. #2 defines the dataclasses that #3 uses. Wait for #2 to merge.

### Q: Can #5, #6, #7 be done in parallel?
**A**: Yes! They're independent collectors. Just all need #2 first.

### Q: What if I find a bug in #2 while working on #8?
**A**: Fix it in a new branch `fix/2-metric-validation`, merge to main, then rebase #8.

### Q: When should I update the sprint board?
**A**: Daily. Mark issues as 🔧 In Progress when starting, 👀 In Review when PR submitted, ✅ Done when merged.

### Q: How do I know if I'm blocking the critical path?
**A**: If you're working on #2, #8, #10, #14, #35, or #37 - you're on the critical path. Prioritize these.

---

## Emergency Contacts

**Project Lead**: @bobmatnyc
**GitHub Project**: https://github.com/users/bobmatnyc/projects/13
**Issues**: https://github.com/bobmatnyc/mcp-vector-search/issues

**Docs**:
- [Sprint Plan](./sprint-plan.md) - Full sprint details
- [Sprint Board](./sprint-board.md) - Visual tracking
- [Dependency Graph](./dependency-graph.txt) - Issue dependencies
- [Project Overview](../projects/structural-code-analysis.md) - High-level view

---

**Ready to start?** Create branch `feature/2-metric-dataclasses` and begin!

```bash
git checkout main
git pull
git checkout -b feature/2-metric-dataclasses
code src/mcp_vector_search/analysis/metrics.py
```
