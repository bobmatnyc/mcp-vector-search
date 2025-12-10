# Sprint Board - Structural Code Analysis

> Visual sprint tracking board

## Sprint Status Legend

```
📋 Backlog     - Not started, dependencies not met
🎯 Ready       - Ready to start, all dependencies met
🔧 In Progress - Actively being developed
👀 In Review   - PR submitted, awaiting review
✅ Done        - PR merged, issue closed
```

---

## Sprint 1: Foundation (Dec 10-23, 2024)

**Goal**: Core dataclasses and configuration infrastructure
**Status**: 🎯 Ready to Start

| Status | # | Issue | Assignee | Est | Blocks |
|--------|---|-------|----------|-----|--------|
| 🎯 | #2 | Core metric dataclasses | TBD | 3d | 13 issues |
| 🎯 | #13 | Threshold configuration system | TBD | 2d | #14 |

**Sprint Goal Met**: When both issues are ✅ Done
**Unblocks**: Sprint 2 (all 6 issues)

---

## Sprint 2: Tier 1 Collectors (Dec 10-23, 2024)

**Goal**: Implement all basic metric collectors
**Status**: 📋 Waiting for #2

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #3 | Cognitive Complexity Collector | TBD | 3d | #2 |
| 📋 | #4 | Cyclomatic Complexity Collector | TBD | 2d | #2 |
| 📋 | #5 | Nesting Depth Collector | TBD | 1d | #2 |
| 📋 | #6 | Parameter Count Collector | TBD | 1d | #2 |
| 📋 | #7 | Method Count Collector | TBD | 1d | #2 |
| 📋 | #9 | Extend ChromaDB metadata schema | TBD | 2d | #2 |

**Parallelization**: All 6 can be worked on simultaneously after #2
**Sprint Goal Met**: When all 6 issues are ✅ Done
**Unblocks**: Sprint 3 (#8, #10)

---

## Sprint 3: Integration & CLI (Dec 16-23, 2024)

**Goal**: Ship Phase 1 (v0.17.0) - Working analysis CLI
**Status**: 📋 Waiting for Sprint 2

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #8 | Integrate collectors with indexer | TBD | 5d | #2-7 |
| 📋 | #10 | Create `analyze --quick` CLI | TBD | 4d | #8, #9 |
| 📋 | #11 | Console reporter | TBD | 2d | #10 |

**Milestone**: v0.17.0 (Phase 1 Complete)
**Critical Path**: #8 → #10 ⚠️
**Sprint Goal Met**: All 3 ✅ Done + v0.17.0 tagged
**Unblocks**: Sprint 4 (Quality Gates)

---

## Sprint 4: Quality Gates (Dec 24-30, 2024)

**Goal**: Ship Phase 2 (v0.18.0) - Code smell detection & SARIF
**Status**: 📋 Waiting for Sprint 3

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #14 | Code smell detection | TBD | 5d | #8, #13 |
| 📋 | #15 | SARIF output format | TBD | 3d | #10, #14 |
| 📋 | #16 | `--fail-on-smell` exit codes | TBD | 1d | #14, #15 |
| 📋 | #17 | Diff-aware analysis | TBD | 3d | #10 |
| 📋 | #18 | Baseline comparison | TBD | 2d | #17 |

**Milestone**: v0.18.0 (Phase 2 Complete)
**Critical Path**: #14 ⚠️
**Sprint Goal Met**: All 5 ✅ Done + v0.18.0 tagged
**Unblocks**: Sprint 5, 6, 7

---

## Sprint 5: Cross-File Analysis (Dec 31, 2024 - Jan 6, 2025)

**Goal**: Ship Phase 3 (v0.19.0) - Coupling & dependency analysis
**Status**: 📋 Waiting for Sprint 3

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #20 | Efferent Coupling Collector | TBD | 4d | #2, #8 |
| 📋 | #21 | Afferent Coupling Collector | TBD | 3d | #20 |
| 📋 | #22 | Instability Index | TBD | 1d | #20, #21 |
| 📋 | #23 | Circular dependency detection | TBD | 3d | #20 |
| 📋 | #24 | SQLite metrics store | TBD | 3d | #2 |
| 📋 | #25 | Trend tracking | TBD | 2d | #24 |
| 📋 | #26 | LCOM4 cohesion metric | TBD | 4d | #2, #8 |

**Milestone**: v0.19.0 (Phase 3 Complete)
**Parallelization**: Two chains (#20-23 and #24-26)
**Sprint Goal Met**: All 7 ✅ Done + v0.19.0 tagged
**Unblocks**: Sprint 6

---

## Sprint 6: Visualization Export (Jan 7-13, 2025)

**Goal**: Ship Phase 4 (v0.20.0) - JSON/HTML reports & tech debt
**Status**: 📋 Waiting for Sprints 3, 4, 5

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #28 | JSON export schema | TBD | 1d | #2 |
| 📋 | #29 | JSON exporter | TBD | 3d | #28, #10 |
| 📋 | #30 | HTML standalone report | TBD | 5d | #29 |
| 📋 | #31 | Halstead metrics collector | TBD | 3d | #2, #8 |
| 📋 | #32 | Technical debt estimation | TBD | 3d | #14, #24 |
| 📋 | #33 | `status --metrics` command | TBD | 2d | #10, #24 |

**Milestone**: v0.20.0 (Phase 4 Complete)
**Parallelization**: #28-30 (export) and #31-33 (metrics)
**Sprint Goal Met**: All 6 ✅ Done + v0.20.0 tagged
**Unblocks**: Sprint 7

---

## Sprint 7: Search Integration (Jan 20 - Feb 3, 2025)

**Goal**: Ship Phase 5 (v0.21.0) - Quality-aware search & MCP
**Status**: 📋 Waiting for Sprints 3, 4, 6

| Status | # | Issue | Assignee | Est | Depends On |
|--------|---|-------|----------|-----|------------|
| 📋 | #35 | Quality filters for search | TBD | 3d | #10, #14 |
| 📋 | #36 | Quality-aware ranking | TBD | 4d | #35 |
| 📋 | #37 | Expose as MCP tools | TBD | 3d | #10, #35 |
| 📋 | #38 | LLM interpretation of analysis | TBD | 5d | #10, #14, #29, #37 |

**Milestone**: v0.21.0 (Phase 5 Complete - PROJECT COMPLETE)
**Critical Path**: #35 → #37 ⚠️ (END OF CRITICAL PATH)
**Sprint Goal Met**: All 4 ✅ Done + v0.21.0 tagged
**PROJECT COMPLETE**: All 5 phases shipped

---

## Sprint 8: EPIC Tracking (Continuous)

**Goal**: Project management and documentation
**Status**: 🔧 Ongoing

| Status | # | Issue | Assignee | Close When |
|--------|---|-------|----------|------------|
| 🎯 | #1 | [EPIC] Core Metrics | PM | Phase 1 complete |
| 📋 | #12 | [EPIC] Quality Gates | PM | Phase 2 complete |
| 📋 | #19 | [EPIC] Cross-File Analysis | PM | Phase 3 complete |
| 📋 | #27 | [EPIC] Visualization Export | PM | Phase 4 complete |
| 📋 | #34 | [EPIC] Search Integration | PM | Phase 5 complete |
| 🔧 | #39 | Project documentation updates | PM | Continuous |

**Note**: EPIC issues are tracking-only, don't block development

---

## Critical Path Tracking

**Path**: #2 → #8 → #10 → #14 → #35 → #37
**Total**: 23 days estimated

| Sprint | Issue | Status | Est | Start | End | Actual |
|--------|-------|--------|-----|-------|-----|--------|
| 1 | #2 | 🎯 | 3d | TBD | TBD | - |
| 3 | #8 | 📋 | 5d | TBD | TBD | - |
| 3 | #10 | 📋 | 4d | TBD | TBD | - |
| 4 | #14 | 📋 | 5d | TBD | TBD | - |
| 7 | #35 | 📋 | 3d | TBD | TBD | - |
| 7 | #37 | 📋 | 3d | TBD | TBD | - |

**Progress**: 0/23 days complete (0%)

---

## Velocity Tracking

| Sprint | Planned | Completed | Points/Day | Status |
|--------|---------|-----------|------------|--------|
| 1 | 12 pts | - | - | Not started |
| 2 | 18 pts | - | - | Not started |
| 3 | 18 pts | - | - | Not started |
| 4 | 22 pts | - | - | Not started |
| 5 | 28 pts | - | - | Not started |
| 6 | 20 pts | - | - | Not started |
| 7 | 20 pts | - | - | Not started |

**Average Planned Velocity**: 19.7 points/sprint

---

## Milestone Checklist

### Phase 1: v0.17.0 (Sprint 3)
- [ ] #1-11 all closed
- [ ] Tests passing (80%+ coverage)
- [ ] Metrics match SonarQube validation
- [ ] <10ms overhead per 1000 LOC
- [ ] User guide: "Quick Analysis" section
- [ ] CHANGELOG.md: v0.17.0 entry
- [ ] Git tag: v0.17.0
- [ ] PyPI release published
- [ ] GitHub release notes

### Phase 2: v0.18.0 (Sprint 4)
- [ ] #12-18 all closed
- [ ] SARIF output validates against schema
- [ ] Quality gates integration tested
- [ ] Diff analysis working on PR workflow
- [ ] User guide: "Quality Gates" section
- [ ] CHANGELOG.md: v0.18.0 entry
- [ ] Git tag: v0.18.0
- [ ] PyPI release published
- [ ] GitHub release notes

### Phase 3: v0.19.0 (Sprint 5)
- [ ] #19-26 all closed
- [ ] SQLite metrics store functional
- [ ] Dependency graph visualization
- [ ] Historical trend tracking working
- [ ] User guide: "Cross-File Analysis" section
- [ ] CHANGELOG.md: v0.19.0 entry
- [ ] Git tag: v0.19.0
- [ ] PyPI release published
- [ ] GitHub release notes

### Phase 4: v0.20.0 (Sprint 6)
- [ ] #27-33 all closed
- [ ] HTML report renders correctly
- [ ] JSON export schema validated
- [ ] Tech debt estimation accurate
- [ ] User guide: "Visualization" section
- [ ] CHANGELOG.md: v0.20.0 entry
- [ ] Git tag: v0.20.0
- [ ] PyPI release published
- [ ] GitHub release notes

### Phase 5: v0.21.0 (Sprint 7)
- [ ] #34-38 all closed
- [ ] MCP tools available in Claude Desktop
- [ ] Quality-aware search functional
- [ ] LLM interpretation working
- [ ] User guide: "Search Integration" section
- [ ] CHANGELOG.md: v0.21.0 entry
- [ ] Git tag: v0.21.0
- [ ] PyPI release published
- [ ] GitHub release notes
- [ ] **PROJECT COMPLETE**

---

## Daily Standup Template

**Date**: ______________________

**Yesterday**:
- Issues completed: _______
- Issues in progress: _______
- Blockers: _______

**Today**:
- Plan to complete: _______
- Plan to start: _______
- Need help with: _______

**Blockers**:
- _______________________

**Critical Path Status**:
- Current critical path issue: #___
- On track: ☐ Yes  ☐ No  ☐ At risk
- Days ahead/behind: _____

---

## Burndown Chart (Manual Tracking)

### Sprint 1 (12 points)
```
Day:  1  2  3  4  5  6  7  8  9 10
Pts: 12 __ __ __ __ __ __ __ __  0
```

### Sprint 2 (18 points)
```
Day:  1  2  3  4  5  6  7  8  9 10
Pts: 18 __ __ __ __ __ __ __ __  0
```

### Sprint 3 (18 points)
```
Day:  1  2  3  4  5  6  7  8  9 10
Pts: 18 __ __ __ __ __ __ __ __  0
```

(Continue for each sprint)

---

## Sprint Retrospective Template

**Sprint**: _____ (Dates: __________)

**What Went Well**:
- ________________________
- ________________________

**What Could Be Improved**:
- ________________________
- ________________________

**Action Items**:
- [ ] ________________________
- [ ] ________________________

**Velocity**:
- Planned: ____ points
- Completed: ____ points
- Carryover: ____ points
- Velocity %: ____

**Critical Path**:
- On track: ☐ Yes  ☐ No
- Slippage: ____ days

---

## Issue Status Quick Update

**Instructions**: Update issue status as work progresses

### Phase 1 (Sprint 1-3)
- [ ] #1 - [EPIC] Core Metrics
- [ ] #2 - Core metric dataclasses ← **START HERE**
- [ ] #3 - Cognitive Complexity
- [ ] #4 - Cyclomatic Complexity
- [ ] #5 - Nesting Depth
- [ ] #6 - Parameter Count
- [ ] #7 - Method Count
- [ ] #8 - Integrate collectors ← **CRITICAL**
- [ ] #9 - ChromaDB metadata
- [ ] #10 - `analyze --quick` CLI ← **CRITICAL**
- [ ] #11 - Console reporter

### Phase 2 (Sprint 4)
- [ ] #12 - [EPIC] Quality Gates
- [ ] #13 - Threshold configuration
- [ ] #14 - Code smell detection ← **CRITICAL**
- [ ] #15 - SARIF output
- [ ] #16 - `--fail-on-smell`
- [ ] #17 - Diff-aware analysis
- [ ] #18 - Baseline comparison

### Phase 3 (Sprint 5)
- [ ] #19 - [EPIC] Cross-File Analysis
- [ ] #20 - Efferent coupling
- [ ] #21 - Afferent coupling
- [ ] #22 - Instability index
- [ ] #23 - Circular dependencies
- [ ] #24 - SQLite metrics store
- [ ] #25 - Trend tracking
- [ ] #26 - LCOM4 cohesion

### Phase 4 (Sprint 6)
- [ ] #27 - [EPIC] Visualization Export
- [ ] #28 - JSON export schema
- [ ] #29 - JSON exporter
- [ ] #30 - HTML standalone report
- [ ] #31 - Halstead metrics
- [ ] #32 - Tech debt estimation
- [ ] #33 - `status --metrics`

### Phase 5 (Sprint 7)
- [ ] #34 - [EPIC] Search Integration
- [ ] #35 - Quality filters ← **CRITICAL**
- [ ] #36 - Quality-aware ranking
- [ ] #37 - MCP tools ← **CRITICAL**
- [ ] #38 - LLM interpretation

### Continuous
- [ ] #39 - Documentation updates

---

**Last Updated**: December 10, 2024
**Next Update**: After Sprint 1 completion
