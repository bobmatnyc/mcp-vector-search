# Sprint Plan Summary - Structural Code Analysis

> Quick reference for sprint planning meetings

## At a Glance

**Total Duration**: 8 weeks (7 development sprints)
**Total Issues**: 39 (34 development + 5 tracking EPICs)
**Critical Path**: 6 issues (#2 → #8 → #10 → #14 → #35 → #37)
**Minimum Duration**: 23 days (critical path only)
**Buffer**: 12 days (52% slack in schedule)

---

## Sprint Overview

```
Sprint 1 (Dec 10-23)  : Foundation          [2 issues,   12 pts]  ████░░░░░░
Sprint 2 (Dec 10-23)  : Tier 1 Collectors  [6 issues,   18 pts]  ██████░░░░
Sprint 3 (Dec 16-23)  : Integration & CLI  [3 issues,   18 pts]  ██████░░░░
Sprint 4 (Dec 24-30)  : Quality Gates      [5 issues,   22 pts]  ███████░░░
Sprint 5 (Dec 31-Jan 6): Cross-File        [7 issues,   28 pts]  █████████░
Sprint 6 (Jan 7-13)    : Visualization     [6 issues,   20 pts]  ███████░░░
Sprint 7 (Jan 20-Feb 3): Search Integration[4 issues,   20 pts]  ███████░░░
Sprint 8 (Ongoing)     : EPIC Tracking     [5 issues,  N/A pts]  Continuous
```

**Heaviest Sprint**: Sprint 5 (28 complexity points)
**Lightest Sprint**: Sprint 1 (12 complexity points)

---

## Critical Path Visualization

```
Week 1     Week 2     Week 3     Week 4     Week 5-8
  |          |          |          |          |
 [#2]───────[#8]──────[#10]──────[#14]──────[#35]─────[#37]
  3d         5d        4d         5d         3d        3d
```

**Total Critical Path Duration**: 23 days

**Slack Time**: 33 working days (8 weeks) - 23 critical path days = 10 days buffer

---

## Sprint Details

### Sprint 1: Foundation (Dec 10-23)
**Dependencies**: None
**Deliverables**: Core dataclasses, threshold configuration

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 2 | Core metric dataclasses | Medium | 13 issues |
| 13 | Threshold configuration | Medium | #14 |

**Critical Path**: #2 ✓
**Start Immediately**: Yes

---

### Sprint 2: Tier 1 Collectors (Dec 10-23)
**Dependencies**: #2
**Deliverables**: All metric collectors, ChromaDB schema

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 3 | Cognitive Complexity Collector | High | #8 |
| 4 | Cyclomatic Complexity Collector | Medium | #8 |
| 5 | Nesting Depth Collector | Low | #8 |
| 6 | Parameter Count Collector | Low | #8 |
| 7 | Method Count Collector | Low | #8 |
| 9 | Extend ChromaDB metadata | Medium | #10 |

**Parallelization**: All 6 issues can be worked on simultaneously after #2
**Critical Path**: #3, #4, #5, #6, #7 → #8 ✓

---

### Sprint 3: Integration & CLI (Dec 16-23)
**Dependencies**: #2-7, #9
**Deliverables**: Working CLI, console reporter
**Milestone**: v0.17.0 (Phase 1 Complete)

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 8 | Integrate collectors | High | 6 issues |
| 10 | `analyze --quick` CLI | Medium | 7 issues |
| 11 | Console reporter | Low | None |

**Critical Path**: #8 → #10 ✓✓
**Milestone Achievement**: Phase 1 complete

---

### Sprint 4: Quality Gates (Dec 24-30)
**Dependencies**: #8, #10, #13
**Deliverables**: Code smell detection, SARIF output, diff analysis
**Milestone**: v0.18.0 (Phase 2 Complete)

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 14 | Code smell detection | High | 4 issues |
| 15 | SARIF output format | Medium | #16 |
| 16 | `--fail-on-smell` exit codes | Low | None |
| 17 | Diff-aware analysis | Medium | #18 |
| 18 | Baseline comparison | Medium | None |

**Critical Path**: #14 ✓
**Milestone Achievement**: Phase 2 complete

---

### Sprint 5: Cross-File Analysis (Dec 31-Jan 6)
**Dependencies**: #2, #8
**Deliverables**: Coupling metrics, dependency graph, SQLite store
**Milestone**: v0.19.0 (Phase 3 Complete)

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 20 | Efferent coupling collector | High | 3 issues |
| 21 | Afferent coupling collector | Medium | #22 |
| 22 | Instability index | Low | None |
| 23 | Circular dependency detection | Medium | None |
| 24 | SQLite metrics store | Medium | 3 issues |
| 25 | Trend tracking | Medium | None |
| 26 | LCOM4 cohesion metric | High | None |

**Parallelization**: #20-23 and #24-26 are independent chains
**Milestone Achievement**: Phase 3 complete

---

### Sprint 6: Visualization Export (Jan 7-13)
**Dependencies**: #2, #8, #10, #14, #24
**Deliverables**: JSON/HTML reports, Halstead metrics, tech debt
**Milestone**: v0.20.0 (Phase 4 Complete)

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 28 | JSON export schema | Low | #29 |
| 29 | JSON exporter | Medium | #30 |
| 30 | HTML standalone report | High | None |
| 31 | Halstead metrics | Medium | None |
| 32 | Tech debt estimation | Medium | None |
| 33 | `status --metrics` command | Low | None |

**Parallelization**: #28-30 (export chain) and #31-33 (independent)
**Milestone Achievement**: Phase 4 complete

---

### Sprint 7: Search Integration (Jan 20-Feb 3)
**Dependencies**: #10, #14
**Deliverables**: Quality-aware search, MCP tools, LLM interpretation
**Milestone**: v0.21.0 (Phase 5 Complete - PROJECT COMPLETE)

| # | Issue | Complexity | Blocks |
|---|-------|------------|--------|
| 35 | Quality filters for search | Medium | 2 issues |
| 36 | Quality-aware ranking | High | None |
| 37 | Expose as MCP tools | Medium | None |
| 38 | LLM interpretation | High | None |

**Critical Path**: #35 → #37 ✓✓ (CRITICAL PATH END)
**Milestone Achievement**: Phase 5 complete, PROJECT COMPLETE

---

### Sprint 8: EPIC Tracking (Ongoing)
**Dependencies**: Various
**Deliverables**: Project management and documentation

| # | Issue | Type | Purpose |
|---|-------|------|---------|
| 1 | [EPIC] Core Metrics | Tracking | Phase 1 |
| 12 | [EPIC] Quality Gates | Tracking | Phase 2 |
| 19 | [EPIC] Cross-File Analysis | Tracking | Phase 3 |
| 27 | [EPIC] Visualization Export | Tracking | Phase 4 |
| 34 | [EPIC] Search Integration | Tracking | Phase 5 |
| 39 | Documentation updates | Continuous | All phases |

**Note**: EPIC issues don't block development, close when phase completes

---

## Dependency Quick Reference

### No Dependencies (Start Anytime)
- #2 ← **START HERE**
- #13

### Depends on #2 Only
- #3, #4, #5, #6, #7, #9, #24, #28

### Depends on #8 (after Sprint 3)
- #10 (also needs #9)
- #14 (also needs #13)
- #20, #26, #31

### Depends on #10 (after Sprint 3)
- #11, #15, #17, #29, #33, #35, #38

### Depends on #14 (after Sprint 4)
- #15, #16, #32, #35, #38

---

## Blocking Analysis

### Most Critical Issues (Block Many)

1. **#2** - Blocks 13 issues (all collectors, stores, schemas)
2. **#10** - Blocks 7 issues (all reporting and search features)
3. **#8** - Blocks 5 issues (CLI, quality gates, coupling)
4. **#14** - Blocks 4 issues (SARIF, filters, tech debt)

### Terminal Issues (Block Nothing)

Safe to defer if needed:
- #11, #16, #18, #22, #23, #25, #26, #30, #31, #32, #33, #36, #37, #38, #39

---

## Risk Mitigation

### High-Risk Items

| Issue | Risk | Mitigation |
|-------|------|------------|
| #2 | SPOF - blocks 13 issues | Extra review, comprehensive tests |
| #8 | Performance overhead | Profile early, optimize hot paths |
| #10 | Blocks all phases | Design for extensibility |
| #14 | Threshold tuning | Start with SonarQube defaults |
| #20 | Complex cross-file | Leverage tree-sitter queries |
| #30 | Frontend complexity | Use minimal CSS, no frameworks |
| #38 | LLM quality | Structured prompts, validation |

---

## Recommended Work Strategy

### Week 1
1. Start #2 immediately (3 days)
2. Start #13 in parallel (can work simultaneously)
3. After #2 complete → fan out to #3-7, #9 (all in parallel)

### Week 2
4. Complete all collectors (#3-7) and schema (#9)
5. Start #8 integration (requires all collectors)
6. After #8 → start #10 CLI

### Week 3
7. Complete #10 CLI
8. Complete #11 console reporter
9. **Ship v0.17.0** (Phase 1)
10. Start #14 code smells (needs #8, #13)

### Week 4
11. Complete #14 and dependent issues (#15-18)
12. **Ship v0.18.0** (Phase 2)
13. Start #20 coupling (needs #2, #8)

### Week 5
14. Complete all cross-file issues (#20-26)
15. **Ship v0.19.0** (Phase 3)

### Week 6
16. Complete visualization (#28-33)
17. **Ship v0.20.0** (Phase 4)

### Week 7-8
18. Complete search integration (#35-38)
19. **Ship v0.21.0** (Phase 5)
20. **PROJECT COMPLETE**

---

## Success Criteria

### Per Sprint
- [ ] All sprint issues → "Done"
- [ ] Tests passing (80%+ coverage)
- [ ] Documentation updated
- [ ] CHANGELOG.md entry

### Per Phase (Milestone)
- [ ] Milestone validation criteria met
- [ ] Version tagged and released
- [ ] User guide updated
- [ ] GitHub release notes

### Overall
- [ ] All 39 issues completed
- [ ] 5 phases shipped (v0.17.0 - v0.21.0)
- [ ] MCP tools in Claude Desktop
- [ ] Quality-aware search functional

---

## Next Actions

1. **Immediate**: Start #2 (Core metric dataclasses)
2. **Day 1**: Create feature branch `feature/2-metric-dataclasses`
3. **Day 3**: Complete #2, fan out to Sprint 2 issues
4. **Week 1 End**: Sprint 1 and partial Sprint 2 complete
5. **Week 2**: Complete Sprint 2 and Sprint 3, ship v0.17.0

---

**Full Plan**: [sprint-plan.md](./sprint-plan.md)
**Dependency Graph**: [dependency-graph.txt](./dependency-graph.txt)
**Project**: https://github.com/users/bobmatnyc/projects/13
