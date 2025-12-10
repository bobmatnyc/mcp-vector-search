# Structural Code Analysis - Sprint Plan

> **Generated**: December 10, 2024
> **Project**: https://github.com/users/bobmatnyc/projects/13
> **Issues**: #1-39

## Executive Summary

**Total Issues**: 39 (including 5 EPIC tracking issues)
**Total Sprints**: 8
**Critical Path**: #2 → #8 → #10 → #14 → #35 → #37 (6 hops)
**Estimated Duration**: 8 weeks (2-week sprints)

## Sprint Organization

Sprints are organized by dependency satisfaction:
- Each sprint contains only tickets whose dependencies are met by previous sprints
- Complexity estimates based on acceptance criteria count from design doc
- Epic issues (#1, #12, #19, #27, #34) are tracking-only and don't block work

---

## Sprint 1: Foundation (Week 1-2)

**Theme**: Core dataclasses and configuration
**Dependencies**: None - can start immediately
**Target**: Dec 10-23, 2024

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #2 | Create metric dataclasses | Medium | None | 13 issues |
| #13 | Threshold configuration system | Medium | None | #14 |

### Acceptance Criteria Summary

**#2 - Metric Dataclasses** (8 criteria):
- Define dataclasses for all Tier 1-3 metrics
- Pydantic validation for field types
- JSON serialization support
- Unit tests for all dataclasses

**#13 - Threshold Configuration** (6 criteria):
- YAML configuration schema for thresholds
- Default threshold presets (strict/recommended/permissive)
- Validation of threshold values
- Integration tests for threshold loading

### Sprint 1 Deliverables
- ✅ Core metric type system established
- ✅ Configuration infrastructure ready
- ✅ Unblocks 14 downstream issues

---

## Sprint 2: Tier 1 Collectors (Week 1-2)

**Theme**: Basic metric collectors
**Dependencies**: #2 (Sprint 1)
**Target**: Dec 10-23, 2024 (parallel with Sprint 1 after #2 completes)

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #3 | Cognitive Complexity Collector | High | #2 | #8 |
| #4 | Cyclomatic Complexity Collector | Medium | #2 | #8 |
| #5 | Nesting Depth Collector | Low | #2 | #8 |
| #6 | Parameter Count Collector | Low | #2 | #8 |
| #7 | Method Count Collector | Low | #2 | #8 |
| #9 | Extend ChromaDB metadata schema | Medium | #2 | #10 |

### Complexity Notes

**High** (#3): Cognitive complexity requires complex AST analysis
**Medium** (#4, #9): Moderate implementation effort
**Low** (#5, #6, #7): Simple tree-sitter queries

### Sprint 2 Deliverables
- ✅ All Tier 1 metric collectors implemented
- ✅ ChromaDB schema extended with metric fields
- ✅ Ready for integration into indexer

---

## Sprint 3: Integration & CLI (Week 2)

**Theme**: Integrate collectors into indexer pipeline
**Dependencies**: #2, #3, #4, #5, #6, #7, #9 (Sprints 1-2)
**Target**: Dec 16-23, 2024

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #8 | Integrate collectors with indexer | High | #2-7 | 6 issues |
| #10 | Create `analyze --quick` CLI | Medium | #8, #9 | 7 issues |
| #11 | Console reporter | Low | #10 | None |

### Critical Path Alert
**#8 and #10 are on the critical path** - delays here cascade to all phases

### Acceptance Criteria Summary

**#8 - Integration** (10 criteria):
- Modify indexer to call all collectors
- Store metrics in ChromaDB metadata
- <10ms overhead per 1000 LOC
- Validation against SonarQube on sample projects

**#10 - CLI Command** (8 criteria):
- `analyze --quick` command with file/directory support
- Display all Tier 1 metrics
- Output format options (table/json/summary)
- Integration tests with sample projects

**#11 - Console Reporter** (4 criteria):
- Rich terminal output with color-coded metrics
- Threshold violation highlighting
- Summary statistics display

### Sprint 3 Deliverables
- ✅ Phase 1 (v0.17.0) COMPLETE
- ✅ Working end-to-end metric collection pipeline
- ✅ User-facing CLI command available
- ✅ Unblocks Phase 2 quality gates

---

## Sprint 4: Quality Gates Foundation (Week 3)

**Theme**: Code smell detection and SARIF output
**Dependencies**: #8, #10, #13 (Sprints 1-3)
**Target**: Dec 24-30, 2024

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #14 | Code smell detection | High | #8, #13 | 4 issues |
| #15 | SARIF output format | Medium | #10, #14 | #16 |
| #16 | `--fail-on-smell` exit codes | Low | #14, #15 | None |
| #17 | Diff-aware analysis | Medium | #10 | #18 |
| #18 | Baseline comparison | Medium | #17 | None |

### Critical Path Alert
**#14 is on the critical path** - required for Phase 5 search integration

### Acceptance Criteria Summary

**#14 - Code Smells** (12 criteria):
- Detect 6 smell types (long method, deep nesting, god class, etc.)
- Use configurable thresholds from #13
- Store smell metadata in ChromaDB
- Unit tests for all smell detectors

**#15 - SARIF Output** (6 criteria):
- Generate SARIF 2.1.0 compliant JSON
- Map smells to SARIF rules
- Include source locations and severity
- Validate against SARIF schema

### Sprint 4 Deliverables
- ✅ Code quality analysis beyond basic metrics
- ✅ CI/CD integration via SARIF
- ✅ Diff-aware analysis for PR workflows
- ✅ Phase 2 (v0.18.0) COMPLETE

---

## Sprint 5: Cross-File Analysis (Week 4)

**Theme**: Coupling metrics and dependency analysis
**Dependencies**: #2, #8 (Sprints 1-3)
**Target**: Dec 31, 2024 - Jan 6, 2025

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #20 | Efferent Coupling Collector | High | #2, #8 | 3 issues |
| #21 | Afferent Coupling Collector | Medium | #20 | #22 |
| #22 | Instability Index | Low | #20, #21 | None |
| #23 | Circular dependency detection | Medium | #20 | None |
| #24 | SQLite metrics store | Medium | #2 | 3 issues |
| #25 | Trend tracking | Medium | #24 | None |
| #26 | LCOM4 cohesion metric | High | #2, #8 | None |

### Complexity Notes

**High** (#20, #26): Complex cross-file analysis and graph algorithms
**Medium** (#21, #23, #24, #25): Moderate implementation effort
**Low** (#22): Simple calculation from #20 and #21

### Sprint 5 Deliverables
- ✅ Full dependency graph analysis
- ✅ Historical metrics tracking
- ✅ Cohesion measurement
- ✅ Phase 3 (v0.19.0) COMPLETE

---

## Sprint 6: Visualization Export (Week 5-6)

**Theme**: JSON/HTML reports and Halstead metrics
**Dependencies**: #2, #8, #10, #14, #24 (Sprints 1-5)
**Target**: Jan 7-13, 2025

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #28 | JSON export schema | Low | #2 | #29 |
| #29 | JSON exporter | Medium | #28, #10 | #30 |
| #30 | HTML standalone report | High | #29 | None |
| #31 | Halstead metrics collector | Medium | #2, #8 | None |
| #32 | Technical debt estimation | Medium | #14, #24 | None |
| #33 | `status --metrics` command | Low | #10, #24 | None |

### Acceptance Criteria Summary

**#28 - JSON Schema** (4 criteria):
- Define JSON schema for all metric types
- Include metadata (timestamp, project, files)
- Schema validation support
- Documentation of schema structure

**#30 - HTML Report** (8 criteria):
- Standalone HTML with embedded CSS/JS
- Interactive charts (complexity distribution, trends)
- Drill-down from project → file → function
- Responsive design for mobile/desktop

### Sprint 6 Deliverables
- ✅ Exportable metrics for dashboards
- ✅ Standalone HTML reports
- ✅ Technical debt quantification
- ✅ Phase 4 (v0.20.0) COMPLETE

---

## Sprint 7: Search Integration (Week 7-8)

**Theme**: Quality-aware search and MCP tools
**Dependencies**: #10, #14 (Sprints 3-4)
**Target**: Jan 20 - Feb 3, 2025

| Issue | Title | Complexity | Dependencies | Blocks |
|-------|-------|------------|--------------|--------|
| #35 | Quality filters for search | Medium | #10, #14 | 2 issues |
| #36 | Quality-aware ranking | High | #35 | None |
| #37 | Expose as MCP tools | Medium | #10, #35 | None |
| #38 | LLM interpretation of analysis | High | #10, #14, #29, #37 | None |

### Critical Path Alert
**#35 and #37 complete the critical path** - final deliverables for Phase 5

### Acceptance Criteria Summary

**#35 - Quality Filters** (6 criteria):
- Add `--max-complexity` filter to search
- Filter by smell presence (e.g., `--no-god-classes`)
- Combine with semantic search
- Integration tests with sample queries

**#38 - LLM Interpretation** (10 criteria):
- Generate natural language summaries of analysis
- Explain why code has high complexity
- Suggest refactoring opportunities
- Context-aware recommendations based on project patterns

### Sprint 7 Deliverables
- ✅ Quality-aware semantic search
- ✅ MCP tool exposure for Claude Desktop
- ✅ LLM-powered analysis interpretation
- ✅ Phase 5 (v0.21.0) COMPLETE
- ✅ **PROJECT COMPLETE**

---

## Sprint 8: EPIC Tracking Issues

**Theme**: Project management and documentation
**Dependencies**: Various
**Target**: Ongoing throughout project

| Issue | Title | Dependencies | Purpose |
|-------|-------|--------------|---------|
| #1 | [EPIC] Core Metrics | None | Track Phase 1 |
| #12 | [EPIC] Quality Gates | Phase 1 | Track Phase 2 |
| #19 | [EPIC] Cross-File Analysis | Phase 2 | Track Phase 3 |
| #27 | [EPIC] Visualization Export | Phase 3 | Track Phase 4 |
| #34 | [EPIC] Search Integration | Phase 4 | Track Phase 5 |
| #39 | Project documentation updates | Various | Continuous |

### Notes
- EPIC issues are tracking-only (don't block development)
- Can be closed when all child issues complete
- #39 is continuous documentation maintenance

---

## Dependency Matrix

### Issues by Dependency Count

**0 Dependencies** (can start immediately):
- #2 (Core dataclasses) ← **START HERE**
- #13 (Threshold configuration)

**1 Dependency**:
- #3, #4, #5, #6, #7, #9 (depend on #2)
- #11, #17 (depend on #10)
- #21, #23 (depend on #20)
- #25 (depends on #24)
- #28 (depends on #2)
- #31 (depends on #2, #8)

**2 Dependencies**:
- #10 (depends on #8, #9) ← **CRITICAL PATH**
- #14 (depends on #8, #13) ← **CRITICAL PATH**
- #15 (depends on #10, #14)
- #18 (depends on #17)
- #20 (depends on #2, #8)
- #22 (depends on #20, #21)
- #26, #29 (multiple)
- #33, #35 (multiple) ← **CRITICAL PATH**

**3+ Dependencies**:
- #8 (depends on #2-7) - **6 dependencies**
- #16 (depends on #14, #15)
- #30 (depends on #29)
- #32 (depends on #14, #24)
- #36 (depends on #35)
- #37 (depends on #10, #35) ← **CRITICAL PATH END**
- #38 (depends on #10, #14, #29, #37) - **4 dependencies**

### Most Blocking Issues

These issues block the most downstream work:

1. **#10** - Blocks 7 issues (#11, #15, #17, #29, #33, #35, #38)
2. **#2** - Blocks 13 issues (all collectors + schema + stores)
3. **#8** - Blocks 5 issues (#10, #14, #20, #26, #31)
4. **#14** - Blocks 4 issues (#15, #16, #32, #35)

### Terminal Issues (don't block anything)

These can be completed anytime after their dependencies:
- #11, #16, #18, #22, #23, #25, #26, #30, #31, #32, #33, #36, #37, #38, #39

---

## Critical Path Analysis

**Longest Dependency Chain**: 6 hops

```
#2 → #8 → #10 → #14 → #35 → #37
```

**Estimated Timeline**:
1. #2: 3 days (dataclasses)
2. #8: 5 days (integration)
3. #10: 4 days (CLI)
4. #14: 5 days (smell detection)
5. #35: 3 days (search filters)
6. #37: 3 days (MCP tools)

**Total**: 23 days (minimum project duration)

**Buffer**: 8-week schedule provides ~35 days, giving 12 days buffer (52% slack)

---

## Complexity Distribution

### By Sprint

| Sprint | Low | Medium | High | Total Issues | Total Complexity |
|--------|-----|--------|------|--------------|------------------|
| 1 | 0 | 2 | 0 | 2 | 12 points |
| 2 | 3 | 2 | 1 | 6 | 18 points |
| 3 | 1 | 1 | 1 | 3 | 18 points |
| 4 | 1 | 3 | 1 | 5 | 22 points |
| 5 | 1 | 4 | 2 | 7 | 28 points |
| 6 | 2 | 3 | 1 | 6 | 20 points |
| 7 | 0 | 2 | 2 | 4 | 20 points |
| 8 | 0 | 0 | 0 | 5 | N/A (tracking) |

**Complexity Scoring**:
- Low = 3 points (simple tree-sitter query or config)
- Medium = 6 points (moderate AST analysis or integration)
- High = 10 points (complex algorithms or cross-file analysis)

### Balanced Workload

Sprints 4-6 are the heaviest (20-28 points each). Consider:
- Parallelizing work where possible (#20-26 in Sprint 5)
- Breaking Sprint 5 into two 1-week sprints if needed

---

## Risk Assessment

### High-Risk Issues

**#8 - Collector Integration** (Sprint 3):
- **Risk**: Performance overhead >10ms per 1000 LOC
- **Mitigation**: Profile early, optimize hot paths
- **Impact**: Blocks 5 downstream issues

**#14 - Code Smell Detection** (Sprint 4):
- **Risk**: Threshold tuning may require iteration
- **Mitigation**: Start with strict SonarQube defaults, make configurable
- **Impact**: On critical path, blocks Phase 5

**#20 - Efferent Coupling** (Sprint 5):
- **Risk**: Complex cross-file dependency resolution
- **Mitigation**: Leverage tree-sitter import/export queries
- **Impact**: Blocks dependency graph features

**#30 - HTML Report** (Sprint 6):
- **Risk**: Frontend complexity may expand scope
- **Mitigation**: Use minimal CSS framework (no React/Vue)
- **Impact**: Terminal node, doesn't block other work

**#38 - LLM Interpretation** (Sprint 7):
- **Risk**: LLM output quality unpredictable
- **Mitigation**: Use structured prompts, validate outputs
- **Impact**: Terminal node, can be enhanced post-launch

### Dependency Bottlenecks

**#2 is a SPOF** (Single Point of Failure):
- Blocks 13 downstream issues
- Must be completed correctly first time
- Recommendation: Extra review attention, comprehensive tests

**#10 blocks Phase 2-5**:
- All later phases depend on the CLI interface
- Recommendation: Design for extensibility, stable API

---

## Recommendations

### Work Strategy

1. **Start with #2 immediately** - It's the critical path start
2. **Parallelize Sprint 2** - All collectors can be developed in parallel after #2
3. **Focus on #8 and #10** - These are the highest-blocking issues
4. **Plan Phase 2 during Phase 1** - Use buffer time for design work
5. **Monitor critical path** - Track #2 → #8 → #10 → #14 → #35 → #37 daily

### Resource Allocation

**Optimal Team Size**: 2-3 developers

**Suggested Assignments**:
- **Developer 1**: Critical path (#2, #8, #10, #14, #35, #37)
- **Developer 2**: Collectors (#3-7), Coupling (#20-23)
- **Developer 3**: Reporting (#11, #15, #28-30), Terminal issues

### Sprint Planning

**Sprint Length**: 2 weeks recommended
- Allows for code review, testing, documentation
- Provides buffer for unexpected complexity
- Aligns with bi-weekly release cadence

**Sprint Goals**:
1. Sprint 1-3: **Ship Phase 1 (v0.17.0)**
2. Sprint 4: **Ship Phase 2 (v0.18.0)**
3. Sprint 5: **Ship Phase 3 (v0.19.0)**
4. Sprint 6: **Ship Phase 4 (v0.20.0)**
5. Sprint 7: **Ship Phase 5 (v0.21.0)**

---

## Success Metrics

### Per Sprint

- [ ] All sprint issues moved to "Done"
- [ ] All tests passing (unit + integration)
- [ ] Code coverage ≥80%
- [ ] Documentation updated
- [ ] CHANGELOG.md entry added

### Per Phase (Milestone)

- [ ] Milestone validation criteria met (see project doc)
- [ ] Version tagged and released to PyPI
- [ ] User guide updated with new features
- [ ] GitHub release notes published

### Overall Project

- [ ] All 39 issues completed
- [ ] 5 phases shipped (v0.17.0 - v0.21.0)
- [ ] MCP tools available in Claude Desktop
- [ ] Quality-aware search functional
- [ ] Documentation complete

---

## Sprint Velocity Tracking

### Planned Velocity

| Sprint | Issues | Complexity Points | Cumulative |
|--------|--------|-------------------|------------|
| 1 | 2 | 12 | 12 |
| 2 | 6 | 18 | 30 |
| 3 | 3 | 18 | 48 |
| 4 | 5 | 22 | 70 |
| 5 | 7 | 28 | 98 |
| 6 | 6 | 20 | 118 |
| 7 | 4 | 20 | 138 |
| 8 | 5 | N/A | N/A |

**Total Complexity**: 138 points
**Average per Sprint**: 19.7 points
**Peak Sprint**: Sprint 5 (28 points)

Use this to track actual velocity and adjust future sprint plans.

---

## Appendix: Issue Quick Reference

### Phase 1: Core Metrics (v0.17.0)

- #1: [EPIC] Core Metrics
- #2: Core metric dataclasses ← **START HERE**
- #3: Cognitive Complexity Collector
- #4: Cyclomatic Complexity Collector
- #5: Nesting Depth Collector
- #6: Parameter Count Collector
- #7: Method Count Collector
- #8: Integrate collectors ← **CRITICAL PATH**
- #9: Extend ChromaDB metadata
- #10: `analyze --quick` CLI ← **CRITICAL PATH**
- #11: Console reporter

### Phase 2: Quality Gates (v0.18.0)

- #12: [EPIC] Quality Gates
- #13: Threshold configuration
- #14: Code smell detection ← **CRITICAL PATH**
- #15: SARIF output format
- #16: `--fail-on-smell` exit codes
- #17: Diff-aware analysis
- #18: Baseline comparison

### Phase 3: Cross-File Analysis (v0.19.0)

- #19: [EPIC] Cross-File Analysis
- #20: Efferent coupling collector
- #21: Afferent coupling collector
- #22: Instability index
- #23: Circular dependency detection
- #24: SQLite metrics store
- #25: Trend tracking
- #26: LCOM4 cohesion metric

### Phase 4: Visualization Export (v0.20.0)

- #27: [EPIC] Visualization Export
- #28: JSON export schema
- #29: JSON exporter
- #30: HTML standalone report
- #31: Halstead metrics
- #32: Technical debt estimation
- #33: `status --metrics` command

### Phase 5: Search Integration (v0.21.0)

- #34: [EPIC] Search Integration
- #35: Quality filters for search ← **CRITICAL PATH**
- #36: Quality-aware ranking
- #37: Expose as MCP tools ← **CRITICAL PATH END**
- #38: LLM interpretation of analysis

### Continuous

- #39: Project documentation updates

---

**Generated by**: Claude Code (Ticketing Agent)
**Last Updated**: December 10, 2024
**Next Review**: After Sprint 1 completion
