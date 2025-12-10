# Project: Structural Code Analysis

> **Status**: Active | **Phase**: 1 of 5 | **Target**: v0.17.0 - v0.21.0

## Quick Links

| Resource | URL |
|----------|-----|
| **GitHub Project** | https://github.com/users/bobmatnyc/projects/13 |
| **Roadmap View** | https://github.com/users/bobmatnyc/projects/13/views/1 |
| **Milestones** | https://github.com/bobmatnyc/mcp-vector-search/milestones |
| **All Issues** | https://github.com/bobmatnyc/mcp-vector-search/issues |
| **Design Document** | [structural-analysis-design.md](../research/mcp-vector-search-structural-analysis-design.md) |
| **PR Workflow** | [pr-workflow-guide.md](../development/pr-workflow-guide.md) |

## Overview

Add structural code analysis capabilities to mcp-vector-search, enabling quality-aware code search and codebase health visualization.

### Key Features

- **Structural Metrics**: Cognitive/cyclomatic complexity, nesting depth, coupling
- **Code Smell Detection**: Long methods, deep nesting, god classes, empty catches
- **Quality Gates**: Configurable thresholds with SARIF output for CI/CD
- **Visualization Export**: JSON/HTML reports for dashboards
- **Search Integration**: Quality-aware ranking and filtering

## Timeline

| Phase | Milestone | Start | Due | Issues | Status |
|-------|-----------|-------|-----|--------|--------|
| 1 | v0.17.0 - Core Metrics | Dec 10 | Dec 23, 2024 | #1-11 | 🎯 Ready |
| 2 | v0.18.0 - Quality Gates | Dec 24 | Dec 30, 2024 | #12-18 | 📋 Backlog |
| 3 | v0.19.0 - Cross-File Analysis | Dec 31 | Jan 6, 2025 | #19-26 | 📋 Backlog |
| 4 | v0.20.0 - Visualization Export | Jan 7 | Jan 13, 2025 | #27-33 | 📋 Backlog |
| 5 | v0.21.0 - Search Integration | Jan 20 | Feb 3, 2025 | #34-37 | 📋 Backlog |

## Workflow

```
📋 Backlog → 🎯 Ready → 🔧 In Progress → 👀 In Review → ✅ Done
```

### Branch Naming

```
feature/<issue-number>-<short-description>
```

Example: `feature/2-metric-dataclasses`

## Critical Path

The minimum timeline is determined by these dependent issues:

```
#2 (dataclasses) → #8 (integrator) → #10 (CLI) → #14 (smells) → #35 (filters) → #37 (MCP)
```

**Start with**: Issue #2 - no blockers, on critical path

## Phase Details

### Phase 1: Core Metrics (v0.17.0)

**Goal**: Tier 1 collectors integrated into indexer with basic analysis command

| Issue | Title | Dependencies | Status |
|-------|-------|--------------|--------|
| #1 | [EPIC] Core Metrics | - | 🎯 Ready |
| #2 | Create metric dataclasses | None | 🎯 Ready |
| #3 | Cognitive Complexity Collector | #2 | 🎯 Ready |
| #4 | Cyclomatic Complexity Collector | #2 | 🎯 Ready |
| #5 | Nesting Depth Collector | #2 | 🎯 Ready |
| #6 | Parameter Count Collector | #2 | 🎯 Ready |
| #7 | Method Count Collector | #2 | 🎯 Ready |
| #8 | Integrate collectors with indexer | #2-7 | 🎯 Ready |
| #9 | Extend ChromaDB metadata | #2 | 🎯 Ready |
| #10 | Create `analyze --quick` CLI | #8, #9 | 🎯 Ready |
| #11 | Console reporter | #10 | 🎯 Ready |

**Validation Criteria**:
- Metrics match SonarQube on sample projects
- <10ms overhead per 1000 LOC

### Phase 2: Quality Gates (v0.18.0)

**Goal**: Threshold configuration, CI integration, diff-aware analysis

| Issue | Title | Dependencies |
|-------|-------|--------------|
| #12 | [EPIC] Quality Gates | Phase 1 |
| #13 | Threshold configuration system | #2 |
| #14 | Code smell detection | #8, #13 |
| #15 | SARIF output format | #10, #14 |
| #16 | `--fail-on-smell` exit codes | #14, #15 |
| #17 | Diff-aware analysis | #10 |
| #18 | Baseline comparison | #17 |

### Phase 3: Cross-File Analysis (v0.19.0)

**Goal**: Coupling metrics, dependency graph, SQLite storage

| Issue | Title | Dependencies |
|-------|-------|--------------|
| #19 | [EPIC] Cross-File Analysis | Phase 2 |
| #20 | Efferent Coupling Collector | #2, #8 |
| #21 | Afferent Coupling Collector | #20 |
| #22 | Instability Index | #20, #21 |
| #23 | Circular dependency detection | #20 |
| #24 | SQLite metrics store | #2 |
| #25 | Trend tracking | #24 |
| #26 | LCOM4 cohesion metric | #2, #8 |

### Phase 4: Visualization Export (v0.20.0)

**Goal**: JSON/HTML reports, Halstead metrics, tech debt estimation

| Issue | Title | Dependencies |
|-------|-------|--------------|
| #27 | [EPIC] Visualization Export | Phase 3 |
| #28 | JSON export schema | #2 |
| #29 | JSON exporter | #28, #10 |
| #30 | HTML standalone report | #29 |
| #31 | Halstead metrics collector | #2, #8 |
| #32 | Technical debt estimation | #14, #24 |
| #33 | `status --metrics` command | #10, #24 |

### Phase 5: Search Integration (v0.21.0)

**Goal**: Quality-aware search ranking, MCP tool exposure, LLM interpretation

| Issue | Title | Dependencies |
|-------|-------|--------------|
| #34 | [EPIC] Search Integration | Phase 4 |
| #35 | Quality filters for search | #10, #14 |
| #36 | Quality-aware ranking | #35 |
| #37 | Expose as MCP tools | #10, #35 |
| #38 | LLM interpretation of analysis | #10, #14, #29, #37 |

## Project Settings

### GitHub Project Configuration

- **Project ID**: 13
- **Project URL**: https://github.com/users/bobmatnyc/projects/13
- **Visibility**: Public
- **Owner**: @bobmatnyc

### Custom Fields

| Field | Type | Purpose |
|-------|------|---------|
| Start Date | Date | When work begins |
| Target Date | Date | Due date for completion |
| Workflow | Single Select | PR-based workflow status |

### Workflow Options

| Option | Color | Description |
|--------|-------|-------------|
| 📋 Backlog | Gray | Dependencies not met |
| 🎯 Ready | Blue | Ready to implement |
| 🔧 In Progress | Yellow | Actively being developed |
| 👀 In Review | Purple | PR awaiting review |
| ✅ Done | Green | PR merged, complete |

## Architecture

### New Module Structure

```
src/mcp_vector_search/
├── analysis/
│   ├── __init__.py
│   ├── metrics.py              # Metric dataclasses
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── base.py             # MetricCollector ABC
│   │   ├── complexity.py       # Cognitive/cyclomatic
│   │   ├── coupling.py         # Efferent/afferent
│   │   ├── cohesion.py         # LCOM4
│   │   ├── smells.py           # Code smell detection
│   │   └── halstead.py         # Halstead metrics
│   ├── aggregators/
│   │   ├── __init__.py
│   │   ├── file.py             # File-level aggregation
│   │   ├── project.py          # Project-level aggregation
│   │   └── trends.py           # Historical tracking
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── console.py          # Rich terminal output
│   │   ├── json.py             # JSON export
│   │   ├── sarif.py            # SARIF for CI
│   │   └── html.py             # Standalone report
│   ├── visualizer/
│   │   ├── __init__.py
│   │   ├── exporter.py         # Viz data export
│   │   └── schemas.py          # Visualization schemas
│   └── thresholds.py           # Threshold management
├── cli/
│   └── commands/
│       └── analyze.py          # New analyze command
```

## References

- [Design Document](../research/mcp-vector-search-structural-analysis-design.md) - Full technical specification
- [PR Workflow Guide](../development/pr-workflow-guide.md) - Branch naming, commit format
- [GitHub Milestones Setup](../development/github-milestones-setup.md) - Automation scripts
- [Dependency Graph](../development/dependency-graph.txt) - Visual dependency map

---

**Created**: December 9, 2024
**Last Updated**: December 9, 2024
**Assignee**: @bobmatnyc
