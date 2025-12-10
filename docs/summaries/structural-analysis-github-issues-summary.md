# Structural Code Analysis - GitHub Issues Summary

**Created**: December 9, 2024
**Repository**: [bobmatnyc/mcp-vector-search](https://github.com/bobmatnyc/mcp-vector-search)
**Total Issues Created**: 37 (5 Epics + 32 Implementation Issues)

## Overview

This document summarizes the GitHub issues created for implementing the Structural Code Analysis feature in mcp-vector-search, based on the design document located at `docs/research/mcp-vector-search-structural-analysis-design.md`.

## Project Structure

The implementation is organized into 5 epics spanning approximately 5 weeks of development:

- **Epic 1**: Core Metrics (Phase 1) - Week 1-2
- **Epic 2**: Quality Gates (Phase 2) - Week 3
- **Epic 3**: Cross-File Analysis (Phase 3) - Week 4
- **Epic 4**: Visualization Export (Phase 4) - Week 5
- **Epic 5**: Search Integration (Phase 5) - Future/Backlog

## Epic Breakdown

### [Epic 1: Core Metrics - Phase 1](https://github.com/bobmatnyc/mcp-vector-search/issues/1)

**Timeline**: Week 1-2
**Focus**: Tier 1 collectors integrated into indexer with basic analysis command

**Issues**:
1. [#2 - Create metric dataclasses and interfaces](https://github.com/bobmatnyc/mcp-vector-search/issues/2)
2. [#3 - Implement Cognitive Complexity Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/3)
3. [#4 - Implement Cyclomatic Complexity Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/4)
4. [#5 - Implement Nesting Depth Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/5)
5. [#6 - Implement Parameter Count Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/6)
6. [#7 - Implement Method Count Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/7)
7. [#8 - Integrate collectors with existing TreeSitter indexer](https://github.com/bobmatnyc/mcp-vector-search/issues/8)
8. [#9 - Extend ChromaDB metadata schema](https://github.com/bobmatnyc/mcp-vector-search/issues/9)
9. [#10 - Create analyze --quick CLI command](https://github.com/bobmatnyc/mcp-vector-search/issues/10)
10. [#11 - Implement console reporter](https://github.com/bobmatnyc/mcp-vector-search/issues/11)

**Deliverables**:
- Tier 1 collectors integrated into indexer
- Extended chunk metadata in ChromaDB
- `analyze --quick` command
- Basic console reporter

**Validation Criteria**:
- Metrics match SonarQube on sample projects
- <10ms overhead per 1000 LOC

---

### [Epic 2: Quality Gates - Phase 2](https://github.com/bobmatnyc/mcp-vector-search/issues/12)

**Timeline**: Week 3
**Focus**: Threshold configuration, CI integration, and diff-aware analysis

**Issues**:
1. [#13 - Create threshold configuration system](https://github.com/bobmatnyc/mcp-vector-search/issues/13)
2. [#14 - Implement code smell detection](https://github.com/bobmatnyc/mcp-vector-search/issues/14)
3. [#15 - Add SARIF output format](https://github.com/bobmatnyc/mcp-vector-search/issues/15)
4. [#16 - Implement --fail-on-smell exit codes](https://github.com/bobmatnyc/mcp-vector-search/issues/16)
5. [#17 - Add diff-aware analysis](https://github.com/bobmatnyc/mcp-vector-search/issues/17)
6. [#18 - Add baseline comparison](https://github.com/bobmatnyc/mcp-vector-search/issues/18)

**Deliverables**:
- Threshold configuration system (YAML-based with presets)
- SARIF output for CI integration
- `--fail-on-smell` exit codes
- Diff-aware analysis (`--changed-only`)

**Validation Criteria**:
- GitHub Actions integration working
- Threshold overrides function correctly

---

### [Epic 3: Cross-File Analysis - Phase 3](https://github.com/bobmatnyc/mcp-vector-search/issues/19)

**Timeline**: Week 4
**Focus**: Tier 4 collectors for coupling analysis and dependency tracking

**Issues**:
1. [#20 - Implement Efferent Coupling Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/20)
2. [#21 - Implement Afferent Coupling Collector](https://github.com/bobmatnyc/mcp-vector-search/issues/21)
3. [#22 - Calculate Instability Index](https://github.com/bobmatnyc/mcp-vector-search/issues/22)
4. [#23 - Implement circular dependency detection](https://github.com/bobmatnyc/mcp-vector-search/issues/23)
5. [#24 - Create SQLite metrics store](https://github.com/bobmatnyc/mcp-vector-search/issues/24)
6. [#25 - Implement trend tracking](https://github.com/bobmatnyc/mcp-vector-search/issues/25)
7. [#26 - Implement LCOM4 cohesion metric](https://github.com/bobmatnyc/mcp-vector-search/issues/26)

**Deliverables**:
- Tier 4 collectors (afferent coupling, circular deps)
- Dependency graph construction
- SQLite metrics store with historical tracking
- Trend tracking for metrics over time

**Validation Criteria**:
- Circular dependencies detected correctly using Tarjan's SCC algorithm
- Historical snapshots recorded and queryable

---

### [Epic 4: Visualization Export - Phase 4](https://github.com/bobmatnyc/mcp-vector-search/issues/27)

**Timeline**: Week 5
**Focus**: JSON export schemas and visualization data for charts/graphs

**Issues**:
1. [#28 - Design visualization JSON export schema](https://github.com/bobmatnyc/mcp-vector-search/issues/28)
2. [#29 - Implement JSON exporter](https://github.com/bobmatnyc/mcp-vector-search/issues/29)
3. [#30 - Create HTML standalone report](https://github.com/bobmatnyc/mcp-vector-search/issues/30)
4. [#31 - Implement Halstead metrics collector](https://github.com/bobmatnyc/mcp-vector-search/issues/31)
5. [#32 - Add technical debt estimation](https://github.com/bobmatnyc/mcp-vector-search/issues/32)
6. [#33 - Extend status command with metrics](https://github.com/bobmatnyc/mcp-vector-search/issues/33)

**Deliverables**:
- JSON export for external visualizers
- All chart data schemas finalized
- Self-contained HTML report with interactive charts
- Halstead complexity metrics
- Technical debt estimation in hours

**Validation Criteria**:
- External visualizer consumes export successfully
- All documented chart types renderable
- HTML report is fully self-contained (<5MB)

---

### [Epic 5: Search Integration - Phase 5 (Future)](https://github.com/bobmatnyc/mcp-vector-search/issues/34)

**Timeline**: Future/Backlog
**Focus**: Quality-aware search ranking and filtering

**Issues**:
1. [#35 - Add quality filters to search](https://github.com/bobmatnyc/mcp-vector-search/issues/35)
2. [#36 - Implement quality-aware ranking](https://github.com/bobmatnyc/mcp-vector-search/issues/36)
3. [#37 - Expose analysis as MCP tools](https://github.com/bobmatnyc/mcp-vector-search/issues/37)

**Deliverables**:
- Quality filters for search (`--max-complexity`, `--no-smells`, `--grade`)
- Quality-aware ranking algorithm blending relevance with code quality
- MCP tool integration for Claude Desktop

**Validation Criteria**:
- Search results ranked by combined quality + relevance score
- Filters work correctly with ChromaDB metadata
- MCP tools accessible from Claude Desktop

---

## Key Metrics and Thresholds

### Cognitive Complexity (Primary Quality Metric)
- **A Grade**: ≤5
- **B Grade**: ≤10
- **C Grade**: ≤15 (SonarQube default threshold)
- **D Grade**: ≤25
- **F Grade**: >25

### Other Metrics
- **Cyclomatic Complexity**: ≤10 (display only)
- **Max Nesting Depth**: ≤4 levels
- **Function Length**: ≤30 lines (warning), ≤50 lines (error)
- **Parameter Count**: ≤5 parameters
- **Method Count per Class**: ≤20 methods
- **Efferent Coupling (Ce)**: ≤20 external dependencies per file
- **LCOM4 Cohesion**: =1 (ideal), ≤2 (acceptable)

### Code Smells Detected
1. **Long Method** - LOC > 30 OR cognitive_complexity > 15
2. **Deep Nesting** - max_depth > 4
3. **Long Parameter List** - param_count > 5
4. **God Class** - methods > 20 AND loc > 500
5. **Large Class** - LOC > 500
6. **Empty Catch** - catch block with pass/empty body
7. **Magic Numbers** - numeric literals outside [-1, 0, 1, 2, 10, 100]

## Technical Debt Model

Based on SonarQube remediation times:
- Long Method: 20 minutes
- Deep Nesting: 15 minutes
- Long Parameter List: 10 minutes
- God Class: 120 minutes (2 hours)
- Empty Catch: 5 minutes
- Magic Numbers: 5 minutes
- High Cognitive Complexity: 30 minutes per function
- Circular Dependency: 60 minutes per cycle

**Severity Multipliers**:
- Error: 1.5x
- Warning: 1.0x

## Visualization Chart Types

1. **Complexity Heatmap** - File grid colored by max cognitive complexity
2. **Dependency Graph** - Force-directed graph of import relationships
3. **Complexity Distribution Histogram** - Function count by grade (A-F)
4. **Smell Treemap** - Rectangles sized by smell count
5. **Trend Line Chart** - Time series of key metrics
6. **Coupling Wheel** - Chord diagram showing import relationships

## Database Schema

### ChromaDB Metadata Extensions
- `cognitive_complexity` (int)
- `cyclomatic_complexity` (int)
- `max_nesting_depth` (int)
- `line_count` (int)
- `parameter_count` (int, optional)
- `method_count` (int, optional)
- `complexity_grade` (str: A-F)
- `has_smells` (bool)
- `smell_count` (int)

### SQLite Metrics Store Tables
- `file_metrics` - File-level metrics (current snapshot)
- `project_snapshots` - Project-wide metrics over time
- `code_smells` - Individual smell instances with locations
- `circular_dependencies` - Detected dependency cycles

## Performance Targets

### Tier 1 Collectors (Free During Indexing)
- Cognitive Complexity: <1ms per 1000 LOC
- Cyclomatic Complexity: <1ms per 1000 LOC
- Nesting Depth: <0.5ms per 1000 LOC
- Parameter/Method Count: O(1) per function/class

### Tier 2 Collectors (Single-Pass)
- Efferent Coupling: <2ms per file
- Halstead Metrics: <2ms per file

### Tier 3 Collectors (Whole-File)
- LCOM4 Cohesion: ~5ms per class

### Overall Target
- **Total overhead**: <10ms per 1000 LOC for Tier 1-2 metrics
- **Memory overhead**: ~100 bytes per function, ~500 bytes per file

## CI/CD Integration

### Exit Codes
- **0**: Success (no issues or within thresholds)
- **1**: Quality gate failure (smells found or thresholds exceeded)
- **2**: Error (invalid configuration, file not found, etc.)

### SARIF Output
- SARIF 2.1.0 compliant
- GitHub Code Scanning compatible
- PR diff annotations supported

### Diff-Aware Analysis
- `--changed-only` flag analyzes only git-changed files
- `--since COMMIT` for custom comparison point
- Works in CI environments (detached HEAD state)

## Reference Links

- **Repository**: https://github.com/bobmatnyc/mcp-vector-search
- **All Issues**: https://github.com/bobmatnyc/mcp-vector-search/issues
- **Design Document**: `docs/research/mcp-vector-search-structural-analysis-design.md`
- **Project Board**: Create a GitHub Project to track these issues

## Next Steps

1. **Create GitHub Project** (requires `project` scope):
   - Run: `gh auth refresh -s project,read:project`
   - Create project and link all 37 issues
   - Organize into columns by epic/phase

2. **Set up Milestones**:
   - Phase 1: Core Metrics (Week 1-2)
   - Phase 2: Quality Gates (Week 3)
   - Phase 3: Cross-File Analysis (Week 4)
   - Phase 4: Visualization Export (Week 5)
   - Phase 5: Search Integration (Future)

3. **Add Labels**:
   - `epic` - for epic issues
   - `enhancement` - already applied to all issues
   - `documentation` - for schema/docs issues
   - `good-first-issue` - for simpler collectors
   - `testing` - for test-heavy issues

4. **Assign Issues**:
   - Assign team members to specific issues
   - Mark dependencies between issues
   - Set priorities within each epic

## Summary Statistics

- **Total Issues**: 37
- **Epic Issues**: 5
- **Implementation Issues**: 32
- **Estimated Timeline**: 5 weeks (4 weeks active + 1 future)
- **Code Smells Detected**: 7 types
- **Metrics Collected**: 15+ different metrics
- **Output Formats**: 4 (console, JSON, SARIF, HTML)
- **Languages Supported**: Python, JavaScript, TypeScript (extensible)

---

**Document Created**: December 9, 2024
**Last Updated**: December 9, 2024
**Status**: All issues created and ready for project board organization
