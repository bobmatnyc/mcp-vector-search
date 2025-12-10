# Structural Code Analysis - Quick Reference Card

**Repository**: [bobmatnyc/mcp-vector-search](https://github.com/bobmatnyc/mcp-vector-search)

## Epic Overview

| Epic | Phase | Timeline | Issues | Focus |
|------|-------|----------|--------|-------|
| [#1](https://github.com/bobmatnyc/mcp-vector-search/issues/1) | Phase 1 | Week 1-2 | #2-#11 (10) | Core Metrics & Collectors |
| [#12](https://github.com/bobmatnyc/mcp-vector-search/issues/12) | Phase 2 | Week 3 | #13-#18 (6) | Quality Gates & CI |
| [#19](https://github.com/bobmatnyc/mcp-vector-search/issues/19) | Phase 3 | Week 4 | #20-#26 (7) | Cross-File Analysis |
| [#27](https://github.com/bobmatnyc/mcp-vector-search/issues/27) | Phase 4 | Week 5 | #28-#33 (6) | Visualization Export |
| [#34](https://github.com/bobmatnyc/mcp-vector-search/issues/34) | Phase 5 | Future | #35-#37 (3) | Search Integration |

## Issue Quick Links by Epic

### Epic 1: Core Metrics (#2-#11)
- [#2](https://github.com/bobmatnyc/mcp-vector-search/issues/2) - Metric dataclasses and interfaces
- [#3](https://github.com/bobmatnyc/mcp-vector-search/issues/3) - Cognitive Complexity Collector
- [#4](https://github.com/bobmatnyc/mcp-vector-search/issues/4) - Cyclomatic Complexity Collector
- [#5](https://github.com/bobmatnyc/mcp-vector-search/issues/5) - Nesting Depth Collector
- [#6](https://github.com/bobmatnyc/mcp-vector-search/issues/6) - Parameter Count Collector
- [#7](https://github.com/bobmatnyc/mcp-vector-search/issues/7) - Method Count Collector
- [#8](https://github.com/bobmatnyc/mcp-vector-search/issues/8) - TreeSitter indexer integration
- [#9](https://github.com/bobmatnyc/mcp-vector-search/issues/9) - ChromaDB metadata schema
- [#10](https://github.com/bobmatnyc/mcp-vector-search/issues/10) - `analyze --quick` CLI command
- [#11](https://github.com/bobmatnyc/mcp-vector-search/issues/11) - Console reporter

### Epic 2: Quality Gates (#13-#18)
- [#13](https://github.com/bobmatnyc/mcp-vector-search/issues/13) - Threshold configuration system
- [#14](https://github.com/bobmatnyc/mcp-vector-search/issues/14) - Code smell detection
- [#15](https://github.com/bobmatnyc/mcp-vector-search/issues/15) - SARIF output format
- [#16](https://github.com/bobmatnyc/mcp-vector-search/issues/16) - `--fail-on-smell` exit codes
- [#17](https://github.com/bobmatnyc/mcp-vector-search/issues/17) - Diff-aware analysis
- [#18](https://github.com/bobmatnyc/mcp-vector-search/issues/18) - Baseline comparison

### Epic 3: Cross-File Analysis (#20-#26)
- [#20](https://github.com/bobmatnyc/mcp-vector-search/issues/20) - Efferent Coupling Collector
- [#21](https://github.com/bobmatnyc/mcp-vector-search/issues/21) - Afferent Coupling Collector
- [#22](https://github.com/bobmatnyc/mcp-vector-search/issues/22) - Instability Index
- [#23](https://github.com/bobmatnyc/mcp-vector-search/issues/23) - Circular dependency detection
- [#24](https://github.com/bobmatnyc/mcp-vector-search/issues/24) - SQLite metrics store
- [#25](https://github.com/bobmatnyc/mcp-vector-search/issues/25) - Trend tracking
- [#26](https://github.com/bobmatnyc/mcp-vector-search/issues/26) - LCOM4 cohesion metric

### Epic 4: Visualization Export (#28-#33)
- [#28](https://github.com/bobmatnyc/mcp-vector-search/issues/28) - Visualization JSON schema
- [#29](https://github.com/bobmatnyc/mcp-vector-search/issues/29) - JSON exporter
- [#30](https://github.com/bobmatnyc/mcp-vector-search/issues/30) - HTML standalone report
- [#31](https://github.com/bobmatnyc/mcp-vector-search/issues/31) - Halstead metrics collector
- [#32](https://github.com/bobmatnyc/mcp-vector-search/issues/32) - Technical debt estimation
- [#33](https://github.com/bobmatnyc/mcp-vector-search/issues/33) - Status command with metrics

### Epic 5: Search Integration (#35-#37)
- [#35](https://github.com/bobmatnyc/mcp-vector-search/issues/35) - Quality filters for search
- [#36](https://github.com/bobmatnyc/mcp-vector-search/issues/36) - Quality-aware ranking
- [#37](https://github.com/bobmatnyc/mcp-vector-search/issues/37) - MCP tool exposure

## Complexity Grades

| Grade | Cognitive Complexity | Color | Quality |
|-------|---------------------|-------|---------|
| A | ≤5 | Green | Excellent |
| B | ≤10 | Green | Good |
| C | ≤15 | Yellow | Acceptable |
| D | ≤25 | Orange | Concerning |
| F | >25 | Red | Poor |

## Metric Thresholds

| Metric | Threshold | Type |
|--------|-----------|------|
| Cognitive Complexity | ≤15 | Quality Gate |
| Cyclomatic Complexity | ≤10 | Display Only |
| Max Nesting Depth | ≤4 | Quality Gate |
| Function Length | ≤30 lines | Warning |
| Parameter Count | ≤5 | Warning |
| Method Count | ≤20 | Warning |
| Efferent Coupling (Ce) | ≤20 | Warning |
| LCOM4 Cohesion | ≤2 | Warning |

## Code Smells

| Smell | Detection Rule | Severity | Minutes |
|-------|---------------|----------|---------|
| long_method | LOC > 30 OR CC > 15 | High | 20 |
| deep_nesting | max_depth > 4 | High | 15 |
| long_parameter_list | param_count > 5 | High | 10 |
| god_class | methods > 20 AND loc > 500 | Medium | 120 |
| large_class | LOC > 500 | High | - |
| empty_catch | catch with pass/empty | High | 5 |
| magic_numbers | literals outside whitelist | Medium | 5 |

## CLI Commands (Planned)

```bash
# Quick analysis (Tier 1-2 metrics)
mcp-vector-search analyze --quick

# Full analysis (all tiers)
mcp-vector-search analyze

# Specific targets
mcp-vector-search analyze --file src/auth.py
mcp-vector-search analyze --directory src/core/

# Output formats
mcp-vector-search analyze --output json > metrics.json
mcp-vector-search analyze --output sarif > results.sarif
mcp-vector-search analyze --output html > report.html

# Quality gates
mcp-vector-search analyze --fail-on-smell
mcp-vector-search analyze --max-complexity 15
mcp-vector-search analyze --quality-gate strict

# Diff-aware (for CI)
mcp-vector-search analyze --changed-only
mcp-vector-search analyze --baseline main

# Visualization export
mcp-vector-search analyze --export-viz ./viz-data/

# Status with metrics
mcp-vector-search status --metrics
```

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| Total Overhead (Tier 1-2) | <10ms per 1000 LOC | During indexing |
| Cognitive Complexity | <1ms per 1000 LOC | Tier 1 |
| Cyclomatic Complexity | <1ms per 1000 LOC | Tier 1 |
| Nesting Depth | <0.5ms per 1000 LOC | Tier 1 |
| Efferent Coupling | <2ms per file | Tier 2 |
| LCOM4 | ~5ms per class | Tier 3 |

## Module Structure

```
src/mcp_vector_search/
├── analysis/
│   ├── metrics.py              # Dataclasses
│   ├── collectors/
│   │   ├── base.py             # Abstract collector
│   │   ├── complexity.py       # CC & cognitive
│   │   ├── coupling.py         # Efferent/afferent
│   │   ├── cohesion.py         # LCOM4
│   │   ├── smells.py           # Smell detection
│   │   └── halstead.py         # Halstead metrics
│   ├── reporters/
│   │   ├── console.py          # Rich terminal
│   │   ├── json.py             # JSON export
│   │   ├── sarif.py            # SARIF for CI
│   │   └── html.py             # Standalone report
│   ├── storage/
│   │   └── metrics_db.py       # SQLite store
│   ├── thresholds.py           # Configuration
│   └── debt.py                 # Tech debt calc
└── cli/commands/analyze.py     # CLI command
```

## Database Tables

- **file_metrics** - Current file-level metrics
- **project_snapshots** - Historical project metrics
- **code_smells** - Individual smell instances
- **circular_dependencies** - Detected cycles

## Exit Codes

- **0** - Success (clean or within thresholds)
- **1** - Quality gate failure
- **2** - Error (invalid config, file not found)

## Key Dependencies

- **SonarQube** - Cognitive complexity algorithm
- **Tree-sitter** - AST parsing (existing)
- **ChromaDB** - Vector storage (existing)
- **SQLite** - Metrics storage (new)
- **Rich** - Terminal formatting
- **Pydantic** - Schema validation
- **Jinja2** - HTML templates
- **Chart.js/D3.js** - Visualizations

---

**Quick Actions**:
- View all issues: https://github.com/bobmatnyc/mcp-vector-search/issues
- View design doc: `docs/research/mcp-vector-search-structural-analysis-design.md`
- View full summary: `docs/summaries/structural-analysis-github-issues-summary.md`
