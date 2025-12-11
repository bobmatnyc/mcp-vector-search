# Phase 4: Visualization Export - Implementation Plan

**Research Date**: 2025-12-11
**Author**: Research Agent
**Epic**: [#27 - Visualization Export](https://github.com/bobmatnyc/mcp-vector-search/issues/27)
**Milestone**: v0.20.0 (Due: January 13, 2025)

## Executive Summary

Phase 4 adds comprehensive visualization and export capabilities to mcp-vector-search. This phase builds on the completed foundation (Phases 1-3) and delivers JSON/HTML exports, Halstead complexity metrics, technical debt estimation, and enhanced CLI reporting.

**Key Deliverables**:
- JSON export schema and exporter for external visualization tools
- Standalone HTML reports with interactive charts
- Halstead complexity metrics (operators, operands, volume, difficulty, effort)
- Technical debt estimation using SonarQube-based remediation time model
- Enhanced `status --metrics` command with comprehensive project health summary

**Critical Path**: Issues #28 → #29 → #30 (schema → exporter → HTML report)

**Estimated Duration**: 6 days (January 7-13, 2025)

**Complexity Assessment**: Medium - Well-defined requirements, existing patterns to follow

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Issue Breakdown](#issue-breakdown)
3. [Dependency Analysis](#dependency-analysis)
4. [Halstead Metrics Research](#halstead-metrics-research)
5. [Technical Debt Estimation Research](#technical-debt-estimation-research)
6. [Implementation Strategy](#implementation-strategy)
7. [Risk Assessment](#risk-assessment)
8. [Testing Strategy](#testing-strategy)
9. [References](#references)

---

## Current State Analysis

### Existing Infrastructure (Phases 1-3 Complete)

**Metrics System** (`src/mcp_vector_search/analysis/metrics.py`):
- ✅ `ChunkMetrics`: Per-function/class metrics with complexity grades
- ✅ `FileMetrics`: Aggregated file-level metrics
- ✅ `ProjectMetrics`: Project-wide aggregates with hotspot identification
- ✅ `CouplingMetrics`: Dependency tracking and instability calculation

**Collectors** (`src/mcp_vector_search/analysis/collectors/`):
- ✅ Cognitive complexity, cyclomatic complexity, nesting depth
- ✅ Parameter count, method count
- ✅ Efferent/afferent coupling, instability calculator
- ✅ LCOM4 cohesion metrics
- ✅ Code smell detection

**Reporters** (`src/mcp_vector_search/analysis/reporters/`):
- ✅ `ConsoleReporter`: Rich terminal output with tables and visualizations
- ✅ `SARIFReporter`: SARIF 2.1.0 format for CI/CD integration

**Storage** (`src/mcp_vector_search/analysis/storage/`):
- ✅ SQLite metrics store with historical tracking
- ✅ Trend tracking for regression detection
- ✅ Baseline comparison system

**Missing Components** (Phase 4 Scope):
- ❌ JSON export schema (Pydantic models)
- ❌ JSON reporter
- ❌ HTML reporter with interactive charts
- ❌ Halstead metrics collector
- ❌ Technical debt estimation module
- ❌ `status --metrics` CLI command

---

## Issue Breakdown

### Issue #27: [EPIC] Visualization Export
**Type**: Epic (tracking issue)
**Dependencies**: None (container for Phase 4 issues)
**Effort**: N/A (planning/tracking only)

**Deliverables**:
- Track completion of all Phase 4 issues
- Validate milestone acceptance criteria
- Update documentation

---

### Issue #28: Design Visualization JSON Export Schema
**Type**: Documentation + Implementation
**Dependencies**: #2 (metric dataclasses) - ✅ COMPLETED
**Blocks**: #29 (JSON exporter)
**Estimated Effort**: 4-6 hours

**Deliverables**:
1. **Pydantic Models** (`analysis/visualizer/schemas.py`):
   - `MetadataSchema`: Project identification, timestamp, tool version
   - `SummarySchema`: Aggregate metrics (files, lines, complexity, debt)
   - `FileDetailSchema`: Per-file breakdown with nested function details
   - `DependencyGraphSchema`: Node/edge representation with cycle detection
   - `SmellLocationSchema`: Smell coordinates for heatmap
   - `TrendDataSchema`: Historical metrics for line charts
   - `VisualizationExport`: Root schema combining all sections

2. **JSON Schema Generation**:
   - Auto-generate JSON Schema from Pydantic models
   - Include type definitions, descriptions, examples
   - Validate against sample exports

3. **Documentation**:
   - Schema specification document
   - Field descriptions and validation rules
   - Example export files (small, medium, large projects)

**Acceptance Criteria**:
- ✅ All six schema sections defined with Pydantic
- ✅ JSON Schema generated and validated
- ✅ Example exports pass schema validation
- ✅ Documentation covers all fields

**Implementation Notes**:
- Use Pydantic v2 for schema validation
- Follow existing patterns from `metrics.py` dataclasses
- Include computed fields (e.g., debt_ratio, health_grade)
- Design for extensibility (additional metrics in future phases)

---

### Issue #29: Implement JSON Exporter
**Type**: Implementation
**Dependencies**: #28 (schema), #10 (CLI) - ✅ COMPLETED
**Blocks**: #30 (HTML report)
**Estimated Effort**: 6-8 hours

**Deliverables**:
1. **JSONReporter Class** (`analysis/reporters/json.py`):
   - Aggregate data from multiple sources:
     - ChromaDB chunks (chunk metrics)
     - SQLite metrics store (file/project history)
     - Coupling collectors (dependency graph)
     - Trend tracker (historical snapshots)
   - Transform to visualization schema format
   - Validate against Pydantic schema before export
   - Pretty-print JSON with 2-space indentation

2. **CLI Integration**:
   - Add `--export-viz PATH` flag to `analyze` command
   - Add `--output json` option for format selection
   - Support both absolute and relative output paths

3. **Performance Optimization**:
   - Lazy loading of historical data
   - Streaming JSON write for large projects
   - Memory-efficient aggregation (chunked processing)

**Acceptance Criteria**:
- ✅ Schema compliance validated by Pydantic
- ✅ All metric sources included (complexity, coupling, smells, trends)
- ✅ Performance: <1 second for 1000-file projects
- ✅ Output size: <10MB for large projects
- ✅ CLI flags functional and documented

**Testing Requirements**:
- Small project export (5 files, ~100 LOC)
- Large project export (500 files, ~50K LOC)
- Schema validation (valid/invalid exports)
- All sections populated correctly
- Edge cases: empty project, no history, no smells

**Implementation Notes**:
- Follow pattern from `SARIFReporter.generate_sarif()`
- Use `ProjectMetrics.to_summary()` as starting point
- Query SQLite for historical trends
- Build dependency graph from coupling data
- Include smell locations for heatmap rendering

---

### Issue #30: Create HTML Standalone Report
**Type**: Implementation
**Dependencies**: #29 (JSON exporter) - REQUIRED
**Estimated Effort**: 10-12 hours

**Deliverables**:
1. **HTMLReporter Class** (`analysis/reporters/html.py`):
   - Consume JSON export from JSONReporter
   - Generate self-contained HTML with Jinja2 templates
   - Embed Chart.js or D3.js (inline, no CDN)
   - Implement six chart types:
     - Complexity heatmap (treemap or grid)
     - Dependency graph (force-directed layout)
     - Complexity distribution (histogram)
     - File size vs complexity (scatter/bubble chart)
     - Trend lines (time series)
     - Coupling chord diagram (circular layout)

2. **Interactive Features**:
   - Tabbed navigation (Summary, Files, Dependencies, Trends)
   - Dark mode toggle
   - Tooltips on hover (file details, metric values)
   - Click to drill down (file → functions)
   - Zoom/pan on large graphs

3. **Responsive Design**:
   - Mobile-friendly layout
   - Print-friendly styles (charts rendered)
   - Accessibility: ARIA labels, keyboard navigation

4. **CLI Integration**:
   - Add `--output html` flag to `analyze` command
   - Support custom output path: `--export-viz report.html`

**Acceptance Criteria**:
- ✅ Single HTML file (no external dependencies)
- ✅ All six visualizations render correctly
- ✅ Interactive features functional (hover, click, zoom)
- ✅ Mobile-responsive design
- ✅ File size <5MB (embedded chart library)
- ✅ Load time <2 seconds on modern browsers

**Testing Requirements**:
- Visual regression tests (screenshot comparison)
- Cross-browser testing (Chrome, Firefox, Safari)
- Mobile device testing (responsive layout)
- Performance testing (large datasets)
- Accessibility audit (WCAG 2.1 AA compliance)

**Implementation Notes**:
- Use Chart.js (smaller than D3.js, easier to embed)
- Jinja2 template: `templates/html_report.jinja2`
- Inline CSS (Tailwind CSS or minimal custom styles)
- Base64-encode chart library for embedding
- Progressive enhancement: core content visible without JS

**Technology Stack Decision**:
- **Chart.js** (Recommended):
  - Size: ~200KB minified
  - Pros: Simpler API, responsive by default, good for standard charts
  - Cons: Less flexible for custom visualizations (dependency graph)
- **D3.js** (Alternative):
  - Size: ~300KB minified
  - Pros: Maximum flexibility, better for dependency graph
  - Cons: Steeper learning curve, more code required
- **Recommendation**: Start with Chart.js for 5 charts, use D3.js only for dependency graph

---

### Issue #31: Implement Halstead Metrics Collector
**Type**: Implementation (Tier 2 Collector)
**Dependencies**: #2 (dataclasses), #8 (integrator) - ✅ COMPLETED
**Estimated Effort**: 8-10 hours

**Deliverables**:
1. **HalsteadCollector Class** (`analysis/collectors/halstead.py`):
   - Extend `MetricCollector` base class
   - Track operators and operands during AST traversal
   - Calculate seven Halstead metrics:
     - n1: Unique operators
     - n2: Unique operands
     - N1: Total operators
     - N2: Total operands
     - Vocabulary (n = n1 + n2)
     - Program Length (N = N1 + N2)
     - Volume (V = N × log₂(n))
     - Difficulty (D = (n1/2) × (N2/n2))
     - Effort (E = D × V)

2. **Operator/Operand Classification**:
   - Python operators: `+, -, *, /, //, %, **, &, |, ^, ~, <<, >>, ==, !=, <, >, <=, >=, and, or, not, in, is, if, for, while, def, class, return, yield, raise, try, except, finally, with, import, from, as, lambda`
   - Python operands: Variables, literals, function names, class names
   - Language-specific mappings for JavaScript, TypeScript, etc.

3. **Integration**:
   - Add `HalsteadMetrics` dataclass to `metrics.py`
   - Update `ChunkMetrics` to include Halstead scores
   - Integrate into indexer workflow

**Acceptance Criteria**:
- ✅ Accurate calculations validated against reference implementations
- ✅ Difficulty threshold: ≤30 for typical code
- ✅ Performance: <2ms per file
- ✅ Correlation with bug density (historical validation)

**Testing Requirements**:
- Simple function: low complexity (n1=5, n2=3, V<100)
- Complex function: high complexity (n1>20, n2>10, V>500)
- Edge cases: empty function, single statement
- Cross-language: Python, JavaScript, TypeScript
- Validate Volume and Difficulty calculations with known examples

**Implementation Notes**:
- Study existing collectors (`complexity.py`, `coupling.py`)
- Reference implementation: [wily library](https://github.com/tonybaloney/wily)
- Operator identification: Use tree-sitter node types
- Operand extraction: Variable names, literals from AST
- Vocabulary size: Use sets for unique counting

**Halstead Formulas** (Reference):
```python
# Basic counts
n1 = len(unique_operators)
n2 = len(unique_operands)
N1 = sum(operator_counts.values())
N2 = sum(operand_counts.values())

# Derived metrics
vocabulary = n1 + n2
length = N1 + N2
volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
effort = difficulty * volume
```

---

### Issue #32: Add Technical Debt Estimation
**Type**: Implementation
**Dependencies**: #14 (smell detection), #24 (SQLite) - ✅ COMPLETED
**Estimated Effort**: 4-6 hours

**Deliverables**:
1. **Debt Estimation Module** (`analysis/debt.py`):
   - Define remediation time constants (SonarQube-based):
     - Long Method: 20 minutes
     - Deep Nesting: 15 minutes
     - Long Parameter List: 10 minutes
     - God Class: 120 minutes (2 hours)
     - Empty Catch: 5 minutes
     - Magic Numbers: 5 minutes
     - High Cognitive Complexity: 30 minutes
     - Circular Dependency: 60 minutes (1 hour)
   - Apply severity scaling:
     - Error-level smells: × 1.5
     - Warning-level smells: × 1.0
     - Info-level smells: × 0.5
   - Aggregate at file and project levels

2. **Integration Points**:
   - Add `technical_debt_hours` to `FileMetrics`
   - Add `total_debt_hours`, `debt_ratio` to `ProjectMetrics`
   - Include in summary output (console, JSON, HTML)
   - Track debt reduction over time (trend tracking)

3. **Reporting Enhancements**:
   - Show debt breakdown by smell type
   - Highlight high-debt files (>2 hours)
   - Compare debt across baseline/current
   - Display debt reduction percentage

**Acceptance Criteria**:
- ✅ Estimates match developer intuition (survey validation)
- ✅ Times configurable via config file
- ✅ Severity scaling applied correctly
- ✅ Breakdown by smell type displayed
- ✅ Debt reduction tracked over time

**Testing Requirements**:
- File with multiple smells: verify total debt calculation
- Severity scaling: error vs warning vs info
- Zero debt: file with no smells
- Debt aggregation: project-level sum
- Trend validation: debt reduction over commits

**Implementation Notes**:
- Remediation times from SonarQube research (see Sources below)
- Times are conservative estimates (novice developer speed)
- Configurable via `config/thresholds.py` or `pyproject.toml`
- Display format: "4h 30m" (human-readable), store as minutes

**Technical Debt Formula**:
```python
def estimate_debt_hours(smells: list[CodeSmell]) -> float:
    """Calculate technical debt in hours."""
    total_minutes = 0
    for smell in smells:
        base_time = REMEDIATION_TIME[smell.name]
        severity_multiplier = SEVERITY_MULTIPLIER[smell.severity]
        total_minutes += base_time * severity_multiplier
    return total_minutes / 60  # Convert to hours
```

**Debt Ratio** (SonarQube-style):
```python
debt_ratio = debt_hours / (development_cost_per_line * total_lines)
# Where development_cost_per_line = 0.5 hours (30 minutes)
```

---

### Issue #33: Extend `status` Command with Metrics
**Type**: Implementation (CLI Enhancement)
**Dependencies**: #10 (CLI), #24 (SQLite) - ✅ COMPLETED
**Estimated Effort**: 4-6 hours

**Deliverables**:
1. **Enhanced Status Command** (`cli/commands/status.py`):
   - Add `--metrics` flag to existing `status` command
   - Display comprehensive project health summary:
     - Files breakdown by language (Python: 150, JS: 80, etc.)
     - Lines breakdown: Code (10K), Comments (2K), Blank (1.5K)
     - Complexity distribution: A-F grades with ASCII bar chart
     - Code smells summary: Count by type and severity
     - Coupling metrics: Avg instability, circular dependencies
     - Technical debt: Total hours, top debtors
   - Add `--verbose` flag for detailed output
   - Use Rich library for colored formatting

2. **Performance Optimization**:
   - Cache metrics in SQLite (avoid full re-analysis)
   - Refresh cache only if files changed (git status check)
   - Performance target: <500ms for cached results

3. **Output Format**:
   - Fit in standard terminal (24 lines maximum)
   - Color coding: Green (good), Yellow (warning), Red (critical)
   - Summary statistics above the fold
   - Details below or via `--verbose`

4. **Graceful Degradation**:
   - Handle missing metrics (no analysis run yet)
   - Prompt user to run `analyze --quick` first
   - Show partial results if some data unavailable

**Acceptance Criteria**:
- ✅ Command `mcp-vector-search status --metrics` displays summary
- ✅ Output fits in standard terminal (24 lines)
- ✅ Performance <500ms (cached)
- ✅ Graceful handling when metrics unavailable

**Testing Requirements**:
- Fresh project (no metrics): prompt to analyze
- Analyzed project: full summary displayed
- Large project (500 files): performance check
- Verbose mode: detailed output
- Terminal width handling: 80/120/160 columns

**Implementation Notes**:
- Query SQLite for latest project snapshot
- Use Rich `Table` for formatted output
- ASCII bar chart: `█` × percentage (scale: 5% = 1 char)
- Check cache freshness: compare file mtimes with last analysis

**Output Format Example**:
```
📊 Project Health Summary

Files: 234 (Python: 180, JS: 40, TS: 14)
Lines: 45,320 (Code: 32,100, Comments: 8,200, Blank: 5,020)

Complexity Distribution:
  A (0-5)    ████████████████████ 45% (102 functions)
  B (6-10)   ████████████ 30% (68)
  C (11-20)  ████████ 18% (41)
  D (21-30)  ████ 5% (11)
  F (31+)    ██ 2% (4) ⚠️

Code Smells: 23 (Errors: 3, Warnings: 15, Info: 5)
  Long Method: 12
  Deep Nesting: 6
  High Complexity: 5

Coupling: Avg Instability 0.42 (Stable ✓)
Tech Debt: 8.5 hours (Top: auth.py 2.5h)

Last Analysis: 2025-01-10 14:23:15
```

---

## Dependency Analysis

### Dependency Graph

```
#27 (Epic) ─┬─> #28 (JSON Schema) ──> #29 (JSON Exporter) ──> #30 (HTML Report)
            │
            ├─> #31 (Halstead Collector) ──> [independent]
            │
            ├─> #32 (Tech Debt Estimation) ──> [independent]
            │
            └─> #33 (status --metrics) ──> [independent]
```

### Critical Path

**Longest chain**: #28 → #29 → #30 (Schema → Exporter → HTML)

**Parallelizable**:
- #31 (Halstead) can start immediately
- #32 (Tech Debt) can start immediately
- #33 (status command) can start immediately after #24 (done)

### Blocking Relationships

| Issue | Blocked By | Blocks | Can Start? |
|-------|-----------|---------|------------|
| #28   | None      | #29     | ✅ Yes (Day 1) |
| #29   | #28       | #30     | Day 2-3 |
| #30   | #29       | None    | Day 4-5 |
| #31   | None      | None    | ✅ Yes (Day 1) |
| #32   | None      | None    | ✅ Yes (Day 1) |
| #33   | None      | None    | ✅ Yes (Day 1) |

### Recommended Implementation Order

**Week 1 (Jan 7-13, 2025)**:

**Day 1-2** (Parallel):
- Start #28 (JSON Schema) - Foundation for exporters
- Start #31 (Halstead) - Independent collector
- Start #32 (Tech Debt) - Independent estimator

**Day 3** (After #28):
- Start #29 (JSON Exporter) - Depends on schema

**Day 4** (Parallel):
- Continue #29 (JSON Exporter)
- Start #33 (status command) - Independent

**Day 5-6** (After #29):
- Start #30 (HTML Report) - Depends on JSON exporter
- Finalize #33 (status command)
- Integration testing

**Day 6** (Final):
- Integration testing (all features together)
- Documentation updates
- Milestone validation

---

## Halstead Metrics Research

### Theory and Background

Halstead complexity metrics, developed by Maurice Halstead in 1977, measure code complexity through operators and operands analysis. These metrics provide quantitative measures of code volume, difficulty, and estimated effort.

**Core Concept**: Treat code as a vocabulary of operators (actions) and operands (data), then derive complexity from their usage patterns.

### Basic Measures

| Metric | Symbol | Definition |
|--------|--------|------------|
| Unique Operators | n1 | Count of distinct operator types |
| Unique Operands | n2 | Count of distinct operand types |
| Total Operators | N1 | Sum of all operator occurrences |
| Total Operands | N2 | Sum of all operand occurrences |

### Derived Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Vocabulary | n = n1 + n2 | Size of the "language" used |
| Length | N = N1 + N2 | Total tokens in program |
| Volume | V = N × log₂(n) | Information content (bits) |
| Difficulty | D = (n1/2) × (N2/n2) | How hard to write/understand |
| Effort | E = D × V | Mental effort required |
| Time | T = E / 18 | Estimated seconds to code |
| Bugs | B = (E^(2/3)) / 3000 | Predicted bug count |

### Operator/Operand Classification (Python)

**Operators** (Examples):
- Arithmetic: `+, -, *, /, //, %, **`
- Logical: `and, or, not, in, is`
- Comparison: `==, !=, <, >, <=, >=`
- Bitwise: `&, |, ^, ~, <<, >>`
- Control flow: `if, for, while, try, except, with`
- Function definition: `def, lambda, return, yield`
- Class definition: `class`
- Import: `import, from, as`

**Operands** (Examples):
- Variables: `x, user_name, counter`
- Literals: `42, "hello", 3.14, True, None`
- Function names: `print, calculate_sum`
- Class names: `MyClass, UserModel`

### Implementation Approach

**Tree-sitter Strategy**:
1. **Traverse AST**: Visit each node in the syntax tree
2. **Classify Nodes**: Determine if node is operator or operand
3. **Track Counts**: Maintain sets (unique) and counters (total)
4. **Calculate Metrics**: Apply formulas at function/file level

**Node Type Mapping** (Python):
```python
OPERATOR_NODES = {
    'binary_operator': ['+', '-', '*', '/', '==', '!=', ...],
    'unary_operator': ['-', 'not', '~'],
    'boolean_operator': ['and', 'or'],
    'comparison_operator': ['<', '>', '<=', '>=', 'in', 'is'],
    'if_statement': ['if'],
    'for_statement': ['for'],
    'while_statement': ['while'],
    'function_definition': ['def'],
    'class_definition': ['class'],
    # ... more node types
}

OPERAND_NODES = {
    'identifier': True,  # Variable names
    'integer': True,     # Numeric literals
    'string': True,      # String literals
    'true': True,        # Boolean literals
    'false': True,
    'none': True,
}
```

### Reference Implementations

**Wily** (Python): https://github.com/tonybaloney/wily
- Comprehensive Python code complexity tool
- Supports Halstead, Cyclomatic, Maintainability Index
- Git integration for historical tracking
- Example of production-quality implementation

**Radon** (Python): https://github.com/rubik/radon
- Static analysis tool for Python
- Includes Halstead metrics, raw metrics, MI
- Lightweight, pure Python
- Good reference for metric calculations

### Validation Strategy

**Test Cases**:
1. **Simple Function**: Verify low complexity
   ```python
   def add(a, b):
       return a + b
   # Expected: n1=2, n2=3, V≈12, D≈1.5
   ```

2. **Complex Function**: Verify high complexity
   ```python
   def process_data(items, filter_fn, transform_fn):
       results = []
       for item in items:
           if filter_fn(item):
               results.append(transform_fn(item))
       return results
   # Expected: n1≈15, n2≈8, V≈100, D≈15
   ```

3. **Cross-Validation**: Compare with SonarQube/Radon
   - Run same code through multiple tools
   - Verify metrics match within 5%

**Thresholds** (Industry Standards):
- Volume: <500 (simple), 500-1000 (moderate), >1000 (complex)
- Difficulty: <10 (easy), 10-30 (moderate), >30 (hard)
- Effort: <10,000 (quick), 10K-50K (moderate), >50K (significant)

### Integration with Existing Metrics

**Correlation Analysis**:
- Halstead Volume ↔ Lines of Code (expect 0.8+ correlation)
- Halstead Difficulty ↔ Cyclomatic Complexity (0.6+ correlation)
- Halstead Effort ↔ Bug Density (historical validation)

**Combined Scoring**:
```python
def compute_complexity_score(chunk: ChunkMetrics) -> float:
    """Combine multiple metrics into unified score."""
    cognitive_weight = 0.4
    cyclomatic_weight = 0.3
    halstead_weight = 0.3

    return (
        chunk.cognitive_complexity * cognitive_weight +
        chunk.cyclomatic_complexity * cyclomatic_weight +
        normalize_halstead(chunk.halstead_difficulty) * halstead_weight
    )
```

---

## Technical Debt Estimation Research

### SonarQube Model

SonarQube uses a remediation time-based approach to quantify technical debt. Each issue type has an associated "fix time" based on empirical developer studies.

**Core Formula**:
```
Technical Debt (hours) = Σ (issue_count × remediation_time × severity_multiplier)
```

**Debt Ratio** (Maintainability):
```
Debt Ratio = Remediation Cost / Development Cost
           = Debt Hours / (Cost Per Line × Total Lines)

# Default: Cost Per Line = 0.5 hours (30 minutes)
```

### Remediation Time Constants

Based on SonarQube research and developer surveys:

| Code Smell | Time (minutes) | Severity | Rationale |
|------------|----------------|----------|-----------|
| Long Method | 20 | Warning | Extract methods, refactor logic |
| Deep Nesting | 15 | Warning | Flatten structure, early returns |
| Long Parameter List | 10 | Info | Introduce parameter object |
| God Class | 120 | Error | Split responsibilities, extract classes |
| Empty Catch | 5 | Warning | Add proper error handling |
| Magic Numbers | 5 | Info | Extract constants |
| High Cognitive Complexity | 30 | Error | Simplify logic, extract helpers |
| Circular Dependency | 60 | Error | Refactor module structure |

### Severity Multipliers

**SonarQube Severity Levels**:
- **Blocker/Critical**: 1.5× (requires immediate attention)
- **Major**: 1.0× (standard remediation time)
- **Minor**: 0.5× (quick fixes)
- **Info**: 0.5× (optional improvements)

**Our Mapping**:
```python
SEVERITY_MULTIPLIER = {
    SmellSeverity.ERROR: 1.5,    # Critical issues
    SmellSeverity.WARNING: 1.0,  # Standard issues
    SmellSeverity.INFO: 0.5,     # Minor issues
}
```

### Accuracy and Validation

**Research Findings** (IEEE Study):
- SonarQube remediation times are generally **overestimated** by 30-70%
- **Code smells** have most accurate estimates (within 20%)
- **Bugs** and **vulnerabilities** have less accurate estimates (50%+ error)
- **Novice developers** take 2-3× longer than estimates
- **Expert developers** are 20-30% faster than estimates

**Implications**:
- Our estimates should be **conservative** (assume novice developers)
- Provide **configurable times** for team-specific calibration
- Track **actual remediation time** to improve estimates over iterations

### Aggregation Strategy

**File-Level Debt**:
```python
def estimate_file_debt(file_metrics: FileMetrics) -> float:
    """Calculate technical debt for a single file."""
    total_minutes = 0
    for chunk in file_metrics.chunks:
        for smell in chunk.smells:
            base_time = REMEDIATION_TIME[smell.name]
            multiplier = SEVERITY_MULTIPLIER[smell.severity]
            total_minutes += base_time * multiplier
    return total_minutes / 60  # Hours
```

**Project-Level Debt**:
```python
def estimate_project_debt(project: ProjectMetrics) -> dict:
    """Calculate project-wide technical debt."""
    total_hours = 0
    debt_by_type = defaultdict(float)

    for file_metrics in project.files.values():
        file_debt = estimate_file_debt(file_metrics)
        total_hours += file_debt

        # Track breakdown by smell type
        for chunk in file_metrics.chunks:
            for smell in chunk.smells:
                time_hours = REMEDIATION_TIME[smell.name] / 60
                debt_by_type[smell.name] += time_hours

    return {
        'total_hours': total_hours,
        'total_days': total_hours / 8,  # 8-hour workdays
        'breakdown': dict(debt_by_type),
        'debt_ratio': calculate_debt_ratio(total_hours, project.total_lines)
    }
```

### Trend Tracking

**Debt Reduction Over Time**:
```python
def calculate_debt_trend(project_id: int, days: int = 30) -> list:
    """Calculate debt trend over last N days."""
    snapshots = metrics_store.get_project_history(project_id, days)

    trend = []
    for snapshot in snapshots:
        debt_hours = estimate_project_debt(snapshot)['total_hours']
        trend.append({
            'timestamp': snapshot.analyzed_at,
            'debt_hours': debt_hours,
            'debt_change': debt_hours - trend[-1]['debt_hours'] if trend else 0
        })

    return trend
```

### Reporting Formats

**Console Output**:
```
Technical Debt Summary

Total Debt: 12.5 hours (1.6 days)
Debt Ratio: 0.08 (8% of codebase)

Breakdown by Type:
  Long Method         6.5h (52%)  ████████████████████
  God Class           4.0h (32%)  ████████████
  Deep Nesting        1.5h (12%)  ████
  High Complexity     0.5h (4%)   ██

Top Debtors:
  src/auth/handler.py    2.5h  ⚠️
  src/api/routes.py      1.8h
  src/db/models.py       1.2h

Trend (Last 7 Days): ↓ 15% (-2.2h)
```

**JSON Export**:
```json
{
  "technical_debt": {
    "total_hours": 12.5,
    "total_days": 1.56,
    "debt_ratio": 0.08,
    "breakdown": {
      "Long Method": 6.5,
      "God Class": 4.0,
      "Deep Nesting": 1.5,
      "High Complexity": 0.5
    },
    "top_files": [
      {"path": "src/auth/handler.py", "debt_hours": 2.5},
      {"path": "src/api/routes.py", "debt_hours": 1.8}
    ],
    "trend": [
      {"date": "2025-01-10", "hours": 12.5, "change": -0.5},
      {"date": "2025-01-09", "hours": 13.0, "change": -0.3}
    ]
  }
}
```

### Configuration

**User-Configurable Times** (`pyproject.toml`):
```toml
[tool.mcp-vector-search.debt]
# Remediation times in minutes
long_method = 20
deep_nesting = 15
long_parameter_list = 10
god_class = 120
empty_catch = 5
magic_numbers = 5
high_cognitive_complexity = 30
circular_dependency = 60

# Severity multipliers
error_multiplier = 1.5
warning_multiplier = 1.0
info_multiplier = 0.5

# Cost assumptions
development_cost_per_line = 0.5  # hours
hours_per_workday = 8
```

---

## Implementation Strategy

### Phase Sequencing

**Week Structure** (6 working days):

**Day 1: Foundations**
- Morning: #28 (JSON Schema) - 4 hours
  - Design Pydantic models
  - Generate JSON Schema
  - Create example exports
- Afternoon: Start #31 (Halstead) - 4 hours
  - Implement basic collector
  - Operator/operand classification

**Day 2: Parallel Development**
- Morning: Complete #31 (Halstead) - 4 hours
  - Finish collector implementation
  - Write unit tests
  - Integrate with metrics system
- Afternoon: Start #32 (Tech Debt) - 4 hours
  - Implement debt estimation module
  - Define remediation time constants

**Day 3: Exporters and CLI**
- Morning: Complete #32 (Tech Debt) - 2 hours
  - Finish integration
  - Write tests
- Morning/Afternoon: Start #29 (JSON Exporter) - 6 hours
  - Implement JSONReporter
  - Data aggregation from multiple sources

**Day 4: JSON and Status**
- Morning: Complete #29 (JSON Exporter) - 4 hours
  - CLI integration
  - Performance optimization
  - Testing
- Afternoon: Start #33 (status --metrics) - 4 hours
  - Implement CLI command
  - Design output format

**Day 5: HTML Report (Part 1)**
- Full day: Start #30 (HTML Report) - 8 hours
  - Design template structure
  - Implement 3 of 6 charts (complexity heatmap, histogram, trends)
  - Basic interactivity

**Day 6: HTML Report (Part 2) + Integration**
- Morning: Complete #30 (HTML Report) - 4 hours
  - Implement remaining 3 charts (dependency graph, scatter, chord)
  - Polish UI/UX
- Afternoon: Integration Testing - 4 hours
  - End-to-end validation
  - Performance testing
  - Documentation updates

### Parallel Work Streams

**Stream A: Exporters** (Critical Path)
- Day 1: #28 (Schema)
- Day 3-4: #29 (JSON Exporter)
- Day 5-6: #30 (HTML Report)

**Stream B: Metrics** (Independent)
- Day 1-2: #31 (Halstead)
- Day 2-3: #32 (Tech Debt)

**Stream C: CLI** (Independent)
- Day 4: #33 (status command)

### Testing Strategy per Issue

**#28 (JSON Schema)**:
- Unit: Pydantic validation (valid/invalid inputs)
- Integration: Schema generation, example exports
- Documentation: Field coverage, examples

**#29 (JSON Exporter)**:
- Unit: Data aggregation, schema compliance
- Integration: CLI flags, file I/O
- Performance: 1000-file project <1 second
- E2E: Export → validate → reimport

**#30 (HTML Report)**:
- Unit: Template rendering, data transformation
- Visual: Screenshot comparisons (pytest-regressions)
- Cross-browser: Chrome, Firefox, Safari
- Accessibility: WCAG 2.1 AA (pa11y)
- Performance: Load time <2 seconds

**#31 (Halstead)**:
- Unit: Operator/operand counting, formula calculations
- Integration: Collector + indexer workflow
- Validation: Cross-check with Radon/wily
- Cross-language: Python, JS, TS

**#32 (Tech Debt)**:
- Unit: Debt calculation, aggregation
- Integration: FileMetrics, ProjectMetrics
- Validation: Developer survey (does it match intuition?)
- Trend: Historical debt reduction

**#33 (status --metrics)**:
- Unit: Metric retrieval, formatting
- Integration: CLI parsing, Rich output
- Performance: <500ms cached
- UX: Terminal width handling, colors

---

## Risk Assessment

### High-Risk Areas

**1. HTML Report Size and Performance**
- **Risk**: Embedded Chart.js + large datasets = >5MB files
- **Mitigation**:
  - Use Chart.js (smaller than D3.js): ~200KB
  - Minify and gzip embedded libraries
  - Limit data points in charts (sample large datasets)
  - Progressive loading: render critical charts first
- **Contingency**: Split into multiple HTML files if size exceeds 10MB

**2. Halstead Metric Accuracy**
- **Risk**: Incorrect operator/operand classification → wrong metrics
- **Mitigation**:
  - Extensive unit tests with known examples
  - Cross-validate with Radon/wily on sample projects
  - Language-specific mappings (Python, JS, TS)
  - Reference SonarQube Halstead implementation
- **Contingency**: Mark Halstead as "experimental" in v0.20.0, stabilize in v0.21.0

**3. Technical Debt Estimation Accuracy**
- **Risk**: Remediation times don't match reality → misleading estimates
- **Mitigation**:
  - Use conservative times (assume novice developers)
  - Make times configurable via pyproject.toml
  - Show estimates as ranges ("2-4 hours") not precise values
  - Include disclaimer: "Estimates based on SonarQube research"
- **Contingency**: User feedback loop to calibrate times per-team

**4. JSON Export Schema Stability**
- **Risk**: Schema changes in v0.21+ break external visualizers
- **Mitigation**:
  - Version schema (v1.0.0) and include in exports
  - Document breaking vs non-breaking changes
  - Provide migration guide for schema updates
  - Use Pydantic for validation and schema evolution
- **Contingency**: Maintain backward compatibility for 1 major version

### Medium-Risk Areas

**5. Cross-Browser Compatibility (HTML Report)**
- **Risk**: Charts render incorrectly in Safari/Firefox
- **Mitigation**:
  - Test on Chrome, Firefox, Safari during development
  - Use Chart.js (better cross-browser support than D3.js)
  - Polyfills for older browsers (if needed)
- **Contingency**: Document supported browsers, warn on unsupported

**6. Performance with Large Projects**
- **Risk**: 10K+ file projects timeout or run slowly
- **Mitigation**:
  - Streaming JSON write (not all in memory)
  - Lazy loading of historical data
  - Chunked processing (files in batches of 100)
  - Performance benchmarks in tests
- **Contingency**: Add `--max-files` limit for large projects

**7. Dependency on Phases 1-3**
- **Risk**: Bugs in earlier phases block Phase 4
- **Mitigation**:
  - Validate Phase 1-3 completion before starting
  - Run full test suite as pre-flight check
  - Fix critical bugs before Phase 4 kickoff
- **Contingency**: Buffer time (1-2 days) for bug fixes

### Low-Risk Areas

**8. CLI Integration**
- **Risk**: Flag conflicts or poor UX
- **Mitigation**: Follow existing CLI patterns (Typer framework)
- **Impact**: Low (easy to fix)

**9. Documentation Updates**
- **Risk**: Outdated docs
- **Mitigation**: Update docs in same PR as features
- **Impact**: Low (no user-facing breakage)

---

## Testing Strategy

### Unit Tests

**Coverage Target**: 85% (maintain existing standard)

**Test Structure**:
```
tests/unit/analysis/
├── visualizer/
│   └── test_schemas.py          # Pydantic validation
├── reporters/
│   ├── test_json_reporter.py    # JSON export
│   └── test_html_reporter.py    # HTML generation
├── collectors/
│   └── test_halstead.py         # Halstead calculations
└── test_debt.py                 # Debt estimation
```

**Key Test Cases**:

**Schemas (#28)**:
```python
def test_visualization_export_schema_valid():
    """Valid export passes Pydantic validation."""
    export = VisualizationExport(
        metadata=MetadataSchema(...),
        summary=SummarySchema(...),
        # ... all fields
    )
    assert export.model_validate(export.model_dump())

def test_schema_handles_empty_project():
    """Schema works with zero files."""
    export = VisualizationExport(
        metadata=MetadataSchema(...),
        summary=SummarySchema(total_files=0),
        files=[]
    )
    assert export.summary.total_files == 0
```

**JSON Exporter (#29)**:
```python
def test_json_reporter_exports_all_sections():
    """All six schema sections included in export."""
    reporter = JSONReporter()
    project = create_sample_project()
    export = reporter.generate_export(project)

    assert 'metadata' in export
    assert 'summary' in export
    assert 'files' in export
    assert 'dependency_graph' in export
    assert 'smell_locations' in export
    assert 'trends' in export

def test_json_export_performance_large_project():
    """Export completes in <1 second for 1000 files."""
    project = create_large_project(files=1000)
    reporter = JSONReporter()

    start = time.time()
    reporter.export_to_file(project, "test.json")
    duration = time.time() - start

    assert duration < 1.0
```

**Halstead Collector (#31)**:
```python
def test_halstead_simple_function():
    """Simple function has low complexity."""
    code = "def add(a, b): return a + b"
    metrics = collect_halstead_metrics(code)

    assert metrics.n1 == 2  # 'def', '+'
    assert metrics.n2 == 3  # 'add', 'a', 'b'
    assert metrics.volume < 20
    assert metrics.difficulty < 2

def test_halstead_cross_validation_radon():
    """Metrics match Radon within 5%."""
    code = load_sample_file("complex_function.py")
    our_metrics = collect_halstead_metrics(code)
    radon_metrics = run_radon(code)

    assert abs(our_metrics.volume - radon_metrics.volume) / radon_metrics.volume < 0.05
```

**Tech Debt (#32)**:
```python
def test_debt_calculation_multiple_smells():
    """Debt aggregates correctly across smells."""
    smells = [
        CodeSmell(name="Long Method", severity=SmellSeverity.WARNING),
        CodeSmell(name="Deep Nesting", severity=SmellSeverity.WARNING),
    ]
    debt = estimate_debt_hours(smells)

    assert debt == (20 + 15) / 60  # 0.58 hours

def test_debt_severity_multiplier():
    """Error smells cost 1.5× base time."""
    smells = [
        CodeSmell(name="God Class", severity=SmellSeverity.ERROR)
    ]
    debt = estimate_debt_hours(smells)

    assert debt == (120 * 1.5) / 60  # 3.0 hours
```

### Integration Tests

**Test Structure**:
```
tests/integration/
├── test_analyze_with_export.py   # CLI end-to-end
├── test_html_report_generation.py
└── test_status_metrics.py
```

**Key Scenarios**:

```python
def test_analyze_export_viz_json():
    """analyze --export-viz generates valid JSON."""
    run_cli(["analyze", "--export-viz", "output.json"])

    assert Path("output.json").exists()
    export = json.loads(Path("output.json").read_text())

    # Validate against schema
    VisualizationExport.model_validate(export)

def test_analyze_export_viz_html():
    """analyze --export-viz output.html generates HTML report."""
    run_cli(["analyze", "--export-viz", "output.html"])

    assert Path("output.html").exists()
    html = Path("output.html").read_text()

    # Check for expected sections
    assert "<title>Code Analysis Report</title>" in html
    assert "Chart.js" in html or "<script>" in html
    assert "<canvas" in html  # Chart canvases

def test_status_metrics_command():
    """status --metrics displays summary."""
    # First analyze project
    run_cli(["analyze", "--quick"])

    # Then check status
    output = run_cli(["status", "--metrics"])

    assert "Project Health Summary" in output
    assert "Complexity Distribution" in output
    assert "Technical Debt" in output
```

### Visual Regression Tests

**HTML Report (#30)**:
```python
@pytest.mark.visual
def test_html_report_visual_regression(page, assert_snapshot):
    """HTML report matches baseline screenshots."""
    # Generate report
    reporter = HTMLReporter()
    reporter.export("test_report.html")

    # Load in browser
    page.goto(f"file://{Path('test_report.html').absolute()}")

    # Capture screenshots
    assert_snapshot(page.screenshot(), "html_report_full.png")
    assert_snapshot(page.locator("#complexity-chart").screenshot(), "complexity_chart.png")
```

**Tools**: Playwright + pytest-regressions

### Performance Benchmarks

**Targets**:
- JSON export: <1 second for 1000 files
- HTML generation: <2 seconds for 1000 files
- status --metrics: <500ms (cached)

**Benchmark Tests**:
```python
@pytest.mark.benchmark
def test_json_export_performance(benchmark):
    """JSON export performance benchmark."""
    project = load_large_project(files=1000)
    reporter = JSONReporter()

    result = benchmark(reporter.generate_export, project)

    assert benchmark.stats.mean < 1.0  # 1 second

@pytest.mark.benchmark
def test_halstead_collector_performance(benchmark):
    """Halstead collector <2ms per file."""
    code = load_sample_file("medium_file.py")

    result = benchmark(collect_halstead_metrics, code)

    assert benchmark.stats.mean < 0.002  # 2 milliseconds
```

### Acceptance Testing

**Manual Test Plan** (Before Release):

**Test Case 1: Small Project Export**
1. Create test project (5 files, ~200 LOC)
2. Run `mcp-vector-search analyze --export-viz report.json`
3. Verify JSON validates against schema
4. Run `mcp-vector-search analyze --export-viz report.html`
5. Open HTML in Chrome, Firefox, Safari
6. Verify all charts render correctly
7. Test interactions: hover, click, zoom

**Test Case 2: Large Project Export**
1. Clone large project (e.g., requests library, ~500 files)
2. Run analysis with export
3. Verify performance targets met
4. Verify HTML file size <5MB
5. Verify load time <2 seconds

**Test Case 3: Status Command**
1. Fresh project (no analysis)
2. Run `status --metrics` → expect prompt to analyze
3. Run `analyze --quick`
4. Run `status --metrics` → expect summary
5. Verify output fits in terminal (24 lines)
6. Verify color coding (green/yellow/red)

**Test Case 4: Debt Estimation**
1. Create file with known smells (1 Long Method, 1 God Class)
2. Run analysis
3. Verify debt calculation: (20 + 120) / 60 = 2.33 hours
4. Check summary output includes debt breakdown

**Test Case 5: Halstead Metrics**
1. Create simple function (known complexity)
2. Run analysis
3. Cross-validate with Radon: `radon hal simple.py`
4. Verify our metrics ≈ Radon metrics (±5%)

---

## References

### GitHub Issues

- [Issue #27: [EPIC] Visualization Export](https://github.com/bobmatnyc/mcp-vector-search/issues/27)
- [Issue #28: Design JSON Export Schema](https://github.com/bobmatnyc/mcp-vector-search/issues/28)
- [Issue #29: Implement JSON Exporter](https://github.com/bobmatnyc/mcp-vector-search/issues/29)
- [Issue #30: Create HTML Standalone Report](https://github.com/bobmatnyc/mcp-vector-search/issues/30)
- [Issue #31: Implement Halstead Metrics](https://github.com/bobmatnyc/mcp-vector-search/issues/31)
- [Issue #32: Add Technical Debt Estimation](https://github.com/bobmatnyc/mcp-vector-search/issues/32)
- [Issue #33: Extend status Command with Metrics](https://github.com/bobmatnyc/mcp-vector-search/issues/33)

### Project Documentation

- [Structural Code Analysis Project](https://github.com/bobmatnyc/mcp-vector-search/blob/main/docs/projects/structural-code-analysis.md)
- [Design Document](https://github.com/bobmatnyc/mcp-vector-search/blob/main/docs/research/mcp-vector-search-structural-analysis-design.md)
- [Project Board](https://github.com/users/bobmatnyc/projects/13)

### Technical References

**Halstead Metrics**:
- [Halstead Complexity Measures - Wikipedia](https://en.wikipedia.org/wiki/Halstead_complexity_measures)
- [Verifysoft → Halstead Metrics](https://www.verifysoft.com/en_halstead_metrics.html)
- [GeeksforGeeks: Halstead's Software Metrics](https://www.geeksforgeeks.org/software-engineering/software-engineering-halsteads-software-metrics/)
- [Measuring Python Complexity with Wily](https://stribny.name/blog/2019/05/measuring-python-code-complexity-with-wily/)

**Technical Debt Estimation**:
- [SonarQube: Understanding Measures and Metrics](https://docs.sonarsource.com/sonarqube-server/user-guide/code-metrics/metrics-definition)
- [IEEE: On the Accuracy of SonarQube Technical Debt Remediation Time](https://ieeexplore.ieee.org/document/8906700/)
- [ResearchGate: Technical Debt Prioritization with SonarQube](https://www.researchgate.net/publication/345632101_On_the_Technical_Debt_Prioritization_and_Cost_Estimation_with_SonarQube_tool)

**Visualization Libraries**:
- [Chart.js Documentation](https://www.chartjs.org/docs/latest/)
- [D3.js Documentation](https://d3js.org/)

**Standards**:
- [SARIF 2.1.0 Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## Appendices

### Appendix A: Estimated Effort Summary

| Issue | Description | Effort (hours) | Risk | Start Date |
|-------|-------------|----------------|------|------------|
| #27   | Epic (tracking) | 0 | Low | Jan 7 |
| #28   | JSON Schema | 4-6 | Low | Jan 7 |
| #29   | JSON Exporter | 6-8 | Medium | Jan 9 |
| #30   | HTML Report | 10-12 | Medium | Jan 11 |
| #31   | Halstead Collector | 8-10 | Medium | Jan 7 |
| #32   | Tech Debt Estimation | 4-6 | Low | Jan 8 |
| #33   | status --metrics | 4-6 | Low | Jan 10 |
| **Total** | **Phase 4** | **36-48** | - | **Jan 7-13** |

### Appendix B: Success Metrics

**Quantitative**:
- ✅ All 7 issues closed (Epic + 6 features)
- ✅ Test coverage ≥85% (maintain existing standard)
- ✅ Performance: JSON export <1s (1000 files)
- ✅ Performance: HTML report <2s (1000 files)
- ✅ Performance: status command <500ms (cached)
- ✅ HTML file size <5MB
- ✅ Zero P0/P1 bugs at release

**Qualitative**:
- ✅ Visualizations render correctly on Chrome, Firefox, Safari
- ✅ HTML report is mobile-responsive
- ✅ Technical debt estimates match developer intuition
- ✅ Halstead metrics cross-validate with Radon (±5%)
- ✅ JSON schema is well-documented
- ✅ status command output is readable and actionable

### Appendix C: Recommended Start

**Issue to Start First**: #28 (JSON Export Schema)

**Rationale**:
1. **Foundation**: Schema defines data structures for all exporters
2. **No Blockers**: Only depends on #2 (dataclasses), which is complete
3. **Critical Path**: Blocks #29 (JSON Exporter) → #30 (HTML Report)
4. **Parallel Work**: While #28 is in progress, start #31 (Halstead) and #32 (Tech Debt)
5. **Clear Deliverable**: Well-defined output (Pydantic models + docs)

**First Day Checklist**:
- [ ] Review Phase 1-3 completion (run test suite)
- [ ] Create feature branch: `feature/28-json-export-schema`
- [ ] Set up module: `src/mcp_vector_search/analysis/visualizer/schemas.py`
- [ ] Define base schemas: MetadataSchema, SummarySchema
- [ ] Generate JSON Schema from Pydantic models
- [ ] Create example exports (small, medium, large)
- [ ] Write unit tests for schema validation
- [ ] Document schema in docs/reference/
- [ ] Submit PR for review

---

**Document Version**: 1.0
**Last Updated**: 2025-12-11
**Next Review**: 2025-01-06 (before Phase 4 kickoff)
