# Structural Code Analysis Design Document

## MCP Vector Search — `analyze` Command & Metrics Integration

**Version**: 1.0
**Status**: Draft
**Last Updated**: December 2024

---

## Overview

### Problem Statement

Developers searching for code need more than semantic similarity—they need quality signals. A search result pointing to a 200-line function with cyclomatic complexity of 47 is technically relevant but practically useless as a reference. Current code search tools treat all results equally regardless of structural quality.

### Opportunity

Since mcp-vector-search already performs full AST traversal during indexing via Tree-sitter, computing structural metrics adds near-zero marginal cost. These metrics enable:

1. **Quality-aware search ranking** — deprioritize overly complex code in results
2. **Codebase health visualization** — interactive dashboards showing structural patterns
3. **Technical debt detection** — identify maintenance hotspots without runtime analysis
4. **Search filtering** — `--max-complexity 15` to exclude problematic code

### Design Principles

- **Zero additional parsing** — all metrics computed during existing AST traversal
- **Low false positive rate** — only metrics with proven correlation to maintainability
- **Visualization-first** — every metric must render meaningfully in charts/graphs
- **Actionable thresholds** — industry-standard defaults that developers can fix

---

## Metrics Specification

### Tier 1: Free During Indexing (O(1) per node)

These metrics require only node counting during the existing Tree-sitter traversal.

#### Cognitive Complexity

**What**: SonarQube's cognitive complexity algorithm — the industry standard for measuring code understandability.

**Calculation**:
```
+1 for each: if, elif, else, for, while, try, catch, switch/match, ternary, and, or
+1 additional per nesting level for control structures
```

**Why it works**: Unlike cyclomatic complexity, a switch with 20 cases scores +1 (not +20) because humans read switches as lookup tables. Research with 216 developers confirmed correlation with perceived difficulty.

**Threshold**: ≤15 per function (SonarQube default)

**Visualization**: Heatmap by file, histogram of function distribution

```python
@dataclass
class CognitiveComplexityMetric:
    score: int
    breakdown: dict[str, int]  # {"if": 3, "nesting_penalty": 2, ...}
    threshold: int = 15

    @property
    def grade(self) -> str:
        if self.score <= 5: return "A"
        if self.score <= 10: return "B"
        if self.score <= 15: return "C"
        if self.score <= 25: return "D"
        return "F"
```

#### Maximum Nesting Depth

**What**: Deepest level of nested control structures in a function.

**Calculation**: Track depth during traversal, record maximum.

**Why it works**: Universally understood signal — everyone agrees 6+ levels of nesting is problematic regardless of what the code does.

**Threshold**: ≤4 levels

**Visualization**: Treemap showing nesting depth by file/function

```python
@dataclass
class NestingDepthMetric:
    max_depth: int
    deepest_location: tuple[int, int]  # (start_line, end_line)
    threshold: int = 4
```

#### Cyclomatic Complexity

**What**: Count of linearly independent paths through code.

**Calculation**:
```
1 + count(if, elif, for, while, case, catch, and, or, ternary)
```

**Why it's secondary**: Same score can mean easy or hard code. A function with 10 sequential `if` statements (CC=11) is easier than one with 4 nested `if` statements (CC=5). Include for completeness but don't gate on it.

**Threshold**: ≤10 per function (display only, not quality gate)

**Visualization**: Scatter plot vs. cognitive complexity to show divergence

#### Function Length

**What**: Lines of code per function (excluding blanks and comments).

**Calculation**: `end_line - start_line` with blank/comment filtering.

**Threshold**: ≤30 lines (warning), ≤50 lines (error)

**Visualization**: Box plot by file/module

#### Parameter Count

**What**: Number of parameters in function signature.

**Calculation**: Count parameter children of function node.

**Threshold**: ≤5 parameters

**Visualization**: Histogram distribution

#### Method Count per Class

**What**: Number of methods in a class definition.

**Calculation**: Count method children during class node traversal.

**Threshold**: ≤20 methods

**Visualization**: Bubble chart (size = method count)

---

### Tier 2: Single-Pass with State (O(n) total)

These metrics require tracking state across the file but don't need cross-file analysis.

#### Efferent Coupling (Ce)

**What**: Count of unique external modules/types this file depends on.

**Calculation**: Collect unique import targets during traversal.

**Why it works**: High Ce means the file is fragile — changes to any dependency can break it.

**Threshold**: ≤20 external dependencies per file

**Visualization**: Dependency wheel showing import relationships

```python
@dataclass
class EfferentCouplingMetric:
    score: int
    dependencies: list[str]  # ["os", "pathlib.Path", "typing.Optional", ...]
    internal_deps: list[str]  # Same-project imports
    external_deps: list[str]  # Third-party imports
    stdlib_deps: list[str]    # Standard library imports
```

#### Halstead Metrics (Simplified)

**What**: Vocabulary and volume metrics based on operators/operands.

**Calculation**:
- n1 = unique operators
- n2 = unique operands
- N1 = total operators
- N2 = total operands
- Volume = (N1 + N2) × log2(n1 + n2)
- Difficulty = (n1/2) × (N2/n2)

**Why include**: Correlates with bug density. High difficulty = hard to modify correctly.

**Threshold**: Difficulty ≤30

**Visualization**: Volume vs. difficulty scatter plot

---

### Tier 3: Whole-File Analysis (O(n) per file)

Requires complete file traversal but no cross-file dependencies.

#### LCOM4 (Lack of Cohesion of Methods)

**What**: Number of connected components in method-variable access graph.

**Calculation**:
1. Build graph: nodes = methods, edges connect methods sharing instance variables or calling each other
2. Count connected components using union-find or DFS
3. Exclude constructors and empty methods

**Why it works**: LCOM4 = 1 means cohesive class. LCOM4 ≥ 2 means the class should split. Unlike LCOM1-3, this variant is practical to compute and interpret.

**Threshold**: = 1 (ideal), ≤ 2 (acceptable)

**Visualization**: Graph visualization showing method clusters within classes

```python
@dataclass
class CohesionMetric:
    lcom4: int
    method_count: int
    connected_components: list[list[str]]  # [["method_a", "method_b"], ["method_c"]]
    shared_variables: dict[str, list[str]]  # {"_config": ["init", "load", "save"]}
```

#### Code Smells (Pattern-Based)

Detectable via Tree-sitter queries without semantic analysis:

| Smell | Detection Rule | Confidence |
|-------|---------------|------------|
| Long Method | LOC > 30 OR cognitive_complexity > 15 | High |
| Long Parameter List | param_count > 5 | High |
| Deep Nesting | max_depth > 4 | High |
| God Class | methods > 20 AND lcom4 > 1 AND loc > 500 | Medium |
| Large Class | LOC > 500 | High |
| Empty Catch | catch block with pass/empty body | High |
| Magic Numbers | numeric literals outside [-1, 0, 1, 2, 10, 100] | Medium |

**Visualization**: Smell distribution pie chart, smell-by-file heatmap

---

### Tier 4: Cross-File Analysis (Dedicated Pass)

Requires the complete codebase index. Run separately from indexing.

#### Afferent Coupling (Ca)

**What**: Count of external files that depend on this file.

**Calculation**: Inverse lookup from efferent coupling data.

**Why it works**: High Ca means the file is load-bearing — changes ripple widely.

**Threshold**: Context-dependent (core utilities naturally have high Ca)

**Visualization**: Dependency graph with node size = Ca

#### Instability Index

**What**: I = Ce / (Ca + Ce)

**Interpretation**:
- I = 0: Completely stable (everyone depends on it, it depends on nothing)
- I = 1: Completely unstable (depends on everything, nothing depends on it)

**Why it works**: Stable Dependencies Principle — dependencies should flow toward stability.

**Visualization**: Instability spectrum showing all files

#### Circular Dependencies

**What**: Cycles in the import graph.

**Calculation**: Tarjan's strongly connected components algorithm on import graph.

**Why it works**: Circular dependencies prevent incremental builds and indicate architectural problems.

**Threshold**: 0 cycles

**Visualization**: Highlighted cycles in dependency graph

```python
@dataclass
class CircularDependency:
    cycle: list[str]  # ["file_a.py", "file_b.py", "file_c.py", "file_a.py"]
    severity: str     # "error" if > 2 files, "warning" if 2 files
```

---

## Data Model Extensions

### Extended Chunk Metadata

Extend the existing chunk metadata schema:

```python
@dataclass
class ChunkMetrics:
    """Structural metrics computed during indexing."""

    # Tier 1 - Always computed
    cognitive_complexity: int
    cyclomatic_complexity: int
    max_nesting_depth: int
    line_count: int
    parameter_count: Optional[int]  # Only for functions

    # Tier 1 - Class-level
    method_count: Optional[int]     # Only for classes

    # Quality grades
    complexity_grade: str           # A-F based on cognitive complexity

    # Smell flags
    smells: list[str]               # ["long_method", "deep_nesting", ...]

@dataclass
class FileMetrics:
    """File-level aggregated metrics."""

    file_path: str
    language: str

    # Size metrics
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int

    # Complexity aggregates
    total_cognitive_complexity: int
    max_cognitive_complexity: int
    avg_cognitive_complexity: float

    # Coupling
    efferent_coupling: int
    imports: list[str]

    # Counts
    function_count: int
    class_count: int

    # Smells
    smell_count: int
    smells: dict[str, int]  # {"long_method": 2, "deep_nesting": 1}

    # Halstead
    halstead_volume: float
    halstead_difficulty: float

@dataclass
class ProjectMetrics:
    """Project-wide aggregated metrics."""

    # Timestamp
    analyzed_at: datetime

    # Size
    total_files: int
    total_lines: int
    total_functions: int
    total_classes: int

    # Languages
    language_distribution: dict[str, int]  # {"python": 45, "javascript": 30}

    # Complexity distribution
    complexity_histogram: dict[str, int]   # {"A": 150, "B": 80, "C": 30, ...}

    # Coupling (Tier 4)
    circular_dependencies: list[CircularDependency]
    avg_instability: float

    # Smells
    total_smells: int
    smell_distribution: dict[str, int]

    # Technical debt estimate
    estimated_debt_hours: float  # Based on SonarQube remediation times
```

### ChromaDB Metadata Extension

Extend chunk metadata stored in ChromaDB:

```python
metadata = {
    # Existing fields
    "file_path": chunk["file_path"],
    "start_line": chunk["start_line"],
    "end_line": chunk["end_line"],
    "language": chunk["language"],
    "chunk_type": chunk.get("chunk_type", "code"),
    "function_name": chunk.get("function_name"),
    "class_name": chunk.get("class_name"),
    "docstring": chunk.get("docstring", ""),

    # NEW: Structural metrics
    "cognitive_complexity": metrics.cognitive_complexity,
    "cyclomatic_complexity": metrics.cyclomatic_complexity,
    "max_nesting_depth": metrics.max_nesting_depth,
    "line_count": metrics.line_count,
    "complexity_grade": metrics.complexity_grade,
    "has_smells": len(metrics.smells) > 0,
    "smell_count": len(metrics.smells),
}
```

### SQLite Metrics Store

For aggregated metrics and time-series tracking:

```sql
-- File-level metrics (updated on each index)
CREATE TABLE file_metrics (
    file_path TEXT PRIMARY KEY,
    language TEXT,
    total_lines INTEGER,
    code_lines INTEGER,
    cognitive_complexity_total INTEGER,
    cognitive_complexity_max INTEGER,
    cyclomatic_complexity_total INTEGER,
    efferent_coupling INTEGER,
    function_count INTEGER,
    class_count INTEGER,
    smell_count INTEGER,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project snapshots (for trend tracking)
CREATE TABLE project_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_files INTEGER,
    total_lines INTEGER,
    total_functions INTEGER,
    avg_cognitive_complexity REAL,
    total_smells INTEGER,
    circular_dependency_count INTEGER,
    estimated_debt_hours REAL
);

-- Individual smells (for drill-down)
CREATE TABLE code_smells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    start_line INTEGER,
    end_line INTEGER,
    smell_type TEXT,
    severity TEXT,  -- 'warning', 'error'
    message TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_path) REFERENCES file_metrics(file_path)
);

-- Circular dependencies
CREATE TABLE circular_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_hash TEXT UNIQUE,  -- Hash of sorted cycle for deduplication
    cycle_files TEXT,        -- JSON array of files in cycle
    file_count INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_file_metrics_language ON file_metrics(language);
CREATE INDEX idx_file_metrics_complexity ON file_metrics(cognitive_complexity_max);
CREATE INDEX idx_code_smells_type ON code_smells(smell_type);
CREATE INDEX idx_code_smells_file ON code_smells(file_path);
```

---

## CLI Interface Design

### New Commands

```bash
# Full analysis (Tier 1-4 metrics)
mcp-vector-search analyze [--output FORMAT] [--threshold-file FILE]

# Quick analysis (Tier 1-2 only, runs during index)
mcp-vector-search analyze --quick

# Specific analysis targets
mcp-vector-search analyze --file src/auth.py
mcp-vector-search analyze --directory src/core/
mcp-vector-search analyze --function "login_handler"

# Output formats
mcp-vector-search analyze --output json > metrics.json
mcp-vector-search analyze --output sarif > results.sarif  # For CI integration
mcp-vector-search analyze --output html > report.html

# Threshold configuration
mcp-vector-search analyze --fail-on-smell          # Exit code 1 if smells found
mcp-vector-search analyze --max-complexity 15      # Fail if any function exceeds
mcp-vector-search analyze --quality-gate strict    # Preset threshold profiles

# Diff-aware analysis (for CI/CD)
mcp-vector-search analyze --changed-only           # Only analyze git-changed files
mcp-vector-search analyze --baseline main          # Compare against branch

# Visualization export
mcp-vector-search analyze --export-viz ./viz-data/ # Export data for visualizer
```

### Enhanced Search with Quality Filters

```bash
# Filter search results by quality
mcp-vector-search search "authentication" --max-complexity 15
mcp-vector-search search "database" --no-smells
mcp-vector-search search "api handler" --grade A,B

# Quality-aware ranking (default behavior)
mcp-vector-search search "login" --quality-weight 0.3  # 30% quality, 70% relevance
```

### Status Command Extension

```bash
$ mcp-vector-search status --metrics

📊 Project Metrics Summary
═══════════════════════════════════════════════════

Files: 247 (Python: 189, JavaScript: 42, TypeScript: 16)
Lines: 45,892 (Code: 38,456, Comments: 4,231, Blank: 3,205)

Complexity Distribution:
  A (≤5)  ████████████████████████████████░░░░░░  68% (412 functions)
  B (≤10) ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  22% (134 functions)
  C (≤15) ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7% (42 functions)
  D (≤25) █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2% (15 functions)
  F (>25) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1% (5 functions)

Code Smells: 23 total
  Long Method:      8
  Deep Nesting:     6
  Long Parameters:  5
  God Class:        2
  Empty Catch:      2

Coupling:
  Circular Dependencies: 1 cycle (auth.py ↔ user.py)
  Avg Instability: 0.42

Technical Debt Estimate: ~18 hours

Run 'mcp-vector-search analyze --verbose' for detailed report
```

---

## Visualization Data Export

### Export Format for Visualizer

```json
{
  "metadata": {
    "project_name": "my-project",
    "analyzed_at": "2024-12-09T10:30:00Z",
    "tool_version": "0.1.0"
  },

  "summary": {
    "total_files": 247,
    "total_lines": 45892,
    "total_functions": 608,
    "total_classes": 89,
    "avg_cognitive_complexity": 6.2,
    "total_smells": 23,
    "estimated_debt_hours": 18.5
  },

  "files": [
    {
      "path": "src/auth/handler.py",
      "language": "python",
      "metrics": {
        "lines": 245,
        "functions": 8,
        "classes": 2,
        "cognitive_complexity_total": 67,
        "cognitive_complexity_max": 23,
        "efferent_coupling": 12,
        "smell_count": 2
      },
      "functions": [
        {
          "name": "validate_token",
          "start_line": 45,
          "end_line": 112,
          "metrics": {
            "cognitive_complexity": 23,
            "cyclomatic_complexity": 18,
            "max_nesting_depth": 5,
            "line_count": 67,
            "parameter_count": 4,
            "complexity_grade": "D"
          },
          "smells": ["long_method", "deep_nesting"]
        }
      ]
    }
  ],

  "dependency_graph": {
    "nodes": [
      {"id": "src/auth/handler.py", "size": 245, "coupling": 12},
      {"id": "src/models/user.py", "size": 180, "coupling": 8}
    ],
    "edges": [
      {"source": "src/auth/handler.py", "target": "src/models/user.py"}
    ],
    "cycles": [
      ["src/auth/handler.py", "src/models/user.py", "src/auth/handler.py"]
    ]
  },

  "smell_locations": [
    {
      "file": "src/auth/handler.py",
      "line": 45,
      "type": "long_method",
      "severity": "warning",
      "message": "Function 'validate_token' has 67 lines (threshold: 30)"
    }
  ],

  "trends": [
    {
      "date": "2024-12-01",
      "total_smells": 28,
      "avg_complexity": 6.8,
      "debt_hours": 22.0
    },
    {
      "date": "2024-12-09",
      "total_smells": 23,
      "avg_complexity": 6.2,
      "debt_hours": 18.5
    }
  ]
}
```

### Visualization Components

The visualizer should support these chart types:

#### 1. Complexity Heatmap
- **Data**: File grid colored by max cognitive complexity
- **Interaction**: Click to drill into file → function breakdown
- **Use case**: Identify maintenance hotspots at a glance

#### 2. Dependency Graph
- **Data**: Nodes = files, edges = imports, node size = LOC or coupling
- **Interaction**: Highlight cycles, filter by directory
- **Use case**: Understand architecture, find circular dependencies

#### 3. Complexity Distribution Histogram
- **Data**: Function count by complexity grade (A-F)
- **Interaction**: Click bar to list functions in that grade
- **Use case**: Track codebase health over time

#### 4. Smell Treemap
- **Data**: Rectangles sized by smell count, colored by severity
- **Interaction**: Drill from project → directory → file → function
- **Use case**: Prioritize refactoring efforts

#### 5. Trend Line Chart
- **Data**: Time series of key metrics (smells, avg complexity, debt hours)
- **Interaction**: Zoom, select date range
- **Use case**: Track improvement over sprints

#### 6. Coupling Wheel
- **Data**: Chord diagram showing import relationships
- **Interaction**: Highlight file to show all connections
- **Use case**: Identify tightly coupled modules

---

## Implementation Architecture

### Module Structure

```
src/mcp_vector_search/
├── analysis/
│   ├── __init__.py
│   ├── metrics.py          # Metric dataclasses and calculations
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract collector interface
│   │   ├── complexity.py   # Cognitive/cyclomatic complexity
│   │   ├── coupling.py     # Efferent/afferent coupling
│   │   ├── cohesion.py     # LCOM4 calculation
│   │   ├── smells.py       # Code smell detection
│   │   └── halstead.py     # Halstead metrics
│   ├── aggregators/
│   │   ├── __init__.py
│   │   ├── file.py         # File-level aggregation
│   │   ├── project.py      # Project-level aggregation
│   │   └── trends.py       # Historical trend tracking
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── console.py      # Rich terminal output
│   │   ├── json.py         # JSON export
│   │   ├── sarif.py        # SARIF format for CI
│   │   └── html.py         # Standalone HTML report
│   ├── visualizer/
│   │   ├── __init__.py
│   │   ├── exporter.py     # Export data for external visualizer
│   │   └── schemas.py      # Visualization data schemas
│   └── thresholds.py       # Configurable threshold management
├── cli/
│   └── commands/
│       └── analyze.py      # New analyze command
```

### Collector Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import tree_sitter

@dataclass
class CollectorContext:
    """Context passed to collectors during traversal."""
    file_path: str
    language: str
    content: str
    tree: tree_sitter.Tree

class MetricCollector(ABC):
    """Base class for metric collectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this collector."""
        pass

    @property
    @abstractmethod
    def tier(self) -> int:
        """Computation tier (1-4) indicating cost."""
        pass

    @abstractmethod
    def collect_node(
        self,
        node: tree_sitter.Node,
        context: CollectorContext,
        depth: int
    ) -> None:
        """Called for each node during traversal."""
        pass

    @abstractmethod
    def finalize_function(
        self,
        function_node: tree_sitter.Node,
        context: CollectorContext
    ) -> dict[str, Any]:
        """Called when exiting a function node. Returns metrics."""
        pass

    @abstractmethod
    def finalize_file(self, context: CollectorContext) -> dict[str, Any]:
        """Called after file traversal. Returns file-level metrics."""
        pass

    def reset(self) -> None:
        """Reset collector state for next file."""
        pass
```

### Integration with Existing Indexer

```python
# In core/indexer.py

class TreeSitterIndexer:
    def __init__(self, collectors: list[MetricCollector] = None):
        self.collectors = collectors or self._default_collectors()

    def _default_collectors(self) -> list[MetricCollector]:
        """Initialize default Tier 1-2 collectors."""
        return [
            CognitiveComplexityCollector(),
            CyclomaticComplexityCollector(),
            NestingDepthCollector(),
            ParameterCountCollector(),
            EfferentCouplingCollector(),
        ]

    async def parse_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse file and collect metrics during traversal."""
        content = file_path.read_text(encoding="utf-8")
        tree = self.parser.parse(content.encode())

        context = CollectorContext(
            file_path=str(file_path),
            language=self._detect_language(file_path),
            content=content,
            tree=tree
        )

        # Reset collectors
        for collector in self.collectors:
            collector.reset()

        chunks = []
        self._traverse_with_metrics(tree.root_node, context, chunks, depth=0)

        # Collect file-level metrics
        file_metrics = {}
        for collector in self.collectors:
            file_metrics.update(collector.finalize_file(context))

        return chunks, file_metrics

    def _traverse_with_metrics(
        self,
        node: tree_sitter.Node,
        context: CollectorContext,
        chunks: list,
        depth: int
    ) -> None:
        """Traverse AST, collecting metrics at each node."""

        # Notify all collectors of this node
        for collector in self.collectors:
            collector.collect_node(node, context, depth)

        # Handle function nodes
        if node.type in self._function_node_types:
            chunk = self._extract_chunk(node, context)

            # Collect function-level metrics from all collectors
            for collector in self.collectors:
                metrics = collector.finalize_function(node, context)
                chunk["metrics"].update(metrics)

            # Compute derived metrics
            chunk["metrics"]["complexity_grade"] = self._compute_grade(
                chunk["metrics"].get("cognitive_complexity", 0)
            )
            chunk["metrics"]["smells"] = self._detect_smells(chunk["metrics"])

            chunks.append(chunk)

        # Recurse into children
        for child in node.children:
            child_depth = depth + 1 if self._is_nesting_node(node) else depth
            self._traverse_with_metrics(child, context, chunks, child_depth)
```

### Cognitive Complexity Collector Implementation

```python
from dataclasses import dataclass, field

@dataclass
class CognitiveComplexityCollector(MetricCollector):
    """Collects cognitive complexity metrics during traversal."""

    name: str = "cognitive_complexity"
    tier: int = 1

    # Nodes that increment complexity
    COMPLEXITY_NODES: set[str] = field(default_factory=lambda: {
        # Control flow
        "if_statement", "elif_clause", "else_clause",
        "for_statement", "while_statement",
        "try_statement", "except_clause",
        "match_statement", "case_clause",
        "conditional_expression",  # ternary
        # Boolean operators
        "and", "or", "boolean_operator",
        # JavaScript/TypeScript additions
        "ternary_expression", "switch_statement", "case",
        "catch_clause", "for_in_statement", "for_of_statement",
    })

    # Nodes that increase nesting level
    NESTING_NODES: set[str] = field(default_factory=lambda: {
        "if_statement", "for_statement", "while_statement",
        "try_statement", "match_statement", "switch_statement",
        "function_definition", "async_function_definition",
        "lambda", "arrow_function", "function_expression",
    })

    # State
    _current_complexity: int = 0
    _current_nesting: int = 0
    _max_nesting: int = 0
    _breakdown: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._current_complexity = 0
        self._current_nesting = 0
        self._max_nesting = 0
        self._breakdown = {}

    def collect_node(
        self,
        node: tree_sitter.Node,
        context: CollectorContext,
        depth: int
    ) -> None:
        # Track nesting
        if node.type in self.NESTING_NODES:
            self._current_nesting += 1
            self._max_nesting = max(self._max_nesting, self._current_nesting)

        # Calculate complexity increment
        if node.type in self.COMPLEXITY_NODES:
            # Base increment
            increment = 1

            # Nesting penalty for control structures (not boolean operators)
            if node.type not in {"and", "or", "boolean_operator"}:
                increment += self._current_nesting

            self._current_complexity += increment

            # Track breakdown
            node_category = self._categorize_node(node.type)
            self._breakdown[node_category] = self._breakdown.get(node_category, 0) + increment

    def finalize_function(
        self,
        function_node: tree_sitter.Node,
        context: CollectorContext
    ) -> dict[str, Any]:
        result = {
            "cognitive_complexity": self._current_complexity,
            "cognitive_complexity_breakdown": self._breakdown.copy(),
            "max_nesting_depth": self._max_nesting,
        }

        # Reset for next function
        self._current_complexity = 0
        self._current_nesting = 0
        self._max_nesting = 0
        self._breakdown = {}

        return result

    def finalize_file(self, context: CollectorContext) -> dict[str, Any]:
        return {}  # File-level aggregation done elsewhere

    def _categorize_node(self, node_type: str) -> str:
        if node_type in {"if_statement", "elif_clause", "else_clause"}:
            return "conditionals"
        if node_type in {"for_statement", "while_statement", "for_in_statement", "for_of_statement"}:
            return "loops"
        if node_type in {"try_statement", "except_clause", "catch_clause"}:
            return "error_handling"
        if node_type in {"and", "or", "boolean_operator"}:
            return "boolean_operators"
        return "other"
```

---

## Quality Gates & Thresholds

### Default Threshold Configuration

```yaml
# .mcp-vector-search/thresholds.yaml

# Quality gate presets
presets:
  strict:
    cognitive_complexity: 10
    cyclomatic_complexity: 8
    max_nesting_depth: 3
    function_lines: 25
    parameter_count: 4
    method_count: 15

  standard:  # Default
    cognitive_complexity: 15
    cyclomatic_complexity: 10
    max_nesting_depth: 4
    function_lines: 30
    parameter_count: 5
    method_count: 20

  relaxed:
    cognitive_complexity: 25
    cyclomatic_complexity: 15
    max_nesting_depth: 5
    function_lines: 50
    parameter_count: 7
    method_count: 30

# Active preset
active_preset: standard

# Override specific thresholds
overrides:
  # Allow higher complexity in test files
  "tests/**":
    cognitive_complexity: 25
    function_lines: 50

  # Stricter for core modules
  "src/core/**":
    cognitive_complexity: 10

# Smell configuration
smells:
  enabled:
    - long_method
    - deep_nesting
    - long_parameter_list
    - god_class
    - empty_catch

  disabled:
    - magic_numbers  # Too noisy for this project

# Exit codes
exit_codes:
  on_smell: true        # Exit 1 if any smell found
  on_threshold: true    # Exit 1 if any threshold exceeded
  severity_filter: error  # Only fail on 'error' severity, not 'warning'
```

### Technical Debt Calculation

Based on SonarQube's remediation time model:

```python
REMEDIATION_TIMES = {
    # Minutes to fix each issue
    "long_method": 20,
    "deep_nesting": 15,
    "long_parameter_list": 10,
    "god_class": 120,
    "empty_catch": 5,
    "magic_numbers": 5,
    "high_cognitive_complexity": 30,  # Per function over threshold
    "circular_dependency": 60,        # Per cycle
}

def estimate_debt_hours(smells: list[CodeSmell], thresholds: Thresholds) -> float:
    """Estimate technical debt in hours."""
    total_minutes = 0

    for smell in smells:
        base_time = REMEDIATION_TIMES.get(smell.type, 10)

        # Scale by severity
        if smell.severity == "error":
            base_time *= 1.5

        total_minutes += base_time

    return total_minutes / 60
```

---

## Performance Considerations

### Indexing Performance Impact

| Collector | Time per 1000 LOC | Memory Overhead |
|-----------|-------------------|-----------------|
| Cognitive Complexity | <1ms | ~100 bytes/function |
| Cyclomatic Complexity | <1ms | ~50 bytes/function |
| Nesting Depth | <0.5ms | ~20 bytes/function |
| Efferent Coupling | <2ms | ~500 bytes/file |
| LCOM4 | ~5ms | ~1KB/class |

**Total overhead**: <10ms per 1000 LOC for Tier 1-2 metrics.

### Caching Strategy

```python
# Content-based caching for expensive computations
class MetricsCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_file_metrics(self, file_path: Path) -> Optional[FileMetrics]:
        """Return cached metrics if file unchanged."""
        cache_key = self._compute_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            return FileMetrics.from_json(cache_file.read_text())
        return None

    def _compute_key(self, file_path: Path) -> str:
        """Hash based on file content + mtime."""
        content = file_path.read_bytes()
        mtime = file_path.stat().st_mtime
        return hashlib.sha256(content + str(mtime).encode()).hexdigest()[:16]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_analysis/test_cognitive_complexity.py

@pytest.mark.parametrize("code,expected_complexity", [
    # Simple function
    ("def foo(): pass", 0),

    # Single if
    ("def foo():\n  if x: pass", 1),

    # Nested if (1 + nesting penalty)
    ("def foo():\n  if x:\n    if y: pass", 3),

    # Boolean operators
    ("def foo():\n  if x and y: pass", 2),

    # Complex real-world example
    ("""
def validate(user, token):
    if not user:                    # +1
        return False
    if not token:                   # +1
        return False
    if user.is_admin:               # +1
        if token.has_scope('admin'): # +2 (nested)
            return True
        elif token.is_supertoken:   # +2 (nested)
            return True
    return user.can_access(token)
""", 7),
])
def test_cognitive_complexity_calculation(code: str, expected_complexity: int):
    collector = CognitiveComplexityCollector()
    tree = parse_python(code)

    # Simulate traversal
    for node in traverse(tree.root_node):
        collector.collect_node(node, context, depth=0)

    metrics = collector.finalize_function(tree.root_node.children[-1], context)
    assert metrics["cognitive_complexity"] == expected_complexity
```

### Integration Tests

```python
# tests/test_analysis/test_full_analysis.py

async def test_analyze_real_project(temp_project_with_code):
    """Test analysis on realistic codebase."""
    analyzer = ProjectAnalyzer(temp_project_with_code)
    results = await analyzer.analyze()

    # Verify structure
    assert results.total_files > 0
    assert results.total_functions > 0

    # Verify metrics computed
    for file_metrics in results.files:
        assert file_metrics.cognitive_complexity_total >= 0
        assert file_metrics.efferent_coupling >= 0

    # Verify smells detected
    assert any(s.type == "long_method" for s in results.smells)

async def test_incremental_analysis(temp_project_with_code):
    """Test that unchanged files use cached metrics."""
    analyzer = ProjectAnalyzer(temp_project_with_code)

    # First run
    start = time.time()
    results1 = await analyzer.analyze()
    first_run_time = time.time() - start

    # Second run (should be faster due to caching)
    start = time.time()
    results2 = await analyzer.analyze()
    second_run_time = time.time() - start

    assert second_run_time < first_run_time * 0.5
    assert results1.total_smells == results2.total_smells
```

---

## Rollout Plan

### Phase 1: Core Metrics (Week 1-2)

**Deliverables**:
- Tier 1 collectors integrated into indexer
- Extended chunk metadata in ChromaDB
- `analyze --quick` command
- Basic console reporter

**Validation**:
- Metrics match SonarQube on sample projects
- <10ms overhead per 1000 LOC

### Phase 2: Quality Gates (Week 3)

**Deliverables**:
- Threshold configuration system
- SARIF output for CI integration
- `--fail-on-smell` exit codes
- Diff-aware analysis

**Validation**:
- GitHub Actions integration working
- Threshold overrides function correctly

### Phase 3: Cross-File Analysis (Week 4)

**Deliverables**:
- Tier 4 collectors (afferent coupling, circular deps)
- Dependency graph construction
- SQLite metrics store
- Trend tracking

**Validation**:
- Circular dependencies detected correctly
- Historical snapshots recorded

### Phase 4: Visualization Export (Week 5)

**Deliverables**:
- JSON export for visualizer
- All chart data schemas finalized
- HTML standalone report
- Documentation

**Validation**:
- Visualizer consumes export successfully
- All documented charts renderable

---

## Open Questions

1. **Search ranking integration**: Should quality metrics affect search result ranking by default, or only when explicitly requested?

2. **Language parity**: Which languages get Tier 3-4 metrics first? Python has best Tree-sitter support, but JavaScript/TypeScript may have more demand.

3. **Baseline management**: How should teams establish and update baseline metrics? Git tag? Explicit snapshot command?

4. **MCP tool exposure**: Which analysis capabilities should be exposed as MCP tools vs. CLI-only?

---

## Appendix: Tree-sitter Query Examples

### Python: Find Functions with High Nesting

```scheme
(function_definition
  body: (block
    (if_statement
      consequence: (block
        (if_statement
          consequence: (block
            (if_statement)))))) @deep_nesting)
```

### JavaScript: Find Empty Catch Blocks

```scheme
(catch_clause
  body: (statement_block) @empty_catch
  (#eq? @empty_catch "{}"))
```

### Python: Count Boolean Operators

```scheme
(boolean_operator
  operator: (_) @op)
```

---

## References

- SonarQube Cognitive Complexity Whitepaper
- CodeClimate 10-Point Technical Debt Assessment
- Semgrep Pattern Syntax Documentation
- Tree-sitter Query Language Reference
