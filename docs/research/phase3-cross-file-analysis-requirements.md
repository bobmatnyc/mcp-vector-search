# Phase 3: Cross-File Analysis Requirements Research

**Project**: mcp-vector-search Structural Code Analysis
**Phase**: 3 of 5 - Cross-File Analysis
**Milestone**: v0.19.0
**Target Date**: Jan 6, 2025
**Research Date**: December 11, 2024
**Researcher**: Claude (Research Agent)

---

## Executive Summary

Phase 3 introduces **cross-file dependency analysis** and **persistent metrics storage** to mcp-vector-search. This phase shifts from single-file metrics (Phases 1-2) to project-wide relationship tracking, enabling:

1. **Coupling Analysis**: Efferent (outgoing) and afferent (incoming) dependencies
2. **Architectural Health**: Instability index and circular dependency detection
3. **Historical Tracking**: SQLite-based metrics store for trend analysis
4. **Advanced Cohesion**: LCOM4 metric for class design quality

**Critical Foundation**: Issue #24 (SQLite metrics store) must be implemented first, as #25 (trend tracking) depends on persistent storage.

**Completion Status of Prerequisites**:
- ✅ Phase 1 (Core Metrics): Complete - Issues #1-11 merged
- ✅ Phase 2 (Quality Gates): Complete - Issues #12-18 merged
- 📋 Phase 3 (Cross-File Analysis): Ready to start

---

## Phase 3 Issues Analysis

### Issue #19: [EPIC] Cross-File Analysis

**Type**: Epic (tracking issue)
**Dependencies**: Phase 2 complete
**Complexity**: N/A (epic)
**Status**: 📋 Backlog

**Purpose**: Parent issue tracking all Phase 3 work.

**Subtasks**:
- #20 Efferent Coupling Collector
- #21 Afferent Coupling Collector
- #22 Instability Index
- #23 Circular dependency detection
- #24 SQLite metrics store
- #25 Trend tracking
- #26 LCOM4 cohesion metric

**Implementation Order**: See Critical Path section below.

---

### Issue #20: Efferent Coupling Collector

**Title**: Implement Efferent Coupling (Ce) metric collector
**Dependencies**: #2 (metric dataclasses - ✅ complete)
**Complexity**: **MEDIUM**
**Implementation Priority**: **2nd** (after #24)

#### Technical Requirements

**Definition**: Efferent coupling (Ce) measures how many external modules/types a file depends on. High Ce indicates fragility—changes to dependencies can break this file.

**Algorithm**:
```python
# During AST traversal, collect import statements
imports = set()

for node in traverse(ast):
    if node.type in ("import_statement", "import_from_statement"):
        # Extract module path
        module = extract_module_name(node)
        imports.add(module)

ce = len(imports)  # Count unique dependencies
```

**Data Model Extension**:
```python
@dataclass
class EfferentCouplingMetric:
    """Efferent coupling metrics for a file."""

    score: int                    # Total unique dependencies
    dependencies: list[str]       # All imports
    internal_deps: list[str]      # Same-project imports
    external_deps: list[str]      # Third-party imports
    stdlib_deps: list[str]        # Standard library imports

    # Classification helper
    def categorize_import(self, module: str, project_root: str) -> str:
        """Classify import as internal/external/stdlib."""
        if is_stdlib(module):
            return "stdlib"
        elif module.startswith(project_root):
            return "internal"
        else:
            return "external"
```

**Integration Points**:
1. Extend `FileMetrics` in `src/mcp_vector_search/analysis/metrics.py`:
   ```python
   @dataclass
   class FileMetrics:
       # ... existing fields ...

       # NEW: Coupling metrics
       efferent_coupling: int = 0
       imports: list[str] = field(default_factory=list)
       internal_imports: list[str] = field(default_factory=list)
       external_imports: list[str] = field(default_factory=list)
   ```

2. Create `src/mcp_vector_search/analysis/collectors/coupling.py`

3. Register collector in `src/mcp_vector_search/core/indexer.py`

**Multi-Language Support**:
- **Python**: `import`, `from ... import`
- **JavaScript/TypeScript**: `import`, `require()`, dynamic `import()`
- **Go**: `import`
- **Rust**: `use`, `extern crate`

**Tree-sitter Queries**:
```scheme
;; Python imports
(import_statement
  name: (dotted_name) @module)

(import_from_statement
  module_name: (dotted_name) @module)

;; JavaScript imports
(import_statement
  source: (string) @module)

;; TypeScript imports (same as JS)
(import_statement
  source: (string) @module)
```

**Testing Requirements**:
- Unit tests with sample files containing known import counts
- Cross-language validation (Python, JS, TS)
- Edge cases: circular imports, star imports, aliased imports

**Estimated Effort**: 2-3 days

---

### Issue #21: Afferent Coupling Collector

**Title**: Implement Afferent Coupling (Ca) metric with reverse dependency lookup
**Dependencies**: #20 (efferent coupling - must complete first)
**Complexity**: **MEDIUM-HIGH**
**Implementation Priority**: **3rd**

#### Technical Requirements

**Definition**: Afferent coupling (Ca) measures how many external files depend on this file. High Ca indicates load-bearing code—changes ripple widely.

**Algorithm**:
```python
# Requires whole-project analysis (can't compute during single file parse)

# Step 1: Collect all efferent couplings during indexing
file_imports = {}  # file_path -> list[imported_modules]

for file_path in project_files:
    ce_metrics = analyze_imports(file_path)
    file_imports[file_path] = ce_metrics.dependencies

# Step 2: Build reverse index (afferent coupling)
file_importers = defaultdict(list)  # module -> list[files_that_import_it]

for file_path, imports in file_imports.items():
    for imported_module in imports:
        file_importers[imported_module].append(file_path)

# Step 3: Compute Ca for each file
for file_path in project_files:
    ca = len(file_importers[file_path])
    store_metric(file_path, afferent_coupling=ca)
```

**Data Model Extension**:
```python
@dataclass
class AfferentCouplingMetric:
    """Afferent coupling metrics for a file."""

    score: int                    # Number of files importing this one
    importers: list[str]          # Files that import this module

    # Relationship context
    is_load_bearing: bool         # True if Ca > threshold (e.g., 10)
    is_utility: bool              # True if Ca high + Ce low
    is_facade: bool               # True if Ca high + Ce high
```

**Integration Points**:
1. Extend `FileMetrics`:
   ```python
   @dataclass
   class FileMetrics:
       # ... existing fields ...

       # NEW: Afferent coupling
       afferent_coupling: int = 0
       importers: list[str] = field(default_factory=list)
   ```

2. Add to `src/mcp_vector_search/analysis/collectors/coupling.py`

3. Create separate analysis pass in indexer (post-processing step)

**Implementation Strategy**:

Option A: **Two-pass indexing** (recommended)
```python
# Pass 1: Collect efferent coupling during normal indexing
for file in files:
    chunks = parse_file(file)
    ce_metrics = extract_efferent_coupling(chunks)
    store_temp(file, ce_metrics)

# Pass 2: Build reverse index for afferent coupling
dependency_graph = build_reverse_index(all_ce_metrics)
for file in files:
    ca = dependency_graph.count_importers(file)
    update_file_metrics(file, afferent_coupling=ca)
```

Option B: **On-demand calculation** (for large codebases)
```python
# Calculate Ca only when requested (e.g., during `analyze` command)
def compute_afferent_coupling(file_path: str) -> int:
    query = "SELECT DISTINCT file_path FROM imports WHERE target = ?"
    importers = db.execute(query, (file_path,)).fetchall()
    return len(importers)
```

**SQLite Schema** (from #24):
```sql
-- Store import relationships
CREATE TABLE file_imports (
    source_file TEXT NOT NULL,
    target_module TEXT NOT NULL,
    import_type TEXT,  -- 'internal', 'external', 'stdlib'
    line_number INTEGER,
    FOREIGN KEY (source_file) REFERENCES file_metrics(file_path)
);

CREATE INDEX idx_imports_target ON file_imports(target_module);
```

**Testing Requirements**:
- Unit tests with mock dependency graphs
- Integration test with multi-file project
- Verify reverse lookup accuracy
- Performance test on large codebases (1000+ files)

**Performance Considerations**:
- For 1000 files with avg 20 imports each: ~20,000 relationships to index
- Reverse index build: O(n * m) where n=files, m=imports_per_file
- Expected time: <100ms for 1000 files

**Estimated Effort**: 3-4 days

---

### Issue #22: Instability Index

**Title**: Calculate Instability Index (I = Ce / (Ca + Ce))
**Dependencies**: #20 (efferent), #21 (afferent)
**Complexity**: **LOW**
**Implementation Priority**: **4th**

#### Technical Requirements

**Definition**: Instability Index measures how "stable" a module is based on its coupling ratio.

**Formula**:
```
I = Ce / (Ca + Ce)

Where:
- I = 0: Completely stable (many dependents, no dependencies)
- I = 1: Completely unstable (no dependents, many dependencies)
- I = 0.5: Balanced
```

**Interpretation**:
- **I < 0.3**: Stable module (good for core utilities, frameworks)
- **0.3 ≤ I ≤ 0.7**: Normal module (business logic, application code)
- **I > 0.7**: Unstable module (leaf nodes, UI components)

**Robert Martin's Stable Dependencies Principle**:
> Dependencies should flow toward stability (I should decrease along dependency chains)

**Algorithm**:
```python
def compute_instability(ce: int, ca: int) -> float:
    """Compute instability index.

    Args:
        ce: Efferent coupling (outgoing dependencies)
        ca: Afferent coupling (incoming dependents)

    Returns:
        Instability score 0.0-1.0
    """
    if ce == 0 and ca == 0:
        return 0.0  # No dependencies = stable by convention

    return ce / (ce + ca)
```

**Data Model Extension**:
```python
@dataclass
class FileMetrics:
    # ... existing fields ...

    # NEW: Instability
    instability: float = 0.0
    stability_category: str = ""  # "stable", "normal", "unstable"

    def compute_instability(self) -> None:
        """Calculate instability from coupling metrics."""
        if self.efferent_coupling == 0 and self.afferent_coupling == 0:
            self.instability = 0.0
            self.stability_category = "stable"
        else:
            self.instability = self.efferent_coupling / (
                self.efferent_coupling + self.afferent_coupling
            )

            # Categorize
            if self.instability < 0.3:
                self.stability_category = "stable"
            elif self.instability > 0.7:
                self.stability_category = "unstable"
            else:
                self.stability_category = "normal"
```

**Integration Points**:
1. Add computation to `FileMetrics.compute_aggregates()`
2. Store in SQLite (Issue #24)
3. Display in console reporter
4. Include in JSON export for visualization

**Visualization**:
```
Instability Spectrum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stable                                    Unstable
|━━━━━━━━━━━|━━━━━━━━━━━━━━━|━━━━━━━━━━━━━━|
0.0        0.3             0.7            1.0

utils.py        ●───────────────────────────── (I=0.12)
config.py       ──●─────────────────────────── (I=0.25)
handler.py      ──────────────●──────────────── (I=0.58)
ui.py           ────────────────────────────●─ (I=0.89)
```

**Testing Requirements**:
- Unit tests with known coupling values
- Edge cases: zero coupling, balanced coupling
- Validate thresholds with real projects

**Estimated Effort**: 1 day

---

### Issue #23: Circular Dependency Detection

**Title**: Detect circular dependencies using Tarjan's SCC algorithm
**Dependencies**: #20 (efferent coupling for dependency graph)
**Complexity**: **HIGH**
**Implementation Priority**: **5th**

#### Technical Requirements

**Definition**: Circular dependencies occur when Module A imports Module B, which imports Module C, which imports Module A (forming a cycle).

**Problem**: Circular dependencies:
- Prevent incremental compilation
- Cause initialization order issues
- Indicate architectural problems
- Make code harder to test and refactor

**Algorithm**: Tarjan's Strongly Connected Components (SCC)
```python
def find_circular_dependencies(dependency_graph: dict[str, list[str]]) -> list[list[str]]:
    """Find all strongly connected components (cycles) in dependency graph.

    Uses Tarjan's algorithm for O(V + E) performance.

    Args:
        dependency_graph: file_path -> list[imported_files]

    Returns:
        List of cycles, where each cycle is a list of file paths
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = set()
    sccs = []

    def strongconnect(node: str) -> None:
        # Set depth index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        on_stack.add(node)
        stack.append(node)

        # Consider successors
        for successor in dependency_graph.get(node, []):
            if successor not in index:
                # Successor not visited, recurse
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                # Successor on stack = part of cycle
                lowlinks[node] = min(lowlinks[node], index[successor])

        # Root node, pop SCC
        if lowlinks[node] == index[node]:
            component = []
            while True:
                successor = stack.pop()
                on_stack.remove(successor)
                component.append(successor)
                if successor == node:
                    break

            # Only report cycles (SCC size > 1)
            if len(component) > 1:
                sccs.append(component)

    # Run algorithm on all nodes
    for node in dependency_graph:
        if node not in index:
            strongconnect(node)

    return sccs
```

**Data Model**:
```python
@dataclass
class CircularDependency:
    """Represents a circular dependency cycle."""

    cycle: list[str]              # File paths forming the cycle
    cycle_length: int             # Number of files in cycle
    severity: str                 # "error" (>2 files) or "warning" (2 files)

    @property
    def cycle_hash(self) -> str:
        """Unique identifier for deduplication."""
        # Sort to handle A→B→A same as B→A→B
        sorted_cycle = sorted(self.cycle)
        return hashlib.sha256(
            "|".join(sorted_cycle).encode()
        ).hexdigest()[:16]

    def to_display(self) -> str:
        """Human-readable cycle representation."""
        return " → ".join(self.cycle) + f" → {self.cycle[0]}"
```

**Integration Points**:
1. Add to `ProjectMetrics`:
   ```python
   @dataclass
   class ProjectMetrics:
       # ... existing fields ...

       # NEW: Circular dependencies
       circular_dependencies: list[CircularDependency] = field(default_factory=list)
       circular_dependency_count: int = 0
   ```

2. Create `src/mcp_vector_search/analysis/dependency_graph.py`

3. Run after afferent coupling analysis (separate pass)

**SQLite Schema** (from #24):
```sql
CREATE TABLE circular_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_hash TEXT UNIQUE,        -- Deduplication key
    cycle_files TEXT NOT NULL,     -- JSON array of file paths
    cycle_length INTEGER NOT NULL,
    severity TEXT NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Console Output**:
```
⚠️  Circular Dependencies Detected

ERROR: 3-file cycle (src/auth/handler.py)
  src/auth/handler.py
    → src/models/user.py
    → src/auth/permissions.py
    → src/auth/handler.py

WARNING: 2-file cycle (src/utils/helpers.py)
  src/utils/helpers.py
    → src/utils/formatters.py
    → src/utils/helpers.py

Recommendation: Refactor to break cycles by introducing interfaces/abstractions
```

**Testing Requirements**:
- Unit tests with known cycle graphs
- Test multi-component cycles
- Test isolated files (no cycles)
- Performance test on 1000+ file graph

**Performance**:
- Tarjan's algorithm: O(V + E) where V=files, E=imports
- For 1000 files, 20k imports: ~20ms

**Estimated Effort**: 3-4 days

---

### Issue #24: SQLite Metrics Store ⭐ CRITICAL FOUNDATION

**Title**: Create SQLite database for persistent metrics storage
**Dependencies**: #2 (metric dataclasses - ✅ complete)
**Complexity**: **MEDIUM**
**Implementation Priority**: **1st - MUST IMPLEMENT FIRST**

#### Why This Is Critical

**Issue #25 (Trend Tracking) DEPENDS ON THIS**. Without persistent storage:
- No historical metrics comparison
- No trend analysis over time
- No baseline management
- Metrics lost after each run

**This is the foundation for Phase 3's value proposition.**

#### Technical Requirements

**Purpose**: Persistent storage layer for:
1. File-level metrics (updated on each index)
2. Project snapshots (for trend tracking)
3. Code smells (for drill-down analysis)
4. Circular dependencies (for architectural monitoring)

**Database Schema**:

```sql
-- Main metrics table (one row per file)
CREATE TABLE file_metrics (
    file_path TEXT PRIMARY KEY,
    language TEXT NOT NULL,

    -- Size metrics
    total_lines INTEGER NOT NULL,
    code_lines INTEGER NOT NULL,
    comment_lines INTEGER DEFAULT 0,
    blank_lines INTEGER DEFAULT 0,

    -- Complexity metrics
    cognitive_complexity_total INTEGER NOT NULL,
    cognitive_complexity_max INTEGER NOT NULL,
    cognitive_complexity_avg REAL NOT NULL,
    cyclomatic_complexity_total INTEGER NOT NULL,
    cyclomatic_complexity_max INTEGER NOT NULL,

    -- Coupling metrics (Phase 3)
    efferent_coupling INTEGER DEFAULT 0,
    afferent_coupling INTEGER DEFAULT 0,
    instability REAL DEFAULT 0.0,

    -- Counts
    function_count INTEGER NOT NULL,
    class_count INTEGER NOT NULL,
    method_count INTEGER DEFAULT 0,

    -- Quality indicators
    smell_count INTEGER DEFAULT 0,
    complexity_grade_distribution TEXT,  -- JSON: {"A": 5, "B": 3, "C": 1}

    -- Metadata
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_hash TEXT,  -- Content hash for change detection

    CHECK (cognitive_complexity_total >= 0),
    CHECK (function_count >= 0)
);

-- Import relationships (for coupling analysis)
CREATE TABLE file_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    target_module TEXT NOT NULL,
    import_type TEXT NOT NULL,  -- 'internal', 'external', 'stdlib'
    line_number INTEGER,

    FOREIGN KEY (source_file) REFERENCES file_metrics(file_path) ON DELETE CASCADE
);

-- Project-wide snapshots (for trend tracking - Issue #25)
CREATE TABLE project_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Size totals
    total_files INTEGER NOT NULL,
    total_lines INTEGER NOT NULL,
    total_code_lines INTEGER NOT NULL,
    total_functions INTEGER NOT NULL,
    total_classes INTEGER NOT NULL,

    -- Complexity aggregates
    avg_cognitive_complexity REAL NOT NULL,
    max_cognitive_complexity INTEGER NOT NULL,

    -- Quality metrics
    total_smells INTEGER NOT NULL,
    smell_distribution TEXT,  -- JSON: {"long_method": 10, "deep_nesting": 5}

    -- Coupling (Phase 3)
    circular_dependency_count INTEGER DEFAULT 0,
    avg_instability REAL DEFAULT 0.0,

    -- Debt estimate
    estimated_debt_hours REAL DEFAULT 0.0,

    -- Language breakdown
    language_distribution TEXT  -- JSON: {"python": 45, "javascript": 30}
);

-- Individual code smells (for detailed reporting)
CREATE TABLE code_smells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    function_name TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    smell_type TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'warning', 'error'
    message TEXT NOT NULL,
    metric_value INTEGER,    -- e.g., actual complexity score
    threshold INTEGER,       -- e.g., configured threshold
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (file_path) REFERENCES file_metrics(file_path) ON DELETE CASCADE
);

-- Circular dependencies (Phase 3 - Issue #23)
CREATE TABLE circular_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_hash TEXT UNIQUE NOT NULL,
    cycle_files TEXT NOT NULL,    -- JSON array of file paths
    cycle_length INTEGER NOT NULL,
    severity TEXT NOT NULL,       -- 'error', 'warning'
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_file_metrics_language ON file_metrics(language);
CREATE INDEX idx_file_metrics_complexity ON file_metrics(cognitive_complexity_max);
CREATE INDEX idx_file_metrics_smells ON file_metrics(smell_count);
CREATE INDEX idx_file_metrics_indexed ON file_metrics(indexed_at);

CREATE INDEX idx_imports_source ON file_imports(source_file);
CREATE INDEX idx_imports_target ON file_imports(target_module);

CREATE INDEX idx_smells_type ON code_smells(smell_type);
CREATE INDEX idx_smells_file ON code_smells(file_path);
CREATE INDEX idx_smells_severity ON code_smells(severity);

CREATE INDEX idx_snapshots_date ON project_snapshots(snapshot_at);
```

**Python API Design**:

```python
# src/mcp_vector_search/analysis/storage/metrics_db.py

from pathlib import Path
import sqlite3
from typing import Any, Optional
from contextlib import contextmanager

class MetricsStore:
    """SQLite-backed persistent metrics storage."""

    def __init__(self, db_path: Path):
        """Initialize metrics store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            # Execute schema SQL from above
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Write operations
    def store_file_metrics(self, metrics: FileMetrics) -> None:
        """Store or update file-level metrics."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO file_metrics (
                    file_path, language, total_lines, code_lines,
                    cognitive_complexity_total, cognitive_complexity_max,
                    cognitive_complexity_avg, cyclomatic_complexity_total,
                    cyclomatic_complexity_max, function_count, class_count,
                    smell_count, indexed_at, file_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.file_path,
                metrics.language,
                metrics.total_lines,
                metrics.code_lines,
                metrics.cognitive_complexity_total,
                metrics.cognitive_complexity_max,
                metrics.cognitive_complexity_avg,
                metrics.cyclomatic_complexity_total,
                metrics.cyclomatic_complexity_max,
                metrics.function_count,
                metrics.class_count,
                metrics.smell_count,
                metrics.indexed_at,
                metrics.file_hash
            ))

    def store_project_snapshot(self, metrics: ProjectMetrics) -> int:
        """Store project-wide snapshot for trend tracking.

        Returns:
            Snapshot ID
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO project_snapshots (
                    total_files, total_lines, total_code_lines,
                    total_functions, total_classes,
                    avg_cognitive_complexity, max_cognitive_complexity,
                    total_smells, estimated_debt_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.total_files,
                metrics.total_lines,
                metrics.total_code_lines,
                metrics.total_functions,
                metrics.total_classes,
                metrics.avg_cognitive_complexity,
                metrics.max_cognitive_complexity,
                metrics.total_smells,
                metrics.estimated_debt_hours
            ))
            return cursor.lastrowid

    def store_code_smell(self, smell: CodeSmell) -> None:
        """Store individual code smell detection."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO code_smells (
                    file_path, function_name, start_line, end_line,
                    smell_type, severity, message, metric_value, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                smell.file_path,
                smell.function_name,
                smell.start_line,
                smell.end_line,
                smell.smell_type,
                smell.severity.value,
                smell.message,
                smell.metric_value,
                smell.threshold
            ))

    # Read operations
    def get_file_metrics(self, file_path: str) -> Optional[FileMetrics]:
        """Retrieve metrics for a specific file."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM file_metrics WHERE file_path = ?",
                (file_path,)
            ).fetchone()

            if row:
                return FileMetrics.from_db_row(row)
            return None

    def get_latest_snapshot(self) -> Optional[ProjectMetrics]:
        """Get most recent project snapshot."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM project_snapshots
                ORDER BY snapshot_at DESC
                LIMIT 1
            """).fetchone()

            if row:
                return ProjectMetrics.from_db_row(row)
            return None

    def get_snapshots_since(
        self,
        since: datetime,
        limit: int = 100
    ) -> list[ProjectMetrics]:
        """Get project snapshots since a given date."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM project_snapshots
                WHERE snapshot_at >= ?
                ORDER BY snapshot_at ASC
                LIMIT ?
            """, (since, limit)).fetchall()

            return [ProjectMetrics.from_db_row(row) for row in rows]

    def get_smells_by_type(self, smell_type: str) -> list[CodeSmell]:
        """Get all smells of a specific type."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM code_smells
                WHERE smell_type = ?
                ORDER BY severity DESC, file_path
            """, (smell_type,)).fetchall()

            return [CodeSmell.from_db_row(row) for row in rows]

    # Cleanup operations
    def cleanup_old_snapshots(self, keep_days: int = 90) -> int:
        """Delete snapshots older than keep_days.

        Returns:
            Number of snapshots deleted
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                DELETE FROM project_snapshots
                WHERE snapshot_at < datetime('now', ? || ' days')
            """, (f"-{keep_days}",))
            return cursor.rowcount
```

**Integration Points**:

1. **Indexer Integration** (`src/mcp_vector_search/core/indexer.py`):
   ```python
   from ..analysis.storage.metrics_db import MetricsStore

   class TreeSitterIndexer:
       def __init__(self, metrics_store: Optional[MetricsStore] = None):
           self.metrics_store = metrics_store or self._default_store()

       async def index_file(self, file_path: Path) -> None:
           # ... existing indexing logic ...

           # Store metrics in SQLite
           if self.metrics_store:
               self.metrics_store.store_file_metrics(file_metrics)

               for smell in code_smells:
                   self.metrics_store.store_code_smell(smell)
   ```

2. **Analyze Command** (`src/mcp_vector_search/cli/commands/analyze.py`):
   ```python
   @click.command()
   @click.option("--snapshot", is_flag=True, help="Create project snapshot")
   def analyze(snapshot: bool):
       """Run code analysis."""

       # ... existing analysis ...

       if snapshot:
           metrics_store.store_project_snapshot(project_metrics)
           console.print("✅ Project snapshot saved")
   ```

**Database Location**:
```
.mcp-vector-search/
├── chroma_db/              # Existing ChromaDB
└── metrics.db              # NEW: SQLite metrics store
```

**Migration Strategy**:
- Database auto-created on first use
- Schema version table for future migrations:
  ```sql
  CREATE TABLE schema_version (
      version INTEGER PRIMARY KEY,
      applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

**Testing Requirements**:
- Unit tests for all CRUD operations
- Test schema creation and migration
- Test concurrent access (indexing + querying)
- Performance test with 10k file metrics

**Estimated Effort**: 3-4 days

---

### Issue #25: Trend Tracking

**Title**: Implement historical trend tracking with snapshot comparison
**Dependencies**: **#24 (SQLite metrics store - CRITICAL)**
**Complexity**: **LOW** (once #24 complete)
**Implementation Priority**: **6th**

#### Technical Requirements

**Definition**: Track how project metrics change over time by comparing snapshots.

**Use Cases**:
1. **Sprint Progress**: "Did we reduce technical debt this sprint?"
2. **Regression Detection**: "Complexity increased by 15% since last release"
3. **Refactoring Impact**: "Before/after comparison of refactored modules"

**Data Already Available** (from #24):
```sql
SELECT * FROM project_snapshots ORDER BY snapshot_at DESC LIMIT 30;
```

**Trend Analysis Functions**:

```python
# src/mcp_vector_search/analysis/trends.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class TrendPoint:
    """Single data point in a trend line."""
    timestamp: datetime
    value: float
    label: str  # e.g., "Sprint 23", "v1.2.0"

@dataclass
class MetricTrend:
    """Trend analysis for a single metric."""
    metric_name: str
    points: list[TrendPoint]

    # Computed statistics
    current_value: float
    previous_value: float
    change_absolute: float
    change_percent: float
    trend_direction: str  # "improving", "declining", "stable"

    @classmethod
    def from_snapshots(
        cls,
        metric_name: str,
        snapshots: list[ProjectMetrics],
        metric_extractor: callable
    ) -> "MetricTrend":
        """Build trend from snapshot history.

        Args:
            metric_name: Display name for metric
            snapshots: List of project snapshots (chronological)
            metric_extractor: Function to extract value from snapshot

        Returns:
            Trend analysis with statistics
        """
        points = [
            TrendPoint(
                timestamp=s.snapshot_at,
                value=metric_extractor(s),
                label=s.snapshot_at.strftime("%Y-%m-%d")
            )
            for s in snapshots
        ]

        current = points[-1].value
        previous = points[-2].value if len(points) > 1 else current

        change_abs = current - previous
        change_pct = (change_abs / previous * 100) if previous != 0 else 0.0

        # Determine trend direction
        if abs(change_pct) < 5:
            direction = "stable"
        elif change_pct < 0:
            # For smells/complexity, decrease is good
            direction = "improving" if metric_name in ["smells", "complexity"] else "declining"
        else:
            direction = "declining" if metric_name in ["smells", "complexity"] else "improving"

        return cls(
            metric_name=metric_name,
            points=points,
            current_value=current,
            previous_value=previous,
            change_absolute=change_abs,
            change_percent=change_pct,
            trend_direction=direction
        )

class TrendAnalyzer:
    """Analyze metric trends over time."""

    def __init__(self, metrics_store: MetricsStore):
        self.store = metrics_store

    def get_recent_trends(
        self,
        days: int = 30,
        metrics: Optional[list[str]] = None
    ) -> dict[str, MetricTrend]:
        """Get trends for recent period.

        Args:
            days: Number of days to analyze
            metrics: List of metric names (None = all)

        Returns:
            Dictionary of metric_name -> MetricTrend
        """
        since = datetime.now() - timedelta(days=days)
        snapshots = self.store.get_snapshots_since(since)

        if len(snapshots) < 2:
            return {}  # Need at least 2 snapshots for trend

        default_metrics = [
            ("total_smells", lambda s: s.total_smells),
            ("avg_complexity", lambda s: s.avg_cognitive_complexity),
            ("debt_hours", lambda s: s.estimated_debt_hours),
            ("total_files", lambda s: s.total_files),
            ("circular_deps", lambda s: s.circular_dependency_count),
        ]

        trends = {}
        for metric_name, extractor in default_metrics:
            if metrics is None or metric_name in metrics:
                trends[metric_name] = MetricTrend.from_snapshots(
                    metric_name, snapshots, extractor
                )

        return trends

    def compare_snapshots(
        self,
        baseline_id: int,
        current_id: int
    ) -> dict[str, Any]:
        """Compare two specific snapshots.

        Returns:
            Comparison report with deltas
        """
        baseline = self.store.get_snapshot_by_id(baseline_id)
        current = self.store.get_snapshot_by_id(current_id)

        return {
            "baseline": baseline.to_dict(),
            "current": current.to_dict(),
            "changes": {
                "smells": {
                    "from": baseline.total_smells,
                    "to": current.total_smells,
                    "delta": current.total_smells - baseline.total_smells,
                    "percent": (current.total_smells - baseline.total_smells)
                               / baseline.total_smells * 100
                },
                # ... similar for other metrics ...
            }
        }
```

**CLI Integration**:

```bash
# Create snapshot (automatic during indexing)
mcp-vector-search index --snapshot

# View recent trends
mcp-vector-search analyze --trends --days 30

# Compare against baseline
mcp-vector-search analyze --baseline v1.0.0

# Export trend data
mcp-vector-search analyze --trends --export json > trends.json
```

**Console Output**:
```
📈 Trend Analysis (Last 30 days)

Code Smells: 23 → 18 (↓ 22% ✅ improving)
  ▁▂▃▅█▇▆▅▃▂▁

Avg Complexity: 6.8 → 6.2 (↓ 9% ✅ improving)
  ▁▂▃▄▅▆▇█▇▆▅

Technical Debt: 22h → 18h (↓ 18% ✅ improving)

Circular Dependencies: 3 → 1 (↓ 67% ✅ improving)

Overall: 📈 Quality improving
```

**Testing Requirements**:
- Unit tests with mock snapshot data
- Test trend calculation accuracy
- Test edge cases (single snapshot, no change)

**Estimated Effort**: 2 days

---

### Issue #26: LCOM4 Cohesion Metric

**Title**: Implement LCOM4 (Lack of Cohesion of Methods) metric
**Dependencies**: #2 (metric dataclasses - ✅ complete), #8 (indexer integration)
**Complexity**: **HIGH**
**Implementation Priority**: **7th**

#### Technical Requirements

**Definition**: LCOM4 measures class cohesion by counting connected components in the method-variable access graph.

**Theory**:
- **LCOM4 = 1**: Highly cohesive (all methods share variables)
- **LCOM4 = 2+**: Class should be split into separate classes
- **LCOM4 = method_count**: No cohesion (each method independent)

**Algorithm**:

1. **Build Graph**: Create undirected graph where:
   - Nodes = methods in class
   - Edges connect methods that:
     - Access same instance variable (`self.x`)
     - Call each other

2. **Count Components**: Use Union-Find or DFS to count connected components

```python
from dataclasses import dataclass, field
from typing import Set, Dict, List

@dataclass
class MethodInfo:
    """Information about a method for cohesion analysis."""
    name: str
    accessed_variables: Set[str]
    called_methods: Set[str]

class LCOM4Calculator:
    """Calculate LCOM4 cohesion metric for a class."""

    def __init__(self):
        self.methods: Dict[str, MethodInfo] = {}

    def add_method(
        self,
        name: str,
        accessed_vars: Set[str],
        called_methods: Set[str]
    ) -> None:
        """Register a method with its dependencies."""
        self.methods[name] = MethodInfo(
            name=name,
            accessed_variables=accessed_vars,
            called_methods=called_methods
        )

    def calculate(self) -> int:
        """Calculate LCOM4 using Union-Find.

        Returns:
            Number of connected components (LCOM4 score)
        """
        if len(self.methods) <= 1:
            return 1  # Single method = cohesive by definition

        # Union-Find data structure
        parent = {name: name for name in self.methods.keys()}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build connections
        method_list = list(self.methods.keys())
        for i, method1 in enumerate(method_list):
            m1 = self.methods[method1]

            for method2 in method_list[i+1:]:
                m2 = self.methods[method2]

                # Connect if they share variables
                if m1.accessed_variables & m2.accessed_variables:
                    union(method1, method2)

                # Connect if one calls the other
                if method2 in m1.called_methods or method1 in m2.called_methods:
                    union(method1, method2)

        # Count unique components
        components = len(set(find(m) for m in self.methods.keys()))
        return components

    def get_components(self) -> List[List[str]]:
        """Get the actual connected components (method groups).

        Returns:
            List of method groups (each group = list of method names)
        """
        if len(self.methods) <= 1:
            return [list(self.methods.keys())]

        # Run calculation first to populate Union-Find
        self.calculate()

        # Group methods by component
        from collections import defaultdict
        parent = {name: name for name in self.methods.keys()}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        components: Dict[str, List[str]] = defaultdict(list)
        for method in self.methods.keys():
            root = find(method)
            components[root].append(method)

        return list(components.values())
```

**Tree-sitter Integration**:

```python
# During class traversal in indexer

class LCOM4Collector(MetricCollector):
    """Collect cohesion metrics for classes."""

    def __init__(self):
        self._current_class: Optional[LCOM4Calculator] = None
        self._current_method: Optional[str] = None
        self._accessed_vars: Set[str] = set()
        self._called_methods: Set[str] = set()

    @property
    def name(self) -> str:
        return "lcom4"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Collect variable accesses and method calls."""

        # Entering a class
        if node.type == "class_definition":
            self._current_class = LCOM4Calculator()

        # Entering a method
        elif node.type in ("function_definition", "method_definition"):
            self._current_method = extract_function_name(node, context)
            self._accessed_vars = set()
            self._called_methods = set()

        # Inside a method, track variable accesses
        elif self._current_method:
            if node.type == "attribute":  # self.x
                var_name = extract_attribute_name(node, context)
                if var_name and var_name.startswith("self."):
                    self._accessed_vars.add(var_name[5:])  # Remove "self."

            elif node.type == "call":  # self.method()
                call_name = extract_call_name(node, context)
                if call_name and call_name.startswith("self."):
                    self._called_methods.add(call_name[5:])

    def finalize_function(self, node: Node, context: CollectorContext) -> dict[str, Any]:
        """Register method with class calculator."""
        if self._current_class and self._current_method:
            self._current_class.add_method(
                self._current_method,
                self._accessed_vars,
                self._called_methods
            )

        # Reset method state
        self._current_method = None
        self._accessed_vars = set()
        self._called_methods = set()

        return {}  # Metrics returned at class level

    def finalize_class(self, node: Node, context: CollectorContext) -> dict[str, Any]:
        """Calculate LCOM4 for completed class."""
        if not self._current_class:
            return {}

        lcom4 = self._current_class.calculate()
        components = self._current_class.get_components()

        result = {
            "lcom4": lcom4,
            "lcom4_components": len(components),
            "lcom4_method_groups": components,
            "is_cohesive": lcom4 == 1
        }

        # Reset for next class
        self._current_class = None

        return result
```

**Data Model Extension**:

```python
@dataclass
class ChunkMetrics:
    # ... existing fields ...

    # NEW: Cohesion metrics (class-level only)
    lcom4: Optional[int] = None
    lcom4_components: Optional[int] = None
    is_cohesive: bool = True
```

**Testing Requirements**:
- Unit tests with sample classes (known LCOM4)
- Test edge cases: single method, no shared variables
- Multi-language support (Python, JS/TS classes)
- Performance test on large classes (50+ methods)

**Estimated Effort**: 4-5 days

---

## Implementation Priority & Critical Path

### Recommended Implementation Order

**Priority ranking based on dependencies and value:**

1. **#24 SQLite Metrics Store** ⭐ CRITICAL
   - **Why first**: Foundation for #25 (trend tracking)
   - **Complexity**: Medium
   - **Effort**: 3-4 days
   - **Blockers**: None
   - **Value**: Enables historical analysis

2. **#20 Efferent Coupling Collector**
   - **Why second**: Foundation for #21, #22, #23
   - **Complexity**: Medium
   - **Effort**: 2-3 days
   - **Blockers**: None (dataclasses exist)
   - **Value**: First step toward dependency analysis

3. **#21 Afferent Coupling Collector**
   - **Why third**: Depends on #20's import data
   - **Complexity**: Medium-High
   - **Effort**: 3-4 days
   - **Blockers**: #20
   - **Value**: Completes coupling analysis

4. **#22 Instability Index**
   - **Why fourth**: Simple calculation once coupling exists
   - **Complexity**: Low
   - **Effort**: 1 day
   - **Blockers**: #20, #21
   - **Value**: Actionable architectural metric

5. **#23 Circular Dependency Detection**
   - **Why fifth**: Architectural quality gate
   - **Complexity**: High
   - **Effort**: 3-4 days
   - **Blockers**: #20 (needs dependency graph)
   - **Value**: Prevents architectural debt

6. **#25 Trend Tracking**
   - **Why sixth**: Leverage completed storage
   - **Complexity**: Low
   - **Effort**: 2 days
   - **Blockers**: #24
   - **Value**: Demonstrates ROI of metrics

7. **#26 LCOM4 Cohesion Metric**
   - **Why seventh**: Advanced metric, standalone
   - **Complexity**: High
   - **Effort**: 4-5 days
   - **Blockers**: None (can be parallel)
   - **Value**: Class design quality

### Critical Path Diagram

```
#24 SQLite Store (3-4d)
  │
  ├──> #25 Trend Tracking (2d)
  │
  └──> (parallel with coupling work)

#20 Efferent Coupling (2-3d)
  │
  ├──> #21 Afferent Coupling (3-4d)
  │      │
  │      ├──> #22 Instability (1d)
  │      │
  │      └──> #23 Circular Deps (3-4d)
  │
  └──> (parallel stream)

#26 LCOM4 (4-5d) [can be parallel]

Total Sequential Time: 18-22 days
With Parallelization: 12-15 days
```

---

## Completion Criteria

### Definition of Done for Phase 3

Each issue is complete when:

✅ **Code Complete**:
- Implementation follows collector pattern
- Type hints and docstrings present
- Passes mypy strict mode

✅ **Tested**:
- Unit tests with 80%+ coverage
- Integration tests with real codebases
- Multi-language validation (Python, JS/TS)

✅ **Integrated**:
- Collectors registered in indexer
- Metrics stored in SQLite
- CLI commands functional
- Console reporter displays metrics

✅ **Documented**:
- API documentation updated
- CHANGELOG.md entry
- Examples in docs/guides/

✅ **Validated**:
- Metrics match reference implementations (SonarQube, etc.)
- Performance benchmarks pass (<10ms overhead per 1000 LOC)

### Phase 3 Success Metrics

**At v0.19.0 release, users should be able to:**

1. ✅ View efferent/afferent coupling for any file
2. ✅ See instability index for all modules
3. ✅ Detect circular dependencies automatically
4. ✅ Track metrics trends over 30/60/90 days
5. ✅ Compare current metrics against baseline
6. ✅ Identify low-cohesion classes with LCOM4
7. ✅ Export dependency graph for visualization

**Performance Targets**:
- Full project analysis (1000 files): <30 seconds
- SQLite storage overhead: <5% indexing time
- Trend query response: <100ms

**Quality Targets**:
- Zero false positives on circular dependency detection
- Coupling metrics match static analysis tools
- SQLite database size <10MB for typical project

---

## Technical Debt & Risks

### Known Risks

1. **Circular Dependency False Positives**
   - Risk: Conditional imports might create cycles that don't actually exist at runtime
   - Mitigation: Document limitations, allow ignore patterns

2. **SQLite Concurrency**
   - Risk: Indexing + CLI query = lock contention
   - Mitigation: Use WAL mode, short transactions

3. **LCOM4 Accuracy**
   - Risk: Dynamic method calls (`getattr`) not detected
   - Mitigation: Document as best-effort, focus on static analysis

4. **Performance on Large Codebases**
   - Risk: 10k+ files might strain memory/SQLite
   - Mitigation: Batch processing, database pagination

### Future Enhancements (Post-Phase 3)

- **Import refactoring suggestions**: Auto-suggest fixes for circular deps
- **Dependency graph visualization**: Interactive HTML export
- **API stability tracking**: Monitor public API coupling over time
- **Machine learning**: Predict complexity increase from diffs

---

## References

### Design Documents

- [Structural Analysis Design](./mcp-vector-search-structural-analysis-design.md) - Full technical specification
- [Project Overview](../projects/structural-code-analysis.md) - Phase tracking

### External References

- **Efferent/Afferent Coupling**: "Object-Oriented Metrics in Practice" (Lanza & Marinescu)
- **Instability Index**: "Agile Software Development" (Robert C. Martin)
- **LCOM4**: "A Metrics Suite for Object-Oriented Design" (Chidamber & Kemerer)
- **Tarjan's Algorithm**: "Depth-First Search and Linear Graph Algorithms" (1972)

### Similar Tools

- **SonarQube**: Coupling metrics, circular dependency detection
- **Code Climate**: Technical debt estimation
- **Sourcegraph**: Dependency graph analysis
- **Understand (SciTools)**: LCOM4 and coupling metrics

---

## Appendix: SQLite Performance Tuning

### Recommended Pragmas

```sql
-- Enable Write-Ahead Logging for better concurrency
PRAGMA journal_mode = WAL;

-- Increase cache size (default 2MB -> 10MB)
PRAGMA cache_size = -10000;

-- Use memory for temp tables
PRAGMA temp_store = MEMORY;

-- Synchronous = NORMAL for balance of safety + speed
PRAGMA synchronous = NORMAL;

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;
```

### Index Strategy

**Essential Indexes** (created in schema):
- `file_metrics.file_path` (PRIMARY KEY)
- `file_metrics.indexed_at` (for trend queries)
- `file_imports.target_module` (for afferent coupling lookup)
- `code_smells.smell_type` (for smell distribution)

**Query Optimization**:
```sql
-- Use EXPLAIN QUERY PLAN to verify index usage
EXPLAIN QUERY PLAN
SELECT COUNT(*) FROM file_imports WHERE target_module = ?;

-- Expected: SEARCH TABLE file_imports USING INDEX idx_imports_target
```

### Backup Strategy

```python
def backup_metrics_db(source_path: Path, backup_dir: Path) -> Path:
    """Create timestamped backup of metrics database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"metrics_backup_{timestamp}.db"

    # Use SQLite online backup API
    source = sqlite3.connect(source_path)
    backup = sqlite3.connect(backup_path)

    with backup:
        source.backup(backup)

    source.close()
    backup.close()

    return backup_path
```

---

## Summary

Phase 3 introduces **persistent metrics storage** and **cross-file dependency analysis** to mcp-vector-search. The critical path starts with **Issue #24 (SQLite metrics store)**, which enables trend tracking and provides a foundation for all other Phase 3 features.

**Key Deliverables**:
1. SQLite database for historical metrics
2. Coupling analysis (efferent, afferent, instability)
3. Circular dependency detection
4. Trend tracking over time
5. LCOM4 cohesion metric

**Implementation Timeline**: 18-22 days sequential, 12-15 days with parallelization

**Next Steps**: Start with Issue #24 to unblock trend tracking, then proceed with coupling analysis (#20-#23).
