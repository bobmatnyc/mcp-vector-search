# Issue #14: Code Smell Detection - Implementation Requirements

**Research Date**: December 11, 2024
**Researcher**: Claude (Research Agent)
**Status**: Ready for Implementation
**Related Design**: [Structural Code Analysis Design](mcp-vector-search-structural-analysis-design.md)

---

## Executive Summary

Issue #14 requires implementing code smell detection as part of the Structural Code Analysis feature (Phase 1/5). The design document provides comprehensive specifications, and significant infrastructure already exists. This research identifies **5 high-priority code smells** ready for implementation using existing collectors and threshold configuration.

**Key Findings**:
- ✅ **Infrastructure Complete**: Metrics dataclasses, threshold configuration, and 5 collectors already implemented
- ✅ **CLI Integration Ready**: `analyze` command exists and can be extended
- ⚠️ **Missing Component**: No `smells.py` collector - this is the core deliverable for Issue #14
- 📋 **Implementation Strategy**: Create pattern-based smell detector using existing metrics

---

## 1. Issue #14 Overview

### Objective
Implement code smell detection for the `mcp-vector-search analyze` command to identify common maintainability issues in codebases.

### Scope
Based on design document section **"Tier 3: Code Smells (Pattern-Based)"**, implement detection for:

1. **Long Method** - Functions with too many lines or high complexity
2. **Deep Nesting** - Excessive nesting depth (>4 levels)
3. **Long Parameter List** - Functions with too many parameters (>5)
4. **God Class** - Classes with too many methods (>20) and low cohesion
5. **Empty Catch** - Exception handlers with no error handling logic

---

## 2. Existing Infrastructure Analysis

### 2.1 Metrics Dataclasses (`src/mcp_vector_search/analysis/metrics.py`)

**Status**: ✅ **Complete** - Ready for integration

```python
@dataclass
class ChunkMetrics:
    """Metrics for a single code chunk (function/class/method)."""

    cognitive_complexity: int = 0
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0
    lines_of_code: int = 0

    # Code smells detected
    smells: list[str] = field(default_factory=list)  # ✅ Already exists

    # Computed grades (A-F scale)
    complexity_grade: str = field(init=False, default="A")

@dataclass
class FileMetrics:
    """Aggregated metrics for an entire file."""
    # ... includes smell_count and smells dict aggregation

@dataclass
class ProjectMetrics:
    """Project-wide metric aggregates."""
    # ... includes smell distribution statistics
```

**Key Observation**: The `smells` field exists but is **never populated**. Issue #14 will implement the logic to detect and populate this field.

### 2.2 Metric Collectors (`src/mcp_vector_search/analysis/collectors/`)

**Status**: ✅ **5 collectors implemented** - Sufficient for all 5 code smells

| Collector | Status | Provides Data For Smell Detection |
|-----------|--------|-----------------------------------|
| `CognitiveComplexityCollector` | ✅ Complete | Long Method (high complexity) |
| `CyclomaticComplexityCollector` | ✅ Complete | Long Method (decision paths) |
| `NestingDepthCollector` | ✅ Complete | Deep Nesting |
| `ParameterCountCollector` | ✅ Complete | Long Parameter List |
| `MethodCountCollector` | ✅ Complete | God Class |

**Example Collector Output**:
```python
# CognitiveComplexityCollector.finalize_function() returns:
{
    "cognitive_complexity": 23,
}

# NestingDepthCollector.finalize_function() returns:
{
    "max_nesting_depth": 5,  # Exceeds threshold of 4
}

# ParameterCountCollector.finalize_function() returns:
{
    "parameter_count": 7,  # Exceeds threshold of 5
}
```

### 2.3 Threshold Configuration (`src/mcp_vector_search/config/thresholds.py`)

**Status**: ✅ **Complete with test coverage** - Ready for use

```python
@dataclass
class SmellThresholds:
    """Thresholds for code smell detection."""

    long_method_lines: int = 50           # LOC threshold
    too_many_parameters: int = 5          # Parameter count threshold
    deep_nesting_depth: int = 4           # Nesting depth threshold
    high_complexity: int = 15             # Cognitive complexity threshold
    god_class_methods: int = 20           # Method count threshold
    feature_envy_external_calls: int = 5  # Future use
```

**YAML Configuration** (`src/mcp_vector_search/config/default_thresholds.yaml`):
```yaml
smells:
  long_method_lines: 50
  too_many_parameters: 5
  deep_nesting_depth: 4
  high_complexity: 15
  god_class_methods: 20
```

**Test Coverage**: 100% coverage in `tests/unit/config/test_thresholds.py` (21 test cases)

### 2.4 CLI Analyze Command (`src/mcp_vector_search/cli/commands/analyze.py`)

**Status**: ✅ **Implemented** - Needs integration with smell detector

**Current Flow**:
```
1. Find analyzable files
2. Parse files → Extract chunks
3. Run collectors → Collect metrics
4. Create ChunkMetrics (cognitive, cyclomatic, nesting, parameters)
5. Aggregate to FileMetrics and ProjectMetrics
6. Display results via ConsoleReporter
```

**Integration Point**:
```python
# Line 388-395: After collecting metrics, detect smells
chunk_metrics = ChunkMetrics(
    cognitive_complexity=cognitive,
    cyclomatic_complexity=complexity,
    max_nesting_depth=0,
    parameter_count=param_count,
    lines_of_code=chunk.end_line - chunk.start_line + 1,
)

# ⚠️ MISSING: Smell detection logic should be called here
# chunk_metrics.smells = SmellDetector.detect(chunk_metrics, thresholds)

file_metrics.chunks.append(chunk_metrics)
```

### 2.5 Missing Component: `smells.py` Collector

**Status**: ⚠️ **NOT FOUND** - This is Issue #14's deliverable

**Expected Location**: `src/mcp_vector_search/analysis/collectors/smells.py`

**Design Specification** (from design doc):
```python
# Expected interface from design document
from .base import MetricCollector

class SmellDetector(MetricCollector):
    """Detects code smells based on metric thresholds."""

    @property
    def name(self) -> str:
        return "smell_detector"

    def collect_node(self, node, context, depth) -> None:
        """Track nodes for pattern-based smell detection."""
        pass

    def finalize_function(self, node, context) -> dict[str, Any]:
        """Return detected smells for completed function."""
        return {"smells": [...]}
```

---

## 3. Code Smells Implementation Requirements

### 3.1 Smell Detection Rules

Based on design document **Table: Code Smells (Pattern-Based)**:

| Smell | Detection Rule | Confidence | Implementation Complexity |
|-------|---------------|------------|---------------------------|
| **Long Method** | `LOC > 50 OR cognitive_complexity > 15` | High | 🟢 Simple |
| **Deep Nesting** | `max_nesting_depth > 4` | High | 🟢 Simple |
| **Long Parameter List** | `parameter_count > 5` | High | 🟢 Simple |
| **God Class** | `method_count > 20 AND loc > 500` | Medium | 🟡 Moderate |
| **Empty Catch** | `catch block with pass/empty body` | High | 🟡 Moderate (requires AST inspection) |

### 3.2 Detection Logic Examples

#### 3.2.1 Long Method (Simple)
```python
def detect_long_method(metrics: ChunkMetrics, thresholds: SmellThresholds) -> str | None:
    """Detect long method smell.

    Rule: LOC > 50 OR cognitive_complexity > 15
    """
    if metrics.lines_of_code > thresholds.long_method_lines:
        return f"long_method:lines:{metrics.lines_of_code}"

    if metrics.cognitive_complexity > thresholds.high_complexity:
        return f"long_method:complexity:{metrics.cognitive_complexity}"

    return None
```

#### 3.2.2 Deep Nesting (Simple)
```python
def detect_deep_nesting(metrics: ChunkMetrics, thresholds: SmellThresholds) -> str | None:
    """Detect deep nesting smell.

    Rule: max_nesting_depth > 4
    """
    if metrics.max_nesting_depth > thresholds.deep_nesting_depth:
        return f"deep_nesting:depth:{metrics.max_nesting_depth}"

    return None
```

#### 3.2.3 Empty Catch (Moderate - AST Required)
```python
def detect_empty_catch(node: Node, context: CollectorContext) -> str | None:
    """Detect empty catch blocks.

    Rule: except/catch clause with only pass/empty body
    """
    language = context.language

    # Check if node is an exception handler
    if node.type in get_node_types(language, "except_clause"):
        # Find the body of the exception handler
        body = node.child_by_field_name("body")

        if body:
            # Check if body is empty or contains only 'pass'
            if len(body.children) == 0:
                return "empty_catch:no_handling"

            # Python: Check for 'pass' statement only
            if language == "python":
                if len(body.children) == 1 and body.children[0].type == "pass_statement":
                    return "empty_catch:pass_only"

            # JavaScript/TypeScript: Check for empty block
            if language in ["javascript", "typescript"]:
                if body.type == "statement_block" and len(body.children) <= 2:  # {} only
                    return "empty_catch:no_handling"

    return None
```

### 3.3 God Class Detection (Complex)

**Challenge**: Requires class-level aggregation, not just function-level metrics.

**Solution Approach**:
1. Track class-level context in `CollectorContext.current_class`
2. Accumulate method count using `MethodCountCollector`
3. Calculate class LOC by summing all methods + class body
4. Detect smell when finalizing class node

**Implementation Note**: God Class detection may be deferred to Phase 2 if class-level finalization is not yet implemented.

---

## 4. Integration Requirements

### 4.1 Smell Detector Architecture

**Option A: Standalone Detector (Recommended)**
```python
# src/mcp_vector_search/analysis/collectors/smells.py

class SmellDetector:
    """Standalone smell detector using post-processing approach."""

    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds = thresholds

    def detect_smells(self, metrics: ChunkMetrics, node: Node | None = None) -> list[str]:
        """Detect all smells for a code chunk.

        Args:
            metrics: Collected metrics for the chunk
            node: Optional AST node for pattern-based detection

        Returns:
            List of detected smell identifiers
        """
        smells = []

        # Metric-based smells (no AST required)
        if smell := self.detect_long_method(metrics):
            smells.append(smell)

        if smell := self.detect_deep_nesting(metrics):
            smells.append(smell)

        if smell := self.detect_long_parameter_list(metrics):
            smells.append(smell)

        # Pattern-based smells (AST required)
        if node:
            if smell := self.detect_empty_catch(node):
                smells.append(smell)

        return smells
```

**Option B: MetricCollector Interface (Alternative)**
```python
# src/mcp_vector_search/analysis/collectors/smells.py

class SmellCollector(MetricCollector):
    """Collector that detects code smells during AST traversal."""

    def __init__(self, thresholds: ThresholdConfig):
        self.thresholds = thresholds
        self._detected_smells: list[str] = []

    @property
    def name(self) -> str:
        return "smell_detector"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Collect pattern-based smells during traversal."""
        # Detect empty catch blocks
        if smell := self._detect_empty_catch_node(node, context):
            self._detected_smells.append(smell)

    def finalize_function(self, node: Node, context: CollectorContext) -> dict[str, Any]:
        """Return all detected smells for this function."""
        return {"smells": self._detected_smells}

    def reset(self) -> None:
        self._detected_smells = []
```

**Recommendation**: Use **Option A (Standalone Detector)** because:
- Simpler integration with existing codebase
- No need to modify indexer traversal logic
- Post-processing approach fits better with current architecture
- Can use metrics collected by other collectors

### 4.2 CLI Integration Points

**File**: `src/mcp_vector_search/cli/commands/analyze.py`

**Integration Steps**:

1. **Import SmellDetector**:
```python
from ...analysis.collectors.smells import SmellDetector
```

2. **Load Threshold Configuration**:
```python
# Line 154: After initializing parser_registry
from ...config.thresholds import ThresholdConfig

# Load thresholds from config file or use defaults
threshold_config = ThresholdConfig.load(
    project_root / ".mcp-vector-search" / "thresholds.yaml"
)

# Initialize smell detector
smell_detector = SmellDetector(threshold_config)
```

3. **Detect Smells in Analysis Loop** (Line 388-395):
```python
# Create chunk metrics
chunk_metrics = ChunkMetrics(
    cognitive_complexity=cognitive,
    cyclomatic_complexity=complexity,
    max_nesting_depth=0,
    parameter_count=param_count,
    lines_of_code=chunk.end_line - chunk.start_line + 1,
)

# NEW: Detect smells
chunk_metrics.smells = smell_detector.detect_smells(chunk_metrics)

file_metrics.chunks.append(chunk_metrics)
```

4. **Update ConsoleReporter** to display smells:
```python
# src/mcp_vector_search/analysis/reporters/console.py

def print_smells(self, project_metrics: ProjectMetrics) -> None:
    """Print code smell summary."""

    # Aggregate smells across all chunks
    smell_counts: dict[str, int] = {}
    smell_locations: list[tuple[str, str, int]] = []  # (file, smell, line)

    for file_path, file_metrics in project_metrics.files.items():
        for chunk in file_metrics.chunks:
            for smell in chunk.smells:
                smell_type = smell.split(":")[0]
                smell_counts[smell_type] = smell_counts.get(smell_type, 0) + 1
                smell_locations.append((file_path, smell, chunk.start_line))

    # Display smell summary
    console.print("\n[bold yellow]Code Smells Detected[/bold yellow]")

    if not smell_counts:
        console.print("[green]✓ No code smells detected![/green]")
        return

    # Print smell counts
    table = Table(title="Smell Distribution")
    table.add_column("Smell Type", style="cyan")
    table.add_column("Count", justify="right", style="yellow")

    for smell_type, count in sorted(smell_counts.items(), key=lambda x: -x[1]):
        table.add_row(smell_type.replace("_", " ").title(), str(count))

    console.print(table)
```

---

## 5. Output Format Requirements

### 5.1 Console Output (Human-Readable)

**Expected Output** (based on design doc section "Status Command Extension"):
```
📊 Code Analysis Results
═══════════════════════════════════════════════════

Files: 247 | Functions: 608 | Classes: 89

Complexity Distribution:
  A (≤5)  ████████████████████████████████░░░░░░  68% (412 functions)
  B (≤10) ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  22% (134 functions)
  C (≤15) ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7% (42 functions)
  D (≤25) █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2% (15 functions)
  F (>25) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1% (5 functions)

⚠️  Code Smells: 23 detected
┌────────────────────────┬───────┐
│ Smell Type             │ Count │
├────────────────────────┼───────┤
│ Long Method            │     8 │
│ Deep Nesting           │     6 │
│ Long Parameter List    │     5 │
│ God Class              │     2 │
│ Empty Catch            │     2 │
└────────────────────────┴───────┘

🔥 Top 5 Complexity Hotspots:
1. src/auth/handler.py::validate_token (Complexity: 23, Grade: D)
   └─ Smells: long_method:complexity:23, deep_nesting:depth:5
2. src/api/routes.py::process_request (Complexity: 18, Grade: C)
   └─ Smells: long_parameter_list:count:7
...
```

### 5.2 JSON Output (Machine-Readable)

**Expected JSON Structure** (for `--json` flag):
```json
{
  "project_root": "/path/to/project",
  "analyzed_at": "2024-12-11T10:30:00Z",
  "summary": {
    "total_files": 247,
    "total_functions": 608,
    "total_classes": 89,
    "total_smells": 23
  },
  "smell_distribution": {
    "long_method": 8,
    "deep_nesting": 6,
    "long_parameter_list": 5,
    "god_class": 2,
    "empty_catch": 2
  },
  "files": [
    {
      "path": "src/auth/handler.py",
      "chunks": [
        {
          "name": "validate_token",
          "start_line": 45,
          "end_line": 112,
          "cognitive_complexity": 23,
          "max_nesting_depth": 5,
          "lines_of_code": 67,
          "complexity_grade": "D",
          "smells": [
            "long_method:complexity:23",
            "deep_nesting:depth:5"
          ]
        }
      ]
    }
  ]
}
```

---

## 6. Testing Requirements

### 6.1 Unit Tests

**File**: `tests/unit/analysis/collectors/test_smells.py`

**Required Test Cases**:
```python
class TestSmellDetector:
    """Test SmellDetector functionality."""

    def test_detect_long_method_by_lines(self):
        """Test detection of long method by line count."""
        metrics = ChunkMetrics(lines_of_code=60)
        thresholds = ThresholdConfig()
        detector = SmellDetector(thresholds)

        smells = detector.detect_smells(metrics)

        assert "long_method:lines:60" in smells

    def test_detect_long_method_by_complexity(self):
        """Test detection of long method by cognitive complexity."""
        metrics = ChunkMetrics(cognitive_complexity=20)
        thresholds = ThresholdConfig()
        detector = SmellDetector(thresholds)

        smells = detector.detect_smells(metrics)

        assert "long_method:complexity:20" in smells

    def test_detect_deep_nesting(self):
        """Test detection of deep nesting smell."""
        metrics = ChunkMetrics(max_nesting_depth=5)
        thresholds = ThresholdConfig()
        detector = SmellDetector(thresholds)

        smells = detector.detect_smells(metrics)

        assert "deep_nesting:depth:5" in smells

    def test_detect_long_parameter_list(self):
        """Test detection of long parameter list smell."""
        metrics = ChunkMetrics(parameter_count=7)
        thresholds = ThresholdConfig()
        detector = SmellDetector(thresholds)

        smells = detector.detect_smells(metrics)

        assert "long_parameter_list:count:7" in smells

    def test_no_smells_for_clean_code(self):
        """Test that clean code has no smells."""
        metrics = ChunkMetrics(
            cognitive_complexity=5,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=20,
        )
        thresholds = ThresholdConfig()
        detector = SmellDetector(thresholds)

        smells = detector.detect_smells(metrics)

        assert len(smells) == 0

    def test_custom_thresholds(self):
        """Test smell detection with custom thresholds."""
        metrics = ChunkMetrics(lines_of_code=60)

        # Custom threshold: allow up to 100 lines
        custom_thresholds = ThresholdConfig(
            smells=SmellThresholds(long_method_lines=100)
        )
        detector = SmellDetector(custom_thresholds)

        smells = detector.detect_smells(metrics)

        # Should NOT detect smell with custom threshold
        assert len(smells) == 0
```

### 6.2 Integration Tests

**File**: `tests/integration/cli/test_analyze_smells.py`

**Required Test Cases**:
```python
async def test_analyze_detects_long_method(tmp_path):
    """Test that analyze command detects long method smell."""
    # Create test file with long method
    test_file = tmp_path / "long_method.py"
    test_file.write_text("""
def very_long_function():
    # 60 lines of code
    pass
""")

    # Run analyze command
    result = await run_analysis(project_root=tmp_path)

    # Verify smell was detected
    assert result.total_smells > 0
    assert "long_method" in result.smell_distribution

async def test_analyze_json_output_includes_smells(tmp_path):
    """Test that JSON output includes smell information."""
    # Create test file with smells
    test_file = tmp_path / "smelly_code.py"
    test_file.write_text("""
def bad_function(a, b, c, d, e, f, g):  # Too many parameters
    if condition:
        if nested:
            if deeply_nested:
                if very_deeply_nested:
                    if extremely_nested:  # Too deep
                        pass
""")

    # Run analyze with JSON output
    result = await run_analysis(project_root=tmp_path, json_output=True)
    json_output = json.loads(result)

    # Verify JSON structure
    assert "smell_distribution" in json_output
    assert "long_parameter_list" in json_output["smell_distribution"]
    assert "deep_nesting" in json_output["smell_distribution"]
```

---

## 7. Implementation Roadmap

### Phase 1: Core Smell Detection (Issue #14)
**Estimated Effort**: 8-12 hours

1. **Create SmellDetector** (`smells.py`) - 3 hours
   - Implement standalone detector class
   - Add detection methods for all 5 smells
   - Write comprehensive docstrings

2. **Integrate with CLI** (`analyze.py`) - 2 hours
   - Load threshold configuration
   - Call smell detector in analysis loop
   - Pass detected smells to ChunkMetrics

3. **Update ConsoleReporter** - 2 hours
   - Add `print_smells()` method
   - Create smell distribution table
   - Integrate into main report flow

4. **Write Tests** - 3 hours
   - Unit tests for SmellDetector (15 test cases)
   - Integration tests for CLI (5 test cases)
   - Verify threshold configuration loading

5. **Documentation** - 2 hours
   - Update CLI help text
   - Add examples to README
   - Document smell detection rules

### Phase 2: Empty Catch Detection (Optional Enhancement)
**Estimated Effort**: 4-6 hours

1. **AST Pattern Detection** - 3 hours
   - Implement tree-sitter queries for exception handlers
   - Detect empty catch blocks in Python, JavaScript, TypeScript
   - Test across multiple languages

2. **Integration** - 1 hour
   - Add to SmellDetector
   - Update tests

3. **Documentation** - 1 hour
   - Add empty catch examples
   - Document language support

### Phase 3: God Class Detection (Future Work)
**Estimated Effort**: 6-8 hours

**Requires**: Class-level finalization in collector infrastructure (not yet implemented)

---

## 8. Recommendations

### 8.1 Immediate Actions (Issue #14)

1. ✅ **Create `smells.py`** with standalone SmellDetector class
2. ✅ **Implement metric-based smells** (Long Method, Deep Nesting, Long Parameter List)
3. ✅ **Integrate with analyze command** using post-processing approach
4. ✅ **Update ConsoleReporter** to display smell distribution
5. ✅ **Write comprehensive tests** (unit + integration)
6. ⚠️ **Defer Empty Catch and God Class** to Phase 2 if time-constrained

### 8.2 Implementation Best Practices

1. **Use Existing Infrastructure**: Don't reinvent the wheel
   - Leverage `ThresholdConfig` for all thresholds
   - Use `ChunkMetrics.smells` field (already exists)
   - Follow existing collector patterns

2. **Keep It Simple**: Start with metric-based detection
   - Long Method, Deep Nesting, Long Parameter List are trivial
   - Empty Catch requires AST inspection (defer if needed)
   - God Class requires class-level aggregation (defer if needed)

3. **Test Coverage**: Aim for 100% coverage
   - Test each smell detection rule independently
   - Test custom threshold configurations
   - Test edge cases (boundary values)

4. **Documentation**: Be thorough
   - Document detection rules clearly
   - Provide examples of each smell type
   - Explain how to customize thresholds

### 8.3 Future Enhancements

1. **Smell Severity Levels**: Classify smells as warning/error
2. **Smell Explanations**: Provide refactoring suggestions
3. **Smell Trends**: Track smell count over time
4. **CI/CD Integration**: Exit with error code if too many smells
5. **Additional Smells**: Feature Envy, Data Clumps, Switch Statements

---

## 9. Open Questions

1. **Should Empty Catch detection be included in Issue #14?**
   - Pro: Specified in design document as "High confidence"
   - Con: Requires AST inspection, increases complexity
   - **Recommendation**: Include if time permits, otherwise defer to Phase 2

2. **Should God Class detection be included in Issue #14?**
   - Pro: Specified in design document
   - Con: Requires class-level finalization (not yet implemented)
   - **Recommendation**: Defer to Phase 2 when class-level infrastructure exists

3. **Should smell detection fail the analyze command (exit code 1)?**
   - Current behavior: Always exit with 0
   - Design doc suggests: `--fail-on-smell` flag
   - **Recommendation**: Implement `--fail-on-smell` flag (respects `fail_on_smell_count` threshold)

4. **How should smells be formatted in JSON output?**
   - Option A: String format `"long_method:complexity:23"`
   - Option B: Structured object `{"type": "long_method", "reason": "complexity", "value": 23}`
   - **Recommendation**: Start with Option A (simpler), migrate to Option B if needed

---

## 10. References

- **Design Document**: `docs/research/mcp-vector-search-structural-analysis-design.md`
- **GitHub Project**: https://github.com/users/bobmatnyc/projects/13
- **Related Issues**: Issue #9 (ChromaDB Metadata), Issue #10 (Complexity Collectors)
- **SonarQube Cognitive Complexity Whitepaper**: Industry standard for complexity measurement
- **Tree-sitter Query Language**: For AST-based pattern detection

---

## Appendix A: Example Code Smells

### A.1 Long Method (Detected)
```python
def validate_user_and_authenticate(username, password, session_id, ip_address, user_agent):
    """This function has too many parameters and is too long."""
    # 60 lines of complex authentication logic
    if not username:
        return False
    if not password:
        return False
    # ... 50 more lines
    return True

# Detected smells:
# - long_method:lines:60
# - long_parameter_list:count:5
```

### A.2 Deep Nesting (Detected)
```python
def process_nested_data(data):
    """This function has excessive nesting."""
    if data:
        for item in data:
            if item.valid:
                if item.status == "active":
                    if item.permissions:
                        if item.permissions.can_access:
                            return True  # Nesting depth: 5
    return False

# Detected smell:
# - deep_nesting:depth:5
```

### A.3 Empty Catch (Detected)
```python
def risky_operation():
    """This function silently swallows exceptions."""
    try:
        perform_critical_operation()
    except Exception:
        pass  # Empty catch - no error handling!

# Detected smell:
# - empty_catch:pass_only
```

### A.4 Clean Code (No Smells)
```python
def calculate_total(items):
    """This is clean, simple code."""
    total = 0
    for item in items:
        total += item.price
    return total

# No smells detected ✓
```

---

**End of Research Report**
