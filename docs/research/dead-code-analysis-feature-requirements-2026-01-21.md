# Dead Code Analysis Feature - Implementation Requirements

**Research Date**: 2026-01-21
**Researcher**: Claude (Research Agent)
**Status**: Ready for Implementation Planning
**Complexity**: Medium

## Executive Summary

This research analyzes the feasibility of adding a `dead-code` analysis command to mcp-vector-search. The project already has **90% of the required infrastructure** in place through its existing call graph analysis, AST parsing, and code smell detection systems. A dead code analyzer can be implemented as a **focused addition** leveraging these existing components.

**Key Finding**: Dead code detection is highly feasible with **estimated 3-4 days** of implementation effort, primarily involving:
- Entry point detection logic (1-2 days)
- Graph reachability traversal (1 day)
- Output formatting and CLI integration (1 day)

---

## 1. Current Infrastructure Assessment

### ✅ 1.1 AST Parsing & Call Graph (Fully Implemented)

**Location**: `src/mcp_vector_search/core/relationships.py`

The project already extracts function calls using Python's AST module:

```python
def extract_function_calls(code: str) -> set[str]:
    """Extract actual function calls from Python code using AST.

    Returns set of function names that are actually called (not just mentioned).
    Avoids false positives from comments, docstrings, and string literals.
    """
    calls = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Handle direct calls: foo()
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                # Handle method calls: obj.foo() - extract 'foo'
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return calls
    except SyntaxError:
        return set()
```

**Strengths**:
- ✅ Avoids false positives from docstrings/comments (AST-based, not regex)
- ✅ Handles both direct calls (`foo()`) and method calls (`obj.foo()`)
- ✅ Used in `_compute_caller_relationships()` to build call graph
- ✅ Tested in `tests/unit/test_graph_builder_ast.py`

**Current Usage**:
```python
def _compute_caller_relationships(self, chunks: list[CodeChunk]) -> dict:
    """Compute which chunks call which other chunks."""
    caller_map = {}

    for chunk in code_chunks:
        function_name = chunk.function_name or chunk.class_name

        # Search other chunks that reference this function
        for other_chunk in chunks:
            # Extract actual function calls using AST
            actual_calls = extract_function_calls(other_chunk.content)

            # Check if this function is actually called
            if function_name in actual_calls:
                # Record caller relationship
                caller_map[chunk_id].append({
                    "file": other_file_path,
                    "chunk_id": other_chunk_id,
                    "name": other_name,
                    "type": other_chunk.chunk_type,
                })
```

**Data Structure**:
- `caller_map[chunk_id] = [caller_info_1, caller_info_2, ...]`
- Each `caller_info` contains: `file`, `chunk_id`, `name`, `type`
- Currently tracks **external callers only** (cross-file calls)
- Internal callers (same-file) are handled in visualization layer

### ✅ 1.2 Graph Traversal Algorithms (Fully Implemented)

**Location**: `src/mcp_vector_search/analysis/collectors/coupling.py`

The project has a complete **DFS-based cycle detection** implementation:

```python
class CircularDependencyDetector:
    """Detects circular dependencies using DFS with three-color marking.

    - WHITE: Unvisited node
    - GRAY: Node in current DFS path (cycle if revisited)
    - BLACK: Fully processed node

    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """

    def detect_cycles(self) -> list[CircularDependency]:
        """Detect all circular dependencies in the import graph."""
        self.colors = {node: NodeColor.WHITE for node in self.graph.adjacency_list}
        self.path = []
        self.cycles = []

        for node in self.graph.adjacency_list:
            if self.colors[node] == NodeColor.WHITE:
                self._dfs(node)

        return self.cycles

    def _dfs(self, node: str) -> None:
        """Depth-first search to detect cycles."""
        self.colors[node] = NodeColor.GRAY  # Mark as in-progress
        self.path.append(node)

        for neighbor in self.graph.get_neighbors(node):
            if self.colors[neighbor] == NodeColor.GRAY:
                # Cycle detected! neighbor is in current path
                self._record_cycle(neighbor)
            elif self.colors[neighbor] == NodeColor.WHITE:
                self._dfs(neighbor)

        self.path.pop()
        self.colors[node] = NodeColor.BLACK  # Mark as complete
```

**Reusability for Dead Code**:
- ✅ The DFS traversal logic can be **directly adapted** for reachability analysis
- ✅ Instead of detecting cycles, we track **which nodes are reachable** from entry points
- ✅ Same O(V+E) complexity applies to reachability traversal

**Proposed Adaptation**:
```python
class ReachabilityAnalyzer:
    """Analyze which functions are reachable from entry points."""

    def find_reachable(self, entry_points: list[str]) -> set[str]:
        """Find all functions reachable from entry points using BFS/DFS."""
        reachable = set()
        visited = set()

        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)
            reachable.add(node)

            for callee in self.call_graph.get_callees(node):
                dfs(callee)

        for entry in entry_points:
            dfs(entry)

        return reachable

    def find_unreachable(self, entry_points: list[str]) -> set[str]:
        """Find all functions NOT reachable from entry points."""
        all_functions = set(self.call_graph.nodes())
        reachable = self.find_reachable(entry_points)
        return all_functions - reachable
```

### ✅ 1.3 Code Smell Detection Framework (Fully Implemented)

**Location**: `src/mcp_vector_search/analysis/collectors/smells.py`

The project has a complete code smell detection system:

```python
class SmellDetector:
    """Detects code smells based on structural metrics.

    Supported Smells:
    - Long Method: lines > 50 OR cognitive_complexity > 15
    - Deep Nesting: max_nesting_depth > 4
    - Long Parameter List: parameter_count > 5
    - God Class: method_count > 20 AND lines > 500
    - Complex Method: cyclomatic_complexity > 10
    """

    def detect(self, metrics: ChunkMetrics, file_path: str, start_line: int) -> list[CodeSmell]:
        """Detect code smells in a single chunk."""
        # Returns list[CodeSmell] with name, severity, location, suggestion
```

**Integration Point for Dead Code**:

Dead code can be added as a **new smell type**:

```python
class SmellDetector:
    def detect_dead_code(
        self,
        chunk: CodeChunk,
        reachable_functions: set[str]
    ) -> list[CodeSmell]:
        """Detect unreachable code (dead code)."""
        function_name = chunk.function_name or chunk.class_name

        # Check if function is reachable from entry points
        if function_name not in reachable_functions:
            return [
                CodeSmell(
                    name="Dead Code",
                    description=f"Function '{function_name}' is not called from any entry point",
                    severity=SmellSeverity.WARNING,
                    location=f"{chunk.file_path}:{chunk.start_line}",
                    metric_value=0,  # 0 incoming edges
                    threshold=1,     # Needs at least 1 caller
                    suggestion="Remove if truly unused, or add to entry points if needed"
                )
            ]
        return []
```

### ✅ 1.4 CLI Integration (Fully Implemented)

**Location**: `src/mcp_vector_search/cli/commands/analyze.py`

The `analyze` command already supports:
- ✅ Multiple output formats (console, JSON, SARIF, markdown)
- ✅ Code smell detection (`--smells/--no-smells`)
- ✅ Quality gates (`--fail-on-smell`, `--severity-threshold`)
- ✅ Git integration (`--changed-only`, `--baseline`)
- ✅ Filtering (`--language`, `--path`)

**Integration Strategy**:

Dead code can be added as a **subcommand** or integrated into the existing smell detection:

**Option A: Separate Subcommand** (Recommended)
```bash
mcp-vector-search analyze dead-code
mcp-vector-search analyze dead-code --entry-points main.py,cli.py
mcp-vector-search analyze dead-code --format json --output dead-code.json
```

**Option B: Integrate into Smell Detection**
```bash
mcp-vector-search analyze --smells  # Includes dead code detection
mcp-vector-search analyze --smell-types dead-code,long-method
```

### ✅ 1.5 Visualization Dead Code Hints (Partially Implemented)

**Location**: Visualization layer (D3.js frontend)

The visualization already shows **red borders** for nodes with no incoming edges:

**From docs**:
> Chunks with no incoming edges (not called by anything) appear with a **red border** in the graph.

**Current Logic** (client-side):
```javascript
// Check if node has incoming caller/imports edges (dead code detection)
function getNodeBorderColor(node) {
    const hasIncomingEdges = links.some(link =>
        link.target.id === node.id &&
        (link.type === 'caller' || link.type === 'imports')
    );

    if (!hasIncomingEdges && node.type === 'function') {
        return "#ff6b6b"; // Red border for potentially dead code
    }

    return "#8b9dc3"; // Default border
}
```

**Strengths**:
- ✅ Already provides visual feedback for potentially unused code
- ✅ Only applies to functions/methods (not noise from imports/comments)
- ✅ Tooltip explains why node is marked

**Limitation**:
- ⚠️ No entry point awareness (marks `main()` as dead if not called by other code)
- ⚠️ Client-side only (no backend dead code analysis command)

---

## 2. Missing Components & Requirements

### ⚠️ 2.1 Entry Point Detection (NOT IMPLEMENTED)

This is the **primary missing piece** for dead code analysis.

**Required Patterns to Detect**:

#### Python Entry Points:
```python
# 1. Main guard
if __name__ == "__main__":
    main()

# 2. CLI entry points (Click)
@click.command()
def cli_main():
    pass

# 3. CLI entry points (Typer)
app = typer.Typer()
@app.command()
def command():
    pass

# 4. FastAPI routes
@app.get("/api/endpoint")
async def endpoint():
    pass

@app.post("/api/data")
def post_data():
    pass

# 5. Flask routes
@app.route("/endpoint")
def endpoint():
    pass

# 6. Test functions
def test_something():  # pytest auto-discovery
    pass

@pytest.fixture
def fixture():
    pass

# 7. Explicit entry points in __init__.py or __main__.py
def main():
    pass

# 8. __all__ exports (public API)
__all__ = ["public_function", "PublicClass"]
```

#### JavaScript/TypeScript Entry Points:
```javascript
// 1. Express routes
app.get('/api/endpoint', handler);
app.post('/data', handler);

// 2. Next.js API routes
export default function handler(req, res) {}

// 3. React components (exported)
export default function Component() {}
export { Component };

// 4. Jest tests
describe('suite', () => {
    test('name', () => {});
});

// 5. Module exports
module.exports = { main };
export { main };
```

**Implementation Strategy**:

```python
class EntryPointDetector:
    """Detect entry points in code across multiple languages and frameworks."""

    def __init__(self):
        self.patterns = {
            'python': PythonEntryPointDetector(),
            'javascript': JavaScriptEntryPointDetector(),
            'typescript': TypeScriptEntryPointDetector(),
        }

    def detect(self, chunks: list[CodeChunk]) -> list[str]:
        """Detect all entry points in codebase."""
        entry_points = []

        for chunk in chunks:
            detector = self.patterns.get(chunk.language)
            if detector and detector.is_entry_point(chunk):
                entry_points.append(chunk.function_name or chunk.class_name)

        return entry_points


class PythonEntryPointDetector:
    """Python-specific entry point detection."""

    def is_entry_point(self, chunk: CodeChunk) -> bool:
        """Check if chunk is an entry point."""

        # 1. Check for if __name__ == "__main__"
        if 'if __name__ == "__main__"' in chunk.content:
            return True

        # 2. Check for CLI decorators
        if '@click.command' in chunk.content or '@app.command' in chunk.content:
            return True

        # 3. Check for FastAPI/Flask routes
        if any(pattern in chunk.content for pattern in [
            '@app.get', '@app.post', '@app.put', '@app.delete',
            '@router.get', '@router.post',
            '@app.route'
        ]):
            return True

        # 4. Check for test functions
        if chunk.function_name and chunk.function_name.startswith('test_'):
            return True

        if '@pytest.fixture' in chunk.content:
            return True

        # 5. Check for __all__ exports (file-level)
        if '__all__' in chunk.content and chunk.chunk_type == 'import':
            # Extract names from __all__ = [...]
            return True

        # 6. Check for main() or cli() in __main__.py or __init__.py
        if chunk.file_path.name in ['__main__.py', '__init__.py']:
            if chunk.function_name in ['main', 'cli', 'run']:
                return True

        return False

    def extract_exported_names(self, file_path: Path) -> list[str]:
        """Extract names from __all__ = [...] in file."""
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            # Extract list items
                            if isinstance(node.value, ast.List):
                                return [
                                    elt.s for elt in node.value.elts
                                    if isinstance(elt, ast.Str)
                                ]
        except Exception:
            pass

        return []
```

**Entry Point Sources**:

1. **Explicit Entry Points** (highest confidence):
   - `if __name__ == "__main__"` blocks
   - CLI command decorators (`@click.command`, `@app.command`)
   - Web framework routes (`@app.get`, `@app.route`)
   - Test functions (`test_*`, `@pytest.fixture`)

2. **Implicit Entry Points** (moderate confidence):
   - `__all__` exports (public API)
   - Functions in `__main__.py` or `__init__.py`
   - Exported functions without leading underscore

3. **User-Specified Entry Points** (full confidence):
   - CLI flag: `--entry-points main.py:main,cli.py:cli`
   - Config file: `.mcp-vector-search/entry-points.json`

### ⚠️ 2.2 Call Graph Inversion (NOT IMPLEMENTED)

Current `caller_map` structure:
```python
caller_map[callee_id] = [caller1, caller2, ...]  # Who calls this function?
```

For reachability, we need:
```python
callee_map[caller_id] = [callee1, callee2, ...]  # What does this function call?
```

**Solution**: Build inverted graph during relationship computation:

```python
class CallGraphBuilder:
    def build_bidirectional_graph(self, chunks: list[CodeChunk]) -> CallGraph:
        """Build both caller→callee and callee→caller mappings."""
        caller_map = {}  # callee → [callers]
        callee_map = {}  # caller → [callees]

        for chunk in chunks:
            function_name = chunk.function_name or chunk.class_name

            # Extract calls from this function
            calls = extract_function_calls(chunk.content)

            for called_function in calls:
                # Populate callee_map: this function calls `called_function`
                if chunk.id not in callee_map:
                    callee_map[chunk.id] = []
                callee_map[chunk.id].append(called_function)

                # Populate caller_map: `called_function` is called by this function
                if called_function not in caller_map:
                    caller_map[called_function] = []
                caller_map[called_function].append({
                    'chunk_id': chunk.id,
                    'name': function_name,
                    'file': str(chunk.file_path),
                })

        return CallGraph(caller_map=caller_map, callee_map=callee_map)
```

### ⚠️ 2.3 False Positive Mitigation (NOT IMPLEMENTED)

Dead code detection will produce false positives for:

#### Dynamic Calls
```python
# getattr calls
handler = getattr(module, function_name)
handler()

# String-based dispatch
handlers = {'create': create_handler, 'delete': delete_handler}
handlers[action]()

# Reflection
method = getattr(self, f'handle_{event_type}')
method()
```

#### Decorator-Registered Handlers
```python
# Plugin systems
@register_plugin('converter')
def my_converter():
    pass

# Event handlers
@app.on_event('startup')
async def startup():
    pass

# Callbacks
@callback_registry.register
def on_complete():
    pass
```

#### String-Based Imports
```python
# importlib
module = importlib.import_module('my.module')
func = getattr(module, 'my_function')

# Dynamic imports
__import__('package.module')
```

**Mitigation Strategies**:

1. **Conservative Defaults**:
   - Treat any function with decorators as potentially alive
   - Mark functions without leading underscore as "potentially public API"
   - Exclude functions in `__init__.py` (package exports)

2. **Pattern Detection**:
   ```python
   def is_likely_dynamic_target(chunk: CodeChunk) -> bool:
       """Check if function is likely called dynamically."""

       # Has decorators (except @staticmethod, @classmethod, @property)
       if chunk.decorators and not is_standard_decorator(chunk.decorators):
           return True

       # Name appears in string literals elsewhere
       if function_appears_in_strings(chunk.function_name):
           return True

       # Public API marker (no leading underscore)
       if not chunk.function_name.startswith('_'):
           return True

       return False
   ```

3. **Confidence Levels**:
   ```python
   @dataclass
   class DeadCodeFinding:
       function_name: str
       file_path: str
       confidence: DeadCodeConfidence  # HIGH, MEDIUM, LOW
       reasons: list[str]  # Why we think it's dead
       caveats: list[str]  # Why it might be alive

   class DeadCodeConfidence(Enum):
       HIGH = "high"      # Private, no incoming edges, no dynamic markers
       MEDIUM = "medium"  # Public or has decorators, but no incoming edges
       LOW = "low"        # Unclear (in __init__.py, appears in strings, etc.)
   ```

4. **User Overrides**:
   ```python
   # .mcp-vector-search/dead-code-config.json
   {
       "exclude_patterns": [
           "**/*_plugin.py",  # Plugin files
           "**/handlers/**",  # Handler directories
           "**/migrations/**" # Migration files (not called directly)
       ],
       "exclude_decorators": [
           "register_handler",
           "celery.task",
           "plugin_registry.register"
       ],
       "always_alive": [
           "utils.py:legacy_function",  # Known false positive
           "api.py:deprecated_endpoint" # Still needed by clients
       ]
   }
   ```

---

## 3. Recommended Implementation Approach

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Dead Code Analyzer                    │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Entry Point      │  │ Call Graph       │           │
│  │ Detector         │  │ Builder          │           │
│  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                      │
│           │ entry_points        │ call_graph           │
│           │                     │                      │
│           └──────────┬──────────┘                      │
│                      │                                 │
│                      ▼                                 │
│            ┌──────────────────┐                        │
│            │ Reachability     │                        │
│            │ Analyzer         │                        │
│            │ (DFS/BFS)        │                        │
│            └────────┬─────────┘                        │
│                     │                                  │
│                     │ unreachable_functions            │
│                     │                                  │
│                     ▼                                  │
│            ┌──────────────────┐                        │
│            │ False Positive   │                        │
│            │ Filter           │                        │
│            └────────┬─────────┘                        │
│                     │                                  │
│                     │ dead_code_findings               │
│                     │                                  │
│                     ▼                                  │
│            ┌──────────────────┐                        │
│            │ Reporter         │                        │
│            │ (Console/JSON/   │                        │
│            │  SARIF/Markdown) │                        │
│            └──────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
src/mcp_vector_search/analysis/
├── collectors/
│   ├── dead_code.py           # NEW: Dead code detector
│   └── entry_points.py        # NEW: Entry point detection
└── reporters/
    └── dead_code_reporter.py  # NEW: Dead code output formatting
```

### 3.3 Implementation Phases

#### Phase 1: Entry Point Detection (1-2 days)

**Files to Create**:
- `src/mcp_vector_search/analysis/collectors/entry_points.py`

**Implementation**:
```python
@dataclass
class EntryPoint:
    """Represents a detected entry point in the codebase."""

    function_name: str
    file_path: str
    chunk_id: str
    entry_type: EntryPointType  # MAIN, CLI, ROUTE, TEST, EXPORT
    confidence: float  # 0.0-1.0

class EntryPointType(Enum):
    MAIN = "main"          # if __name__ == "__main__"
    CLI = "cli"            # @click.command, @app.command
    ROUTE = "route"        # @app.get, @app.route
    TEST = "test"          # test_*, @pytest.fixture
    EXPORT = "export"      # __all__, __init__.py
    USER_DEFINED = "user"  # User-specified entry points

class EntryPointDetector:
    """Detect entry points across multiple languages and frameworks."""

    def detect_all(self, chunks: list[CodeChunk]) -> list[EntryPoint]:
        """Detect all entry points in codebase."""
        pass

    def is_entry_point(self, chunk: CodeChunk) -> bool:
        """Check if chunk is an entry point."""
        pass

    def extract_all_exports(self, file_path: Path) -> list[str]:
        """Extract __all__ exports from file."""
        pass
```

**Tests**:
- `tests/unit/analysis/collectors/test_entry_points.py`
  - Test `if __name__ == "__main__"` detection
  - Test CLI decorator detection
  - Test route decorator detection
  - Test test function detection
  - Test `__all__` export extraction

#### Phase 2: Reachability Analysis (1 day)

**Files to Create**:
- `src/mcp_vector_search/analysis/collectors/dead_code.py`

**Implementation**:
```python
@dataclass
class DeadCodeFinding:
    """Represents a dead code detection result."""

    function_name: str
    file_path: str
    start_line: int
    end_line: int
    chunk_id: str
    confidence: DeadCodeConfidence
    reasons: list[str]
    caveats: list[str]

class DeadCodeAnalyzer:
    """Analyze code reachability to detect dead code."""

    def __init__(
        self,
        call_graph: CallGraph,
        entry_detector: EntryPointDetector,
    ):
        self.call_graph = call_graph
        self.entry_detector = entry_detector

    def analyze(self, chunks: list[CodeChunk]) -> list[DeadCodeFinding]:
        """Find all unreachable code from entry points."""

        # 1. Detect entry points
        entry_points = self.entry_detector.detect_all(chunks)

        # 2. Build reachability set via DFS
        reachable = self._find_reachable(entry_points)

        # 3. Identify unreachable functions
        all_functions = {c.function_name for c in chunks if c.function_name}
        unreachable = all_functions - reachable

        # 4. Filter false positives and assign confidence
        findings = self._filter_false_positives(unreachable, chunks)

        return findings

    def _find_reachable(self, entry_points: list[EntryPoint]) -> set[str]:
        """DFS to find all reachable functions."""
        reachable = set()
        visited = set()

        def dfs(function_name: str):
            if function_name in visited:
                return
            visited.add(function_name)
            reachable.add(function_name)

            # Visit all callees
            for callee in self.call_graph.get_callees(function_name):
                dfs(callee)

        for entry in entry_points:
            dfs(entry.function_name)

        return reachable

    def _filter_false_positives(
        self,
        unreachable: set[str],
        chunks: list[CodeChunk],
    ) -> list[DeadCodeFinding]:
        """Apply heuristics to reduce false positives."""
        findings = []

        for function_name in unreachable:
            chunk = self._find_chunk(function_name, chunks)
            if not chunk:
                continue

            confidence = self._calculate_confidence(chunk)
            reasons = self._generate_reasons(chunk)
            caveats = self._generate_caveats(chunk)

            findings.append(DeadCodeFinding(
                function_name=function_name,
                file_path=str(chunk.file_path),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_id=chunk.id,
                confidence=confidence,
                reasons=reasons,
                caveats=caveats,
            ))

        return findings

    def _calculate_confidence(self, chunk: CodeChunk) -> DeadCodeConfidence:
        """Determine confidence level for dead code finding."""

        # HIGH confidence: private, no decorators, no string references
        if (chunk.function_name.startswith('_') and
            not chunk.decorators and
            not self._appears_in_strings(chunk.function_name)):
            return DeadCodeConfidence.HIGH

        # MEDIUM confidence: public or has decorators
        if not chunk.function_name.startswith('_') or chunk.decorators:
            return DeadCodeConfidence.MEDIUM

        # LOW confidence: unclear
        return DeadCodeConfidence.LOW
```

**Tests**:
- `tests/unit/analysis/collectors/test_dead_code.py`
  - Test reachability from single entry point
  - Test reachability with multiple entry points
  - Test transitive reachability (A→B→C)
  - Test cyclic reachability (A→B→C→A)
  - Test confidence calculation

#### Phase 3: CLI Integration & Output (1 day)

**Files to Modify**:
- `src/mcp_vector_search/cli/commands/analyze.py`

**Implementation**:

```python
# Option 1: Subcommand approach
@analyze_app.command()
def dead_code(
    project_root: Path = typer.Option(None, ...),
    entry_points: str = typer.Option(
        None,
        "--entry-points",
        help="Comma-separated list of entry point patterns (e.g., 'main.py:main,cli.py:cli')"
    ),
    confidence: str = typer.Option(
        "medium",
        "--min-confidence",
        help="Minimum confidence level: high, medium, low"
    ),
    format: str = typer.Option("console", ...),
    output: Path = typer.Option(None, ...),
) -> None:
    """Detect dead code (unreachable functions)."""

    # 1. Load chunks from database
    chunks = load_chunks(project_root)

    # 2. Build call graph
    call_graph = CallGraphBuilder().build(chunks)

    # 3. Detect entry points
    entry_detector = EntryPointDetector()
    if entry_points:
        # User-specified entry points
        entry_list = parse_entry_points(entry_points)
    else:
        # Auto-detect entry points
        entry_list = entry_detector.detect_all(chunks)

    # 4. Analyze dead code
    analyzer = DeadCodeAnalyzer(call_graph, entry_detector)
    findings = analyzer.analyze(chunks)

    # 5. Filter by confidence
    min_conf = DeadCodeConfidence[confidence.upper()]
    filtered = [f for f in findings if f.confidence >= min_conf]

    # 6. Output results
    if format == "console":
        print_dead_code_report(filtered, entry_list)
    elif format == "json":
        print_json_report(filtered)
    elif format == "sarif":
        write_sarif_report(filtered, output)
```

**Output Formats**:

**Console**:
```
Dead Code Analysis Report
═════════════════════════

Entry Points Detected: 5
  • main() in main.py (MAIN)
  • cli() in cli.py (CLI)
  • test_integration() in tests/test_api.py (TEST)
  • /api/users in api/users.py (ROUTE)
  • PublicAPI in __init__.py (EXPORT)

Dead Code Findings: 12

[HIGH CONFIDENCE] (7 findings)
  • _legacy_converter in utils/converters.py:45-67
    Reason: Private function, no incoming calls

  • _deprecated_handler in handlers/old.py:12-34
    Reason: Private function, no incoming calls

[MEDIUM CONFIDENCE] (5 findings)
  • public_utility in utils/misc.py:89-102
    Reason: No incoming calls
    Caveat: Public function (no leading underscore)

  • @deprecated decorated_function in api/legacy.py:56-78
    Reason: No incoming calls
    Caveat: Has decorator (might be dynamically called)

Summary:
  Total Functions: 245
  Reachable: 233 (95.1%)
  Unreachable: 12 (4.9%)
    - High confidence: 7
    - Medium confidence: 5
    - Low confidence: 0
```

**JSON**:
```json
{
  "summary": {
    "total_functions": 245,
    "reachable": 233,
    "unreachable": 12,
    "confidence_breakdown": {
      "high": 7,
      "medium": 5,
      "low": 0
    }
  },
  "entry_points": [
    {
      "function_name": "main",
      "file_path": "main.py",
      "entry_type": "MAIN",
      "confidence": 1.0
    }
  ],
  "findings": [
    {
      "function_name": "_legacy_converter",
      "file_path": "utils/converters.py",
      "start_line": 45,
      "end_line": 67,
      "confidence": "HIGH",
      "reasons": [
        "Private function (starts with _)",
        "No incoming calls from reachable code"
      ],
      "caveats": []
    }
  ]
}
```

**SARIF** (for GitHub Code Scanning):
```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "mcp-vector-search",
          "informationUri": "https://github.com/user/mcp-vector-search",
          "rules": [
            {
              "id": "dead-code",
              "shortDescription": {
                "text": "Unreachable code detected"
              },
              "fullDescription": {
                "text": "Function is not called from any entry point and appears to be dead code"
              },
              "defaultConfiguration": {
                "level": "warning"
              }
            }
          ]
        }
      },
      "results": [
        {
          "ruleId": "dead-code",
          "level": "warning",
          "message": {
            "text": "Function '_legacy_converter' is unreachable from entry points"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "utils/converters.py"
                },
                "region": {
                  "startLine": 45,
                  "endLine": 67
                }
              }
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 4. Estimated Complexity

### Overall Assessment: **MEDIUM**

**Breakdown**:
- ✅ **Low Complexity**: Call graph and DFS traversal (already implemented)
- ✅ **Low Complexity**: CLI integration and output formatting (existing patterns)
- ⚠️ **Medium Complexity**: Entry point detection (new logic, multiple patterns)
- ⚠️ **Medium Complexity**: False positive mitigation (heuristics needed)

**Estimated Effort**: **3-4 days** (single developer)

| Phase | Task | Effort | Complexity |
|-------|------|--------|------------|
| 1 | Entry Point Detection | 1-2 days | Medium |
| 2 | Reachability Analysis | 1 day | Low |
| 3 | CLI Integration & Output | 1 day | Low |
| 4 | Testing & Documentation | 0.5 days | Low |

---

## 5. Potential Limitations & Caveats

### 5.1 Known False Positives

1. **Dynamic Dispatch**:
   ```python
   # Will mark as dead, but is actually alive
   handlers = {'create': create_user, 'delete': delete_user}
   handlers[action]()
   ```

   **Mitigation**: Mark functions with certain decorators as "potentially alive"

2. **Reflection & getattr**:
   ```python
   # Will mark as dead, but is actually alive
   method = getattr(self, f'handle_{event_type}')
   method()
   ```

   **Mitigation**: Search for function name in string literals

3. **Plugin Systems**:
   ```python
   # Will mark as dead, but registered in plugin registry
   @register_plugin('converter')
   def my_converter():
       pass
   ```

   **Mitigation**: Exclude specific decorator patterns

4. **External Calls** (API endpoints called by clients):
   ```python
   # Will mark as dead if no internal callers
   @app.get("/api/endpoint")
   def endpoint():
       pass
   ```

   **Mitigation**: Treat route handlers as entry points

### 5.2 Known False Negatives

1. **Truly Dead Entry Points**:
   ```python
   # Is an entry point, but never actually invoked
   @click.command()
   def deprecated_command():
       pass
   ```

   **Mitigation**: None (by design, entry points are always considered alive)

2. **Conditional Entry Points**:
   ```python
   if DEBUG:
       @app.get("/debug")
       def debug_endpoint():
           pass
   ```

   **Mitigation**: Static analysis can't handle runtime conditions

### 5.3 Language Support

**Current Scope**: Python only

**Reason**:
- AST parsing uses Python's `ast` module
- Entry point patterns are Python-specific

**Future Extension**:
- JavaScript/TypeScript: Use tree-sitter for AST parsing
- Implement language-specific entry point detectors
- Reuse same reachability algorithm (language-agnostic)

---

## 6. Integration with Existing Features

### 6.1 Integration with Code Smell Detection

Dead code can be added as a **new smell type**:

```python
# In SmellDetector.detect_all()
class SmellDetector:
    def detect_all(
        self,
        file_metrics: FileMetrics,
        file_path: str,
        dead_code_analyzer: DeadCodeAnalyzer | None = None  # NEW
    ) -> list[CodeSmell]:
        """Detect all code smells including dead code."""

        all_smells = []

        # Existing smell detection
        for chunk in file_metrics.chunks:
            chunk_smells = self.detect(chunk, file_path)
            all_smells.extend(chunk_smells)

        # NEW: Dead code detection
        if dead_code_analyzer:
            dead_code_findings = dead_code_analyzer.get_findings_for_file(file_path)
            for finding in dead_code_findings:
                all_smells.append(
                    CodeSmell(
                        name="Dead Code",
                        description=f"Function '{finding.function_name}' is unreachable",
                        severity=SmellSeverity.WARNING,
                        location=f"{file_path}:{finding.start_line}",
                        metric_value=0,
                        threshold=1,
                        suggestion="Remove if truly unused, or ensure it's called from entry points"
                    )
                )

        return all_smells
```

**Usage**:
```bash
# Dead code included in smell detection
mcp-vector-search analyze --smells

# Dead code only
mcp-vector-search analyze dead-code

# Dead code with specific entry points
mcp-vector-search analyze dead-code --entry-points main.py:main,cli.py:cli
```

### 6.2 Integration with Visualization

The visualization already has **red border hints** for nodes with no incoming edges.

**Enhancement**: Use backend dead code analysis instead of client-side heuristic:

```javascript
// Current (client-side):
function getNodeBorderColor(node) {
    const hasIncomingEdges = links.some(link => link.target.id === node.id);
    if (!hasIncomingEdges && node.type === 'function') {
        return "#ff6b6b"; // Red for dead code
    }
}

// Enhanced (backend-driven):
function getNodeBorderColor(node) {
    // Use backend dead code analysis results
    if (node.dead_code_confidence === 'HIGH') {
        return "#ff0000"; // Bright red for high confidence
    } else if (node.dead_code_confidence === 'MEDIUM') {
        return "#ff6b6b"; // Orange-red for medium confidence
    }
}
```

**Benefits**:
- ✅ Entry point awareness (doesn't mark `main()` as dead)
- ✅ Confidence levels (different colors for HIGH/MEDIUM/LOW)
- ✅ Consistent with CLI analysis

### 6.3 Integration with CI/CD

Dead code detection can be used as a **quality gate**:

```yaml
# .github/workflows/ci.yml
- name: Analyze code quality
  run: |
    mcp-vector-search analyze dead-code \
      --format sarif \
      --output dead-code.sarif \
      --min-confidence high

- name: Upload SARIF results
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: dead-code.sarif
```

**Result**: Dead code findings appear in GitHub Security tab

---

## 7. Comparison with Alternative Approaches

### 7.1 Alternative: Vulture (Python Dead Code Detector)

**Vulture** is an existing Python dead code detector.

**Strengths**:
- ✅ Mature, well-tested
- ✅ Fast (AST-based)
- ✅ Supports Python 2 and 3

**Limitations**:
- ❌ Python-only (no multi-language support)
- ❌ No entry point auto-detection (requires manual whitelist)
- ❌ High false positive rate (no confidence levels)
- ❌ No integration with existing mcp-vector-search infrastructure

**Why Build Our Own**:
1. **Multi-language future**: Tree-sitter support for JS/TS/Go/Rust
2. **Existing infrastructure**: Reuse call graph, AST parsing, CLI patterns
3. **Confidence levels**: Reduce false positive noise
4. **Integration**: Works with existing `analyze` command and visualization

### 7.2 Alternative: Coverage-Based Dead Code Detection

**Approach**: Use test coverage data to identify uncovered code.

**Example**:
```bash
pytest --cov=. --cov-report=json
# Parse coverage.json to find 0% coverage functions
```

**Strengths**:
- ✅ No false positives (if code is never executed, it's truly dead)
- ✅ Easy to implement (parse existing coverage data)

**Limitations**:
- ❌ Requires comprehensive test suite (doesn't work without tests)
- ❌ Only detects code unused **by tests** (not production usage)
- ❌ Doesn't identify architectural dead code (code paths never reached in production)

**Why Not Use This**:
- mcp-vector-search aims to work on **any codebase**, not just well-tested ones
- Static analysis finds dead code **before** running tests
- Complements coverage (static + dynamic analysis = comprehensive)

---

## 8. Recommended Next Steps

### Immediate Actions

1. **Validate Approach** (1 hour):
   - Review this document with stakeholders
   - Confirm entry point patterns for target languages
   - Agree on false positive mitigation strategy

2. **Prototype Entry Point Detection** (2-3 hours):
   - Implement basic `PythonEntryPointDetector`
   - Test on mcp-vector-search codebase
   - Validate detected entry points manually

3. **Spike Reachability Analysis** (2 hours):
   - Adapt existing DFS code from `CircularDependencyDetector`
   - Run on sample call graph
   - Measure performance on large codebases

### Implementation Plan

**Sprint 1** (Week 1):
- Day 1-2: Entry point detection implementation + tests
- Day 3: Reachability analysis implementation + tests
- Day 4: CLI integration + basic console output
- Day 5: Documentation + code review

**Sprint 2** (Week 2):
- Day 1-2: False positive mitigation + confidence scoring
- Day 3: JSON/SARIF output formats
- Day 4: Integration with existing `analyze` command
- Day 5: End-to-end testing + bug fixes

**Sprint 3** (Week 3):
- Day 1-2: Visualization integration (backend-driven red borders)
- Day 3: Performance optimization (large codebases)
- Day 4: User documentation + examples
- Day 5: Release preparation

---

## 9. Conclusion

**Dead code analysis is highly feasible** for mcp-vector-search with **3-4 days of focused effort**. The project already has 90% of the required infrastructure:

✅ **Existing Infrastructure** (Reusable):
- AST-based call extraction (`extract_function_calls()`)
- DFS graph traversal (`CircularDependencyDetector`)
- Code smell detection framework (`SmellDetector`)
- CLI patterns and output formatting (`analyze` command)

⚠️ **Missing Components** (New Work):
- Entry point detection (1-2 days)
- Call graph inversion for reachability (0.5 days)
- False positive mitigation (1 day)
- Output formatting (0.5 days)

**Recommended Approach**:
1. Start with **Python-only** support (leverage existing AST infrastructure)
2. Implement as **subcommand** (`mcp-vector-search analyze dead-code`)
3. Use **confidence levels** (HIGH/MEDIUM/LOW) to reduce false positive noise
4. Integrate with **existing visualization** (backend-driven red borders)
5. Support **CI/CD integration** via SARIF output

**Risk Assessment**: **LOW**

The implementation reuses well-tested components and follows established patterns in the codebase. The primary risk is **false positive management**, which is mitigated through confidence scoring and user configuration.

---

## Appendix A: Code Examples

### A.1 Example Entry Point Detection

```python
# Test case: Detect entry points in CLI application
def test_entry_point_detection():
    code = '''
import click

@click.command()
def main():
    """CLI entry point."""
    process_data()

def process_data():
    """Helper function."""
    pass

if __name__ == "__main__":
    main()
'''

    chunks = parse_code(code)
    detector = EntryPointDetector()
    entry_points = detector.detect_all(chunks)

    assert len(entry_points) == 1
    assert entry_points[0].function_name == "main"
    assert entry_points[0].entry_type == EntryPointType.CLI
```

### A.2 Example Reachability Analysis

```python
# Test case: Detect unreachable function
def test_reachability_analysis():
    call_graph = CallGraph()

    # main() → process() → helper()
    call_graph.add_edge("main", "process")
    call_graph.add_edge("process", "helper")

    # orphan() is not called
    call_graph.add_node("orphan")

    analyzer = ReachabilityAnalyzer(call_graph)
    reachable = analyzer.find_reachable(entry_points=["main"])

    assert reachable == {"main", "process", "helper"}

    unreachable = analyzer.find_unreachable(entry_points=["main"])

    assert unreachable == {"orphan"}
```

### A.3 Example False Positive Filtering

```python
# Test case: Filter false positives
def test_false_positive_filtering():
    finding = DeadCodeFinding(
        function_name="public_api",
        file_path="api.py",
        start_line=10,
        end_line=20,
        chunk_id="chunk_123",
        confidence=DeadCodeConfidence.MEDIUM,
        reasons=["No incoming calls"],
        caveats=["Public function (no leading underscore)"]
    )

    # Should NOT be HIGH confidence (public function)
    assert finding.confidence != DeadCodeConfidence.HIGH

    # Should have caveat about being public
    assert any("public" in caveat.lower() for caveat in finding.caveats)
```

---

## Appendix B: References

### Existing Infrastructure
- **Call Graph**: `src/mcp_vector_search/core/relationships.py`
- **AST Parsing**: `extract_function_calls()` function
- **Graph Algorithms**: `src/mcp_vector_search/analysis/collectors/coupling.py`
- **Code Smells**: `src/mcp_vector_search/analysis/collectors/smells.py`
- **CLI Commands**: `src/mcp_vector_search/cli/commands/analyze.py`

### Research Documents
- **Circular Dependency Detection**: `docs/research/cycle-detection-analysis-2025-12-06.md`
- **Visualization Architecture**: `docs/research/visualization-architecture-analysis-2025-12-06.md`
- **AST Circular Dependency Fix**: `docs/development/ast_circular_dependency_fix.md`

### External Tools
- **Vulture**: https://github.com/jendrikseipp/vulture (Python dead code detector)
- **SARIF**: https://sarifweb.azurewebsites.net/ (Static Analysis Results Interchange Format)

---

**End of Research Document**
