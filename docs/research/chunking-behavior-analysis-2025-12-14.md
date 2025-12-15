# MCP Vector Search: Chunking Behavior Analysis

**Date**: December 14, 2024
**Researcher**: Claude Code (Research Agent)
**Issue**: Investigation into why SemanticIndexer class appears to be indexed as single chunk

## Executive Summary

The mcp-vector-search indexer **correctly extracts classes AND individual methods as separate chunks**. The SemanticIndexer class (1208 lines) is indexed as **33 total chunks**:
- 1 class chunk (entire class definition)
- 31 function chunks (individual methods)
- 1 imports chunk (module-level imports)

The system uses a **dual-level chunking strategy** where both the complete class AND its individual methods are indexed separately, enabling both class-level and method-level semantic search.

## Investigation Findings

### 1. How Chunks Are Created

**File**: `src/mcp_vector_search/parsers/python.py`

The `_extract_chunks_from_tree()` method uses Tree-sitter AST traversal with a recursive `visit_node()` function:

```python
def visit_node(node, current_class=None):
    """Recursively visit AST nodes."""
    node_type = node.type

    if node_type == "function_definition":
        chunks.extend(
            self._extract_function(node, lines, file_path, current_class)
        )
    elif node_type == "class_definition":
        class_chunks = self._extract_class(node, lines, file_path)
        chunks.extend(class_chunks)

        # Visit class methods with class context
        class_name = self._get_node_name(node)
        for child in node.children:
            visit_node(child, class_name)  # ← RECURSION with class context
```

**Key Behavior**:
1. When encountering a `class_definition` node:
   - Extracts the ENTIRE class as a single chunk (lines 98-100)
   - Then RECURSES into class children with `class_name` context (lines 103-105)
2. When encountering `function_definition` nodes inside a class:
   - Extracts each method as a separate chunk
   - Sets `class_name` property to link method to its parent class

### 2. Method Extraction Inside Classes

**File**: `src/mcp_vector_search/parsers/python.py` (lines 137-182)

The `_extract_function()` method accepts an optional `class_name` parameter:

```python
def _extract_function(
    self, node, lines: list[str], file_path: Path, class_name: str | None = None
) -> list[CodeChunk]:
    """Extract function definition as a chunk."""
    chunks = []

    function_name = self._get_node_name(node)
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1

    # ... extract metadata ...

    chunk = self._create_chunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        chunk_type="function",
        function_name=function_name,
        class_name=class_name,  # ← Links method to class
        # ... other metadata ...
    )
```

**When `class_name` is provided**: The function chunk is marked as belonging to that class, enabling method-level search within class context.

### 3. Tree-sitter Query Patterns

**No explicit `.scm` query files** - The system uses Tree-sitter's node type checking directly in Python code.

Node types extracted:
- `"module"` → Creates imports chunk
- `"class_definition"` → Creates class chunk
- `"function_definition"` → Creates function/method chunk
- Recursion ensures nested structures are traversed

### 4. Database Verification

**Query Results** (actual data from `.mcp-vector-search/chroma.sqlite3`):

```sql
-- Chunks for indexer.py
Total: 33 chunks

Breakdown by type:
- function: 31 chunks  (individual methods)
- class:     1 chunk   (SemanticIndexer class)
- imports:   1 chunk   (module-level imports)
```

**Sample chunks**:
```
class SemanticIndexer:           # 1208 lines → 1 class chunk
    def __init__(self, ...):     # → 1 function chunk
    def index_project(self, ...):  # → 1 function chunk
    def index_file(self, ...):   # → 1 function chunk
    # ... 28 more methods ...     # → 28 more function chunks
```

### 5. Hierarchy Building

**File**: `src/mcp_vector_search/core/indexer.py` (lines 1124-1229)

The `_build_chunk_hierarchy()` method establishes parent-child relationships AFTER parsing:

```python
def _build_chunk_hierarchy(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Build parent-child relationships between chunks.

    Logic:
    - Module chunks (chunk_type="module") have depth 0
    - Class chunks have depth 1, parent is module
    - Method chunks have depth 2, parent is class
    - Function chunks outside classes have depth 1, parent is module
    """
    # ...

    for func in function_chunks:
        if func.class_name:
            # Find parent class
            parent_class = next(
                (c for c in class_chunks if c.class_name == func.class_name), None
            )
            if parent_class:
                func.parent_chunk_id = parent_class.chunk_id
                func.chunk_depth = parent_class.chunk_depth + 1
                parent_class.child_chunk_ids.append(func.chunk_id)
```

**Purpose**: Links method chunks to their parent class chunk for hierarchical queries and visualization.

## Test Results

**Test File**: `test_chunking.py` (created during investigation)

```python
class Calculator:
    """A simple calculator class."""

    def __init__(self):
        """Initialize calculator."""
        self.result = 0

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
```

**Output**:
```
Total chunks extracted: 4

Chunk 1:
  Type: class
  Class: Calculator
  Function: None
  Lines: 2-15
  Depth: 1

Chunk 2:
  Type: function
  Class: Calculator
  Function: __init__
  Lines: 5-7
  Depth: 2

Chunk 3:
  Type: function
  Class: Calculator
  Function: add
  Lines: 9-11
  Depth: 2

Chunk 4:
  Type: function
  Class: Calculator
  Function: subtract
  Lines: 13-15
  Depth: 2

✅ Working as expected: Both class and individual methods extracted
```

## Expected vs. Actual Behavior

### Expected Behavior
✅ **CONFIRMED**: The system should create:
1. ONE chunk for the complete class definition
2. SEPARATE chunks for each method within the class

### Actual Behavior
✅ **CORRECT**: For SemanticIndexer (1208 lines):
- 1 class chunk covering entire class (lines 41-1249)
- 31 individual method chunks
- 1 imports chunk

### Why This Design?

**Dual-Level Chunking Strategy**:

1. **Class-level chunk**: Enables semantic search for entire class patterns
   - Useful for: "Find classes that handle database connections"
   - Returns: Complete class definition with all methods

2. **Method-level chunks**: Enables granular search for specific functionality
   - Useful for: "Find methods that parse files"
   - Returns: Specific methods without entire class context

3. **Hierarchy metadata**: Links methods to their parent class
   - Enables: "Show me all methods in SemanticIndexer class"
   - Visualization: Class → Methods tree structure

## Recommendations

### 1. Current Behavior: KEEP AS-IS ✅

The dual-level chunking is a **design feature, not a bug**. It provides:
- Maximum search flexibility (class-level OR method-level)
- Hierarchical relationships for visualization
- Better search relevance (methods ranked separately from class)

### 2. Potential Improvements (Future Work)

**Option A: Configurable Chunking Strategies**
```python
class ChunkingStrategy(Enum):
    DUAL_LEVEL = "dual"      # Current: class + methods (default)
    METHODS_ONLY = "methods" # Only extract methods (no class chunk)
    CLASSES_ONLY = "classes" # Only extract classes (no method chunks)
```

**Option B: Smart Class Chunking**
For very large classes (>500 lines), consider:
- Breaking class chunk into logical sections (constructor, public methods, private methods)
- Keeping method chunks separate
- Adding section-level metadata

**Option C: Search Result Deduplication**
When searching returns both class chunk AND its methods:
- Detect parent-child relationships
- Optionally group results by class
- Show "Expand to see 31 methods" UI pattern

### 3. Documentation Needs

Add to user documentation:
- Explain dual-level chunking strategy
- Show examples of class-level vs method-level search
- Document when to use each search approach

## Files Analyzed

```
src/mcp_vector_search/
├── core/
│   ├── indexer.py (lines 41-1249, 33 chunks)
│   │   └── SemanticIndexer class (1208 lines)
│   └── models.py (CodeChunk dataclass definition)
└── parsers/
    ├── base.py (BaseParser abstract class)
    └── python.py (PythonParser with Tree-sitter)
        ├── _extract_chunks_from_tree() (lines 83-135)
        ├── _extract_function() (lines 137-182)
        ├── _extract_class() (lines 184-220)
        └── visit_node() (lines 90-118)
```

## Conclusion

The mcp-vector-search indexer correctly implements a **dual-level chunking strategy** where classes are indexed both as complete entities AND as collections of individual methods. This is intentional design that enables both broad class-level searches and granular method-level searches.

The reported issue of "SemanticIndexer being one chunk" is a **misunderstanding of the chunking model**, not a bug. The system actually creates:
- 1 class chunk (for class-level search)
- 31 method chunks (for method-level search)
- Hierarchical relationships linking methods to their parent class

**Recommendation**: NO CODE CHANGES NEEDED. Consider adding documentation to explain the dual-level chunking strategy to users.

---

**Memory Updates for Future Reference**:

```json
{
  "remember": [
    "mcp-vector-search uses dual-level chunking: classes are indexed both as complete entities AND as individual methods",
    "SemanticIndexer class (1208 lines) is indexed as 33 chunks: 1 class + 31 methods + 1 imports",
    "Chunking happens in PythonParser._extract_chunks_from_tree() with recursive visit_node()",
    "Hierarchy building occurs after parsing in SemanticIndexer._build_chunk_hierarchy()",
    "This is intentional design for flexible search (class-level OR method-level queries)"
  ]
}
```
