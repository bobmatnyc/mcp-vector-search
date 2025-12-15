# Class Chunking Behavior Analysis

**Date:** 2025-12-14
**Research Type:** Code Analysis - Parser Behavior Investigation
**Status:** Complete

## Executive Summary

The `SemanticIndexer` class in `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py` is being indexed as BOTH:
1. A single 1208-line class chunk (lines 41-1248, complexity: 132.0)
2. 31 individual method chunks (properly extracted with parent relationships)

**This is by design**, not a bug. The parser creates a hierarchical representation where:
- The **class chunk** represents the full class scope for class-level searches
- **Method chunks** enable fine-grained search at the function level
- **Parent-child relationships** link methods to their containing class

## Context

**File Analyzed:** `src/mcp_vector_search/core/indexer.py`
- Total lines: 1248
- `SemanticIndexer` class: Lines 41-1248 (1208 lines)
- Complexity score: 132.0
- Methods extracted: 31 individual chunks

**Question:** Why is the class indexed as a single large chunk instead of only individual methods?

## Findings

### 1. Parser Architecture

The Python parser (`src/mcp_vector_search/parsers/python.py`) uses a dual-chunking strategy:

```python
# Lines 98-105 in python.py
elif node_type == "class_definition":
    class_chunks = self._extract_class(node, lines, file_path)  # Creates CLASS chunk
    chunks.extend(class_chunks)

    # Visit class methods with class context
    class_name = self._get_node_name(node)
    for child in node.children:
        visit_node(child, class_name)  # Creates METHOD chunks
```

**Key Insight:** Both the class AND its methods are intentionally extracted as separate chunks.

### 2. Actual Indexed Structure

Analysis of `.mcp-vector-search/chunk-graph.json` reveals:

```
SemanticIndexer Chunks:
├── Class Chunk (ID: 65a5fb62554a2f2b)
│   ├── Lines: 41-1248 (1208 lines)
│   ├── Type: class
│   └── Complexity: 132.0
│
└── Method Chunks (31 total, examples):
    ├── __init__ (lines 44-138, 95 lines)
    ├── _default_collectors (lines 140-160, 21 lines)
    ├── _collect_metrics (lines 162-198, 37 lines)
    ├── index_project (lines 278-415, 138 lines)
    └── ... (27 more methods)
```

**Parent-Child Relationships:** All 31 methods have `parent_id: 65a5fb62554a2f2b` (the class chunk ID).

### 3. Why This Design?

This dual-chunking strategy serves multiple purposes:

#### Use Case 1: Class-Level Queries
```
User: "Find all classes related to indexing"
→ Returns the class chunk with full class docstring and overview
```

#### Use Case 2: Method-Level Queries
```
User: "How does the index_project method work?"
→ Returns the specific 138-line method chunk
```

#### Use Case 3: Context-Aware Search
```
User: "Find methods in SemanticIndexer that handle metadata"
→ Uses parent_id relationships to filter methods within the class
```

### 4. Comparison with Other Large Classes

Analyzed other large files in the codebase:

| File | Lines | Structure |
|------|-------|-----------|
| `scripts.py` | 4046 | Template file (likely different chunking) |
| `html_report.py` | 2895 | Multiple classes + functions |
| `server.py` | 1543 | MCP server implementation |
| `indexer.py` | 1248 | **SemanticIndexer class (this case)** |

**Pattern:** All class definitions follow the same dual-chunking approach:
- Class-level chunk for the entire class
- Individual chunks for each method

### 5. Configuration Options

Checked `src/mcp_vector_search/config/constants.py`:

```python
DEFAULT_CHUNK_SIZE = 50  # Lines per chunk for text/fallback parsing
TEXT_CHUNK_SIZE = 30     # Lines per text/markdown chunk
```

**Important:** These constants apply to **fallback parsing** (when Tree-sitter is unavailable) or text files. They do NOT control AST-based chunking, which is driven by syntactic structure (functions, classes, methods).

## Technical Deep Dive

### Tree-Sitter Node Traversal

The parser uses recursive node traversal:

```python
def visit_node(node, current_class=None):
    node_type = node.type

    if node_type == "function_definition":
        # Extract function chunk
        chunks.extend(self._extract_function(node, lines, file_path, current_class))

    elif node_type == "class_definition":
        # Extract class chunk (entire class body)
        class_chunks = self._extract_class(node, lines, file_path)
        chunks.extend(class_chunks)

        # THEN visit children to extract methods
        class_name = self._get_node_name(node)
        for child in node.children:
            visit_node(child, class_name)  # Methods get class_name in context
```

**Result:**
- Class chunk: `chunk_type="class"`, contains entire class (41-1248)
- Method chunks: `chunk_type="function"`, `class_name="SemanticIndexer"`, parent_id links to class

### Method Extraction with Class Context

Methods are extracted with class awareness:

```python
def _extract_function(self, node, lines, file_path, class_name=None):
    # class_name is passed from parent class traversal
    chunk = self._create_chunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        chunk_type="function",
        function_name=function_name,
        class_name=class_name,  # ← Links method to class
        chunk_depth=2 if class_name else 1,  # ← Indicates nesting level
    )
```

## Options for Alternative Chunking

If method-only chunking (without class-level chunks) is desired, there are three approaches:

### Option 1: Skip Class Chunk Creation (Simple)

Modify `python.py` lines 98-105:

```python
elif node_type == "class_definition":
    # SKIP class chunk extraction
    # class_chunks = self._extract_class(node, lines, file_path)
    # chunks.extend(class_chunks)

    # Only visit class methods
    class_name = self._get_node_name(node)
    for child in node.children:
        visit_node(child, class_name)
```

**Pros:**
- Eliminates large class chunks
- Methods still indexed with class context
- Simple one-line change

**Cons:**
- Loses class-level metadata (docstrings, decorators)
- Can't search for "classes related to X"
- Breaks hierarchical relationships in graph

### Option 2: Conditional Chunking Based on Size

Add size threshold to decide whether to chunk the class:

```python
elif node_type == "class_definition":
    class_name = self._get_node_name(node)
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    class_size = end_line - start_line + 1

    # Only create class chunk if reasonably sized
    MAX_CLASS_CHUNK_SIZE = 500  # configurable threshold
    if class_size <= MAX_CLASS_CHUNK_SIZE:
        class_chunks = self._extract_class(node, lines, file_path)
        chunks.extend(class_chunks)
    else:
        # For large classes, create a metadata-only chunk (just docstring + signature)
        metadata_chunk = self._extract_class_metadata(node, lines, file_path)
        chunks.extend(metadata_chunk)

    # Always visit methods
    for child in node.children:
        visit_node(child, class_name)
```

**Pros:**
- Balances granularity with overview
- Small classes still get single chunk
- Large classes split into methods + metadata

**Cons:**
- More complex logic
- Requires tuning threshold
- Need to implement `_extract_class_metadata()`

### Option 3: Class Summary Chunk

Create a lightweight class chunk with just signature + docstring (first ~50 lines):

```python
def _extract_class_summary(self, node, lines, file_path):
    """Extract class signature, docstring, and first few attributes only."""
    class_name = self._get_node_name(node)
    start_line = node.start_point[0] + 1

    # Find where methods start (typically after __init__ or class vars)
    summary_end_line = min(start_line + 50, node.end_point[0] + 1)

    content = self._get_line_range(lines, start_line, summary_end_line)
    docstring = self._extract_docstring(node, lines)

    return self._create_chunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=summary_end_line,
        chunk_type="class_summary",
        class_name=class_name,
        docstring=docstring,
    )
```

**Pros:**
- Keeps class overview without bloat
- Fixed small chunk size
- Preserves searchability for classes

**Cons:**
- Arbitrary cutoff point
- Might miss important class-level code
- Complexity in determining "summary" boundary

## Recommendations

### Current Behavior is Intentional and Valuable

The dual-chunking strategy (class + methods) is **working as designed** and provides:
1. **Hierarchical search**: Find classes, then drill into methods
2. **Context preservation**: Class docstrings and decorators available
3. **Graph relationships**: Parent-child links enable sophisticated queries
4. **Flexibility**: Users can search at class or method granularity

### When to Consider Alternatives

Only modify if:
1. **Storage concerns**: Large class chunks use significant embedding storage
2. **Search quality**: Class chunks dominate search results unfairly
3. **Performance**: Embedding generation slow for huge chunks
4. **User feedback**: Users prefer method-only results

### Suggested Configuration Option

Add a configuration setting to control behavior:

```python
# config/constants.py
CLASS_CHUNKING_STRATEGY = "dual"  # Options: "dual", "methods_only", "summary"
MAX_CLASS_CHUNK_SIZE = 500  # Only apply to "dual" mode
```

```python
# parsers/python.py
elif node_type == "class_definition":
    class_name = self._get_node_name(node)

    if CLASS_CHUNKING_STRATEGY == "dual":
        # Current behavior
        class_chunks = self._extract_class(node, lines, file_path)
        chunks.extend(class_chunks)
    elif CLASS_CHUNKING_STRATEGY == "summary":
        # Summary-only chunk
        summary = self._extract_class_summary(node, lines, file_path)
        chunks.extend(summary)
    # else: "methods_only" - skip class chunk

    # Always visit methods
    for child in node.children:
        visit_node(child, class_name)
```

## Conclusion

**The `SemanticIndexer` class is being indexed correctly according to the current design.** The 1208-line class chunk exists alongside 31 method chunks to enable:
- Class-level semantic search
- Method-level granular search
- Hierarchical code relationships

**No bug exists.** The behavior is intentional and serves valid use cases.

**If modification is desired**, the recommended approach is:
1. Add `CLASS_CHUNKING_STRATEGY` configuration option
2. Implement conditional chunking based on strategy
3. Default to current "dual" mode for backward compatibility
4. Allow users to opt into "methods_only" or "summary" modes

## Code Examples

### Current Chunking Output (Simplified)

```json
{
  "chunks": [
    {
      "id": "65a5fb62554a2f2b",
      "type": "class",
      "name": "SemanticIndexer",
      "start_line": 41,
      "end_line": 1248,
      "complexity": 132.0,
      "content": "class SemanticIndexer:\n    \"\"\"Main indexer...\"\"\"\n    ..."
    },
    {
      "id": "abc123...",
      "type": "function",
      "name": "__init__",
      "class_name": "SemanticIndexer",
      "parent_id": "65a5fb62554a2f2b",
      "start_line": 44,
      "end_line": 138,
      "content": "def __init__(self, ...):\n    ..."
    },
    {
      "id": "def456...",
      "type": "function",
      "name": "index_project",
      "class_name": "SemanticIndexer",
      "parent_id": "65a5fb62554a2f2b",
      "start_line": 278,
      "end_line": 415,
      "content": "async def index_project(self, ...):\n    ..."
    }
  ]
}
```

### Proposed "methods_only" Output

```json
{
  "chunks": [
    {
      "id": "abc123...",
      "type": "function",
      "name": "__init__",
      "class_name": "SemanticIndexer",
      "start_line": 44,
      "end_line": 138,
      "content": "def __init__(self, ...):\n    ..."
    },
    {
      "id": "def456...",
      "type": "function",
      "name": "index_project",
      "class_name": "SemanticIndexer",
      "start_line": 278,
      "end_line": 415,
      "content": "async def index_project(self, ...):\n    ..."
    }
  ]
}
```

**Note:** No class-level chunk, but `class_name` metadata preserved on methods.

## References

**Files Analyzed:**
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py` (lines 83-220)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/base.py` (lines 165-222)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/constants.py` (lines 14-16)
- `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/chunk-graph.json` (runtime data)

**Key Code Sections:**
- `visit_node()` recursive traversal (python.py:90-118)
- `_extract_class()` class chunk creation (python.py:184-220)
- `_extract_function()` method chunk creation (python.py:137-182)

---

**Research completed:** 2025-12-14
**Analyst:** Claude (Research Agent)
