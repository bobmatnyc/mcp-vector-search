# AST Extraction Pipeline Analysis for Knowledge Graph Foundation (Issue #97)

**Research Date:** 2026-02-15
**Research Agent:** Claude Sonnet 4.5
**Issue:** #97 - Knowledge Graph Performance Foundation
**Focus:** Understanding current AST extraction pipeline and integration points for entity/relationship extraction

---

## Executive Summary

The mcp-vector-search codebase has a **robust, multi-language AST extraction pipeline** built on tree-sitter with extensive metadata capture capabilities. The architecture is well-suited for knowledge graph enhancement with **minimal structural changes required**. Key findings:

1. **Strong Foundation**: CodeChunk model already captures rich entity metadata (functions, classes, decorators, parameters, types)
2. **Performance-Optimized**: Multi-process parsing (14 workers on M4 Max), two-phase architecture (chunks → embeddings)
3. **Relationship Infrastructure Exists**: RelationshipStore already extracts function calls using Python AST
4. **Integration-Ready**: Clear extension points for entity/relationship extraction without disrupting existing flow
5. **Import Tracking Present**: CodeChunk.imports field exists but **not currently populated** by parsers

---

## 1. Current Chunk Processing Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Chunk Creation (Fast, Durable)
├─ File Discovery (FileDiscovery)
│  └─ Glob patterns, ignore rules, file extensions
├─ Multiprocess Parsing (ChunkProcessor)
│  ├─ ProcessPoolExecutor (14 workers on M4 Max)
│  ├─ Parser Selection (ParserRegistry)
│  └─ AST Traversal (tree-sitter)
├─ Chunk Extraction
│  ├─ Functions/Methods
│  ├─ Classes/Interfaces
│  ├─ Modules/Imports
│  └─ Docstrings/Comments
├─ Hierarchy Building
│  ├─ Parent-child relationships (class → method)
│  ├─ Depth calculation (0=module, 1=class, 2=method)
│  └─ chunk_id linkage (via parent_chunk_id)
└─ Storage → chunks.lance (LanceDB)

Phase 2: Embedding (Resumable)
├─ Pending Chunk Retrieval (chunks_backend)
├─ Batch Embedding Generation (SentenceTransformer)
└─ Storage → vectors.lance (LanceDB)

Post-Index: Relationship Computation (Optional)
├─ Semantic Relationships (vector similarity)
└─ Caller Relationships (AST-based call analysis)
```

### Key Files and Responsibilities

| Component | File | Role |
|-----------|------|------|
| **Orchestration** | `core/indexer.py` | SemanticIndexer - coordinates phases |
| **Chunk Processing** | `core/chunk_processor.py` | ChunkProcessor - multiprocess parsing coordinator |
| **Data Model** | `core/models.py` | CodeChunk - 23 fields including hierarchy/metadata |
| **Parser Base** | `parsers/base.py` | BaseParser - abstract interface, complexity calculation |
| **Python Parser** | `parsers/python.py` | PythonParser - tree-sitter AST traversal |
| **Extractors** | `parsers/python_helpers/node_extractors.py` | FunctionExtractor, ClassExtractor, ModuleExtractor |
| **Metadata** | `parsers/python_helpers/metadata_extractor.py` | Decorators, parameters, return types |
| **Relationships** | `core/relationships.py` | RelationshipStore - pre-computed caller/semantic links |

---

## 2. Parser Architecture

### BaseParser Design

The `BaseParser` abstract class provides:

```python
class BaseParser(ABC):
    def __init__(self, language: str)

    @abstractmethod
    async def parse_file(self, file_path: Path) -> list[CodeChunk]

    @abstractmethod
    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]

    def _calculate_complexity(self, node, language: str | None = None) -> float
        """Cyclomatic complexity from AST node"""

    def _create_chunk(self, ...) -> CodeChunk
        """Factory method for CodeChunk creation with all metadata"""
```

**Key Capabilities:**
- ✅ Complexity scoring (cyclomatic complexity via decision point counting)
- ✅ Hierarchical chunk creation (parent_chunk_id, chunk_depth)
- ✅ Language-agnostic interface (15 parsers: Python, JS/TS, Rust, Go, Java, etc.)
- ✅ Graceful degradation (FallbackParser for unsupported languages)

### Tree-sitter Integration

**Python Parser Example:**

```python
class PythonParser(BaseParser):
    def _initialize_parser(self):
        from tree_sitter_language_pack import get_language, get_parser
        self._language = get_language("python")
        self._parser = get_parser("python")

    def _extract_chunks_from_tree(self, tree, content: str, file_path: Path):
        def visit_node(node, current_class=None):
            if node.type == "function_definition":
                chunks.extend(self._function_extractor.extract(node, ...))
            elif node.type == "class_definition":
                chunks.extend(self._class_extractor.extract(node, ...))
                # Visit methods with class context
                for child in node.children:
                    visit_node(child, class_name)
            elif node.type == "module":
                module_chunk = self._module_extractor.extract(node, ...)
                # Visit all module-level children
```

**AST Node Types Currently Handled:**

| Language | Node Types | Extractor Class |
|----------|-----------|----------------|
| Python | `function_definition`, `class_definition`, `module`, `import_statement`, `import_from_statement` | FunctionExtractor, ClassExtractor, ModuleExtractor |
| JavaScript/TypeScript | Similar patterns | JS/TS specific extractors |
| Rust, Go, Java, etc. | Language-specific AST nodes | Dedicated parsers |

---

## 3. CodeChunk Data Model

### Current Schema (23 Fields)

```python
@dataclass
class CodeChunk:
    # Core Identification
    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # "code", "function", "class", "method", "module", "imports"

    # Entity Metadata (ALREADY CAPTURED)
    function_name: str | None = None
    class_name: str | None = None
    docstring: str | None = None
    imports: list[str] = None  # ⚠️ Field exists but NOT POPULATED

    # Complexity Metrics
    complexity_score: float = 0.0

    # Hierarchical Relationships (ALREADY BUILT)
    chunk_id: str | None = None  # SHA256 hash of file:type:name:lines:content_hash
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = None
    chunk_depth: int = 0  # 0=module, 1=class/function, 2=method

    # Enhanced Metadata (ALREADY CAPTURED)
    decorators: list[str] = None
    parameters: list[dict] = None  # [{"name": "x", "type": "int", "default": "0"}]
    return_type: str | None = None
    type_annotations: dict[str, str] = None

    # Monorepo Support
    subproject_name: str | None = None
    subproject_path: str | None = None
```

### What's Already Captured (Per Chunk)

✅ **Entity Identity:**
- Function/method names
- Class names
- Chunk type classification

✅ **Structural Metadata:**
- Decorators (`@property`, `@staticmethod`, etc.)
- Parameters with types and defaults
- Return type annotations
- Docstrings

✅ **Hierarchical Context:**
- Parent-child relationships (method → class → module)
- Depth in code structure
- Unique deterministic IDs (content-aware hashing)

✅ **Complexity Metrics:**
- Cyclomatic complexity score
- Decision point counting

### What's Missing (For KG)

❌ **Relationship Data:**
- Function calls (which functions does this chunk call?)
- Import dependencies (which modules/packages are imported?)
- Class inheritance (which classes does this class extend?)
- Method overrides (which parent methods are overridden?)

❌ **Cross-File References:**
- Import resolution (which file does "from foo import bar" refer to?)
- Call targets (which file/chunk is the called function in?)

❌ **Entity-Level Storage:**
- Entities stored as chunk metadata, not as first-class graph nodes
- No dedicated entity table/collection for querying

---

## 4. Integration Points for Knowledge Graph

### Option A: Extend Existing Extractors (Recommended)

**Approach:** Enhance `MetadataExtractor` and `NodeExtractorBase` classes to capture relationships during AST traversal.

**Implementation Pattern:**

```python
# In parsers/python_helpers/metadata_extractor.py

class MetadataExtractor:
    @staticmethod
    def extract_function_calls(node) -> list[dict]:
        """Extract function calls from AST node.

        Returns:
            [{"name": "foo", "line": 42, "type": "call"}]
        """
        calls = []
        for child in ast.walk(node):
            if child.type == "call":
                if child.func.type == "identifier":
                    calls.append({
                        "name": child.func.text.decode(),
                        "line": child.start_point[0] + 1,
                        "type": "call"
                    })
                elif child.func.type == "attribute":
                    # Handle method calls: obj.method()
                    calls.append({
                        "name": child.func.attr.text.decode(),
                        "line": child.start_point[0] + 1,
                        "type": "method_call",
                        "receiver": child.func.value.text.decode()
                    })
        return calls

    @staticmethod
    def extract_imports(node) -> list[dict]:
        """Extract import statements from AST.

        Returns:
            [{"module": "os", "names": ["path", "listdir"], "line": 1}]
        """
        imports = []
        for child in node.children:
            if child.type == "import_statement":
                # import foo, bar
                imports.append({
                    "type": "import",
                    "modules": [name.text.decode() for name in child.children if name.type == "dotted_name"],
                    "line": child.start_point[0] + 1
                })
            elif child.type == "import_from_statement":
                # from foo import bar, baz
                module = None
                names = []
                for subchild in child.children:
                    if subchild.type == "dotted_name" and module is None:
                        module = subchild.text.decode()
                    elif subchild.type in ("identifier", "aliased_import"):
                        names.append(subchild.text.decode())
                imports.append({
                    "type": "from_import",
                    "module": module,
                    "names": names,
                    "line": child.start_point[0] + 1
                })
        return imports

    @staticmethod
    def extract_class_inheritance(node) -> list[str]:
        """Extract base classes from class definition.

        Returns:
            ["BaseClass", "MixinClass"]
        """
        bases = []
        for child in node.children:
            if child.type == "argument_list":
                # class Foo(Bar, Baz): ...
                for base in child.children:
                    if base.type in ("identifier", "attribute"):
                        bases.append(base.text.decode())
        return bases
```

**Modification Sites:**

1. **parsers/python_helpers/node_extractors.py:**
   - FunctionExtractor.extract() → add calls extraction
   - ClassExtractor.extract() → add inheritance extraction
   - ModuleExtractor.extract() → populate imports properly

2. **core/models.py (CodeChunk):**
   ```python
   # ADD THESE FIELDS:
   calls: list[dict] = None  # [{"name": "foo", "line": 42, "type": "call"}]
   inherits_from: list[str] = None  # ["BaseClass", "MixinClass"]
   # POPULATE EXISTING FIELD:
   imports: list[str] = None  # Currently defined but not populated
   ```

3. **parsers/base.py (BaseParser._create_chunk):**
   ```python
   def _create_chunk(
       self,
       ...,
       calls: list[dict] | None = None,  # NEW
       inherits_from: list[str] | None = None,  # NEW
   ) -> CodeChunk:
       return CodeChunk(
           ...,
           calls=calls or [],
           inherits_from=inherits_from or [],
       )
   ```

**Advantages:**
- ✅ Minimal architectural changes
- ✅ Leverages existing AST traversal (no duplicate parsing)
- ✅ Zero performance overhead (same tree-sitter pass)
- ✅ Backward compatible (new fields optional)
- ✅ Works with all 15 language parsers (add gradually)

**Performance Impact:**
- **Negligible** - AST traversal already happens, just extract more data
- **No extra parsing passes** - relationships extracted during existing walk
- **Storage increase** - ~20% more data per chunk (3-5 calls/chunk, 5-10 imports/file)

### Option B: Post-Processing Relationship Extraction

**Approach:** Add a phase between chunking and embedding that analyzes chunks to extract relationships.

```python
# In core/chunk_processor.py

class ChunkProcessor:
    def extract_relationships(self, chunks: list[CodeChunk]) -> dict[str, Any]:
        """Extract relationships between chunks after parsing.

        Returns:
            {
                "calls": [{"source_id": "abc123", "target_name": "foo", "line": 42}],
                "imports": [{"source_id": "abc123", "module": "os"}],
                "inheritance": [{"source_id": "abc123", "base": "BaseClass"}]
            }
        """
        relationships = {"calls": [], "imports": [], "inheritance": []}

        for chunk in chunks:
            # Re-parse chunk content to extract relationships
            if chunk.language == "python":
                tree = ast.parse(chunk.content)
                # Extract calls, imports, inheritance
                # ...

        return relationships
```

**Disadvantages:**
- ❌ Requires re-parsing (performance hit)
- ❌ Duplicate AST traversal (already done in parse phase)
- ❌ More complex error handling (chunk may not be valid standalone code)
- ❌ Harder to maintain (separate from main parsing logic)

**Use Case:**
- Languages without tree-sitter support (fallback parser)
- Post-hoc analysis of existing indices (migration)

---

## 5. Already-Extracted vs. Needs-to-be-Extracted Data

### ✅ Already Extracted (Available in CodeChunk)

| Data Type | Field | Example | Extracted By |
|-----------|-------|---------|--------------|
| Function names | `function_name` | `"calculate_complexity"` | FunctionExtractor |
| Class names | `class_name` | `"PythonParser"` | ClassExtractor |
| Method-class relationships | `parent_chunk_id` | `"abc123"` → parent class chunk | ChunkProcessor.build_chunk_hierarchy() |
| Decorators | `decorators` | `["@property", "@staticmethod"]` | MetadataExtractor.extract_decorators() |
| Parameters | `parameters` | `[{"name": "x", "type": "int"}]` | MetadataExtractor.extract_parameters() |
| Return types | `return_type` | `"list[str]"` | MetadataExtractor.extract_return_type() |
| Docstrings | `docstring` | `"Calculate complexity..."` | DocstringExtractor |
| Complexity | `complexity_score` | `5.0` | BaseParser._calculate_complexity() |
| Hierarchical depth | `chunk_depth` | `2` (method level) | ChunkProcessor.build_chunk_hierarchy() |
| Chunk type | `chunk_type` | `"function"`, `"class"`, `"method"` | Parser-specific logic |

### ❌ Needs to be Extracted (For KG)

| Data Type | Storage Target | Example | Extraction Source |
|-----------|---------------|---------|-------------------|
| **Function calls** | `CodeChunk.calls` | `[{"name": "foo", "line": 42}]` | AST `call` nodes |
| **Import statements** | `CodeChunk.imports` | `["os", "pathlib.Path"]` | AST `import_statement` nodes |
| **Class inheritance** | `CodeChunk.inherits_from` | `["BaseClass", "MixinClass"]` | AST `argument_list` in class def |
| **Method overrides** | New field or metadata | `{"overrides": "BaseClass.foo"}` | Cross-reference with parent class |
| **Call targets (resolved)** | Separate entity graph | `{"call": "foo", "target_chunk_id": "xyz789"}` | Cross-file name resolution |
| **Import resolution** | Separate entity graph | `{"import": "mymodule.foo", "target_file": "src/mymodule.py"}` | File path resolution |

### Performance Baseline (Current State)

From research docs and codebase analysis:

**Phase 1: Chunking (Current Speed)**
- **Small Project** (50 files): ~2 seconds
- **Medium Project** (500 files): ~20 seconds
- **Large Project** (5000 files): ~3-4 minutes

**Breakdown (Large Project):**
```
Total: 3m 20s (200 seconds)
├─ File Discovery: 2s (1%)
├─ Tree-sitter AST Parsing: 2m 45s (82%)
├─ Chunk Extraction: 30s (15%)
└─ Hierarchy Building: 3s (2%)
```

**Bottleneck:** Tree-sitter AST parsing (CPU-bound, already parallelized with 14 workers)

**Adding Relationship Extraction:**
- **Estimated Overhead:** +5-10% (relationship extraction happens during existing AST walk)
- **New Total Time:** ~3m 30s - 3m 50s (10-30 seconds added)
- **Storage Increase:** +20% (calls, imports, inheritance data)

---

## 6. Recommended Integration Strategy

### Phase 1: Enhance Extractors (1-2 days)

**Goal:** Capture relationships during AST traversal without performance impact.

**Tasks:**
1. Add relationship extraction methods to `MetadataExtractor`:
   - `extract_function_calls(node)`
   - `extract_imports(node)`
   - `extract_class_inheritance(node)`

2. Extend `CodeChunk` model:
   ```python
   calls: list[dict] = None  # NEW
   inherits_from: list[str] = None  # NEW
   # imports already exists - just populate it
   ```

3. Modify extractors to capture relationships:
   - `FunctionExtractor.extract()` → add calls
   - `ClassExtractor.extract()` → add inheritance
   - `ModuleExtractor.extract()` → populate imports properly

4. Update `BaseParser._create_chunk()` to accept new fields

**Testing:**
- Parse 100 Python files
- Verify calls/imports/inheritance captured
- Measure performance impact (expect <5% overhead)

### Phase 2: Entity-Level Storage (2-3 days)

**Goal:** Store entities and relationships in queryable graph structure.

**Approach:**
- Add new LanceDB table: `entities.lance`
- Schema:
  ```python
  {
      "entity_id": str,  # Unique ID
      "entity_type": str,  # "function", "class", "module"
      "name": str,
      "file_path": str,
      "chunk_id": str,  # Link back to original chunk
      "metadata": dict,  # Decorators, params, return type, etc.
      "relationships": list[dict]  # [{"type": "calls", "target": "foo"}]
  }
  ```

**Tasks:**
1. Create `EntityBackend` (similar to `ChunksBackend`)
2. Extract entities from chunks during indexing
3. Build relationship index for fast lookup
4. Add API endpoints:
   - `/api/entities/{entity_id}`
   - `/api/entities/{entity_id}/relationships`
   - `/api/entities/search?query=foo`

### Phase 3: Cross-File Resolution (3-4 days)

**Goal:** Resolve calls and imports to actual target chunks/files.

**Approach:**
- Build symbol table during indexing:
  ```python
  symbol_table = {
      "function_name": [{"chunk_id": "abc", "file": "foo.py"}],
      "ClassName": [{"chunk_id": "xyz", "file": "bar.py"}]
  }
  ```
- Resolve relationships:
  ```python
  for chunk in chunks:
      for call in chunk.calls:
          target_chunk = resolve_symbol(call["name"], symbol_table)
          if target_chunk:
              chunk.calls[i]["target_chunk_id"] = target_chunk.chunk_id
  ```

**Challenges:**
- Namespace conflicts (multiple functions with same name)
- Import path resolution (relative imports, package structure)
- Cross-language calls (Python → TypeScript via API)

---

## 7. Performance Considerations

### Current Bottlenecks

From `docs/research/performance-optimization-indexing-visualization-2025-12-16.md`:

```
Total Indexing Time: 7m 15s (large project)
├─ File Parsing: 3m 20s (46%) ← BOTTLENECK
│  └─ Tree-sitter AST: 2m 45s
│  └─ Chunk extraction: 35s
├─ Embedding Generation: 2m 50s (39%)
├─ Relationship Computation: 1m 18s (18%)
└─ Database Insertion: 45s (10%)
```

### Optimizations Already in Place

1. **Multiprocessing:**
   - `ProcessPoolExecutor` with 14 workers (M4 Max)
   - Auto-detection: `_detect_optimal_workers()`
   - Environment override: `MCP_VECTOR_SEARCH_MAX_WORKERS`

2. **Two-Phase Architecture:**
   - Phase 1: Fast chunking (no embeddings) → `chunks.lance`
   - Phase 2: Resumable embedding → `vectors.lance`
   - Crash recovery (pending/processing/complete status)

3. **Batch Processing:**
   - Default batch size: 32 files
   - Environment override: `MCP_VECTOR_SEARCH_BATCH_SIZE`
   - Auto-optimization based on codebase profile

4. **Incremental Indexing:**
   - File hash change detection
   - Skip unchanged files
   - Delta updates only

### Impact of Relationship Extraction

**Expected Performance:**

| Operation | Current | With KG Extraction | Overhead |
|-----------|---------|-------------------|----------|
| AST Parsing | 2m 45s | 2m 45s | 0% (same tree) |
| Chunk Extraction | 35s | 40s | +5s (+14%) |
| Total Chunking | 3m 20s | 3m 25s | +5s (+2.5%) |

**Why Minimal Impact:**
- Relationship extraction happens during **existing** AST traversal
- No additional parsing passes required
- Data structures already in memory
- Incremental cost: extracting a few more node types

**Storage Impact:**

| Data | Size/Chunk | Total (5000 files, 25K chunks) |
|------|-----------|-------------------------------|
| Current chunks | 2KB | 50MB |
| Calls (avg 3/chunk) | +200B | +5MB |
| Imports (avg 5/file) | +100B | +0.5MB |
| Inheritance (classes only) | +50B | +1MB |
| **Total** | **~2.3KB** | **~57MB (+14%)** |

---

## 8. Existing Relationship Infrastructure

### RelationshipStore (core/relationships.py)

**Purpose:** Pre-compute chunk relationships at index time for instant visualization.

**Current Capabilities:**

```python
class RelationshipStore:
    def compute_and_store(
        self,
        chunks: list[CodeChunk],
        database: VectorDatabase,
    ) -> dict:
        """Compute relationships and save to relationships.json"""

        # 1. Semantic relationships (vector similarity)
        semantic_links = []
        for chunk in chunks:
            similar = database.search(chunk.content, k=5)
            semantic_links.append({
                "source": chunk.chunk_id,
                "targets": [s.chunk_id for s in similar]
            })

        # 2. Caller relationships (AST-based)
        caller_map = {}
        for chunk in chunks:
            calls = extract_function_calls(chunk.content)  # Uses Python AST
            caller_map[chunk.chunk_id] = list(calls)

        return {
            "semantic": semantic_links,
            "callers": caller_map
        }
```

**Key Function:**

```python
def extract_function_calls(code: str) -> set[str]:
    """Extract actual function calls from Python code using AST.

    Returns set of function names that are actually called.
    Avoids false positives from comments/docstrings.
    """
    calls = set()
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)  # Direct call: foo()
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)  # Method call: obj.foo()
    return calls
```

**Limitations:**
- Only extracts call **names**, not call **targets** (which chunk is being called?)
- Python-only (uses Python `ast` module, not tree-sitter)
- Post-processing (runs after chunking, not during AST traversal)
- Performance bottleneck: O(n²) for cross-chunk resolution

**Opportunity:**
- Move call extraction into `MetadataExtractor` (during tree-sitter pass)
- Extract for all languages (not just Python)
- Store in `CodeChunk.calls` (not separate file)
- Enable cross-file resolution (symbol table lookup)

---

## 9. Comparison: Current vs. Proposed Architecture

### Current State

```
┌─────────────────┐
│   File Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tree-sitter    │ ← Parse AST
│  AST Parsing    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunk Extraction│ ← Extract functions/classes
│  (Extractors)   │    (names, types, metadata)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CodeChunk Model │ ← Store entity data
│  (23 fields)    │    (NO relationships)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  chunks.lance   │ ← Phase 1 storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-Processing │ ← SEPARATE pass
│ Relationships   │    (RelationshipStore)
│ (relationships  │    (Python AST, not tree-sitter)
│  .json)         │
└─────────────────┘
```

### Proposed Architecture (KG Integration)

```
┌─────────────────┐
│   File Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tree-sitter    │ ← Parse AST (SAME)
│  AST Parsing    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Enhanced        │ ← Extract entities + relationships
│ Extraction      │    (functions, classes, calls, imports, inheritance)
│ (Extractors +   │    ← NEW: MetadataExtractor methods
│  Metadata       │        - extract_function_calls()
│  Extractor)     │        - extract_imports()
│                 │        - extract_class_inheritance()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Enhanced        │ ← Store entities + relationships
│ CodeChunk Model │    ← NEW FIELDS:
│  (26 fields)    │        - calls: list[dict]
│                 │        - inherits_from: list[str]
│                 │        - imports: list[str] (populated)
└────────┬────────┘
         │
         ├────────────────┬────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│chunks.lance │  │entities.lance│  │ relationships│
│             │  │              │  │  .json       │
│ (Phase 1)   │  │ (NEW: entity │  │ (Enhanced:   │
│             │  │  graph)      │  │  with targets│
└─────────────┘  └──────────────┘  └──────────────┘
```

**Key Differences:**

1. **Single AST Pass:** Relationships extracted during initial tree-sitter traversal (not post-processing)
2. **Richer CodeChunk:** Entities and relationships stored together
3. **Entity Graph:** New `entities.lance` table for graph queries
4. **Language-Agnostic:** Uses tree-sitter (works for all 15 languages), not Python AST
5. **Performance:** No duplicate parsing, minimal overhead (<5%)

---

## 10. Implementation Roadmap

### Milestone 1: Relationship Extraction (1 week)

**Goal:** Capture calls, imports, inheritance during AST traversal.

**Deliverables:**
- [ ] Add `extract_function_calls()` to `MetadataExtractor`
- [ ] Add `extract_imports()` to `MetadataExtractor`
- [ ] Add `extract_class_inheritance()` to `MetadataExtractor`
- [ ] Extend `CodeChunk` with `calls`, `inherits_from` fields
- [ ] Update `FunctionExtractor`, `ClassExtractor`, `ModuleExtractor`
- [ ] Add integration tests (100 Python files)
- [ ] Measure performance impact (expect <5% overhead)

**Success Criteria:**
- All Python files extract calls/imports/inheritance
- CodeChunk.calls populated with actual call data
- CodeChunk.imports populated (not empty)
- CodeChunk.inherits_from populated for classes
- Performance overhead <10%

### Milestone 2: Entity Storage (1 week)

**Goal:** Store entities in queryable graph structure.

**Deliverables:**
- [ ] Design entity schema (entity_id, type, name, relationships)
- [ ] Create `EntityBackend` (LanceDB table)
- [ ] Extract entities from CodeChunks during indexing
- [ ] Build relationship index (source → targets)
- [ ] Add API endpoints: `/api/entities/{id}`, `/api/entities/search`
- [ ] Add unit tests for entity CRUD operations

**Success Criteria:**
- Entities stored in `entities.lance`
- Fast lookup by entity_id (<10ms)
- Relationship queries return targets
- API endpoints functional

### Milestone 3: Cross-File Resolution (2 weeks)

**Goal:** Resolve calls/imports to actual target chunks.

**Deliverables:**
- [ ] Build symbol table (name → [chunks])
- [ ] Implement symbol resolution (handle namespaces)
- [ ] Resolve function calls to target chunks
- [ ] Resolve imports to target files
- [ ] Handle ambiguous names (multiple candidates)
- [ ] Add visualizer integration (show cross-file edges)
- [ ] Performance testing (large monorepo)

**Success Criteria:**
- 80%+ of calls resolved to target chunks
- Import paths resolved to actual files
- Visualizer displays cross-file relationships
- Resolution time <100ms per chunk

---

## 11. Risk Assessment

### Low Risk (Minimal Impact)

✅ **Extending Extractors:**
- Isolated changes to `MetadataExtractor`
- Doesn't modify core parsing logic
- Backward compatible (new fields optional)

✅ **Adding CodeChunk Fields:**
- Dataclass fields are optional (defaults to None)
- Existing code continues to work
- Storage format handles missing fields

### Medium Risk (Careful Testing Required)

⚠️ **Performance Impact:**
- Relationship extraction adds CPU work
- Mitigation: Profile with large codebases, optimize hot paths
- Fallback: Make extraction optional (flag to disable)

⚠️ **Storage Overhead:**
- +20% storage for relationship data
- Mitigation: Compress JSON, use efficient encodings
- Fallback: Prune old relationships, implement TTL

### High Risk (Requires Design Review)

🔴 **Cross-File Resolution Accuracy:**
- Namespace conflicts (multiple functions with same name)
- Import path resolution (relative imports, complex package structures)
- Mitigation: Implement fuzzy matching, confidence scores
- Fallback: Store unresolved relationships, resolve on-demand

🔴 **Schema Evolution:**
- Entity graph schema may evolve frequently
- Migration path for existing indices
- Mitigation: Version entity schema, write migrations
- Fallback: Reindex from scratch (if migration fails)

---

## 12. Recommendations

### Immediate Actions (This Week)

1. **Prototype Relationship Extraction:**
   - Start with `MetadataExtractor.extract_function_calls()`
   - Test on 10 Python files
   - Measure overhead (should be <5%)

2. **Extend CodeChunk Model:**
   - Add `calls: list[dict] = None`
   - Add `inherits_from: list[str] = None`
   - Populate `imports: list[str]` (currently unused)

3. **Update One Extractor:**
   - Choose `FunctionExtractor`
   - Add call extraction
   - Verify calls stored in CodeChunk

### Next Steps (Week 2-3)

4. **Expand to All Python Extractors:**
   - `ClassExtractor` → inheritance
   - `ModuleExtractor` → imports

5. **Add Entity Storage:**
   - Create `entities.lance` table
   - Extract entities during indexing
   - Build relationship index

6. **Performance Testing:**
   - Index 1000 Python files
   - Measure Phase 1 time impact
   - Optimize if overhead >10%

### Long-Term (Month 2+)

7. **Cross-File Resolution:**
   - Build symbol table
   - Resolve calls to targets
   - Handle namespaces

8. **Multi-Language Support:**
   - Extend to JavaScript/TypeScript
   - Add Rust, Go parsers
   - Unified relationship schema

9. **Visualization Integration:**
   - Update D3.js graph to show KG relationships
   - Add filtering (show only calls, only inheritance)
   - Performance testing (10K+ entities)

---

## 13. Conclusion

The mcp-vector-search codebase is **exceptionally well-architected** for knowledge graph integration. The existing AST extraction pipeline, hierarchical chunk model, and relationship infrastructure provide a solid foundation. Key takeaways:

**Strengths:**
- ✅ Rich entity metadata already captured (functions, classes, decorators, parameters)
- ✅ Hierarchical relationships already built (parent_chunk_id, chunk_depth)
- ✅ Performance-optimized multiprocess parsing (14 workers)
- ✅ Two-phase architecture (fast chunking, resumable embedding)
- ✅ Relationship extraction infrastructure exists (RelationshipStore)

**Opportunities:**
- 🎯 Capture relationships during AST traversal (not post-processing)
- 🎯 Populate unused `imports` field in CodeChunk
- 🎯 Add `calls` and `inherits_from` fields to CodeChunk
- 🎯 Create entity-level storage for graph queries
- 🎯 Enable cross-file resolution (symbol table)

**Minimal Changes Required:**
- ~100 lines of code in `MetadataExtractor`
- 3 new fields in `CodeChunk` (2 new + 1 populated)
- Update 3 extractors (Function, Class, Module)
- <5% performance overhead (estimated)

The integration path is clear, low-risk, and can be implemented incrementally without disrupting existing functionality. The performance foundation is strong enough to support knowledge graph queries at scale.

---

## Appendix A: File References

**Core Files:**
- `src/mcp_vector_search/core/models.py` - CodeChunk data model (23 fields)
- `src/mcp_vector_search/core/chunk_processor.py` - Multiprocess parsing orchestration
- `src/mcp_vector_search/core/indexer.py` - Two-phase indexing pipeline
- `src/mcp_vector_search/core/relationships.py` - Relationship extraction (post-processing)

**Parser Architecture:**
- `src/mcp_vector_search/parsers/base.py` - BaseParser abstract class
- `src/mcp_vector_search/parsers/python.py` - Python tree-sitter parser
- `src/mcp_vector_search/parsers/registry.py` - Parser selection logic

**Python Helpers:**
- `src/mcp_vector_search/parsers/python_helpers/metadata_extractor.py` - Decorators, params, types
- `src/mcp_vector_search/parsers/python_helpers/node_extractors.py` - Function/Class/Module extraction
- `src/mcp_vector_search/parsers/python_helpers/docstring_extractor.py` - Docstring parsing

**Research Documents:**
- `docs/research/performance-optimization-indexing-visualization-2025-12-16.md` - Performance baselines
- `docs/research/semantic-relationship-parallelization-analysis-2025-12-20.md` - Relationship computation
- `docs/research/phase3-cross-file-analysis-requirements.md` - Cross-file analysis requirements

**Performance:**
- Multiprocess worker detection: `chunk_processor._detect_optimal_workers()`
- Batch size optimization: `indexer.apply_auto_optimizations()`
- Two-phase storage: `chunks_backend.py`, `vectors_backend.py`

---

## Appendix B: Sample Data Structures

### Current CodeChunk (Simplified)

```python
CodeChunk(
    content="def calculate_complexity(node, language='python'):\n    complexity = 1.0\n    ...",
    file_path=Path("src/parsers/base.py"),
    start_line=67,
    end_line=163,
    language="python",
    chunk_type="function",
    function_name="calculate_complexity",
    class_name="BaseParser",
    docstring="Calculate cyclomatic complexity from AST node...",
    complexity_score=5.0,
    chunk_id="a1b2c3d4e5f6g7h8",
    parent_chunk_id="x9y8z7w6v5u4t3s2",  # Parent class chunk
    child_chunk_ids=[],
    chunk_depth=2,  # Method inside class
    decorators=[],
    parameters=[
        {"name": "node", "type": None, "default": None},
        {"name": "language", "type": "str", "default": "'python'"}
    ],
    return_type="float",
    imports=[],  # ⚠️ EMPTY (not populated)
)
```

### Enhanced CodeChunk (With KG Data)

```python
CodeChunk(
    # ... same fields as above ...

    # NEW: Relationship data extracted during AST traversal
    calls=[
        {"name": "hasattr", "line": 82, "type": "builtin_call"},
        {"name": "count_decision_points", "line": 160, "type": "call"}
    ],
    inherits_from=[],  # Empty (not a class)
    imports=[  # NOW POPULATED
        "typing.Pattern",
        "pathlib.Path"
    ],
)
```

### Entity Record (New Entity Graph)

```python
{
    "entity_id": "a1b2c3d4e5f6g7h8",
    "entity_type": "function",
    "name": "calculate_complexity",
    "qualified_name": "BaseParser.calculate_complexity",
    "file_path": "src/parsers/base.py",
    "chunk_id": "a1b2c3d4e5f6g7h8",  # Link to CodeChunk
    "metadata": {
        "parameters": ["node", "language"],
        "return_type": "float",
        "decorators": [],
        "docstring": "Calculate cyclomatic complexity..."
    },
    "relationships": [
        {
            "type": "calls",
            "target_name": "hasattr",
            "target_chunk_id": None,  # Builtin, no chunk
            "line": 82
        },
        {
            "type": "calls",
            "target_name": "count_decision_points",
            "target_chunk_id": "z9y8x7w6v5u4t3s2",  # Resolved to nested function
            "line": 160
        },
        {
            "type": "member_of",
            "target_name": "BaseParser",
            "target_chunk_id": "x9y8z7w6v5u4t3s2"
        }
    ]
}
```

---

**End of Research Document**
