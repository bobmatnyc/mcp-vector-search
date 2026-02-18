# AST-Based Knowledge Graph Generation Architecture

**Date:** 2026-02-16
**Investigator:** Research Agent
**Context:** Design an AST-based approach to automatically generate the Knowledge Graph from tree-sitter AST
**Issue:** Current manual string parsing in `kg_builder.py` is fragile (IMPORTS shows 0 count despite fixes)

---

## Executive Summary

The project **already has extensive AST infrastructure** via tree-sitter parsers that extract rich metadata during indexing. However, the Knowledge Graph builder (`kg_builder.py`) **ignores this metadata** and attempts to re-parse chunks using manual string parsing. This creates a disconnect where:

1. **AST extraction populates**: `chunk.imports`, `chunk.calls`, `chunk.inherits_from` during parsing
2. **KG builder reads**: These fields but they contain **structured AST data** (dictionaries for imports)
3. **KG builder expects**: Simple strings, causing parsing failures
4. **Result**: IMPORTS = 0 despite having data

**Recommendation:** The KG builder should **consume AST metadata directly** instead of re-parsing. The infrastructure is already there—we just need to fix the data format mismatch.

---

## Current AST Infrastructure (Already Exists)

### 1. Tree-Sitter Parsers

**Location:** `src/mcp_vector_search/parsers/`

**Available Parsers:**
- Python (`python.py`)
- JavaScript/TypeScript (`javascript.py`)
- Rust (`rust.py`)
- Go (`go.py`)
- Java (`java.py`)
- C# (`csharp.py`)
- Ruby (`ruby.py`)
- Dart (`dart.py`)
- PHP (`php.py`)

**Primary Parser:** Python (most mature, has helper modules)

### 2. Python AST Extraction Helpers

**Location:** `src/mcp_vector_search/parsers/python_helpers/`

**Key Modules:**

#### `metadata_extractor.py`
Contains static methods for extracting AST metadata:

```python
class MetadataExtractor:
    @staticmethod
    def extract_imports(node, source_code: bytes) -> list[dict]
        """Returns: [{"module": "os", "names": ["path"], "alias": None}, ...]"""

    @staticmethod
    def extract_function_calls(node, source_code: bytes) -> list[str]
        """Returns: ['print', 'self.save', 'db.query']"""

    @staticmethod
    def extract_class_bases(node, source_code: bytes) -> list[str]
        """Returns: ['BaseModel', 'ABC', 'Generic[T]']"""

    @staticmethod
    def extract_decorators(node, lines: list[str]) -> list[str]
        """Returns: ['@property', '@staticmethod']"""

    @staticmethod
    def extract_parameters(node) -> list[dict]
        """Returns: [{"name": "x", "type": "int", "default": "0"}, ...]"""

    @staticmethod
    def extract_return_type(node) -> str | None
        """Returns: "list[str]" or None"""
```

**Import Extraction Implementation** (Lines 182-329):
- Traverses AST to find `import_statement` and `import_from_statement` nodes
- Handles both `import x` and `from x import y` formats
- Extracts module names, imported symbols, and aliases
- Returns structured dictionaries with fields: `module`, `names`, `alias`

**Function Call Extraction** (Lines 124-179):
- Walks AST tree recursively to find `call` nodes
- Extracts function names from identifiers and attributes
- Handles simple calls (`print()`) and method calls (`self.save()`)
- Returns list of called function names

**Class Inheritance Extraction** (Lines 332-360):
- Finds `argument_list` in `class_definition` nodes
- Extracts base class names from arguments
- Returns list of parent class names

#### `node_extractors.py`
Contains extractors for different code elements:

```python
class FunctionExtractor(NodeExtractorBase):
    def extract(self, node, lines, file_path, class_name=None) -> list[CodeChunk]
        """Extracts function as chunk with:
        - Docstring
        - Decorators
        - Parameters (with types)
        - Return type
        - Function calls (via MetadataExtractor.extract_function_calls)
        """

class ClassExtractor(NodeExtractorBase):
    def extract(self, node, lines, file_path) -> list[CodeChunk]
        """Extracts class as chunk with:
        - Docstring
        - Decorators
        - Base classes (via MetadataExtractor.extract_class_bases)
        - Class skeleton (no method bodies)
        """

class ModuleExtractor(NodeExtractorBase):
    def extract(self, node, lines, file_path) -> CodeChunk | None
        """Extracts module-level imports as chunk with:
        - Structured imports (via MetadataExtractor.extract_imports)
        """
```

**Key Point:** These extractors **already call** `MetadataExtractor` methods during parsing and populate `CodeChunk` fields.

### 3. CodeChunk Model

**Location:** `src/mcp_vector_search/core/models.py`

**Relevant Fields:**
```python
@dataclass
class CodeChunk:
    # ... basic fields ...

    imports: list[str] = None       # ❌ Type mismatch - should be list[dict]
    calls: list[str] = None         # ✅ Correctly typed
    inherits_from: list[str] = None # ✅ Correctly typed
```

**Type Mismatch Issue:**
- `MetadataExtractor.extract_imports()` returns `list[dict]` with structure:
  ```python
  [
      {"module": "pathlib", "names": ["Path"], "alias": None},
      {"module": "typing", "names": ["Optional", "Dict"], "alias": None}
  ]
  ```
- But `CodeChunk.imports` is typed as `list[str]`
- This creates serialization/parsing confusion in KG builder

### 4. Chunk Creation Flow

**Location:** `src/mcp_vector_search/parsers/base.py`

**Method:** `BaseParser._create_chunk()` (Lines 167-247)

```python
def _create_chunk(
    self,
    # ... other params ...
    imports: list[dict] | None = None,  # ← Accepts list[dict]
    calls: list[str] | None = None,
    inherits_from: list[str] | None = None,
) -> CodeChunk:
    # ...
    return CodeChunk(
        # ...
        imports=imports or [],      # ← Passes list[dict] to list[str] field
        calls=calls or [],
        inherits_from=inherits_from or [],
    )
```

**Data Flow:**
1. Tree-sitter parses file → AST nodes
2. `MetadataExtractor.extract_imports(node)` → `list[dict]`
3. `ModuleExtractor.extract()` → passes to `_create_chunk(imports=list[dict])`
4. `CodeChunk` created with `imports: list[dict]` (despite type hint saying `list[str]`)
5. Chunk serialized to ChromaDB with structured import data
6. KG builder reads chunk, expects `list[str]`, gets `list[dict]` → **parsing fails**

---

## Root Cause of IMPORTS = 0

**Location:** `src/mcp_vector_search/core/kg_builder.py` (Lines 878-908)

### Current Implementation

```python
def _extract_code_entity(self, chunk: CodeChunk) -> ...:
    import_relationships: list[CodeRelationship] = []

    if hasattr(chunk, "imports") and chunk.imports:
        for imp in chunk.imports:
            # Parse import statement to extract module name
            import_str = imp.get("module", "") if isinstance(imp, dict) else str(imp)
            module_name = self._parse_module_name(import_str)  # ← STRING PARSING

            if module_name:
                module_id = f"module:{module_name}"
                # Create module entity...
```

### The Problem

1. **Chunk has structured data**: `chunk.imports = [{"module": "os", "names": ["path"], ...}]`
2. **KG builder checks type**: `isinstance(imp, dict)` → **True**
3. **Extracts module**: `imp.get("module")` → `"os"` ✅
4. **Calls string parser**: `self._parse_module_name("os")`
5. **Parser expects import statement**: `"import os"` or `"from os import path"`
6. **Gets simple string**: `"os"`
7. **Parsing logic fails**: No "import" or "from" keyword detected
8. **Returns**: `None`
9. **Module skipped**: No relationship created

### Why `_parse_module_name` Fails

**Method:** `kg_builder.py:826-872`

```python
def _parse_module_name(self, import_statement: str) -> str | None:
    if not import_statement or not isinstance(import_statement, str):
        return None

    import_statement = import_statement.strip()

    # Handle "from X import Y" format
    if import_statement.startswith("from "):  # ← Expects "from os import path"
        # ...
        return module

    # Handle "import X" format
    elif import_statement.startswith("import "):  # ← Expects "import os"
        # ...
        return module

    return None  # ← Returns None for "os" (no keyword)
```

**Actual input:** `"os"` (just the module name)
**Expected input:** `"import os"` or `"from os import path"` (full statement)
**Result:** Returns `None` → No IMPORTS relationship created

---

## Why Manual Parsing is Fragile

The current approach has multiple layers of parsing:

```
Source Code
    ↓
Tree-sitter AST ← PARSE LAYER 1 (robust, language-aware)
    ↓
CodeChunk.imports = [{"module": "os", ...}]  ← Structured AST data
    ↓
Serialized to ChromaDB metadata
    ↓
KG Builder reads chunk
    ↓
_parse_module_name(import_str) ← PARSE LAYER 2 (fragile, regex-based)
    ↓
Returns None (fails on "os" without "import" keyword)
    ↓
IMPORTS = 0
```

**Problems:**
1. **Redundant parsing**: Re-parsing data already extracted by AST
2. **Data loss**: Structured AST data (`names`, `alias`) discarded
3. **Format mismatch**: Parser expects different format than AST provides
4. **Fragile regex**: Manual string parsing breaks on edge cases
5. **Language-specific**: Must implement parsing for each language
6. **No validation**: Errors silently return `None`

---

## AST-Based Architecture (Proposed)

### Core Principle

**Don't re-parse what's already parsed.** The AST has already extracted structured relationship data—consume it directly.

### Data Flow

```
Source Code
    ↓
Tree-sitter AST (PARSE ONCE)
    ↓
MetadataExtractor (extracts structured data)
    ↓
CodeChunk with AST metadata:
    - imports: list[dict]     ← {"module": "os", "names": ["path"], "alias": None}
    - calls: list[str]        ← ["print", "db.save"]
    - inherits_from: list[str] ← ["BaseModel", "ABC"]
    ↓
Serialized to ChromaDB
    ↓
KG Builder (NO PARSING - direct consumption)
    ↓
    for imp in chunk.imports:
        module_id = f"module:{imp['module']}"  ← Direct access
        create_relationship(chunk_id, module_id, "IMPORTS")
    ↓
IMPORTS > 0 ✅
```

### Implementation Changes

#### 1. Fix CodeChunk Type Annotation

**File:** `src/mcp_vector_search/core/models.py`

```python
@dataclass
class CodeChunk:
    # ...
    imports: list[dict] = None  # ← Change from list[str] to list[dict]
    calls: list[str] = None
    inherits_from: list[str] = None
```

**Structure of `imports`:**
```python
[
    {
        "module": "pathlib",        # Module being imported
        "names": ["Path", "PurePath"],  # Specific names imported (or ["*"])
        "alias": None               # Import alias (e.g., "pd" for pandas)
    },
    {
        "module": "typing",
        "names": ["Optional", "Dict"],
        "alias": None
    }
]
```

#### 2. Simplify KG Builder Import Processing

**File:** `src/mcp_vector_search/core/kg_builder.py`

**Replace** `_extract_code_entity()` import handling (Lines 878-908):

```python
# BEFORE (manual parsing):
if hasattr(chunk, "imports") and chunk.imports:
    for imp in chunk.imports:
        import_str = imp.get("module", "") if isinstance(imp, dict) else str(imp)
        module_name = self._parse_module_name(import_str)  # ← Remove this
        if module_name:
            module_id = f"module:{module_name}"
            # ...

# AFTER (direct AST consumption):
if hasattr(chunk, "imports") and chunk.imports:
    for imp in chunk.imports:
        # imp is already dict with {"module": "os", "names": [...], "alias": ...}
        module_name = imp.get("module")
        if module_name and not module_name.startswith("."):  # Skip relative imports
            module_id = f"module:{module_name}"

            # Create module entity
            module_entity = CodeEntity(
                id=module_id,
                name=module_name,
                entity_type="module",
                file_path=None,
            )
            module_entities.append(module_entity)

            # Create IMPORTS relationship
            import_relationships.append(
                CodeRelationship(
                    source_id=chunk_id,
                    target_id=module_id,
                    relationship_type="imports",
                )
            )
```

**Key Changes:**
- ❌ Remove `_parse_module_name()` call
- ✅ Access `imp["module"]` directly
- ✅ Skip relative imports (`.` prefix check)
- ✅ No string parsing required
- ✅ Works for all languages with AST extraction

#### 3. Remove Dead Code

**Delete:** `_parse_module_name()` method (Lines 826-872)

**Reason:** No longer needed—AST already extracted module names

#### 4. Enhance Other Relationships

**CALLS Relationships** (Already works):
```python
if hasattr(chunk, "calls") and chunk.calls:
    for called in chunk.calls:  # ← Already list[str]
        target_id = self._resolve_entity(called)
        if target_id:
            relationships["CALLS"].append(
                CodeRelationship(
                    source_id=chunk_id,
                    target_id=target_id,
                    relationship_type="calls",
                )
            )
```

**INHERITS Relationships** (Already works):
```python
if hasattr(chunk, "inherits_from") and chunk.inherits_from:
    for base in chunk.inherits_from:  # ← Already list[str]
        target_id = self._resolve_entity(base)
        if target_id:
            relationships["INHERITS"].append(
                CodeRelationship(
                    source_id=chunk_id,
                    target_id=target_id,
                    relationship_type="inherits",
                )
            )
```

**Why these work:**
- AST extractors return `list[str]` for calls and inheritance
- KG builder consumes them directly
- No parsing needed

---

## Additional AST Capabilities (Future Enhancements)

The AST infrastructure can extract even more relationships:

### 1. TYPE_OF Relationships

**What:** Links variables/parameters to their type annotations

**AST Data Available:**
- Function parameters: `[{"name": "x", "type": "int", "default": "0"}, ...]`
- Return types: `"list[str]"`
- Class attributes with type hints

**Example:**
```python
def process(data: DataFrame, config: Config) -> Result:
    pass
```

**Relationships:**
- `(process, TYPE_OF, DataFrame)` ← Parameter type
- `(process, TYPE_OF, Config)` ← Parameter type
- `(process, RETURNS, Result)` ← Return type

**Implementation:**
```python
if hasattr(chunk, "parameters") and chunk.parameters:
    for param in chunk.parameters:
        param_type = param.get("type")
        if param_type:
            relationships["TYPE_OF"].append(
                CodeRelationship(
                    source_id=f"{chunk_id}:param:{param['name']}",
                    target_id=f"type:{param_type}",
                    relationship_type="type_of",
                )
            )
```

### 2. DECORATES Relationships

**What:** Links decorators to decorated functions/classes

**AST Data Available:**
- `chunk.decorators = ["@property", "@staticmethod", "@app.route('/api')"]`

**Example:**
```python
@app.route('/users')
@require_auth
def get_users():
    pass
```

**Relationships:**
- `(app.route, DECORATES, get_users)`
- `(require_auth, DECORATES, get_users)`

**Implementation:**
```python
if hasattr(chunk, "decorators") and chunk.decorators:
    for decorator in chunk.decorators:
        # Extract decorator name (strip @ and arguments)
        dec_name = decorator.lstrip("@").split("(")[0]
        relationships["DECORATES"].append(
            CodeRelationship(
                source_id=f"decorator:{dec_name}",
                target_id=chunk_id,
                relationship_type="decorates",
            )
        )
```

### 3. USES_FRAMEWORK Relationships

**What:** Links code entities to frameworks/libraries

**AST Data Available:**
- Import statements identify frameworks
- Common patterns: `fastapi`, `django`, `flask`, `pytest`

**Example:**
```python
from fastapi import FastAPI, APIRouter
from pytest import fixture
```

**Relationships:**
- `(module, USES_FRAMEWORK, FastAPI)`
- `(module, USES_FRAMEWORK, pytest)`

**Implementation:**
```python
FRAMEWORK_MODULES = {
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "pytest": "pytest",
    "pandas": "pandas",
    "numpy": "numpy",
}

if hasattr(chunk, "imports") and chunk.imports:
    for imp in chunk.imports:
        module = imp.get("module")
        if module in FRAMEWORK_MODULES:
            framework = FRAMEWORK_MODULES[module]
            relationships["USES_FRAMEWORK"].append(
                CodeRelationship(
                    source_id=chunk_id,
                    target_id=f"framework:{framework}",
                    relationship_type="uses_framework",
                )
            )
```

### 4. COMPLEXITY_SCORE Enhancement

**What:** Use AST-based cyclomatic complexity in relationships

**AST Data Available:**
- `chunk.complexity_score` (already calculated)

**Example:**
```python
def complex_function():  # complexity_score = 15
    if x:
        if y:
            while z:
                for i in range(10):
                    # ...
```

**Relationships:**
- `(complex_function, HAS_COMPLEXITY, HIGH)` ← Complexity > 10
- `(simple_function, HAS_COMPLEXITY, LOW)` ← Complexity < 5

**Implementation:**
```python
if chunk.complexity_score > 10:
    complexity_level = "HIGH"
elif chunk.complexity_score > 5:
    complexity_level = "MEDIUM"
else:
    complexity_level = "LOW"

relationships["HAS_COMPLEXITY"].append(
    CodeRelationship(
        source_id=chunk_id,
        target_id=f"complexity:{complexity_level}",
        relationship_type="has_complexity",
    )
)
```

### 5. CONTAINS_PATTERN Relationships

**What:** Identify design patterns from AST structure

**AST Patterns:**
- Singleton: Class with private constructor and `getInstance()` method
- Factory: Class with static methods returning instances
- Decorator: Function taking function as argument
- Context Manager: Class with `__enter__` and `__exit__` methods

**Example:**
```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Relationships:**
- `(Singleton, CONTAINS_PATTERN, singleton)`

**Implementation:**
```python
def _detect_patterns(chunk: CodeChunk) -> list[str]:
    patterns = []

    # Singleton pattern
    if (chunk.chunk_type == "class" and
        any(d == "@singleton" for d in chunk.decorators)):
        patterns.append("singleton")

    # Context manager
    if (chunk.chunk_type == "class" and
        any(m in chunk.calls for m in ["__enter__", "__exit__"])):
        patterns.append("context_manager")

    # Factory pattern
    if (chunk.function_name and
        chunk.function_name.startswith("create_") and
        chunk.return_type):
        patterns.append("factory")

    return patterns

# Usage:
patterns = self._detect_patterns(chunk)
for pattern in patterns:
    relationships["CONTAINS_PATTERN"].append(
        CodeRelationship(
            source_id=chunk_id,
            target_id=f"pattern:{pattern}",
            relationship_type="contains_pattern",
        )
    )
```

---

## Implementation Plan

### Phase 1: Fix IMPORTS (Immediate - 1 day)

**Goal:** Get IMPORTS count > 0 by fixing type mismatch

**Tasks:**
1. ✅ Update `CodeChunk.imports` type annotation to `list[dict]`
2. ✅ Simplify `_extract_code_entity()` to consume `imp["module"]` directly
3. ✅ Remove `_parse_module_name()` method
4. ✅ Add unit test for import relationship creation
5. ✅ Verify with `mcp-vector-search kg build`

**Files to Modify:**
- `src/mcp_vector_search/core/models.py` (Line 23)
- `src/mcp_vector_search/core/kg_builder.py` (Lines 878-908, 826-872)

**Test:**
```bash
mcp-vector-search index --force
mcp-vector-search kg build --force
mcp-vector-search kg query "MATCH ()-[r:IMPORTS]->() RETURN count(r)"
# Expected: count > 0
```

### Phase 2: Enhance AST Consumption (Short-term - 3 days)

**Goal:** Add more AST-based relationships

**Tasks:**
1. Add TYPE_OF relationships from parameter types
2. Add DECORATES relationships from decorators
3. Add USES_FRAMEWORK relationships from imports
4. Add HAS_COMPLEXITY relationships from complexity scores

**Files to Modify:**
- `src/mcp_vector_search/core/kg_builder.py` (Add new relationship extractors)
- `src/mcp_vector_search/core/knowledge_graph.py` (Add new relationship types to schema)

**Test:**
```bash
mcp-vector-search kg query "MATCH ()-[r:TYPE_OF]->() RETURN count(r)"
mcp-vector-search kg query "MATCH ()-[r:DECORATES]->() RETURN count(r)"
mcp-vector-search kg query "MATCH ()-[r:USES_FRAMEWORK]->() RETURN count(r)"
```

### Phase 3: Pattern Detection (Medium-term - 1 week)

**Goal:** Automatically detect design patterns from AST structure

**Tasks:**
1. Implement pattern detection algorithms
2. Add CONTAINS_PATTERN relationship type
3. Build pattern detection rules (singleton, factory, etc.)
4. Add pattern entity type to KG schema

**Files to Modify:**
- `src/mcp_vector_search/core/kg_builder.py` (Add `_detect_patterns()` method)
- `src/mcp_vector_search/core/knowledge_graph.py` (Add Pattern node type)

### Phase 4: Multi-Language Support (Long-term - 2 weeks)

**Goal:** Extend AST-based KG to all supported languages

**Tasks:**
1. Audit AST extraction in non-Python parsers
2. Standardize AST metadata format across languages
3. Test KG generation with JavaScript, Rust, Go, etc.
4. Add language-specific relationship types if needed

**Files to Modify:**
- `src/mcp_vector_search/parsers/javascript.py`
- `src/mcp_vector_search/parsers/rust.py`
- `src/mcp_vector_search/parsers/go.py`
- (Other language parsers)

---

## Benefits of AST-Based Approach

### 1. Correctness
- ✅ No string parsing errors
- ✅ Language-aware extraction
- ✅ Handles edge cases (nested imports, aliases, etc.)
- ✅ Type-safe data structures

### 2. Performance
- ✅ Single parse pass (tree-sitter during indexing)
- ✅ No re-parsing in KG builder
- ✅ Direct data access (no regex)
- ✅ Faster KG build times

### 3. Maintainability
- ✅ Less code (remove parsing logic)
- ✅ Centralized AST extraction (MetadataExtractor)
- ✅ Easier to add new relationships
- ✅ Language-agnostic approach

### 4. Extensibility
- ✅ AST provides rich metadata for new relationship types
- ✅ Easy to add TYPE_OF, DECORATES, etc.
- ✅ Pattern detection from AST structure
- ✅ Framework/library detection

### 5. Consistency
- ✅ Same data used for indexing and KG
- ✅ No data format translation
- ✅ Single source of truth (AST)
- ✅ Reproducible results

---

## Risks and Mitigations

### Risk 1: Backward Compatibility

**Problem:** Existing ChromaDB chunks may have old import format

**Mitigation:**
- Check type at runtime: `isinstance(imp, dict)`
- Support both `list[dict]` (new) and `list[str]` (old)
- Add migration script to re-index old chunks
- Graceful degradation for old data

**Code:**
```python
if hasattr(chunk, "imports") and chunk.imports:
    for imp in chunk.imports:
        if isinstance(imp, dict):
            # New format (AST-based)
            module_name = imp.get("module")
        else:
            # Old format (string-based) - fallback
            module_name = self._parse_module_name(str(imp))

        if module_name:
            # Create relationship...
```

### Risk 2: Language Coverage

**Problem:** Non-Python parsers may not extract full AST metadata

**Mitigation:**
- Audit each language parser for AST extraction
- Implement missing extractors (copy pattern from Python)
- Fall back to basic parsing if AST unavailable
- Document language support matrix

**Priority:**
1. Python (✅ Full AST support)
2. JavaScript/TypeScript (⚠️ Needs audit)
3. Rust, Go, Java (⚠️ Needs audit)
4. Others (❌ Basic support)

### Risk 3: AST Data Size

**Problem:** Storing full AST metadata in chunks increases storage

**Mitigation:**
- AST data is already being stored (no change)
- Only store metadata (not full AST tree)
- Metadata size: ~100-500 bytes per chunk (negligible)
- ChromaDB compresses metadata automatically

### Risk 4: Re-indexing Required

**Problem:** Fixing type mismatch requires re-indexing old projects

**Mitigation:**
- Make re-indexing optional (graceful degradation)
- Add `mcp-vector-search migrate` command
- Document migration process
- Provide auto-migration on first `kg build`

---

## Testing Strategy

### Unit Tests

**File:** `tests/unit/core/test_kg_builder.py`

```python
def test_import_extraction_from_ast():
    """Test that KG builder correctly consumes AST import data."""
    chunk = CodeChunk(
        content="import os\nfrom pathlib import Path",
        file_path=Path("test.py"),
        start_line=1,
        end_line=2,
        language="python",
        chunk_type="imports",
        imports=[
            {"module": "os", "names": ["*"], "alias": None},
            {"module": "pathlib", "names": ["Path"], "alias": None},
        ],
    )

    builder = KGBuilder(kg, project_root)
    entity, relationships, modules = builder._extract_code_entity(chunk)

    # Should create 2 module entities
    assert len(modules) == 2
    assert any(m.name == "os" for m in modules)
    assert any(m.name == "pathlib" for m in modules)

    # Should create 2 IMPORTS relationships
    assert len(relationships["IMPORTS"]) == 2


def test_calls_extraction_from_ast():
    """Test that KG builder correctly consumes AST calls data."""
    chunk = CodeChunk(
        content="def foo():\n    print('hello')\n    self.save()",
        file_path=Path("test.py"),
        start_line=1,
        end_line=3,
        language="python",
        chunk_type="function",
        function_name="foo",
        calls=["print", "self.save"],
    )

    builder = KGBuilder(kg, project_root)
    entity, relationships, modules = builder._extract_code_entity(chunk)

    # Should create CALLS relationships
    assert len(relationships["CALLS"]) == 2


def test_inherits_extraction_from_ast():
    """Test that KG builder correctly consumes AST inheritance data."""
    chunk = CodeChunk(
        content="class Foo(BaseModel, ABC):\n    pass",
        file_path=Path("test.py"),
        start_line=1,
        end_line=2,
        language="python",
        chunk_type="class",
        class_name="Foo",
        inherits_from=["BaseModel", "ABC"],
    )

    builder = KGBuilder(kg, project_root)
    entity, relationships, modules = builder._extract_code_entity(chunk)

    # Should create INHERITS relationships
    assert len(relationships["INHERITS"]) == 2
```

### Integration Tests

**File:** `tests/integration/test_ast_kg.py`

```python
def test_full_ast_to_kg_pipeline():
    """Test complete flow: Parse → AST → Chunk → KG."""
    # 1. Parse Python file
    parser = PythonParser()
    chunks = await parser.parse_file(Path("sample.py"))

    # 2. Verify AST metadata populated
    assert any(chunk.imports for chunk in chunks)
    assert any(chunk.calls for chunk in chunks)

    # 3. Build KG from chunks
    builder = KGBuilder(kg, project_root)
    stats = builder.build_from_chunks_sync(chunks)

    # 4. Verify relationships created
    assert stats["imports"] > 0
    assert stats["calls"] > 0

    # 5. Query KG
    result = kg.conn.execute("MATCH ()-[r:IMPORTS]->() RETURN count(r)")
    import_count = result.get_next()[0]
    assert import_count > 0
```

### End-to-End Tests

```bash
# Test full CLI workflow
mcp-vector-search index --force /path/to/project
mcp-vector-search kg build --force

# Query specific relationship types
mcp-vector-search kg query "MATCH (c:CodeEntity)-[r:IMPORTS]->(m:CodeEntity) WHERE m.entity_type = 'module' RETURN c.name, m.name LIMIT 10"

mcp-vector-search kg query "MATCH (c:CodeEntity)-[r:CALLS]->(f:CodeEntity) RETURN c.name, f.name LIMIT 10"

mcp-vector-search kg query "MATCH (c:CodeEntity)-[r:INHERITS]->(b:CodeEntity) RETURN c.name, b.name LIMIT 10"
```

---

## Conclusion

The project **already has robust AST infrastructure** via tree-sitter parsers. The issue is not missing AST capabilities—it's that the KG builder **ignores the AST metadata** and attempts fragile string parsing instead.

**Root Cause:** Type mismatch between AST data format (`list[dict]`) and expected format (`list[str]`) causes parsing to fail.

**Solution:** Remove manual parsing and consume AST metadata directly. This requires:
1. Fix type annotation: `CodeChunk.imports: list[dict]`
2. Simplify KG builder: `imp["module"]` instead of `_parse_module_name()`
3. Remove dead code: Delete string parsing methods

**Impact:**
- ✅ IMPORTS relationships will work immediately
- ✅ More reliable, faster, maintainable KG generation
- ✅ Easier to add new relationship types (TYPE_OF, DECORATES, etc.)
- ✅ Language-agnostic approach (works for all tree-sitter parsers)

**Next Steps:**
1. Implement Phase 1 fixes (1 day)
2. Test with real project (`mcp-vector-search` itself)
3. Verify IMPORTS > 0 in KG stats
4. Plan Phase 2 enhancements (TYPE_OF, DECORATES, etc.)

---

## Appendix: Key Files

### AST Extraction
- `src/mcp_vector_search/parsers/python_helpers/metadata_extractor.py`
  - `extract_imports()` (Lines 182-329)
  - `extract_function_calls()` (Lines 124-179)
  - `extract_class_bases()` (Lines 332-360)
- `src/mcp_vector_search/parsers/python_helpers/node_extractors.py`
  - `FunctionExtractor.extract()` (Lines 26-86)
  - `ClassExtractor.extract()` (Lines 92-140)
  - `ModuleExtractor.extract()` (Lines 146-181)

### Data Models
- `src/mcp_vector_search/core/models.py`
  - `CodeChunk` (Lines 11-50) - **Type mismatch on Line 23**

### KG Builder
- `src/mcp_vector_search/core/kg_builder.py`
  - `_extract_code_entity()` (Lines 791-975) - **Broken import handling**
  - `_parse_module_name()` (Lines 826-872) - **Delete this method**

### Integration Points
- `src/mcp_vector_search/parsers/base.py`
  - `_create_chunk()` (Lines 167-247) - Accepts `list[dict]` but passes to `list[str]` field

---

**Research Complete**
**Recommendation:** Proceed with Phase 1 implementation to fix IMPORTS relationships immediately.
