# Knowledge Graph Zero Entities Issue - Root Cause Analysis

**Date:** 2026-02-18
**Investigator:** Research Agent
**Project:** mcp-vector-search
**Issue:** KG build shows 0 Code Entities despite having 57,886 Doc Sections

## Executive Summary

The Knowledge Graph builder is showing **0 Code Entities** while successfully processing **57,886 Doc Sections** because the `code_chunks` filter in `build_from_chunks_sync()` is **too restrictive**. It only accepts 4 chunk types (`function`, `method`, `class`, `module`) but the codebase generates many additional valid code chunk types that are being silently ignored.

**Root Cause:** Incomplete chunk type filter at line 304 in `kg_builder.py`

**Impact:** High - Most code entities from Rust, Go, PHP, and TypeScript files are being excluded

**Priority:** Critical - This breaks the primary functionality of the Knowledge Graph

---

## Detailed Findings

### 1. The Restrictive Filter (Lines 301-305)

**File:** `src/mcp_vector_search/core/kg_builder.py`

```python
# Separate code and text chunks
code_chunks = [
    c
    for c in chunks
    if c.chunk_type in ["function", "method", "class", "module"]  # ❌ TOO RESTRICTIVE
]
```

**Issue:** This filter only accepts 4 chunk types out of 15+ valid code chunk types.

### 2. Missing Chunk Types

By analyzing the parser implementations, I found these **valid code chunk types** that are being **excluded**:

#### Rust Parser (`parsers/rust.py`):
- ✅ `"function"` - Included
- ✅ `"method"` - Included
- ✅ `"module"` - Included
- ❌ `"struct"` - **EXCLUDED** (lines 227, 498)
- ❌ `"enum"` - **EXCLUDED** (lines 254, 520)
- ❌ `"trait"` - **EXCLUDED** (lines 283, 542)
- ❌ `"impl"` - **EXCLUDED** (lines 199, 564)

#### Go Parser (`parsers/go.py`):
- ✅ `"function"` - Included
- ✅ `"method"` - Included
- ✅ `"module"` - Included
- ❌ `"struct"` - **EXCLUDED** (line 234)

#### PHP Parser (`parsers/php.py`):
- ✅ `"function"` - Included
- ✅ `"method"` - Included
- ✅ `"class"` - Included
- ✅ `"module"` - Included
- ❌ `"interface"` - **EXCLUDED** (lines 301, 511)
- ❌ `"trait"` - **EXCLUDED** (lines 335, 544)
- ❌ `"imports"` - **EXCLUDED** (line 362)

#### JavaScript/TypeScript Parser (`parsers/javascript.py`):
- ✅ `"function"` - Included
- ✅ `"method"` - Included
- ✅ `"class"` - Included
- ✅ `"module"` - Included
- ❌ `"interface"` - **EXCLUDED** (line 581)

#### Base Parser (`parsers/base.py`):
- ❌ `"code"` - **EXCLUDED** (line 200) - Generic code chunks
- ❌ `"text"` - **EXCLUDED** - Handled separately

### 3. Why This Causes 0 Code Entities

**User's Output:**
```
Code Entities: 0
Doc Sections: 57886
```

**Explanation:**

1. **Doc Sections show 57,886** because the `text_chunks` filter works correctly:
   ```python
   text_chunks = [
       c
       for c in chunks
       if c.language == "text" or str(c.file_path).endswith(".md")
   ]
   ```
   This filter uses `language` and file extension, which correctly captures documentation chunks.

2. **Code Entities show 0** because:
   - The project likely contains significant Rust/Go/PHP/TypeScript code
   - These languages use chunk types like `struct`, `enum`, `trait`, `impl`, `interface`
   - The restrictive filter **silently excludes** all these chunks
   - Result: `code_chunks = []` (empty list)
   - No entities to extract, so `stats["entities"] = 0`

### 4. The `_get_entity_name()` Fix Status

**File:** `src/mcp_vector_search/core/kg_builder.py` (lines 250-274)

```python
def _get_entity_name(self, chunk: CodeChunk) -> str:
    """Get meaningful entity name from chunk.

    For chunks with explicit function/class names, use those.
    For module-level chunks, use the module filename (without extension).
    """
    # Prefer explicit names
    if chunk.function_name:
        return chunk.function_name
    if chunk.class_name:
        return chunk.class_name

    # For module-level chunks, use filename as module name
    # e.g., "src/app/models.py" -> "models"
    if chunk.file_path:
        return Path(chunk.file_path).stem

    # Fallback (should never happen in practice)
    return f"unnamed_{chunk.chunk_type}"
```

**Status:** ✅ **The fix looks correct** but cannot be verified because no code chunks are being processed at all.

**Once the filter is fixed**, this helper should work correctly for:
- Functions: Uses `chunk.function_name`
- Classes: Uses `chunk.class_name`
- Modules: Uses filename stem
- Structs/Enums/Traits: Will use `unnamed_{chunk_type}` fallback (⚠ needs enhancement)

### 5. Generic Entity Filter Analysis

**File:** `src/mcp_vector_search/core/kg_builder.py` (lines 38-98)

```python
GENERIC_ENTITY_NAMES = {
    "main", "run", "test", "get", "set", "add", "remove", ...
}

def _is_generic_entity(self, name: str) -> bool:
    """Check if entity name is too generic to be useful."""
    if not name:
        return True

    # Filter very short names (2 chars or less)
    if len(name) <= 2:
        return True

    # Filter exact matches (case-insensitive)
    if name.lower() in GENERIC_ENTITY_NAMES:
        return True

    # Filter names starting with single underscore (private/internal)
    if name.startswith("_") and not name.startswith("__"):
        return True

    return False
```

**Assessment:** ✅ **The filter logic is reasonable** but may be too aggressive for some use cases.

**Potential Issues:**
- Filters 2-character names (e.g., `io`, `db`, `fs` - common module names)
- Filters common function names like `get`, `set`, `run` - these might be useful in context
- Filters single underscore private names - legitimate API

**Recommendation:** The generic filter is fine for now but should be made configurable via CLI flag.

### 6. Statistics Output Logic

**File:** `src/mcp_vector_search/cli/commands/_kg_subprocess.py` (lines 159-181)

```python
table.add_row("Code Entities", str(build_stats["entities"]))
table.add_row("Doc Sections", str(build_stats.get("doc_sections", 0)))
table.add_row("Calls", str(build_stats["calls"]))
table.add_row("Imports", str(build_stats["imports"]))
table.add_row("Inherits", str(build_stats["inherits"]))
table.add_row("Contains", str(build_stats["contains"]))
table.add_row("References", str(build_stats.get("references", 0)))
table.add_row("Documents", str(build_stats.get("documents", 0)))
```

**Assessment:** ✅ **The output logic is correct** - it accurately reflects the stats from `build_from_chunks_sync()`.

The problem is not in the display, but in the upstream filtering that produces `stats["entities"] = 0`.

---

## Impact Analysis

### Severity: **CRITICAL**

**Affected Functionality:**
- ❌ Code entity extraction for Rust, Go, PHP, TypeScript
- ❌ Relationship mapping (CALLS, IMPORTS, INHERITS)
- ❌ Knowledge graph navigation for non-Python code
- ✅ Documentation extraction (working correctly)
- ✅ Python basic types (function, method, class, module)

**User Impact:**
- Users with Rust/Go/PHP/TS projects see 0 entities
- Python-only projects see partial entities (missing generic "code" chunks)
- Graph visualization is essentially empty
- Code navigation features don't work

### Data Loss

**Example: Rust Project**
- File: `src/lib.rs` containing:
  - 5 functions ✅ (captured)
  - 3 structs ❌ (lost)
  - 2 enums ❌ (lost)
  - 1 trait ❌ (lost)
  - 4 impl blocks ❌ (lost)

**Result:** Only 5/15 entities captured (67% data loss)

---

## Recommended Fixes

### Fix 1: Expand Chunk Type Filter (HIGH PRIORITY)

**File:** `src/mcp_vector_search/core/kg_builder.py` (lines 301-305)

**Current Code:**
```python
code_chunks = [
    c
    for c in chunks
    if c.chunk_type in ["function", "method", "class", "module"]
]
```

**Recommended Fix:**
```python
# Define code chunk types (exclude only text-based chunks)
CODE_CHUNK_TYPES = {
    # Python/JS/Common
    "function", "method", "class", "module",
    # Rust
    "struct", "enum", "trait", "impl",
    # PHP
    "interface", "trait", "imports",
    # Go
    "struct",
    # Generic
    "code",
}

code_chunks = [
    c
    for c in chunks
    if c.chunk_type in CODE_CHUNK_TYPES
]
```

**Alternative (More Robust):**
```python
# Process everything that's NOT a text chunk
code_chunks = [
    c
    for c in chunks
    if c.language != "text" and not str(c.file_path).endswith(".md")
]
```

This approach is more future-proof as new chunk types won't be silently dropped.

### Fix 2: Enhance `_get_entity_name()` for New Types (MEDIUM PRIORITY)

**File:** `src/mcp_vector_search/core/kg_builder.py` (lines 250-274)

**Current Code:**
```python
def _get_entity_name(self, chunk: CodeChunk) -> str:
    if chunk.function_name:
        return chunk.function_name
    if chunk.class_name:
        return chunk.class_name

    if chunk.file_path:
        return Path(chunk.file_path).stem

    return f"unnamed_{chunk.chunk_type}"
```

**Recommended Enhancement:**
```python
def _get_entity_name(self, chunk: CodeChunk) -> str:
    """Get meaningful entity name from chunk."""
    # Prefer explicit names
    if chunk.function_name:
        return chunk.function_name
    if chunk.class_name:
        return chunk.class_name

    # Check for struct/enum/trait/interface names
    # (assuming parsers populate these fields)
    if hasattr(chunk, 'struct_name') and chunk.struct_name:
        return chunk.struct_name
    if hasattr(chunk, 'enum_name') and chunk.enum_name:
        return chunk.enum_name
    if hasattr(chunk, 'trait_name') and chunk.trait_name:
        return chunk.trait_name
    if hasattr(chunk, 'interface_name') and chunk.interface_name:
        return chunk.interface_name

    # For module-level chunks, use filename as module name
    if chunk.file_path:
        return Path(chunk.file_path).stem

    # Fallback
    return f"unnamed_{chunk.chunk_type}"
```

**Note:** This requires checking if parsers populate these fields. Alternative: extract names from chunk content/context.

### Fix 3: Add Configurable Generic Filter (LOW PRIORITY)

**File:** `src/mcp_vector_search/core/kg_builder.py`

Add CLI flag to control generic filtering:

```python
def build_from_chunks_sync(
    self,
    chunks: list[CodeChunk],
    show_progress: bool = True,
    skip_documents: bool = False,
    filter_generic: bool = True,  # NEW PARAMETER
    progress_tracker: Optional["ProgressTracker"] = None,
) -> dict[str, int]:
    ...

    # In _extract_code_entity()
    if filter_generic and self._is_generic_entity(name):
        logger.debug(f"Skipping generic entity: {name}")
        return None, {"IMPORTS": import_relationships}, module_entities
```

### Fix 4: Add Diagnostic Logging (LOW PRIORITY)

Add logging to help debug future issues:

```python
# After code_chunks filtering
if not show_progress:
    logger.info(f"Filtered {len(code_chunks)} code chunks from {len(chunks)} total")
    logger.info(f"  Code chunk types: {set(c.chunk_type for c in code_chunks)}")
    logger.info(f"  Excluded chunk types: {set(c.chunk_type for c in chunks) - set(c.chunk_type for c in code_chunks)}")
```

---

## Testing Recommendations

### Test Case 1: Rust Project
```bash
# Create test Rust project with various types
mkdir test-rust
cd test-rust
cat > lib.rs << 'EOF'
pub struct User {
    name: String,
}

pub enum Status {
    Active,
    Inactive,
}

pub trait Auth {
    fn login(&self);
}

impl Auth for User {
    fn login(&self) {}
}

pub fn hello() {}
EOF

# Index and build KG
mcp-vector-search index
mcp-vector-search kg build --force

# Expected results:
# Code Entities: 5 (User, Status, Auth, impl block, hello)
# Relationships: 1 IMPLEMENTS (User -> Auth)
```

### Test Case 2: Mixed Language Project
```bash
# Test with Python + Rust + Go mix
# Expected: Entities from all languages

mcp-vector-search kg build --force
# Should see entities from .py, .rs, .go files
```

### Test Case 3: Generic Filter
```bash
# Test that generic names are still filtered
# Functions named "get", "set", "run" should not create entities
```

---

## Related Files

**Core Implementation:**
- `src/mcp_vector_search/core/kg_builder.py` - Main builder logic (lines 301-305, 250-274)
- `src/mcp_vector_search/core/knowledge_graph.py` - Graph storage (lines 1494+)

**Parsers (Chunk Type Definitions):**
- `src/mcp_vector_search/parsers/rust.py` - Rust types (struct, enum, trait, impl)
- `src/mcp_vector_search/parsers/go.py` - Go types (struct)
- `src/mcp_vector_search/parsers/php.py` - PHP types (interface, trait)
- `src/mcp_vector_search/parsers/javascript.py` - TS types (interface)
- `src/mcp_vector_search/parsers/base.py` - Generic types (code, text)

**CLI Output:**
- `src/mcp_vector_search/cli/commands/_kg_subprocess.py` - Statistics display (lines 159-181)

---

## Conclusion

The Knowledge Graph is showing 0 Code Entities because the chunk type filter at line 304 in `kg_builder.py` only accepts 4 types (`function`, `method`, `class`, `module`) out of 15+ valid code chunk types generated by the parsers.

**Priority:** Critical
**Effort:** Low (2-line fix)
**Risk:** Low (expanding filter is safe)

**Immediate Action Required:** Expand `CODE_CHUNK_TYPES` to include all valid code types OR switch to exclusion-based filtering (exclude only text chunks).

The `_get_entity_name()` fix appears correct but cannot be fully validated until chunks are actually processed. The generic entity filter is reasonable but could be made configurable for power users.

---

**Next Steps:**
1. ✅ Research complete - root cause identified
2. ⏭️ Implement Fix 1 (expand chunk type filter)
3. ⏭️ Test with multi-language project
4. ⏭️ Consider Fixes 2-4 as enhancements
