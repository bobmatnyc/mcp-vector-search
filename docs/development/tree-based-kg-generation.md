# Tree-Based KG Generation Implementation

**Date**: 2026-02-16
**Status**: ✅ Complete

## Problem Statement

The knowledge graph builder was re-parsing already-parsed AST data with fragile string parsers:

- **Root Cause**: `_parse_module_name()` expected full import statements like "import os" but received just "os" from AST dict
- **Impact**: IMPORTS relationship count = 0 (broken)
- **User Mandate**: "we shouldn't be using manual parsing to build KG entities and relationships"

## Solution

Replaced manual string parsing with direct AST consumption.

### Changes Made

#### 1. Fixed IMPORTS Extraction (kg_builder.py)

**Before (fragile string parsing)**:
```python
import_str = imp.get("module", "") if isinstance(imp, dict) else str(imp)
module_name = self._parse_module_name(import_str)  # Returns None!
```

**After (direct AST consumption)**:
```python
# AST provides: {"module": "os", "names": ["path"], "alias": None}
if isinstance(imp, dict):
    module_name = imp.get("module")
else:
    # Fallback for legacy string format (won't happen with tree-sitter)
    module_name = str(imp).split(".")[0] if imp else None

# Skip relative imports (., .., etc.) and empty modules
if module_name and not module_name.startswith("."):
    # Create relationship...
```

**Locations Fixed**:
- Line 846-873: `_extract_code_entity()` method
- Line 1006-1034: `_process_chunk_kg()` method

#### 2. Deleted Dead Code

Removed `_parse_module_name()` method (lines 826-857) - fragile string parser no longer needed.

#### 3. Verified Other Relationships

Confirmed CALLS and INHERITS already consume AST directly:
- `chunk.calls`: list[str] from tree-sitter `call_expression` nodes
- `chunk.inherits_from`: list[str] from tree-sitter `argument_list` nodes

## Results

### Before
```
✗ IMPORTS: 0       (broken - manual parser failed)
✓ CALLS: 4,250     (working - using AST)
✓ INHERITS: 56     (working - using AST)
```

### After
```
✓ IMPORTS: 3,395   (fixed - direct AST consumption)
✓ CALLS: 4,250     (still working)
✓ INHERITS: 56     (still working)
✓ FOLLOWS: 5,206   (already implemented!)
```

## Architecture Principles

### Tree-Based Extraction Pattern

All KG relationships now follow the tree-based pattern:

1. **AST provides structured data** (dicts, lists) - no string parsing needed
2. **Direct field access** - `imp.get("module")` instead of regex parsing
3. **Fallback for legacy** - handle old string format gracefully
4. **Skip invalid data** - relative imports, empty values

### AST Data Formats

**Python Imports** (`chunk.imports`):
```python
[
    {"module": "os", "names": ["path", "environ"], "alias": None},
    {"module": "typing", "names": ["List", "Dict"], "alias": None},
    {"module": "numpy", "names": ["*"], "alias": "np"}
]
```

**Function Calls** (`chunk.calls`):
```python
["print", "len", "self.save", "obj.method"]
```

**Inheritance** (`chunk.inherits_from`):
```python
["BaseClass", "MixinA", "MixinB"]
```

## LOC Delta

```
LOC Delta:
- Added: 12 lines (comments explaining AST structure)
- Removed: 32 lines (_parse_module_name method)
- Net Change: -20 lines
- Phase: Cleanup (eliminate manual parsing)
```

## Testing

```bash
# Rebuild KG with force flag
mcp-vector-search kg build --force --verbose

# Verify IMPORTS > 0
# Output: ✓ 3,395 imports
```

## Benefits

1. **Reliability**: No more fragile regex parsing - AST is always correct
2. **Maintainability**: Less code (20 lines removed)
3. **Clarity**: Direct field access is easier to understand
4. **Performance**: No string parsing overhead
5. **Consistency**: All relationships use same tree-based pattern

## Future Work

### Phase 2: FOLLOWS Relationship (Already Done!)

The FOLLOWS relationship is already implemented with 5,206 relationships extracted.

### Phase 3: Text Tree Integration

Use `markdown-it-py` for text documents:

```python
from markdown_it import MarkdownIt

md = MarkdownIt()
tokens = md.parse(content)
# Walk tokens to extract sections with hierarchy
```

This would replace any manual paragraph/heading parsing in `text.py`.

## Related Documentation

- **AST Structure**: `src/mcp_vector_search/parsers/python_helpers/metadata_extractor.py`
- **Tree-Sitter Parser**: `src/mcp_vector_search/parsers/python.py`
- **KG Schema**: `src/mcp_vector_search/core/knowledge_graph.py`
