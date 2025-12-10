# Search Result Filtering for Boilerplate Names

**Research Date**: 2025-12-10
**Status**: Complete
**Purpose**: Identify where to add language-specific exclusions for common boilerplate names (like "main", "__init__") in search results

## Executive Summary

The current search implementation returns semantically similar results but includes common boilerplate names that clutter results. This research identifies:

1. **Where filtering should be applied**: Post-search in `_rerank_results()` method
2. **What fields are available for filtering**: `function_name`, `class_name`, `chunk_type`, `language`
3. **Supported languages**: Python, JavaScript, TypeScript, Dart, PHP, Ruby, HTML, Text
4. **Recommended approach**: Multi-layered filtering with configurable exclusion lists

## Key Findings

### Search Architecture Flow

```
User Query
    ↓
SemanticSearchEngine.search() [search.py:109]
    ↓
_preprocess_query() [search.py:287] - Query expansion
    ↓
VectorDatabase.search() [database.py:325] - Vector search with filters
    ↓
_enhance_result() [search.py:437] - Add context lines
    ↓
_rerank_results() [search.py:483] ← **BEST PLACE TO ADD FILTERING**
    ↓
Return filtered & ranked results
```

### Critical Files

#### 1. `/src/mcp_vector_search/core/search.py`
**Purpose**: Main search orchestration and result ranking
**Key Methods**:
- `search()` (line 109): Main search entry point
- `_rerank_results()` (line 483): **RECOMMENDED FILTER LOCATION** - applies boosting/penalties
- `_preprocess_query()` (line 287): Query expansion

**Current Boosting/Penalty Logic** (lines 59-69):
```python
_BOOST_EXACT_IDENTIFIER = 0.15      # Exact match in function/class name
_BOOST_PARTIAL_IDENTIFIER = 0.05    # Partial word match
_BOOST_FILE_NAME_EXACT = 0.08       # File name exact match
_BOOST_FILE_NAME_PARTIAL = 0.03     # File name partial match
_BOOST_FUNCTION_CHUNK = 0.05        # Function chunk type
_BOOST_CLASS_CHUNK = 0.03           # Class chunk type
_BOOST_SOURCE_FILE = 0.02           # Source file extension
_BOOST_SHALLOW_PATH = 0.02          # Short path depth
_PENALTY_TEST_FILE = -0.02          # Test file penalty
_PENALTY_DEEP_PATH = -0.01          # Deep path penalty
```

#### 2. `/src/mcp_vector_search/core/models.py`
**Purpose**: Data models for search results
**Key Models**:
- `SearchResult` (line 132): Available fields for filtering
  - `function_name: str | None` (line 143)
  - `class_name: str | None` (line 146)
  - `chunk_type: str` (line 142) - "code", "function", "class", "comment", "docstring"
  - `language: str` (line 139)
  - `similarity_score: float` (line 140)
  - `file_path: Path` (line 136)

- `CodeChunk` (line 10): Source data model
  - All SearchResult fields plus:
  - `decorators: list[str]` (line 35)
  - `parameters: list[dict]` (line 36)
  - `return_type: str | None` (line 37)

#### 3. `/src/mcp_vector_search/core/database.py`
**Purpose**: Vector database interface and ChromaDB implementation
**Key Methods**:
- `search()` (line 325): Performs vector search with pre-filtering
- `_build_where_clause()` (line 602 & 1114): Builds filter conditions for ChromaDB

**Current Filter Support**:
```python
# Simple filters (line 602-614)
where = {}
for key, value in filters.items():
    if isinstance(value, list):
        where[key] = {"$in": value}      # Multiple values
    elif isinstance(value, str) and value.startswith("!"):
        where[key] = {"$ne": value[1:]}  # Negation
    else:
        where[key] = value               # Exact match
```

#### 4. `/src/mcp_vector_search/parsers/registry.py`
**Purpose**: Language parser registration
**Supported Languages** (line 33-65):
1. **Python**: `.py`, `.pyw`
2. **JavaScript**: `.js`, `.jsx`, `.mjs`
3. **TypeScript**: `.ts`, `.tsx`
4. **Dart**: `.dart`
5. **PHP**: `.php`, `.phtml`
6. **Ruby**: `.rb`, `.rake`, `.gemspec`
7. **HTML**: `.html`
8. **Text**: `.txt`
9. **Fallback**: All other extensions

## Language-Specific Boilerplate Patterns

### Python
**Common Boilerplate Functions**:
- `__init__` - Constructor
- `__str__` - String representation
- `__repr__` - Object representation
- `__eq__` - Equality comparison
- `__len__` - Length
- `__getitem__` - Index access
- `main` - Entry point
- `setUp` - Test setup
- `tearDown` - Test teardown

**Common Boilerplate Classes**:
- `TestCase` - Base test class (but subclasses are meaningful)

### JavaScript/TypeScript
**Common Boilerplate Functions**:
- `constructor` - Class constructor
- `render` - React render (common but meaningful)
- `componentDidMount` - React lifecycle
- `componentWillUnmount` - React lifecycle
- `main` - Entry point
- `index` - Default export

**Common Boilerplate Properties**:
- `get` - Generic getter
- `set` - Generic setter

### Dart
**Common Boilerplate Functions**:
- `build` - Widget build (Flutter - very common)
- `dispose` - Cleanup
- `initState` - State initialization
- `main` - Entry point

### PHP
**Common Boilerplate Functions**:
- `__construct` - Constructor
- `__destruct` - Destructor
- `__toString` - String conversion
- `index` - Controller action
- `main` - Entry point

### Ruby
**Common Boilerplate Functions**:
- `initialize` - Constructor
- `to_s` - String conversion
- `to_h` - Hash conversion
- `main` - Entry point
- `setup` - Test setup
- `teardown` - Test teardown

## Recommended Implementation Approach

### Option 1: Post-Search Filtering (RECOMMENDED)

**Where**: In `SemanticSearchEngine._rerank_results()` after line 572

**Advantages**:
- No database re-indexing required
- Easy to configure and update
- Can be user-controlled via flags
- Preserves vector search accuracy
- Can apply smart context-aware filtering

**Implementation Strategy**:

```python
# Add to search.py after line 69 (with other constants)
_BOILERPLATE_EXCLUSIONS = {
    "python": {
        "functions": {
            "__init__", "__str__", "__repr__", "__eq__", "__len__",
            "__getitem__", "__setitem__", "__delitem__", "__iter__",
            "setUp", "tearDown", "test_main"
        },
        "partial_functions": {"main"},  # Only exclude if standalone
        "classes": set(),  # Usually don't exclude classes
    },
    "javascript": {
        "functions": {
            "constructor", "componentDidMount", "componentWillUnmount",
            "componentDidUpdate", "shouldComponentUpdate"
        },
        "partial_functions": {"main", "index", "render"},
        "classes": set(),
    },
    "typescript": {
        "functions": {
            "constructor", "componentDidMount", "componentWillUnmount",
            "componentDidUpdate", "shouldComponentUpdate"
        },
        "partial_functions": {"main", "index", "render"},
        "classes": set(),
    },
    "dart": {
        "functions": {"build", "dispose", "initState"},
        "partial_functions": {"main"},
        "classes": set(),
    },
    "php": {
        "functions": {
            "__construct", "__destruct", "__toString", "__call",
            "__get", "__set"
        },
        "partial_functions": {"index", "main"},
        "classes": set(),
    },
    "ruby": {
        "functions": {
            "initialize", "to_s", "to_h", "to_a",
            "setup", "teardown"
        },
        "partial_functions": {"main"},
        "classes": set(),
    },
}

def _is_boilerplate(self, result: SearchResult, query: str) -> bool:
    """Check if result is common boilerplate.

    Args:
        result: Search result to check
        query: Original search query (to avoid over-filtering)

    Returns:
        True if result should be filtered out
    """
    # Don't filter if user explicitly searched for this name
    query_lower = query.lower()
    if result.function_name and result.function_name.lower() in query_lower:
        return False
    if result.class_name and result.class_name.lower() in query_lower:
        return False

    # Get exclusion set for language
    exclusions = self._BOILERPLATE_EXCLUSIONS.get(result.language, {})

    # Check function name
    if result.function_name:
        # Exact match exclusions
        if result.function_name in exclusions.get("functions", set()):
            return True
        # Partial match exclusions (only if chunk is just this function)
        if result.function_name in exclusions.get("partial_functions", set()):
            # Only exclude if this is a standalone function, not part of larger context
            if result.chunk_type == "function" and result.line_count < 10:
                return True

    # Check class name (usually more permissive)
    if result.class_name:
        if result.class_name in exclusions.get("classes", set()):
            return True

    return False

# In _rerank_results(), after line 572:
# Filter out boilerplate (optional, controlled by parameter)
if filter_boilerplate:  # New parameter
    results = [r for r in results if not self._is_boilerplate(r, query)]
```

### Option 2: Pre-Search Database Filtering

**Where**: In `ChromaVectorDatabase._build_where_clause()` (line 602 or 1114)

**Advantages**:
- Reduces vector search workload
- More efficient for large codebases

**Disadvantages**:
- Harder to make context-aware
- Requires database metadata updates
- Less flexible for users

### Option 3: Hybrid Approach (BEST)

**Strategy**:
1. **Pre-filter** (Database level): Remove truly noise chunks (e.g., empty `__init__`)
2. **Post-filter** (Search level): Context-aware filtering with user control
3. **Ranking adjustment**: Penalize boilerplate instead of removing

**Implementation**:

```python
# In _rerank_results(), add boilerplate penalty
_PENALTY_BOILERPLATE = -0.10  # Significant penalty

# Apply penalty instead of filtering
if self._is_boilerplate(result, query):
    score += self._PENALTY_BOILERPLATE
```

## CLI Integration

### Proposed Search Flags

Add to `/src/mcp_vector_search/cli/commands/search.py`:

```python
filter_boilerplate: bool = typer.Option(
    True,  # Default: filter common boilerplate
    "--filter-boilerplate/--no-filter-boilerplate",
    help="Filter out common boilerplate names (main, __init__, etc.)",
    rich_help_panel="🔍 Filters",
)

boilerplate_penalty: float = typer.Option(
    0.10,
    "--boilerplate-penalty",
    help="Penalty for boilerplate matches (0.0-1.0, 0=no penalty)",
    min=0.0,
    max=1.0,
    rich_help_panel="🎯 Search Options",
)
```

### Usage Examples

```bash
# Default: Filter boilerplate
mcp-vector-search search "authentication"

# Include boilerplate
mcp-vector-search search "authentication" --no-filter-boilerplate

# Penalize instead of filter
mcp-vector-search search "authentication" --boilerplate-penalty 0.15

# Explicitly search for boilerplate (auto-detected)
mcp-vector-search search "__init__ method"  # Won't filter __init__
```

## Configuration File Support

### Proposed Configuration Schema

Add to project configuration (`.mcp-vector-search/config.json`):

```json
{
  "search": {
    "filter_boilerplate": true,
    "boilerplate_penalty": 0.10,
    "custom_exclusions": {
      "python": {
        "functions": ["custom_boilerplate_func"],
        "classes": []
      }
    }
  }
}
```

## Testing Considerations

### Test Cases Needed

1. **Exact name filtering**: Search "main" should NOT filter when explicitly searched
2. **Context-aware filtering**: "main in authentication.py" vs standalone "main()"
3. **Language-specific filtering**: Python `__init__` vs JavaScript `constructor`
4. **User override**: `--no-filter-boilerplate` flag works
5. **Penalty vs filtering**: Compare result quality with both approaches

### Test Files

- `/tests/unit/core/test_search.py` - Add boilerplate filtering tests
- `/tests/integration/test_search_filtering.py` - End-to-end filtering tests

## Performance Impact

**Estimated Impact**: Minimal to none

- **Post-search filtering**: O(n) where n = number of results (typically 10-100)
- **Set lookups**: O(1) for each function/class name check
- **Total overhead**: <1ms for typical search results

## Migration Path

### Phase 1: Add Penalty-Based Ranking (Low Risk)
1. Add `_BOILERPLATE_EXCLUSIONS` constant
2. Add `_is_boilerplate()` method
3. Apply penalty in `_rerank_results()`
4. Add tests
5. Monitor impact on search quality

### Phase 2: Add User Controls (Medium Risk)
1. Add CLI flags (`--filter-boilerplate`, `--boilerplate-penalty`)
2. Add configuration file support
3. Add custom exclusion lists
4. Update documentation

### Phase 3: Optimize with Pre-filtering (Optional)
1. Add database-level metadata flags
2. Implement pre-filtering in `_build_where_clause()`
3. Benchmark performance improvements

## Open Questions

1. **Should we filter or penalize?**
   - Recommendation: Start with penalty, add filtering as opt-in

2. **What about context?**
   - `main()` in `auth.py` might be meaningful
   - Recommendation: Only filter standalone boilerplate chunks (<10 lines)

3. **User customization?**
   - Allow users to add custom exclusions
   - Recommendation: Support via config file in Phase 2

4. **Framework-specific boilerplate?**
   - React lifecycle methods, Django views, etc.
   - Recommendation: Start with language-agnostic, expand later

## Next Steps

1. **Immediate**: Implement Option 3 (Hybrid) with penalty-based ranking
2. **Short-term**: Add CLI flags and basic configuration support
3. **Medium-term**: Gather user feedback and adjust exclusion lists
4. **Long-term**: Add framework-specific exclusions (React, Django, etc.)

## References

- Search implementation: `/src/mcp_vector_search/core/search.py`
- Data models: `/src/mcp_vector_search/core/models.py`
- Database layer: `/src/mcp_vector_search/core/database.py`
- Parser registry: `/src/mcp_vector_search/parsers/registry.py`
- CLI search: `/src/mcp_vector_search/cli/commands/search.py`

## Appendix: Complete Exclusion Reference

### Python Dunder Methods (Complete List)

**Object Lifecycle**:
- `__new__`, `__init__`, `__del__`

**String Representation**:
- `__str__`, `__repr__`, `__format__`, `__bytes__`

**Comparison**:
- `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`, `__hash__`

**Attribute Access**:
- `__getattr__`, `__setattr__`, `__delattr__`, `__dir__`

**Callable**:
- `__call__`

**Container**:
- `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`
- `__iter__`, `__reversed__`, `__next__`

**Context Manager**:
- `__enter__`, `__exit__`

**Descriptor**:
- `__get__`, `__set__`, `__delete__`, `__set_name__`

**Numeric**:
- `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`
- `__pow__`, `__and__`, `__or__`, `__xor__`, `__lshift__`, `__rshift__`
- And their `__r*__` and `__i*__` variants

**Recommendation**: Only filter the most common (lifecycle, string, comparison)

---

**End of Research Document**
