# AttentionInterface Error Investigation

**Date:** 2026-02-06
**Error:** `'AttentionInterface' object has no attribute 'get_interface'`
**Tool:** `search_code` failing in MCP server

## Error Analysis

### What We Know

1. **Error Location**: Occurs when calling `search_code` MCP tool
2. **Error Message**: `AttributeError: 'AttentionInterface' object has no attribute 'get_interface'`
3. **Component**: Related to transformers/sentence-transformers embedding model loading

### Code Investigation

#### AttentionInterface Class Structure

**File:** `.venv-mcp/lib/python3.13/site-packages/transformers/modeling_utils.py:6250`

```python
class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions.
    """
    _global_mapping = {
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "paged_attention": paged_attention_forward,
        "sdpa": sdpa_attention_forward,
        "sdpa_paged": sdpa_attention_paged_forward,
        "eager_paged": eager_paged_attention_forward,
    }
```

#### GeneralInterface Class

**File:** `.venv-mcp/lib/python3.13/site-packages/transformers/utils/generic.py:1094`

```python
class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one.
    """
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())
```

**KEY FINDING**: `GeneralInterface` does NOT have a `get_interface()` method.

### Available Methods

The `GeneralInterface` class (parent of `AttentionInterface`) provides:
- `__getitem__(key)` - Dictionary-style access
- `__setitem__(key, value)` - Dictionary-style assignment
- `__delitem__(key)` - Dictionary-style deletion
- `__iter__()` - Iteration support
- `__len__()` - Length support
- `register(key, value)` - Class method to register globally
- `valid_keys()` - Returns list of available keys

**Missing**: `get_interface()` method

### Package Versions

- **transformers**: 4.56.2
- **sentence-transformers**: 5.1.2

### Root Cause Analysis

The error suggests that somewhere in the code (likely in transformers or sentence-transformers), there's a call to:

```python
attention_interface.get_interface()
```

But this method doesn't exist on `AttentionInterface` or its parent `GeneralInterface`.

#### Possible Causes:

1. **Version Mismatch**: The code calling `get_interface()` expects a different version of transformers
2. **API Change**: Transformers may have changed the API between versions
3. **Missing Implementation**: The method should exist but wasn't implemented
4. **Wrong Object Type**: Code is receiving an `AttentionInterface` when it expects something else

### Where the Call Might Be

Based on the error occurring during `search_code`, the call chain is likely:

```
search_code (MCP handler)
  → search_engine.search()
    → database.query()
      → embedding_function()
        → SentenceTransformerEmbeddingFunction
          → model.encode()
            → [transformers internals]
              → attention_interface.get_interface()  ❌ ERROR HERE
```

### Next Steps for Resolution

1. **Check for recent transformers/sentence-transformers updates**
   - Version 4.56.2 released: September 2024
   - Version 5.1.2 released: January 2025

2. **Search for get_interface calls in dependencies**
   - Need to grep through transformers source for `get_interface` usage
   - Check if method was renamed or removed in recent versions

3. **Potential Fixes**:
   - Downgrade transformers to earlier version where `get_interface` existed
   - Upgrade to newer version where issue is fixed
   - Patch `GeneralInterface` to add missing `get_interface` method
   - Use alternative embedding model that doesn't trigger this code path

4. **Workaround**:
   - Switch to ChromaDB's built-in embedding function (already done in `create_embedding_function`)
   - Use different model that doesn't hit this code path

### File Locations

- **AttentionInterface**: `.venv-mcp/lib/python3.13/site-packages/transformers/modeling_utils.py:6250`
- **GeneralInterface**: `.venv-mcp/lib/python3.13/site-packages/transformers/utils/generic.py:1094`
- **Embedding Function**: `src/mcp_vector_search/core/embeddings.py:662`
- **Search Engine**: `src/mcp_vector_search/core/search.py:107`
- **MCP Handler**: `src/mcp_vector_search/mcp/search_handlers.py:99`

## Recommendations

### Immediate Action

1. **Check actual traceback**: Get full Python traceback to identify exact call site
2. **Test with different model**: Try `all-MiniLM-L6-v2` or `paraphrase-MiniLM-L6-v2` to see if issue is model-specific

### Long-term Solution

1. **Pin transformers version**: Add explicit version constraint if older version works
2. **File upstream bug**: If this is a regression in transformers, report to HuggingFace
3. **Add error handling**: Wrap embedding calls with better error messages

## Status

**Investigation Status:** In Progress
**Root Cause:** Method `get_interface()` called on `AttentionInterface` but method doesn't exist
**Location:** Somewhere in transformers/sentence-transformers code path during embedding generation
**Next Step:** Get full traceback from actual error to identify exact call site
