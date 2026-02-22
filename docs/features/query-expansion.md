# Query Expansion with Code Synonym Dictionary

## Overview

Query expansion improves semantic search recall by automatically substituting code-specific synonyms in search queries. When enabled (default), the system generates multiple query variants and merges results to find the most relevant code.

## How It Works

### Expansion Strategy

1. **Tokenize query** — Split query into words (respecting word boundaries)
2. **Match synonyms** — Find tokens that match built-in or custom synonym keys
3. **Generate variants** — Create variants by substituting ONE synonym per variant
4. **Search all variants** — Run vector search for each variant
5. **Merge results** — Deduplicate by chunk_id, keep highest similarity score

### Example

```python
# Query: "auth middleware"
# Expands to:
[
    "auth middleware",           # Original
    "authentication middleware", # Synonym for 'auth'
    "authorize middleware",
    "login middleware",
    "session middleware",
    "credential middleware",
    "oauth middleware",
    "auth interceptor",          # Synonym for 'middleware'
    "auth filter",
    "auth hook",
    "auth plugin",
    "auth handler",
]
```

## Built-in Synonyms

The system includes 25 built-in synonym groups covering common code concepts:

| Key | Synonyms |
|-----|----------|
| `auth` | authentication, authorize, login, session, credential, oauth |
| `delete` | remove, destroy, drop, purge, clean, erase |
| `create` | new, init, initialize, build, make, generate, construct |
| `error` | exception, fault, failure, crash, bug, issue |
| `config` | configuration, settings, options, preferences, env |
| `db` | database, storage, persistence, datastore, repo, repository |
| `api` | endpoint, route, handler, controller, resource |
| `test` | spec, assertion, verify, validate, check, expect |
| `async` | concurrent, parallel, await, coroutine, future, promise |
| `cache` | memoize, store, buffer, preload |
| `parse` | extract, tokenize, analyze, process, decode |
| `render` | display, draw, paint, show, present, view |
| `fetch` | get, retrieve, load, download, pull, request |
| `send` | post, push, emit, dispatch, publish, transmit |
| `log` | trace, debug, print, output, record, audit |
| `user` | account, profile, member, identity, principal |
| `file` | document, path, stream, blob, resource |
| `search` | find, query, lookup, filter, match, grep |
| `update` | modify, patch, change, edit, mutate, alter |
| `serialize` | encode, marshal, dump, stringify, format |
| `deserialize` | decode, unmarshal, load, parse |
| `validate` | check, verify, sanitize, assert, ensure |
| `transform` | convert, map, translate, adapt, morph |
| `middleware` | interceptor, filter, hook, plugin, handler |
| `deploy` | release, publish, ship, rollout, launch |

**Total**: 25 groups, 137 synonyms (average 5.48 per group)

## Custom Synonyms

You can extend the built-in synonyms by creating a custom synonyms file:

**Location**: `.mcp-vector-search/synonyms.json`

**Format**:
```json
{
  "myterm": ["synonym1", "synonym2", "synonym3"],
  "db": ["additional_db_term"],
  "custom": ["variant1", "variant2"]
}
```

**Behavior**:
- Custom synonyms are **merged** with built-in synonyms
- If a key already exists, custom synonyms are **appended** to the list
- Custom synonyms support the same bidirectional expansion as built-in ones

## CLI Usage

### Enable Expansion (Default)

```bash
# Expansion enabled by default
mcp-vector-search search "auth middleware"

# Explicitly enable
mcp-vector-search search "auth middleware" --expand
mcp-vector-search search "auth middleware" -E
```

### Disable Expansion

```bash
# Disable expansion
mcp-vector-search search "auth middleware" --no-expand
mcp-vector-search search "auth middleware" -N
```

### Examples

```bash
# Search with expansion (finds more results)
mcp-vector-search search "delete user" --expand

# Search without expansion (original query only)
mcp-vector-search search "delete user" --no-expand

# Combine with other filters
mcp-vector-search search "auth" --expand --language python --limit 10
```

## MCP Integration

### Tool Schema Parameter

The `search_code` MCP tool includes an `expand` boolean parameter:

```json
{
  "query": "auth middleware",
  "limit": 10,
  "expand": true
}
```

**Default**: `true` (expansion enabled)

### Example MCP Call

```json
{
  "tool": "search_code",
  "arguments": {
    "query": "authentication middleware",
    "limit": 10,
    "expand": true
  }
}
```

## Python API Usage

```python
from pathlib import Path
from mcp_vector_search.core.search import SemanticSearchEngine
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.factory import create_database

# Create search engine
embedding_function, _ = create_embedding_function("sentence-transformers/all-MiniLM-L6-v2")
database = create_database(
    persist_directory=Path(".mcp-vector-search/lance"),
    embedding_function=embedding_function,
)
search_engine = SemanticSearchEngine(
    database=database,
    project_root=Path("."),
    enable_query_expansion=True,  # Enable expansion (default)
)

# Search with expansion (default)
results_with_expansion = await search_engine.search(
    query="auth middleware",
    limit=10,
    expand=True,  # Generates multiple variants
)

# Search without expansion
results_without_expansion = await search_engine.search(
    query="auth middleware",
    limit=10,
    expand=False,  # Uses original query only
)
```

## Performance Impact

### Pros
- **Improved recall**: Finds more relevant results by matching synonyms
- **Better coverage**: Catches different terminology for the same concept
- **Bidirectional**: Expands both ways (auth → authentication, authentication → auth)

### Cons
- **More API calls**: Generates N variants = N vector searches
- **Slightly slower**: Proportional to number of variants (typically 5-15 variants)
- **Potential noise**: May match less relevant synonyms in some cases

### Optimization
- Variants are searched in parallel (async)
- Results are deduplicated by chunk_id
- Only the **best score** per chunk is kept

## Implementation Details

### Query Expander Module

**File**: `src/mcp_vector_search/core/query_expander.py`

**Key Classes**:
- `QueryExpander`: Main class for query expansion
  - `expand(query: str) -> list[str]`: Generates query variants
  - `get_synonyms(key: str) -> list[str]`: Retrieves synonyms for a key
  - `get_stats() -> dict`: Returns expander statistics

**Key Features**:
- Token-based expansion (respects word boundaries)
- Bidirectional mapping (key ↔ synonym)
- Custom synonyms loaded from `.mcp-vector-search/synonyms.json`
- One substitution per variant (prevents combinatorial explosion)

### Integration Points

1. **SemanticSearchEngine** (`src/mcp_vector_search/core/search.py`)
   - Initialized with `QueryExpander` instance
   - Expands query before search routing
   - Merges results from all variants

2. **CLI** (`src/mcp_vector_search/cli/commands/search.py`)
   - `--expand/--no-expand` flags (-E/-N short flags)
   - Passes `expand` parameter to search engine

3. **MCP Tool** (`src/mcp_vector_search/mcp/tool_schemas.py`)
   - `expand` boolean parameter in `search_code` schema
   - Default: `true`

4. **MCP Handler** (`src/mcp_vector_search/mcp/search_handlers.py`)
   - Extracts `expand` parameter from tool call
   - Passes to search engine

## Testing

### Unit Tests

```bash
# Test query expander directly
uv run python -c "
from mcp_vector_search.core.query_expander import QueryExpander

qe = QueryExpander()
print(qe.expand('auth middleware'))
print(qe.expand('delete user from database'))
print(qe.get_stats())
"
```

### Integration Tests

```bash
# Compare results with/without expansion
uv run mcp-vector-search search "auth middleware" --expand --limit 5
uv run mcp-vector-search search "auth middleware" --no-expand --limit 5
```

### Expected Behavior

- **With expansion**: More results, slightly higher scores, more variants matched
- **Without expansion**: Fewer results, original query only, faster execution

## Future Enhancements

1. **Context-aware expansion**
   - Use project-specific terminology from codebase
   - Learn synonyms from commit messages and documentation

2. **Adaptive expansion**
   - Adjust expansion aggressiveness based on result quality
   - Disable expansion if initial results are already high quality

3. **Domain-specific dictionaries**
   - Web frameworks (React, FastAPI, Django)
   - Testing frameworks (pytest, Jest, JUnit)
   - Database systems (SQL, NoSQL, ORM)

4. **Synonym scoring**
   - Weight synonyms by relevance
   - Boost common synonyms over rare ones

5. **Query analysis**
   - Detect query intent (search vs. filter vs. navigation)
   - Adjust expansion strategy based on intent

## Related Features

- **Hybrid Search**: Combines vector and BM25 keyword search
- **MMR Diversity**: Reranks results for diversity
- **Knowledge Graph**: Enriches results with code relationships
- **Quality Ranking**: Combines relevance and code quality scores

## References

- Issue #104: Query expansion with code synonym dictionary
- `src/mcp_vector_search/core/query_expander.py`: Implementation
- `docs/research/code-embedding-models-research-2026-02-20.md`: Embedding model comparison

---

**Generated**: 2026-02-21
**Author**: Claude Code (Opus 4.6)
**Status**: ✅ Implemented
