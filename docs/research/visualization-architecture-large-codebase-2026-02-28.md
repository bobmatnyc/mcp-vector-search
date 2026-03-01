# Visualization Architecture Analysis: Large Codebase Optimization

**Date:** 2026-02-28
**Scope:** Complete audit of the D3.js visualization system for scalability issues with large codebases (Duetto: 17K files, 114K chunks)

---

## 1. Architecture Map

### File Locations

| Component | File |
|-----------|------|
| CLI orchestration | `src/mcp_vector_search/cli/commands/visualize/cli.py` |
| Graph data builder | `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` |
| FastAPI HTTP server | `src/mcp_vector_search/cli/commands/visualize/server.py` |
| HTML template shell | `src/mcp_vector_search/cli/commands/visualize/templates/base.py` |
| All JavaScript (D3, frontend logic) | `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (~7000 lines) |
| JSON exporter | `src/mcp_vector_search/cli/commands/visualize/exporters/json_exporter.py` |
| Persistent graph file | `.mcp-vector-search/visualization/chunk-graph.json` |

### Data Flow

```
mcp-vector-search visualize serve
    │
    ├─ cli.py: Reads all chunks from LanceDB via get_all_chunks()
    ├─ graph_builder.build_graph_data(): builds full graph dict with ALL content
    ├─ json_exporter.export_to_json(): writes chunk-graph.json (OPT_INDENT_2)
    └─ server.start_visualization_server():
           ┌─ FastAPI /api/graph-status   → checks file exists + size
           ├─ FastAPI /api/graph-initial  → reads full file, filters to depth 0-2 dirs
           ├─ FastAPI /api/graph-expand/{id} → reads full file, extracts children
           ├─ FastAPI /api/graph-code-chunks → reads full file, filters code types
           ├─ FastAPI /api/relationships/{id} → reads full file, scans all nodes (O(n²))
           ├─ FastAPI /api/callers/{id}   → reads full file, scans all nodes (O(n²))
           └─ FastAPI /api/graph          → legacy, reads + returns full file
```

### API Endpoints (complete list)

| Endpoint | Purpose | Caching |
|----------|---------|---------|
| `GET /api/graph-status` | File exists + size check | no-cache |
| `GET /api/graph` | Legacy full dump | no-cache |
| `GET /api/graph-initial` | Top-level dirs (depth 0-2), ~100-200 nodes | no-cache |
| `GET /api/graph-expand/{node_id}` | Children of one node | max-age=300 |
| `GET /api/graph-code-chunks` | All code+structural nodes for treemap/sunburst | max-age=300 |
| `GET /api/kg-graph` | Knowledge graph full dump | no-cache |
| `GET /api/kg-expand/{node_id}` | KG node neighbors | none |
| `GET /api/relationships/{chunk_id}` | Semantic + caller relationships | max-age=300 |
| `GET /api/callers/{chunk_id}` | Callers of a function (AST scan) | max-age=300 |
| `GET /api/chunks` | Chunks for a file_id | no-cache |
| `GET /api/graph-data` | Streaming version of full file | no-cache |

---

## 2. Issue 1: API Mismatch — RESOLVED

The original bug report said the frontend expected `/api/graph-initial` and `/api/graph-expand/{node}` but the backend only had `/api/graph`.

**Current status: Both endpoints exist.** The backend has all three:
- `/api/graph` (legacy full-dump, labeled as such in comments)
- `/api/graph-initial` (lines 157-275 of server.py)
- `/api/graph-expand/{node_id}` (lines 277-379 of server.py)

The frontend calls the correct endpoints:
```javascript
// scripts.py line 236
const response = await fetch('/api/graph-initial');

// scripts.py line 293-294
const url = `/api/graph-expand/${encodeURIComponent(nodeId)}`;
const response = await fetch(url);
```

This issue appears to have been fixed already. No action needed here.

---

## 3. Issue 2: Massive JSON Size — Critical and Unresolved

### Measured Data (mcp-vector-search itself, 36K nodes)

```
File: chunk-graph.json
Total size:              27.6 MB (29,015,805 bytes)
Total nodes:             36,928
Total links:             36,216

Node type distribution:
  text:      30,960 nodes (83.8%) — markdown paragraphs from .md files
  function:   4,180 nodes (11.3%)
  file:         705 nodes  (1.9%)
  class:        595 nodes  (1.6%)
  imports:      370 nodes  (1.0%)
  directory:     70 nodes  (0.2%)
  method:        30 nodes  (0.1%)

Size breakdown:
  content + docstring fields:  9.99 MB  (42%)
  metadata/structure fields:  10.75 MB  (45%)
  links:                       2.94 MB  (12%)
```

### Fields Per Node Type

**Chunk nodes** (function/class/method/text/imports/module) include ALL of:
```json
{
  "id": "7a1999a3...",          // 16 bytes
  "name": "my_function",        // variable
  "type": "function",           // 8 bytes
  "file_path": "/abs/path/...", // 48 bytes avg
  "start_line": 42,
  "end_line": 95,
  "complexity": 3.4,
  "parent_id": "file_a0af589d",
  "depth": 2,
  "content": "def my_function...",  // <-- THE BLOAT, avg 1800 bytes for functions
  "docstring": "...",               // <-- ADDITIONAL BLOAT
  "language": "python",
  "quality_score": 0.85,           // only function/class/method
  "smell_count": 1,
  "smells": ["Long Method"],
  "complexity_grade": "A",
  "lines_of_code": 53
}
```

**Location in graph_builder.py (line 549-562):**
```python
node = {
    "id": chunk_id,
    "name": chunk_name,
    "type": chunk.chunk_type,
    "file_path": file_path_str,
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    "complexity": chunk.complexity_score,
    "parent_id": parent_id,
    "depth": chunk.chunk_depth,
    "content": chunk.content,      # ← full source code
    "docstring": chunk.docstring,  # ← full docstring
    "language": chunk.language,
}
```

### Extrapolation to Duetto (17K files, 114K chunks)

| Scenario | Estimated JSON Size |
|----------|---------------------|
| Python codebase (avg 290B/chunk) | ~87 MB |
| C++ codebase (avg 1800B/chunk) | ~261 MB |
| C++ with large functions (avg 5.5KB/chunk) | ~600 MB |

The reported 300-700MB is entirely consistent with large C++ functions whose source is embedded verbatim.

### Root Cause

1. **`content` field:** Full source code of every chunk — the dominant driver.
2. **`docstring` field:** Full docstring text of every chunk.
3. **No deduplication:** `file_path` is a full absolute path repeated in every node (~48 bytes × 36K nodes = 1.7 MB just for paths).
4. **Indented JSON:** `json_exporter.py` uses `OPT_INDENT_2` — pretty-printing adds ~15-20% to file size.
5. **`text` chunk explosion:** Markdown files produce one node per paragraph/heading. A 1000-line CHANGELOG = ~200+ text nodes. For Duetto this could be enormous.

---

## 4. Issue 3: No Caching — Critical and Unresolved

### The Re-Parse Problem

Every API request reads and parses the full `chunk-graph.json` from disk:

```python
# This pattern is repeated in EVERY endpoint handler:
with open(graph_file, "rb") as f:
    data = orjson.loads(f.read())  # ← reads entire file on EVERY call
```

Endpoints that do this:
- `/api/graph` — full re-read every call
- `/api/graph-initial` — reads 87-600MB to return ~60KB
- `/api/graph-expand/{id}` — reads 87-600MB to return one node's children
- `/api/graph-code-chunks` — reads 87-600MB to return code nodes
- `/api/relationships/{id}` — reads 87-600MB then does O(n) AST scan
- `/api/callers/{id}` — reads 87-600MB then does O(n²) scan

### The O(n²) Relationship Scan

`/api/relationships/{chunk_id}` and `/api/callers/{chunk_id}` perform:
```python
for node in data.get("nodes", []):
    if node.get("type") != "chunk":
        continue
    content = node.get("content", "")
    if function_name in extract_calls(content):  # AST.parse() per node!
```

For 114K chunks: `114,000 × ast.parse()` per single relationship request. This is O(n) AST parses, each of which is ~10-100ms. Total: minutes per request.

### HTTP Cache Headers (Browser-Level)

Some endpoints have `Cache-Control: max-age=300` (5 minutes):
- `/api/graph-expand/{node_id}` — correct, but server still re-reads file
- `/api/graph-code-chunks` — correct, same problem
- `/api/relationships/{id}` — correct, same problem
- `/api/callers/{id}` — correct, same problem

These cache headers only help the browser avoid repeat fetches. The server still re-reads the full JSON file on every unique request.

There is **no server-side in-memory cache** of any parsed data.

---

## 5. Frontend Data Consumption Analysis

### What the Frontend Actually Needs from Nodes

The frontend uses `content` and `docstring` only in two places:

1. **Code viewer panel** (`showChunkViewer()` in scripts.py line 1843):
   ```javascript
   if (chunkData.content) {
       html += `<pre><code>${escapeHtml(chunkData.content)}</code></pre>`;
   }
   ```
   This is only triggered when a user **clicks** a specific chunk node.

2. **`calculateNodeSizes()`** (line 596): Uses `content.length` as fallback if `start_line`/`end_line` unavailable — but `start_line` and `end_line` are always populated, making this fallback dead code.

3. **`collapseSingleChildChains()`** (lines 499, 520): Copies `content` into a `collapsed_chunk` object when a file has only one chunk. Still display-only.

**Conclusion:** `content` and `docstring` are only needed when a user clicks a specific chunk. They are never needed for rendering the tree, treemap, or sunburst visualizations. Loading them for all 114K chunks upfront is pure waste.

### What Each Visualization Mode Actually Needs

| Field | Tree | Treemap | Sunburst | Code Viewer |
|-------|------|---------|----------|-------------|
| id | yes | yes | yes | yes |
| name | yes | yes | yes | yes |
| type | yes | yes | yes | yes |
| file_path | yes | yes | yes | yes |
| start_line, end_line | sizing | yes | yes | display |
| complexity | color | yes | yes | display |
| quality_score | no | color | color | display |
| smell_count, smells | report | tooltip | tooltip | display |
| complexity_grade | no | label | label | display |
| parent_id | tree | hierarchy | hierarchy | no |
| depth | yes | no | no | no |
| **content** | **no** | **no** | **no** | **yes (on click only)** |
| **docstring** | **no** | **no** | **no** | **yes (on click only)** |
| language | no | no | no | syntax highlight |
| lines_of_code | sizing | sizing | sizing | display |

---

## 6. Specific Recommendations

### Recommendation 1: Strip Content from Graph JSON (Highest Impact)

**Problem:** `content` + `docstring` = 42% of JSON file.
**Fix:** Remove these fields from `build_graph_data()` in `graph_builder.py`:

```python
# In graph_builder.py, change line 549-562:
node = {
    "id": chunk_id,
    "name": chunk_name,
    "type": chunk.chunk_type,
    "file_path": file_path_str,
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    "complexity": chunk.complexity_score,
    "parent_id": parent_id,
    "depth": chunk.chunk_depth,
    # REMOVE: "content": chunk.content,
    # REMOVE: "docstring": chunk.docstring,
    "language": chunk.language,
}
```

**Add a new endpoint** `/api/chunk-content/{chunk_id}` that fetches content on demand:
```python
@app.get("/api/chunk-content/{chunk_id}")
async def get_chunk_content(chunk_id: str) -> Response:
    # Query LanceDB directly for this one chunk's content
    # Return: {"content": "...", "docstring": "..."}
```

**Frontend change** in `showChunkViewer()`: fetch `/api/chunk-content/{id}` when a node is clicked, then show the code.

**Impact for Duetto:**
- 114K chunks × avg 1800B content = ~195 MB eliminated
- File size: 600 MB → ~400 MB (still too large, needs combination with other fixes)

### Recommendation 2: Server-Side In-Memory Cache (Required)

**Problem:** Every request re-reads and re-parses the full JSON file from disk.
**Fix:** Parse once at startup, keep in memory:

```python
# In server.py create_app():
import asyncio

# Module-level cache
_graph_cache: dict | None = None
_graph_cache_mtime: float = 0.0
_graph_cache_lock = asyncio.Lock()

async def get_graph_data(viz_dir: Path) -> dict:
    global _graph_cache, _graph_cache_mtime
    graph_file = viz_dir / "chunk-graph.json"

    try:
        current_mtime = graph_file.stat().st_mtime
    except FileNotFoundError:
        return {"nodes": [], "links": []}

    async with _graph_cache_lock:
        if _graph_cache is None or current_mtime > _graph_cache_mtime:
            with open(graph_file, "rb") as f:
                _graph_cache = orjson.loads(f.read())
            _graph_cache_mtime = current_mtime

    return _graph_cache
```

**Add pre-built indexes** for O(1) lookups:
```python
_node_index: dict[str, dict] | None = None  # id → node
_children_index: dict[str, list[str]] | None = None  # node_id → [child_ids]
_file_chunks_index: dict[str, list[str]] | None = None  # file_id → [chunk_ids]
```

**Impact:** `/api/graph-expand` goes from 87MB disk read → O(1) dict lookup.

### Recommendation 3: Separate Structural and Content Data Files

**Problem:** Even with caching, loading a 600MB file at startup is slow and wastes RAM.
**Fix:** Split `chunk-graph.json` into two files at export time:

- `chunk-graph-meta.json`: All nodes WITHOUT content/docstring (~140MB for Duetto)
- `chunk-content.json`: Map of `chunk_id → {content, docstring}` (~400MB for Duetto, but never loaded into memory)

The server serves `chunk-graph-meta.json` for all graph endpoints, and queries `chunk-content.json` only for the content endpoint.

### Recommendation 4: Strip Text/Documentation Chunks from Default Graph

**Problem:** 83.8% of nodes in the measured graph are `text` type (markdown paragraphs). These have `complexity=0`, `quality_score=undefined`, and serve no purpose in the tree/treemap/sunburst visualizations.

The frontend already excludes them from treemap/sunburst:
```javascript
// scripts.py line 4619
const children = allChildren.filter(child => {
    const t = child.type;
    return t === 'directory' || t === 'file' || codeChunkTypes.has(t);
});
```

**Fix:** The `--code-only` flag in `_export_chunks()` already filters these:
```python
chunks = [c for c in chunks if c.chunk_type not in ["text", "comment", "docstring"]]
```

**Make `--code-only` the default for the visualization command.** The full text-inclusive export is only needed for semantic search, not for visualization.

**Impact:** For mcp-vector-search itself: 36,928 nodes → ~5,961 nodes (84% reduction in node count).

### Recommendation 5: Node-Level Indexing in the JSON (LanceDB IDs)

**Problem:** `/api/relationships/{chunk_id}` and `/api/callers/{chunk_id}` do O(n) full scans.
**Fix for callers:** Pre-compute the call graph at export time and store it as a `callers` map in a separate sidecar file.

Better alternative: Replace the in-JSON AST scan with direct LanceDB queries. The server already has the database path — it can query LanceDB for the chunk content on demand instead of scanning the JSON.

### Recommendation 6: Compact JSON Format

**Problem:** `OPT_INDENT_2` in `json_exporter.py` adds ~15-20% overhead.
**Fix:**
```python
# Change from:
f.write(orjson.dumps(graph_data, option=orjson.OPT_INDENT_2))
# To:
f.write(orjson.dumps(graph_data))  # compact, no indent needed for machine consumption
```

**Impact:** ~15% file size reduction with zero functionality change.

---

## 7. Priority Matrix

| Fix | Effort | Impact for Duetto | Risk |
|-----|--------|-------------------|------|
| Make `--code-only` default | Low (1 line) | 84% fewer nodes | Low |
| Compact JSON (no indent) | Low (1 line) | -15% size | Low |
| In-memory graph cache | Medium (50 lines) | O(1) reads | Low |
| Strip content from graph JSON | Medium (30 lines + new API endpoint) | -37% size + fast code viewer | Low |
| Node index maps (id→node, parent→children) | Medium (30 lines) | O(1) expand | Low |
| Separate content sidecar file | Medium (100 lines) | Full separation | Medium |
| Replace AST scan with LanceDB query | High (new DB connection in server) | O(1) relationships | Medium |

---

## 8. Quick Win Implementation Order

**Phase 1 (immediate, <1 day):**
1. Change `OPT_INDENT_2` to no option in `json_exporter.py` (1 line)
2. Make `--code-only` default in `visualize serve` (change default in `cli.py`)
3. Add in-memory graph cache to `server.py` with a `_graph_data` module-level variable

**Phase 2 (1-2 days):**
4. Remove `content` and `docstring` from `build_graph_data()` node dict
5. Add `/api/chunk-content/{chunk_id}` endpoint that queries LanceDB
6. Update `showChunkViewer()` in scripts.py to fetch content on click

**Phase 3 (3-5 days):**
7. Build node index maps (`_node_by_id`, `_children_by_parent`) in the server cache
8. Rewrite `/api/graph-expand`, `/api/relationships`, `/api/callers` to use index maps instead of scanning

**Expected Duetto result after all phases:** 600MB → ~50MB file (with code-only + no content), and all API endpoints become O(1) or O(k) where k = number of children of the expanded node.

---

## 9. Memory Implications of Server-Side Caching

For Duetto at 114K chunks with the optimized (no-content) format:
- Estimated JSON size: ~55 MB
- Parsed Python dict: ~3-5x JSON size = ~165-275 MB RAM
- This is acceptable for a local development server (modern machines have 8-32GB)

If RAM is a concern, an alternative is a lightweight SQLite database built from the graph JSON at startup, allowing indexed queries without loading everything into memory.

---

*Research saved to: `docs/research/visualization-architecture-large-codebase-2026-02-28.md`*
*No ticket context provided — file-based capture only.*
