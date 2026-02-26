# Visualization Issues: Treemap/Sunburst Complexity and Hierarchy

**Date**: 2026-02-24
**Project**: mcp-vector-search
**Investigator**: Research Agent

---

## Summary

Two distinct issues were identified in the treemap and sunburst visualizations:

1. **Issue 1 (Complexity/Quality/Smells)**: The data pipeline is **mostly working** but has a schema mismatch bug and a data routing bug. Complexity values exist in the `chunks` table but a wrong column name (`complexity` vs `complexity_score`) causes them to fall back to `0.0` when reading through the batch path. The `compute_quality_metrics()` function then compensates with a LOC-based heuristic, so some visual encoding does appear — but it is inaccurate.

2. **Issue 2 (Hierarchy for AST chunks)**: The `parent_name` field in the `chunks` table is **always empty** (no nested chunk hierarchy), and `chunk_depth` / `parent_chunk_id` are not stored in the `chunks` table schema. This means the file→class→method hierarchy that treemap and sunburst need does not exist in the data. The `buildASTHierarchy()` function builds a flat Language→Type→Chunk grouping instead, which works but loses the class-method nesting.

---

## Issue 1: Complexity/Quality/Smell Visual Encoding

### Root Cause: Double Schema Mismatch

**A. `chunks` table column name is `complexity`, not `complexity_score`**

File: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/lancedb_backend.py`

- The `chunks` table schema (defined in `chunks_backend.py:66`) has column `complexity` (int32).
- The `vectors` table (defined in `vectors_backend.py`) does **not** have a `complexity` column at all.
- `_batch_dict_to_chunk()` at **line 1146** correctly reads `batch_dict.get("complexity", ...)` → maps to `CodeChunk.complexity_score`. This path works.
- `_row_to_chunk()` at **line 1228** reads `row.get("complexity", 0)` → also correct for the `chunks` table. This path works.

So complexity reading is correct IF the `chunks` table is used. Let's verify: `cli.py:161-164` calls `create_database(collection_name="chunks")` → `LanceVectorDatabase` opens the `chunks` table. **This path is correct.**

**B. The actual data confirms complexity is non-zero**

```
Code chunks in chunks table: 4805 (functions + classes + methods)
Complexity stats:
  non-zero: 4805 (100% have complexity data)
  zero:     0
  max:      537
  mean:     4.59
```

**C. graph_builder.py: `compute_quality_metrics()` is only called for chunk_types `function`, `method`, `class`**

`graph_builder.py:565`:
```python
if chunk.chunk_type in ("function", "method", "class"):
    quality_metrics = compute_quality_metrics(chunk)
    node.update(quality_metrics)
```

This is correct. However, `chunk.chunk_type == "text"` (30,960 out of 36,153 = 86% of chunks) gets **no** quality metrics at all — only `complexity: chunk.complexity_score` which will be 0.0 for text chunks.

**D. `getNodeColor()` in D3 (scripts.py:4768) reads `d.data.complexity`**

```javascript
function getNodeColor(d) {
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return blendComplexityQuality(d.data.complexity, d.data.quality_score);
    }
    // ...aggregation for parent nodes...
}
```

`node["complexity"] = chunk.complexity_score` is set at `graph_builder.py:556`. For `text` chunks this is 0.0.

### The Real Problem: `text` chunk_type dominates (86%) with zero complexity

The database has 36,153 total chunks:
- `text`: 30,960 (85.6%) — no complexity metrics set, `complexity = 0.0`
- `function`: 4,180 (11.6%) — HAS complexity (avg 4.59, max 537)
- `class`: 595 (1.6%) — HAS complexity
- `imports`: 370 (1.0%) — no complexity
- `method`: 30 — HAS complexity
- `module`: 16 — no complexity

**Effect**: When treemap/sunburst sizes cells by `lines_of_code`, text chunks (which can be large) dominate visually. Since `complexity = 0.0` for text chunks, `getComplexityColor(0)` returns gray (`#6e7681`). The entire treemap appears gray except for a small portion of function/class nodes.

**But wait — `compute_quality_metrics()` has a fallback at line 55-61:**
```python
if raw_complexity == 0:
    raw_complexity = max(1, lines_of_code / 5)
```

This only runs for `function`, `method`, `class` chunk types (per line 565 gate). Text chunks never enter this path, so they remain gray.

### What IS working

For function/class/method nodes:
- `compute_quality_metrics()` correctly computes quality, smells, and grade
- The D3 `blendComplexityQuality()` and `getSmellBorderColor()` functions work correctly
- The tooltip shows complexity, smells, and quality score

### What is NOT working correctly

1. `text` chunks (86% of data) always show gray in treemap/sunburst — no complexity encoding
2. The `buildFileHierarchy()` processes `treeData` which is the D3 tree structure built from `allNodes` — but the initial graph-initial API only returns directory nodes (depth 0-2), not chunk nodes. **Chunk nodes are loaded lazily via expand. So treemap/sunburst that call `buildFileHierarchy()` on unexpanded tree data will see mostly directory nodes with `complexity=0`.**

### Files and line numbers

| File | Line | Issue |
|------|------|-------|
| `graph_builder.py` | 552-556 | `node["complexity"] = chunk.complexity_score` — text chunks always 0.0 |
| `graph_builder.py` | 565 | Quality metrics only computed for `function/method/class`, not `text` |
| `server.py` | 200-215 | `graph-initial` API only returns dir nodes (depth 0-2) — no chunk nodes |
| `scripts.py` | 4835 | `buildFileHierarchy()` reads from `treeData` which may be unexpanded |
| `scripts.py` | 4768-4771 | `getNodeColor()` reads `d.data.complexity` — works if data present |

---

## Issue 2: Do Treemap/Sunburst Work with AST Chunks?

### Finding: Two Modes Exist — Both Have Issues

The JS frontend has two hierarchy-building modes:

#### Mode A: `buildFileHierarchy()` (default `currentGroupingMode === 'file'`)

**Structure**: root → directories → files → chunks
**Source**: Reads from `treeData` (D3 tree built from `allNodes`)
**Problem**: `treeData` is built from progressively-loaded nodes. On initial load, only directory nodes (depth 0-2) are included. Chunk nodes require clicking to expand files. So **on initial treemap/sunburst render, chunk nodes are missing entirely** — only colored directory rectangles appear with `complexity=0`.

After full expansion (user clicks all directories), file and chunk nodes appear. The hierarchy is correct (file → class/function), but **class→method nesting is broken** because:
- `parent_chunk_id` is not stored in the `chunks` table (not in schema)
- `chunk_depth` is not stored in the `chunks` table (not in schema)
- `parent_name` field in `chunks` table is always empty

So all function/class/method chunks appear as flat children of their file node, not as class→method nested trees.

#### Mode B: `buildASTHierarchy()` (`currentGroupingMode === 'ast'`)

**Structure**: root → Language → Type (Functions/Classes/Methods) → individual chunks
**Source**: Reads from `cachedChunkNodes || allNodes.filter(chunkTypes.includes(type))`
**Problem**: This uses ALL loaded `allNodes` which include chunk nodes loaded via expansion. BUT the hierarchy is flat grouping by language+type — not the actual AST hierarchy (class containing its methods).

**This mode actually works reasonably well** for treemap/sunburst because:
- It correctly groups chunks by language and type
- Each chunk node has `complexity`, `quality_score`, `smell_count` from `compute_quality_metrics()`
- D3 `sum(d => d.lines_of_code)` sizes boxes by code size

But it has a dependency: chunk nodes must be loaded first (via file expansion). On initial page load before any expansion, `allNodes` only has directory nodes — so `buildASTHierarchy()` returns an empty or nearly-empty tree.

### The Missing Hierarchy Fields in chunks Table

**`chunks` table schema** (`chunks_backend.py`):
```
chunk_id, file_path, file_hash, content, language, start_line, end_line, chunk_type,
name, start_char, end_char, parent_name, hierarchy_path, docstring, signature,
complexity, token_count, calls, imports, inherits_from, last_author, last_modified,
commit_hash, embedding_status, embedding_batch_id, created_at, updated_at, error_message
```

**Missing**: `parent_chunk_id`, `chunk_depth`, `child_chunk_ids`

The `parent_name` field exists but is always empty (`""`) in the database — so even the name-based parent relationship is unavailable.

**The `vectors` table schema** (`vectors_backend.py`) also lacks `parent_chunk_id`, `chunk_depth`.

**The old `LanceDB schema`** in `lancedb_backend.py:line 71-73** DID have these fields:
```python
pa.field("parent_chunk_id", pa.string()),
pa.field("child_chunk_ids", pa.string()),
pa.field("chunk_depth", pa.int32()),
```

But this is the OLD schema for the legacy `lancedb_backend.py` (which opens the `vectors` table). The **new two-phase schema** (`chunks.lance` + `vectors.lance`) dropped these fields from both tables.

### Actual chunk types in the database

```
text: 30960    # Documentation/markdown text (NO hierarchy, NO parent)
function: 4180  # Top-level functions (parent_name='')
class: 595      # Classes (parent_name='')
imports: 370    # Import blocks
method: 30      # Methods (parent_name='' — should link to class but doesn't!)
module: 16      # Module-level code
```

**Only 30 method chunks exist** and none have `parent_name` set to their class. This means no class→method nesting is possible at all with current data.

### Should treemap/sunburst be removed?

**No — but they need data to be loaded first.**

The treemap and sunburst views ARE fundamentally compatible with AST chunk data, especially in `ast` grouping mode. The visual encoding (complexity→color, smells→border, LOC→size) is well-designed and works correctly when chunk nodes are present.

**The core problem is architectural: progressive loading means chunk nodes aren't available at treemap/sunburst render time.**

---

## Specific Files Needing Changes

### Fix 1: Load All Chunk Nodes Before Treemap/Sunburst Render

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Lines 4826-4835** (`renderTreemap`) and **5260-5270** (`renderSunburst`):

Current:
```javascript
const hierarchyData = currentGroupingMode === 'ast' ? buildASTHierarchy() : buildFileHierarchy();
```

Problem: If chunk nodes haven't been expanded yet, both functions return trees with no leaf-level quality data.

**Fix**: Add a pre-render check that loads all chunk nodes before rendering treemap/sunburst. Add a dedicated `/api/graph-chunks` endpoint or use `/api/graph-expand-all` to pre-fetch all file-level nodes.

Alternative (simpler): When switching to treemap/sunburst mode, trigger expansion of all visible file nodes automatically.

### Fix 2: Add `parent_chunk_id` and `chunk_depth` to chunks Table Schema

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/chunks_backend.py`

Current schema (around line 44-80) lacks `parent_chunk_id`, `chunk_depth`, `child_chunk_ids`.

Add:
```python
pa.field("parent_chunk_id", pa.string()),   # ID of parent chunk (class for method)
pa.field("chunk_depth", pa.int32()),         # 0=top-level, 1=class method, 2=nested
pa.field("child_chunk_ids", pa.list_(pa.string())),  # IDs of child chunks
```

Also add to the `vectors` table schema in `vectors_backend.py` for denormalization.

### Fix 3: Populate parent_name/parent_chunk_id During Indexing

**File**: Wherever the AST parsing/chunking happens (likely `indexer.py` or the parser module)

The `parent_name` field is never populated for `method` chunk types. Need to:
1. Set `parent_name = class_name` for method chunks
2. Set `parent_chunk_id` to the class chunk's `chunk_id` for method chunks
3. Set `chunk_depth = 1` for methods inside classes

### Fix 4: Ensure Text Chunks Get Reasonable Complexity for Sizing

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

Lines 549-556: Currently `node["complexity"] = chunk.complexity_score` is 0.0 for text chunks.

For treemap sizing purposes (not quality encoding), text chunks should get `lines_of_code` set properly. The sizing in D3 uses `lines_of_code` (`scripts.py:4839-4843`), not complexity, so this is less critical.

### Fix 5: Dedicated Bulk-Load API for Treemap/Sunburst Mode

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py`

Add a new endpoint `/api/graph-chunks-only` that returns ALL function/class/method chunk nodes (not directories) for direct use by treemap/sunburst. This avoids the progressive loading bottleneck:

```python
@app.get("/api/graph-chunks-only")
async def get_chunks_only():
    """Return all code chunks (function/class/method) for treemap/sunburst."""
    # Filter to code chunks only
    chunk_types = {"function", "class", "method"}
    chunk_nodes = [n for n in all_nodes if n.get("type") in chunk_types]
    return Response(content=orjson.dumps({"nodes": chunk_nodes}), ...)
```

---

## Data Flow Diagram

```
chunks.lance table
  ├── chunk_type: function/class/method → HAS complexity (int, avg 4.6, max 537)
  ├── chunk_type: text → complexity=0 (no quality metrics)
  └── parent_name: ALWAYS "" (no class→method nesting)
          ↓
lancedb_backend._batch_dict_to_chunk()
  ├── maps 'complexity' → CodeChunk.complexity_score  ✓ CORRECT
  └── sets parent_chunk_id=None, chunk_depth=0 always  ✗ BROKEN
          ↓
graph_builder.build_graph_data()
  ├── node["complexity"] = chunk.complexity_score (0.0 for text chunks)
  ├── if chunk_type in (function/method/class):
  │     compute_quality_metrics() → adds quality_score, smell_count, smells, complexity_grade ✓
  └── parent_id = chunk.parent_chunk_id or file_nodes[file_path]["id"] (all → file node)
          ↓
graph-initial API (server.py)
  └── Returns ONLY directory nodes (depth 0-2) — NO chunk nodes
          ↓
Frontend JavaScript
  ├── allNodes = [directory nodes only]
  ├── buildASTHierarchy() → EMPTY (no chunkTypes in allNodes)
  └── buildFileHierarchy() → only directories visible
          ↓
treemap/sunburst renders with only directory nodes → ALL GRAY (complexity=0)
```

After user expands files:
```
allNodes += [file nodes + chunk nodes]
buildASTHierarchy() → Language→Type→Chunk (WORKS but flat, no class nesting)
buildFileHierarchy() → dir→file→chunk (WORKS but all chunks are flat under file)
```

---

## Recommended Fix Priority

**Priority 1 (Quick fix — makes treemap/sunburst immediately useful)**:
- Add `/api/graph-chunks-only` endpoint that returns all function/class/method nodes
- In `renderTreemap()` and `renderSunburst()`, if chunk nodes are absent, auto-fetch them first
- Use `ast` grouping mode as default for treemap/sunburst (avoids need for full tree expansion)

**Priority 2 (Data pipeline fix — enables proper hierarchy)**:
- Add `parent_chunk_id` and `chunk_depth` to `chunks` table schema
- Populate these fields during AST parsing (method → parent class)
- Update `_batch_dict_to_chunk` and `_row_to_chunk` to read new fields

**Priority 3 (Long-term — remove text chunk noise)**:
- Add `--code-only` as default for treemap/sunburst mode
- Or filter text chunks from hierarchy building in `buildFileHierarchy()` and `buildASTHierarchy()`

---

## Saved To

`docs/research/visualization-treemap-sunburst-issues-2026-02-24.md`
