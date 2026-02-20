# Visualization System Research: "visualize index" Command Design

**Research Date**: 2026-02-20
**Project**: mcp-vector-search
**Objective**: Design a "visualize index" command that pre-builds all JSON necessary for visualization

## Executive Summary

The current visualization system generates JSON on-demand via FastAPI endpoints, which works but can be slow for large codebases. This research investigates what data structures are needed and proposes a `mcp-vector-search visualize index` command to pre-build all visualization JSON files for instant loading.

### Key Findings

1. **Current System Architecture**: Uses progressive loading with FastAPI server generating JSON from database queries
2. **Two Main Data Sources**: Chunk graph (19MB) and KG graph (6.6MB)
3. **Performance Bottleneck**: Initial graph generation takes time; semantic relationships computed lazily
4. **Pre-build Opportunity**: Most data is static once indexed; only relationships need lazy loading

### Recommendation

Implement `mcp-vector-search visualize index` command that:
- Pre-builds all static JSON files (chunk-graph.json, kg-graph.json, directory metadata)
- Stores in `.mcp-vector-search/visualization/` directory
- Includes stale detection to trigger regeneration when index changes
- Maintains lazy-loading for expensive operations (semantic search, caller detection)

---

## Current Visualization System Architecture

### 1. Command Structure

The visualization system is located at `src/mcp_vector_search/cli/commands/visualize/`:

```
visualize/
├── cli.py                 # Main CLI commands (export, serve)
├── server.py              # FastAPI server with progressive loading
├── graph_builder.py       # Graph data construction from chunks
├── layout_engine.py       # Layout algorithms (unused in current impl)
├── state_manager.py       # Visualization state management
├── exporters/
│   ├── json_exporter.py   # Export to JSON (uses orjson)
│   └── html_exporter.py   # Export to HTML
└── templates/
    ├── base.py            # HTML template generation
    ├── scripts.py         # JavaScript for D3.js visualization
    └── styles.py          # CSS styles
```

### 2. Current Commands

**`mcp-vector-search visualize export`**
- Exports chunk graph to `chunk-graph.json` (default location)
- Fetches all chunks from database
- Builds graph structure with nodes and links
- Options: `--file` filter, `--code-only` filter

**`mcp-vector-search visualize serve`**
- Starts FastAPI server on port 8501
- Auto-generates `chunk-graph.json` if missing
- Auto-generates `kg-graph.json` if missing
- Detects stale files (index newer than graph)
- Serves progressive loading endpoints

### 3. Data Flow

```
Index Time (mcp-vector-search index):
  → Chunks stored in vectors.lance
  → Metadata in chunks.lance
  → Directory structure in directory_index.json
  → Index metadata in index_metadata.json
  → Knowledge graph in knowledge_graph/ (KuzuDB)

Visualization Time (mcp-vector-search visualize serve):
  1. Check for chunk-graph.json
  2. If missing/stale: Run _export_chunks()
     → Query database.get_all_chunks()
     → Call build_graph_data()
     → Export to JSON via json_exporter.py
  3. Check for kg-graph.json
  4. If missing/stale: Run _export_kg_data()
     → Load KnowledgeGraph
     → Call kg.get_visualization_data()
     → Export to JSON
  5. Start FastAPI server
  6. Serve progressive loading endpoints
```

---

## JSON Data Structures

### 1. Chunk Graph JSON (`chunk-graph.json`)

**Size**: 19 MB (current project)
**Location**: `.mcp-vector-search/visualization/chunk-graph.json`

**Structure**:
```json
{
  "nodes": [
    {
      "id": "dir_795c7b65",
      "name": "docs",
      "type": "directory",
      "file_path": "docs",
      "start_line": 0,
      "end_line": 0,
      "complexity": 0,
      "depth": 1,
      "dir_path": "docs",
      "parent_id": null,
      "parent_path": null,
      "file_count": 0,
      "subdirectory_count": 17,
      "total_chunks": 287,
      "languages": {},
      "is_package": false,
      "last_modified": 1771363256.457739
    },
    {
      "id": "file_a1b2c3d4",
      "name": "main.py",
      "type": "file",
      "file_path": "src/main.py",
      "depth": 2,
      "parent_id": "dir_xyz123",
      "chunk_count": 15
    },
    {
      "id": "chunk_xyz789",
      "name": "calculate_total",
      "type": "function",
      "file_path": "src/main.py",
      "start_line": 42,
      "end_line": 58,
      "complexity": 5,
      "parent_id": "file_a1b2c3d4",
      "depth": 3,
      "content": "def calculate_total(...)...",
      "docstring": "Calculate the total sum",
      "language": "python",
      "cognitive_complexity": 8,
      "cyclomatic_complexity": 3,
      "complexity_grade": "B",
      "code_smells": ["long-function"],
      "smell_count": 1,
      "quality_score": 0.85,
      "lines_of_code": 16
    }
  ],
  "links": [
    {
      "source": "dir_xyz123",
      "target": "file_a1b2c3d4",
      "type": "dir_containment"
    },
    {
      "source": "file_a1b2c3d4",
      "target": "chunk_xyz789",
      "type": "file_containment"
    },
    {
      "source": "chunk_parent",
      "target": "chunk_child",
      "type": "chunk_hierarchy"
    },
    {
      "source": "chunk_caller",
      "target": "chunk_callee",
      "type": "semantic",
      "similarity": 0.85
    }
  ],
  "metadata": {
    "total_chunks": 5432,
    "total_files": 234,
    "languages": {
      "python": 180,
      "javascript": 45,
      "markdown": 9
    },
    "is_monorepo": false,
    "subprojects": []
  },
  "trends": {
    "entries": [
      {
        "date": "2026-02-20",
        "files": 234,
        "chunks": 5432,
        "lines_of_code": 45678,
        "avg_complexity": 3.2,
        "health_score": 0.87
      }
    ]
  }
}
```

**Node Types**:
- `monorepo`: Root node for monorepo projects
- `subproject`: Subproject in monorepo
- `directory`: Directory nodes with metadata
- `file`: File nodes with chunk counts
- `function`, `class`, `method`: Code chunks
- `text`, `comment`, `docstring`: Documentation chunks
- `imports`, `module`: Import/module chunks

**Link Types**:
- `monorepo_containment`: Monorepo → Subproject
- `subproject_containment`: Subproject → Top-level chunks
- `dir_containment`: Directory → Subdirectory/File
- `file_containment`: File → Top-level chunks
- `chunk_hierarchy`: Parent chunk → Child chunk (e.g., class → method)
- `semantic`: Semantic similarity between chunks (lazy-loaded)
- `caller`: Function call relationship (lazy-loaded)
- `dependency`: Inter-project dependencies (monorepo)

### 2. Knowledge Graph JSON (`kg-graph.json`)

**Size**: 6.6 MB (current project)
**Location**: `.mcp-vector-search/visualization/kg-graph.json`

**Structure**:
```json
{
  "nodes": [
    {
      "id": "07cd7bae0fca1396",
      "name": "connection_pooling_example",
      "type": "imports",
      "file_path": "examples/connection_pooling_example.py",
      "group": 0
    },
    {
      "id": "module:asyncio",
      "name": "asyncio",
      "type": "module",
      "file_path": "",
      "group": 2
    },
    {
      "id": "function:calculate_total",
      "name": "calculate_total",
      "type": "function",
      "file_path": "src/main.py",
      "group": 1
    }
  ],
  "links": [
    {
      "source": "chunk_id_1",
      "target": "module:asyncio",
      "type": "imports",
      "weight": 1.0
    },
    {
      "source": "function:caller",
      "target": "function:callee",
      "type": "calls",
      "weight": 1.0
    },
    {
      "source": "class:child",
      "target": "class:parent",
      "type": "inherits",
      "weight": 1.0
    }
  ],
  "metadata": {
    "is_monorepo": false,
    "subprojects": [],
    "total_entities": 1234,
    "total_relationships": 3456
  }
}
```

**Node Types** (from KuzuDB):
- `CodeEntity`: Functions, classes, files
- `DocSection`: Documentation sections
- `Tag`: Topic tags
- `Person`: Authors (git)
- `Project`: Project metadata
- `Repository`: Git repository
- `Branch`: Git branches
- `module`: External/internal modules

**Link Types**:
- `calls`: Function call relationships
- `imports`: Import relationships
- `inherits`: Class inheritance
- `contains`: File contains entity
- `authored`: Person authored code
- `tagged`: Entity tagged with topic

### 3. Directory Index JSON (`directory_index.json`)

**Size**: 71 KB (current project)
**Location**: `.mcp-vector-search/directory_index.json`

**Structure**:
```json
{
  "directories": [
    {
      "path": "docs",
      "name": "docs",
      "parent_path": null,
      "file_count": 0,
      "subdirectory_count": 18,
      "total_chunks": 320,
      "languages": {},
      "depth": 1,
      "is_package": false,
      "last_modified": 1771573960.3094437
    }
  ],
  "file_to_directory": {
    "src/main.py": "src",
    "docs/README.md": "docs"
  }
}
```

### 4. Index Metadata JSON (`index_metadata.json`)

**Size**: 78 KB (current project)
**Location**: `.mcp-vector-search/index_metadata.json`

**Structure**:
```json
{
  "index_version": "2.5.55",
  "indexed_at": "2026-02-20T13:47:20.553897+00:00",
  "file_mtimes": {
    "/Users/masa/Projects/mcp-vector-search/CHANGELOG.md": 1770347648.900154,
    "/Users/masa/Projects/mcp-vector-search/pyproject.toml": 1771567454.4464188
  }
}
```

### 5. KG Metadata JSON (`knowledge_graph/kg_metadata.json`)

**Size**: 892 KB (current project)
**Location**: `.mcp-vector-search/knowledge_graph/kg_metadata.json`

**Structure**: Contains KuzuDB schema and metadata information.

### 6. Other Supporting JSON Files

**`config.json`** (1.5 KB): Project configuration
**`vendor_metadata.json`** (267 B): Vendor dependency metadata
**`schema_version.json`** (76 B): Database schema version
**`progress.json`** (385 B): Indexing progress tracking
**`search_history.json`** (7.1 KB): Search query history

---

## Current JSON Generation Process

### 1. Chunk Graph Generation (`graph_builder.py::build_graph_data`)

**Input Sources**:
- `database.get_all_chunks()` → Fetches all chunks from vectors.lance/chunks.lance
- `DirectoryIndex` → Loads from `directory_index.json`
- `TrendTracker` → Loads from `trends.json` (if exists)
- `ProjectManager` → Project metadata

**Process**:
1. Collect subprojects (monorepo detection)
2. Build directory nodes from DirectoryIndex
3. Build file nodes from chunks
4. Build chunk nodes (function, class, method, etc.)
5. Create containment links (directory → file → chunk hierarchy)
6. Parse inter-project dependencies (package.json in monorepos)
7. Load trend data for time series
8. **Skip relationship computation** (lazy-loaded via API)

**Performance**:
- Fast: ~5-10 seconds for 5000+ chunks
- Memory efficient: Streams from database
- No semantic search at build time (avoids 5+ minute computation)

**Output**: Returns dict with nodes, links, metadata, trends

### 2. KG Graph Generation (`knowledge_graph.py::get_visualization_data`)

**Input Sources**:
- KuzuDB queries on `knowledge_graph/code_kg` database
- Queries: CodeEntity, DocSection, Tag, Person, Project nodes
- Relationship queries: CALLS, IMPORTS, INHERITS, CONTAINS, AUTHORED

**Process**:
1. Query all CodeEntity nodes
2. Query all DocSection nodes
3. Query all Tag nodes
4. Query all Person nodes
5. Query all Project nodes
6. Detect monorepo structure
7. Query all relationship edges
8. Assign group IDs for visualization coloring

**Performance**:
- Moderate: ~2-5 seconds for moderate-sized KGs
- Memory efficient: Streams from KuzuDB
- Pre-computed relationships (already in KG)

**Output**: Returns dict with nodes, links, metadata

### 3. JSON Export (`exporters/json_exporter.py::export_to_json`)

**Process**:
1. Takes graph_data dict
2. Uses `orjson` for fast serialization (5-10x faster than stdlib json)
3. Writes with `OPT_INDENT_2` for readability
4. Ensures parent directory exists

**Performance**:
- Fast: ~1-2 seconds for 19MB JSON
- Uses orjson for optimal performance

---

## FastAPI Server Endpoints

### Progressive Loading Endpoints

**`/api/graph-status`** (GET)
- Returns: `{"ready": bool, "size": int}`
- Checks if chunk-graph.json exists and has content

**`/api/graph-initial`** (GET)
- Returns: Top-level nodes only (root dirs, depth 0-2)
- Purpose: Fast initial render (~100-200 nodes)
- Filters from full chunk-graph.json in memory

**`/api/graph-expand/{node_id}`** (GET)
- Returns: Direct children of node_id
- Purpose: Progressive loading on node expansion
- Fetches from chunk-graph.json in memory

**`/api/relationships/{chunk_id}`** (GET)
- Returns: Semantic neighbors + callers for a chunk
- **Lazy-loaded**: Computed on-demand when node expanded
- Uses AST parsing to detect function calls
- Uses Jaccard similarity for semantic matching
- Caches for 5 minutes

**`/api/callers/{chunk_id}`** (GET)
- Returns: Callers for a specific chunk
- **Lazy-loaded**: AST parsing on-demand
- Caches for 5 minutes

**`/api/kg-graph`** (GET)
- Returns: Full KG graph (nodes, links, metadata)
- Loads from kg-graph.json

**`/api/kg-expand/{node_id}`** (GET)
- Returns: Neighbors within N hops
- **Dynamic**: Queries KuzuDB on-demand
- Aggregates if >30 children

### Static File Endpoints

**`/`** (GET) → Serves `index.html`
**`/{path:path}`** (GET) → Serves static files from viz_dir

---

## Performance Analysis

### Current System Performance

**Initial Load**:
1. Check graph-status: <100ms
2. Fetch /api/graph-initial: ~200-500ms (19MB file read + filter)
3. Render initial tree: ~500ms (200 nodes)
4. **Total: ~1-2 seconds** for first view

**Node Expansion**:
1. Fetch /api/graph-expand/{node_id}: ~100-300ms
2. Render children: ~100ms
3. **Total: ~200-400ms** per expand

**Relationship Loading** (Expensive):
1. Fetch /api/relationships/{chunk_id}: **~2-5 seconds**
   - AST parsing of all chunks: ~1-3 seconds
   - Jaccard similarity computation: ~1-2 seconds
2. **Total: 2-5 seconds** per chunk relationship query

### Bottlenecks

1. **Relationship computation**: AST parsing + semantic search is slow
2. **Large file reads**: 19MB chunk-graph.json loaded into memory
3. **No caching**: Every server restart reloads everything
4. **Cold start**: First graph generation takes ~10-15 seconds

### Optimization Opportunities

1. **Pre-build JSON files**: Avoid generation on every serve
2. **Stale detection**: Regenerate only when index changes
3. **Lazy relationship loading**: Keep current approach (it's good)
4. **Incremental updates**: Update only changed files (future)

---

## Proposed "visualize index" Command

### Command Design

**Name**: `mcp-vector-search visualize index`

**Purpose**: Pre-build all static JSON files for instant visualization loading

**Usage**:
```bash
# Build visualization index
mcp-vector-search visualize index

# Force rebuild even if up-to-date
mcp-vector-search visualize index --force

# Verbose output
mcp-vector-search visualize index --verbose
```

### What It Should Pre-Build

**1. Chunk Graph (`chunk-graph.json`)**
- Location: `.mcp-vector-search/visualization/chunk-graph.json`
- Size: ~19 MB (current project)
- Content: Nodes, links (hierarchy only), metadata, trends
- Excludes: Semantic relationships (lazy-loaded)

**2. Knowledge Graph (`kg-graph.json`)**
- Location: `.mcp-vector-search/visualization/kg-graph.json`
- Size: ~6.6 MB (current project)
- Content: KG nodes, relationship links, metadata
- Includes: Pre-computed relationships from KuzuDB

**3. HTML Template (`index.html`)**
- Location: `.mcp-vector-search/visualization/index.html`
- Size: ~290 KB (with embedded JavaScript)
- Content: D3.js visualization UI
- Generated: From templates/base.py, templates/scripts.py, templates/styles.py

**4. Metadata Cache (Optional)**
- Location: `.mcp-vector-search/visualization/metadata.json`
- Size: ~5-10 KB
- Content: Build timestamp, file hashes, version info
- Purpose: Stale detection

### Storage Location

**Primary**: `.mcp-vector-search/visualization/`

```
.mcp-vector-search/
├── visualization/
│   ├── chunk-graph.json      # Pre-built chunk graph
│   ├── kg-graph.json          # Pre-built KG graph
│   ├── index.html             # Visualization UI
│   ├── metadata.json          # Build metadata
│   └── favicon.ico            # (optional)
├── knowledge_graph/           # KuzuDB database
├── lance/                     # LanceDB vectors/chunks
├── directory_index.json       # Directory structure
├── index_metadata.json        # Index metadata
└── config.json                # Project config
```

### Stale Detection Strategy

**Triggers for Regeneration**:
1. `index_metadata.json` timestamp > `chunk-graph.json` timestamp
2. `knowledge_graph/code_kg` mtime > `kg-graph.json` timestamp
3. `--force` flag provided
4. Missing visualization files

**Metadata File Structure**:
```json
{
  "build_version": "2.5.55",
  "built_at": "2026-02-20T14:30:00Z",
  "index_timestamp": 1771567440.553897,
  "kg_timestamp": 1771573960.309443,
  "chunk_graph_hash": "sha256:abc123...",
  "kg_graph_hash": "sha256:def456...",
  "total_nodes": 5432,
  "total_links": 8765,
  "build_duration_seconds": 12.5
}
```

### Implementation Plan

**Step 1: Create `visualize index` command**
- Add to `cli.py` as new command
- Options: `--force`, `--verbose`

**Step 2: Build chunk graph**
- Reuse existing `_export_chunks()` logic
- Save to `.mcp-vector-search/visualization/chunk-graph.json`

**Step 3: Build KG graph**
- Reuse existing `_export_kg_data()` logic
- Save to `.mcp-vector-search/visualization/kg-graph.json`

**Step 4: Generate HTML**
- Call `export_to_html()` to create index.html
- Save to `.mcp-vector-search/visualization/index.html`

**Step 5: Create metadata file**
- Gather timestamps from index_metadata.json and KG database
- Calculate file hashes (optional)
- Save build metadata

**Step 6: Add stale detection to `serve`**
- Check metadata.json timestamps
- Compare with index_metadata.json and KG database
- Auto-regenerate if stale (or warn user)

### Modified Serve Command Behavior

**Current**: Generates JSON on every serve (or if missing)
**Proposed**: Use pre-built JSON, warn if stale

```python
@app.command()
def serve(
    port: int = 8501,
    auto_regenerate: bool = True,  # NEW: Auto-regen if stale
) -> None:
    """Start visualization server using pre-built index."""

    # Check if visualization index exists
    viz_dir = project_root / ".mcp-vector-search" / "visualization"

    if not viz_dir.exists() or not (viz_dir / "chunk-graph.json").exists():
        console.print("[yellow]Visualization index not found. Building now...[/yellow]")
        # Call visualize index command
        ctx.invoke(index_command)

    # Check if stale
    is_stale = check_visualization_stale(viz_dir)

    if is_stale:
        if auto_regenerate:
            console.print("[yellow]Visualization index is stale. Rebuilding...[/yellow]")
            ctx.invoke(index_command)
        else:
            console.print("[yellow]⚠ Warning: Visualization index is stale.[/yellow]")
            console.print("[dim]Run 'mcp-vector-search visualize index' to rebuild[/dim]")

    # Start server (uses pre-built JSON)
    start_visualization_server(port, viz_dir, auto_open=True)
```

---

## Lazy-Loading Strategy (Keep Current Approach)

### What Should NOT Be Pre-Built

**1. Semantic Relationships**
- **Why**: Expensive to compute (5+ minutes for 5000 chunks)
- **Current**: Lazy-loaded via `/api/relationships/{chunk_id}`
- **Keep**: On-demand computation when user expands node

**2. Caller Detection**
- **Why**: Requires AST parsing of all chunks (1-3 seconds per query)
- **Current**: Lazy-loaded via `/api/callers/{chunk_id}`
- **Keep**: On-demand with 5-minute cache

**3. KG Node Expansion**
- **Why**: Dynamic queries on KuzuDB (unpredictable neighborhood sizes)
- **Current**: Lazy-loaded via `/api/kg-expand/{node_id}`
- **Keep**: On-demand with aggregation for >30 children

### Rationale for Lazy Loading

**Performance**: Pre-computing all relationships would take 20+ minutes
**Memory**: Full relationship graph would be 100+ MB
**UX**: Users rarely need all relationships; on-demand is faster
**Caching**: 5-minute cache provides good balance of freshness and performance

---

## File List (All Relevant Source Files)

### Core Visualization Files

1. **`src/mcp_vector_search/cli/commands/visualize/cli.py`**
   - Main CLI commands (export, serve)
   - Entry point for proposed `index` command

2. **`src/mcp_vector_search/cli/commands/visualize/server.py`**
   - FastAPI server implementation
   - Progressive loading endpoints
   - Lazy relationship loading

3. **`src/mcp_vector_search/cli/commands/visualize/graph_builder.py`**
   - `build_graph_data()` function
   - Node and link construction
   - Directory/file/chunk hierarchy

4. **`src/mcp_vector_search/cli/commands/visualize/state_manager.py`**
   - Visualization state management
   - Expansion tracking

5. **`src/mcp_vector_search/cli/commands/visualize/layout_engine.py`**
   - Layout algorithms (currently unused)

6. **`src/mcp_vector_search/cli/commands/visualize/exporters/json_exporter.py`**
   - `export_to_json()` function
   - orjson serialization

7. **`src/mcp_vector_search/cli/commands/visualize/exporters/html_exporter.py`**
   - `export_to_html()` function
   - HTML template generation

8. **`src/mcp_vector_search/cli/commands/visualize/templates/base.py`**
   - HTML structure generation

9. **`src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`**
   - JavaScript for D3.js visualization
   - Progressive loading logic

10. **`src/mcp_vector_search/cli/commands/visualize/templates/styles.py`**
    - CSS styles

### Data Source Files

11. **`src/mcp_vector_search/core/knowledge_graph.py`**
    - `get_visualization_data()` method (line 3139)
    - KuzuDB queries for KG export

12. **`src/mcp_vector_search/core/directory_index.py`**
    - `DirectoryIndex` class
    - Directory metadata management

13. **`src/mcp_vector_search/analysis/trends.py`**
    - `TrendTracker` class
    - Time series data loading

14. **`src/mcp_vector_search/core/project.py`**
    - `ProjectManager` class
    - Project metadata

15. **`src/mcp_vector_search/core/database.py`**
    - `get_all_chunks()` method
    - Database access

### Configuration Files

16. **`.mcp-vector-search/directory_index.json`** (71 KB)
17. **`.mcp-vector-search/index_metadata.json`** (78 KB)
18. **`.mcp-vector-search/config.json`** (1.5 KB)
19. **`.mcp-vector-search/knowledge_graph/kg_metadata.json`** (892 KB)

### Output Files (Generated)

20. **`.mcp-vector-search/visualization/chunk-graph.json`** (19 MB)
21. **`.mcp-vector-search/visualization/kg-graph.json`** (6.6 MB)
22. **`.mcp-vector-search/visualization/index.html`** (290 KB)

---

## Design Considerations

### Why Pre-Build?

**1. Performance**
- Avoid 10-15 second generation on every serve
- Instant loading for visualization server

**2. Consistency**
- Same JSON used across multiple serve sessions
- Reproducible builds

**3. CI/CD Integration**
- Pre-build in CI pipeline
- Deploy static visualization alongside docs

**4. Offline Use**
- Copy visualization directory to share
- No need to run indexing on recipient's machine

### Why NOT Pre-Build Everything?

**1. Relationships are expensive**
- 5+ minutes to compute all semantic relationships
- 20+ minutes for full caller detection
- 100+ MB JSON if pre-computed

**2. Dynamic queries are better**
- KG expansion depends on user navigation
- On-demand caching provides good UX
- Aggregation logic handles large neighborhoods

**3. Incremental updates**
- Future: Only regenerate changed files
- Pre-building everything blocks incremental approach

### Trade-offs

**Storage**:
- Pre-built: ~26 MB total (chunk-graph + kg-graph + html)
- On-demand: 0 MB (generated dynamically)
- Decision: Storage is cheap, time is expensive

**Maintenance**:
- Pre-built: Need to detect staleness, regenerate
- On-demand: Always fresh
- Decision: Staleness detection is simple (compare timestamps)

**Flexibility**:
- Pre-built: Fixed at build time
- On-demand: Can apply filters dynamically
- Decision: Pre-build default view, support filters via API parameters

---

## Implementation Checklist

### Phase 1: Basic Index Command (MVP)

- [ ] Create `visualize index` command in `cli.py`
- [ ] Reuse `_export_chunks()` to generate chunk-graph.json
- [ ] Reuse `_export_kg_data()` to generate kg-graph.json
- [ ] Call `export_to_html()` to generate index.html
- [ ] Save all files to `.mcp-vector-search/visualization/`
- [ ] Add success message with file paths and sizes

### Phase 2: Stale Detection

- [ ] Create `metadata.json` structure
- [ ] Write build metadata after index generation
- [ ] Implement `check_visualization_stale()` function
- [ ] Compare index_metadata.json timestamp
- [ ] Compare knowledge_graph mtime
- [ ] Add `--force` flag to skip staleness check

### Phase 3: Serve Integration

- [ ] Modify `serve` command to check for pre-built index
- [ ] Add `--auto-regenerate` flag (default: true)
- [ ] Warn if stale (unless auto-regenerate)
- [ ] Remove on-demand generation (use pre-built only)
- [ ] Keep lazy-loading endpoints unchanged

### Phase 4: Polish

- [ ] Add `--verbose` flag for detailed output
- [ ] Add progress indicators during build
- [ ] Add file size reporting
- [ ] Add build duration reporting
- [ ] Update documentation

### Phase 5: Future Enhancements

- [ ] Incremental updates (only changed files)
- [ ] Multiple visualization presets (code-only, full, etc.)
- [ ] Compression option for large graphs
- [ ] Export to static HTML (no server needed)

---

## Conclusion

The proposed `mcp-vector-search visualize index` command will:

1. **Pre-build static JSON files** (chunk-graph.json, kg-graph.json, index.html)
2. **Store in `.mcp-vector-search/visualization/`** for easy access
3. **Detect staleness** via timestamp comparison with index_metadata.json
4. **Maintain lazy-loading** for expensive operations (relationships, callers)
5. **Integrate with serve command** for auto-regeneration if stale

This approach provides:
- **Fast visualization startup** (1-2 seconds instead of 10-15)
- **Reproducible builds** (same JSON every time)
- **Efficient storage** (~26 MB total)
- **Good UX** (lazy-loading for expensive queries)

The implementation reuses existing code (`_export_chunks`, `_export_kg_data`, `export_to_html`) and requires minimal changes to the current architecture.

---

## Next Steps

1. **Implement MVP** (Phase 1) in `cli.py`
2. **Test with large codebase** (10,000+ chunks)
3. **Add stale detection** (Phase 2)
4. **Update serve command** (Phase 3)
5. **Document new workflow** in README

**Estimated Implementation Time**: 4-6 hours for MVP, 8-10 hours for complete solution

---

*Research conducted by: Research Agent*
*Date: 2026-02-20*
*Project: mcp-vector-search*
*Version: 2.5.55*
