# ChromaDB Usage and Baseline Storage Investigation

**Date**: December 11, 2024
**Investigator**: Research Agent
**Purpose**: Investigate current ChromaDB usage and evaluate potential for baseline storage consolidation

---

## Executive Summary

**Current Architecture**:
- ChromaDB stores code chunks with **extensive structural metrics** (20+ metadata fields)
- Baselines stored as **JSON files** in `~/.mcp-vector-search/baselines/`
- ChromaDB uses SQLite backend with normalized metadata storage
- Structural metrics **already integrated** into ChromaDB metadata schema

**Key Finding**: Baseline data **could** be stored in ChromaDB as a separate collection, but the current JSON-based approach has significant advantages for baseline-specific use cases.

**Recommendation**: **Keep current architecture** (JSON baselines) with potential ChromaDB collection for query optimization in future phases.

---

## Current ChromaDB Architecture

### 1. Collection Structure

**Single Collection**: `code_search`
- **Dimension**: 384 (sentence-transformers embedding size)
- **Total Chunks**: 10,498 indexed chunks (sample project)
- **Database Location**: `.mcp-vector-search/chroma.sqlite3`
- **Database Size**: 73 MB (for mcp-vector-search codebase)

### 2. Metadata Schema (Per Chunk)

ChromaDB stores **20 metadata fields** per code chunk:

#### Code Location Fields
```python
file_path: str              # Absolute path to source file
start_line: int             # Starting line number
end_line: int               # Ending line number
language: str               # Programming language (python, javascript, etc.)
```

#### Code Structure Fields
```python
chunk_type: str             # function, class, method, text, code
chunk_id: str               # Unique identifier (SHA-256 hash)
parent_chunk_id: str        # Hierarchical parent reference
child_chunk_ids: str        # JSON list of child IDs
chunk_depth: int            # Depth in code hierarchy
```

#### Function/Class Metadata
```python
function_name: str          # Function name (if applicable)
class_name: str             # Class name (if applicable)
docstring: str              # Docstring content
decorators: str             # JSON list of decorators
parameters: str             # JSON list of parameters
return_type: str            # Return type annotation
type_annotations: str       # JSON dict of type hints
```

#### Structural Metrics (Phase 1/2 Integration)
```python
cognitive_complexity: int   # Cognitive complexity score (0-100+)
cyclomatic_complexity: int  # Cyclomatic complexity (1-N)
max_nesting_depth: int      # Maximum nesting level (0-N)
parameter_count: int        # Number of function parameters
lines_of_code: int          # Lines of code in chunk
complexity_grade: str       # A-F grade based on cognitive complexity
code_smells: str            # JSON list of detected smells
smell_count: int            # Count of code smells
```

#### Monorepo Support
```python
subproject_name: str        # Subproject identifier (for monorepos)
subproject_path: str        # Relative path from monorepo root
```

### 3. Sample Record

```sql
embedding_id: /Users/masa/Projects/mcp-vector-search/CHANGELOG.md:3:6

metadata = {
    "file_path": "/Users/masa/Projects/mcp-vector-search/CHANGELOG.md",
    "start_line": 3,
    "end_line": 6,
    "language": "text",
    "chunk_type": "text",
    "cognitive_complexity": 0,
    "cyclomatic_complexity": 5,
    "complexity_grade": "A",
    "lines_of_code": 4,
    "code_smells": "[]",
    "smell_count": 0,
    "chunk_id": "0ab6feb4620e046b",
    "parent_chunk_id": "",
    "child_chunk_ids": "[]",
    ...
}
```

### 4. ChromaDB Backend Structure

ChromaDB uses **normalized SQLite schema**:

```sql
embeddings                 -- Core embedding storage
  ├── id (PK)
  ├── segment_id
  ├── embedding_id
  ├── seq_id
  └── created_at

embedding_metadata         -- Normalized key-value metadata
  ├── id (FK to embeddings)
  ├── key
  ├── string_value
  ├── int_value
  ├── float_value
  └── bool_value
```

**Indexes**:
- `embedding_metadata_int_value` (key, int_value)
- `embedding_metadata_float_value` (key, float_value)
- `embedding_metadata_string_value` (key, string_value)
- Full-text search index on string_value (trigram tokenizer)

**Efficiency**: 10,498 chunks × 20 metadata fields = **209,960 rows** in `embedding_metadata` table.

---

## Current Baseline Storage Architecture

### 1. Storage Location

**Primary Directory**: `~/.mcp-vector-search/baselines/`
- **File Format**: JSON (`.json`)
- **Naming Convention**: `{baseline_name}.json` (e.g., `main-branch.json`, `v1.2.0.json`)
- **Current Status**: **Empty** (no baselines created yet in sample project)

### 2. Baseline File Structure

**Top-Level Schema** (from `BaselineManager`):

```json
{
    "version": "1.0",
    "baseline_name": "main-branch",
    "created_at": "2025-12-11T15:30:00Z",
    "tool_version": "v0.18.0",
    "description": "Baseline before refactoring",

    "git_info": {
        "commit": "abc123def456...",
        "branch": "main",
        "remote": "origin"
    },

    "project": {
        "path": "/path/to/project",
        "file_count": 42,
        "function_count": 156,
        "class_count": 23
    },

    "aggregate_metrics": {
        "cognitive_complexity": {
            "sum": 1250,
            "avg": 8.0,
            "max": 47,
            "grade_distribution": {
                "A": 120,
                "B": 25,
                "C": 8,
                "D": 2,
                "F": 1
            }
        },
        "cyclomatic_complexity": {...},
        "nesting_depth": {...},
        "parameter_count": {...}
    },

    "files": {
        "src/core/indexer.py": {
            "file_path": "src/core/indexer.py",
            "total_lines": 500,
            "code_lines": 380,
            "comment_lines": 80,
            "blank_lines": 40,
            "function_count": 12,
            "class_count": 2,
            "method_count": 8,
            "total_complexity": 95,
            "avg_complexity": 7.9,
            "max_complexity": 23,
            "chunks": [
                {
                    "cognitive_complexity": 23,
                    "cyclomatic_complexity": 15,
                    "max_nesting_depth": 4,
                    "parameter_count": 5,
                    "lines_of_code": 120,
                    "smells": ["long_method", "too_many_parameters"],
                    "complexity_grade": "D"
                }
            ]
        }
    }
}
```

### 3. Design Decisions (from `baseline/manager.py`)

**Why JSON?**
- **Human readability**: Baselines are reference snapshots developers need to understand
- **Simplicity**: No schema migrations required
- **Portability**: Easy to share via git, email, or CI artifacts
- **Git-friendly**: Text-based format works well with version control

**Storage Strategy**:
- **Atomic writes**: Temp file + rename for data integrity
- **Git traceability**: Includes commit hash, branch, remote
- **Version tracking**: Tool version for compatibility validation
- **Performance**: Save O(n) ~50-100ms for 100 files, Load O(n) ~20-50ms

**Error Handling**:
- `BaselineNotFoundError`: Baseline doesn't exist
- `BaselineExistsError`: Baseline already exists (use `overwrite=True`)
- `BaselineCorruptedError`: JSON parsing failed or invalid structure
- `OSError`: Filesystem permission issues (propagated with clear message)

---

## ChromaDB vs JSON Baseline Storage: Analysis

### Option A: Current Architecture (JSON Files) ✅ CURRENT

**Pros**:
1. **Human-Readable**: Developers can open baselines in text editor
2. **Git-Friendly**: Baselines can be committed to repo for team sharing
3. **Portable**: Copy baseline files between machines/CI systems
4. **Simple**: No schema migrations, no database coupling
5. **Fast Load**: JSON deserialization ~20-50ms for 100 files
6. **Semantic Separation**: Baselines are reference snapshots, not search data
7. **Backup-Friendly**: Standard filesystem backups work
8. **Diff-Friendly**: Text diff tools work on JSON baselines

**Cons**:
1. **No Query Optimization**: Can't filter baselines by complexity range without loading
2. **Linear Search**: Finding specific files across baselines requires loading all
3. **Storage Duplication**: Same project metrics might be stored in both ChromaDB and baselines
4. **No Relational Queries**: Can't easily query "show all functions with complexity > 20 across 5 baselines"

**Use Cases Optimized For**:
- ✅ Baseline comparison (current vs. historical)
- ✅ CI/CD quality gates (compare PR against baseline)
- ✅ Team collaboration (share baseline snapshots)
- ✅ Manual inspection of historical metrics
- ✅ Backup and restore

### Option B: ChromaDB Collection for Baselines (Proposed Alternative)

**Architecture**:
```python
# New collection: "baselines"
collection = client.get_or_create_collection(
    name="baselines",
    metadata={"description": "Historical baseline snapshots"}
)

# Store each baseline as document
baseline_doc = {
    "id": f"baseline_{baseline_name}_{timestamp}",
    "baseline_name": "main-branch",
    "created_at": "2025-12-11T15:30:00Z",
    "git_commit": "abc123",
    "metrics_json": json.dumps(metrics_dict)  # Full metrics as JSON string
}
```

**Pros**:
1. **Query Optimization**: SQL indexes on complexity metrics
2. **Cross-Baseline Queries**: "Find all functions with complexity > 20 across all baselines"
3. **Unified Storage**: All project data in one ChromaDB database
4. **Vector Search**: Could search baselines by semantic similarity (future)
5. **Relational Joins**: Link baseline data to current index data
6. **Efficient Filtering**: ChromaDB metadata filtering for baseline queries

**Cons**:
1. **Not Human-Readable**: Binary SQLite database, requires tools to inspect
2. **Not Git-Friendly**: Can't commit baselines to repo easily
3. **Database Coupling**: Baselines tied to ChromaDB lifecycle
4. **Schema Migrations**: Need migration strategy for baseline schema changes
5. **Backup Complexity**: Must backup entire ChromaDB database
6. **Portability Loss**: Can't easily share individual baselines
7. **Semantic Mismatch**: Baselines are reference data, not embedding-based search data
8. **Increased Complexity**: More ChromaDB collections to manage

**Use Cases Optimized For**:
- ✅ Cross-baseline analytics queries
- ✅ Trend visualization dashboards
- ✅ Automated quality gate queries
- ⚠️  Team collaboration (less portable)
- ⚠️  Manual inspection (requires tools)
- ❌ Git-based baseline sharing

### Option C: Hybrid Approach (Future Phase 3)

**Architecture**:
- **Primary Storage**: JSON files (`~/.mcp-vector-search/baselines/`)
- **Query Cache**: ChromaDB collection (`baselines_cache`)
- **Synchronization**: Lazy load baseline JSON → ChromaDB on demand

**Workflow**:
```python
# Save baseline: Always write JSON
manager.save_baseline("main", metrics)
→ saves to ~/.mcp-vector-search/baselines/main.json

# Query baseline: Check cache first
results = baseline_query.filter(complexity > 20)
→ if baseline in ChromaDB cache: query ChromaDB
→ else: load JSON → populate cache → query ChromaDB

# Cache invalidation: Delete ChromaDB cache on baseline update
manager.save_baseline("main", metrics, overwrite=True)
→ deletes main.json
→ evicts main from ChromaDB cache
```

**Pros**:
1. **Best of Both Worlds**: Human-readable JSON + query optimization
2. **Git-Friendly**: JSON files can be committed
3. **Query Performance**: ChromaDB cache for complex queries
4. **Graceful Degradation**: Falls back to JSON if ChromaDB unavailable
5. **No Data Loss**: JSON is source of truth
6. **Incremental Adoption**: Add caching without breaking existing baselines

**Cons**:
1. **Complexity**: Two storage systems to maintain
2. **Synchronization**: Cache invalidation logic required
3. **Storage Overhead**: Data stored twice (JSON + ChromaDB)
4. **Development Effort**: Significant implementation work

---

## Recommendation

### Primary Recommendation: **Keep Current Architecture (JSON Baselines)**

**Rationale**:

1. **Current Phase**: Project is in Phase 2 (Quality Gates). Baseline comparison is a new feature (Issue #18: "Baseline comparison"). ChromaDB consolidation would add complexity before baseline usage patterns are established.

2. **Design Alignment**: Baselines are **reference snapshots** for comparison, not search data. ChromaDB is optimized for **embedding-based semantic search**, not tabular metric storage.

3. **User Experience**: Developers need to **inspect, share, and version-control baselines**. JSON files excel at this. ChromaDB would require custom tooling.

4. **Portability**: JSON baselines can be:
   - Committed to git (team-wide baselines)
   - Copied to CI/CD systems
   - Shared via email/Slack
   - Diffed with standard tools

5. **Performance**: JSON loading (~20-50ms) is **already fast enough** for baseline comparison use cases. Query optimization not needed yet.

6. **Simplicity**: Current implementation is clean, tested, and documented. ChromaDB consolidation would add:
   - Schema migration logic
   - Cache invalidation
   - Backup/restore complexity
   - Cross-platform sync issues

### Secondary Recommendation: **Evaluate Hybrid Approach in Phase 3**

**When to Reconsider**:
- **Phase 3: Cross-File Analysis** introduces SQLite metrics store (Issue #24)
- If users frequently query "show all functions with complexity > 20 across 5 baselines"
- If baseline files grow > 50 MB (current: ~100 KB per 100 files)
- If trend tracking (Issue #25) requires time-series queries

**Phase 3 Architecture Proposal**:
- **Primary Storage**: JSON files (backward compatible)
- **Metrics Store**: SQLite database (Issue #24) for time-series tracking
- **ChromaDB**: Code chunks only (embedding-based search)
- **Query Layer**: Unified API that queries appropriate backend

**Benefits**:
- Separates concerns: ChromaDB for search, SQLite for analytics, JSON for snapshots
- Maintains human-readable baselines
- Adds query optimization where needed (SQLite is better than ChromaDB for tabular analytics)
- Avoids ChromaDB schema complexity

---

## Technical Details: ChromaDB Metadata Storage

### Current Metadata Integration

**Database Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/chroma.sqlite3`

**Structural Metrics Already Stored in ChromaDB**:

```python
# From database.py: add_chunks() method
metadata = {
    # Original fields
    "file_path": str(chunk.file_path),
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    "language": chunk.language,
    "chunk_type": chunk.chunk_type,

    # Structural metrics (Phase 1/2)
    "cognitive_complexity": chunk.cognitive_complexity,  # NEW
    "cyclomatic_complexity": chunk.cyclomatic_complexity,  # NEW
    "max_nesting_depth": chunk.max_nesting_depth,  # NEW
    "parameter_count": chunk.parameter_count,  # NEW
    "lines_of_code": chunk.lines_of_code,  # NEW
    "complexity_grade": chunk.complexity_grade,  # NEW
    "code_smells": json.dumps(chunk.smells),  # NEW (JSON string)
    "smell_count": len(chunk.smells),  # NEW

    # Hierarchy fields
    "chunk_id": chunk.chunk_id or "",
    "parent_chunk_id": chunk.parent_chunk_id or "",
    "child_chunk_ids": json.dumps(chunk.child_chunk_ids or []),
    "chunk_depth": chunk.chunk_depth,

    # Enhanced metadata
    "decorators": json.dumps(chunk.decorators or []),
    "parameters": json.dumps(chunk.parameters or []),
    "return_type": chunk.return_type or "",
    "type_annotations": json.dumps(chunk.type_annotations or {}),

    # Monorepo support
    "subproject_name": chunk.subproject_name or "",
    "subproject_path": chunk.subproject_path or "",
}

# Integration point: metrics from collectors can be merged
if metrics and chunk.chunk_id and chunk.chunk_id in metrics:
    chunk_metrics = metrics[chunk.chunk_id]
    metadata.update(chunk_metrics)  # Merge ChunkMetrics.to_metadata()
```

**Key Observation**: ChromaDB **already stores** all structural metrics needed for baselines. The difference is:
- **ChromaDB**: Per-chunk metrics for **current codebase** (used for search filtering)
- **Baselines**: Aggregated project metrics for **historical snapshots** (used for comparison)

### Storage Overhead Analysis

**Current Storage** (JSON Baselines):
```
100 files × 10 functions each = 1000 functions
1000 functions × ~500 bytes JSON = 500 KB per baseline
10 baselines = 5 MB total

Growth: Linear with project size
Query: O(n) load entire baseline
```

**Alternative Storage** (ChromaDB Collection):
```
100 files × 10 functions = 1000 records in ChromaDB
1000 records × 20 metadata fields × 40 bytes = 800 KB (normalized)
+ Vector overhead (not needed for baselines): 1000 × 384 × 4 bytes = 1.5 MB
= 2.3 MB per baseline in ChromaDB

Growth: Linear but with 4x overhead (vector storage not needed)
Query: O(log n) with indexes
```

**Verdict**: JSON is **more storage-efficient** for baseline snapshots because:
- No vector embedding overhead (baselines don't need semantic search)
- No normalized table overhead (baseline data is naturally hierarchical)
- JSON compression is effective on repetitive metric data

---

## Implementation Notes

### Current Code Locations

**Baseline Management**:
- `src/mcp_vector_search/analysis/baseline/manager.py` (622 lines)
  - `BaselineManager`: Save/load/list/delete baselines
  - `GitInfo`: Git metadata tracking
  - `BaselineMetadata`: Baseline file metadata
  - Storage: `~/.mcp-vector-search/baselines/`

**ChromaDB Integration**:
- `src/mcp_vector_search/core/database.py` (1490 lines)
  - `ChromaVectorDatabase`: Main ChromaDB interface
  - `PooledChromaVectorDatabase`: Connection pooling
  - `add_chunks()`: Stores chunks with structural metrics
  - Storage: `.mcp-vector-search/chroma.sqlite3`

**Metrics Dataclasses**:
- `src/mcp_vector_search/analysis/metrics.py`
  - `ChunkMetrics`: Per-function metrics with `to_metadata()` method
  - `FileMetrics`: Per-file aggregates
  - `ProjectMetrics`: Project-wide aggregates

### Current Workflow

**Indexing with Metrics**:
```python
# Indexer collects structural metrics
metrics = collector.collect_metrics(chunks)

# Metrics added to ChromaDB as metadata
await db.add_chunks(chunks, metrics=metrics)
→ Stores in ChromaDB with metadata fields

# Baseline saved separately
await baseline_manager.save_baseline("main", project_metrics)
→ Stores in ~/.mcp-vector-search/baselines/main.json
```

**Baseline Comparison**:
```python
# Load baseline from JSON
baseline = baseline_manager.load_baseline("main")

# Analyze current codebase
current_metrics = analyzer.analyze_project(project_root)

# Compare
comparison = compare_metrics(baseline, current_metrics)
→ Shows added/removed/changed functions
→ Highlights complexity increases
```

### Potential ChromaDB Collection Schema (If Implemented)

**Collection Name**: `baselines`

**Document Structure**:
```python
{
    "id": "baseline_main_1702319400",  # baseline_{name}_{timestamp}
    "baseline_name": "main",
    "created_at": "2025-12-11T15:30:00Z",
    "git_commit": "abc123",
    "git_branch": "main",
    "tool_version": "v0.18.0",
    "project_path": "/path/to/project",
    "total_files": 100,
    "total_functions": 1000,

    # Store full metrics as JSON string (for human readability)
    "metrics_json": "{...full ProjectMetrics...}",

    # Extract key aggregates for filtering
    "avg_complexity": 8.5,
    "max_complexity": 47,
    "total_complexity": 8500,
    "grade_a_count": 800,
    "grade_f_count": 5,
}
```

**Query Examples**:
```python
# Find baselines where average complexity increased
results = collection.query(
    query_texts=[""],
    where={"$and": [
        {"avg_complexity": {"$gt": 10}},
        {"git_branch": "main"}
    ]}
)

# Find baselines with F-grade functions
results = collection.query(
    query_texts=[""],
    where={"grade_f_count": {"$gt": 0}}
)
```

**Challenges**:
- **No Full-Text Search**: Metrics are numeric, not semantic
- **No Embedding Benefit**: Baselines don't need similarity search
- **Complex JSON**: Full metrics stored as JSON string (defeats purpose of structured storage)
- **Limited Filtering**: ChromaDB metadata filtering is less powerful than SQL

---

## Next Steps

### Immediate (Phase 2 - Current)

1. **Keep JSON baselines** as implemented in Issue #18
2. **Document baseline workflow** for users:
   - How to create baselines
   - How to share baselines with team (git commit)
   - How to use baselines in CI/CD
3. **Add baseline examples** to documentation:
   - `docs/guides/baseline-comparison.md`
   - Example baseline files in `examples/baselines/`

### Phase 3 (Cross-File Analysis)

4. **Implement SQLite metrics store** (Issue #24):
   - Separate database: `.mcp-vector-search/metrics.db`
   - Time-series table: `metric_snapshots`
   - Supports trend tracking (Issue #25)
   - Keep baselines as JSON snapshots

5. **Evaluate hybrid approach**:
   - If users need cross-baseline queries
   - If baseline files grow > 50 MB
   - Consider SQLite (not ChromaDB) for metric analytics

### Future Optimization

6. **Baseline compression** (if file size becomes issue):
   - gzip JSON files (transparent to users)
   - Store only deltas from previous baseline
   - Archive old baselines to separate directory

7. **Baseline indexing** (if query performance needed):
   - SQLite FTS5 index on baseline JSON
   - Separate index file: `.mcp-vector-search/baseline_index.db`
   - Maintain JSON as source of truth

---

## Conclusion

The current architecture of storing baselines as JSON files in `~/.mcp-vector-search/baselines/` is **well-suited for the intended use case**. While ChromaDB could technically store baseline data, it would introduce complexity without significant benefits at this stage.

**Key Takeaway**: ChromaDB excels at **embedding-based semantic search** over code chunks. Baselines are **tabular metric snapshots** for comparison and trending. These are fundamentally different data models with different access patterns.

**Recommendation**: Maintain separation of concerns. Let ChromaDB handle search, SQLite handle metric analytics (Phase 3), and JSON handle baseline snapshots.

---

## Appendix: File Locations

### Analyzed Files

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py` (1490 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py` (298 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py` (622 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/metrics.py` (269+ lines)

### Database Files

- **ChromaDB**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/chroma.sqlite3` (73 MB)
- **Baselines**: `/Users/masa/.mcp-vector-search/baselines/` (empty, no baselines created yet)

### Documentation

- `/Users/masa/Projects/mcp-vector-search/docs/projects/structural-code-analysis.md`
- `/Users/masa/Projects/mcp-vector-search/docs/research/mcp-vector-search-structural-analysis-design.md`

---

**Investigation Complete** | Total Files Analyzed: 4 | Database Inspected: Yes | Recommendation: Keep JSON baselines
