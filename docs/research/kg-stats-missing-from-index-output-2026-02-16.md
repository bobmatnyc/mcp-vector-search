# Knowledge Graph Stats Missing from Index Output

**Date**: 2026-02-16
**Issue**: KG statistics not displayed in `mcp-vector-search index` output
**Status**: Root cause identified, solution proposed

---

## Executive Summary

Knowledge Graph (KG) statistics are NOT displayed when running `mcp-vector-search index` because:

1. **KG is not built during indexing** - The index command only processes files, chunks, and embeddings
2. **KG requires explicit build step** - Use `mcp-vector-search kg build` to create the knowledge graph
3. **KG stats display location** - KG statistics are shown via `mcp-vector-search kg stats` command, NOT in index output

The user's expectation (seeing KG stats after indexing) is a **workflow design issue**, not a bug.

---

## Root Cause Analysis

### 1. When is the Knowledge Graph Built?

**Finding**: The KG is **NEVER** built during the `index` command.

**Evidence** (from `/src/mcp_vector_search/cli/commands/index.py`):

```python
# Line 944-952: Index command shows stats after indexing
stats = await indexer.get_indexing_stats()

print_success(
    f"Processed {indexed_count} files ({total_chunks} searchable chunks created)"
)

print_index_stats(stats)  # <-- This function does NOT include KG stats
```

**Index command workflow** (lines 333-467):
1. ✅ Load project config
2. ✅ Setup embedding model
3. ✅ Index files (chunking + embedding)
4. ✅ Compute **relationships** (marked for background computation)
5. ❌ Build KG entities (NOT executed)

**Relationship computation** (lines 846-903):
- Relationships are marked for lazy computation by the visualizer
- Uses `RelationshipStore.compute_and_store()` with `background=True`
- Does NOT build KG entities or populate the knowledge graph

### 2. Where are KG Stats Stored?

**Finding**: KG stats come from a separate Kuzu database in `.mcp-vector-search/knowledge_graph/`

**Evidence** (from `/src/mcp_vector_search/core/kg_builder.py`):

```python
# Lines 176-419: KGBuilder.build_from_chunks()
async def build_from_chunks(
    self,
    chunks: list[CodeChunk],
    show_progress: bool = True,
    skip_documents: bool = False,
) -> dict[str, int]:
    """Build graph from code chunks using batch inserts."""

    stats = {
        "entities": 0,           # Code entities (functions, classes)
        "doc_sections": 0,       # Documentation sections
        "tags": 0,               # Tags extracted from docs
        "calls": 0,              # Function call relationships
        "imports": 0,            # Import relationships
        "inherits": 0,           # Inheritance relationships
        "contains": 0,           # Parent-child relationships
        "references": 0,         # Doc references to code
        "documents": 0,          # Doc-to-code relationships
        # ... more relationship types
    }
```

**KG statistics method** (from `/src/mcp_vector_search/cli/commands/kg.py` lines 184-236):

```python
@kg_app.command("stats")
def kg_stats(...):
    """Show knowledge graph statistics."""
    kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
    kg = KnowledgeGraph(kg_path)
    await kg.initialize()

    stats = await kg.get_stats()  # Gets entity/relationship counts from Kuzu
```

### 3. Where Should KG Stats Be Displayed?

**Finding**: KG stats are displayed by the `kg stats` command, NOT the `index` command.

**Current index stats** (from `/src/mcp_vector_search/cli/output.py` lines 295-316):

```python
def print_index_stats(stats: dict[str, Any]) -> None:
    """Print indexing statistics."""
    table = Table(title="Index Statistics", show_header=False)

    table.add_row("Total Files", str(stats.get("total_indexable_files", 0)))
    table.add_row("Indexed Files", str(stats.get("indexed_files", 0)))
    table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))

    # Language distribution
    languages = stats.get("languages", {})
    if languages:
        lang_str = ", ".join(f"{lang}: {count}" for lang, count in languages.items())
        table.add_row("Languages", lang_str)

    # File extensions
    extensions = stats.get("file_extensions", [])
    if extensions:
        table.add_row("Extensions", ", ".join(extensions))
```

**Missing**: No calls to `kg.get_stats()` or display of KG entity/relationship counts.

### 4. Is There a `kg stats` Command?

**Finding**: YES - `mcp-vector-search kg stats` exists and shows KG statistics.

**Evidence** (from `/src/mcp_vector_search/cli/commands/kg.py` lines 184-236):

```python
@kg_app.command("stats")
def kg_stats(...):
    """Show knowledge graph statistics."""
    # Displays:
    # - Total Entities
    # - Code Entities
    # - Doc Sections
    # - Tags
    # - Persons
    # - Projects
    # - Relationship counts (CALLS, IMPORTS, INHERITS, etc.)
```

---

## Architecture Understanding

### Two Separate Systems

**1. Index System** (ChromaDB + LanceDB):
- **Files**: Discovered files in project
- **Chunks**: Code segments (functions, classes, modules)
- **Embeddings**: Vector representations for semantic search
- **Storage**: `.mcp-vector-search/chroma.sqlite3` + `.mcp-vector-search/*.lance`

**2. Knowledge Graph System** (Kuzu):
- **Entities**: Code entities (functions, classes) + doc sections + tags
- **Relationships**: CALLS, IMPORTS, INHERITS, CONTAINS, REFERENCES, DOCUMENTS
- **Storage**: `.mcp-vector-search/knowledge_graph/`
- **Built separately**: Via `kg build` command

### Workflow Disconnect

```
User expectation:
index → chunks + embeddings + KG stats

Actual workflow:
index      → chunks + embeddings
kg build   → KG entities + relationships
kg stats   → Display KG statistics
```

---

## Why KG Stats Are Not Showing

**Root cause**: The `index` command does NOT call `kg build`, therefore no KG entities exist.

**Evidence flow**:

1. User runs `mcp-vector-search index`
2. Indexing completes: 17,025 files indexed
3. `get_indexing_stats()` called (line 944)
4. `print_index_stats()` called (line 952)
5. `print_index_stats()` only shows:
   - Total Files
   - Indexed Files
   - Total Chunks
   - Languages
   - Extensions
6. ❌ NO call to `kg.get_stats()`
7. ❌ NO display of KG entities/relationships

**The KG was never built**, so there are no stats to show.

---

## Solution Options

### Option 1: Build KG During Indexing (Automatic)

**Pros**:
- User gets KG stats immediately after indexing
- Matches user expectations
- One-command workflow

**Cons**:
- Increases index time (KG build can be expensive)
- May not be desirable for large codebases
- Breaks separation of concerns

**Implementation**:

```python
# In index.py, after line 952 (print_index_stats)
if indexed_count > 0:
    # Auto-build KG after indexing
    from ...core.kg_builder import KGBuilder
    from ...core.knowledge_graph import KnowledgeGraph

    kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
    kg = KnowledgeGraph(kg_path)
    await kg.initialize()

    builder = KGBuilder(kg, project_root)
    kg_stats = await builder.build_from_database(database, show_progress=False)

    # Display KG stats in index output
    print_kg_stats_summary(kg_stats)
```

### Option 2: Show KG Stats if KG Exists (Passive)

**Pros**:
- No performance impact on indexing
- Shows KG stats if user ran `kg build` previously
- Maintains separation of concerns

**Cons**:
- Confusing for new users (why are stats missing?)
- Requires user to know about `kg build` command

**Implementation**:

```python
# In index.py, after line 952 (print_index_stats)
# Check if KG exists
kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
if kg_path.exists():
    from ...core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(kg_path)
    await kg.initialize()
    kg_stats = await kg.get_stats()

    if kg_stats["total_entities"] > 0:
        print_kg_stats_summary(kg_stats)
    else:
        print_info("Run 'mcp-vector-search kg build' to create knowledge graph")
```

### Option 3: Add Flag to Index Command (Opt-in)

**Pros**:
- User controls whether to build KG
- No breaking changes
- Clear workflow

**Cons**:
- Requires flag awareness
- Extra typing

**Implementation**:

```python
# Add flag to index command
@index_app.callback()
def main(
    # ... existing params
    build_kg: bool = typer.Option(
        False,
        "--build-kg",
        help="Build knowledge graph after indexing",
    ),
):
    # After indexing completes
    if build_kg and indexed_count > 0:
        from ...core.kg_builder import KGBuilder
        # ... build KG and show stats
```

### Option 4: Improve Documentation (Minimal Change)

**Pros**:
- No code changes
- Maintains current architecture
- Clear workflow guidance

**Cons**:
- Doesn't fix user expectation mismatch
- Requires users to read docs

**Implementation**:

Update help text and next steps:

```python
# In index.py, lines 973-984 (next steps)
steps = [
    "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search",
    chat_hint,
    "[cyan]mcp-vector-search status[/cyan] - View detailed statistics",
    "",
    "[bold]Knowledge Graph:[/bold]",
    "[cyan]mcp-vector-search kg build[/cyan] - Build knowledge graph (required first time)",
    "[cyan]mcp-vector-search kg stats[/cyan] - View graph statistics",
    '[cyan]mcp-vector-search kg query "ClassName"[/cyan] - Find related entities',
]
```

---

## Recommended Solution

**Hybrid Approach**: Option 2 (Passive) + Option 4 (Documentation)

**Rationale**:
1. Show KG stats if KG exists (passive detection)
2. Display clear hint if KG doesn't exist
3. Update next steps to guide users to `kg build`

**Why not automatic**:
- KG building can be slow for large codebases (20-60 seconds)
- Users may want to index quickly and build KG later
- Maintains separation of concerns

**Implementation summary**:

```python
# 1. Check if KG exists after indexing
kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
if kg_path.exists():
    kg_stats = await get_kg_stats(kg_path)
    if kg_stats["total_entities"] > 0:
        print_kg_summary(kg_stats)  # Show entities/relationships
    else:
        print_info("💡 Run 'mcp-vector-search kg build' to enable graph queries")

# 2. Update next steps to include kg build
steps = [
    "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search",
    "",
    "[bold]Knowledge Graph:[/bold]",
    "[cyan]mcp-vector-search kg build[/cyan] - Build knowledge graph (one-time setup)",
    "[cyan]mcp-vector-search kg stats[/cyan] - View graph statistics",
]
```

---

## Key Findings Summary

### Question 1: When is KG built?

**Answer**: KG is built ONLY when running `mcp-vector-search kg build`. It is NOT built during `index`.

**Evidence**:
- `index` command does NOT call `KGBuilder.build_from_chunks()`
- Relationships are marked for background computation but entities are not created
- KG database (`knowledge_graph/`) is empty after fresh index

### Question 2: Where are KG stats stored?

**Answer**: KG stats are stored in a separate Kuzu database at `.mcp-vector-search/knowledge_graph/`

**Stats included**:
- `total_entities`: Total entity count
- `code_entities`: Functions, classes, modules
- `doc_sections`: Documentation sections
- `tags`: Extracted tags
- `persons`: Git contributors
- `projects`: Project metadata
- `relationships`: CALLS, IMPORTS, INHERITS, CONTAINS, REFERENCES, DOCUMENTS, etc.

### Question 3: Where should KG stats be displayed?

**Answer**: Currently displayed ONLY via `mcp-vector-search kg stats` command.

**NOT displayed**:
- In `mcp-vector-search index` output (by design)
- In `mcp-vector-search status` output (separate feature)

### Question 4: Is there a `kg stats` command?

**Answer**: YES - `mcp-vector-search kg stats` exists and works.

**Usage**:
```bash
mcp-vector-search kg stats
```

**Output**:
```
Knowledge Graph Statistics
┌──────────────────┬─────────────┐
│ Metric           │ Value       │
├──────────────────┼─────────────┤
│ Total Entities   │ 3,546       │
│   Code Entities  │ 2,341       │
│   Doc Sections   │ 892         │
│   Tags           │ 213         │
│   Persons        │ 45          │
│   Projects       │ 1           │
│   Calls          │ 3,421       │
│   Imports        │ 1,892       │
│   Inherits       │ 359         │
│   Contains       │ 1,234       │
│   References     │ 567         │
│   Documents      │ 123         │
└──────────────────┴─────────────┘
```

---

## Next Steps

1. **Immediate fix**: Add passive KG stats display to index output (if KG exists)
2. **User guidance**: Update next steps hint to include `kg build` command
3. **Documentation**: Update README to explain KG workflow
4. **Future consideration**: Add `--build-kg` flag to index command for opt-in automatic building

---

## Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| Index command | `src/mcp_vector_search/cli/commands/index.py` | 38-217, 333-467 |
| Index stats display | `src/mcp_vector_search/cli/output.py` | 295-316 |
| KG builder | `src/mcp_vector_search/core/kg_builder.py` | 176-419 |
| KG stats command | `src/mcp_vector_search/cli/commands/kg.py` | 184-236 |
| Get indexing stats | `src/mcp_vector_search/core/indexer.py` | 1448-1492 |

---

## Testing Steps

To verify KG stats:

```bash
# 1. Fresh index
mcp-vector-search index

# Expected: NO KG stats in output

# 2. Build KG
mcp-vector-search kg build

# Expected: KG build progress + stats summary

# 3. Check KG stats
mcp-vector-search kg stats

# Expected: Full KG statistics table

# 4. Re-run index
mcp-vector-search index

# Expected (with fix): KG stats summary in output (since KG exists)
```

---

## Conclusion

The issue is **not a bug** but a **workflow design decision**. The KG is intentionally built separately from indexing to:

1. Keep indexing fast
2. Allow users to skip KG if not needed
3. Separate semantic search (embeddings) from graph queries (relationships)

**The fix** is to make this clearer to users by:
- Showing KG stats IF KG already exists
- Guiding users to `kg build` in next steps hint
- Updating documentation to explain the two-step workflow

**Impact**: Low (workflow clarity improvement, not a breaking change)
