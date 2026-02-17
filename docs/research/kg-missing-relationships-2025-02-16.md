# Knowledge Graph Missing Relationships Investigation

**Date:** 2025-02-16
**Investigator:** Research Agent
**Issue:** Multiple relationship types showing 0 counts despite having extraction code

## Executive Summary

The Knowledge Graph build shows 0 for several relationship types (DOCUMENTS, HAS_TAG, LINKS_TO, AUTHORED, MODIFIED, PART_OF) because the CLI uses a **synchronous subprocess implementation** (`build_from_chunks_sync`) that intentionally skips git-based relationship extraction and async-only features. The full-featured **async implementation** (`build_from_database`) with all relationship types is only used during background indexing and MCP server operations.

## Problem Statement

Current KG build output shows:
```
│ Imports       │     0 │  ← Skipped due to invalid module:[] targets
│ Documents     │     0 │  ← Should link doc sections to code entities
│ Has Tag       │     0 │  ← Should link entities to tags (21 tags exist!)
│ Links To      │     0 │  ← Should be extracted from markdown links
│ Authored      │     0 │  ← Should come from git blame/history
│ Modified      │     0 │  ← Should come from git blame/history
│ Part Of       │     0 │  ← Should link code to projects
```

## Root Cause Analysis

### Two Build Paths in Codebase

**Path 1: Synchronous CLI Build** (Used by `mcp-vector-search kg build`)
- **File:** `src/mcp_vector_search/cli/commands/_kg_subprocess.py`
- **Method:** `KGBuilder.build_from_chunks_sync()`
- **Location:** Lines 145-150 in `_kg_subprocess.py`
- **Purpose:** Thread-safe subprocess to avoid LanceDB/Kuzu conflicts
- **Limitations:**
  - No git history extraction (AUTHORED, MODIFIED)
  - No project relationships (PART_OF)
  - Skips DOCUMENTS extraction with warning message
  - Only processes chunks loaded from JSON file

**Path 2: Async Database Build** (Used during indexing and MCP operations)
- **Files:**
  - `src/mcp_vector_search/core/indexer.py` (line 251)
  - `src/mcp_vector_search/mcp/kg_handlers.py` (line 98)
- **Method:** `KGBuilder.build_from_database()`
- **Location:** Lines 3026-3217 in `kg_builder.py`
- **Purpose:** Full-featured build with database access and git integration
- **Features:**
  - Git author extraction via `_extract_git_authors()` (line 3117)
  - Project entity extraction via `_extract_project_info()` (line 3118)
  - Language/framework detection (line 3122)
  - AUTHORED relationships via `_extract_authorship_fast()` (line 3174)
  - PART_OF relationships in batch (lines 3176-3193)
  - Full DOCUMENTS relationship extraction

## Detailed Analysis by Relationship Type

### 1. DOCUMENTS (0 count)

**Expected Behavior:**
Links documentation sections to code entities they describe.

**Code Location:**
- **Extraction method:** `_extract_documents_relationships()` (lines 1501-1616 in `kg_builder.py`)
- **What it does:** Uses semantic similarity to match doc sections with code entities

**Why it's 0:**
Synchronous build explicitly skips with warning:
```python
# Line 546-555 in kg_builder.py (build_from_chunks_sync)
if not skip_documents and text_chunks and code_chunks:
    if progress_tracker:
        progress_tracker.warning(
            "Skipping DOCUMENTS extraction in synchronous mode"
        )
```

**Async Path:**
Called in `build_from_database()` at lines 705-709 and 777-781.

### 2. HAS_TAG (0 count)

**Expected Behavior:**
Links documentation sections to their frontmatter tags.

**Code Location:**
- **Extraction method:** `_extract_doc_sections()` (lines 1074-1080 in `kg_builder.py`)
- **What it does:** Extracts YAML frontmatter tags and creates HAS_TAG relationships

**Why it's 0:**
HAS_TAG relationships ARE collected in sync mode (lines 332, 650 in relationship dict), and ARE inserted (lines 491-538 in batch insertion loop). However, they're being **filtered out during validation**:

```python
# Lines 497-514 in kg_builder.py
for rel in rels:
    source_ok = rel.source_id in valid_entity_ids
    target_ok = rel.target_id in valid_entity_ids

    if source_ok and target_ok:
        valid_rels.append(rel)
    else:
        skipped += 1  # Relationships with invalid IDs are skipped
```

**Root Cause:** Tag IDs might not match between extraction and validation.

**Evidence:**
- Tags are inserted (line 394-398): `self.kg.add_tags_batch_sync(list(tags))`
- Relationships are created (line 1074-1080) with target: `f"tag:{tag}"`
- But validation might fail if tag format doesn't match

### 3. LINKS_TO (0 count)

**Expected Behavior:**
Links documentation sections that reference external URLs.

**Code Location:**
- **Extraction method:** `_extract_doc_sections()` (lines 1090-1097 in `kg_builder.py`)
- **What it does:** Extracts markdown links and creates LINKS_TO relationships

**Why it's 0:**
Same as HAS_TAG - collected but potentially filtered during validation. No external URL nodes are created, so `target_id` validation fails.

**Code:**
```python
# Lines 1090-1097 in kg_builder.py
for url in doc_urls:
    relationships["LINKS_TO"].append(
        CodeRelationship(
            source_id=section_id,
            target_id=url,  # Uses raw URL as target
            relationship_type="links_to",
        )
    )
```

**Issue:** No URL nodes exist in graph to validate against.

### 4. AUTHORED (0 count)

**Expected Behavior:**
Links Person entities to code entities they authored (from git history).

**Code Location:**
- **Extraction methods:**
  - `_extract_git_authors()` (lines 1629-1694) - Creates Person entities
  - `_extract_authorship_fast()` (lines 2844-2946) - Creates AUTHORED relationships
- **What it does:** Uses `git log` to map file authors to code entities

**Why it's 0:**
**Not called in synchronous build at all.** Only invoked in `build_from_database()` at line 3174:

```python
# Line 3174 in build_from_database (async only)
await self._extract_authorship_fast(entity_files, persons, stats)
```

**Synchronous build has no equivalent.**

### 5. MODIFIED (0 count)

**Expected Behavior:**
Tracks Person entities who modified code entities (from git history).

**Code Location:**
- **Extraction method:** `_extract_git_history()` (lines 2551-2842)
- **What it does:** Uses `git log` to track file modification history

**Why it's 0:**
**Not implemented in either build path.** No call sites found in codebase.

**Status:** Appears to be **dead code** - defined but never invoked.

### 6. PART_OF (0 count)

**Expected Behavior:**
Links code entities to Project entity.

**Code Location:**
- **Extraction method:** Batch creation in `build_from_database()` (lines 3176-3193)
- **What it does:** Links all CodeEntity nodes to Project node

**Why it's 0:**
**Not called in synchronous build.** Only in async `build_from_database()`:

```python
# Lines 3176-3193 in build_from_database
if project:
    logger.info("Creating PART_OF relationships...")
    part_of_count = await self.kg.add_part_of_batch(
        entity_ids, project.id
    )
    stats["part_of"] = part_of_count
```

**Requires Project entity which is only extracted in async path:**
```python
# Line 3118 in build_from_database
project = await self._extract_project_info(stats)
```

### 7. IMPORTS (0 count - Known Issue)

**Expected Behavior:**
Links code entities that import other modules.

**Why it's 0:**
Invalid target IDs - relationships use `module:[]` format which doesn't exist in graph.

**Status:** Known issue, not part of this investigation.

## Architecture Decision Trade-offs

### Why Two Build Paths?

**Synchronous Path Rationale:**
- **Safety:** Avoids LanceDB background threads that cause Kuzu segfaults
- **Isolation:** Runs in separate subprocess with no database connection
- **Speed:** Faster for basic relationship extraction
- **Simplicity:** No async complexity, easier debugging

**Async Path Rationale:**
- **Full Features:** Supports all relationship types
- **Git Integration:** Can run subprocess git commands in async context
- **Database Access:** Direct VectorDatabase queries for embeddings
- **Production Use:** Used during normal indexing workflow

### Current Usage

**CLI Users Run:**
```bash
mcp-vector-search kg build
```
- Uses **synchronous path**
- Gets partial graph (CALLS, CONTAINS, INHERITS, REFERENCES only)
- Skips DOCUMENTS, AUTHORED, PART_OF

**Background Indexing:**
```python
# In Indexer.build_graph()
await builder.build_from_database(
    self.database,
    show_progress=False,
    skip_documents=True,  # Fast mode
)
```
- Uses **async path**
- Gets full graph including git relationships
- Can optionally skip DOCUMENTS for speed

## Recommendations

### Option 1: Document Current Behavior (Low Effort)

**Action:**
Update CLI help text to clarify relationship coverage:

```bash
mcp-vector-search kg build
  Creates basic knowledge graph with code relationships (CALLS, CONTAINS, etc.)
  Note: Git history (AUTHORED, PART_OF) requires background indexing.
  Use 'mcp-vector-search index' for full graph.
```

**Pros:**
- No code changes
- Sets correct expectations
- Existing async path already works

**Cons:**
- CLI users don't get full graph
- Two-tier feature set

### Option 2: Add Sync Git Extraction (Medium Effort)

**Action:**
1. Create sync version of `_extract_git_authors()`
2. Create sync version of `_extract_authorship_fast()`
3. Create sync version of `_extract_project_info()`
4. Call them in `build_from_chunks_sync()` after relationship insertion

**Code Changes:**
```python
# In build_from_chunks_sync(), after Phase 3:

# PHASE 4: Extract git relationships (sync-safe)
if not skip_git:
    persons = self._extract_git_authors_sync(stats)
    project = self._extract_project_info_sync(stats)

    if persons:
        self._extract_authorship_fast_sync(code_entities, persons, stats)

    if project:
        self._create_part_of_relationships_sync(code_entities, project, stats)
```

**Pros:**
- CLI users get full graph
- Consistent behavior across paths
- No async complexity

**Cons:**
- Code duplication (sync + async versions)
- Git operations might be slow in subprocess
- Still skip DOCUMENTS (too expensive)

### Option 3: Unify Build Paths (High Effort)

**Action:**
Refactor to use single async build path with subprocess isolation strategy.

**Approach:**
1. Make subprocess run async build method
2. Use asyncio subprocess for git operations
3. Handle LanceDB threading issue differently (maybe lazy import)

**Pros:**
- Single source of truth
- All features in CLI
- Cleaner architecture

**Cons:**
- High refactoring cost
- Risk of reintroducing Kuzu segfaults
- Complex async subprocess handling

### Option 4: Fix HAS_TAG/LINKS_TO in Sync Build (Quick Win)

**Action:**
Fix tag ID validation to unblock these relationships in current sync build.

**Root Cause:**
Target validation fails for tags because:
1. Tags are created with ID: `tag:{name}`
2. Relationships reference: `f"tag:{tag}"`
3. But query returns different format?

**Investigation Needed:**
```python
# Debug validation at line 497-514
print(f"Tag IDs in Kuzu: {tag_ids}")
print(f"Relationship targets: {[r.target_id for r in rels if r.relationship_type == 'has_tag']}")
```

**If IDs match:** Bug is elsewhere (likely in batch insert logic)
**If IDs don't match:** Fix tag ID format consistency

**Pros:**
- Quick fix for 2 relationship types
- No major refactoring
- Improves current CLI experience

**Cons:**
- Doesn't address git relationships
- Doesn't fix DOCUMENTS

## Implementation Priority

**Immediate (Quick Wins):**
1. ✅ Document current behavior in CLI help
2. 🔧 Fix HAS_TAG validation issue
3. 🔧 Fix LINKS_TO (may require adding URL nodes)

**Short Term (Medium Effort):**
4. 🚀 Add sync git extraction for AUTHORED/PART_OF
5. 🗑️ Remove or fix MODIFIED (dead code cleanup)

**Long Term (Major Refactoring):**
6. 🏗️ Unify build paths (if justified by user needs)
7. 🔍 Implement proper DOCUMENTS in sync mode

## Testing Strategy

### Verification Steps

**For HAS_TAG Fix:**
```bash
# 1. Index a project with frontmatter tags
mcp-vector-search index --force

# 2. Build KG
mcp-vector-search kg build --force

# 3. Query tags
mcp-vector-search kg query "MATCH (t:Tag) RETURN count(t)"

# 4. Query HAS_TAG relationships
mcp-vector-search kg query "MATCH ()-[r:HAS_TAG]->() RETURN count(r)"

# Expected: count(r) > 0
```

**For Git Relationships:**
```bash
# After implementing sync git extraction:
mcp-vector-search kg build --force

mcp-vector-search kg query "MATCH (p:Person) RETURN count(p)"
# Expected: > 0

mcp-vector-search kg query "MATCH ()-[r:AUTHORED]->() RETURN count(r)"
# Expected: > 0

mcp-vector-search kg query "MATCH ()-[r:PART_OF]->() RETURN count(r)"
# Expected: equals CodeEntity count
```

### Regression Testing

Ensure async path still works:
```bash
# Full async build via indexing
mcp-vector-search index --force

# Check KG via MCP
# (Use MCP client to verify kg_status tool shows all relationships)
```

## Conclusion

The missing relationships are **by design in the synchronous CLI build**, not bugs. The codebase maintains two build paths with different feature sets:

- **Sync CLI:** Fast, safe, basic relationships only
- **Async Background:** Full-featured, includes git history and complex relationships

**Recommended Action:**
Implement **Option 4 (Fix HAS_TAG)** + **Option 2 (Add Sync Git)** for best CLI user experience without major refactoring.

**Priority Order:**
1. Fix HAS_TAG validation (should work in sync build)
2. Document sync vs async differences
3. Add sync git extraction for AUTHORED/PART_OF
4. Defer DOCUMENTS to async-only (too expensive)
5. Remove or fix MODIFIED (dead code)

---

## Appendix: Key Code References

### Build Path Entry Points

**Synchronous (CLI):**
- `src/mcp_vector_search/cli/commands/_kg_subprocess.py:145`
- Calls: `builder.build_from_chunks_sync()`

**Async (Indexing):**
- `src/mcp_vector_search/core/indexer.py:251`
- Calls: `await builder.build_from_database()`

### Relationship Extraction Methods

| Method | Lines | Sync? | Async? | Called? |
|--------|-------|-------|--------|---------|
| `_extract_code_entity()` | 791-975 | ✅ | ✅ | ✅ Both |
| `_extract_doc_sections()` | 977-1127 | ✅ | ✅ | ✅ Both |
| `_extract_documents_relationships()` | 1501-1616 | ❌ | ✅ | ⚠️ Async only |
| `_extract_git_authors()` | 1629-1694 | ❌ | ✅ | ⚠️ Async only |
| `_extract_project_info()` | 1696-1780 | ❌ | ✅ | ⚠️ Async only |
| `_extract_authorship_fast()` | 2844-2946 | ❌ | ✅ | ⚠️ Async only |
| `_extract_git_history()` | 2551-2842 | ❌ | ✅ | ❌ Never called |

### Relationship Types Coverage

**Sync Build (CLI):**
- ✅ CALLS - Works
- ✅ CONTAINS - Works
- ✅ INHERITS - Works
- ✅ REFERENCES - Works
- ✅ FOLLOWS - Works
- ✅ DEMONSTRATES - Works
- ❌ IMPORTS - Skipped (invalid targets)
- ❌ DOCUMENTS - Skipped (too expensive)
- ❌ HAS_TAG - Collected but filtered (validation bug)
- ❌ LINKS_TO - Collected but filtered (no URL nodes)
- ❌ AUTHORED - Not extracted (git required)
- ❌ MODIFIED - Not implemented
- ❌ PART_OF - Not extracted (project required)

**Async Build (Indexing):**
- ✅ All of the above
- ✅ AUTHORED - From git log
- ✅ PART_OF - To Project entity
- ⚠️ DOCUMENTS - Optional (skip_documents flag)
- ❌ MODIFIED - Still not implemented

---

**Investigation Complete**
**Next Steps:** Share findings with team and prioritize fixes based on user impact.
