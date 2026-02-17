# KG Missing Relationships - Fix Status

## Summary
Task: Enable missing KG relationships in sync build after subprocess isolation fixed threading issues.

## Completed ✅

### 1. HAS_TAG (0 → 8)
**Status**: FIXED
**Commits**: `819f51a` - fix: extract frontmatter from full file to handle chunked markdown

**Root Cause**:
Frontmatter was being extracted from `chunk.content` instead of full file. When files are chunked (e.g., ISS-0001.md has frontmatter spanning lines 1-37 but chunks were 1-12, 14-16, etc.), the YAML frontmatter was split across chunks, causing parsing to fail.

**Solution**:
1. Read full file for frontmatter: `Path(chunk.file_path).read_text()` instead of `chunk.content`
2. Attach tags only to first section (not every section in loop)
3. Handle files without headers (add tags even when early return happens)

**Results**:
- Tags: 21 (lang only) → 25 (includes 'bug', 'issue', 'indexing')
- HAS_TAG relationships: 0 → 8

**Files Modified**:
- `src/mcp_vector_search/core/kg_builder.py` (_extract_doc_sections method)

---

## In Progress 🚧

### 2. AUTHORED, PART_OF, Projects, Persons
**Status**: INCOMPLETE (needs KG sync methods)
**Commits**: `b18df5e` - wip: add git metadata extraction to sync KG build

**What Was Done**:
1. Added Phase 4 to `build_from_chunks_sync` for git metadata extraction
2. Created sync extraction methods:
   - `_extract_git_authors_sync()` - Extract persons from git log
   - `_extract_project_info_sync()` - Extract project from repo metadata
   - `_extract_authorship_sync()` - Create AUTHORED relationships
   - `_extract_part_of_sync()` - Create PART_OF relationships

**What's Missing**:
Need to add these methods to `src/mcp_vector_search/core/knowledge_graph.py`:

```python
def add_persons_batch_sync(self, persons: list[Person], batch_size: int = 500) -> int:
    """Batch insert persons using UNWIND (synchronous)."""
    total = 0
    for i in range(0, len(persons), batch_size):
        batch = persons[i : i + batch_size]
        params = [
            {
                "id": p.id,
                "name": p.name,
                "email_hash": p.email_hash,
                "commits_count": p.commits_count,
                "first_commit": p.first_commit,
                "last_commit": p.last_commit,
            }
            for p in batch
        ]

        try:
            self._execute_query(
                """
                UNWIND $batch AS p
                MERGE (n:Person {id: p.id})
                ON CREATE SET
                    n.name = p.name,
                    n.email_hash = p.email_hash,
                    n.commits_count = p.commits_count,
                    n.first_commit = p.first_commit,
                    n.last_commit = p.last_commit
                ON MATCH SET
                    n.name = p.name,
                    n.commits_count = p.commits_count,
                    n.last_commit = p.last_commit
                """,
                {"batch": params},
            )
            total += len(batch)
        except Exception as e:
            logger.error(f"Failed to insert persons batch: {e}")
            raise

    return total

def add_project_sync(self, project: Project) -> None:
    """Add or update a project (synchronous)."""
    if not self._initialized:
        raise RuntimeError(
            "KnowledgeGraph not initialized. Call initialize_sync() first."
        )

    try:
        self._execute_query(
            """
            MERGE (p:Project {id: $id})
            ON CREATE SET p.name = $name, p.description = $description, p.repo_url = $repo_url
            ON MATCH SET p.name = $name, p.description = $description, p.repo_url = $repo_url
            """,
            {
                "id": project.id,
                "name": project.name,
                "description": project.description or "",
                "repo_url": project.repo_url or "",
            },
        )
    except Exception as e:
        logger.error(f"Failed to add project {project.id}: {e}")
        raise

def add_authored_relationship_sync(
    self, person_id: str, entity_id: str, timestamp: str, commit_sha: str, lines: int
) -> None:
    """Add an AUTHORED relationship (synchronous)."""
    if not self._initialized:
        raise RuntimeError(
            "KnowledgeGraph not initialized. Call initialize_sync() first."
        )

    try:
        self._execute_query(
            """
            MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
            CREATE (p)-[r:AUTHORED]->(e)
            SET r.timestamp = $timestamp,
                r.commit_sha = $commit_sha,
                r.lines = $lines
            """,
            {
                "person_id": person_id,
                "entity_id": entity_id,
                "timestamp": timestamp,
                "commit_sha": commit_sha,
                "lines": lines,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to add AUTHORED relationship: {e}")

def add_part_of_batch_sync(
    self, entity_ids: list[str], project_id: str, batch_size: int = 500
) -> int:
    """Batch create PART_OF relationships (synchronous)."""
    if not self._initialized:
        raise RuntimeError(
            "KnowledgeGraph not initialized. Call initialize_sync() first."
        )

    total = 0
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i : i + batch_size]

        try:
            self._execute_query(
                """
                UNWIND $entity_ids AS entity_id
                MATCH (e:CodeEntity {id: entity_id}), (p:Project {id: $project_id})
                CREATE (e)-[:PART_OF]->(p)
                """,
                {"entity_ids": batch, "project_id": project_id},
            )
            total += len(batch)
        except Exception as e:
            logger.error(f"Failed to create PART_OF relationships batch: {e}")
            raise

    return total
```

**Testing After Implementation**:
```bash
mcp-vector-search kg build --force
mcp-vector-search kg stats
```

Expected results:
- Persons: 0 → ~5-10 (number of git contributors)
- Projects: 0 → 1
- AUTHORED: 0 → ~500-1000 (person → code entities)
- PART_OF: 0 → ~3374 (all code entities → project)

---

## Not Started ❌

### 3. DOCUMENTS
**Status**: SKIPPED (too expensive)
**Reason**: Requires semantic similarity computation between doc sections and code entities (O(n*m) complexity)

**Decision**: Keep skipped in sync mode. Can be enabled later with:
```bash
# In async mode only
mcp-vector-search kg build --async  # (if we implement async CLI flag)
```

**Alternative**: Implement incremental DOCUMENTS extraction in background worker.

---

### 4. IMPORTS
**Status**: TODO
**Reason**: Not yet implemented in either sync or async mode

**Current**: 0 (should extract from code)
**Expected**: ~500-1000 (import statements in Python files)

**Implementation Notes**:
- Extract import statements during code entity parsing
- Need to parse imports from tree-sitter AST
- Link to external packages or internal modules
- Example: `import numpy as np` → `(CodeEntity)-[:IMPORTS]->(Package:numpy)`

---

## Summary Stats

| Relationship | Before | After (Current) | Target |
|--------------|--------|-----------------|--------|
| HAS_TAG      | 0      | 8 ✅            | 8      |
| AUTHORED     | 0      | 0 🚧            | 500+   |
| PART_OF      | 0      | 0 🚧            | 3374   |
| DOCUMENTS    | 0      | 0 ⏸️            | (skip) |
| IMPORTS      | 0      | 0 ❌            | 500+   |
| Persons      | 0      | 0 🚧            | 5-10   |
| Projects     | 0      | 0 🚧            | 1      |

**Legend**:
- ✅ Complete
- 🚧 In progress (needs KG sync methods)
- ⏸️ Skipped (by design)
- ❌ Not started

---

## Next Steps

1. **Immediate**: Add 4 sync methods to `knowledge_graph.py` (see code above)
2. **Test**: Run `mcp-vector-search kg build --force` and verify counts
3. **Future**: Consider implementing IMPORTS extraction
4. **Optional**: Implement incremental DOCUMENTS extraction

---

## References

- **HAS_TAG Fix Commit**: 819f51a
- **WIP Git Extraction Commit**: b18df5e
- **Files Modified**:
  - `src/mcp_vector_search/core/kg_builder.py` (✅ complete)
  - `src/mcp_vector_search/core/knowledge_graph.py` (🚧 needs sync methods)
