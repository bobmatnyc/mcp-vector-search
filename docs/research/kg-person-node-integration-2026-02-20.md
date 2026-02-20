# KG Person Node Integration Research

**Date:** 2026-02-20
**Project:** mcp-vector-search
**Task:** Understand where Person nodes and AUTHORED/MODIFIED relationships should be integrated

---

## Executive Summary

The Knowledge Graph (KG) already has **complete infrastructure** for Person nodes and AUTHORED/MODIFIED relationships. The implementation exists in both schema and builder code. However, **the current implementation only tracks the MOST RECENT author** per file (AUTHORED relationship only, no MODIFIED relationships are created).

**Key Finding:** Person node creation and AUTHORED relationships are already implemented in `kg_builder.py`. The task is to **extend** the existing implementation to:
1. Track **ALL authors** per file (not just most recent)
2. Differentiate between **first author (AUTHORED)** and **subsequent authors (MODIFIED)**

---

## 1. KG Schema (Kuzu Database)

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/knowledge_graph.py`

### Person Node Schema

```python
# Line 417-426
CREATE NODE TABLE IF NOT EXISTS Person (
    id STRING PRIMARY KEY,
    name STRING,
    email_hash STRING,
    commits_count INT64 DEFAULT 0,
    first_commit STRING,
    last_commit STRING
)
```

**Fields:**
- `id`: Format `person:<email_hash>` (SHA256 hash of email)
- `name`: Author name from git log
- `email_hash`: SHA256 hash of email for privacy
- `commits_count`: Total commits by this person
- `first_commit`: ISO timestamp of first commit
- `last_commit`: ISO timestamp of last commit

### AUTHORED Relationship Schema

```python
# Line 447-462
CREATE REL TABLE IF NOT EXISTS AUTHORED (
    FROM Person TO CodeEntity,
    timestamp STRING,
    commit_sha STRING,
    lines_authored INT64 DEFAULT 0,
    MANY_MANY
)
```

**Relationship:** Person -[AUTHORED]-> CodeEntity
**Properties:**
- `timestamp`: ISO timestamp when file was created
- `commit_sha`: Git commit hash
- `lines_authored`: Number of lines contributed (currently set to 0)

### MODIFIED Relationship Schema

```python
# Line 464-479
CREATE REL TABLE IF NOT EXISTS MODIFIED (
    FROM Person TO CodeEntity,
    timestamp STRING,
    commit_sha STRING,
    lines_changed INT64 DEFAULT 0,
    MANY_MANY
)
```

**Relationship:** Person -[MODIFIED]-> CodeEntity
**Properties:**
- `timestamp`: ISO timestamp of modification
- `commit_sha`: Git commit hash
- `lines_changed`: Number of lines modified (currently set to 0)

**Schema Status:** ✅ **COMPLETE** - Both relationships are fully defined in schema

---

## 2. KG Builder Implementation

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/kg_builder.py`

### Main Build Pipeline

The KG build happens in **4 phases** (line 276-636 in `build_from_chunks_sync`):

```python
# PHASE 1: Collect all data from chunks
# - Extract CodeEntity nodes
# - Extract DocSection nodes
# - Extract Tags
# - Extract relationships (CALLS, IMPORTS, INHERITS, CONTAINS, etc.)

# PHASE 2: Insert entities into Kuzu
# - Batch insert CodeEntity nodes
# - Batch insert DocSection nodes
# - Batch insert Tags

# PHASE 3: Build relationships
# - Validate entity IDs exist in Kuzu
# - Insert relationships (CALLS, IMPORTS, etc.)

# PHASE 4: Extract git metadata (THIS IS WHERE PERSON NODES GO)
# Line 606-620:
persons = self._extract_git_authors_sync(stats, progress_tracker)
project = self._extract_project_info_sync(stats, progress_tracker)

if code_entities and persons:
    self._extract_authorship_sync(
        code_entities, persons, stats, progress_tracker
    )

if code_entities and project:
    self._extract_part_of_sync(code_entities, project, stats, progress_tracker)
```

**Integration Point:** Phase 4 - "Extracting git metadata" (line 601-620)

---

## 3. Person Extraction (ALREADY IMPLEMENTED)

**Method:** `_extract_git_authors_sync` (line 1861-1933)

### Current Implementation

```python
def _extract_git_authors_sync(self, stats, progress_tracker) -> dict[str, Person]:
    """Extract Person entities from git log (synchronous)."""
    persons = {}

    # Get ALL commits with author info
    result = subprocess.run(
        ["git", "log", "--format=%H|%an|%ae|%aI", "--all"],
        cwd=self.project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    for line in result.stdout.strip().split("\n"):
        commit_sha, name, email, timestamp = line.split("|")[:4]
        email_hash = self._hash_email(email)
        person_id = f"person:{email_hash}"

        if person_id not in persons:
            persons[person_id] = Person(
                id=person_id,
                name=name,
                email_hash=email_hash,
                commits_count=1,
                first_commit=timestamp,
                last_commit=timestamp,
            )
        else:
            persons[person_id].commits_count += 1
            # Update first/last commit timestamps

    # Batch insert into Kuzu
    if persons:
        person_list = list(persons.values())
        stats["persons"] = self.kg.add_persons_batch_sync(person_list)

    return persons
```

**Git Command Used:**
```bash
git log --format=%H|%an|%ae|%aI --all
```

**Output Format:**
```
<commit_hash>|<author_name>|<author_email>|<ISO_timestamp>
```

**Status:** ✅ **FULLY IMPLEMENTED** - Person nodes are created and inserted

---

## 4. AUTHORED Relationship Extraction (PARTIAL IMPLEMENTATION)

**Method:** `_extract_authorship_sync` (line 3248-3347)

### Current Implementation (FILE-LEVEL, MOST RECENT AUTHOR ONLY)

```python
def _extract_authorship_sync(
    self,
    code_entities: list[CodeEntity],
    persons: dict[str, Person],
    stats: dict,
    progress_tracker
) -> None:
    """Extract AUTHORED relationships using git log (synchronous)."""

    # STEP 1: Build file -> entity mapping
    file_to_entities = {}
    for entity in code_entities:
        rel_path = str(Path(entity.file_path).relative_to(self.project_root))
        if rel_path not in file_to_entities:
            file_to_entities[rel_path] = []
        file_to_entities[rel_path].append(entity.id)

    # STEP 2: Get file -> most recent author mapping
    file_author_map = {}  # file_path -> (person_id, timestamp, commit_sha)

    result = subprocess.run(
        ["git", "log", "--format=%H|%an|%ae|%aI", "--name-only", "-n", "500"],
        cwd=self.project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    current_commit = None
    current_email = None
    current_time = None

    for line in result.stdout.split("\n"):
        if "|" in line:
            # Commit line
            parts = line.split("|")
            current_commit = parts[0]
            current_email = parts[2]
            current_time = parts[3]
        elif line.strip() and current_email:
            # File line
            file_path = line.strip()
            if file_path not in file_author_map:  # ⚠️ ONLY FIRST (MOST RECENT)
                email_hash = self._hash_email(current_email)
                person_id = f"person:{email_hash}"
                file_author_map[file_path] = (person_id, current_time, current_commit)

    # STEP 3: Create AUTHORED relationships
    for file_path, entity_ids in file_to_entities.items():
        if file_path in file_author_map:
            person_id, timestamp, commit_sha = file_author_map[file_path]

            if person_id in persons:
                for entity_id in entity_ids[:5]:  # Limit 5 entities per file
                    self.kg.add_authored_relationship_sync(
                        person_id, entity_id, timestamp, commit_sha, 0
                    )
                    authored_count += 1

    stats["authored"] = authored_count
```

**Git Command Used:**
```bash
git log --format=%H|%an|%ae|%aI --name-only -n 500
```

**Limitation:**
- Only processes **last 500 commits** (`-n 500`)
- Only captures **MOST RECENT author** per file (line 3313: `if file_path not in file_author_map`)
- Does **NOT create MODIFIED relationships**

**Status:** ⚠️ **PARTIAL** - Only tracks most recent author, not all contributors

---

## 5. MODIFIED Relationship Extraction (NOT IMPLEMENTED)

**Current Status:** ❌ **NO IMPLEMENTATION**

The `MODIFIED` relationship is:
- Defined in schema ✅
- Has helper methods in `knowledge_graph.py` ✅
- **NOT called anywhere in kg_builder.py** ❌

**Methods Available (Not Used):**
- `knowledge_graph.py` line 918-976: `async def add_modified_relationship()`
- No sync version exists (would need to be added)

---

## 6. Data Flow Diagram

```
chunks.lance (CodeChunk)
    |
    | file_path, start_line, end_line, content
    v
KGBuilder.build_from_chunks_sync()
    |
    +-- PHASE 1: Extract entities from chunks
    |       └── code_entities: list[CodeEntity]
    |
    +-- PHASE 2: Insert entities into Kuzu
    |       └── CodeEntity nodes created
    |
    +-- PHASE 3: Build code relationships
    |       └── CALLS, IMPORTS, INHERITS, CONTAINS
    |
    +-- PHASE 4: Extract git metadata ⭐ PERSON NODE INTEGRATION POINT
            |
            +-- _extract_git_authors_sync()
            |       |
            |       | git log --format=%H|%an|%ae|%aI --all
            |       |
            |       └── Person nodes created (person:<email_hash>)
            |
            +-- _extract_authorship_sync()
                    |
                    | git log --format=%H|%an|%ae|%aI --name-only -n 500
                    |
                    | For each file:
                    |   - Get MOST RECENT author from git log
                    |   - Create Person -[AUTHORED]-> CodeEntity
                    |
                    └── AUTHORED relationships created
                            (ONLY for most recent author per file)
```

---

## 7. Chunks.lance Schema

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py`

### CodeChunk Fields (Relevant to Person Extraction)

```python
@dataclass
class CodeChunk:
    file_path: Path              # ✅ Available for git log
    start_line: int              # ✅ Available (not used currently)
    end_line: int                # ✅ Available (not used currently)

    # Git blame metadata (ALREADY IN SCHEMA!)
    last_author: str | None      # ⚠️ NOT POPULATED during indexing
    last_modified: str | None    # ⚠️ NOT POPULATED during indexing
    commit_hash: str | None      # ⚠️ NOT POPULATED during indexing
```

**Status:**
- Fields exist in schema ✅
- Fields are **NOT populated** during indexing ❌
- Would need to add git blame extraction to chunking process

---

## 8. Recommended Git Commands for Implementation

### Current Implementation (File-Level)

```bash
# Get most recent author per file (current implementation)
git log --format=%H|%an|%ae|%aI --name-only -n 500
```

**Pros:** Fast (single command for 500 commits)
**Cons:** Only gets most recent author per file

### Proposed Enhancement (All Authors per File)

#### Option A: Get ALL authors per file (chronological)

```bash
# For each file: get all commits that modified it
git log --format=%H|%aN|%aI --follow -- <file_path>
```

**Output:**
```
<commit_hash>|<author_name>|<timestamp>
<commit_hash>|<author_name>|<timestamp>
...
```

**Strategy:**
1. First commit = AUTHORED relationship
2. Subsequent commits = MODIFIED relationships

**Pros:**
- Complete authorship history
- Chronological order (first = author, rest = modifiers)
- Handles file renames (`--follow`)

**Cons:**
- One git command per file (slower for large repos)
- Would need batching/parallelization

#### Option B: Batch approach with --name-only (faster)

```bash
# Get ALL commits (remove -n 500 limit)
git log --format=%H|%an|%ae|%aI --name-only --all
```

**Modified parsing logic:**
```python
# For each file, collect ALL authors (not just first)
file_authors = {}  # file_path -> list[(person_id, timestamp, commit_sha)]

for line in result.stdout.split("\n"):
    if "|" in line:
        # Commit line
        current_commit, current_name, current_email, current_time = line.split("|")
    elif line.strip() and current_email:
        # File line
        file_path = line.strip()
        if file_path not in file_authors:
            file_authors[file_path] = []

        # ⭐ KEY CHANGE: Collect ALL authors (not just first)
        file_authors[file_path].append((person_id, current_time, current_commit))

# After collecting all:
for file_path, authors in file_authors.items():
    # Sort by timestamp (oldest first)
    authors.sort(key=lambda x: x[1])

    first_author = authors[0]  # AUTHORED
    subsequent_authors = authors[1:]  # MODIFIED

    # Create AUTHORED relationship for first author
    self.kg.add_authored_relationship_sync(
        first_author[0], entity_id, first_author[1], first_author[2], 0
    )

    # Create MODIFIED relationships for other authors
    for person_id, timestamp, commit_sha in subsequent_authors:
        self.kg.add_modified_relationship_sync(
            person_id, entity_id, timestamp, commit_sha, 0
        )
```

**Pros:**
- Single git command (very fast)
- Gets complete history

**Cons:**
- Large output for repos with many commits
- Memory usage for large repos

#### Option C: Hybrid approach (recommended)

```bash
# Step 1: Fast batch collection (current approach)
git log --format=%H|%an|%ae|%aI --name-only --all

# Step 2: For files with >1 author, get detailed history
git log --format=%H|%aN|%aI --follow -- <file_path>
```

**Strategy:**
1. First pass: Identify files with multiple authors (batch)
2. Second pass: Get detailed history only for multi-author files
3. Single-author files: Create AUTHORED relationship only

**Pros:**
- Fast for most files (single command)
- Detailed history only where needed

---

## 9. Recommended Integration Point

### Exact Location in kg_builder.py

**Method:** `_extract_authorship_sync` (line 3248-3347)

**Modification Required:**

```python
# CURRENT CODE (line 3310-3321):
elif line.strip() and current_email:
    file_path = line.strip()
    if file_path not in file_author_map:  # ⚠️ ONLY FIRST
        email_hash = self._hash_email(current_email)
        person_id = f"person:{email_hash}"
        file_author_map[file_path] = (person_id, current_time, current_commit)

# PROPOSED CHANGE:
elif line.strip() and current_email:
    file_path = line.strip()
    email_hash = self._hash_email(current_email)
    person_id = f"person:{email_hash}"

    # ⭐ NEW: Collect ALL authors per file
    if file_path not in file_author_map:
        file_author_map[file_path] = []
    file_author_map[file_path].append((person_id, current_time, current_commit))
```

**Then modify relationship creation (line 3330-3341):**

```python
# CURRENT CODE:
for file_path, entity_ids in file_to_entities.items():
    if file_path in file_author_map:
        person_id, timestamp, commit_sha = file_author_map[file_path]

        if person_id in persons:
            for entity_id in entity_ids[:5]:
                self.kg.add_authored_relationship_sync(
                    person_id, entity_id, timestamp, commit_sha, 0
                )

# PROPOSED CHANGE:
for file_path, entity_ids in file_to_entities.items():
    if file_path in file_author_map:
        authors = file_author_map[file_path]  # List of (person_id, timestamp, sha)

        # Sort by timestamp (oldest first)
        authors.sort(key=lambda x: x[1])

        # First author = AUTHORED
        first_author = authors[0]
        if first_author[0] in persons:
            for entity_id in entity_ids[:5]:
                self.kg.add_authored_relationship_sync(
                    first_author[0], entity_id, first_author[1], first_author[2], 0
                )

        # Subsequent authors = MODIFIED
        for person_id, timestamp, commit_sha in authors[1:]:
            if person_id in persons:
                for entity_id in entity_ids[:5]:
                    # ⚠️ Need to add sync version of add_modified_relationship
                    self.kg.add_modified_relationship_sync(
                        person_id, entity_id, timestamp, commit_sha, 0
                    )
```

### New Method Required

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/knowledge_graph.py`

**Add sync version of add_modified_relationship:**

```python
def add_modified_relationship_sync(
    self,
    person_id: str,
    entity_id: str,
    timestamp: str,
    commit_sha: str,
    lines_changed: int,
) -> None:
    """Add a MODIFIED relationship (synchronous).

    Args:
        person_id: ID of the Person node
        entity_id: ID of the CodeEntity node
        timestamp: ISO timestamp of modification
        commit_sha: Git commit hash
        lines_changed: Number of lines changed
    """
    try:
        self._execute_query(
            """
            MATCH (p:Person {id: $person_id}), (e:CodeEntity {id: $entity_id})
            CREATE (p)-[r:MODIFIED]->(e)
            SET r.timestamp = $timestamp,
                r.commit_sha = $commit_sha,
                r.lines_changed = $lines
            """,
            {
                "person_id": person_id,
                "entity_id": entity_id,
                "timestamp": timestamp,
                "commit_sha": commit_sha,
                "lines": lines_changed,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to add MODIFIED relationship: {e}")
```

---

## 10. Summary of Findings

### ✅ What Exists

1. **Person Node Schema** - Fully defined in Kuzu
2. **AUTHORED Relationship Schema** - Fully defined
3. **MODIFIED Relationship Schema** - Fully defined
4. **Person Node Creation** - Implemented in `_extract_git_authors_sync`
5. **AUTHORED Relationship Creation** - Implemented in `_extract_authorship_sync` (file-level, most recent author only)
6. **CodeChunk git fields** - Defined in models.py (not populated)

### ❌ What's Missing

1. **Tracking ALL authors per file** - Current implementation only tracks most recent
2. **MODIFIED relationship creation** - Schema exists but no sync method or caller
3. **Differentiation between first author and modifiers** - All treated as AUTHORED currently
4. **Complete git history** - Limited to 500 commits (`-n 500`)

### 🎯 Recommended Implementation

**Minimal Change (File-Level Authorship):**

1. Remove `-n 500` limit from git log command (line 3287)
2. Change `file_author_map` from single author to list of authors (line 3282)
3. Sort authors by timestamp and differentiate first (AUTHORED) vs rest (MODIFIED)
4. Add `add_modified_relationship_sync` method to `knowledge_graph.py`
5. Call sync MODIFIED method for non-first authors

**Estimated Impact:**
- ~50 lines of code changes in `kg_builder.py`
- ~20 lines for new sync method in `knowledge_graph.py`
- No schema changes required
- No changes to CLI or database structure

**Performance Considerations:**
- Current: 1 git command (500 commits)
- Proposed: 1 git command (ALL commits)
- For large repos: May need to handle large git log output (e.g., stream processing)

---

## 11. Alternative: Chunk-Level Git Blame (More Accurate)

If we want **line-level attribution** (who wrote specific lines in each chunk):

**Use existing:** `src/mcp_vector_search/core/git_blame.py`

```python
class GitBlameCache:
    def get_blame_for_range(
        self, file_path: Path, start_line: int, end_line: int
    ) -> BlameInfo | None:
        """Get git blame info for a line range."""
```

**Integration in chunking process:**
```python
# In chunker, for each chunk:
blame_cache = GitBlameCache(project_root)
blame = blame_cache.get_blame_for_range(file_path, start_line, end_line)

chunk.last_author = blame.author
chunk.last_modified = blame.timestamp
chunk.commit_hash = blame.commit_hash
```

**Then in KG builder:**
```python
# Use chunk.last_author to create Person nodes and relationships
# More accurate but slower (one git blame per file during indexing)
```

---

## 12. Conclusion

**The KG infrastructure for Person nodes is COMPLETE.** The task is to enhance the existing `_extract_authorship_sync` method to:

1. Track ALL authors per file (not just most recent)
2. Create AUTHORED relationships for first author
3. Create MODIFIED relationships for subsequent authors
4. Add missing `add_modified_relationship_sync` method

**Recommended approach:** File-level git log with complete history (Option B from Section 8).

**Minimal code changes required:** ~70 lines across 2 files.

**No schema changes needed:** All tables and relationships already exist.
