# KG Feature Investigation: YAML Frontmatter, Tag Indices, Time Decay

**Date:** 2026-03-11
**Scope:** `/Users/masa/Projects/mcp-vector-search`
**Files examined:**
- `src/mcp_vector_search/core/kg_builder.py` (~4700 lines)
- `src/mcp_vector_search/core/knowledge_graph.py` (~4900 lines)
- `src/mcp_vector_search/mcp/kg_handlers.py`
- `src/mcp_vector_search/core/models.py`
- `src/mcp_vector_search/core/result_ranker.py`

---

## Feature 1: YAML Frontmatter Parsing

### Status: YES — Fully Supported

The KG reads YAML frontmatter from markdown/text documents during `_extract_doc_sections()`.

### How It Works

**`_extract_frontmatter(content: str) -> dict | None`**
Location: `kg_builder.py:2406`

```python
def _extract_frontmatter(self, content: str) -> dict | None:
    if not content.startswith("---"):
        return None
    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return None
    frontmatter_str = content[3 : end_match.start() + 3]
    return yaml.safe_load(frontmatter_str)
```

- Uses `yaml.safe_load` — correct and safe
- Triggered for every text/markdown chunk in `_extract_doc_sections()` (`kg_builder.py:1509`)
- Reads the **full file** from disk (not just the chunk) to ensure frontmatter is always captured even when the first chunk doesn't include the `---` block

**Fast path:** If `chunk.tags` is already populated (set during the chunking/indexing phase via `CodeChunk.tags`), the file read is skipped for tag extraction but the full file is still read to get `related` links.

### Fields Extracted from Frontmatter

| Field | Used For | Notes |
|-------|----------|-------|
| `tags` | Tag nodes + HAS_TAG edges | Primary field |
| `categories` | Tag nodes + HAS_TAG edges | Alias fallback |
| `keywords` | Tag nodes + HAS_TAG edges | Alias fallback |
| `labels` | Tag nodes + HAS_TAG edges | Alias fallback |
| `related` | LINKS_TO / RELATED_TO edges | Cross-doc references |

### Fields NOT Extracted from Frontmatter

| Field | Status | Notes |
|-------|--------|-------|
| `title` | NOT read from frontmatter | Document title comes from first H1 heading or filename stem |
| `date` / `created_at` | NOT read | No frontmatter date extraction |
| `author` | NOT read | Authors come from git blame only |
| `entities` | NOT read | No explicit entities field handling |
| `description` | NOT read | No description field handling |

### Code Locations

- Extraction: `kg_builder.py:1538–1590` (batch path), `kg_builder.py:2156–2204` (deprecated async path)
- `_extract_frontmatter`: `kg_builder.py:2406–2436`
- `CodeChunk.tags` field: `models.py:261–290`

---

## Feature 2: Entity/Tag Indices

### Status: PARTIAL — Tags indexed; dedicated entity/author/date indices absent

### What IS Indexed

**Tag nodes and HAS_TAG relationship:**
- Schema: `Tag(id STRING PRIMARY KEY, name STRING)` (`knowledge_graph.py:299–312`)
- Relationship: `HAS_TAG (FROM DocSection TO Tag, MANY_MANY)` (`knowledge_graph.py:416–430`)
- Tags sourced from frontmatter fields: `tags`, `categories`, `keywords`, `labels`
- Tags also auto-created for code block languages: `lang:python`, `lang:typescript`, etc. (via DEMONSTRATES edges)
- Tag nodes are queryable via Cypher: `MATCH (t:Tag) RETURN t.name`

**Document categorization (IA Topics):**
- Schema: `Topic(id STRING PRIMARY KEY, name STRING, parent_id STRING)` (`knowledge_graph.py:335–348`)
- Relationship: `HAS_TOPIC (FROM Document TO Topic, MANY_MANY)`
- Documents are auto-classified into 7 IA groups (Orientation, Guides & Tutorials, Architecture & Design, API Reference, Operations, Lifecycle, Testing) based on filename patterns (`_classify_document`, `kg_builder.py:1706`)
- The `doc_category` field on Document nodes is the primary index for this

**Author/Person index:**
- Schema: `Person(id, name, email_hash, commits_count, first_commit, last_commit)` (`knowledge_graph.py:522–538`)
- Relationships: `AUTHORED (Person -> CodeEntity)`, `MODIFIED (Person -> CodeEntity)`
- Authors come exclusively from git log, NOT from frontmatter `author` field

### What is MISSING

| Index Type | Status | Gap |
|------------|--------|-----|
| Frontmatter `author` -> Person node | NOT implemented | Would need to parse `author:` field and link to Person |
| Frontmatter `entities` -> CodeEntity lookup | NOT implemented | No `entities:` field handling |
| Frontmatter `date` / `created_at` | NOT implemented | Document nodes have `last_modified STRING` field but it is always set to `""` (empty string) |
| Tag-based search API | PARTIAL | Tags exist in KG but `kg_query` handler does not expose tag queries; no `find_by_tag` method |
| Explicit `title` frontmatter override | NOT implemented | Title always taken from H1 or filename |

### Critical Gap: `last_modified` Always Empty

`Document.last_modified` is defined in the dataclass and schema but never populated:

```python
# kg_builder.py:2074 — _build_document_nodes()
doc = Document(
    id=doc_id,
    file_path=file_path,
    title=title,
    doc_category=doc_category,
    word_count=word_count,
    section_count=len(sections),
    # last_modified=  <-- NEVER SET, defaults to None -> stored as ""
)
```

---

## Feature 3: Time Decay

### Status: NO — Not implemented anywhere in the KG

### Investigation Results

**KG query methods examined for time-based scoring:**

| Method | Location | Time decay? |
|--------|----------|-------------|
| `find_related()` | `knowledge_graph.py:2869` | No — pure graph traversal |
| `get_call_graph()` | `knowledge_graph.py:2928` | No — structural only |
| `get_inheritance_tree()` | `knowledge_graph.py:3360` | No — structural only |
| `get_document_ontology()` | `knowledge_graph.py:4672` | No — categorical grouping only |
| `get_ia_tree()` | `knowledge_graph.py:4822` | No — hierarchical only |
| `get_entity_history()` | `knowledge_graph.py:3310` | Returns commit timestamp as data, does NOT use it for ranking |
| `trace_execution_flow()` | `knowledge_graph.py` | No — BFS traversal by depth |

**Vector search result ranker (`result_ranker.py`):**
Scoring factors are: exact identifier match, partial identifier match, file name match, content word match, chunk type (function/class), file type (source vs test), path depth, boilerplate penalty, NLP entity match. No time-based component exists.

**`_compute_documents_score()` (`kg_builder.py:2460`):**
This scores doc-to-entity relevance (0.0–1.0) using: entity name in doc title (0.4), mention count (0.2), README proximity (0.3), contextual keywords (0.1). No recency component.

**Temporal data that exists but is unused for scoring:**
- `CodeEntity.commit_sha` — stored but only used for `kg_callers_at_commit` ancestor queries
- `DocSection.commit_sha` — stored, never queried for ranking
- `Person.first_commit` / `Person.last_commit` — stored, never used for decay
- `Commit.timestamp` — stored in Commit nodes but no decay query runs against it
- `Document.last_modified` — schema field exists but always empty (see above)

---

## Effort Estimates

### Feature 1 (YAML Frontmatter) — Already implemented
No work needed for basic parsing. Gaps to address:

| Enhancement | Effort | Notes |
|-------------|--------|-------|
| Read `title` from frontmatter (override H1) | 1–2h | Add `frontmatter.get("title")` before H1 scan |
| Read `date`/`created_at` from frontmatter → populate `Document.last_modified` | 2–4h | Parse ISO or date string, normalize to ISO, set on Document |
| Read `author` from frontmatter → link to Person node | 4–8h | Create Person if not exists, add AUTHORED edge |
| Read `entities` from frontmatter → explicit CodeEntity lookup | 4–8h | Fuzzy-match entity names, add REFERENCES edges |

### Feature 2 (Tag/Entity Indices) — Partially implemented

| Enhancement | Effort | Notes |
|-------------|--------|-------|
| Expose `find_by_tag` in `kg_query` handler | 2–4h | Cypher: `MATCH (t:Tag {name:$tag})<-[:HAS_TAG]-(s:DocSection)` |
| Populate `Document.last_modified` from filesystem `os.stat()` | 1–2h | `Path(file_path).stat().st_mtime` during `_build_document_nodes()` |
| Frontmatter `author` -> Person linkage | 4–8h | See above |

### Feature 3 (Time Decay) — Not implemented

Implementing true time decay in KG queries would require:

| Step | Effort | Notes |
|------|--------|-------|
| 1. Populate `Document.last_modified` reliably (filesystem stat or frontmatter date) | 1–2h | Prerequisite for all decay |
| 2. Add decay parameter to `get_document_ontology` / `find_related` | 4–8h | Parse stored timestamp strings, compute age in days, apply `score *= exp(-lambda * age_days)` |
| 3. Expose decay half-life as a configurable parameter | 2–4h | Add `decay_half_life_days` to config |
| 4. Apply decay to vector search results (not just KG) | 8–16h | Requires `last_modified` in LanceDB index, modify `result_ranker.py` scoring |

**Total estimated effort for full time decay:** 15–30h

**Simplest useful implementation (doc-level decay in ontology queries only):** ~6–10h if `last_modified` is populated first.

---

## Summary Table

| Feature | Supported | Key File | Key Gap |
|---------|-----------|----------|---------|
| YAML frontmatter parsing | YES | `kg_builder.py:2406` | `title`, `date`, `author`, `entities` fields not read |
| Tag indices (tags/categories/keywords/labels) | YES | `kg_builder.py:1554–1577`, schema `knowledge_graph.py:299` | No `find_by_tag` query API exposed |
| Author/entity indices from frontmatter | NO | — | Authors only from git log |
| `Document.last_modified` population | NO | `kg_builder.py:2074` | Field defined but never set |
| Time decay in KG queries | NO | — | No implementation anywhere |
| Time decay in vector search ranking | NO | `result_ranker.py` | No recency factor in scoring |
