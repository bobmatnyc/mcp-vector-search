# Document Ontology Feasibility Research
**Date:** 2026-02-24
**Project:** mcp-vector-search
**Scope:** Can the existing infrastructure support a document ontology, and what needs to be built?

---

## Executive Summary

mcp-vector-search already has substantial infrastructure that maps cleanly onto document ontology
requirements. A document ontology is achievable with moderate effort, largely as an extension of
existing systems rather than a greenfield build. The clearest path is to extend the Kuzu KG schema
with document-level node types and populate them during the existing `kg build` pipeline.

---

## 1. What Already Exists

### 1.1 Kuzu Knowledge Graph — Core Infrastructure

**File:** `src/mcp_vector_search/core/knowledge_graph.py`
**Builder:** `src/mcp_vector_search/core/kg_builder.py`

The KG is a fully operational Kuzu (embedded graph database) instance with the following
existing node types:

| Node Type | Key Fields | Relevance to Doc Ontology |
|-----------|-----------|---------------------------|
| `CodeEntity` | id, name, entity_type, file_path | Foundation — files are already nodes |
| `DocSection` | id, name, file_path, level, line_start, line_end, doc_type | **Direct precursor** to document nodes |
| `Tag` | id, name | Topic taxonomy already exists |
| `Person` | id, name, email_hash, commits_count | Authorship tracking |
| `Project` | id, name, description, repo_url | Top-level project node |
| `Repository` | id, name, url, default_branch | Version control context |
| `ProgrammingLanguage` | id, name, version, file_extensions | Technology taxonomy |
| `ProgrammingFramework` | id, name, version, category | Framework taxonomy |

Existing relationship types in Kuzu:

| Relationship | From → To | Ontology Use |
|-------------|-----------|--------------|
| `CALLS` | CodeEntity → CodeEntity | Code dependency |
| `IMPORTS` | CodeEntity → CodeEntity | Code dependency |
| `INHERITS` | CodeEntity → CodeEntity | Code hierarchy |
| `CONTAINS` | CodeEntity → CodeEntity | Structural hierarchy |
| `FOLLOWS` | DocSection → DocSection | Section ordering (sequential structure) |
| `REFERENCES` | DocSection → CodeEntity | Docs reference code |
| `DOCUMENTS` | DocSection → CodeEntity | Docs document code |
| `HAS_TAG` | DocSection → Tag | Topic taxonomy |
| `DEMONSTRATES` | DocSection → Tag | Language/tech demonstration |
| `LINKS_TO` | DocSection → DocSection | Cross-document links |
| `AUTHORED` | Person → CodeEntity | Authorship |
| `MODIFIED` | Person → CodeEntity | Change tracking |
| `PART_OF` | CodeEntity → Project | Project membership |
| `WRITTEN_IN` | CodeEntity → ProgrammingLanguage | Language tagging |
| `USES_FRAMEWORK` | CodeEntity → ProgrammingFramework | Framework tagging |

**Critical finding:** `DocSection` already exists with `doc_type` field, `level` (markdown heading
level 1-6), and `file_path`. It is already populated during `kg build` for `.md` files. The
`FOLLOWS`, `HAS_TAG`, `DEMONSTRATES`, `LINKS_TO`, and `REFERENCES` relationships are all built.

### 1.2 Text Parser — Markdown Structure Extraction

**File:** `src/mcp_vector_search/parsers/text.py`

The `TextParser` handles `.txt`, `.md`, and `.markdown` files. It:
- Extracts YAML frontmatter (including `tags`, `categories`, `keywords`, `labels` fields)
- Performs paragraph-based chunking with fallback to line-based chunking
- Attaches frontmatter tags to all chunks from a given file
- Does NOT currently extract heading hierarchy or section structure at parse time

**Gap:** The parser does not preserve heading-level structure as chunk metadata. Heading extraction
happens later in `kg_builder.py` via `_extract_headers()`, which is only invoked during KG
build — not indexing.

### 1.3 KGBuilder — Document Section Pipeline

**File:** `src/mcp_vector_search/core/kg_builder.py`

The `_extract_doc_sections()` method (line 1165) already:
- Extracts markdown headers via `_extract_headers(chunk.content, chunk.start_line)`
- Creates `DocSection` nodes from each header (with `level`, `line_start`, `line_end`)
- Builds `FOLLOWS` relationships between sequential sections
- Extracts `related:` frontmatter for `LINKS_TO` relationships
- Extracts backtick code references for `REFERENCES` relationships
- Extracts code block languages for `DEMONSTRATES` relationships
- Uses `NLPExtractor.nlp_code_refs` if available on the chunk

**Gap:** There is no `Document` (file-level) node type — only `DocSection` (heading-level) nodes.
The file path is stored as a field on each `DocSection`, but there is no first-class document
node that aggregates sections. This is the primary structural gap for a document ontology.

### 1.4 Wiki Generator — Concept Ontology

**File:** `src/mcp_vector_search/core/wiki.py`

The `WikiGenerator` is a separate, parallel ontology system that:
- Extracts concept names from chunk metadata (function names, class names, directory names,
  docstring keywords)
- Uses LLM-based semantic grouping into 5–10 top-level categories
- Produces a `WikiOntology` with `WikiConcept` nodes (hierarchical: root categories → child
  concepts)
- Each concept references the chunk IDs where it appears (frequency tracking)
- Supports flat (no-LLM) fallback and file-based caching with TTL

**Relevance:** This is the closest existing system to a document ontology's "topics and concepts"
dimension. It operates on code concepts (function names, classes), not document-level topics. It
is not connected to the Kuzu KG.

### 1.5 NLP Extractor — Entity Extraction

**File:** `src/mcp_vector_search/core/nlp_extractor.py`

`NLPExtractor` runs on docstrings during parsing (`BaseParser._create_chunk`) and produces:
- `nlp_keywords`: YAKE-extracted keywords (or simple word frequency fallback)
- `nlp_code_refs`: Backtick `` `code` `` references
- `nlp_technical_terms`: CamelCase, ACRONYMS, `snake_case` patterns
- `action_verbs`: Returns, Raises, Creates, etc.

These are stored on `CodeChunk` and available in KGBuilder for relationship extraction. The same
extractor could be applied to document content for topic/concept extraction.

### 1.6 Story Module — Document Catalog

**Files:** `src/mcp_vector_search/story/extractor.py`, `src/mcp_vector_search/story/models.py`

The `StoryExtractor._extract_docs()` method already scans the filesystem for documentation files:
- Root-level docs: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `DESIGN.md`, `ROADMAP.md`
- `docs/**/*.md` (recursive)
- Any `.md` files with "design", "spec", "architecture", "proposal" in name

The `DocReference` model captures: `path`, `title` (from first `# heading`), `word_count`,
`last_modified`.

**Gap:** This is informational only — `DocReference` objects are used for story narrative
generation and are not persisted to the KG.

### 1.7 Vector Embeddings — Semantic Clustering Foundation

**File:** `src/mcp_vector_search/core/vectors_backend.py`

LanceDB stores embeddings for all indexed chunks. The backend provides:
- `search()` — cosine similarity search with IVF_PQ approximate nearest neighbor index
- `get_chunk_vector()` / `get_chunk_vectors_batch()` — vector retrieval by chunk ID
- `search_by_file()` — file-scoped search

**Clustering potential:** LanceDB does not natively support clustering algorithms (k-means,
HDBSCAN). Vector clustering would require pulling embeddings into numpy/scikit-learn and running
clustering there. This is feasible but is not a built-in feature. The vector retrieval API
(`get_chunk_vectors_batch`) provides the data needed to run external clustering.

---

## 2. What Is Missing (Specific Gaps)

### Gap 1: No Document-Level Node in KG (Critical)

There is no `Document` node type in the Kuzu schema. The current `DocSection` node maps to
individual markdown headings. There is no file-level aggregation node that would represent
"README.md" as a single entity with metadata like document type, category, and cross-document
relationships.

What would need to be added:
```
CREATE NODE TABLE Document (
    id STRING PRIMARY KEY,       -- e.g. "doc:README.md"
    file_path STRING,            -- relative path
    title STRING,                -- first H1 heading or filename
    doc_category STRING,         -- "guide", "api_doc", "config", "readme", etc.
    word_count INT64,
    last_modified STRING,        -- ISO timestamp
    language STRING,             -- "markdown", "rst", "plaintext"
    commit_sha STRING
)
```

### Gap 2: No Document Categorization Logic (Critical)

No code currently classifies documents into types (API docs, guides, READMEs, config files,
changelogs, etc.). This classification would need to be written. It could be:
- **Rule-based**: filename patterns (`README*`, `CHANGELOG*`, `API*`, `CONTRIBUTING*`), file
  location (`docs/api/`, `docs/guide/`), frontmatter `type:` field
- **LLM-based**: prompt the LLM to classify based on title + first paragraph

### Gap 3: No Cross-Document Relationship Extraction (Critical)

There is a `LINKS_TO` relationship (DocSection → DocSection), populated from `related:` frontmatter
fields. However:
- It only works if the author explicitly lists related documents in frontmatter
- It does not detect in-text markdown links (`[text](other-doc.md)`)
- There is no "depends-on", "supersedes", or "references" relationship at the document level

### Gap 4: No Hierarchical Document Taxonomy (Moderate)

The `Tag` system provides flat topic tagging. There is no tree structure for topics (e.g.,
"Authentication" → "OAuth2" → "JWT"). The `WikiOntology` from `wiki.py` provides LLM-based
hierarchical categorization but is not persisted to the KG and is not connected to documents.

### Gap 5: TextParser Does Not Extract Section Hierarchy at Index Time (Moderate)

The markdown heading hierarchy is extracted only during `kg build` (in `_extract_headers()`),
not during indexing. This means:
- Embeddings are stored without heading-level metadata
- Vector search results cannot filter by "section type" (H1 vs H2 vs H3)
- The full file has to be re-read from disk during KG build (line 1201 in `kg_builder.py`)

If heading level were stored as a metadata field on `CodeChunk` at parse time, it would be
available for both vector search filtering and KG ingestion without re-reading files.

### Gap 6: No Entity Extraction from Document Content (Moderate)

The `NLPExtractor` runs on code docstrings but not on plain document content. For document
ontology, we would want to extract:
- Named entities (class names, function names mentioned in prose)
- Topic keywords across the full document body (not just frontmatter tags)
- Cross-references to other documents (in-text links and mentions)

### Gap 7: WikiOntology Not Connected to KG (Minor)

`WikiGenerator` produces a rich concept ontology with LLM-generated category names but it is:
- Stored as a JSON file cache, not in Kuzu
- Code-concept focused (function/class names), not document-topic focused
- Not queryable via the KG query API

Connecting the WikiOntology to the Kuzu KG would enable graph traversal from code entities to
wiki concepts to documents, which is the full ontology vision.

---

## 3. Architecture Proposal

### Approach: Extend the Existing KG Rather Than Build Parallel

The KG pipeline is the right place for a document ontology. The proposal is to:
1. Add a `Document` node type to Kuzu schema
2. Extend `_extract_doc_sections()` in `KGBuilder` to also emit `Document` nodes
3. Add document-level relationship extraction
4. Optionally extend `TextParser` to emit heading metadata at index time

### Proposed New Kuzu Schema Additions

```cypher
-- New node type: Document (file-level)
CREATE NODE TABLE Document (
    id STRING PRIMARY KEY,         -- "doc:path/to/file.md"
    file_path STRING,
    title STRING,                  -- First H1 or filename
    doc_category STRING,           -- "readme", "guide", "api", "config", "changelog"
    word_count INT64,
    section_count INT64,
    language STRING,               -- "markdown", "rst"
    last_modified STRING
)

-- New node type: Topic (hierarchical, LLM-assigned)
CREATE NODE TABLE Topic (
    id STRING PRIMARY KEY,         -- "topic:authentication"
    name STRING,
    parent_id STRING               -- For hierarchy; null = root
)
```

```cypher
-- New relationship types
CREATE REL TABLE CONTAINS_SECTION (FROM Document TO DocSection, MANY_MANY)
CREATE REL TABLE HAS_TOPIC       (FROM Document TO Topic, MANY_MANY)
CREATE REL TABLE RELATED_TO      (FROM Document TO Document, MANY_MANY)
CREATE REL TABLE SUPERSEDES      (FROM Document TO Document, MANY_MANY)
CREATE REL TABLE DESCRIBES       (FROM Document TO CodeEntity, MANY_MANY)
```

### Proposed Build Pipeline Changes

**Step 1 (during `_extract_doc_sections`):**
- Group sections by `file_path`
- Create one `Document` node per unique file
- Create `CONTAINS_SECTION` edges linking document to each heading section
- Run rule-based document categorization using filename/path patterns

**Step 2 (new: `_extract_doc_relationships`):**
- Parse in-text markdown links `[text](url)` to discover cross-document references
- Emit `RELATED_TO` relationships
- Detect `supersedes:` or `replaces:` frontmatter fields for `SUPERSEDES`

**Step 3 (optional: LLM topic assignment):**
- Batch documents with their section titles
- Call LLM to assign 1-3 topic labels per document
- Create `Topic` nodes and `HAS_TOPIC` edges
- LLM already invoked in WikiGenerator — can reuse same client

### Data Flow Diagram

```
.md files
   |
   v
TextParser (index time)
   |-- paragraphs → CodeChunk (language="text", tags=frontmatter_tags)
   |-- [NEW] heading levels stored as chunk metadata
   |
   v
KGBuilder._extract_doc_sections() [extended]
   |-- DocSection nodes (already works)
   |-- [NEW] Document nodes (one per file)
   |-- [NEW] CONTAINS_SECTION edges
   |-- [NEW] RELATED_TO from in-text links
   |-- [NEW] DESCRIBES from doc → code entity matching
   |
   v
Kuzu KG
   |
   +--> kg query / kg_query MCP tool (already works for new node types)
   +--> [NEW] mvs kg ontology CLI command
   +--> [NEW] kg_ontology MCP tool
```

---

## 4. Estimated Effort by Component

| Component | Effort | Description |
|-----------|--------|-------------|
| Kuzu schema: `Document` + `Topic` nodes | XS (2h) | Add 2 `CREATE NODE TABLE` statements to `_create_schema()` |
| Kuzu schema: new relationships | XS (2h) | Add 5 `CREATE REL TABLE` statements |
| `KGBuilder._extract_doc_sections` extension | S (1d) | Group by file, emit Document nodes, in-text link parsing |
| Rule-based doc categorization | S (0.5d) | Filename pattern matching, frontmatter `type:` field |
| `TextParser` heading metadata | S (1d) | Store heading level on CodeChunk; no semantic change |
| `Document` → `CodeEntity` (`DESCRIBES`) | M (2d) | Match doc entity mentions to KG entities; NLP-based |
| LLM topic assignment pipeline | M (2d) | Reuse WikiGenerator LLM client, batch documents |
| `Topic` hierarchy + `HAS_TOPIC` edges | M (2d) | Connect WikiOntology to Kuzu |
| New CLI: `mvs kg ontology` | S (1d) | Query and display document ontology |
| New MCP tool: `kg_ontology` | S (1d) | Expose document ontology over MCP |
| Vector clustering for auto-topic discovery | L (1w) | Pull embeddings, run HDBSCAN, label clusters |

**Quick wins total: ~1 week** (schema, categorization, CLI display)
**Full ontology: ~3-4 weeks** (including topic hierarchy, entity linking, MCP tool)

---

## 5. Quick Wins (Minimal Changes)

These can be done with minimal risk to existing functionality:

### QW1: Add `Document` Node to Kuzu Schema (2h)
Just add the `CREATE NODE TABLE Document` and `CONTAINS_SECTION` relationship to
`_create_schema()` in `knowledge_graph.py`. No other changes needed yet.
The schema addition is non-destructive (`IF NOT EXISTS`).

### QW2: Emit `Document` Nodes in `_extract_doc_sections()` (0.5d)
Collect unique `file_path` values from `doc_sections` list after extraction. Create one
`Document` node per file. Wire up `CONTAINS_SECTION` edges. No LLM required.
This gives immediate value: the KG can answer "what documents exist in the project?"

### QW3: Rule-Based Document Categorization (0.5d)
In the same `_extract_doc_sections()` pass, classify each document:
- `README*` → "readme"
- `CHANGELOG*`, `HISTORY*` → "changelog"
- `CONTRIBUTING*` → "contributing"
- `docs/api/**` → "api_doc"
- `docs/guide/**` → "guide"
- `*.config.*`, `pyproject.toml`, etc. → "config"
- Frontmatter `type: guide` → use frontmatter value
No LLM required. Deterministic and fast.

### QW4: In-Text Markdown Link Extraction (`RELATED_TO` edges) (0.5d)
Parse `[text](./other-doc.md)` patterns in chunk content. Already have
`_extract_code_refs()` as a pattern — add a similar `_extract_doc_links()` function.
Emit `RELATED_TO` relationships between documents.

### QW5: Surface `DocSection.level` in KG Stats/Visualization (1h)
The `level` field is already stored on `DocSection` nodes (heading H1–H6). The visualization
and stats commands could filter/color by level. Zero new infrastructure needed.

---

## 6. CLI Feature vs. MCP Tool

**Both are appropriate.** The infrastructure is shared; the surface is different.

| Feature | CLI | MCP Tool |
|---------|-----|----------|
| `kg build` (existing) | Yes | Yes (`kg_build`) |
| Document ontology view | `mvs kg ontology` | `kg_ontology` |
| Query by doc category | `mvs kg query --type readme` | `kg_query` (extend) |
| Topic hierarchy browse | `mvs kg topics` | New `kg_topics` tool |
| Cross-doc relationships | `mvs kg related README.md` | `kg_query` (extend) |

The MCP tool is higher priority because it enables LLM agents (including Claude) to navigate
the document structure programmatically during code review, wiki generation, and story generation.

The CLI is useful for humans inspecting what the ontology looks like.

---

## 7. Key Dependencies and Risks

**Risk 1: Kuzu schema migrations are destructive**
Kuzu does not support `ALTER TABLE`. Adding new node types requires a `--force` rebuild of
the KG. The `IF NOT EXISTS` guard handles this for new tables, but existing databases will need
a rebuild to populate `Document` nodes.

**Risk 2: `DOCUMENTS` relationship is currently disabled in subprocess mode**
The comment at line 666 of `kg_builder.py` notes that `DOCUMENTS` (DocSection → CodeEntity)
is skipped in subprocess mode due to O(n×m) complexity (57k docs × 4k entities = 228M
comparisons). A document-to-code-entity `DESCRIBES` relationship would have the same
scalability constraint. Solution: use vector similarity (search for entity names in doc
embeddings) rather than brute-force string matching.

**Risk 3: LanceDB does not have native clustering**
Auto-topic discovery via embedding clustering requires pulling all document vectors into
memory and running external clustering (scikit-learn/HDBSCAN). This is a separate processing
step not currently in the pipeline. Not a blocker for quick wins (rule-based categorization
works without it).

**Risk 4: Kuzu threading constraints**
The KG build runs in an isolated subprocess specifically because Kuzu segfaults with background
threads. Any new node type additions must follow the same subprocess isolation pattern. This
is already established architecture; new node types are additions, not architectural changes.

---

## 8. Appendix: Current KG Schema (Complete)

```
Nodes:    CodeEntity, DocSection, Tag, Person, Project, Repository,
          Branch, Commit, ProgrammingLanguage, ProgrammingFramework

Edges:    CALLS, IMPORTS, INHERITS, CONTAINS (CodeEntity → CodeEntity)
          FOLLOWS, LINKS_TO (DocSection → DocSection)
          REFERENCES, DOCUMENTS (DocSection → CodeEntity)
          HAS_TAG, DEMONSTRATES (DocSection → Tag)
          AUTHORED, MODIFIED (Person → CodeEntity)
          PART_OF (CodeEntity → Project)
          WRITTEN_IN (CodeEntity → ProgrammingLanguage)
          USES_FRAMEWORK (CodeEntity → ProgrammingFramework)
          FRAMEWORK_FOR (ProgrammingFramework → ProgrammingLanguage)
          MODIFIES (Commit → CodeEntity)
          COMMITTED_TO (Commit → Branch)
          BRANCHED_FROM (Branch → Repository)
          BELONGS_TO (Commit → Repository)
```

---

*Research conducted: 2026-02-24*
*Files analyzed: knowledge_graph.py, kg_builder.py, kg_handlers.py, wiki.py, wiki_handlers.py,*
*story/extractor.py, story/models.py, parsers/text.py, parsers/base.py, nlp_extractor.py,*
*core/models.py (CodeChunk), vectors_backend.py, cli/commands/kg.py*
