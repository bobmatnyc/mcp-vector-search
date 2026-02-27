# Information Architecture Root Node: KG Feasibility Analysis

**Date**: 2026-02-27
**Project**: mcp-vector-search
**Scope**: Feasibility of adding an IA root node to the existing Knowledge Graph

---

## Executive Summary

Adding an Information Architecture (IA) root node to the existing KG is **highly feasible** with low risk. The schema already has most necessary scaffolding in place (`Topic` node, `HAS_TOPIC` relationship, `Document`/`DocSection` nodes with `doc_category`). The primary work is in (1) populating the unused `Topic` node during `KGBuilder.build_from_chunks_sync`, (2) introducing a single `IANode` root (or reusing `Topic`) to anchor the hierarchy, and (3) adding a `get_ia_tree` query method. No new dependencies are needed; the change is entirely additive.

---

## 1. Existing KG Architecture

### 1.1 Node Types (Kuzu schema in `knowledge_graph.py`)

| Node Table | Key Fields | Status |
|---|---|---|
| `CodeEntity` | id, name, entity_type, file_path | Fully populated |
| `DocSection` | id, name, file_path, level, line_start, line_end | Fully populated |
| `Document` | id, file_path, title, doc_category, word_count, section_count | Fully populated |
| `Tag` | id, name | Populated via HAS_TAG |
| `Topic` | id, name, parent_id | Schema exists, **never populated** |
| `Person` | id, name, email_hash | Fully populated |
| `Project` | id, name, description, repo_url | Fully populated |
| `Repository`, `Branch`, `Commit` | version control metadata | Fully populated |
| `ProgrammingLanguage`, `ProgrammingFramework` | tech stack | Fully populated |

### 1.2 Relationship Tables

Relevant to IA:
- `HAS_TOPIC` (Document → Topic) — schema exists, **never used**
- `RELATED_TO` (Document → Document) — populated
- `CONTAINS_SECTION` (Document → DocSection) — populated
- `HAS_TAG` (DocSection → Tag) — populated
- `PART_OF` (CodeEntity → Project) — populated

### 1.3 Document Categorization (in `KGBuilder._classify_document`)

The builder already classifies every document file into one of ~24 categories:
`readme`, `guide`, `api_doc`, `config`, `changelog`, `design`, `spec`, `roadmap`,
`contributing`, `license`, `research`, `bugfix`, `performance`, `setup`, `tutorial`,
`example`, `deployment`, `release_notes`, `upgrade_guide`, `troubleshooting`,
`migration`, `faq`, `security`, `test_doc`, `internal`, `report`, `other`.

This classification is the direct foundation for an IA hierarchy.

### 1.4 Current `get_document_ontology` Method

`KnowledgeGraph.get_document_ontology(category=None)` returns:
```json
{
  "categories": {
    "guide": [ {"id": ..., "title": ..., "sections": [...], "cross_references": [...], "tags": [...]} ],
    "api_doc": [...],
    ...
  },
  "total_documents": N,
  "total_sections": M,
  "total_cross_references": K
}
```

This is a flat grouping by category string — essentially one level above the document. The IA root would add a second level above this (thematic groups), making the hierarchy three levels deep:

```
IA Root
  └── Thematic Group (e.g., "User-Facing Docs")
        └── Category (e.g., "guide", "readme", "tutorial")
              └── Document
                    └── DocSection
```

### 1.5 The Unused `Topic` Node

`Topic` was created in the schema with a `parent_id` field — clearly intended for a hierarchical taxonomy. It is wired into the relationship routing logic (lines 1992, 2469 in knowledge_graph.py) but `KGBuilder` never calls any method to create Topic nodes or HAS_TOPIC relationships. This is a **stub that was designed for exactly this purpose**.

---

## 2. Python Library Analysis

### 2.1 Libraries Evaluated

#### anytree
- **What it is**: Pure-Python n-ary tree library. `Node` objects with parent/children, resolver, finder, iterators, and renderers.
- **Pros**: Elegant API, no heavy dependencies, renders as ASCII tree. Excellent for building in-memory IA hierarchies from KG query results before serializing.
- **Cons**: In-memory only; not a graph DB. No Cypher integration. Adding it just for tree traversal during build is overkill when Kuzu's own recursive Cypher queries can traverse `parent_id` chains.
- **Verdict**: Useful for output rendering (CLI display of the IA tree) but unnecessary as a core data structure.

#### networkx
- **What it is**: General-purpose graph library. DiGraph supports directed hierarchies.
- **Pros**: Already widely understood; excellent for computing graph metrics (centrality, shortest path) on the IA structure.
- **Cons**: Another large dependency (not currently in pyproject.toml); redundant with Kuzu for storage. Best suited if you need algorithmic analysis (e.g., "find the most-linked IA node").
- **Verdict**: Not needed for basic IA root. Worth considering if later you want cross-hierarchy analytics.

#### RDFLib
- **What it is**: Python library for RDF/OWL ontology representation (SKOS, Dublin Core, schema.org).
- **Pros**: Standards-compliant; SKOS `skos:broader`/`skos:narrower` is designed exactly for document taxonomies. Enables export to Turtle/JSON-LD for interoperability.
- **Cons**: Heavy dependency; introduces the full semantic web stack. Kuzu is not RDF-aware, so bridging the two requires custom serialization. Strong overkill for a code search tool.
- **Verdict**: Not recommended unless the project plans to publish the ontology externally.

#### SKOS (via RDFLib or as a convention)
- **What it is**: Simple Knowledge Organization System — a W3C standard for hierarchical concept schemes.
- **Pros**: Industry standard for information architecture. `skos:Concept`, `skos:ConceptScheme`, `skos:broader`, `skos:narrower`, `skos:related` map cleanly to the existing KG relationships.
- **Cons**: No native Python SKOS library that doesn't require RDFLib. Best used as a conceptual model, not an implementation dependency.
- **Verdict**: Use SKOS as the **conceptual template** for designing the IA structure; implement it natively in Kuzu using existing node/rel tables.

#### Built-in Python (dataclasses + dict trees)
- **Pros**: Zero new dependencies. Kuzu can already traverse `parent_id` chains in `Topic` nodes using recursive Cypher. The `get_document_ontology` method shows the pattern: build a result dict by querying Kuzu.
- **Cons**: Recursive Kuzu queries have limited depth by default; must be explicit about traversal depth.
- **Verdict**: **Recommended approach.** All the scaffolding is already in Kuzu. No new Python library is needed.

### 2.2 Recommendation

Use **no new library**. The existing `Topic` node table with `parent_id` + the `HAS_TOPIC` relationship table is all the storage infrastructure needed. Implement the IA hierarchy as Kuzu `Topic` nodes with parent-child `parent_id` references, and a new `get_ia_tree()` method on `KnowledgeGraph` that traverses this structure.

For CLI display, `rich.tree.Tree` is already imported in `kg.py` — it can render the IA hierarchy without any new dependency.

---

## 3. Proposed IA Structure for Code Documentation

### 3.1 IA Hierarchy Design

Based on the 24 existing `doc_category` values, a natural two-tier grouping is:

```
IA Root ("project:mcp-vector-search")  [Project node, already exists]
  |
  +-- "Orientation" (Topic)
  |     +-- readme          (doc_category values that map here)
  |     +-- contributing
  |     +-- license
  |     +-- changelog
  |
  +-- "Guides & Tutorials" (Topic)
  |     +-- guide
  |     +-- tutorial
  |     +-- example
  |     +-- setup
  |     +-- faq
  |     +-- troubleshooting
  |
  +-- "Architecture & Design" (Topic)
  |     +-- design
  |     +-- spec
  |     +-- research
  |     +-- internal
  |     +-- report
  |
  +-- "API Reference" (Topic)
  |     +-- api_doc
  |
  +-- "Operations" (Topic)
  |     +-- deployment
  |     +-- config
  |     +-- performance
  |     +-- security
  |
  +-- "Lifecycle" (Topic)
  |     +-- roadmap
  |     +-- release_notes
  |     +-- upgrade_guide
  |     +-- migration
  |     +-- bugfix
  |
  +-- "Testing" (Topic)
        +-- test_doc
        +-- other
```

### 3.2 Node Representation

Each IA node maps to a `Topic` node in Kuzu:

```
Topic { id: "ia:orientation",              name: "Orientation",           parent_id: "project:mcp-vector-search" }
Topic { id: "ia:guides",                   name: "Guides & Tutorials",    parent_id: "project:mcp-vector-search" }
Topic { id: "ia:architecture",             name: "Architecture & Design", parent_id: "project:mcp-vector-search" }
Topic { id: "ia:api_reference",            name: "API Reference",         parent_id: "project:mcp-vector-search" }
Topic { id: "ia:operations",              name: "Operations",            parent_id: "project:mcp-vector-search" }
Topic { id: "ia:lifecycle",               name: "Lifecycle",             parent_id: "project:mcp-vector-search" }
Topic { id: "ia:testing",                 name: "Testing",               parent_id: "project:mcp-vector-search" }
```

Category-level nodes (second tier):
```
Topic { id: "ia:cat:readme",    name: "README",    parent_id: "ia:orientation" }
Topic { id: "ia:cat:guide",     name: "Guides",    parent_id: "ia:guides" }
...
```

Documents link to category-level Topic nodes via `HAS_TOPIC`:
```
(Document {doc_category: "guide"}) -[:HAS_TOPIC]-> (Topic {id: "ia:cat:guide"})
```

### 3.3 Alternative: Lightweight Approach (No New Topic Nodes for Categories)

Since `doc_category` is already a property on `Document`, an even lighter approach skips the category-level Topic tier entirely and maps Documents directly to the 7 thematic group Topic nodes:

```
(Document {doc_category: "guide"}) -[:HAS_TOPIC]-> (Topic {id: "ia:guides"})
(Document {doc_category: "tutorial"}) -[:HAS_TOPIC]-> (Topic {id: "ia:guides"})
(Document {doc_category: "design"}) -[:HAS_TOPIC]-> (Topic {id: "ia:architecture"})
```

This is simpler to implement (7 nodes instead of 7+24) and queries are faster. The `doc_category` property is still preserved on the Document node for fine-grained filtering. **This is the recommended starting point.**

---

## 4. Integration Points with Existing KG

### 4.1 `KGBuilder.build_from_chunks_sync` (kg_builder.py)

**Integration point**: The document processing section (`_extract_documents` or equivalent) that currently creates `Document` nodes and calls `kg.upsert_document(...)`. After each document is classified, the builder would:
1. Look up the thematic group for the document's `doc_category`
2. Create the Topic node if it doesn't exist (idempotent via MERGE)
3. Create a `HAS_TOPIC` relationship: Document → Topic

The mapping (doc_category → thematic group) is a simple static dict — 5 lines of Python.

### 4.2 `KnowledgeGraph._create_schema` (knowledge_graph.py)

No schema changes required. `Topic` and `HAS_TOPIC` tables already exist. Optionally, add an `ia_level` integer property to `Topic` to distinguish root/group/category levels:

```sql
-- Optional enhancement only:
ALTER TABLE Topic ADD COLUMN ia_level INT64 DEFAULT 0;
```

However, Kuzu's `ALTER TABLE ADD COLUMN` support is limited in v0.7; it is safer to encode level in the `id` prefix convention (`ia:` prefix = group, `ia:cat:` prefix = category, `project:` = root) and compute level at query time.

### 4.3 New `get_ia_tree` Method on `KnowledgeGraph`

Add alongside `get_document_ontology`. The method:
1. Queries all `Topic` nodes ordered by `parent_id`, `name`
2. Queries all `Document` → `HAS_TOPIC` → `Topic` edges
3. Assembles a nested dict tree rooted at the `Project` node
4. Returns the tree as JSON-serializable dict

The query pattern is identical to `get_document_ontology` — iterating Kuzu result rows and building Python dicts.

### 4.4 `handle_kg_ontology` (kg_handlers.py)

Either:
- Extend the existing `handle_kg_ontology` to optionally return the IA tree view (`args.get("view") == "ia"`)
- Or add a new `handle_kg_ia` handler and a `kg_ia` MCP tool

The latter is cleaner (single-responsibility) but requires adding one entry to `tool_schemas.py` and `server.py`.

### 4.5 CLI (`kg.py`)

The existing `kg ontology` CLI command (which uses `rich.tree.Tree` for display) can be extended with a `--ia` flag to show the IA hierarchy view. Since `rich.tree.Tree` is already imported, the rendering cost is zero.

### 4.6 Visualization (`get_visualization_data`)

The `get_visualization_data` method already queries and returns multiple node types for D3.js. To include Topic/IA nodes in the force-directed graph, add a `Topic` query block (same pattern as the existing `Person`, `Project` blocks at lines 3508-3531). IA nodes would appear as a distinct color/group in the force-directed layout, visually anchoring document clusters.

---

## 5. Effort Estimate

| Component | Effort | Notes |
|---|---|---|
| IA category-to-group mapping dict | 0.5 hours | Static Python dict, ~30 lines |
| `KGBuilder`: create Topic nodes + HAS_TOPIC rels | 2 hours | Follows existing `upsert_document` pattern |
| `KnowledgeGraph.get_ia_tree()` | 2 hours | Follows `get_document_ontology` pattern |
| `KGHandlers.handle_kg_ia()` | 0.5 hours | Trivial wrapper |
| `tool_schemas.py`: new `kg_ia` schema | 0.5 hours | Copy-adapt from `kg_ontology` |
| `server.py`: route kg_ia | 0.25 hours | One elif clause |
| CLI `--ia` flag on `kg ontology` | 1 hour | Rich Tree rendering |
| Visualization: Topic nodes in D3 data | 1 hour | Copy-adapt existing node query blocks |
| Tests | 2 hours | Unit tests for builder + query method |
| **Total** | **~10 hours** | No new dependencies, purely additive |

---

## 6. Risks and Concerns

### 6.1 Kuzu Recursive Query Limitations (Low Risk)
Kuzu supports recursive pattern matching but the syntax differs from Neo4j's `*1..n` shorthand. The `Topic` hierarchy is at most 2 levels deep (root → group → category), so no recursive query is needed — a simple two-hop `MATCH (root)-[:HAS_TOPIC*1..2]->(doc)` suffices. At 2 hops this is deterministic and fast.

### 6.2 Topic/HAS_TOPIC Tables Are Empty After Existing KG Builds (Medium Risk)
Any projects that built the KG before this change would have empty `Topic` tables. The fix is automatic: the next `kg build --force` rebuilds from scratch, or alternatively add a migration path that populates Topics from existing Documents without a full rebuild (query all Documents, derive their group, insert Topics + HAS_TOPIC edges).

### 6.3 doc_category "other" Catch-All (Low Risk)
A non-trivial fraction of documents may land in `other`. The IA structure can include an "Uncategorized" group for these, or the classifier can be improved separately. This does not block the IA implementation.

### 6.4 HAS_TOPIC Relationship Directionality (Low Risk)
The existing schema defines `HAS_TOPIC FROM Document TO Topic`. This means to traverse from Topic to Documents you must reverse the edge direction in Cypher (`MATCH (t:Topic)<-[:HAS_TOPIC]-(d:Document)`). This is supported in Kuzu and the existing codebase already uses reverse-direction traversal elsewhere (e.g., `called_by` relationships). Not a blocker.

### 6.5 Kuzu Thread-Safety Constraint (Known, Managed)
The existing code already handles Kuzu's single-thread requirement via the subprocess isolation pattern in `_build_kg_in_subprocess`. Topic node creation happens inside `build_from_chunks_sync` which is already in the subprocess, so no additional threading concerns arise.

### 6.6 Over-Engineering Risk (Medium)
If the IA root is intended purely to help humans browse documentation, the existing `get_document_ontology` (flat grouping by `doc_category`) may already be sufficient. The added value of the thematic group tier is primarily navigation UX. Before implementing, confirm that the two-level hierarchy (7 thematic groups above 24 categories) provides meaningfully better navigation than the current flat 24-category view.

---

## 7. Decision Recommendation

**Proceed with implementation.** The scaffold is already in place (Topic node, HAS_TOPIC relationship). The implementation is additive, risk-free, and can be done in ~10 hours without any new dependencies.

**Recommended approach (in priority order):**

1. **Start with the lightweight variant**: Map Documents directly to 7 thematic Topic nodes (skip the category-level middle tier). This is 3 hours of work and delivers the IA root immediately.

2. **Add the category tier later** if navigation testing reveals users need finer-grained grouping. The category Topic nodes can be inserted on top of the existing structure without breaking anything.

3. **Do not add anytree, networkx, or RDFLib**. The existing Kuzu + Python dict pattern (demonstrated by `get_document_ontology`) is sufficient.

4. **Use the existing `rich.tree.Tree`** for CLI rendering — already imported, zero cost.

5. **Expose via a new `kg_ia` MCP tool** rather than overloading `kg_ontology`, to keep the API surface clean.

---

## Appendix: Key File Locations

| File | Relevance |
|---|---|
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/knowledge_graph.py` | Kuzu schema (`_create_schema`), `get_document_ontology`, `get_visualization_data` |
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/kg_builder.py` | `_classify_document`, `build_from_chunks_sync`, entity upsert patterns |
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/kg_handlers.py` | MCP handler wrappers |
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/tool_schemas.py` | MCP tool schemas for kg_build, kg_stats, kg_query, kg_ontology |
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py` | Tool routing (elif chain) |
| `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/kg.py` | CLI commands, Rich Tree rendering |
| `/Users/masa/Projects/mcp-vector-search/pyproject.toml` | Dependencies (kuzu>=0.7.0 already present) |
