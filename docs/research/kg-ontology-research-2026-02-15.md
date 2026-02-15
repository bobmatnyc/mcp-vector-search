# Knowledge Graph Ontology Research for mcp-vector-search

**Date:** 2026-02-15
**Researcher:** Claude (Research Agent)
**Purpose:** Identify ontologies and entity/relationship types to extend the Knowledge Graph for documentation and project artifacts

---

## Executive Summary

This research identifies ontologies and schemas that can enhance mcp-vector-search's Knowledge Graph to better handle documentation (markdown, text files) and project work items (issues, features, tasks). The current KG focuses on code entities (Function, Class, Module) with relationships (CALLS, IMPORTS, INHERITS, CONTAINS). The extension should add ~10k text chunks to the existing ~5k code chunks while maintaining semantic relationships.

**Key Findings:**
1. **Dublin Core** provides foundational metadata relationships (replaces, references, requires)
2. **DCAT** offers catalog/dataset structures applicable to documentation collections
3. **CodeMeta** defines software-specific metadata including documentation links
4. **PROV-O** enables provenance tracking (creation, derivation, attribution)
5. **Schema.org** provides TechArticle/SoftwareApplication types with dependency modeling
6. **SIOC** supports discussion/community content structures (applicable to issue tracking)

---

## 1. Documentation Ontologies

### 1.1 Dublin Core Metadata Terms (DCMI)

**Source:** https://www.dublincore.org/specifications/dublin-core/dcmi-terms/
**Maturity:** W3C Standard, widely adopted
**Complexity:** Simple (foundational vocabulary)

#### Core Entity Types
- **Resource** - Any digital or physical item (base type)
- **Collection** - Aggregation of resources
- **Dataset** - Structured data collection
- **Text** - Textual content (documentation, articles)

#### Key Relationship Types

| Relationship | Description | Use Case in mcp-vector-search |
|--------------|-------------|-------------------------------|
| **replaces** | Resource supplants another | Track doc versioning (v2 replaces v1) |
| **isReplacedBy** | Inverse of replaces | Backward links to newer docs |
| **references** | Cites or points to resource | Doc mentions code entity or other doc |
| **isReferencedBy** | Inverse of references | Find all docs that mention this code |
| **requires** | Prerequisite dependency | "Read Getting Started before API Guide" |
| **isRequiredBy** | Inverse of requires | What depends on this doc |
| **conformsTo** | Adheres to standard/spec | Doc follows API spec v2.0 |
| **isVersionOf** | Version relationship | Multiple versions of same doc |
| **hasVersion** | Inverse version link | Root doc -> all versions |
| **source** | Derivation lineage | Doc derived from template |

#### Implementation Mapping

**Current KG:**
```cypher
(CodeEntity)-[CALLS]->(CodeEntity)
(CodeEntity)-[IMPORTS]->(CodeEntity)
```

**Extended with Dublin Core:**
```cypher
// Documentation entities
(DocEntity {type: 'doc', path: 'README.md', title: 'Getting Started'})
(DocEntity {type: 'doc', path: 'API.md', title: 'API Reference'})

// Documentation relationships
(DocEntity)-[REFERENCES {type: 'code_reference'}]->(CodeEntity)
(DocEntity)-[REQUIRES {prerequisite: true}]->(DocEntity)
(DocEntity)-[REPLACES {version: '2.0'}]->(DocEntity)
```

**Complexity:** Simple - Just add node type `DocEntity` and 3-4 new edge types

---

### 1.2 DCAT (Data Catalog Vocabulary)

**Source:** https://www.w3.org/TR/vocab-dcat-3/
**Maturity:** W3C Recommendation
**Complexity:** Medium (hierarchical structure)

#### Core Entity Types

| Entity | Description | Mapping to mcp-vector-search |
|--------|-------------|------------------------------|
| **Catalog** | Dataset collection | Documentation library/wiki |
| **Dataset** | Data collection | Single documentation file or section |
| **Distribution** | Specific format | Markdown source vs rendered HTML |
| **DatasetSeries** | Grouped datasets | Multi-part tutorial series |
| **CatalogRecord** | Metadata entry | Index entry for doc with timestamps |

#### Key Relationship Types

| Relationship | Description | Use Case |
|--------------|-------------|----------|
| **dataset** | Catalog contains dataset | "docs/" contains all markdown files |
| **distribution** | Dataset has format | README.md (source) + README.html (rendered) |
| **inSeries** | Part of series | "Part 2 of Authentication Guide" |
| **previousVersion** | Version chain | Track doc evolution over commits |
| **replaces** | Supersedes resource | New guide replaces old one |

#### Implementation Benefits
- **Hierarchical organization**: Group docs by category (guides/, api/, examples/)
- **Versioning support**: Track documentation changes over git commits
- **Format tracking**: Link markdown source to rendered output

**Complexity:** Medium - Requires hierarchical modeling and metadata tracking

---

### 1.3 Schema.org TechArticle

**Source:** https://schema.org/TechArticle
**Maturity:** Production-ready, widely used
**Complexity:** Simple

#### Entity Type: TechArticle

**Properties:**
- `dependencies` - Prerequisites needed (e.g., "Read Installation Guide first")
- `proficiencyLevel` - Beginner/Expert skill level
- `articleSection` - Category (Tutorial, Reference, Troubleshooting)
- `teaches` - Learning outcomes
- `isPartOf` - Belongs to larger doc set
- `hasPart` - Contains sub-sections

#### Implementation Example

```cypher
// TechArticle as documentation node
(TechArticle {
  title: "FastAPI Authentication",
  dependencies: ["Installation Guide", "Basic Concepts"],
  proficiencyLevel: "Intermediate",
  articleSection: "Security",
  path: "docs/auth.md"
})

// Relationships
(TechArticle)-[DEPENDS_ON]->(TechArticle)  // Prerequisites
(TechArticle)-[IS_PART_OF]->(DocCollection) // Belongs to guide series
(TechArticle)-[DOCUMENTS]->(CodeEntity)    // Explains code
```

**Complexity:** Simple - Just add metadata properties to DocEntity nodes

---

### 1.4 PROV-O (Provenance Ontology)

**Source:** https://www.w3.org/TR/prov-o/
**Maturity:** W3C Recommendation
**Complexity:** Medium (temporal tracking)

#### Core Entity Types

| Type | Description | Use in mcp-vector-search |
|------|-------------|--------------------------|
| **Entity** | Artifact/resource | Code file, documentation, chunk |
| **Activity** | Process over time | Indexing run, doc generation, refactor |
| **Agent** | Responsible actor | Developer, bot, CI/CD system |

#### Key Relationships

| Relationship | Description | Use Case |
|--------------|-------------|----------|
| **wasGeneratedBy** | Entity created by activity | Doc generated from template |
| **wasDerivedFrom** | Transformation source | New doc derived from old version |
| **wasAttributedTo** | Responsibility | "Author: John Doe" |
| **used** | Activity consumed entity | Indexing activity used Python files |
| **wasAssociatedWith** | Agent influenced activity | Bot performed indexing |

#### Implementation for Temporal Tracking

```cypher
// Track documentation provenance
(DocEntity)-[WAS_GENERATED_BY]->(Activity {type: 'doc_generation', timestamp: '2026-02-15'})
(DocEntity)-[WAS_ATTRIBUTED_TO]->(Agent {name: 'docs-bot', type: 'automation'})
(DocEntity)-[WAS_DERIVED_FROM]->(DocEntity {version: '1.0'})

// Track code evolution
(CodeEntity)-[WAS_GENERATED_BY]->(Activity {type: 'commit', sha: 'abc123'})
(CodeEntity)-[WAS_ATTRIBUTED_TO]->(Agent {name: 'developer', email: 'dev@example.com'})
```

**Complexity:** Medium - Requires temporal tracking and activity logging

---

### 1.5 CodeMeta Software Metadata

**Source:** https://codemeta.github.io/terms/
**Maturity:** Production (v3.0, 2023)
**Complexity:** Simple

#### Software Documentation Properties

| Property | Description | KG Application |
|----------|-------------|----------------|
| **readme** | Link to README file | (Project)-[HAS_README]->(DocEntity) |
| **softwareHelp** | Help resources | (Code)-[HAS_HELP]->(DocEntity) |
| **releaseNotes** | Version changelog | (Release)-[HAS_NOTES]->(DocEntity) |
| **buildInstructions** | Installation docs | (Project)-[HAS_BUILD_DOCS]->(DocEntity) |
| **softwareRequirements** | Dependencies | (Code)-[REQUIRES_SOFTWARE]->(Library) |
| **hasSourceCode** | Code location | (Doc)-[DOCUMENTS_CODE]->(CodeEntity) |

#### Implementation Example

```cypher
// Project-level documentation
(Project {name: 'mcp-vector-search'})
  -[HAS_README]->(DocEntity {path: 'README.md'})
  -[HAS_BUILD_DOCS]->(DocEntity {path: 'INSTALL.md'})

// API documentation links
(CodeEntity {type: 'function', name: 'search_code'})
  -[HAS_HELP]->(DocEntity {path: 'docs/api/search.md', section: 'search_code'})
```

**Complexity:** Simple - Add project-level nodes and doc links

---

## 2. Project/Work Ontologies

### 2.1 SIOC (Semantically-Interlinked Online Communities)

**Source:** https://www.w3.org/wiki/SIOC
**Maturity:** Established (multiple implementations)
**Complexity:** Medium

#### Core Entity Types

| Type | Description | Application to Issue Tracking |
|------|-------------|-------------------------------|
| **Post** | Individual content item | Issue, comment, pull request |
| **Thread** | Discussion sequence | Issue with comments |
| **Forum** | Community space | Project issue tracker, repository |
| **User** | Community member | Developer, contributor |

#### Key Relationships

| Relationship | Description | Use Case |
|--------------|-------------|----------|
| **has_creator** | Authored by user | Issue created by developer |
| **has_reply** | Response relationship | Comment replies to issue |
| **has_parent** | Thread hierarchy | Sub-issue under epic |
| **related_to** | Topical connection | Related issues |
| **has_container** | Belongs to forum | Issue belongs to project tracker |

#### Implementation for Issue Tracking

```cypher
// Issues and PRs as Posts
(Issue {
  id: 'GH-123',
  title: 'Add authentication',
  type: 'feature',
  status: 'open'
})-[HAS_CREATOR]->(User {name: 'developer'})
 -[RELATED_TO {type: 'blocks'}]->(Issue {id: 'GH-124'})
 -[REFERENCES]->(CodeEntity {name: 'auth_handler'})
 -[HAS_REPLY]->(Comment {text: 'Working on this'})

// Project as Forum
(Project {name: 'mcp-vector-search'})
  -[HAS_CONTAINER]->(Issue)
```

**Complexity:** Medium - Requires user/activity modeling

---

### 2.2 DOAP (Description of a Project)

**Source:** https://github.com/edumbill/doap/wiki
**Maturity:** Widely used in open source
**Complexity:** Simple

#### Core Entity Types

| Type | Description | KG Application |
|------|-------------|----------------|
| **Project** | Software project | mcp-vector-search project node |
| **Version** | Release/version | v2.2.0, v2.3.0 releases |
| **Repository** | Code repository | GitHub repo, local git |
| **BugDatabase** | Issue tracker | GitHub Issues, Jira |

#### Key Relationships

```cypher
// Project structure
(Project {name: 'mcp-vector-search'})
  -[HAS_REPOSITORY]->(Repository {url: 'github.com/...'})
  -[HAS_BUG_DATABASE]->(BugTracker {type: 'github-issues'})
  -[HAS_RELEASE]->(Version {number: '2.2.21', date: '2026-02-15'})

// Link to code and docs
(Project)-[CONTAINS]->(CodeEntity)
(Project)-[HAS_DOCUMENTATION]->(DocEntity)
```

**Complexity:** Simple - Add project-level nodes

---

## 3. Recommended Entity Types for Text/Documentation

### 3.1 Core Documentation Entities

```python
@dataclass
class DocEntity:
    """Documentation node in knowledge graph."""
    id: str                          # Unique identifier (chunk_id)
    name: str                        # Display name (file name or section title)
    entity_type: str                 # doc_file, doc_section, doc_paragraph, guide, reference, tutorial
    file_path: str                   # Source file path
    title: str | None = None         # Parsed title (from markdown # header)
    section: str | None = None       # Section name (## subsection)
    doc_type: str | None = None      # readme, api_reference, guide, tutorial, changelog
    format: str = 'markdown'         # markdown, text, rst, html
    commit_sha: str | None = None    # Git commit (for versioning)

    # Metadata from ontologies
    dependencies: list[str] = None   # Prerequisites (schema.org TechArticle)
    proficiency_level: str = None    # beginner, intermediate, advanced
    article_section: str = None      # Category (tutorial, reference, etc.)
    version: str | None = None       # Document version
```

### 3.2 Project/Work Entities

```python
@dataclass
class WorkItemEntity:
    """Work item (issue, feature, task) node."""
    id: str                          # Issue ID (GH-123, JIRA-456)
    name: str                        # Issue title
    entity_type: str                 # issue, feature, epic, task, bug
    status: str                      # open, in_progress, closed
    priority: str = 'medium'         # low, medium, high, critical

    # SIOC properties
    creator: str | None = None       # Author/creator
    created_at: str | None = None    # Timestamp

    # Relationships (stored as edge lists)
    blocks: list[str] = None         # IDs of issues this blocks
    related_to: list[str] = None     # Related issues
    references_code: list[str] = None # Code entities mentioned
```

### 3.3 Hybrid Text/Code Entity

**Option:** Extend existing `CodeChunk` model to support documentation:

```python
@dataclass
class CodeChunk:
    # Existing code properties
    content: str
    file_path: Path
    chunk_type: str  # Expand: code, function, class, doc_text, doc_section, doc_paragraph

    # NEW: Documentation-specific properties
    doc_title: str | None = None         # Markdown header title
    doc_section: str | None = None       # Section hierarchy (1.2.3)
    doc_type: str | None = None          # readme, guide, api_reference
    doc_dependencies: list[str] = None   # Prerequisite docs
    code_references: list[str] = None    # Code entities mentioned in text
```

**Advantage:** Unified model, simpler implementation
**Disadvantage:** Less semantic clarity between code and docs

---

## 4. Recommended Relationship Types for Text/Documentation

### 4.1 Documentation-to-Code Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **DOCUMENTS** | DocEntity | CodeEntity | Doc explains/describes code | HIGH |
| **REFERENCES** | DocEntity | CodeEntity | Doc mentions code entity | HIGH |
| **HAS_EXAMPLE** | DocEntity | CodeEntity | Doc contains code example | MEDIUM |
| **DEFINED_IN** | CodeEntity | DocEntity | Code documented in specific doc | MEDIUM |

#### Implementation Example

```cypher
// Documentation explains code
(DocEntity {path: 'docs/auth.md', section: 'login'})
  -[DOCUMENTS {detail_level: 'detailed'}]->(CodeEntity {name: 'login_handler'})

// Doc references multiple entities
(DocEntity {path: 'README.md'})
  -[REFERENCES {mention_count: 3}]->(CodeEntity {name: 'search_code'})
  -[REFERENCES {mention_count: 1}]->(CodeEntity {name: 'index_project'})

// Code has API documentation
(CodeEntity {type: 'function', name: 'search_similar'})
  -[DEFINED_IN {section: 'API Reference'}]->(DocEntity {path: 'API.md'})
```

---

### 4.2 Documentation-to-Documentation Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **REQUIRES** | DocEntity | DocEntity | Prerequisite reading | HIGH |
| **RELATES_TO** | DocEntity | DocEntity | Topically similar | MEDIUM |
| **IS_PART_OF** | DocEntity | DocEntity | Belongs to doc set | HIGH |
| **REPLACES** | DocEntity | DocEntity | Supersedes old doc | MEDIUM |
| **REFERENCES** | DocEntity | DocEntity | Cross-reference link | MEDIUM |
| **DERIVES_FROM** | DocEntity | DocEntity | Based on template/source | LOW |

#### Implementation Example

```cypher
// Prerequisite chain
(DocEntity {path: 'advanced-search.md'})
  -[REQUIRES {order: 1}]->(DocEntity {path: 'basic-search.md'})
  -[REQUIRES {order: 2}]->(DocEntity {path: 'installation.md'})

// Documentation hierarchy
(DocEntity {path: 'user-guide.md', type: 'guide'})
  -[HAS_PART]->(DocEntity {path: 'user-guide.md', section: 'Getting Started'})
  -[HAS_PART]->(DocEntity {path: 'user-guide.md', section: 'Advanced Usage'})

// Version replacement
(DocEntity {path: 'API-v2.md', version: '2.0'})
  -[REPLACES {reason: 'api_update'}]->(DocEntity {path: 'API-v1.md', version: '1.0'})

// Topical relationships (from vector similarity)
(DocEntity {path: 'authentication.md'})
  -[RELATES_TO {similarity: 0.85}]->(DocEntity {path: 'authorization.md'})
```

---

### 4.3 Work Item Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **BLOCKS** | WorkItem | WorkItem | Dependency blocker | HIGH |
| **RELATED_TO** | WorkItem | WorkItem | Related work | MEDIUM |
| **IMPLEMENTS** | CodeEntity | WorkItem | Code implements feature | HIGH |
| **DOCUMENTED_IN** | WorkItem | DocEntity | Issue described in doc | LOW |
| **REFERENCES_CODE** | WorkItem | CodeEntity | Issue mentions code | MEDIUM |

#### Implementation Example

```cypher
// Issue blocking chain
(Issue {id: 'GH-125', title: 'Add OAuth'})
  -[BLOCKS {reason: 'prerequisite'}]->(Issue {id: 'GH-126', title: 'Add user roles'})

// Code implements feature
(CodeEntity {name: 'oauth_handler', file: 'auth.py'})
  -[IMPLEMENTS {status: 'completed'}]->(Issue {id: 'GH-125'})

// Issue references code and docs
(Issue {id: 'GH-127', title: 'Fix search performance'})
  -[REFERENCES_CODE]->(CodeEntity {name: 'search_code'})
  -[DOCUMENTED_IN]->(DocEntity {path: 'docs/performance.md'})
```

---

### 4.4 Provenance/Temporal Relationships

| Relationship | Source | Target | Description | Priority |
|--------------|--------|--------|-------------|----------|
| **GENERATED_BY** | Entity | Activity | Created by process | LOW |
| **DERIVED_FROM** | Entity | Entity | Evolved from previous | MEDIUM |
| **ATTRIBUTED_TO** | Entity | Agent | Created by person/bot | LOW |
| **VERSION_OF** | Entity | Entity | Version relationship | MEDIUM |

#### Implementation Example

```cypher
// Track document evolution
(DocEntity {path: 'API.md', commit: 'abc123'})
  -[DERIVED_FROM]->(DocEntity {path: 'API.md', commit: '456def'})
  -[ATTRIBUTED_TO]->(Agent {name: 'developer', type: 'human'})
  -[GENERATED_BY]->(Activity {type: 'doc_update', timestamp: '2026-02-15'})

// Code-doc co-evolution
(CodeEntity {name: 'search_code', commit: 'abc123'})
  -[GENERATED_BY]->(Activity {type: 'commit', sha: 'abc123'})
(DocEntity {path: 'docs/search.md', commit: 'abc123'})
  -[GENERATED_BY]->(Activity {type: 'commit', sha: 'abc123'})
  -[DOCUMENTS]->(CodeEntity {name: 'search_code'})
```

---

## 5. Implementation Priority and Complexity

### Phase 1: Essential Documentation Links (Simple, High Impact)

**Entities:**
- Add `DocEntity` node type alongside `CodeEntity`
- Properties: `id`, `name`, `entity_type`, `file_path`, `doc_type`, `title`

**Relationships:**
- `DOCUMENTS` (DocEntity -> CodeEntity) - "doc explains code"
- `REFERENCES` (DocEntity -> CodeEntity) - "doc mentions code"
- `REQUIRES` (DocEntity -> DocEntity) - "prerequisite reading"

**Implementation Effort:** 1-2 days
**Value:** HIGH - Enables doc-code navigation and prerequisite chains

**Example Queries:**
```cypher
// Find all docs that explain this function
MATCH (d:DocEntity)-[DOCUMENTS]->(c:CodeEntity {name: 'search_code'})
RETURN d.path, d.title

// Find prerequisite docs for this guide
MATCH (d:DocEntity {path: 'advanced-search.md'})-[REQUIRES*]->(prereq:DocEntity)
RETURN prereq.path, prereq.title ORDER BY depth

// Find all code referenced in this doc
MATCH (d:DocEntity {path: 'README.md'})-[REFERENCES]->(c:CodeEntity)
RETURN c.name, c.type, c.file_path
```

---

### Phase 2: Documentation Hierarchy (Medium, Medium Impact)

**Entities:**
- Add `DocCollection` node for grouped docs (guide series, API reference)
- Properties: `collection_name`, `category`, `order`

**Relationships:**
- `IS_PART_OF` (DocEntity -> DocCollection) - "belongs to guide series"
- `HAS_PART` (DocCollection -> DocEntity) - inverse relationship
- `RELATES_TO` (DocEntity -> DocEntity) - topical similarity (from vector search)

**Implementation Effort:** 2-3 days
**Value:** MEDIUM - Enables browsing doc collections and finding related content

**Example Queries:**
```cypher
// Get all docs in "User Guide" series
MATCH (c:DocCollection {name: 'User Guide'})-[HAS_PART]->(d:DocEntity)
RETURN d.path, d.title ORDER BY d.order

// Find topically related docs (similarity > 0.8)
MATCH (d:DocEntity {path: 'authentication.md'})-[r:RELATES_TO]->(related:DocEntity)
WHERE r.similarity > 0.8
RETURN related.path, related.title, r.similarity
```

---

### Phase 3: Work Item Integration (Medium, High Impact for Teams)

**Entities:**
- Add `WorkItem` node (issues, features, tasks)
- Properties: `id`, `title`, `status`, `priority`, `creator`

**Relationships:**
- `IMPLEMENTS` (CodeEntity -> WorkItem) - "code implements feature"
- `BLOCKS` (WorkItem -> WorkItem) - "issue blocks another"
- `REFERENCES_CODE` (WorkItem -> CodeEntity) - "issue mentions code"
- `DOCUMENTED_IN` (WorkItem -> DocEntity) - "feature described in doc"

**Implementation Effort:** 3-4 days (requires GitHub/Jira API integration)
**Value:** HIGH - Connects code/docs to project management

**Example Queries:**
```cypher
// Find all code implementing this feature
MATCH (c:CodeEntity)-[IMPLEMENTS]->(w:WorkItem {id: 'GH-125'})
RETURN c.name, c.file_path

// Find blocking chain for an issue
MATCH (w:WorkItem {id: 'GH-126'})<-[BLOCKS*]-(blocker:WorkItem)
RETURN blocker.id, blocker.title, blocker.status

// Find all docs related to this issue
MATCH (w:WorkItem {id: 'GH-125'})-[DOCUMENTED_IN]->(d:DocEntity)
RETURN d.path, d.title
```

---

### Phase 4: Versioning & Provenance (Complex, Low Immediate Value)

**Entities:**
- Add `Activity` node (commits, indexing runs)
- Add `Agent` node (developers, bots)

**Relationships:**
- `REPLACES` (Entity -> Entity) - version supersession
- `DERIVED_FROM` (Entity -> Entity) - evolution tracking
- `GENERATED_BY` (Entity -> Activity) - provenance
- `ATTRIBUTED_TO` (Entity -> Agent) - authorship

**Implementation Effort:** 4-5 days (requires git history parsing)
**Value:** LOW initially, HIGH for long-term maintenance and auditing

**Example Queries:**
```cypher
// Track documentation evolution over time
MATCH path = (latest:DocEntity {path: 'API.md', commit: 'abc123'})-[DERIVED_FROM*]->(oldest:DocEntity)
RETURN [node IN nodes(path) | node.commit] AS version_history

// Find who last updated this code
MATCH (c:CodeEntity {name: 'search_code'})-[ATTRIBUTED_TO]->(a:Agent)
RETURN a.name, a.email
```

---

## 6. Recommended Implementation Approach

### 6.1 Current State Analysis

**Existing KG Structure:**
```python
# knowledge_graph.py
class CodeEntity:
    id: str
    name: str
    entity_type: str  # file, class, function, module
    file_path: str
    commit_sha: str | None

# Relationships: CALLS, IMPORTS, INHERITS, CONTAINS
```

**Text Chunks:** Currently stored in vector DB as `CodeChunk` with `chunk_type='text'` but no graph relationships.

---

### 6.2 Phase 1 Implementation Plan (Documentation Links)

**Step 1:** Extend KG schema for DocEntity

```python
# knowledge_graph.py

@dataclass
class DocEntity:
    """Documentation node in knowledge graph."""
    id: str                          # chunk_id from CodeChunk
    name: str                        # File name or section title
    entity_type: str                 # doc_file, doc_section, doc_paragraph
    file_path: str                   # Source markdown/text file
    doc_type: str = 'general'        # readme, api_reference, guide, tutorial
    title: str | None = None         # Parsed from markdown header
    commit_sha: str | None = None    # Git commit

# Add new relationship types
RELATIONSHIP_TYPES = ['CALLS', 'IMPORTS', 'INHERITS', 'CONTAINS',
                      'DOCUMENTS', 'REFERENCES', 'REQUIRES']  # NEW
```

**Step 2:** Create Kuzu schema for DocEntity

```python
def _create_schema(self):
    # Existing CodeEntity table
    self.conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS CodeEntity (
            id STRING PRIMARY KEY,
            name STRING,
            entity_type STRING,
            file_path STRING,
            commit_sha STRING,
            created_at TIMESTAMP DEFAULT current_timestamp()
        )
    """)

    # NEW: DocEntity table
    self.conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS DocEntity (
            id STRING PRIMARY KEY,
            name STRING,
            entity_type STRING,
            file_path STRING,
            doc_type STRING,
            title STRING,
            commit_sha STRING,
            created_at TIMESTAMP DEFAULT current_timestamp()
        )
    """)

    # NEW: Doc-to-Code relationships
    self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS DOCUMENTS (
            FROM DocEntity TO CodeEntity,
            detail_level STRING DEFAULT 'brief',
            confidence DOUBLE DEFAULT 1.0,
            MANY_MANY
        )
    """)

    self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS REFERENCES (
            FROM DocEntity TO CodeEntity,
            mention_count INT DEFAULT 1,
            context STRING,
            MANY_MANY
        )
    """)

    # NEW: Doc-to-Doc relationships
    self.conn.execute("""
        CREATE REL TABLE IF NOT EXISTS REQUIRES (
            FROM DocEntity TO DocEntity,
            order INT DEFAULT 1,
            reason STRING,
            MANY_MANY
        )
    """)
```

**Step 3:** Parse documentation and extract references

```python
# text_parser.py enhancement

async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
    """Enhanced to extract code references."""
    chunks = []

    for para_info in self._extract_paragraphs(content):
        chunk = self._create_chunk(...)

        # NEW: Extract code references from text
        code_refs = self._extract_code_references(para_info["content"])
        chunk.nlp_code_refs = code_refs  # Already exists in CodeChunk

        chunks.append(chunk)

    return chunks

def _extract_code_references(self, text: str) -> list[str]:
    """Extract code entity references from markdown/text.

    Patterns:
    - Backtick code: `function_name`, `ClassName`
    - Markdown links: [search_code](link)
    - Module paths: `module.submodule.function`
    """
    import re

    refs = []

    # Backtick code references
    backtick_pattern = r'`([a-zA-Z_][a-zA-Z0-9_\.]*)`'
    refs.extend(re.findall(backtick_pattern, text))

    # Markdown links with code names
    link_pattern = r'\[([a-zA-Z_][a-zA-Z0-9_\.]*)\]\('
    refs.extend(re.findall(link_pattern, text))

    return list(set(refs))  # Deduplicate
```

**Step 4:** Build KG relationships during indexing

```python
# indexer.py enhancement

async def _build_graph_relationships(self, chunks: list[CodeChunk]):
    """Build KG relationships including doc-code links."""

    # Existing: Build code relationships (CALLS, IMPORTS, etc.)
    await self._build_code_relationships(chunks)

    # NEW: Build documentation relationships
    await self._build_doc_relationships(chunks)

async def _build_doc_relationships(self, chunks: list[CodeChunk]):
    """Build documentation-to-code relationships."""
    kg = self.database.knowledge_graph

    for chunk in chunks:
        if chunk.chunk_type in ['text', 'doc_section', 'doc_paragraph']:
            # Create DocEntity node
            doc_entity = DocEntity(
                id=chunk.chunk_id,
                name=chunk.file_path.name,
                entity_type=chunk.chunk_type,
                file_path=str(chunk.file_path),
                doc_type=self._infer_doc_type(chunk.file_path),
                title=self._extract_title(chunk.content),
            )
            await kg.add_doc_entity(doc_entity)

            # Create REFERENCES relationships
            for code_ref in chunk.nlp_code_refs:
                # Find matching CodeEntity by name
                code_entity_id = await self._find_code_entity(code_ref)
                if code_entity_id:
                    rel = DocRelationship(
                        source_id=chunk.chunk_id,
                        target_id=code_entity_id,
                        relationship_type='REFERENCES',
                        mention_count=chunk.content.count(code_ref),
                    )
                    await kg.add_doc_relationship(rel)

def _infer_doc_type(self, file_path: Path) -> str:
    """Infer documentation type from file name."""
    name_lower = file_path.name.lower()

    if 'readme' in name_lower:
        return 'readme'
    elif 'api' in name_lower or 'reference' in name_lower:
        return 'api_reference'
    elif 'guide' in name_lower or 'tutorial' in name_lower:
        return 'guide'
    elif 'changelog' in name_lower or 'release' in name_lower:
        return 'changelog'
    else:
        return 'general'
```

---

### 6.3 Example Queries After Phase 1

**Query 1: Find all documentation for a function**

```python
# MCP tool: get_function_docs
async def get_function_docs(function_name: str) -> list[dict]:
    """Find all docs that document or reference a function."""
    kg = self.database.knowledge_graph

    result = await kg.conn.execute("""
        MATCH (c:CodeEntity {name: $name})<-[r:DOCUMENTS|REFERENCES]-(d:DocEntity)
        RETURN d.file_path AS doc_path,
               d.title AS doc_title,
               type(r) AS relationship,
               r.detail_level AS detail,
               r.mention_count AS mentions
        ORDER BY relationship DESC, mentions DESC
    """, {"name": function_name})

    docs = []
    while result.has_next():
        row = result.get_next()
        docs.append({
            "doc_path": row[0],
            "title": row[1],
            "relationship": row[2],  # DOCUMENTS or REFERENCES
            "detail": row[3],
            "mentions": row[4],
        })

    return docs
```

**Query 2: Find prerequisite docs**

```python
async def get_prerequisites(doc_path: str) -> list[dict]:
    """Find all prerequisite docs (recursive)."""
    kg = self.database.knowledge_graph

    result = await kg.conn.execute("""
        MATCH path = (d:DocEntity {file_path: $path})-[r:REQUIRES*]->(prereq:DocEntity)
        RETURN prereq.file_path AS doc_path,
               prereq.title AS title,
               r[0].order AS order,
               length(path) AS depth
        ORDER BY depth, order
    """, {"path": doc_path})

    prereqs = []
    while result.has_next():
        row = result.get_next()
        prereqs.append({
            "path": row[0],
            "title": row[1],
            "order": row[2],
            "depth": row[3],
        })

    return prereqs
```

**Query 3: Find related docs via shared code references**

```python
async def find_related_docs(doc_path: str, limit: int = 10) -> list[dict]:
    """Find docs that reference similar code entities."""
    kg = self.database.knowledge_graph

    result = await kg.conn.execute("""
        MATCH (d1:DocEntity {file_path: $path})-[:REFERENCES]->(c:CodeEntity)<-[:REFERENCES]-(d2:DocEntity)
        WHERE d1 <> d2
        WITH d2, count(c) AS shared_refs
        RETURN d2.file_path AS doc_path,
               d2.title AS title,
               shared_refs
        ORDER BY shared_refs DESC
        LIMIT $limit
    """, {"path": doc_path, "limit": limit})

    related = []
    while result.has_next():
        row = result.get_next()
        related.append({
            "path": row[0],
            "title": row[1],
            "shared_references": row[2],
        })

    return related
```

---

### 6.4 Testing Strategy

**Unit Tests:**
```python
# tests/test_doc_kg.py

@pytest.mark.asyncio
async def test_create_doc_entity(knowledge_graph):
    doc = DocEntity(
        id='doc-123',
        name='README.md',
        entity_type='doc_file',
        file_path='README.md',
        doc_type='readme',
        title='Getting Started',
    )

    await knowledge_graph.add_doc_entity(doc)

    # Verify entity exists
    result = await knowledge_graph.conn.execute(
        "MATCH (d:DocEntity {id: $id}) RETURN d",
        {"id": 'doc-123'}
    )
    assert result.has_next()

@pytest.mark.asyncio
async def test_doc_references_code(knowledge_graph):
    # Create code entity
    code = CodeEntity(...)
    await knowledge_graph.add_entity(code)

    # Create doc entity
    doc = DocEntity(...)
    await knowledge_graph.add_doc_entity(doc)

    # Create REFERENCES relationship
    rel = DocRelationship(
        source_id=doc.id,
        target_id=code.id,
        relationship_type='REFERENCES',
        mention_count=3,
    )
    await knowledge_graph.add_doc_relationship(rel)

    # Verify relationship
    result = await knowledge_graph.conn.execute("""
        MATCH (d:DocEntity {id: $doc_id})-[r:REFERENCES]->(c:CodeEntity {id: $code_id})
        RETURN r.mention_count
    """, {"doc_id": doc.id, "code_id": code.id})

    assert result.has_next()
    assert result.get_next()[0] == 3
```

---

### 6.5 Performance Considerations

**Indexing Performance:**
- Phase 1 adds ~10k DocEntity nodes + relationships
- Kuzu can handle this efficiently (milliseconds per relationship)
- Batch inserts for better performance (100-1000 relationships per transaction)

**Query Performance:**
- Simple queries (1-2 hops): <10ms
- Recursive queries (REQUIRES*): <50ms with proper indexing
- Consider caching for frequently accessed doc hierarchies

**Storage:**
- DocEntity nodes: ~1KB each * 10k = 10MB
- Relationships: ~100 bytes each * 30k = 3MB
- Total additional KG storage: ~13MB (negligible)

---

## 7. Alternative Approaches Considered

### 7.1 Flat Metadata vs. Graph Relationships

**Option A:** Store doc references as metadata in `CodeChunk`
```python
class CodeChunk:
    # ...existing fields...
    referenced_in_docs: list[str] = None  # List of doc paths
```

**Pros:** Simpler implementation, no KG changes
**Cons:** No bidirectional queries, no relationship attributes, limited query flexibility

**Option B:** Graph relationships (RECOMMENDED)
```cypher
(DocEntity)-[REFERENCES {mention_count: 3}]->(CodeEntity)
```

**Pros:** Rich queries, relationship metadata, bidirectional navigation
**Cons:** Additional complexity, requires KG schema changes

**Decision:** Option B (graph) provides significantly more value for querying and navigation.

---

### 7.2 Unified Entity vs. Separate DocEntity

**Option A:** Extend `CodeEntity` to handle docs
```python
class CodeEntity:
    entity_type: str  # file, class, function, module, doc_file, doc_section
```

**Pros:** Simpler schema, single entity type
**Cons:** Confusing semantics (doc is not "code"), different property requirements

**Option B:** Separate `DocEntity` (RECOMMENDED)
```python
class DocEntity:
    # Doc-specific properties
    doc_type: str
    title: str
    dependencies: list[str]
```

**Pros:** Clear semantics, type-specific properties, easier to extend
**Cons:** Slightly more complex schema

**Decision:** Option B (separate) provides better semantic clarity and extensibility.

---

## 8. Summary and Next Steps

### Key Takeaways

1. **Dublin Core** provides foundational relationship vocabulary (replaces, references, requires)
2. **DCAT** offers hierarchical catalog structures for organizing documentation
3. **Schema.org TechArticle** adds metadata like dependencies and proficiency levels
4. **PROV-O** enables temporal tracking and provenance (useful for versioning)
5. **SIOC** models community content structures (applicable to issue tracking)
6. **CodeMeta** defines software-specific metadata including documentation links

### Recommended Implementation Path

**Phase 1 (1-2 weeks):** Documentation Links
- Add `DocEntity` node type
- Implement `DOCUMENTS`, `REFERENCES`, `REQUIRES` relationships
- Parse code references from markdown text
- Enable queries: "find docs for function", "get prerequisites"

**Phase 2 (2-3 weeks):** Documentation Hierarchy
- Add `DocCollection` for grouped docs
- Implement `IS_PART_OF`, `RELATES_TO` relationships
- Build doc category browser
- Enable queries: "find related docs", "browse guide series"

**Phase 3 (3-4 weeks):** Work Item Integration
- Add `WorkItem` nodes (issues, features)
- Implement `IMPLEMENTS`, `BLOCKS`, `REFERENCES_CODE` relationships
- Integrate with GitHub/Jira APIs
- Enable queries: "find code for feature", "show blocking chain"

**Phase 4 (optional, 4-5 weeks):** Versioning & Provenance
- Add `Activity`, `Agent` nodes
- Implement `DERIVED_FROM`, `GENERATED_BY`, `ATTRIBUTED_TO`
- Parse git history for temporal tracking
- Enable queries: "track doc evolution", "find recent changes"

### Success Metrics

- **Coverage:** >80% of markdown files have at least one REFERENCES relationship
- **Utility:** Users successfully navigate from code to docs (and vice versa)
- **Performance:** Doc queries complete in <50ms
- **Adoption:** Doc relationships improve search relevance (measured via user feedback)

### References

- Dublin Core: https://www.dublincore.org/specifications/dublin-core/dcmi-terms/
- DCAT: https://www.w3.org/TR/vocab-dcat-3/
- Schema.org TechArticle: https://schema.org/TechArticle
- PROV-O: https://www.w3.org/TR/prov-o/
- CodeMeta: https://codemeta.github.io/terms/
- SIOC: https://www.w3.org/wiki/SIOC

---

**End of Research Document**
