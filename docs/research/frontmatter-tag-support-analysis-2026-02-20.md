# Frontmatter Tag Support Analysis for KG

**Date**: 2026-02-20
**Scope**: Assessment of existing infrastructure for frontmatter tag support in knowledge graph

## Quick Findings

### 1. KG Tag Schema - YES, EXISTS
**Location**: `src/mcp_vector_search/core/knowledge_graph.py`

```python
class Tag:
    """Tag node for knowledge graph."""
    id: str  # Tag identifier
    name: str  # Tag name
```

**Status**: Fully implemented with schema tables:
- `Tag` node table created in schema
- Ready to store tag metadata

### 2. HAS_TAG Relationship - YES, EXISTS
**Location**: `src/mcp_vector_search/core/knowledge_graph.py:365-379`

```sql
CREATE REL TABLE IF NOT EXISTS HAS_TAG (
    FROM DocSection TO Tag,
    relationship_score FLOAT
)
```

**Status**:
- Fully implemented relationship type
- Used for DocSection → Tag relationships
- Also have `DEMONSTRATES` relationship for language tags

### 3. Chunk Processor - NO frontmatter parsing
**Location**: `src/mcp_vector_search/core/chunk_processor.py`

**Findings**:
- No frontmatter extraction code
- Calls parser registry for file-specific parsing
- Deduplicates and enriches chunks with git blame, subproject info
- Does NOT handle YAML frontmatter itself

**Implication**: Frontmatter parsing must happen in specific parsers (text/markdown) or kg_builder

### 4. CodeChunk Model - NO tags field
**Location**: `src/mcp_vector_search/core/models.py:10-89`

**Current Fields**:
- `content`, `file_path`, `start_line`, `end_line`, `language`
- `chunk_type`, `function_name`, `class_name`, `docstring`
- `complexity_score`, `chunk_id`, `parent_chunk_id`, `child_chunk_ids`
- `decorators`, `parameters`, `return_type`, `type_annotations`
- `subproject_name`, `subproject_path`
- `nlp_keywords`, `nlp_code_refs`, `nlp_technical_terms`
- `last_author`, `last_modified`, `commit_hash`

**Missing**: `tags: list[str]` field

### 5. KG Builder - YES, HAS frontmatter parsing
**Location**: `src/mcp_vector_search/core/kg_builder.py:1530-1560`

**Comprehensive frontmatter extraction exists**:

```python
def _extract_frontmatter(self, content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content.

    Frontmatter format:
    ---
    title: "Document Title"
    tags: [api, rest]
    related: [other-doc.md]
    category: guides
    ---
    """
```

**Implementation Details**:
- Uses `yaml.safe_load()` to parse YAML
- Extracts `tags` field (supports list or string)
- Flattens nested lists
- Handles `related` field for LINKS_TO relationships
- Called in `_extract_doc_sections()` at line 1148 and 1304
- Already creates Tag nodes and HAS_TAG relationships (lines 1373-1387, 1375)

**Tag Processing**:
- Extracts frontmatter tags from FULL file (not chunks)
- Creates Tag entities via `kg.add_tag()`
- Creates HAS_TAG relationships from DocSection → Tag
- Also creates language tags as `lang:{language}`

### 6. Text/Markdown Parser - MINIMAL frontmatter support
**Location**: `src/mcp_vector_search/parsers/text.py`

**Current Implementation**:
- No frontmatter parsing
- Paragraph-based chunking using `\n\n` splits
- Line-based fallback chunking
- Supported extensions: `.txt`, `.md`, `.markdown`

**Limitation**: Frontmatter is parsed by kg_builder (after chunks created), not by parser

## Current Architecture Flow

```
Markdown File with Frontmatter
    ↓
TextParser.parse_file()
    ↓ (creates CodeChunk objects - NO tags extracted yet)
ChunkProcessor.parse_file()
    ↓ (enriches chunks, no frontmatter handling)
KGBuilder._extract_doc_sections()
    ↓
_extract_frontmatter() reads FULL file
    ↓
Extracts tags from YAML
    ↓
kg.add_tag() + HAS_TAG relationships
```

## Work Assessment

### Scope: MODERATE
Frontmatter tag support infrastructure 60% complete. Missing pieces are:

1. **CodeChunk tags field** (TRIVIAL - 5 min)
   - Add `tags: list[str] = None` to CodeChunk dataclass
   - Update `__post_init__()`, `to_dict()`, `from_dict()`

2. **Text parser frontmatter extraction** (TRIVIAL - 15 min)
   - Add frontmatter extraction to TextParser
   - Populate CodeChunk.tags before returning chunks
   - Filter frontmatter from chunk content

3. **Pass tags through pipeline** (TRIVIAL - 10 min)
   - ChunkProcessor already passes chunks through
   - KGBuilder already has tag extraction
   - Just ensure tags flow from parser → chunk → KG

4. **Testing** (MODERATE - 1-2 hours)
   - Test frontmatter parsing with various YAML formats
   - Test tag creation and HAS_TAG relationships
   - Test edge cases (empty tags, nested lists, strings)
   - Integration tests with real markdown files

### Blockers: NONE
- All infrastructure exists in KG
- YAML parsing already done
- Just need to wire frontmatter through parser layer

### Risk Factors: LOW
- Frontmatter extraction logic proven (already in kg_builder)
- Tag schema stable
- No breaking changes needed

## Implementation Checklist

- [ ] Add `tags` field to `CodeChunk` in models.py
- [ ] Update `CodeChunk.__post_init__()` to initialize tags
- [ ] Update `CodeChunk.to_dict()` to include tags
- [ ] Update `CodeChunk.from_dict()` to load tags
- [ ] Add frontmatter extraction to TextParser
- [ ] Test with markdown files containing frontmatter
- [ ] Verify HAS_TAG relationships in KG
- [ ] Update documentation with frontmatter format

## Example Frontmatter Format

```markdown
---
title: "API Authentication Guide"
tags: [authentication, security, oauth2]
related: [jwt-tokens.md, session-management.md]
category: guides
---

# API Authentication

Content here...
```

## Next Steps

1. Confirm exact requirements for frontmatter format
2. Decide: Should chunk.tags duplicate KG tags or be separate?
3. Plan integration test strategy
4. Consider: Should frontmatter be stripped from chunk content?
