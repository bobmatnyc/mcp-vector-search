# Phase 1 Text Relationships Usage Guide

## Quick Start

### Rebuild Knowledge Graph with Doc Support

```bash
# Full rebuild (required after Phase 1 implementation)
mcp-vector-search kg build --force

# Expected output:
# Knowledge Graph Statistics
# ┌─────────────────┬───────┐
# │ Metric          │ Count │
# ├─────────────────┼───────┤
# │ Code Entities   │ 4067  │
# │ Doc Sections    │ 10482 │
# │ Calls           │ 5410  │
# │ Imports         │ 4066  │
# │ Inherits        │ 56    │
# │ Contains        │ 2713  │
# │ References      │ 146   │
# │ Documents       │ 0     │
# │ Follows         │ 4476  │
# └─────────────────┴───────┘
```

### View Enhanced Statistics

```bash
mcp-vector-search kg stats

# Output shows:
# - Total Entities (code + docs)
# - Code Entities (functions, classes, modules)
# - Doc Sections (markdown headers)
# - All relationship types including new ones
```

## Understanding the New Entity Types

### DocSection

Represents a documentation section (markdown header):

**Properties:**
- `id`: Unique identifier (generated from chunk_id + line number)
- `name`: Section title (markdown header text)
- `file_path`: Source markdown file
- `level`: Header level (1-6, where 1 is `#`, 2 is `##`, etc.)
- `line_start`: Starting line number
- `line_end`: Ending line number
- `doc_type`: Type of documentation (currently "section")

**Example:**
```markdown
# MCP Vector Search  ← Level 1 DocSection
## Features          ← Level 2 DocSection
### Installation    ← Level 3 DocSection
```

## Understanding the New Relationship Types

### REFERENCES (Doc → Code)

Created when a doc section mentions a code entity using backticks.

**Example:**
```markdown
## Usage

Use the `VectorDatabase` class for storage.
Call `search()` method to find code.
```

**Relationships Created:**
- DocSection("Usage") → REFERENCES → CodeEntity("VectorDatabase")
- DocSection("Usage") → REFERENCES → CodeEntity("search")

### FOLLOWS (Doc → Doc)

Links documentation sections in reading order within a file.

**Example:**
```markdown
# Title            ← Section 1
## Introduction    ← Section 2
### Background     ← Section 3
## Installation    ← Section 4
```

**Relationships Created:**
- Section 1 → FOLLOWS → Section 2
- Section 2 → FOLLOWS → Section 3
- Section 3 → FOLLOWS → Section 4

### DOCUMENTS (Doc → Code)

Placeholder for Phase 2. Will use semantic similarity to match docs to code.

**Coming in Phase 2:**
```python
# Will analyze embeddings to detect:
DocSection("Vector Database Guide") → DOCUMENTS → CodeEntity("VectorDatabase")
# Even without explicit backtick references
```

## Query Examples

### Find Docs that Reference a Code Entity

```bash
# Query for documentation mentioning VectorDatabase
mcp-vector-search kg query "VectorDatabase" --relationship references

# Expected output:
# Documentation References for 'VectorDatabase'
# ┌─────────────────────────┬──────────────────┬──────────┐
# │ Doc Title               │ File             │ Lines    │
# ├─────────────────────────┼──────────────────┼──────────┤
# │ Quick Start             │ README.md        │ 40-60    │
# │ API Reference           │ docs/api.md      │ 100-150  │
# └─────────────────────────┴──────────────────┴──────────┘
```

### Navigate Document Structure

```bash
# Find sections that follow a specific section
# (Coming soon - requires query enhancement)
```

### Find Related Entities (Code + Docs)

```bash
# This includes both code and doc entities
mcp-vector-search kg query "search" --hops 2

# Returns:
# - Code entities related to "search" (via calls, imports)
# - Doc sections that reference "search" (via REFERENCES)
# - Doc sections that follow related sections (via FOLLOWS)
```

## Python API Usage

### Query Doc-Code References

```python
from mcp_vector_search.core.knowledge_graph import KnowledgeGraph
from pathlib import Path

async def find_docs_for_code(entity_name: str):
    """Find all docs that reference a code entity."""
    kg = KnowledgeGraph(Path(".mcp-vector-search/knowledge_graph"))
    await kg.initialize()

    # Get docs that reference this entity
    docs = await kg.get_doc_references(entity_name, relationship="references")

    for doc in docs:
        print(f"{doc['title']} ({doc['file_path']}:{doc['line_start']})")

    await kg.close()

# Usage
import asyncio
asyncio.run(find_docs_for_code("VectorDatabase"))
```

### Add Custom DocSection

```python
from mcp_vector_search.core.knowledge_graph import DocSection, KnowledgeGraph
from pathlib import Path

async def add_custom_doc():
    """Add a custom documentation section."""
    kg = KnowledgeGraph(Path(".mcp-vector-search/knowledge_graph"))
    await kg.initialize()

    doc = DocSection(
        id="custom-doc-1",
        name="Custom Documentation",
        file_path="docs/custom.md",
        level=2,
        line_start=10,
        line_end=50,
        doc_type="section"
    )

    await kg.add_doc_section(doc)
    await kg.close()

# Usage
import asyncio
asyncio.run(add_custom_doc())
```

### Create Custom Relationships

```python
from mcp_vector_search.core.knowledge_graph import CodeRelationship, KnowledgeGraph
from pathlib import Path

async def link_doc_to_code():
    """Create a REFERENCES relationship."""
    kg = KnowledgeGraph(Path(".mcp-vector-search/knowledge_graph"))
    await kg.initialize()

    rel = CodeRelationship(
        source_id="doc:custom-doc-1:10",  # Doc section ID
        target_id="code-entity-123",       # Code entity ID
        relationship_type="references"
    )

    await kg.add_relationship(rel)
    await kg.close()

# Usage
import asyncio
asyncio.run(link_doc_to_code())
```

## Integration with Existing Features

### Works with Vector Search

```bash
# 1. Search for code
mcp-vector-search search "database connection"

# 2. Find related docs in KG
mcp-vector-search kg query "VectorDatabase" --relationship references

# 3. Get full context: code implementation + documentation
```

### Works with Analysis Tools

```bash
# Find complexity hotspots
mcp-vector-search analyze hotspots

# Check which hotspots have documentation
mcp-vector-search kg query "complex_function" --relationship references
```

### Works with Wiki

```bash
# Generate codebase wiki (includes docs)
mcp-vector-search wiki generate

# Navigate wiki with doc relationships
```

## Use Cases

### 1. Find Undocumented Code

```python
# Pseudo-code (coming in Phase 2)
# Find code entities with no REFERENCES relationships
SELECT e.name
FROM CodeEntity e
WHERE NOT EXISTS (
    MATCH (d:DocSection)-[:REFERENCES]->(e)
)
```

### 2. Documentation Coverage Report

```python
# Pseudo-code
# Calculate % of code entities with doc references
total_code = count(CodeEntity)
documented_code = count(DISTINCT CodeEntity with REFERENCES)
coverage = documented_code / total_code * 100
```

### 3. Navigate Related Documentation

```python
# Find all docs in a section's context
# 1. Get current section
# 2. Follow FOLLOWS relationships
# 3. Get referenced code
# 4. Find other docs that reference same code
```

### 4. Documentation Quality Analysis

```python
# Identify docs with:
# - Many REFERENCES (well-documented sections)
# - Few REFERENCES (might need more examples)
# - Broken references (references to non-existent code)
```

## Debugging Tips

### Check if Docs Were Indexed

```bash
# Verify text chunks exist
mcp-vector-search status

# Look for "text: X chunks" in output
# Should show ~10,652 text chunks
```

### Validate Header Extraction

```python
# Test extraction logic
python3 test_doc_extraction.py

# Expected output:
# Testing header extraction:
#   Level 1: MCP Vector Search (line 1)
#   Level 2: ✨ Features (line 5)
#   ...
```

### Check Relationship Counts

```bash
# If References = 0, check:
# 1. Are code entities being matched?
# 2. Are backtick references being extracted?
# 3. Run with debug logging enabled
```

### Rebuild if Needed

```bash
# Force full rebuild
mcp-vector-search kg build --force

# Test with small sample
mcp-vector-search kg build --force --limit 100
```

## Performance Considerations

### Build Time
- **Full build**: ~2-3 minutes (15,026 chunks)
- **Text processing**: Adds ~30% overhead
- **Header extraction**: O(n) per chunk (regex matching)
- **Reference resolution**: O(m) lookups (where m = # of refs)

### Database Size
- **Before Phase 1**: ~12MB (code only)
- **After Phase 1**: ~52MB (+10,482 doc sections, +4,622 relationships)
- **Storage growth**: ~40MB for documentation layer

### Query Performance
- **Doc reference queries**: O(1) with indexed lookups
- **FOLLOWS traversal**: O(k) where k = # of sections in file
- **Cross-entity queries**: O(h) where h = hop count

## Limitations & Future Work

### Current Limitations

1. **No Semantic Matching**
   - DOCUMENTS relationships are empty
   - Relies only on explicit backtick references
   - **Phase 2 will add**: Embedding-based doc-code matching

2. **Simple Reference Extraction**
   - Only captures `backtick` references
   - Misses natural language mentions
   - **Future**: Add NER for class/function name detection

3. **Single File FOLLOWS**
   - No cross-file doc relationships
   - **Future**: Link related docs across files

4. **No Hierarchy Inference**
   - Flat FOLLOWS (sequential only)
   - **Future**: Parent-child relationships for nested sections

### Planned Phase 2 Features

1. **Semantic DOCUMENTS Relationships**
   ```python
   # Use embeddings to match docs to code
   similarity = cosine_similarity(doc_embedding, code_embedding)
   if similarity > 0.7:
       create_relationship(doc, code, "DOCUMENTS")
   ```

2. **Topic Extraction**
   ```python
   # Extract technical topics from doc sections
   topics = extract_topics(doc_section.content)
   for topic in topics:
       create_doc_section(topic, doc_type="topic")
   ```

3. **Co-occurrence Analysis**
   ```python
   # Detect frequently co-mentioned entities
   if mentioned_together(doc, code, threshold=5):
       strengthen_relationship(doc, code, weight=2.0)
   ```

4. **Enhanced Queries**
   ```bash
   # Find most documented code
   mcp-vector-search kg top-documented --limit 10

   # Find orphaned code (no docs)
   mcp-vector-search kg orphaned-code

   # Suggest documentation improvements
   mcp-vector-search kg doc-gaps
   ```

## Support & Feedback

### Report Issues
- GitHub Issues: mcp-vector-search/issues
- Include KG stats output
- Provide sample markdown with backtick refs

### Request Features
- Phase 2 feature requests welcome
- Suggest new relationship types
- Propose query enhancements

### Contribute
- Add new doc_type values
- Improve reference extraction regex
- Enhance query methods

---

**Last Updated**: February 15, 2025
**Version**: 2.2.21
**Phase**: 1 Complete ✅
