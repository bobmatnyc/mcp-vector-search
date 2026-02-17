# Tree-Based KG Generation Without Manual Parsing

**Date**: 2026-02-16
**Objective**: Eliminate ALL manual parsing for knowledge graph generation by using tree-walking approaches for both code and text.

## Executive Summary

The project can achieve **zero manual parsing** for KG relationship extraction by leveraging:

1. **Tree-sitter for code** (already in use) - Direct AST relationships
2. **markdown-it-py for text** (already installed) - Token tree for document sections
3. **Unified tree-walking strategy** - One pattern for both code and docs

This eliminates manual string parsing, regex patterns, and brittle text extraction in favor of structured tree traversal.

---

## Current State Analysis

### What's Already in Place

**tree-sitter integration** (pyproject.toml line 33-34):
```toml
"tree-sitter>=0.20.1",
"tree-sitter-language-pack>=0.9.0",
```

**markdown-it-py installed** (found in .venv):
- Produces token tree with hierarchical structure
- Already used elsewhere in project (rich library dependency)
- Supports heading levels, nesting, parent-child relationships

**AST extraction helpers** (src/mcp_vector_search/parsers/python_helpers/):
- `metadata_extractor.py` - Extracts calls, imports, inheritance from tree-sitter nodes
- `node_extractors.py` - Walks AST to find entities
- `class_skeleton_generator.py` - Traverses class structure

### Current Text Parser (Paragraph-Based)

**File**: `src/mcp_vector_search/parsers/text.py`

**Current approach** (lines 89-142):
- `_extract_paragraphs()` - Manual line-by-line parsing
- Identifies paragraphs by empty line detection
- No hierarchical structure (no heading awareness)
- No relationship extraction (CONTAINS, FOLLOWS)

**Limitation**: Creates flat `CodeChunk` list with no KG relationships.

### Current KG Builder (Manual Parsing)

**File**: `src/mcp_vector_search/core/kg_builder.py`

**Current approach**:
- Processes `CodeChunk` metadata fields (calls, imports, inherits_from)
- Relationships extracted at parse time, not build time
- No document section relationships (docs treated as flat text)

---

## Direct AST Relationships (Code)

### Already Extracted by tree-sitter

**File**: `src/mcp_vector_search/parsers/python_helpers/metadata_extractor.py`

#### 1. Function Calls (CALLS relationship)

**Method**: `extract_function_calls()` (lines 124-179)

**How it works**:
```python
def extract_function_calls(node, source_code: bytes) -> list[str]:
    """Walk AST to find call_expression nodes."""
    calls = []

    def _walk_tree(n):
        if n.type == "call":
            call_name = _extract_call_name(n)
            calls.append(call_name)
        for child in n.children:
            _walk_tree(child)

    _walk_tree(node)
    return calls
```

**Relationship**: `(Function A)-[:CALLS]->(Function B)`

**Example**:
```python
def process_data():
    validate_input()  # Creates CALLS relationship
    transform()       # Creates CALLS relationship
```

#### 2. Imports (IMPORTS relationship)

**Method**: `extract_imports()` (lines 182-329)

**How it works**:
```python
def extract_imports(node, source_code: bytes) -> list[dict]:
    """Find import_statement and import_from_statement nodes."""
    imports = []

    def _walk_tree(n):
        if n.type == "import_statement":
            # Extract: import os, sys
        elif n.type == "import_from_statement":
            # Extract: from typing import List, Dict

        if n.type in ("module", "expression_statement"):
            for child in n.children:
                _walk_tree(child)

    return imports
```

**Relationship**: `(Module A)-[:IMPORTS]->(Module B)`

**Example**:
```python
from typing import List, Dict  # Creates IMPORTS relationship
import os                       # Creates IMPORTS relationship
```

#### 3. Inheritance (INHERITS relationship)

**Method**: `extract_class_bases()` (lines 332-361)

**How it works**:
```python
def extract_class_bases(node, source_code: bytes) -> list[str]:
    """Extract base classes from argument_list."""
    bases = []

    for child in node.children:
        if child.type == "argument_list":
            for arg_child in child.children:
                if arg_child.type in ("identifier", "attribute"):
                    bases.append(arg_child.text.decode("utf-8"))

    return bases
```

**Relationship**: `(Class A)-[:INHERITS]->(Class B)`

**Example**:
```python
class Dog(Animal, Mammal):  # Creates INHERITS relationships
    pass
```

#### 4. Contains (CONTAINS relationship)

**Implicit from AST structure**:

**Parent-child traversal**:
```python
# File contains class
(File)-[:CONTAINS]->(Class)

# Class contains methods
(Class)-[:CONTAINS]->(Method)

# Function contains nested function
(Function)-[:CONTAINS]->(NestedFunction)
```

**Source**: AST node hierarchy (parent/child pointers in tree-sitter)

#### 5. Sequential (FOLLOWS relationship)

**Source**: AST node ordering

**Example**:
```python
def func1():
    pass

def func2():  # FOLLOWS func1
    pass

def func3():  # FOLLOWS func2
    pass
```

**Extraction**: Use `node.next_sibling` in tree-sitter to find sequential order.

---

## Text Tree Structure (Documentation)

### markdown-it-py Token Tree

**Discovered**: markdown-it-py produces hierarchical token tree (tested above).

**Sample output**:
```
heading_open (h1) level=0 nesting=1
  inline level=1
heading_close (h1) level=0 nesting=-1

heading_open (h2) level=0 nesting=1
  inline level=1
heading_close (h2) level=0 nesting=-1

heading_open (h3) level=0 nesting=1
  inline level=1
heading_close (h3) level=0 nesting=-1
```

### Relationships from Token Tree

#### 1. Document Structure (CONTAINS)

**Hierarchy detection**:

```markdown
# Title               (level 1)
## Section 1          (level 2) - Child of Title
### Subsection 1.1    (level 3) - Child of Section 1
## Section 2          (level 2) - Child of Title
```

**Algorithm**:
```python
def extract_section_hierarchy(tokens):
    """Build parent-child relationships from heading levels."""
    sections = []
    stack = []  # Track current hierarchy

    for token in tokens:
        if token.type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2

            # Pop stack until we find parent level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            section = {
                "level": level,
                "parent": stack[-1] if stack else None
            }

            sections.append(section)
            stack.append(section)

    return sections
```

**Relationship**: `(Section A)-[:CONTAINS]->(Section B)`

#### 2. Sequential Order (FOLLOWS)

**Token order in list**:

```python
def extract_sequential_order(tokens):
    """Use token index to determine FOLLOWS relationships."""
    headings = [t for t in tokens if t.type == "heading_open"]

    for i in range(len(headings) - 1):
        current = headings[i]
        next_heading = headings[i + 1]

        # Create FOLLOWS relationship
        # (Section i)-[:FOLLOWS]->(Section i+1)
```

**Relationship**: `(Section A)-[:FOLLOWS]->(Section B)`

#### 3. Code References (REFERENCES)

**Extract from inline code blocks**:

```python
def extract_code_references(tokens):
    """Find `code` tokens and link to code entities."""
    refs = []

    for token in tokens:
        if token.type == "code_inline":
            code_ref = token.content  # e.g., "parse_file"
            refs.append(code_ref)

    return refs
```

**Relationship**: `(DocSection)-[:REFERENCES]->(Function)`

**Example**:
```markdown
The `parse_file` method handles...
  ^-- Creates REFERENCES relationship to parse_file function
```

---

## Unified Tree-Walking Strategy

### Design Pattern

**One approach for both code and text**:

```python
class TreeWalker:
    """Unified tree walker for AST and token trees."""

    def walk(self, tree, callbacks):
        """Generic tree traversal with callbacks."""

        def _traverse(node, depth=0):
            # Pre-order callback
            if callbacks.get("pre"):
                callbacks["pre"](node, depth)

            # Visit children
            children = self._get_children(node)
            for child in children:
                _traverse(child, depth + 1)

            # Post-order callback
            if callbacks.get("post"):
                callbacks["post"](node, depth)

        _traverse(tree)

    def _get_children(self, node):
        """Polymorphic child access."""
        # tree-sitter AST: node.children
        # markdown-it-py: filter by nesting level
        pass
```

### Relationship Extraction Pipeline

**Unified extraction flow**:

```
┌────────────────┐
│  Parse Source  │
└────────┬───────┘
         │
    ┌────▼─────┐
    │ Tree-sitter (code)
    │ markdown-it-py (text)
    └────┬─────┘
         │
┌────────▼───────────┐
│  Tree Walker       │
│  - Pre-order       │
│  - Post-order      │
│  - Level tracking  │
└────────┬───────────┘
         │
┌────────▼────────────┐
│ Relationship Types  │
│ - CONTAINS          │
│ - CALLS             │
│ - IMPORTS           │
│ - INHERITS          │
│ - FOLLOWS           │
│ - REFERENCES        │
└────────┬────────────┘
         │
    ┌────▼─────┐
    │  Kuzu KG  │
    └──────────┘
```

---

## Proposed Implementation

### Phase 1: Enhance Text Parser

**File**: `src/mcp_vector_search/parsers/text.py`

**Add markdown-it-py integration**:

```python
from markdown_it import MarkdownIt

class TextParser(BaseParser):
    def __init__(self):
        super().__init__("text")
        self.md = MarkdownIt()

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse markdown into hierarchical chunks."""

        # Parse with markdown-it-py
        tokens = self.md.parse(content)

        # Extract sections from token tree
        sections = self._extract_sections(tokens)

        # Create CodeChunk with relationships
        chunks = []
        for section in sections:
            chunk = self._create_chunk(
                content=section["content"],
                file_path=file_path,
                start_line=section["start_line"],
                end_line=section["end_line"],
                chunk_type="doc_section",
                parent_chunk_id=section.get("parent_id"),  # CONTAINS
                # Store sequential order for FOLLOWS
            )
            chunks.append(chunk)

        return chunks

    def _extract_sections(self, tokens) -> list[dict]:
        """Walk token tree to extract sections with relationships."""
        sections = []
        stack = []
        current_content = []

        for i, token in enumerate(tokens):
            if token.type == "heading_open":
                level = int(token.tag[1])

                # Finalize previous section
                if stack:
                    sections.append({
                        "content": "".join(current_content),
                        "level": stack[-1]["level"],
                        "parent_id": stack[-2]["id"] if len(stack) > 1 else None,
                        "start_line": stack[-1]["start_line"],
                        "end_line": token.map[0] - 1,
                    })
                    current_content = []

                # Pop stack to find parent
                while stack and stack[-1]["level"] >= level:
                    stack.pop()

                # Get heading text from next inline token
                heading_text = tokens[i + 1].content if i + 1 < len(tokens) else ""

                section_id = f"{file_path}:{token.map[0]}:{heading_text}"
                stack.append({
                    "id": section_id,
                    "level": level,
                    "start_line": token.map[0],
                })

            elif token.type in ("paragraph_open", "list_item_open", "code_block"):
                # Accumulate content for current section
                current_content.append(self._token_to_text(token))

        return sections
```

### Phase 2: Extend KG Schema

**File**: `src/mcp_vector_search/core/knowledge_graph.py`

**Add new relationship type**:

```python
class KnowledgeGraph:
    def create_schema(self):
        """Create Kuzu schema with relationship types."""

        # Existing relationships
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS CALLS(FROM CodeEntity TO CodeEntity)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM CodeEntity TO CodeEntity)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS INHERITS(FROM CodeEntity TO CodeEntity)")

        # NEW: Sequential relationships
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS FOLLOWS(FROM CodeEntity TO CodeEntity)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS FOLLOWS_DOC(FROM DocSection TO DocSection)")

        # NEW: Document-code references
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS REFERENCES(FROM DocSection TO CodeEntity, ref_type STRING)")
```

### Phase 3: Update KG Builder

**File**: `src/mcp_vector_search/core/kg_builder.py`

**Add FOLLOWS extraction**:

```python
class KGBuilder:
    def build_from_chunks(self, chunks: list[CodeChunk]):
        """Extract entities and relationships from chunks."""

        # Group chunks by file
        by_file = {}
        for chunk in chunks:
            by_file.setdefault(chunk.file_path, []).append(chunk)

        # Extract FOLLOWS relationships per file
        for file_path, file_chunks in by_file.items():
            # Sort by line number
            sorted_chunks = sorted(file_chunks, key=lambda c: c.start_line)

            for i in range(len(sorted_chunks) - 1):
                current = sorted_chunks[i]
                next_chunk = sorted_chunks[i + 1]

                # Same depth = sibling, create FOLLOWS
                if current.chunk_depth == next_chunk.chunk_depth:
                    self.kg.add_relationship(
                        from_id=current.chunk_id,
                        to_id=next_chunk.chunk_id,
                        rel_type="FOLLOWS"
                    )
```

---

## Benefits of Tree-Based Approach

### 1. Zero Manual Parsing

**Before** (manual string parsing):
```python
# Brittle regex patterns
heading_pattern = r'^(#{1,6})\s+(.+)$'
matches = re.findall(heading_pattern, line)
```

**After** (tree walking):
```python
# Structured tree traversal
if token.type == "heading_open":
    level = int(token.tag[1])
    parent = stack[-1] if stack else None
```

**Benefit**: No regex, no edge cases, no false positives.

### 2. Consistent Relationship Extraction

**All relationships use tree structure**:
- CONTAINS: Parent-child in tree
- FOLLOWS: Sibling order in tree
- CALLS: AST node type detection
- IMPORTS: AST node type detection
- INHERITS: AST node type detection
- REFERENCES: Token content extraction

**No manual relationship detection** - structure is inherent in tree.

### 3. Language-Agnostic

**Same tree walker works for**:
- Python (tree-sitter-python)
- JavaScript (tree-sitter-javascript)
- Rust (tree-sitter-rust)
- Markdown (markdown-it-py)
- **Any language with tree-sitter grammar**

**No per-language regex patterns needed**.

### 4. Robust to Format Changes

**Tree structure handles**:
- Different indentation styles
- Comment styles
- Code formatting variations
- Markdown dialect differences

**Manual parsers break on format changes** - tree parsers adapt automatically.

### 5. Performance

**Tree-sitter is fast**:
- Incremental parsing (re-parse only changed regions)
- Memory-efficient (streaming)
- C-based parser (native speed)

**markdown-it-py is fast**:
- Single-pass parsing
- No backtracking
- Optimized token stream

---

## Implementation Checklist

### Code Relationships (Already Extracted)

- [x] CALLS - `extract_function_calls()` in metadata_extractor.py
- [x] IMPORTS - `extract_imports()` in metadata_extractor.py
- [x] INHERITS - `extract_class_bases()` in metadata_extractor.py
- [x] CONTAINS - AST parent-child structure (implicit)
- [ ] FOLLOWS - Need to add sequential extraction

### Document Relationships (To Implement)

- [ ] CONTAINS - Add heading hierarchy extraction to text.py
- [ ] FOLLOWS - Add sequential section ordering to text.py
- [ ] REFERENCES - Add inline code reference extraction to text.py

### Infrastructure Updates

- [ ] Add FOLLOWS relationship to KG schema (knowledge_graph.py)
- [ ] Add REFERENCES relationship to KG schema (knowledge_graph.py)
- [ ] Update KGBuilder to process FOLLOWS (kg_builder.py)
- [ ] Update KGBuilder to process REFERENCES (kg_builder.py)
- [ ] Add markdown-it-py section extraction (text.py)

---

## Specific Libraries and Tools

### For Code (Already Using)

**tree-sitter** (v0.20.1+):
- C-based parser with Python bindings
- Grammars: Python, JavaScript, TypeScript, Rust, Go, Java, C#, Ruby, PHP, Dart
- **Location**: pyproject.toml line 33

**tree-sitter-language-pack** (v0.9.0+):
- Pre-built grammar binaries
- **Location**: pyproject.toml line 34

### For Text (Already Available)

**markdown-it-py**:
- CommonMark compliant markdown parser
- Token tree with nesting information
- **Status**: Already installed (via rich dependency)
- **API**: `MarkdownIt().parse(text)` returns token list

### Alternative (Not Recommended)

**tree-sitter-markdown** (v0.5.1):
- Exists but "not recommended where correctness is important"
- Designed for syntax highlighting, not parsing
- Many inaccuracies according to maintainers

**Recommendation**: Use markdown-it-py for text (more mature, better documented).

---

## Files That Handle Text/Markdown Currently

### Parser Layer

**`src/mcp_vector_search/parsers/text.py`**:
- Current: Paragraph-based manual parsing
- Needed: Add markdown-it-py tree extraction
- Lines 89-142: `_extract_paragraphs()` to be replaced

**`src/mcp_vector_search/parsers/base.py`**:
- Base parser interface
- No changes needed (tree-based parsing is internal)

### KG Layer

**`src/mcp_vector_search/core/kg_builder.py`**:
- Current: Processes CodeChunk metadata
- Needed: Add FOLLOWS relationship extraction
- Line 122+: `KGBuilder` class

**`src/mcp_vector_search/core/knowledge_graph.py`**:
- Current: Schema definition
- Needed: Add FOLLOWS and REFERENCES relationship tables
- Line 36+: `DocSection` dataclass (already exists)

### Analysis Layer

**`src/mcp_vector_search/analysis/reporters/markdown.py`**:
- Generates markdown reports (output, not parsing)
- No changes needed

---

## Next Steps

### Immediate Actions

1. **Implement markdown-it-py section extraction in text.py**:
   - Replace `_extract_paragraphs()` with `_extract_sections()`
   - Use token tree to build parent-child hierarchy
   - Extract inline code references

2. **Add FOLLOWS to KG schema**:
   - Update `knowledge_graph.py` with FOLLOWS relationship table
   - Add FOLLOWS_DOC for document sections

3. **Update KGBuilder**:
   - Add sequential relationship extraction for code entities
   - Add sequential relationship extraction for doc sections

### Testing Strategy

**Unit tests**:
- Test markdown-it-py section extraction with sample markdown
- Test FOLLOWS relationship creation from chunk ordering
- Test REFERENCES extraction from inline code blocks

**Integration tests**:
- Index sample project with markdown docs
- Query KG for CONTAINS relationships (heading hierarchy)
- Query KG for FOLLOWS relationships (sequential order)
- Query KG for REFERENCES relationships (doc-to-code links)

### Success Criteria

- [ ] Zero regex patterns in relationship extraction
- [ ] All relationships from tree structure
- [ ] Markdown docs have hierarchical structure in KG
- [ ] Code entities have sequential ordering in KG
- [ ] Documentation links to code entities in KG

---

## Conclusion

**Zero manual parsing is achievable** by:

1. Using tree-sitter AST for code relationships (already 90% done)
2. Using markdown-it-py token tree for document relationships (needs implementation)
3. Applying unified tree-walking strategy for both (design pattern)

**Key insight**: Both code and text are trees. Walk the tree, extract relationships from structure, no parsing needed.

**Immediate win**: Add markdown-it-py section extraction to text.py (1-2 hours of work, eliminates all paragraph parsing logic).

**Long-term win**: Language-agnostic, robust, fast, maintainable relationship extraction with no manual patterns.
