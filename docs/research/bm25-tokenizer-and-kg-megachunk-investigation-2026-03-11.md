# Investigation: BM25 Tokenizer and KG Mega-Chunk Problems

**Date:** 2026-03-11
**Scope:** `search_hybrid` quality issues — BM25 result count disparity and KnowledgeGraph mega-chunk
**Files examined:**
- `src/mcp_vector_search/core/bm25_backend.py`
- `src/mcp_vector_search/core/bm25_builder.py`
- `src/mcp_vector_search/core/search.py`
- `src/mcp_vector_search/mcp/hybrid_search_handler.py`
- `src/mcp_vector_search/parsers/python.py`
- `src/mcp_vector_search/parsers/python_helpers/node_extractors.py`
- `src/mcp_vector_search/parsers/python_helpers/class_skeleton_generator.py`
- `src/mcp_vector_search/parsers/python_helpers/fallback_parser.py`
- `src/mcp_vector_search/core/knowledge_graph.py`

**Installed versions:** lancedb 0.29.2, rank-bm25 0.2.2, tree-sitter 0.25.2, tree-sitter-language-pack 0.13.0

---

## Investigation 1: BM25 / Full-Text Search Index Configuration

### 1. Where is the BM25 index created?

There is **no LanceDB FTS index** (`create_fts_index` / `create_scalar_index`) anywhere in the codebase. The search grep for `create_fts_index`, `create_scalar_index`, `FtsIndexConfig`, `tokenizer`, and `analyzer` returned zero matches in LanceDB-related files.

The system uses a **custom BM25 implementation** backed by the `rank-bm25` library (`BM25Okapi`), stored as a pickle file at:

```
<project_root>/.mcp-vector-search/bm25_index.pkl
```

**Index build entry point:**
- `src/mcp_vector_search/core/bm25_builder.py` — `build_bm25_index()` (Phase 3 of indexing)
- `src/mcp_vector_search/core/bm25_backend.py` — `BM25Backend.build_index()` / `BM25Backend.search()`

### 2. What tokenizer/analyzer is configured?

The tokenizer is a custom static method: `BM25Backend._tokenize()` at **`bm25_backend.py:301-329`**.

```python
@staticmethod
def _tokenize(text: str) -> list[str]:
    import re
    text_lower = text.lower()
    # First pass: preserve dotted/hyphenated/slashed compound identifiers as single tokens
    compound_tokens = re.findall(r"[\w][\w.\-/]*[\w]", text_lower)
    # Second pass: also index individual word components for partial matching
    word_tokens = re.findall(r"\w+", text_lower)
    # Combine: compound forms first, then individual components not already in compound
    tokens = compound_tokens + [t for t in word_tokens if t not in compound_tokens]
    tokens = [t for t in tokens if t]
    return tokens
```

**Key behavior verified by direct execution:**

```
Input:  "find_by_tag_docs _detect_optimal_write_buffer_size decay_half_life_days"
compound_tokens: ['find_by_tag_docs', '_detect_optimal_write_buffer_size', 'decay_half_life_days']
word_tokens:     ['find_by_tag_docs', '_detect_optimal_write_buffer_size', 'decay_half_life_days']
combined tokens: ['find_by_tag_docs', '_detect_optimal_write_buffer_size', 'decay_half_life_days']
```

**The tokenizer does NOT split on underscores.** Snake_case identifiers are preserved as single compound tokens verbatim. The `\w` character class in Python regex matches `[a-zA-Z0-9_]`, so underscores bind identifier components together.

### 3. What column is being FTS-indexed?

The BM25 corpus text for each chunk is assembled at **`bm25_backend.py:90-117`** by combining:

```python
combined_text = " ".join([
    chunk["content"],     # primary: full code content
    chunk["name"],        # repeated TWICE for boosting
    chunk["name"],
    chunk["file_path"],   # for path-based searches
    chunk["chunk_type"],  # e.g. "function", "class"
])
```

Columns come from `chunks.lance` (or `vectors.lance` in the lazy-build path). The column is `content` for the body, supplemented by `name` (function/class name), `file_path`, and `chunk_type`.

### 4. Existing handling of identifiers, camelCase, or snake_case?

- **Snake_case:** Preserved as-is. `find_by_tag_docs` indexes as the single token `find_by_tag_docs`.
- **camelCase:** No camelCase splitting. `DatabaseConnection` indexes as the single token `databaseconnection` (lowercased).
- **Dotted/hyphenated identifiers:** Explicitly preserved (e.g. `getstream.io` stays as one token AND `getstream`, `io` are indexed separately).
- **No sub-word splitting** for any identifier style.

### 5. Root cause of the BM25 result count disparity (3-10 vs 23-30)

The result count disparity is **correct and expected behavior**, not a tokenizer bug.

**BM25 is an exact-token matcher.** A query of `find_by_tag_docs` tokenizes to the single token `find_by_tag_docs`. `BM25Okapi.get_scores()` returns a score > 0.0 only for corpus documents that contain that exact token. The filter at `bm25_backend.py:188` enforces `score > 0.0`. For a specific identifier like `find_by_tag_docs`, only:
- The function chunk itself
- Any chunks that call or reference it (callers, docstrings, test files)

...will match. That naturally produces 3-10 results.

**Semantic search** returns 23-30 because dense-vector embeddings capture conceptual similarity — a query about `find_by_tag_docs` is semantically similar to any code about tags, documents, lookup, filtering, etc.

The result count difference is a property of keyword vs. semantic retrieval, not a defect. The BM25 results for identifier queries are **higher precision** (all matches are genuinely relevant); the semantic results have higher recall but lower precision.

**Where this becomes a problem:** The KG mega-chunk (Investigation 2) pollutes BM25 ranking for any KG-related term. That is the actual actionable defect.

---

## Investigation 2: KnowledgeGraph Mega-Chunk Problem

### 1. How is `knowledge_graph.py` currently chunked?

Chunking uses the **tree-sitter Python parser** (`PythonParser` + `ClassExtractor` + `FunctionExtractor`). Tree-sitter IS available at runtime (tree-sitter-language-pack 0.13.0 is installed and functional in the project venv).

The parser visits AST nodes recursively (`python.py:179-219`). For each `class_definition` node, `ClassExtractor.extract()` is called, producing **one chunk** for the entire class with:
- `content` = class skeleton (method signatures + docstrings, no bodies) from `ClassSkeletonGenerator`
- `start_line` = first line of the class node
- `end_line` = last line of the class node

Methods inside the class are then visited as `function_definition` children, each producing **individual method chunks**.

### 2. Is there a max_lines or max_tokens limit per chunk?

**No.** There is no maximum chunk size enforced anywhere in the chunking pipeline. The parsers extract AST-level boundaries (class definition, function definition) and create one chunk per syntactic unit, regardless of how large that unit is. No `max_lines`, `max_tokens`, or `MAX_CHUNK_SIZE` constant exists in the chunker or parsers.

### 3. Why does `knowledge_graph.py` produce such a large single chunk?

**Verified by live parsing** (`.venv/bin/python3`):

```
Total chunks from knowledge_graph.py: 87

class      KnowledgeGraph    lines  222-5348  lines_in_file=5127  content_len=5995
```

The `KnowledgeGraph` class spans lines 222-5348 (5,127 source lines, 73 methods). The CLASS chunk has:
- `start_line = 222`, `end_line = 5348` — the full span of the class in the file
- `content` = skeleton of 245 lines (method signatures with `...` bodies), 5,995 characters

The skeleton is **not** the full class body — `ClassSkeletonGenerator` correctly strips method bodies. However, the chunk metadata (`start_line`, `end_line`) reflects the entire class span. The skeleton itself contains ALL 73 method signatures and their one-line docstrings, making it a ~245-line, ~6 KB BM25 document that contains tokens for every single method name in the class.

**BM25 impact:** The KnowledgeGraph class skeleton chunk contains tokens like `find_by_tag_docs`, `add_entity`, `add_relationship`, `trace_execution_flow`, etc. — every method name in the class. Any BM25 query for a KG method name will score this class skeleton chunk highly because it contains all KG terms, giving it outsized IDF advantage. It will appear as a top result alongside (or above) the actual method chunk.

### 4. Why does the chunk problem exist? Is this a class-body split issue?

The architecture is correct in intent: the class skeleton chunk gives a high-level overview of the class interface; individual method chunks provide searchable implementations. The problem is **the class skeleton chunk is too large** because:

1. `KnowledgeGraph` has 73 methods. The skeleton includes all 73 signatures + docstrings.
2. `_create_schema` (lines 327-888, 562 source lines, 19,562 char content) is itself a massive method chunk — it contains inline Kuzu schema DDL strings, making it the single largest chunk in the codebase.

There is **no split at method boundaries inside the class skeleton**. The class chunk is always one unit for the entire class, regardless of class size.

### 5. Other files with similar mega-chunk problems

**Live scan of all `chunk.end_line - chunk.start_line > 200` cases:**

| Chunk type | Name | File | Source lines spanned | Content chars |
|---|---|---|---|---|
| `function` | `get_all_scripts` | `scripts.py` | 6,684 | 244,868 |
| `class` | `KnowledgeGraph` | `knowledge_graph.py` | 5,127 | 5,995 |
| `class` | `KGBuilder` | `kg_builder.py` | 4,748 | 5,210 |
| `class` | `HTMLReportGenerator` | `html_report.py` | 2,857 | 1,706 |
| `class` | `SemanticIndexer` | `indexer.py` | 2,550 | 4,006 |
| `class` | `SemanticSearchEngine` | `search.py` | 1,736 | 2,731 |
| `class` | `LanceVectorDatabase` | `lancedb_backend.py` | 1,435 | 2,115 |
| `class` | `VectorsBackend` | `vectors_backend.py` | 1,338 | 2,219 |
| `class` | `ChunksBackend` | `chunks_backend.py` | 1,206 | 2,129 |
| `function` | `_generate_scripts` | `html_report.py` | 1,175 | 44,590 |
| `function` | `create_app` | `server.py` | 908 | 35,293 |
| `function` | `_generate_styles` | `html_report.py` | 877 | 16,430 |
| `function` | `_create_schema` | `knowledge_graph.py` | 562 | 19,562 |

**The `get_all_scripts` chunk** (scripts.py, lines 29-6712, 244,868 chars) is by far the worst offender. It is a single Python function returning a 6,000+ line JavaScript string literal. This single BM25 document will dominate any search query that matches JavaScript D3 visualization terms.

**`_create_schema`** (knowledge_graph.py:327-888) is 562 source lines containing Kuzu schema DDL strings — it dominates BM25 for any Kuzu or schema-related query.

**`_generate_scripts`** and `create_app` are similar: large functions that contain embedded multi-line string literals.

---

## Key Findings Summary

### Investigation 1: BM25

| Question | Finding |
|---|---|
| FTS index location | No LanceDB FTS index. Custom rank-bm25 BM25Okapi at `.mcp-vector-search/bm25_index.pkl` |
| Tokenizer | `BM25Backend._tokenize()` at `bm25_backend.py:301`. Two-pass regex: compound then word tokens |
| Tokenizer splits on `_`? | **No.** `\w` matches `_`; snake_case is preserved as a single token |
| Column indexed | `content` + `name` (×2) + `file_path` + `chunk_type` concatenated |
| camelCase or snake_case handling | None. Single-token preservation only; no sub-word splitting |
| Why 3-10 vs 23-30 results? | **Expected behavior.** BM25 = exact keyword match; semantic = conceptual similarity. Both are correct |

### Investigation 2: KG Mega-Chunk

| Question | Finding |
|---|---|
| How is `knowledge_graph.py` chunked? | Tree-sitter AST: 1 class chunk (skeleton) + 73 individual method chunks = 87 chunks total |
| Max chunk size limit? | **None.** No `max_lines`, `max_tokens`, or size guard in any parser or chunker |
| Why is the class chunk large? | Class skeleton includes all 73 method signatures. `start_line=222, end_line=5348` (full class span) |
| Is the content the full body? | No — skeleton only (245 lines, 5,995 chars). But the metadata range `222-5348` is misleading |
| BM25 impact | Skeleton contains every method name as a token → scores highly for any KG-related query |
| Other mega-chunks? | Yes: `scripts.py:get_all_scripts` (244,868 chars), `_generate_scripts` (44,590), `create_app` (35,293), `_create_schema` (19,562), `_generate_styles` (16,430) |

---

## Actionable Fix Recommendations

### Fix 1: BM25 — Add sub-word tokenization for identifiers

While the tokenizer correctly preserves `find_by_tag_docs` as a token, it does **not** also index `find`, `by`, `tag`, `docs` as separate tokens. This means a query of `tag docs` or `find tag` will NOT match the `find_by_tag_docs` function. Adding a camelCase/snake_case splitter as a supplementary pass (not a replacement) would increase recall.

**Suggested addition to `_tokenize()` in `bm25_backend.py`:**
```python
# Third pass: split snake_case and camelCase into sub-words
subword_tokens = re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", text_lower)
tokens = compound_tokens + [t for t in word_tokens if t not in compound_tokens] \
       + [t for t in subword_tokens if t not in set(compound_tokens + word_tokens)]
```

This would add `find`, `by`, `tag`, `docs` alongside `find_by_tag_docs`.

### Fix 2: KG — Limit class skeleton chunk size

Add a `MAX_SKELETON_METHODS` guard in `ClassSkeletonGenerator.generate_from_node()`. If the class has more than N methods (e.g. 30), split the skeleton into multiple class-shard chunks (e.g. `KnowledgeGraph [part 1/3]`).

Alternatively, drop the class skeleton chunk entirely for classes exceeding a threshold. The individual method chunks already provide full searchability for every method.

### Fix 3: KG — Cap mega-method chunks

Add a `MAX_CHUNK_LINES` constant (e.g. 300) in `FunctionExtractor.extract()`. If a function's source span exceeds the limit, emit multiple sub-chunks splitting at logical boundaries (blank lines, comment blocks). This would apply to `_create_schema`, `get_all_scripts`, `_generate_scripts`, `create_app`, and `get_visualization_data`.

### Fix 4: Exclude non-source files from indexing

`scripts.py` (6,712 lines of JS-in-Python) is a template file, not application logic. It should be excluded from the chunk index via the file discovery filter. Its presence creates a single 244,868-character BM25 document that will dominate any search matching D3/JavaScript terms.
