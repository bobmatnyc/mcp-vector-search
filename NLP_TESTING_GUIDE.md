# NLP Entity Extraction Testing Guide

## Quick Start

### 1. Install Dependencies
```bash
uv pip install yake==0.4.8
```

### 2. Reindex Database
```bash
# Delete old database
rm -rf .mcp-vector-search/*.lance

# Reindex with NLP extraction enabled
mcp-vector-search index --force
```

### 3. Test Search Quality

#### Baseline Weak Queries
These queries had low precision before NLP extraction:

```bash
# Query 1: "how embeddings are generated"
# Previous score: 0.639
# Target: 0.75+
mcp-vector-search search "how embeddings are generated" --limit 5

# Query 2: "class for database connections"
# Previous score: 0.695
# Target: 0.80+
mcp-vector-search search "class for database connections" --limit 5
```

#### Expected Improvements
- Results should include more relevant chunks with higher similarity scores
- Code references in docstrings (e.g., `` `LanceDB` ``, `` `VectorDatabase` ``) should boost relevance
- Technical terms (CamelCase, ACRONYMS) should improve matching

### 4. Verify NLP Fields in Database

#### Check Chunk Metadata
```python
from mcp_vector_search.core.database import VectorDatabase
from pathlib import Path

# Load database
db = VectorDatabase(Path(".mcp-vector-search"))

# Search for a chunk with docstring
results = await db.search("VectorDatabase", limit=1)

# Inspect NLP fields
result = results[0]
print(f"Keywords: {result.nlp_keywords}")
print(f"Code refs: {result.nlp_code_refs}")
print(f"Technical terms: {result.nlp_technical_terms}")
```

### 5. Compare Precision@5 Metrics

#### Before NLP (Baseline)
- Average precision@5: 88%
- Weak query performance: 0.639-0.695

#### After NLP (Target)
- Average precision@5: 92%
- Weak query performance: 0.75-0.80+

#### How to Measure
```bash
# Run benchmark suite
mcp-vector-search benchmark --queries test_queries.txt
```

## Debugging Tips

### Check NLP Extraction
```python
from mcp_vector_search.core.nlp_extractor import NLPExtractor

extractor = NLPExtractor()

# Test extraction on sample docstring
docstring = """
Perform semantic search using `VectorDatabase`.

Returns:
    SearchResult objects with similarity scores
"""

entities = extractor.extract(docstring)
print(entities)
```

### Verify Parser Integration
```python
from mcp_vector_search.parsers.python import PythonParser
from pathlib import Path

parser = PythonParser()

# Parse a file
chunks = await parser.parse_file(Path("your_file.py"))

# Check NLP fields
for chunk in chunks:
    if chunk.docstring:
        print(f"Chunk: {chunk.function_name or chunk.class_name}")
        print(f"Keywords: {chunk.nlp_keywords}")
        print(f"Code refs: {chunk.nlp_code_refs}")
        print(f"Technical terms: {chunk.nlp_technical_terms}")
        print()
```

### Inspect Search Boost
```python
from mcp_vector_search.core.result_ranker import ResultRanker

ranker = ResultRanker()

# Rerank results with NLP boost
reranked = ranker.rerank_results(results, query="VectorDatabase")

# Check boosted scores
for result in reranked:
    print(f"File: {result.file_path}")
    print(f"Score: {result.similarity_score}")
    print(f"Keywords: {result.nlp_keywords}")
    print()
```

## Performance Validation

### Extraction Speed
Expected: <5ms per chunk

```python
import time
from mcp_vector_search.core.nlp_extractor import NLPExtractor

extractor = NLPExtractor()

docstring = """
Long docstring with multiple paragraphs...
"""

start = time.perf_counter()
entities = extractor.extract(docstring)
elapsed = (time.perf_counter() - start) * 1000

print(f"Extraction time: {elapsed:.2f}ms")
assert elapsed < 5.0, "Extraction too slow!"
```

### Indexing Overhead
Expected: <10% increase in indexing time

```bash
# Time indexing before NLP
time mcp-vector-search index --force

# Note: Compare with baseline indexing time
# Overhead should be <10%
```

## Test Cases

### 1. Code Reference Extraction
Test that backtick references are extracted:

```python
# Docstring with code references
docstring = "Use `VectorDatabase` to store and `search()` to query."

entities = extractor.extract(docstring)
assert "VectorDatabase" in entities.code_references
assert "search()" in entities.code_references
```

### 2. Technical Term Extraction
Test that CamelCase and ACRONYMS are extracted:

```python
docstring = "The DatabaseConnection uses HTTP protocol and API calls."

entities = extractor.extract(docstring)
assert "DatabaseConnection" in entities.technical_terms
assert "HTTP" in entities.technical_terms
assert "API" in entities.technical_terms
```

### 3. Action Verb Extraction
Test that docstring verbs are extracted:

```python
docstring = """
Returns SearchResult objects.
Raises ValueError if invalid.
Creates embeddings asynchronously.
"""

entities = extractor.extract(docstring)
assert "returns" in entities.action_verbs
assert "raises" in entities.action_verbs
assert "creates" in entities.action_verbs
```

### 4. Empty Docstring Handling
Test that empty/missing docstrings don't crash:

```python
# No docstring
chunk = parser._create_chunk(
    content="def foo(): pass",
    file_path=Path("test.py"),
    start_line=1,
    end_line=1,
    docstring=None
)

assert chunk.nlp_keywords == []
assert chunk.nlp_code_refs == []
assert chunk.nlp_technical_terms == []
```

## Success Criteria Checklist

- [ ] YAKE dependency installed
- [ ] Database reindexed with NLP extraction
- [ ] NLP fields populated in chunks
- [ ] Search results show improved scores
- [ ] Weak queries improved (0.639 → 0.75+)
- [ ] Precision@5 improved (88% → 92%)
- [ ] Extraction time <5ms per chunk
- [ ] Indexing overhead <10%
- [ ] All tests pass

## Troubleshooting

### Issue: YAKE import fails
**Solution**: Install with `uv pip install yake==0.4.8`
**Fallback**: Extractor will use simple keyword extraction

### Issue: NLP fields empty after reindex
**Cause**: Chunks may not have docstrings
**Check**: Verify docstrings exist in source files

### Issue: No boost in search results
**Cause**: Query terms may not match NLP entities
**Debug**: Print NLP fields for top results and compare with query terms

### Issue: Indexing too slow
**Cause**: YAKE extraction may be slow for large codebases
**Solution**: Profile extraction time and consider caching

## Next Steps After Testing

1. **Validate metrics**: Compare precision@5 before/after
2. **Tune boost weights**: Adjust NLP boost factors if needed
3. **Monitor performance**: Track extraction time during indexing
4. **Gather feedback**: Test with real queries from users
5. **Iterate**: Refine extraction patterns based on results

---

**Testing Priority**: High
**Estimated Time**: 30 minutes
**Expected Impact**: 4-5% precision improvement
