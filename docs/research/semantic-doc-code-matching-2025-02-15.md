# Research: Semantic Matching Between Documentation and Code Entities

**Date**: 2025-02-15
**Status**: Proposal
**Goal**: Create DOCUMENTS relationships in knowledge graph to connect doc sections with semantically related code entities

## Executive Summary

The knowledge graph currently has 4,067 code entities and 10,483 doc sections with only 146 REFERENCES edges (backtick mentions). To create meaningful DOCUMENTS relationships, we can leverage existing embeddings infrastructure OR implement lightweight keyword-based matching. Current state analysis reveals that embeddings exist only in ChromaDB (not in Lance), while chunks.lance has 15,028 pending chunks awaiting embedding.

**Recommendation**: Implement a **hybrid approach** combining keyword matching (fast, high precision) with optional embedding similarity (slower, higher recall) to balance performance and quality.

---

## Current State Analysis

### Data Architecture

```
.mcp-vector-search/
├── lance/
│   └── chunks.lance          # 15,028 chunks, NO embeddings (all pending)
├── code_search.lance/        # Empty (no tables)
├── knowledge_graph/
│   └── code_kg               # Kuzu database
│       ├── CodeEntity: 4,067 nodes
│       ├── DocSection: 10,483 nodes
│       ├── REFERENCES: 146 edges (backtick mentions)
│       └── DOCUMENTS: 0 edges (TARGET)
└── (ChromaDB persistence elsewhere - contains embeddings)
```

### Key Findings

1. **Embeddings NOT in Lance**: The `chunks.lance` table does NOT contain embeddings (no `vector` column)
   - All 15,028 chunks have `embedding_status='pending'`
   - Embeddings are stored in ChromaDB's persistent storage (separate from Lance)
   - Lance is used for Phase 1 (parsing/chunking) tracking only

2. **Two-Phase Indexing Architecture**:
   - **Phase 1** (chunks.lance): Parse files → extract chunks → track with file_hash
   - **Phase 2** (ChromaDB): Generate embeddings → store in ChromaDB for search
   - **Phase 3** (KG): Extract entities/relationships → store in Kuzu

3. **Available Data for Matching**:
   - **In chunks.lance**: content, docstring, name, file_path, language, chunk_type
   - **In KG CodeEntity**: id, name, entity_type, file_path
   - **In KG DocSection**: id, name, file_path, level, line_start, line_end
   - **Embeddings**: Available in ChromaDB (not easily accessible from Lance/KG)

4. **Current REFERENCES Logic** (kg_builder.py:387-403):
   - Extracts backtick code references from doc content
   - Uses regex: `` `([^`\n]+)` `` (inline code, not blocks)
   - Resolves entity names via `_entity_map` lookup
   - Creates REFERENCES edge only if entity found

---

## Problem Definition

### What is DOCUMENTS?

A **DOCUMENTS** relationship indicates that a documentation section **semantically explains or describes** a code entity, beyond just mentioning it.

**Examples**:

| Doc Section | Code Entity | DOCUMENTS? | REFERENCES? |
|------------|-------------|------------|-------------|
| "## Authentication Flow" explains OAuth2Handler class | OAuth2Handler | ✓ | ✓ |
| "Use `login()` to authenticate" | login() | ✗ | ✓ |
| README section on database module | database.py module | ✓ | Maybe |
| Tutorial on vector search | VectorDatabase class | ✓ | Maybe |

**Key Distinction**:
- **REFERENCES**: Mentions entity name (usually in backticks)
- **DOCUMENTS**: Provides semantic explanation/tutorial/description

---

## Proposed Approaches

### Approach 1: Keyword-Based Matching (RECOMMENDED)

**Strategy**: Match entity names in doc section titles and content using fuzzy matching and proximity scoring.

#### Algorithm

```python
def compute_documents_score(doc_section: DocSection, entity: CodeEntity) -> float:
    """
    Compute semantic relevance score (0.0-1.0) between doc and code entity.

    Returns score based on:
    - Entity name in doc title (0.4)
    - Entity name in doc content (0.2)
    - File proximity (0.2)
    - Contextual keywords (0.2)
    """
    score = 0.0

    # 1. Entity name in doc section title (strong signal)
    if entity.name.lower() in doc_section.name.lower():
        score += 0.4

    # 2. Entity name in doc content (moderate signal)
    # Already captured by REFERENCES, but check frequency
    doc_content = get_section_content(doc_section)
    mentions = doc_content.lower().count(entity.name.lower())
    if mentions >= 2:  # Multiple mentions = explaining
        score += 0.2
    elif mentions == 1:
        score += 0.1

    # 3. File proximity (same directory or nearby)
    if Path(doc_section.file_path).parent == Path(entity.file_path).parent:
        score += 0.2
    elif is_readme_for_module(doc_section.file_path, entity.file_path):
        score += 0.3  # README.md documenting module

    # 4. Contextual keywords (function, class, API reference, usage)
    doc_lower = doc_content.lower()
    if entity.entity_type == "function":
        if any(kw in doc_lower for kw in ["function", "method", "call", "usage"]):
            score += 0.1
    elif entity.entity_type == "class":
        if any(kw in doc_lower for kw in ["class", "object", "instance", "inherit"]):
            score += 0.1

    return min(score, 1.0)  # Cap at 1.0

def extract_documents_relationships():
    """
    Extract DOCUMENTS relationships using keyword matching.
    """
    threshold = 0.5  # Minimum score to create edge

    for doc_section in get_all_doc_sections():
        for entity in get_nearby_entities(doc_section):  # Filter by file proximity
            score = compute_documents_score(doc_section, entity)

            if score >= threshold:
                create_edge(
                    source=doc_section.id,
                    target=entity.id,
                    type="DOCUMENTS",
                    weight=score
                )
```

#### Pros & Cons

**Pros**:
- ✅ Fast (no embedding computation needed)
- ✅ High precision (keyword matching is explicit)
- ✅ Works with current data (no new infrastructure)
- ✅ Explainable results (score breakdown available)
- ✅ Can run immediately (no waiting for embeddings)

**Cons**:
- ❌ Lower recall (misses semantic similarity without keywords)
- ❌ Name-dependent (requires entity name in doc)
- ❌ No understanding of paraphrasing (e.g., "authentication handler" ≠ "OAuth2Handler")

**Expected Coverage**: 30-50% of true DOCUMENTS relationships (high precision, moderate recall)

---

### Approach 2: Embedding Similarity (OPTIONAL ENHANCEMENT)

**Strategy**: Compare doc section embeddings with code entity embeddings using cosine similarity.

#### Requirements

1. **Generate Doc Section Embeddings**:
   - Extract content for each DocSection (already available in chunks.lance)
   - Generate embeddings using existing SentenceTransformer model
   - Store in new Lance table `doc_embeddings.lance`

2. **Access Code Entity Embeddings**:
   - Query ChromaDB for code chunk embeddings by chunk_id
   - OR: Export embeddings from ChromaDB → Lance for faster access

3. **Compute Similarity**:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity

   def compute_embedding_similarity(doc_section, entity):
       doc_embedding = get_doc_embedding(doc_section.id)
       code_embedding = get_code_embedding(entity.id)
       return cosine_similarity([doc_embedding], [code_embedding])[0][0]
   ```

#### Algorithm

```python
async def extract_documents_via_embeddings(threshold=0.75):
    """
    Extract DOCUMENTS relationships using embedding similarity.

    Args:
        threshold: Minimum cosine similarity (0.75 = strong semantic match)
    """
    for doc_section in get_all_doc_sections():
        doc_embedding = await get_or_compute_doc_embedding(doc_section)

        # Find nearest neighbors in embedding space
        candidates = find_nearby_entities_by_file(doc_section.file_path)

        for entity in candidates:
            code_embedding = await get_code_embedding(entity.id)
            similarity = cosine_similarity([doc_embedding], [code_embedding])[0][0]

            if similarity >= threshold:
                await kg.add_relationship(
                    source_id=doc_section.id,
                    target_id=entity.id,
                    relationship_type="documents",
                    weight=similarity
                )
```

#### Pros & Cons

**Pros**:
- ✅ High recall (captures semantic similarity)
- ✅ Language-agnostic (works across paraphrasing)
- ✅ Leverages existing embedding infrastructure

**Cons**:
- ❌ Slow (requires embedding generation for all docs)
- ❌ Requires ChromaDB access (not directly in Lance)
- ❌ Higher complexity (embedding pipeline integration)
- ❌ Less explainable (cosine similarity is opaque)
- ❌ May require reindexing (15,028 pending chunks)

**Expected Coverage**: 70-85% of true DOCUMENTS relationships (high recall, moderate precision)

---

### Approach 3: Hybrid Approach (BEST BALANCE)

**Strategy**: Combine keyword matching (Approach 1) with optional embedding similarity (Approach 2).

#### Algorithm

```python
async def extract_documents_hybrid(
    keyword_threshold=0.5,
    embedding_threshold=0.75,
    use_embeddings=False  # Can be enabled later
):
    """
    Hybrid DOCUMENTS extraction with fallback logic.

    Phase 1 (Fast): Keyword matching for high-confidence pairs
    Phase 2 (Slow): Embedding similarity for remaining pairs (optional)
    """
    # Phase 1: Keyword-based matching (always runs)
    for doc_section in get_all_doc_sections():
        for entity in get_nearby_entities(doc_section):
            keyword_score = compute_documents_score(doc_section, entity)

            if keyword_score >= keyword_threshold:
                await kg.add_relationship(
                    source_id=doc_section.id,
                    target_id=entity.id,
                    relationship_type="documents",
                    weight=keyword_score
                )
                continue  # Skip embedding check (already confident)

    # Phase 2: Embedding similarity (optional, for low keyword scores)
    if use_embeddings:
        for doc_section in get_all_doc_sections():
            for entity in get_nearby_entities(doc_section):
                # Skip if already has DOCUMENTS edge
                if has_documents_edge(doc_section.id, entity.id):
                    continue

                # Compute embedding similarity
                similarity = await compute_embedding_similarity(doc_section, entity)

                if similarity >= embedding_threshold:
                    await kg.add_relationship(
                        source_id=doc_section.id,
                        target_id=entity.id,
                        relationship_type="documents",
                        weight=similarity
                    )
```

#### Pros & Cons

**Pros**:
- ✅ Best of both worlds (precision + recall)
- ✅ Fast initial pass (keyword matching)
- ✅ Optional enhancement (embedding similarity)
- ✅ Progressive improvement (start simple, enhance later)

**Cons**:
- ❌ More complex implementation (two algorithms)
- ❌ Requires tuning two thresholds

**Expected Coverage**: 40-60% (keyword only) → 75-90% (with embeddings)

---

## Implementation Roadmap

### Phase 1: Keyword-Based Matching (Week 1)

**Goal**: Implement Approach 1 to create initial DOCUMENTS relationships.

**Tasks**:
1. Extract doc section content from chunks.lance by file_path + line range
2. Implement `compute_documents_score()` function
3. Add `extract_documents_relationships()` to kg_builder.py
4. Run initial pass to create DOCUMENTS edges
5. Validate quality with manual spot-checks

**Deliverables**:
- ~1,500-3,000 DOCUMENTS edges (estimated 30-50% coverage)
- Fast execution (<5 minutes for 10K doc sections)

**Success Criteria**:
- Precision ≥80% (manual validation of 50 random samples)
- Zero false positives for unrelated entities

---

### Phase 2: Embedding Similarity (Optional, Week 2-3)

**Goal**: Enhance DOCUMENTS extraction with embedding-based semantic matching.

**Prerequisites**:
1. Complete Phase 2 indexing (generate embeddings for all 15,028 chunks)
2. Extract embeddings from ChromaDB → Lance for faster access

**Tasks**:
1. Generate embeddings for doc sections (using existing model)
2. Store doc embeddings in `doc_embeddings.lance` table
3. Implement `compute_embedding_similarity()` function
4. Run embedding pass for remaining doc-code pairs
5. Validate quality improvements

**Deliverables**:
- ~3,000-5,000 additional DOCUMENTS edges (70-85% total coverage)
- Slower execution (~30-60 minutes for 10K doc sections)

**Success Criteria**:
- Recall improvement ≥30% over keyword-only approach
- Precision maintained ≥70% (acceptable trade-off)

---

## Data Requirements

### Currently Available

| Data | Location | Fields | Count |
|------|----------|--------|-------|
| Code chunks | chunks.lance | content, docstring, name, file_path | 15,028 |
| Code entities | KG (CodeEntity) | id, name, entity_type, file_path | 4,067 |
| Doc sections | KG (DocSection) | id, name, file_path, level, line_start | 10,483 |
| Code embeddings | ChromaDB | 384-dim vectors | ~4,000 (estimated) |

### Needs to be Computed

| Data | Approach 1 | Approach 2 | Approach 3 |
|------|-----------|-----------|-----------|
| Doc section content extraction | ✓ Required | ✓ Required | ✓ Required |
| Doc section embeddings | ✗ Not needed | ✓ Required | ⚠️ Optional |
| Embedding similarity matrix | ✗ Not needed | ✓ Required | ⚠️ Optional |

---

## Performance Estimates

### Approach 1: Keyword-Based

- **Preprocessing**: Extract doc content from chunks.lance (~2 min)
- **Matching**: Score 10,483 docs × ~50 nearby entities = 524K pairs (~5 min)
- **Total**: ~10 minutes for full pass
- **Expected edges**: 1,500-3,000 DOCUMENTS relationships

### Approach 2: Embedding Similarity

- **Preprocessing**: Generate 10,483 doc embeddings (~20 min on M4 Max)
- **Matching**: Cosine similarity for 524K pairs (~30 min)
- **Total**: ~60 minutes for full pass
- **Expected edges**: 4,000-6,000 DOCUMENTS relationships

### Approach 3: Hybrid

- **Phase 1 (Keyword)**: ~10 minutes
- **Phase 2 (Embeddings)**: ~30 minutes (only for remaining pairs)
- **Total**: ~40 minutes for full pass
- **Expected edges**: 5,000-7,000 DOCUMENTS relationships

---

## Quality Metrics

### Precision (Manual Validation)

Sample 50 random DOCUMENTS edges and check if doc truly documents entity:

- **Target precision**: ≥80% (Approach 1), ≥70% (Approach 2)

### Recall (Coverage Estimate)

Estimate how many "true" DOCUMENTS relationships exist:

- **Approach 1**: 30-50% recall (keyword-dependent)
- **Approach 2**: 70-85% recall (semantic similarity)
- **Approach 3**: 75-90% recall (best of both)

### Edge Distribution

Expected DOCUMENTS edge distribution:

| Scenario | Edges | Precision | Recall |
|----------|-------|-----------|--------|
| Keyword-only | 1,500-3,000 | 80-85% | 30-50% |
| Embedding-only | 4,000-6,000 | 70-75% | 70-85% |
| Hybrid | 5,000-7,000 | 75-80% | 75-90% |

---

## Recommendations

### Immediate Action (This Week)

**Implement Approach 1 (Keyword-Based Matching)**:

1. Add `compute_documents_score()` to kg_builder.py
2. Integrate into `build_from_chunks()` workflow
3. Run initial pass on existing 10,483 doc sections
4. Validate quality with 50 random samples

**Benefits**:
- ✅ Fast implementation (~2-3 hours)
- ✅ Immediate value (1,500-3,000 DOCUMENTS edges)
- ✅ No new infrastructure needed
- ✅ High precision (≥80% expected)

### Future Enhancement (Next Sprint)

**Add Approach 2 (Embedding Similarity)** as optional flag:

1. Complete Phase 2 indexing (generate embeddings)
2. Export embeddings from ChromaDB → Lance
3. Add `--use-embeddings` flag to kg build command
4. Run embedding pass for improved recall

**Benefits**:
- ✅ Higher recall (+30-40% coverage)
- ✅ Captures semantic relationships
- ✅ Progressive enhancement (optional feature)

### Long-Term Strategy

**Hybrid Approach (Approach 3)**:

- Use keyword matching as **default** (fast, high precision)
- Enable embedding similarity via **config flag** (slow, high recall)
- Monitor precision/recall trade-offs with analytics
- Tune thresholds based on user feedback

---

## Example Queries Enabled

Once DOCUMENTS relationships exist, the KG enables powerful queries:

```cypher
-- Find all documentation for a specific code entity
MATCH (d:DocSection)-[r:DOCUMENTS]->(e:CodeEntity {name: "VectorDatabase"})
RETURN d.name, d.file_path, r.weight
ORDER BY r.weight DESC

-- Find code entities without documentation
MATCH (e:CodeEntity)
WHERE NOT EXISTS {
  MATCH (d:DocSection)-[:DOCUMENTS]->(e)
}
RETURN e.name, e.entity_type, e.file_path

-- Find most-documented entities (popular/important code)
MATCH (d:DocSection)-[:DOCUMENTS]->(e:CodeEntity)
RETURN e.name, e.entity_type, count(d) as doc_count
ORDER BY doc_count DESC
LIMIT 10

-- Find documentation quality score (DOCUMENTS vs REFERENCES)
MATCH (e:CodeEntity)
OPTIONAL MATCH (d1:DocSection)-[:DOCUMENTS]->(e)
OPTIONAL MATCH (d2:DocSection)-[:REFERENCES]->(e)
RETURN e.name,
       count(DISTINCT d1) as documented_by,
       count(DISTINCT d2) as referenced_by,
       count(DISTINCT d1) * 1.0 / count(DISTINCT d2) as doc_quality_ratio
ORDER BY doc_quality_ratio DESC
```

---

## Conclusion

**Recommended Path Forward**:

1. **Start with Approach 1** (Keyword-Based Matching):
   - Implement this week
   - Fast, high-precision results
   - No new infrastructure needed

2. **Enhance with Approach 2** (Embedding Similarity) later:
   - Add as optional feature
   - Requires embedding infrastructure
   - Improves recall by 30-40%

3. **Monitor and tune**:
   - Collect precision/recall metrics
   - Adjust thresholds based on user feedback
   - Iterate on scoring algorithm

**Expected Outcome**:
- Initial implementation: 1,500-3,000 DOCUMENTS edges (30-50% coverage, ≥80% precision)
- Enhanced implementation: 5,000-7,000 DOCUMENTS edges (75-90% coverage, ≥75% precision)
- Timeline: Week 1 (keyword), Week 2-3 (embeddings)

---

## Appendix A: Code Snippets

### Extract Doc Section Content

```python
def get_section_content(doc_section: DocSection) -> str:
    """
    Extract content for a doc section from chunks.lance.

    Uses file_path and line_start/line_end to find matching chunks.
    """
    # Query chunks.lance for matching file and line range
    chunks = query_chunks_by_file_and_lines(
        file_path=doc_section.file_path,
        start_line=doc_section.line_start,
        end_line=doc_section.line_end
    )

    # Concatenate chunk contents
    return "\n".join(chunk.content for chunk in chunks)
```

### File Proximity Scoring

```python
def is_readme_for_module(readme_path: str, module_path: str) -> bool:
    """
    Check if README.md documents a specific module.

    Examples:
      - README.md in src/core/ documents src/core/database.py
      - docs/api.md documents src/api/
    """
    readme_dir = Path(readme_path).parent
    module_dir = Path(module_path).parent

    return readme_dir == module_dir or readme_dir == module_dir.parent
```

### Scoring Breakdown Example

```python
# Doc: "## VectorDatabase Class" (README.md)
# Entity: VectorDatabase (class in database.py)

score_breakdown = {
    "name_in_title": 0.4,      # "VectorDatabase" in "VectorDatabase Class"
    "name_in_content": 0.2,    # Multiple mentions in section
    "file_proximity": 0.3,     # README.md in same directory
    "context_keywords": 0.1,   # "class" keyword found
    "total": 1.0
}

# Result: High-confidence DOCUMENTS edge with weight=1.0
```

---

## Appendix B: Alternative Approaches Considered

### Path-Based Heuristics (Rejected)

**Idea**: Use file path patterns to infer documentation relationships:
- `docs/api/database.md` documents `src/database.py`
- `README.md` documents all files in same directory

**Why Rejected**: Too coarse-grained, high false positive rate

### NLP Co-Reference Resolution (Future Work)

**Idea**: Use NLP models (e.g., spaCy) to detect when docs refer to entities:
- "The database class" → VectorDatabase
- "This function" → search()

**Why Deferred**: High complexity, requires NLP pipeline integration

### LLM-Based Classification (Future Work)

**Idea**: Use LLM (Claude/GPT-4) to classify doc-entity pairs:
- Prompt: "Does this documentation section explain this code entity?"
- Input: (doc_content, entity_signature)
- Output: Yes/No + confidence

**Why Deferred**: Expensive, slow, requires API integration

---

**Research completed by**: Claude Research Agent
**Next steps**: Implement Approach 1 (Keyword-Based Matching) in kg_builder.py
