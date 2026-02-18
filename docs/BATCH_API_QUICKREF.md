# Knowledge Graph Batch API - Quick Reference

> **TL;DR:** Use batch methods for 50-60% faster knowledge graph builds. Replace individual `add_*` calls with `add_*_batch` methods.

---

## Quick Start

### Before (Slow)
```python
for entity in entities:
    await kg.add_entity(entity)  # 1 query per entity
```

### After (Fast)
```python
await kg.add_entities_batch(entities)  # 1 query per 500 entities
```

---

## API Reference

### Core Batch Methods

#### `add_entities_batch(entities, batch_size=500)`
Insert code entities in batches.

```python
entities = [CodeEntity(id="1", name="foo", ...), ...]
count = await kg.add_entities_batch(entities, batch_size=500)
```

**Returns:** Number of entities inserted

---

#### `add_doc_sections_batch(docs, batch_size=500)`
Insert documentation sections in batches.

```python
docs = [DocSection(id="1", name="Section", ...), ...]
count = await kg.add_doc_sections_batch(docs)
```

**Returns:** Number of doc sections inserted

---

#### `add_tags_batch(tag_names, batch_size=500)`
Insert tags in batches (auto-deduplicates).

```python
tags = ["python", "rust", "go"]
count = await kg.add_tags_batch(tags)
```

**Returns:** Number of unique tags inserted

---

#### `add_relationships_batch(relationships, batch_size=500)`
Insert relationships in batches (auto-groups by type).

```python
rels = [
    CodeRelationship(source_id="1", target_id="2", relationship_type="calls"),
    CodeRelationship(source_id="2", target_id="3", relationship_type="imports"),
]
count = await kg.add_relationships_batch(rels)
```

**Returns:** Number of relationships inserted

**Supported types:** CALLS, IMPORTS, INHERITS, CONTAINS, REFERENCES, DOCUMENTS, FOLLOWS, HAS_TAG, DEMONSTRATES, LINKS_TO

---

#### `add_part_of_batch(entity_ids, project_id, batch_size=500)`
Create PART_OF relationships in batches.

```python
entity_ids = ["entity:1", "entity:2", "entity:3"]
count = await kg.add_part_of_batch(entity_ids, "project:myproject")
```

**Returns:** Number of relationships created

---

## CLI Usage

### Standard Build (Batch Optimized)
```bash
mcp-vector-search kg build
```
**Time:** ~3-4 minutes (moderate repo)

### Fast Build (Skip DOCUMENTS)
```bash
mcp-vector-search kg build --skip-documents
```
**Time:** ~2-3 minutes (25-30% faster)

### Force Rebuild
```bash
mcp-vector-search kg build --force
```

### Test with Limit
```bash
mcp-vector-search kg build --limit 100
```

### Benchmark Performance
```bash
time mcp-vector-search kg build --force
```

---

## Common Patterns

### Pattern 1: Collect-Then-Batch
```python
# Phase 1: Collect entities (no DB writes)
entities = []
relationships = []

for chunk in chunks:
    entity = extract_entity(chunk)
    entities.append(entity)

    rels = extract_relationships(chunk)
    relationships.extend(rels)

# Phase 2: Batch insert
await kg.add_entities_batch(entities)
await kg.add_relationships_batch(relationships)
```

### Pattern 2: Mixed Types
```python
# Collect all data
code_entities = [...]
doc_sections = [...]
tags = [...]
relationships = [...]

# Batch insert in order
await kg.add_entities_batch(code_entities)
await kg.add_doc_sections_batch(doc_sections)
await kg.add_tags_batch(tags)
await kg.add_relationships_batch(relationships)
```

### Pattern 3: Large Datasets
```python
# Process in chunks of 1000 entities at a time
for i in range(0, len(entities), 1000):
    batch = entities[i:i+1000]
    await kg.add_entities_batch(batch, batch_size=500)
```

---

## Performance Tips

### ✅ Do

- **Collect first, insert later:** Minimize DB round trips
- **Use batch methods for >10 entities:** Significant speedup
- **Group by type:** Relationships auto-grouped, but pre-grouping helps
- **Adjust batch_size:** Default 500 is optimal for most cases

### ❌ Don't

- **Mix batch and individual calls:** Stick to one approach
- **Make batch_size too small:** <100 reduces benefits
- **Make batch_size too large:** >1000 may hit query limits
- **Skip error handling:** Always check return counts

---

## Error Handling

All batch methods have automatic fallback:

```python
try:
    # Batch insert (fast)
    count = await kg.add_entities_batch(entities)
except Exception as e:
    # Automatically falls back to individual inserts
    # Logs error and continues
    pass
```

**No action required:** Batch failures automatically degrade gracefully.

---

## Troubleshooting

### Issue: Batch insert failed
**Symptom:** Error message "Batch insert failed: ..."

**Solution:** Check logs - method automatically falls back to individual inserts. Partial data is still inserted.

### Issue: Slow build despite batching
**Symptom:** Build still takes >5 minutes

**Possible causes:**
1. Large DOCUMENTS extraction (use `--skip-documents`)
2. Too many relationships (>10k per type)
3. Disk I/O bottleneck (check disk speed)

**Solution:**
```bash
# Skip expensive operations
mcp-vector-search kg build --skip-documents

# Limit dataset for testing
mcp-vector-search kg build --limit 500
```

### Issue: Memory usage high
**Symptom:** High memory during build

**Cause:** Collecting all entities before insertion

**Solution:**
- Use streaming approach (future feature)
- Process in smaller chunks:
```python
for chunk_batch in chunks_batched(all_chunks, size=1000):
    entities = [extract_entity(c) for c in chunk_batch]
    await kg.add_entities_batch(entities)
```

---

## Migration Guide

### Step 1: Identify Loops
Find patterns like:
```python
for entity in entities:
    await kg.add_entity(entity)
```

### Step 2: Collect Entities
Change to:
```python
# Collect first
entities = [extract_entity(chunk) for chunk in chunks]

# Batch insert
await kg.add_entities_batch(entities)
```

### Step 3: Test
```bash
mcp-vector-search kg build --force --limit 100
```

### Step 4: Benchmark
```bash
time mcp-vector-search kg build --force
```

---

## Performance Expectations

| Repo Size | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Small (100 chunks) | 30 sec | 15 sec | 50% |
| Medium (500 chunks) | 8.5 min | 3.5 min | 59% |
| Large (2000 chunks) | 35 min | 15 min | 57% |
| XL (10k chunks) | 3 hrs | 1.5 hrs | 50% |

*With `--skip-documents`: Add 25-30% additional speedup*

---

## Advanced Usage

### Custom Batch Sizes
```python
# Small batches for testing
await kg.add_entities_batch(entities, batch_size=100)

# Large batches for production
await kg.add_entities_batch(entities, batch_size=1000)
```

### Relationship Type Filtering
```python
# Only insert specific relationship types
calls_rels = [r for r in relationships if r.relationship_type == "calls"]
await kg.add_relationships_batch(calls_rels)
```

### Progress Tracking
```python
from rich.progress import Progress

with Progress() as progress:
    task = progress.add_task("Inserting entities", total=len(entities))

    for i in range(0, len(entities), 500):
        batch = entities[i:i+500]
        await kg.add_entities_batch(batch)
        progress.update(task, advance=len(batch))
```

---

## References

- **Full Documentation:** `docs/kg_batch_optimization.md`
- **Implementation Summary:** `BATCH_OPTIMIZATION_SUMMARY.md`
- **Tests:** `tests/unit/core/test_kg_batch.py`

---

## Version Info

- **Added in:** v2.2.22
- **Status:** Stable
- **Backward Compatible:** Yes

---

## Quick Comparison

| Feature | Individual Insert | Batch Insert |
|---------|------------------|--------------|
| Speed | ⭐ Slow | ⭐⭐⭐⭐⭐ Fast |
| Memory | ⭐⭐⭐⭐⭐ Low | ⭐⭐⭐ Moderate |
| Complexity | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐⭐ Easy |
| Error Handling | ⭐⭐⭐ Per-item | ⭐⭐⭐⭐ Auto-fallback |
| Recommended | Small datasets (<10) | Large datasets (10+) |

---

**Need help?** See full documentation in `docs/kg_batch_optimization.md`
