# GraphCodeBERT Default Model Changes

## Summary

Switched the default embedding model from device-dependent (MiniLM on CPU/MPS, GraphCodeBERT on CUDA) to **GraphCodeBERT for all devices**. Added a `--preset` CLI option to allow fast indexing with MiniLM when needed.

## Rationale

- **GraphCodeBERT provides superior code understanding** (data-flow aware)
- Quality over speed by default aligns with semantic search use case
- Users can opt-in to fast mode via `--preset fast` when speed is preferred
- Previous default (MiniLM on CPU) was optimized for speed but sacrificed quality

## Changes

### 1. `src/mcp_vector_search/core/embeddings.py`

**Changed `_default_model_for_device()`:**
```python
def _default_model_for_device() -> str:
    """Return the default embedding model.

    GraphCodeBERT provides superior code understanding (data-flow aware).
    Use --preset fast for MiniLM if indexing speed is preferred.
    """
    return "microsoft/graphcodebert-base"
```

**Added `get_model_dimension()`:**
```python
def get_model_dimension(model_name: str | None = None) -> int:
    """Return the embedding dimension for the given model.

    Args:
        model_name: Model name (None = use default)

    Returns:
        Embedding dimension (384 for MiniLM, 768 for code models)
    """
    if model_name is None:
        model_name = _default_model_for_device()
    if "minilm" in model_name.lower():
        return 384
    # GraphCodeBERT, CodeBERT, and most code models are 768d
    return 768
```

### 2. `src/mcp_vector_search/core/vectors_backend.py`

**Changed `DEFAULT_VECTOR_DIMENSION`:**
```python
DEFAULT_VECTOR_DIMENSION = 768  # GraphCodeBERT (default model)
```

### 3. `src/mcp_vector_search/core/lancedb_backend.py`

**Changed `LANCEDB_SCHEMA`:**
```python
# Default schema for 768-dimensional embeddings (GraphCodeBERT default)
LANCEDB_SCHEMA = _create_lance_schema(768)
```

### 4. `src/mcp_vector_search/cli/commands/index.py`

**Added `--preset` option:**
```python
preset: str = typer.Option(
    "",
    "--preset",
    help="Model preset: 'fast' (MiniLM, 384d, ~10x faster) or 'quality' (GraphCodeBERT, 768d, better code understanding). Default: quality.",
    rich_help_panel="⚡ Performance",
),
```

**Added preset handling in `run_indexing()`:**
```python
# Handle preset
if preset == "fast":
    # Override to MiniLM for fast indexing
    print_info("Preset: fast (MiniLM-L6-v2, 384d)")
    config_model = "sentence-transformers/all-MiniLM-L6-v2"
elif preset == "quality" or preset == "":
    # Default: GraphCodeBERT for best code understanding
    config_model = config.embedding_model  # None = auto-select (GraphCodeBERT)
else:
    print_warning(f"Unknown preset '{preset}', using default (quality)")
    config_model = config.embedding_model

# Override embedding model if specified (takes precedence over preset)
if embedding_model_override:
    logger.info(f"Overriding embedding model: {embedding_model_override}")
    config_model = embedding_model_override

# Apply the model selection
if config_model != config.embedding_model:
    config = config.model_copy(update={"embedding_model": config_model})
```

**Updated model display:**
```python
# Display embedding model with preset info
if preset == "fast":
    print_info("Embedding model: sentence-transformers/all-MiniLM-L6-v2 (fast preset)")
elif config.embedding_model:
    print_info(f"Embedding model: {config.embedding_model}")
else:
    print_info("Embedding model: auto (GraphCodeBERT - code-optimized)")
```

## Usage

### Default behavior (GraphCodeBERT, 768d)
```bash
mcp-vector-search index -f
# Shows: Embedding model: auto (GraphCodeBERT - code-optimized)
```

### Fast preset (MiniLM, 384d, ~10x faster)
```bash
mcp-vector-search index -f --preset fast
# Shows: Preset: fast (MiniLM-L6-v2, 384d)
# Shows: Embedding model: sentence-transformers/all-MiniLM-L6-v2 (fast preset)
```

### Quality preset (explicit, same as default)
```bash
mcp-vector-search index -f --preset quality
# Shows: Embedding model: auto (GraphCodeBERT - code-optimized)
```

### Override with specific model (takes precedence)
```bash
mcp-vector-search index -f --embedding-model microsoft/codebert-base
# Shows: Embedding model: microsoft/codebert-base
```

## Priority Order

1. **Environment variable** (`MCP_VECTOR_SEARCH_EMBEDDING_MODEL`) - highest priority
2. **`--embedding-model` flag** - explicit override
3. **`--preset` flag** - fast/quality presets
4. **Config file** (`embedding_model` field)
5. **Default** (GraphCodeBERT) - if none of the above specified

## Migration Impact

- **Existing indexes with 384d** will need force rebuild (`-f` flag) to switch to 768d
- **Force rebuild behavior**: old tables are dropped and recreated with new dimensions
- **No data loss**: chunks are re-parsed and re-embedded with new model
- **Users can continue using MiniLM** via `--preset fast` or `--embedding-model`

## Testing

All tests pass:
- ✓ Default uses GraphCodeBERT (768d)
- ✓ `--preset fast` uses MiniLM (384d)
- ✓ `--preset quality` uses GraphCodeBERT (768d)
- ✓ Dimension detection works correctly (384 for MiniLM, 768 for others)

## References

- Research doc: `docs/research/embedding-model-evaluation-codebert-vs-minilm-2026-02-19.md`
- GraphCodeBERT paper: "GraphCodeBERT: Pre-training Code Representations with Data Flow"
- Model: `microsoft/graphcodebert-base` (768 dimensions)
