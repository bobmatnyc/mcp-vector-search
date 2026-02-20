# index-code Architecture Research

**Date**: 2026-02-20
**Research Type**: Implementation-ready architectural analysis
**Purpose**: Detailed specifications for implementing `index-code` CLI command

## Executive Summary

This document provides complete implementation details for the `index-code` command that will create a separate code-specific vector index using CodeT5+ embeddings, enabling dual-index architecture for enhanced semantic code search.

---

## 1. VectorsBackend Architecture (`vectors_backend.py`)

### 1.1 Table Schema

**Location**: Lines 29-57
**Function**: `_create_vectors_schema(vector_dim: int) -> pa.Schema`

```python
pa.schema([
    # Identity
    pa.field("chunk_id", pa.string()),

    # Vector embedding (dynamic dimension)
    pa.field("vector", pa.list_(pa.float32(), vector_dim)),

    # Denormalized search fields
    pa.field("file_path", pa.string()),
    pa.field("content", pa.string()),
    pa.field("language", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("chunk_type", pa.string()),
    pa.field("name", pa.string()),
    pa.field("hierarchy_path", pa.string()),

    # Metadata
    pa.field("embedded_at", pa.string()),  # ISO timestamp
    pa.field("model_version", pa.string()),  # Embedding model
])
```

**Key Insights**:
- **Dynamic dimensions**: Schema function takes `vector_dim` parameter
- **Fixed-size vectors**: Uses `pa.list_(pa.float32(), vector_dim)` not variable-length
- **Denormalized design**: All search fields included (no JOINs needed)
- **Default dimension**: 768D (GraphCodeBERT) at line 62, 97

### 1.2 Constructor

**Location**: Lines 99-110
**Signature**: `__init__(db_path: Path, vector_dim: int | None = None)`

```python
def __init__(self, db_path: Path, vector_dim: int | None = None):
    self.db_path = Path(db_path)  # LanceDB directory
    self._db = None                # LanceDB connection
    self._table = None             # LanceDB table
    self.vector_dim = vector_dim   # Auto-detected or explicit
```

**Key Insights**:
- `db_path` is the LanceDB directory (e.g., `.mcp-vector-search/lance`)
- `vector_dim` is optional - auto-detected from first batch if None
- Table is lazily initialized (created on first `add_vectors()`)

### 1.3 Table Name Configuration

**Location**: Line 96
**Constant**: `TABLE_NAME = "vectors"`

**For index-code implementation**:
- Use `table_name="code_vectors"` to create separate table
- Creates `code_vectors.lance/` directory alongside `vectors.lance/`
- Requires modifying VectorsBackend to support custom table names

**Proposed modification**:
```python
def __init__(self, db_path: Path, vector_dim: int | None = None, table_name: str = "vectors"):
    self.db_path = Path(db_path)
    self._db = None
    self._table = None
    self.vector_dim = vector_dim
    self.table_name = table_name  # NEW: Support custom table names
```

### 1.4 add_vectors() Method

**Location**: Lines 352-546
**Signature**: `async def add_vectors(chunks_with_vectors: list[dict], model_version: str = "all-MiniLM-L6-v2") -> int`

**Required fields per chunk**:
```python
{
    "chunk_id": str,           # Unique identifier
    "vector": list[float],     # Embedding (dimension must match)
    "file_path": str,
    "content": str,
    "language": str,
    "start_line": int,
    "end_line": int,
    "chunk_type": str,         # function, class, method, etc.
    "name": str,               # Function/class name
    "hierarchy_path": str,     # Optional, defaults to ""
}
```

**Key behaviors**:
- **Auto-dimension detection**: Lines 401-407 - detects dimension from first vector
- **Validation**: Lines 410-414 - ensures all vectors match detected dimension
- **Deduplication**: Lines 450, 478-479, 532-533 - calls `_delete_chunk_ids()` before append
- **Schema creation**: Lines 440-448 - creates PyArrow table with explicit schema
- **Table creation**: Lines 454-505 - creates table on first write, appends thereafter
- **Dimension mismatch handling**: Lines 465-497, 507-530 - drops and recreates if mismatch

**Critical for index-code**:
- Must pass correct `model_version` parameter (e.g., "Salesforce/codet5p-110m-embedding")
- All chunks must have same vector dimension
- Deduplication is automatic (no manual duplicate handling needed)

### 1.5 search() Method

**Location**: Lines 548-674
**Signature**: `async def search(query_vector: list[float], limit: int = 10, filters: dict[str, Any] | None = None) -> list[dict]`

**Search implementation**:
```python
# Line 605: Uses cosine metric
search = self._table.search(query_vector).metric("cosine").limit(limit)

# Lines 608-627: Metadata filters
if filters:
    # Supports: language, file_path, chunk_type filters
    # String filters: key = 'value'
    # List filters: key IN ('val1', 'val2')
    search = search.where(where_clause)

# Lines 630-656: Result conversion
results = search.to_list()
for result in results:
    distance = result.get("_distance", 0.0)
    similarity = max(0.0, 1.0 - (distance / 2.0))  # Cosine distance to similarity
```

**Key insights**:
- **Cosine distance metric**: Returns `_distance` in [0, 2] range
- **Similarity conversion**: `similarity = 1.0 - (distance / 2.0)`
- **Filtering support**: Metadata filters via SQL WHERE clause
- **Error handling**: Lines 664-674 - corruption auto-recovery

---

## 2. Embeddings Architecture (`embeddings.py`)

### 2.1 CodeBERTEmbeddingFunction Class

**Location**: Lines 373-553
**Constructor signature**: `__init__(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", timeout: float = 300.0)`

**Key implementation details**:
```python
# Lines 391-394: Environment variable override (HIGHEST PRIORITY)
env_model = os.environ.get("MCP_VECTOR_SEARCH_EMBEDDING_MODEL")
if env_model:
    model_name = env_model

# Lines 397-398: Device detection
device = _detect_device()  # Returns "mps", "cuda", or "cpu"

# Lines 419-422: Model loading (with stdout suppression)
with suppress_stdout_stderr():
    self.model = SentenceTransformer(
        model_name, device=device, trust_remote_code=True
    )

# Line 428: Get actual dimensions from loaded model
actual_dims = self.model.get_sentence_embedding_dimension()
```

**Properties exposed**:
- `self.model_name` (line 423): Original model identifier string
- `self.device` (line 425): Compute device ("mps", "cuda", "cpu")
- `self.timeout` (line 424): Embedding timeout in seconds
- `self.model`: SentenceTransformer instance

**Key methods**:
- `__call__(input: list[str]) -> list[list[float]]` (lines 483-512): Generate embeddings
- `embed_query(text: str) -> list[float]` (lines 543-552): Single query embedding

### 2.2 Model Loading

**Technology**: SentenceTransformer (line 91 import)
**trust_remote_code**: Required for CodeT5+ (line 421)

**Device detection** (lines 97-149):
```python
def _detect_device() -> str:
    # Priority: env var > CUDA > MPS > CPU
    # Apple Silicon defaults to CPU (faster for transformers)
    # CUDA: Check torch.cuda.is_available()
    # Returns: "mps", "cuda", or "cpu"
```

### 2.3 Dimension Detection

**Function**: `get_model_dimensions(model_name: str) -> int`
**Location**: `config/defaults.py` lines 506-533

```python
MODEL_SPECIFICATIONS = {
    "Salesforce/SFR-Embedding-Code-400M_R": {
        "dimensions": 1024,
        "type": "code",
    },
    "microsoft/graphcodebert-base": {
        "dimensions": 768,
        "type": "code",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "type": "general",
    },
}

def get_model_dimensions(model_name: str) -> int:
    if model_name in MODEL_SPECIFICATIONS:
        return MODEL_SPECIFICATIONS[model_name]["dimensions"]
    # Fallback: pattern matching
    raise ValueError(f"Unknown model: {model_name}")
```

**For CodeT5+ (NEW)**:
- Must add to `MODEL_SPECIFICATIONS` dictionary
- Dimension: 768 (verified from Salesforce/codet5p-110m-embedding docs)
- Type: "code"

### 2.4 create_embedding_function() Factory

**Location**: Lines 819-854
**Signature**: `create_embedding_function(model_name: str | None = None, cache_dir: Path | None = None, cache_size: int = 1000)`

```python
# Lines 837-844: Environment variable handling
env_model = os.environ.get("MCP_VECTOR_SEARCH_EMBEDDING_MODEL")
if env_model:
    model_name = env_model
elif model_name is None:
    model_name = _default_model_for_device()

# Line 848: Create embedding function
embedding_function = CodeBERTEmbeddingFunction(model_name)

# Lines 850-852: Optional cache
if cache_dir:
    cache = EmbeddingCache(cache_dir, cache_size)

return embedding_function, cache
```

**Usage for index-code**:
```python
# Create CodeT5+ embedding function
embedding_func, cache = create_embedding_function(
    model_name="Salesforce/codet5p-110m-embedding",
    cache_dir=project_root / ".mcp-vector-search" / "cache" / "code",
    cache_size=1000
)
```

---

## 3. Index Command Architecture (`cli/commands/index.py`)

### 3.1 Command Registration

**File**: `src/mcp_vector_search/cli/commands/index.py`
**Location**: Likely lines 100-200 (persisted output, not fully visible)

**Expected pattern** (based on other commands):
```python
from typer import Typer

app = Typer()

@app.command("main")  # or just function name becomes command
def index(
    project_root: Path = typer.Argument(Path.cwd()),
    model: str = typer.Option("code", "--model", "-m"),
    force: bool = typer.Option(False, "--force", "-f"),
    # ... other options
):
    """Index project for semantic search."""
    # Implementation
```

**CLI integration**: `src/mcp_vector_search/cli/main.py`
- Line 68: `app = create_enhanced_typer()` creates main app
- Commands registered as subcommands via imports

### 3.2 Model Option and Presets

**Location**: `config/defaults.py` lines 215-224

```python
DEFAULT_EMBEDDING_MODELS = {
    "code": "sentence-transformers/all-MiniLM-L6-v2",     # Default
    "graphcodebert": "microsoft/graphcodebert-base",      # 768D
    "precise": "Salesforce/SFR-Embedding-Code-400M_R",    # 1024D
    "fast": "sentence-transformers/all-MiniLM-L6-v2",     # 384D
}
```

**Proposed for index-code**:
- Remove `--model` option with presets
- Add `--model-name` option for direct model specification
- Default: `"Salesforce/codet5p-110m-embedding"` or similar CodeT5+ model

### 3.3 Indexer Instantiation

**From `core/indexer.py`** (lines 78-241):

```python
class SemanticIndexer:
    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        config: ProjectConfig | None = None,
        max_workers: int | None = None,
        batch_size: int | None = None,
        # ... other params
    ):
        # Lines 239-241: Two-phase backends
        index_path = project_root / ".mcp-vector-search" / "lance"
        self.chunks_backend = ChunksBackend(index_path)
        self.vectors_backend = VectorsBackend(index_path)
```

**Key components**:
1. **ChunksBackend**: Stores parsed chunks (Phase 1, shared across all indexes)
2. **VectorsBackend**: Stores embeddings (Phase 2, separate per model)
3. **Embedding function**: Created via `create_embedding_function()`

---

## 4. Search Integration (`core/search.py`)

### 4.1 VectorsBackend Detection

**Location**: Lines 494-533
**Method**: `_check_vectors_backend()`

```python
def _check_vectors_backend(self) -> None:
    # Line 504-514: Detect index path from database
    if hasattr(self.database, "persist_directory"):
        index_path = self.database.persist_directory

        # Handle both old and new path formats
        if index_path.name == "lance":
            lance_path = index_path
        else:
            lance_path = index_path / "lance"

        # Line 517-518: Check if vectors.lance exists
        vectors_path = lance_path / "vectors.lance"
        if vectors_path.exists() and vectors_path.is_dir():
            # Line 520-523: Instantiate VectorsBackend
            vectors_backend = VectorsBackend(lance_path)
            self._vectors_backend = vectors_backend
```

**For code_vectors support**:
- Need to detect both `vectors.lance` and `code_vectors.lance`
- Decision logic: Which table to query?
- Options:
  1. Query both, merge results
  2. User flag: `--code-only` or `--general-only`
  3. Automatic: Query code_vectors for code queries, vectors for general

### 4.2 Search Implementation

**Location**: Lines 617-729
**Method**: `async def _search_vectors_backend(query, limit, filters, threshold)`

```python
# Lines 640-652: Get embedding function from database
if hasattr(self.database, "_embedding_function"):
    embedding_func = self.database._embedding_function
else:
    raise SearchError("Cannot access embedding function")

# Line 692: Generate query embedding
query_vector = embedding_func([query])[0]

# Lines 695-697: Search vectors backend
raw_results = await self._vectors_backend.search(
    query_vector, limit=limit, filters=filters
)

# Lines 700-723: Convert to SearchResult format
for result in raw_results:
    distance = result.get("_distance", 1.0)
    similarity = max(0.0, 1.0 - (distance / 2.0))

    search_result = SearchResult(
        file_path=Path(result["file_path"]),
        content=result["content"],
        start_line=result["start_line"],
        end_line=result["end_line"],
        language=result.get("language"),
        similarity_score=similarity,
        rank=idx + 1,
        chunk_type=result.get("chunk_type"),
        function_name=result.get("name"),
    )
```

**Critical issue**: Model mismatch detection (lines 655-689)
- Checks stored `model_version` against current model
- Warns if mismatch detected
- Raises error if dimension mismatch

**For dual-index search**:
- Must ensure query uses matching embedding model
- code_vectors queries → CodeT5+ embeddings
- vectors queries → MiniLM embeddings

### 4.3 Result Ranking

**Location**: Lines 169
**Method**: `_result_ranker.rerank_results(enhanced_results, query)`

**From `result_ranker.py`** (not read, but referenced):
- Re-ranks results based on query relevance
- Considers similarity scores, chunk types, etc.

---

## 5. Indexer Workflow (`core/indexer.py`)

### 5.1 Chunk Collection and Filtering

**Location**: Lines 1-300 (partial read)

**Key components**:
1. **FileDiscovery** (line 159): Find files to index
2. **ChunkProcessor** (line 212): Parse files into chunks
3. **ChunksBackend** (line 240): Store chunks (Phase 1)
4. **VectorsBackend** (line 241): Store embeddings (Phase 2)

**Chunk types** (inferred from schema):
- `function`: Top-level functions
- `class`: Class definitions
- `method`: Class methods
- Others: Varies by language parser

### 5.2 Code vs Text Chunks

**Distinction mechanism** (inferred):
- **chunk_type field**: Values like "function", "class", "method" indicate code
- **language field**: Programming languages vs "markdown", "text"
- **Filtering for code-only index**:
  ```python
  code_chunk_types = {"function", "class", "method", "interface", "struct"}
  code_chunks = [
      chunk for chunk in all_chunks
      if chunk["chunk_type"] in code_chunk_types
  ]
  ```

**File-level filtering**:
- Exclude documentation: `.md`, `.rst`, `.txt`
- Include code only: `.py`, `.js`, `.ts`, etc.

### 5.3 Embedding Pipeline

**Producer/Consumer pattern** (from indexer.py context):
- **Producer**: ChunkProcessor yields chunks
- **Consumer**: Embedding function processes batches
- **Batch size**: Configurable, default 256 (line 146)

**From BatchEmbeddingProcessor** (`embeddings.py` lines 555-777):
```python
class BatchEmbeddingProcessor:
    def __init__(
        self,
        embedding_function: CodeBERTEmbeddingFunction,
        cache: EmbeddingCache | None = None,
        batch_size: int | None = None,
    ):
        # Line 574-576: Auto-detect batch size
        if batch_size is None:
            batch_size = _detect_optimal_batch_size()

    async def process_batch(self, contents: list[str]) -> list[list[float]]:
        # Lines 677-685: Check cache
        # Lines 693-723: Generate embeddings
        # Lines 720-723: Use parallel or sequential
        if use_parallel and len(uncached_contents) >= 16:
            new_embeddings = await self.embed_batches_parallel(...)
        else:
            new_embeddings = await self._sequential_embed(...)
```

**Key insights**:
- Caching reduces redundant embeddings
- Parallel processing for large batches (16+ items)
- Optimal batch size: 128-512 depending on GPU

---

## 6. CLI Registration Pattern

### 6.1 Main App Structure

**File**: `src/mcp_vector_search/cli/main.py`
**Lines**: 1-100

```python
# Line 68: Create main Typer app
app = create_enhanced_typer(
    name="mcp-vector-search",
    help="...",
)

# Commands imported and registered as subcommands
# Pattern: from .commands.module import app as module_app
# Then: app.add_typer(module_app, name="command-name")
```

### 6.2 Command Registration Examples

**From grep results**:
- `@app.command("main")` - Primary command in submodule
- `@app.command("status")` - Subcommand
- `@app.command()` - Use function name as command name

**Expected for index-code**:
```python
# In src/mcp_vector_search/cli/commands/index_code.py
from typer import Typer

app = Typer()

@app.command("main")  # Will be accessible as: mcp-vector-search index-code
def index_code(
    project_root: Path = typer.Argument(Path.cwd()),
    model_name: str = typer.Option(
        "Salesforce/codet5p-110m-embedding",
        "--model-name",
        "-m",
        help="CodeT5+ model for embeddings"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force rebuild code index"
    ),
):
    """Index code chunks with CodeT5+ embeddings."""
    # Implementation
```

**Registration in main.py**:
```python
# In main.py (add near other command imports)
from .commands.index_code import app as index_code_app

# Register subcommand
app.add_typer(index_code_app, name="index-code")
```

---

## 7. File Watcher Integration (`core/watcher.py`)

### 7.1 FileWatcher Class

**Location**: Lines 122-150
**Constructor signature**:
```python
def __init__(
    self,
    project_root: Path,
    config: ProjectConfig,
    indexer: SemanticIndexer,
    database: VectorDatabase,
):
    self.project_root = project_root
    self.config = config
    self.indexer = indexer
    self.database = database
    self.observer: Observer | None = None
    self.handler: CodeFileHandler | None = None
    self.is_running = False
```

### 7.2 Event Handler

**Location**: Lines 18-120
**Class**: `CodeFileHandler`

```python
class CodeFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        file_extensions: list[str],
        ignore_patterns: list[str],
        callback: Callable[[str, str], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        debounce_delay: float = 1.0,
    ):
        # Event handling with debouncing

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
```

**Key features**:
- **Debouncing**: Lines 88-100 - delays processing for 1 second
- **Async callback**: Lines 116-119 - calls async indexer methods
- **Event types**: created, modified, deleted, moved

**For dual-index support**:
- Watcher needs to trigger BOTH index updates:
  1. Update `vectors.lance` (general embeddings)
  2. Update `code_vectors.lance` (CodeT5+ embeddings)
- Approach: Extend callback to handle multiple backends

---

## 8. Implementation Requirements

### 8.1 VectorsBackend Modifications

**Required changes**:
1. Add `table_name` parameter to constructor
2. Replace hardcoded `TABLE_NAME` with instance variable
3. Update all references to use `self.table_name`

**Example**:
```python
class VectorsBackend:
    def __init__(
        self,
        db_path: Path,
        vector_dim: int | None = None,
        table_name: str = "vectors",  # NEW PARAMETER
    ):
        self.db_path = Path(db_path)
        self._db = None
        self._table = None
        self.vector_dim = vector_dim
        self.table_name = table_name  # NEW ATTRIBUTE

    # Update all methods to use self.table_name instead of TABLE_NAME
    async def initialize(self):
        if self.table_name in table_names:  # Was: TABLE_NAME
            self._table = self._db.open_table(self.table_name)
```

### 8.2 Embedding Function Setup

**Required steps**:
1. Add CodeT5+ model to `MODEL_SPECIFICATIONS` in `config/defaults.py`
2. Create embedding function with CodeT5+ model:
   ```python
   from mcp_vector_search.core.embeddings import create_embedding_function

   embedding_func, cache = create_embedding_function(
       model_name="Salesforce/codet5p-110m-embedding",
       cache_dir=project_root / ".mcp-vector-search" / "cache" / "code",
       cache_size=1000,
   )
   ```

**Model specification** (add to defaults.py):
```python
"Salesforce/codet5p-110m-embedding": {
    "dimensions": 768,  # Verify from model docs
    "context_length": 512,
    "type": "code",
    "description": "CodeT5+ 110M: Code-specific embeddings",
},
```

### 8.3 Chunk Filtering for Code-Only

**Required logic**:
```python
def filter_code_chunks(chunks: list[dict]) -> list[dict]:
    """Filter chunks to only include code (exclude docs)."""
    CODE_CHUNK_TYPES = {
        "function", "class", "method", "interface",
        "struct", "enum", "trait", "module"
    }

    DOC_LANGUAGES = {"markdown", "text", "rst"}

    return [
        chunk for chunk in chunks
        if chunk["chunk_type"] in CODE_CHUNK_TYPES
        and chunk["language"] not in DOC_LANGUAGES
    ]
```

### 8.4 Index Command Implementation

**File**: `src/mcp_vector_search/cli/commands/index_code.py` (NEW)

**Structure**:
```python
import asyncio
from pathlib import Path
import typer
from mcp_vector_search.core.factory import create_database
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.vectors_backend import VectorsBackend

app = typer.Typer()

@app.command("main")
async def index_code(
    project_root: Path = typer.Argument(Path.cwd()),
    model_name: str = typer.Option(
        "Salesforce/codet5p-110m-embedding",
        "--model-name",
        "-m",
    ),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Index code chunks with CodeT5+ embeddings."""

    # 1. Create database and indexer (standard setup)
    database = await create_database(project_root)
    indexer = SemanticIndexer(database, project_root)

    # 2. Create CodeT5+ embedding function
    embedding_func, cache = create_embedding_function(
        model_name=model_name,
        cache_dir=project_root / ".mcp-vector-search" / "cache" / "code",
    )

    # 3. Create code_vectors backend
    lance_path = project_root / ".mcp-vector-search" / "lance"
    code_vectors_backend = VectorsBackend(
        lance_path,
        vector_dim=768,  # CodeT5+ dimension
        table_name="code_vectors",
    )
    await code_vectors_backend.initialize()

    # 4. Get chunks from chunks_backend (Phase 1 data)
    all_chunks = await indexer.chunks_backend.get_all_chunks()

    # 5. Filter to code-only chunks
    code_chunks = filter_code_chunks(all_chunks)

    # 6. Generate embeddings
    contents = [chunk["content"] for chunk in code_chunks]
    embeddings = await embedding_func(contents)

    # 7. Prepare chunks with vectors
    chunks_with_vectors = []
    for chunk, vector in zip(code_chunks, embeddings):
        chunks_with_vectors.append({
            **chunk,
            "vector": vector,
        })

    # 8. Add to code_vectors table
    await code_vectors_backend.add_vectors(
        chunks_with_vectors,
        model_version=model_name,
    )

    print(f"✓ Indexed {len(code_chunks)} code chunks with CodeT5+")
```

### 8.5 Search Integration

**Modifications to `core/search.py`**:

1. **Detect code_vectors table** (in `_check_vectors_backend()`):
   ```python
   code_vectors_path = lance_path / "code_vectors.lance"
   if code_vectors_path.exists():
       self._code_vectors_backend = VectorsBackend(
           lance_path, table_name="code_vectors"
       )
   ```

2. **Decide which backend to query**:
   ```python
   async def search(self, query: str, use_code_vectors: bool = False, ...):
       backend = (
           self._code_vectors_backend if use_code_vectors and self._code_vectors_backend
           else self._vectors_backend
       )
       results = await self._search_vectors_backend(query, backend, ...)
   ```

3. **Add CLI flag**:
   ```python
   @app.command("search")
   def search_command(
       query: str,
       use_code_vectors: bool = typer.Option(
           False, "--code", "-c",
           help="Search code_vectors index (CodeT5+)"
       ),
   ):
       # Pass flag to search engine
   ```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Test VectorsBackend with custom table_name**:
```python
async def test_custom_table_name():
    backend = VectorsBackend(
        db_path=tmp_path,
        table_name="code_vectors"
    )
    await backend.initialize()

    # Verify table created with correct name
    assert (tmp_path / "code_vectors.lance").exists()
```

**Test chunk filtering**:
```python
def test_filter_code_chunks():
    chunks = [
        {"chunk_type": "function", "language": "python"},
        {"chunk_type": "text", "language": "markdown"},
    ]

    code_chunks = filter_code_chunks(chunks)
    assert len(code_chunks) == 1
    assert code_chunks[0]["chunk_type"] == "function"
```

### 9.2 Integration Tests

**Test end-to-end indexing**:
```python
async def test_index_code_command():
    # 1. Create test project
    # 2. Run index-code command
    # 3. Verify code_vectors.lance created
    # 4. Verify embeddings have correct dimensions
    # 5. Test search with CodeT5+ embeddings
```

### 9.3 Performance Tests

**Compare search quality**:
- Run same queries on `vectors` vs `code_vectors`
- Measure relevance scores
- Collect user feedback on result quality

**Benchmark embedding speed**:
- Measure CodeT5+ vs MiniLM throughput
- Test on different hardware (CPU, CUDA, MPS)

---

## 10. Deployment Considerations

### 10.1 Backward Compatibility

**Ensure existing installations work**:
- `vectors.lance` remains default table
- `index-code` is optional enhancement
- Existing search commands use `vectors.lance` by default

### 10.2 Migration Path

**For users with existing indexes**:
1. Keep existing `vectors.lance` intact
2. Add `index-code` command to CLI help
3. Provide migration guide in docs
4. Optional: Auto-detect and suggest `index-code` after regular index

### 10.3 Documentation Requirements

**Add to user docs**:
- What is `index-code` and why use it?
- Performance comparison: MiniLM vs CodeT5+
- When to use `--code` search flag
- Disk space requirements (dual indexes)

**Add to developer docs**:
- VectorsBackend architecture with custom tables
- Embedding function selection strategy
- Testing strategy for dual-index features

---

## 11. Open Questions

### 11.1 Architecture Decisions

**Q1**: Should search query BOTH indexes by default or require flag?
- **Option A**: Query both, merge results (complexity: scoring normalization)
- **Option B**: User chooses via `--code` flag (simpler, clearer)
- **Recommendation**: Option B for v1, Option A for v2

**Q2**: How to handle model updates?
- **Challenge**: CodeT5+ embeddings incompatible with MiniLM
- **Solution**: Version tracking in `model_version` field, warn on mismatch

**Q3**: File watcher support for dual indexes?
- **Challenge**: Update both indexes on file change
- **Solution**: Extend watcher callback to trigger both backends

### 11.2 Performance Optimization

**Q1**: Batch processing for large codebases?
- **Current**: 256 file batch size (line 146 in indexer.py)
- **CodeT5+**: May need smaller batches due to model size
- **Action**: Test and tune batch size for CodeT5+

**Q2**: Cache management for dual embeddings?
- **Current**: Single cache directory
- **Proposal**: Separate cache dirs: `cache/general/` and `cache/code/`

---

## 12. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Modify `VectorsBackend` to support custom `table_name`
- [ ] Add CodeT5+ model specs to `MODEL_SPECIFICATIONS`
- [ ] Test VectorsBackend with `code_vectors` table
- [ ] Add unit tests for custom table names

### Phase 2: CLI Command
- [ ] Create `cli/commands/index_code.py`
- [ ] Implement `index_code()` function
- [ ] Add chunk filtering logic
- [ ] Register command in `cli/main.py`
- [ ] Add CLI help text and documentation

### Phase 3: Search Integration
- [ ] Extend `_check_vectors_backend()` to detect code_vectors
- [ ] Add `--code` flag to search command
- [ ] Update `_search_vectors_backend()` to select backend
- [ ] Handle model mismatch warnings

### Phase 4: Testing
- [ ] Unit tests for VectorsBackend table_name
- [ ] Unit tests for chunk filtering
- [ ] Integration test: index-code command
- [ ] Integration test: search with --code flag
- [ ] Performance benchmarks: MiniLM vs CodeT5+

### Phase 5: Documentation
- [ ] Update README with index-code usage
- [ ] Add developer guide for dual-index architecture
- [ ] Create migration guide for existing users
- [ ] Add performance comparison section

### Phase 6: Future Enhancements
- [ ] File watcher support for dual indexes
- [ ] Automatic query routing (code vs general)
- [ ] Result merging and score normalization
- [ ] Model update and migration tools

---

## Appendix A: File Paths Reference

```
Project Structure:
├── src/mcp_vector_search/
│   ├── core/
│   │   ├── vectors_backend.py      # Lines 29-944: Schema, add_vectors(), search()
│   │   ├── embeddings.py           # Lines 373-854: CodeBERTEmbeddingFunction
│   │   ├── indexer.py              # Lines 1-300: SemanticIndexer setup
│   │   ├── search.py               # Lines 494-729: VectorsBackend integration
│   │   └── watcher.py              # Lines 18-150: File watcher
│   ├── cli/
│   │   ├── main.py                 # Lines 1-100: Main app, command registration
│   │   └── commands/
│   │       ├── index.py            # Existing index command
│   │       └── index_code.py       # NEW: Code-specific index command
│   └── config/
│       └── defaults.py             # Lines 215-573: Model specs, dimensions
└── .mcp-vector-search/
    └── lance/                      # LanceDB directory
        ├── vectors.lance/          # General embeddings (MiniLM)
        └── code_vectors.lance/     # NEW: Code embeddings (CodeT5+)
```

---

## Appendix B: Key Line Numbers

### vectors_backend.py
- **Schema definition**: Lines 29-57
- **Constructor**: Lines 99-110
- **TABLE_NAME constant**: Line 96
- **add_vectors()**: Lines 352-546
- **search()**: Lines 548-674
- **Dimension detection**: Lines 401-407
- **Deduplication**: Lines 478-479, 532-533

### embeddings.py
- **CodeBERTEmbeddingFunction.__init__()**: Lines 376-477
- **Model loading**: Lines 419-422
- **Dimension detection**: Line 428
- **create_embedding_function()**: Lines 819-854
- **BatchEmbeddingProcessor**: Lines 555-777

### search.py
- **VectorsBackend detection**: Lines 494-533
- **_search_vectors_backend()**: Lines 617-729
- **Model mismatch check**: Lines 655-689

### indexer.py
- **Constructor**: Lines 78-255
- **Backend initialization**: Lines 239-241
- **Batch size**: Lines 126-148

### defaults.py
- **MODEL_SPECIFICATIONS**: Lines 227-273
- **get_model_dimensions()**: Lines 506-533
- **DEFAULT_EMBEDDING_MODELS**: Lines 215-224

---

## Research Classification

**Type**: Actionable Implementation Research

**Outputs**:
- Architectural specifications ✓
- Implementation checklist ✓
- Code examples ✓
- Testing strategy ✓

**Next Steps**:
1. Begin Phase 1 implementation (VectorsBackend modification)
2. Create unit tests for custom table names
3. Implement index-code CLI command
4. Integration testing with CodeT5+ model

---

**Research completed**: 2026-02-20
**Document version**: 1.0
**Status**: Ready for implementation
