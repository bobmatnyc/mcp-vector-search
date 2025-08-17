# MCP Vector Search - Lightweight ChromaDB Implementation PRD

## Executive Summary

This PRD outlines the development of **`mcp-vector-search`** - a CLI-first Python package that provides high-precision semantic code search with optional MCP integration, optimized for lightweight local deployment using ChromaDB. Based on competitive analysis showing persistent precision problems in existing tools and zero dedicated MCP semantic search solutions, this package addresses a clear market gap with modern Python development practices.

**Key Finding**: 61% of developers spend 30+ minutes daily searching for code, while GitHub's 6+ year struggle with semantic search and developer frustration with "pages of irrelevant content" demonstrates the precision challenge. The MCP ecosystem lacks any dedicated semantic code search servers, presenting a first-mover opportunity.

### Design Principles

**CLI-First Development**: Robust command-line interface for immediate testing and local workflows  
**Pure Python Implementation**: Leveraging Tree-sitter language packs for multi-language support  
**Lightweight Local Deployment**: ChromaDB for zero-config vector storage  
**Project-Aware**: Simple initialization with persistent configuration and indexing  
**Precision-Focused**: AST-based understanding to solve relevance problems plaguing existing tools  
**Local-First Privacy**: Complete on-device processing with no external dependencies  
**Modern Python Standards**: Type-safe, async-first, production-ready codebase  
**MCP-Ready Architecture**: Designed for seamless MCP server integration as Phase 2

---

## Competitive Landscape Analysis

### Market Gaps Identified

**No MCP-Native Solutions**: Despite explosive MCP adoption since November 2024, zero dedicated semantic code search servers exist. Current MCP implementations focus on repository management rather than deep semantic understanding.

**Persistent Precision Problems**: GitHub Code Search after 6+ years still suffers from duplicate results, poor ranking, and irrelevant content. Developer trust issues with 31% distrusting AI tool accuracy create opportunity for precision-focused alternatives.

**Privacy-First Demand**: Tabnine markets as "the only air-gapped AI platform," indicating clear market demand. Regulated industries require local processing for HIPAA, SOC2, and security clearance compliance.

### Technical Performance Benchmarks

**Tree-sitter Dominance**: 36x performance improvement over alternatives, 80ms for 6000-line files, incremental parsing with sub-100ms latency  
**ChromaDB Local Performance**: Single-machine deployment, SQLite backend, sub-10ms vector similarity queries  
**GitHub's Scale Challenge**: 115TB processed into 25TB index, sub-100ms p99 response times at 640 queries/second  
**Sourcegraph Hybrid Success**: 30%+ completion acceptance rates using dense-sparse vector fusion

---

## Technology Stack - Lightweight & Modern

### Core Dependencies

**Parsing**: Tree-sitter (36x performance advantage, 80+ languages, incremental updates)  
**Embeddings**: sentence-transformers with CodeBERT models (local inference)  
**Vector Database**: ChromaDB (SQLite backend, zero-config deployment, Python-native)  
**File Watching**: Watchdog (cross-platform, efficient change detection)  
**CLI Framework**: Typer (type-safe CLI with automatic help generation)  
**Async Runtime**: asyncio with aiofiles for non-blocking I/O  
**Configuration**: Pydantic for type-safe settings management  
**Package Management**: UV for ultra-fast dependency resolution and installation  
**Build Backend**: Hatchling (PEP 621 compliant, modern Python standards)

### Modern Python Best Practices Integration

**Type Safety**: Complete type annotations using mypy-strict mode  
**Code Quality**: Black formatting, Ruff linting, isort import sorting  
**Testing**: pytest with async support, fixtures, and parametrized tests  
**Logging**: Structured logging with loguru for better debugging  
**Error Handling**: Custom exception hierarchy with contextual error information  
**Async Patterns**: async/await throughout, proper resource management  
**Configuration**: Environment-aware settings with validation  
**Dependency Management**: Poetry for reproducible builds  
**Documentation**: Sphinx with type hint integration  
**CI/CD**: GitHub Actions with comprehensive testing matrix

---

## Core Architecture

### CLI-First Interface Design

```bash
# Project initialization
mcp-vector-search init [--config CONFIG_FILE]

# Indexing operations  
mcp-vector-search index [--watch] [--incremental] [--extensions EXTS]
mcp-vector-search reindex [--force] [--path PATH]

# Search operations
mcp-vector-search search "authentication middleware" [--limit 10] [--files "*.py"]
mcp-vector-search similar --file src/auth.py --function login_handler
mcp-vector-search context "implement rate limiting" [--focus security,middleware]

# Project management
mcp-vector-search status [--verbose] [--health-check]
mcp-vector-search config [--show] [--set key=value] [--global]
mcp-vector-search clean [--cache] [--index] [--logs]

# Development and debugging
mcp-vector-search benchmark [--query-file QUERIES] [--report FORMAT]
mcp-vector-search explain --query "your search" [--debug]
mcp-vector-search validate [--index] [--config]
```

### Project Structure - Modern Python Standards

```
mcp-vector-search/
├── pyproject.toml              # Poetry + hatchling configuration
├── README.md                   # Installation and usage guide
├── .python-version             # pyenv version specification
├── .pre-commit-config.yaml     # Git hooks for code quality
├── mypy.ini                    # Type checking configuration
├── pytest.ini                 # Test configuration
├── src/
│   └── mcp_vector_search/
│       ├── __init__.py
│       ├── py.typed            # PEP 561 type information
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py         # Typer CLI application
│       │   ├── commands/       # Command implementations
│       │   │   ├── __init__.py
│       │   │   ├── init.py     # Project initialization
│       │   │   ├── index.py    # Indexing operations
│       │   │   ├── search.py   # Search operations
│       │   │   ├── config.py   # Configuration management
│       │   │   └── status.py   # Status and diagnostics
│       │   └── output.py       # Rich formatting and display
│       ├── core/
│       │   ├── __init__.py
│       │   ├── exceptions.py   # Custom exception hierarchy
│       │   ├── project.py      # Project detection and management
│       │   ├── indexer.py      # Tree-sitter based code parsing
│       │   ├── embeddings.py   # Sentence-transformers integration
│       │   ├── database.py     # ChromaDB vector operations
│       │   ├── search.py       # Semantic search engine
│       │   └── watcher.py      # File system monitoring
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract parser interface
│       │   ├── python.py       # Python-specific parsing
│       │   ├── javascript.py   # JavaScript/TypeScript parsing
│       │   └── registry.py     # Language parser registry
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py     # Pydantic configuration schemas
│       │   └── defaults.py     # Default configurations
│       └── mcp/                # Phase 2: MCP server integration
│           ├── __init__.py
│           ├── server.py       # MCP protocol implementation
│           └── tools.py        # MCP tool definitions
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_cli/             # CLI command tests
│   ├── test_core/            # Core functionality tests
│   ├── test_parsers/         # Parser implementation tests
│   └── fixtures/             # Test codebases and examples
├── docs/
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Documentation index
│   ├── api/                  # API documentation
│   └── guides/               # User guides
└── scripts/
    ├── dev-setup.py          # Development environment setup
    ├── benchmark.py          # Performance benchmarking
    └── migrate-config.py     # Configuration migration tools
```

---

## Modern Python Development Standards

### Type Safety & Code Quality

```python
# Type-safe configuration with Pydantic
from pydantic import BaseSettings, Field, validator
from pathlib import Path
from typing import Optional, List, Dict, Any

class ProjectConfig(BaseSettings):
    """Type-safe project configuration with validation."""
    
    project_root: Path = Field(..., description="Project root directory")
    index_path: Path = Field(default=".mcp-vector-search", description="Index storage path")
    file_extensions: List[str] = Field(default=[".py", ".js", ".ts"], description="File extensions to index")
    embedding_model: str = Field(default="microsoft/codebert-base", description="Embedding model name")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_chunk_size: int = Field(default=512, gt=0, description="Maximum chunk size in tokens")
    languages: List[str] = Field(default=[], description="Detected programming languages")
    
    @validator('project_root', 'index_path')
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are absolute and normalized."""
        return v.resolve()
    
    @validator('file_extensions')
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Ensure extensions start with dot."""
        return [ext if ext.startswith('.') else f'.{ext}' for ext in v]
    
    class Config:
        env_prefix = "MCP_VECTOR_SEARCH_"
        case_sensitive = False

# Async-first database operations with proper error handling
from chromadb import AsyncClient
from chromadb.config import Settings
import asyncio
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        ...

class VectorDatabase:
    """ChromaDB-based vector database with async operations."""
    
    def __init__(self, db_path: Path, embedding_function: EmbeddingFunction) -> None:
        self.db_path = db_path
        self.embedding_function = embedding_function
        self._client: Optional[AsyncClient] = None
        self._collection = None
    
    async def __aenter__(self) -> "VectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.db_path),
                anonymized_telemetry=False
            )
            
            self._client = AsyncClient(settings)
            self._collection = await self._client.get_or_create_collection(
                name="code_embeddings",
                embedding_function=self.embedding_function
            )
            
        except Exception as e:
            raise DatabaseInitializationError(f"Failed to initialize ChromaDB: {e}") from e
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """Add documents to the collection with proper error handling."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")
        
        try:
            await self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise DocumentAdditionError(f"Failed to add documents: {e}") from e
    
    async def search(
        self,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")
        
        try:
            results = await self._collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e
    
    async def close(self) -> None:
        """Close database connection."""
        if self._client:
            # ChromaDB async client cleanup
            self._client = None
            self._collection = None
```

### Exception Hierarchy & Error Handling

```python
# Custom exception hierarchy for better error handling
class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.context = context or {}

class DatabaseError(MCPVectorSearchError):
    """Database-related errors."""
    pass

class DatabaseInitializationError(DatabaseError):
    """Database initialization failed."""
    pass

class DatabaseNotInitializedError(DatabaseError):
    """Operation attempted on uninitialized database."""
    pass

class DocumentAdditionError(DatabaseError):
    """Failed to add documents to database."""
    pass

class SearchError(DatabaseError):
    """Search operation failed."""
    pass

class ParsingError(MCPVectorSearchError):
    """Code parsing errors."""
    pass

class EmbeddingError(MCPVectorSearchError):
    """Embedding generation errors."""
    pass

class ConfigurationError(MCPVectorSearchError):
    """Configuration validation errors."""
    pass

# Error handling with context preservation
from loguru import logger
import traceback
from typing import TypeVar, Callable, Any

T = TypeVar('T')

def handle_errors(
    operation_name: str,
    reraise: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for consistent error handling and logging."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except MCPVectorSearchError:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in {operation_name}",
                    error=str(e),
                    traceback=traceback.format_exc(),
                    args=args,
                    kwargs=kwargs
                )
                if reraise:
                    raise MCPVectorSearchError(
                        f"Unexpected error in {operation_name}: {e}",
                        context={
                            "operation": operation_name,
                            "original_error": str(e),
                            "error_type": type(e).__name__
                        }
                    ) from e
                return None
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except MCPVectorSearchError:
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in {operation_name}",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                if reraise:
                    raise MCPVectorSearchError(
                        f"Unexpected error in {operation_name}: {e}",
                        context={
                            "operation": operation_name,
                            "original_error": str(e)
                        }
                    ) from e
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
```

### Structured Logging & Monitoring

```python
# Structured logging with performance monitoring
from loguru import logger
import time
from functools import wraps
from typing import Any, Dict
import sys

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured: bool = True
) -> None:
    """Configure structured logging with performance monitoring."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with rich formatting
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler with JSON format for structured logs
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="1 month",
            serialize=structured
        )

def monitor_performance(
    operation_name: str,
    log_threshold_ms: float = 100.0
) -> Callable:
    """Decorator to monitor operation performance."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                if duration_ms > log_threshold_ms:
                    logger.warning(
                        f"Slow operation: {operation_name}",
                        duration_ms=duration_ms,
                        threshold_ms=log_threshold_ms
                    )
                else:
                    logger.debug(
                        f"Operation completed: {operation_name}",
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Operation failed: {operation_name}",
                    duration_ms=duration_ms,
                    error=str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                if duration_ms > log_threshold_ms:
                    logger.warning(
                        f"Slow operation: {operation_name}",
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Operation failed: {operation_name}",
                    duration_ms=duration_ms,
                    error=str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
```

### Testing Framework & Fixtures

```python
# Comprehensive testing setup with async support
import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from mcp_vector_search.core.database import VectorDatabase
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.config.settings import ProjectConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_project_dir() -> AsyncGenerator[Path, None]:
    """Create a temporary project directory with sample code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create sample Python file
        (project_path / "src").mkdir()
        (project_path / "src" / "main.py").write_text("""
def hello_world():
    '''A simple greeting function.'''
    return "Hello, World!"

class Calculator:
    '''A basic calculator class.'''
    
    def add(self, a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b
""")
        
        # Create sample JavaScript file
        (project_path / "src" / "utils.js").write_text("""
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
    }
    
    emit(event, ...args) {
        if (this.events[event]) {
            this.events[event].forEach(listener => listener(...args));
        }
    }
}
""")
        
        yield project_path

@pytest.fixture
async def mock_embedding_function() -> AsyncMock:
    """Mock embedding function for testing."""
    mock = AsyncMock()
    mock.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(10)]
    return mock

@pytest.fixture
async def vector_db(
    temp_project_dir: Path,
    mock_embedding_function: AsyncMock
) -> AsyncGenerator[VectorDatabase, None]:
    """Create a test vector database."""
    db_path = temp_project_dir / ".mcp-vector-search"
    db = VectorDatabase(db_path, mock_embedding_function)
    
    async with db:
        yield db

@pytest.fixture
def project_config(temp_project_dir: Path) -> ProjectConfig:
    """Create a test project configuration."""
    return ProjectConfig(
        project_root=temp_project_dir,
        index_path=temp_project_dir / ".mcp-vector-search",
        file_extensions=[".py", ".js", ".ts"],
        embedding_model="test-model",
        similarity_threshold=0.7
    )

@pytest.fixture
def project_manager(
    temp_project_dir: Path,
    project_config: ProjectConfig
) -> ProjectManager:
    """Create a test project manager."""
    return ProjectManager(temp_project_dir, project_config)

# Parametrized tests for multiple scenarios
@pytest.mark.parametrize("query,expected_results", [
    ("hello world function", 1),
    ("calculator class", 1),
    ("add two numbers", 1),
    ("javascript debounce", 1),
    ("event emitter pattern", 1),
    ("nonexistent functionality", 0),
])
async def test_semantic_search_precision(
    vector_db: VectorDatabase,
    query: str,
    expected_results: int
):
    """Test semantic search precision across different queries."""
    # Add sample documents
    await vector_db.add_documents(
        documents=["Hello world function", "Calculator class", "Add two numbers"],
        metadatas=[{"file": "main.py"}, {"file": "main.py"}, {"file": "main.py"}],
        ids=["1", "2", "3"]
    )
    
    results = await vector_db.search(
        query_texts=[query],
        n_results=10
    )
    
    assert len(results.get("documents", [[]])[0]) >= expected_results

# Property-based testing with hypothesis
from hypothesis import given, strategies as st

@given(
    chunk_size=st.integers(min_value=64, max_value=2048),
    similarity_threshold=st.floats(min_value=0.1, max_value=0.9)
)
async def test_configuration_validation(chunk_size: int, similarity_threshold: float):
    """Property-based testing for configuration validation."""
    config = ProjectConfig(
        project_root=Path("/tmp"),
        max_chunk_size=chunk_size,
        similarity_threshold=similarity_threshold
    )
    
    assert config.max_chunk_size == chunk_size
    assert config.similarity_threshold == similarity_threshold
    assert 0.0 <= config.similarity_threshold <= 1.0

# Performance benchmarking tests
@pytest.mark.benchmark
async def test_indexing_performance(
    temp_project_dir: Path,
    vector_db: VectorDatabase,
    benchmark
):
    """Benchmark indexing performance."""
    documents = [f"Function {i} for testing performance" for i in range(1000)]
    metadatas = [{"file": f"test_{i}.py"} for i in range(1000)]
    ids = [str(i) for i in range(1000)]
    
    result = await benchmark(
        vector_db.add_documents,
        documents,
        metadatas,
        ids
    )
    
    # Assert performance targets
    assert benchmark.stats.mean < 5.0  # Less than 5 seconds for 1000 documents

@pytest.mark.benchmark
async def test_search_performance(
    vector_db: VectorDatabase,
    benchmark
):
    """Benchmark search performance."""
    # Pre-populate database
    documents = [f"Function {i} implementation" for i in range(100)]
    metadatas = [{"file": f"test_{i}.py"} for i in range(100)]
    ids = [str(i) for i in range(100)]
    
    await vector_db.add_documents(documents, metadatas, ids)
    
    result = await benchmark(
        vector_db.search,
        query_texts=["function implementation"],
        n_results=10
    )
    
    # Assert performance targets: sub-100ms search
    assert benchmark.stats.mean < 0.1
```

---

## ChromaDB Integration Architecture

### Lightweight Database Setup

```python
# ChromaDB integration optimized for local development
from chromadb import AsyncClient, Collection
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

class CodeBERTEmbeddingFunction:
    """ChromaDB-compatible embedding function using CodeBERT."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base") -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

class ChromaVectorDatabase:
    """Lightweight ChromaDB implementation for code search."""
    
    def __init__(self, persist_directory: Path) -> None:
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_directory),
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        self.client: Optional[AsyncClient] = None
        self.collection: Optional[Collection] = None
        self.embedding_function = CodeBERTEmbeddingFunction()
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self.client = AsyncClient(self.settings)
            
            # Create or get collection with embedding function
            self.collection = await self.client.get_or_create_collection(
                name="code_search",
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Semantic code search collection",
                    "embedding_model": self.embedding_function.model_name
                }
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise DatabaseInitializationError(f"ChromaDB initialization failed: {e}")
    
    async def add_code_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> None:
        """Add code chunks to the database."""
        if not self.collection:
            raise DatabaseNotInitializedError("Database not initialized")
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Create searchable text combining content and context
            searchable_text = self._create_searchable_text(chunk)
            documents.append(searchable_text)
            
            # Store comprehensive metadata
            metadata = {
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "language": chunk["language"],
                "chunk_type": chunk.get("chunk_type", "code"),
                "function_name": chunk.get("function_name"),
                "class_name": chunk.get("class_name"),
                "docstring": chunk.get("docstring", ""),
                "imports": chunk.get("imports", []),
                "complexity": chunk.get("complexity_score", 0)
            }
            
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)
            
            # Create unique ID
            chunk_id = f"{chunk['file_path']}:{chunk['start_line']}:{chunk['end_line']}"
            ids.append(chunk_id)
        
        try:
            await self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} code chunks to database")
            
        except Exception as e:
            logger.error(f"Failed to add code chunks: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}")
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with filtering."""
        if not self.collection:
            raise DatabaseNotInitializedError("Database not initialized")
        
        try:
            # Build ChromaDB where clause from filters
            where_clause = self._build_where_clause(filters) if filters else None
            
            results = await self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process and filter results
            processed_results = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0], 
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= similarity_threshold:
                        result = {
                            "content": doc,
                            "metadata": metadata,
                            "similarity_score": similarity,
                            "rank": i + 1
                        }
                        processed_results.append(result)
            
            logger.debug(f"Found {len(processed_results)} results for query: {query}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Semantic search failed: {e}")
    
    def _create_searchable_text(self, chunk: Dict[str, Any]) -> str:
        """Create optimized searchable text from code chunk."""
        parts = []
        
        # Add main content
        parts.append(chunk["content"])
        
        # Add contextual information
        if chunk.get("function_name"):
            parts.append(f"Function: {chunk['function_name']}")
        
        if chunk.get("class_name"):
            parts.append(f"Class: {chunk['class_name']}")
        
        if chunk.get("docstring"):
            parts.append(f"Documentation: {chunk['docstring']}")
        
        # Add language and file context
        parts.append(f"Language: {chunk['language']}")
        parts.append(f"File: {Path(chunk['file_path']).name}")
        
        return "\n".join(parts)
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle list filters with $in operator
                where[key] = {"$in": value}
            elif isinstance(value, str) and value.startswith("!"):
                # Handle negation
                where[key] = {"$ne": value[1:]}
            else:
                # Direct equality
                where[key] = value
        
        return where
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.collection:
            raise DatabaseNotInitializedError("Database not initialized")
        
        try:
            count = await self.collection.count()
            
            # Get sample metadata for language distribution
            sample_results = await self.collection.get(
                limit=1000,
                include=["metadatas"]
            )
            
            language_counts = {}
            file_counts = {}
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    lang = metadata.get("language", "unknown")
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
                    file_path = metadata.get("file_path", "unknown")
                    file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            return {
                "total_chunks": count,
                "languages": language_counts,
                "files_indexed": len(file_counts),
                "database_path": str(self.persist_directory),
                "embedding_model": self.embedding_function.model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def reset_database(self) -> None:
        """Reset the database (useful for development)."""
        if self.client:
            try:
                await self.client.reset()
                logger.info("Database reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset database: {e}")
                raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self.client:
            # ChromaDB async client doesn't require explicit closing
            self.client = None
            self.collection = None
            logger.info("Database connections closed")
```

---

## Performance Optimization for ChromaDB

### Memory Management & Caching

```python
# Optimized caching and batch processing for ChromaDB
from functools import lru_cache
import hashlib
from typing import Tuple, Union
import pickle
import aiofiles
import json

class EmbeddingCache:
    """LRU cache for embeddings with disk persistence."""
    
    def __init__(self, cache_dir: Path, max_size: int = 1000) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self._memory_cache: Dict[str, List[float]] = {}
    
    def _hash_content(self, content: str) -> str:
        """Generate cache key from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get_embedding(self, content: str) -> Optional[List[float]]:
        """Get cached embedding for content."""
        cache_key = self._hash_content(content)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    embedding = json.loads(content)
                    
                    # Add to memory cache
                    if len(self._memory_cache) < self.max_size:
                        self._memory_cache[cache_key] = embedding
                    
                    return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    async def store_embedding(self, content: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        cache_key = self._hash_content(content)
        
        # Store in memory cache
        if len(self._memory_cache) < self.max_size:
            self._memory_cache[cache_key] = embedding
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(embedding))
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

class BatchEmbeddingProcessor:
    """Batch processing for efficient embedding generation."""
    
    def __init__(
        self,
        embedding_function: CodeBERTEmbeddingFunction,
        cache: EmbeddingCache,
        batch_size: int = 32
    ) -> None:
        self.embedding_function = embedding_function
        self.cache = cache
        self.batch_size = batch_size
    
    async def process_batch(self, contents: List[str]) -> List[List[float]]:
        """Process a batch of content for embeddings."""
        embeddings = []
        uncached_contents = []
        uncached_indices = []
        
        # Check cache for each content
        for i, content in enumerate(contents):
            cached_embedding = await self.cache.get_embedding(content)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_contents.append(content)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached content
        if uncached_contents:
            logger.debug(f"Generating {len(uncached_contents)} new embeddings")
            
            new_embeddings = []
            for i in range(0, len(uncached_contents), self.batch_size):
                batch = uncached_contents[i:i + self.batch_size]
                batch_embeddings = self.embedding_function(batch)
                new_embeddings.extend(batch_embeddings)
            
            # Cache new embeddings and fill placeholders
            for i, (content, embedding) in enumerate(zip(uncached_contents, new_embeddings)):
                await self.cache.store_embedding(content, embedding)
                embeddings[uncached_indices[i]] = embedding
        
        return embeddings
```

### Incremental Indexing with File Watching

```python
# Optimized file watching with incremental updates
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import asyncio
from typing import Set, Callable, Awaitable
from pathlib import Path
import time

class CodeFileHandler(FileSystemEventHandler):
    """Handle file system events for code files."""
    
    def __init__(
        self,
        file_extensions: Set[str],
        on_file_change: Callable[[Path, str], Awaitable[None]],
        debounce_seconds: float = 1.0
    ) -> None:
        self.file_extensions = file_extensions
        self.on_file_change = on_file_change
        self.debounce_seconds = debounce_seconds
        self._pending_changes: Dict[Path, float] = {}
        self._debounce_task: Optional[asyncio.Task] = None
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "modified")
    
    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "created")
    
    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "deleted")
    
    def _queue_change(self, file_path: Path, change_type: str) -> None:
        """Queue file change with debouncing."""
        if file_path.suffix not in self.file_extensions:
            return
        
        # Ignore temporary and hidden files
        if file_path.name.startswith('.') or file_path.name.endswith('.tmp'):
            return
        
        current_time = time.time()
        self._pending_changes[file_path] = current_time
        
        # Schedule debounced processing
        if self._debounce_task:
            self._debounce_task.cancel()
        
        self._debounce_task = asyncio.create_task(
            self._process_pending_changes()
        )
    
    async def _process_pending_changes(self) -> None:
        """Process pending changes after debounce period."""
        await asyncio.sleep(self.debounce_seconds)
        
        current_time = time.time()
        changes_to_process = []
        
        for file_path, timestamp in list(self._pending_changes.items()):
            if current_time - timestamp >= self.debounce_seconds:
                changes_to_process.append(file_path)
                del self._pending_changes[file_path]
        
        # Process changes
        for file_path in changes_to_process:
            try:
                if file_path.exists():
                    await self.on_file_change(file_path, "modified")
                else:
                    await self.on_file_change(file_path, "deleted")
            except Exception as e:
                logger.error(f"Failed to process file change {file_path}: {e}")

class IncrementalIndexer:
    """Incremental indexing with file watching."""
    
    def __init__(
        self,
        project_root: Path,
        database: ChromaVectorDatabase,
        parser_registry: "ParserRegistry",
        file_extensions: Set[str]
    ) -> None:
        self.project_root = project_root
        self.database = database
        self.parser_registry = parser_registry
        self.file_extensions = file_extensions
        self.observer: Optional[Observer] = None
        self._index_lock = asyncio.Lock()
    
    async def start_watching(self) -> None:
        """Start file system watching."""
        handler = CodeFileHandler(
            file_extensions=self.file_extensions,
            on_file_change=self._handle_file_change,
            debounce_seconds=1.0
        )
        
        self.observer = Observer()
        self.observer.schedule(
            handler,
            str(self.project_root),
            recursive=True
        )
        
        self.observer.start()
        logger.info(f"Started watching {self.project_root} for changes")
    
    def stop_watching(self) -> None:
        """Stop file system watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped file watching")
    
    @monitor_performance("incremental_index_file")
    async def _handle_file_change(self, file_path: Path, change_type: str) -> None:
        """Handle individual file changes."""
        async with self._index_lock:
            try:
                if change_type == "deleted":
                    await self._remove_file_from_index(file_path)
                else:
                    await self._reindex_file(file_path)
                
                logger.info(f"Processed {change_type} for {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to handle file change {file_path}: {e}")
    
    async def _reindex_file(self, file_path: Path) -> None:
        """Reindex a single file."""
        # Remove existing entries for this file
        await self._remove_file_from_index(file_path)
        
        # Parse and index the file
        parser = self.parser_registry.get_parser(file_path.suffix)
        if not parser:
            logger.warning(f"No parser available for {file_path}")
            return
        
        try:
            chunks = await parser.parse_file(file_path)
            if chunks:
                await self.database.add_code_chunks(chunks)
                logger.debug(f"Reindexed {len(chunks)} chunks from {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
    
    async def _remove_file_from_index(self, file_path: Path) -> None:
        """Remove all chunks for a file from the index."""
        try:
            # Query for all chunks from this file
            results = await self.database.collection.get(
                where={"file_path": str(file_path)},
                include=["metadatas"]
            )
            
            if results["ids"]:
                await self.database.collection.delete(ids=results["ids"])
                logger.debug(f"Removed {len(results['ids'])} chunks for {file_path}")
        
        except Exception as e:
            logger.warning(f"Failed to remove file from index {file_path}: {e}")
```

---

## Implementation Plan - Lightweight Deployment

### Phase 1: CLI Foundation (4-6 weeks)

**Goal**: Establish core CLI functionality with ChromaDB-based semantic search

**Key Components**:
1. **Project Initialization**: `mcp-vector-search init` creates `.mcp-vector-search/` directory with configuration, sets up ChromaDB database, initializes Tree-sitter parsers
2. **Indexing System**: Tree-sitter parsing with CodeBERT embeddings, incremental updates via file watching, Git-aware change detection
3. **Search Engine**: Semantic search with AST-aware chunking, context-sensitive ranking, multiple result formats
4. **Configuration Management**: Pydantic-based settings with environment variables, extensible language support

**Technical Deliverables**:
- Type-safe CLI with Typer and comprehensive help
- ChromaDB integration with optimized embedding caching
- Tree-sitter integration with language pack support
- CodeBERT embedding pipeline with batch processing
- File watching with intelligent debouncing and reindexing
- Comprehensive test suite with async support and benchmarks

### Phase 2: MCP Server Integration (2-3 weeks)

**Goal**: Seamless MCP protocol support maintaining CLI feature parity

**Key Components**:
1. **MCP Protocol Implementation**: Standards-compliant server using official Python SDK
2. **Tool Mapping**: Direct mapping from CLI commands to MCP tools
3. **Session Management**: Project context awareness across MCP calls
4. **Error Handling**: Structured error responses with proper MCP formatting

**Technical Deliverables**:
- MCP server with all CLI functionality exposed as tools
- Integration examples for Claude Desktop and other MCP clients
- Performance optimization for MCP request/response patterns
- Comprehensive documentation for MCP client configuration

### Phase 3: Advanced Features (3-4 weeks)

**Goal**: Performance optimization and advanced semantic capabilities

**Key Components**:
1. **Performance Optimization**: Query result caching, embedding optimization, parallel processing
2. **Advanced Search**: Intent detection, query rewriting, cross-language semantic understanding
3. **Integration Ecosystem**: IDE plugins via Language Server Protocol, Git hooks for automatic reindexing
4. **Analytics and Insights**: Search pattern analysis, codebase complexity metrics

---

## Success Metrics and Validation

### Performance Benchmarks

**Search Response Time**: <100ms for typical queries (ChromaDB local deployment advantage)  
**Indexing Speed**: <5s for 1000 files (Tree-sitter + batch embedding processing)  
**Memory Efficiency**: <100MB baseline + 1MB per 1000 indexed files (ChromaDB SQLite backend)  
**Precision Metrics**: >80% relevant results in top 5 (AST-aware ranking improvement)  
**Installation Size**: <50MB total package size (no external database requirements)

### Competitive Differentiation

**Zero-Config Deployment**: ChromaDB embedded database eliminates setup complexity  
**MCP-First Advantage**: First dedicated semantic search MCP server in growing ecosystem  
**Precision Focus**: AST-aware ranking addresses the "pages of irrelevant content" problem  
**Local-First Privacy**: Complete on-device processing with embedded vector database  
**Modern Python Standards**: Type-safe, async-first, production-ready codebase  
**CLI-First UX**: Immediate productivity without complex setup or cloud dependencies

### Validation Approach

**Developer Testing**: CLI workflows with real codebases across languages  
**Precision Evaluation**: Benchmark against CodeSearchNet dataset and real-world queries  
**Performance Profiling**: Memory and CPU usage under realistic workloads with ChromaDB  
**Integration Testing**: MCP protocol compliance and client compatibility  
**Installation Testing**: Zero-config deployment across platforms (Windows, macOS, Linux)

---

## Development Environment Setup

### Modern Python Toolchain

```toml
# pyproject.toml optimized for UV
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mcp-vector-search"
dynamic = ["version"]
description = "CLI-first semantic code search with MCP integration"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Code Generators",
]
keywords = ["semantic-search", "code-search", "mcp", "vector-database"]
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.4.20",
    "sentence-transformers>=2.2.2",
    "tree-sitter>=0.20.1",
    "tree-sitter-languages>=1.8.0",
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "watchdog>=3.0.0",
    "aiofiles>=23.0.0",
    "loguru>=0.7.0",
    "httpx>=0.25.0",
]

# UV-optimized dependency groups
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-benchmark>=4.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "hypothesis>=6.88.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "coverage[toml]>=7.3.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "sphinx-autodoc-typehints>=1.25.0",
]
mcp = [
    "mcp>=0.1.0",  # Official MCP Python SDK
]

[project.urls]
Homepage = "https://github.com/yourusername/mcp-vector-search"
Documentation = "https://mcp-vector-search.readthedocs.io"
Repository = "https://github.com/yourusername/mcp-vector-search"
"Bug Tracker" = "https://github.com/yourusername/mcp-vector-search/issues"

[project.scripts]
mcp-vector-search = "mcp_vector_search.cli.main:app"

[tool.hatch.version]
path = "src/mcp_vector_search/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/mcp_vector_search"]

# UV-specific configuration
[tool.uv]
# Faster resolution with pre-releases allowed for development
prerelease = "if-necessary-or-explicit"

# Lock file management
lock = true

# Development server configuration
dev-dependencies = [
    "pytest-xdist>=3.3.0",  # Parallel test execution
    "pytest-watch>=4.2.0",  # File watching for tests
]

# Performance optimizations
[tool.uv.sources]
# Pin critical dependencies for consistent builds
chromadb = { version = ">=0.4.20" }
sentence-transformers = { version = ">=2.2.2" }

# Development tools configuration
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?
```

### UV-Powered Development Workflow

```bash
# Project setup with UV (single command)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init mcp-vector-search --package
cd mcp-vector-search

# Development environment setup (ultra-fast)
uv sync --dev  # Creates venv and installs all dependencies in seconds
uv run pre-commit install  # Set up git hooks

# Daily development commands
uv run pytest                    # Run tests
uv run black src tests          # Format code
uv run ruff check src tests     # Lint code
uv run mypy src                 # Type check
uv run mcp-vector-search init   # Test CLI

# Add new dependencies (with automatic lock file updates)
uv add chromadb
uv add --dev pytest-benchmark
```

### Development Scripts

```python
#!/usr/bin/env python3
# scripts/dev-setup.py
"""UV-based development environment setup script."""

import subprocess
import sys
import shutil
from pathlib import Path

def run_uv_command(cmd: list[str], description: str) -> None:
    """Run a UV command with error handling."""
    print(f"🔧 {description}...")
    try:
        subprocess.run(["uv"] + cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ {description} completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        sys.exit(1)

def check_uv_installed() -> bool:
    """Check if UV is installed."""
    return shutil.which("uv") is not None

def main() -> None:
    """Set up development environment with UV."""
    print("🚀 Setting up mcp-vector-search development environment\n")
    
    # Check UV installation
    if not check_uv_installed():
        print("❌ UV not found. Install with:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    print("✅ UV found, proceeding with setup\n")
    
    # Sync dependencies (creates venv and installs everything)
    run_uv_command(
        ["sync", "--dev"],
        "Installing all dependencies with UV"
    )
    
    # Install pre-commit hooks
    run_uv_command(
        ["run", "pre-commit", "install"],
        "Setting up pre-commit hooks"
    )
    
    # Run initial code quality checks
    run_uv_command(
        ["run", "black", "--check", "src", "tests"],
        "Checking code formatting"
    )
    
    run_uv_command(
        ["run", "ruff", "check", "src", "tests"],
        "Running linter checks"
    )
    
    run_uv_command(
        ["run", "mypy", "src"],
        "Running type checking"
    )
    
    # Run tests
    run_uv_command(
        ["run", "pytest", "tests/", "-v"],
        "Running test suite"
    )
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("  • Run 'uv run mcp-vector-search init' to initialize a test project")
    print("  • Run 'uv run pytest' to run tests")
    print("  • Use 'uv add <package>' to add new dependencies")
    print("  • All commands prefixed with 'uv run' use the project venv automatically")

if __name__ == "__main__":
    main()
```

### UV Configuration Files

```toml
# uv.lock will be generated automatically
# This replaces Poetry.lock or requirements.txt

# .python-version (UV respects this for Python version)
3.11

# pyproject.toml additions for UV optimization
[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-benchmark>=4.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "hypothesis>=6.88.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "coverage[toml]>=7.3.0",
]

[tool.uv.sources]
# Pin to specific versions for reproducibility
chromadb = { version = ">=0.4.20" }
```

---

## Conclusion

This updated PRD leverages ChromaDB for lightweight local deployment while incorporating modern Python development best practices. The architecture eliminates external database dependencies, reducing installation complexity to a simple `pip install`. The comprehensive type safety, async-first design, and production-ready error handling ensure the codebase meets professional standards while maintaining the precision focus that differentiates this package from existing tools.

The ChromaDB integration provides excellent performance for local development while the modern Python standards ensure maintainability, testability, and extensibility. This approach creates a solid foundation for both CLI and MCP integration phases while addressing the market gaps identified in the competitive analysis.
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*" = ["B011", "B018"]

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
strict_equality = true
show_error_codes = true

[[tool.mypy.overrides]]
module = [
    "tree_sitter",
    "tree_sitter_languages",
    "watchdog.*",
    "chromadb.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=mcp_vector_search",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "benchmark: marks tests as benchmarks",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

### UV-Powered Development Workflow

```bash
# Project setup with UV (single command)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init mcp-vector-search --package
cd mcp-vector-search

# Development environment setup (ultra-fast)
uv sync --dev  # Creates venv and installs all dependencies in seconds
uv run pre-commit install  # Set up git hooks

# Daily development commands
uv run pytest                    # Run tests
uv run black src tests          # Format code
uv run ruff check src tests     # Lint code
uv run mypy src                 # Type check
uv run mcp-vector-search init   # Test CLI

# Add new dependencies (with automatic lock file updates)
uv add chromadb
uv add --dev pytest-benchmark
```

### Development Scripts

```python
#!/usr/bin/env python3
# scripts/dev-setup.py
"""UV-based development environment setup script."""

import subprocess
import sys
import shutil
from pathlib import Path

def run_uv_command(cmd: list[str], description: str) -> None:
    """Run a UV command with error handling."""
    print(f"🔧 {description}...")
    try:
        subprocess.run(["uv"] + cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ {description} completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        sys.exit(1)

def check_uv_installed() -> bool:
    """Check if UV is installed."""
    return shutil.which("uv") is not None

def main() -> None:
    """Set up development environment with UV."""
    print("🚀 Setting up mcp-vector-search development environment\n")
    
    # Check UV installation
    if not check_uv_installed():
        print("❌ UV not found. Install with:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    print("✅ UV found, proceeding with setup\n")
    
    # Sync dependencies (creates venv and installs everything)
    run_uv_command(
        ["sync", "--dev"],
        "Installing all dependencies with UV"
    )
    
    # Install pre-commit hooks
    run_uv_command(
        ["run", "pre-commit", "install"],
        "Setting up pre-commit hooks"
    )
    
    # Run initial code quality checks
    run_uv_command(
        ["run", "black", "--check", "src", "tests"],
        "Checking code formatting"
    )
    
    run_uv_command(
        ["run", "ruff", "check", "src", "tests"],
        "Running linter checks"
    )
    
    run_uv_command(
        ["run", "mypy", "src"],
        "Running type checking"
    )
    
    # Run tests
    run_uv_command(
        ["run", "pytest", "tests/", "-v"],
        "Running test suite"
    )
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("  • Run 'uv run mcp-vector-search init' to initialize a test project")
    print("  • Run 'uv run pytest' to run tests")
    print("  • Use 'uv add <package>' to add new dependencies")
    print("  • All commands prefixed with 'uv run' use the project venv automatically")

if __name__ == "__main__":
    main()
```

### UV Configuration Files

```toml
# uv.lock will be generated automatically
# This replaces Poetry.lock or requirements.txt

# .python-version (UV respects this for Python version)
3.11

# pyproject.toml additions for UV optimization
[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-benchmark>=4.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "hypothesis>=6.88.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "coverage[toml]>=7.3.0",
]

[tool.uv.sources]
# Pin to specific versions for reproducibility
chromadb = { version = ">=0.4.20" }
```

---

## Conclusion

This updated PRD leverages ChromaDB for lightweight local deployment while incorporating modern Python development best practices. The architecture eliminates external database dependencies, reducing installation complexity to a simple `pip install`. The comprehensive type safety, async-first design, and production-ready error handling ensure the codebase meets professional standards while maintaining the precision focus that differentiates this package from existing tools.

The ChromaDB integration provides excellent performance for local development while the modern Python standards ensure maintainability, testability, and extensibility. This approach creates a solid foundation for both CLI and MCP integration phases while addressing the market gaps identified in the competitive analysis.