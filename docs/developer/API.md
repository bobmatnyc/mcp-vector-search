# Internal API Reference

## 🎯 Overview

This document provides detailed API reference for MCP Vector Search internal modules. These APIs are intended for developers contributing to the project or building extensions.

> **Note**: This is internal API documentation. For user-facing CLI documentation, see [README.md](../../README.md).

---

## 🏗️ Core APIs

### SemanticIndexer

**Location**: `src/mcp_vector_search/core/indexer.py`

Main class responsible for indexing code files and generating vector embeddings.

```python
class SemanticIndexer:
    """Indexes code files for semantic search."""
    
    def __init__(
        self,
        database: VectorDatabase,
        embeddings: EmbeddingGenerator,
        parser_registry: ParserRegistry,
    ) -> None:
        """Initialize the indexer with required dependencies."""
    
    async def index_files(
        self,
        files: List[Path],
        show_progress: bool = True,
    ) -> IndexingResult:
        """Index multiple files concurrently.
        
        Args:
            files: List of file paths to index.
            show_progress: Whether to show progress bar.
            
        Returns:
            IndexingResult with statistics and any errors.
        """
    
    async def index_file(
        self,
        file_path: Path,
        force_reindex: bool = False,
    ) -> List[CodeChunk]:
        """Index a single file.
        
        Args:
            file_path: Path to the file to index.
            force_reindex: Whether to reindex even if unchanged.
            
        Returns:
            List of generated code chunks.
            
        Raises:
            IndexingError: If file cannot be indexed.
            ParseError: If file cannot be parsed.
        """
    
    def get_supported_extensions(self) -> Set[str]:
        """Get file extensions supported by registered parsers."""
```

### SemanticSearch

**Location**: `src/mcp_vector_search/core/search.py`

Handles semantic search queries and result ranking.

```python
class SemanticSearch:
    """Performs semantic search over indexed code."""
    
    def __init__(
        self,
        database: VectorDatabase,
        embeddings: EmbeddingGenerator,
    ) -> None:
        """Initialize search with database and embedding generator."""
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for code similar to query.
        
        Args:
            query: Search query string.
            limit: Maximum number of results.
            threshold: Minimum similarity score (0.0-1.0).
            filters: Optional metadata filters.
            
        Returns:
            List of search results sorted by similarity.
        """
    
    async def search_similar_code(
        self,
        code_chunk: CodeChunk,
        limit: int = 5,
    ) -> List[SearchResult]:
        """Find code similar to given chunk."""
    
    def explain_search(
        self,
        query: str,
        result: SearchResult,
    ) -> SearchExplanation:
        """Explain why a result matched the query."""
```

### VectorDatabase

**Location**: `src/mcp_vector_search/core/database.py`

Abstract interface for vector database operations.

```python
class VectorDatabase(ABC):
    """Abstract base class for vector databases."""
    
    @abstractmethod
    async def store_chunks(
        self,
        chunks: List[CodeChunk],
        collection_name: str = "default",
    ) -> None:
        """Store code chunks in the database."""
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
    
    @abstractmethod
    async def delete_chunks(
        self,
        chunk_ids: List[str],
    ) -> None:
        """Delete chunks by ID."""
    
    @abstractmethod
    async def get_stats(self) -> DatabaseStats:
        """Get database statistics."""

class ChromaDatabase(VectorDatabase):
    """ChromaDB implementation of vector database."""
    
    def __init__(
        self,
        db_path: Path,
        collection_name: str = "code_chunks",
    ) -> None:
        """Initialize ChromaDB instance."""
```

---

## 🔤 Parser APIs

### BaseParser

**Location**: `src/mcp_vector_search/parsers/base.py`

Abstract base class for all language parsers.

```python
class BaseParser(ABC):
    """Abstract base class for code parsers."""
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Language name (e.g., 'python', 'javascript')."""
    
    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Supported file extensions (e.g., ['.py', '.pyi'])."""
    
    @abstractmethod
    def parse(self, content: str, file_path: Optional[Path] = None) -> List[CodeChunk]:
        """Parse code content into chunks.
        
        Args:
            content: Source code content.
            file_path: Optional file path for context.
            
        Returns:
            List of parsed code chunks.
            
        Raises:
            ParseError: If content cannot be parsed.
        """
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if parser can handle the given file."""
        return file_path.suffix.lower() in self.file_extensions
    
    def extract_metadata(
        self,
        content: str,
        file_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Extract metadata from code content."""
```

### PythonParser

**Location**: `src/mcp_vector_search/parsers/python.py`

Python-specific parser implementation.

```python
class PythonParser(BaseParser):
    """Parser for Python code using AST."""
    
    @property
    def language(self) -> str:
        return "python"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".py", ".pyi", ".pyw"]
    
    def parse(self, content: str, file_path: Optional[Path] = None) -> List[CodeChunk]:
        """Parse Python code using AST."""
    
    def extract_functions(self, tree: ast.AST) -> List[FunctionInfo]:
        """Extract function definitions from AST."""
    
    def extract_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Extract class definitions from AST."""
    
    def extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract import statements from AST."""
```

### ParserRegistry

**Location**: `src/mcp_vector_search/parsers/registry.py`

Registry for managing language parsers.

```python
class ParserRegistry:
    """Registry for code parsers."""
    
    def __init__(self) -> None:
        """Initialize empty registry."""
    
    def register(self, parser: BaseParser) -> None:
        """Register a parser for its supported languages."""
    
    def get_parser(self, language: str) -> Optional[BaseParser]:
        """Get parser for specific language."""
    
    def get_parser_for_file(self, file_path: Path) -> Optional[BaseParser]:
        """Get appropriate parser for file based on extension."""
    
    def get_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
    
    def list_languages(self) -> List[str]:
        """List all supported languages."""

# Global registry instance
parser_registry = ParserRegistry()
```

---

## 📊 Data Models

### CodeChunk

**Location**: `src/mcp_vector_search/core/models.py`

Represents a chunk of code with metadata.

```python
@dataclass
class CodeChunk:
    """Represents a chunk of code for indexing."""
    
    id: str                          # Unique identifier
    content: str                     # Code content
    file_path: Path                  # Source file path
    language: str                    # Programming language
    chunk_type: ChunkType           # Function, class, module, etc.
    start_line: int                 # Starting line number
    end_line: int                   # Ending line number
    metadata: Dict[str, Any]        # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChunk":
        """Create from dictionary."""
    
    @property
    def line_count(self) -> int:
        """Number of lines in chunk."""
    
    @property
    def summary(self) -> str:
        """Brief summary of chunk content."""
```

### SearchResult

**Location**: `src/mcp_vector_search/core/models.py`

Represents a search result with similarity score.

```python
@dataclass
class SearchResult:
    """Represents a search result."""
    
    chunk: CodeChunk                # Matching code chunk
    similarity_score: float         # Similarity score (0.0-1.0)
    rank: int                       # Result ranking
    explanation: Optional[str]      # Why this matched
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
    
    @property
    def is_high_confidence(self) -> bool:
        """Whether this is a high-confidence match."""
        return self.similarity_score >= 0.8
```

---

## ⚙️ Configuration APIs

### Settings

**Location**: `src/mcp_vector_search/config/settings.py`

Pydantic-based configuration management.

```python
class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    path: Path = Path(".mcp-vector-search/db")
    collection_name: str = "code_chunks"
    
class EmbeddingSettings(BaseSettings):
    """Embedding configuration."""
    
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_embeddings: bool = True

class IndexingSettings(BaseSettings):
    """Indexing configuration."""
    
    chunk_size: int = 1000
    overlap: int = 200
    exclude_patterns: List[str] = [
        "*.pyc", "*.pyo", "__pycache__/",
        "node_modules/", ".git/", ".venv/"
    ]
    
class Settings(BaseSettings):
    """Main application settings."""
    
    database: DatabaseSettings = DatabaseSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    indexing: IndexingSettings = IndexingSettings()
    
    class Config:
        env_prefix = "MCP_"
        env_file = ".env"

# Global settings instance
settings = Settings()
```

---

## 🔧 Utility APIs

### FileUtils

**Location**: `src/mcp_vector_search/core/utils.py`

File system utilities.

```python
class FileUtils:
    """File system utilities."""
    
    @staticmethod
    async def read_file_async(file_path: Path) -> str:
        """Read file content asynchronously."""
    
    @staticmethod
    def detect_language(file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
    
    @staticmethod
    def should_exclude(file_path: Path, patterns: List[str]) -> bool:
        """Check if file should be excluded based on patterns."""
    
    @staticmethod
    def find_project_root(start_path: Path) -> Optional[Path]:
        """Find project root directory."""
```

### EmbeddingGenerator

**Location**: `src/mcp_vector_search/core/embeddings.py`

Text embedding generation.

```python
class EmbeddingGenerator:
    """Generates text embeddings for semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize with specified model."""
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding vector dimension."""
    
    def preprocess_code(self, code: str, language: str) -> str:
        """Preprocess code for better embeddings."""
```

---

## 🚨 Exception Hierarchy

**Location**: `src/mcp_vector_search/core/exceptions.py`

Custom exception classes.

```python
class MCPVectorSearchError(Exception):
    """Base exception for all MCP Vector Search errors."""

class ConfigurationError(MCPVectorSearchError):
    """Configuration-related errors."""

class IndexingError(MCPVectorSearchError):
    """Indexing operation errors."""

class ParseError(MCPVectorSearchError):
    """Code parsing errors."""

class DatabaseError(MCPVectorSearchError):
    """Vector database errors."""

class SearchError(MCPVectorSearchError):
    """Search operation errors."""

class FileSystemError(MCPVectorSearchError):
    """File system operation errors."""
```

---

## 🔗 Extension Points

### Adding Custom Parsers

```python
# 1. Create parser class
class MyLanguageParser(BaseParser):
    @property
    def language(self) -> str:
        return "mylang"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".ml", ".mylang"]
    
    def parse(self, content: str, file_path: Optional[Path] = None) -> List[CodeChunk]:
        # Implementation here
        pass

# 2. Register parser
from mcp_vector_search.parsers.registry import parser_registry
parser_registry.register(MyLanguageParser())
```

### Adding Custom Database Backends

```python
# 1. Implement VectorDatabase interface
class MyVectorDatabase(VectorDatabase):
    async def store_chunks(self, chunks: List[CodeChunk]) -> None:
        # Implementation here
        pass
    
    async def search_similar(self, query_embedding: List[float], **kwargs) -> List[SearchResult]:
        # Implementation here
        pass

# 2. Use in indexer
database = MyVectorDatabase()
indexer = SemanticIndexer(database, embeddings, parser_registry)
```

---

## 📚 Usage Examples

### Basic Indexing

```python
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.database import ChromaDatabase
from mcp_vector_search.core.embeddings import EmbeddingGenerator
from mcp_vector_search.parsers.registry import parser_registry

# Initialize components
database = ChromaDatabase(Path(".mcp-vector-search/db"))
embeddings = EmbeddingGenerator()
indexer = SemanticIndexer(database, embeddings, parser_registry)

# Index files
files = [Path("src/main.py"), Path("src/utils.py")]
result = await indexer.index_files(files)
print(f"Indexed {result.chunks_created} chunks")
```

### Basic Search

```python
from mcp_vector_search.core.search import SemanticSearch

# Initialize search
search = SemanticSearch(database, embeddings)

# Perform search
results = await search.search("authentication logic", limit=5)
for result in results:
    print(f"Score: {result.similarity_score:.3f}")
    print(f"File: {result.chunk.file_path}")
    print(f"Content: {result.chunk.content[:100]}...")
```
