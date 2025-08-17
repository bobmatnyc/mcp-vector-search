# Project Structure

## 📁 Directory Overview

```
mcp-vector-search/
├── src/mcp_vector_search/          # Main package source
│   ├── __init__.py                 # Package initialization
│   ├── py.typed                    # Type checking marker
│   ├── cli/                        # Command-line interface
│   ├── core/                       # Core functionality
│   ├── parsers/                    # Language parsers
│   ├── config/                     # Configuration management
│   └── mcp/                        # MCP integration (future)
├── tests/                          # Test suite
├── scripts/                        # Development scripts
├── docs/                           # Documentation
├── pyproject.toml                  # Project configuration
├── uv.lock                         # Dependency lock file
├── README.md                       # Main documentation
├── CLAUDE.md                       # Documentation index
├── DEVELOPMENT.md                  # Development workflow
├── LICENSE                         # MIT license
└── .gitignore                      # Git ignore rules
```

---

## 🏗️ Architecture Layers

### Layer 1: CLI Interface
**Purpose**: User interaction and command routing
**Location**: `src/mcp_vector_search/cli/`

```
cli/
├── __init__.py                     # CLI package
├── main.py                         # Entry point & Typer app
├── output.py                       # Rich terminal formatting
└── commands/                       # Command implementations
    ├── __init__.py
    ├── init.py                     # Project initialization
    ├── index.py                    # Codebase indexing
    ├── search.py                   # Semantic search
    ├── watch.py                    # File watching
    ├── status.py                   # Project statistics
    └── config.py                   # Configuration management
```

### Layer 2: Core Engine
**Purpose**: Business logic and algorithms
**Location**: `src/mcp_vector_search/core/`

```
core/
├── __init__.py                     # Core package
├── models.py                       # Data models & types
├── exceptions.py                   # Custom exceptions
├── project.py                      # Project management
├── indexer.py                      # Code indexing logic
├── search.py                       # Search algorithms
├── database.py                     # Vector DB abstraction
├── embeddings.py                   # Text embedding generation
└── watcher.py                      # File system monitoring
```

### Layer 3: Language Support
**Purpose**: Code parsing and analysis
**Location**: `src/mcp_vector_search/parsers/`

```
parsers/
├── __init__.py                     # Parser package
├── base.py                         # Abstract parser interface
├── registry.py                     # Parser registration
├── python.py                       # Python AST parsing
├── javascript.py                   # JavaScript/TypeScript parsing
└── [future languages]              # Go, Rust, Java, etc.
```

### Layer 4: Configuration
**Purpose**: Settings and defaults
**Location**: `src/mcp_vector_search/config/`

```
config/
├── __init__.py                     # Config package
├── settings.py                     # Pydantic settings
└── defaults.py                     # Default configurations
```

---

## 🔄 Data Flow

### Indexing Flow
```
1. CLI Command (index.py)
   ↓
2. Project Manager (project.py)
   ↓
3. File Discovery & Filtering
   ↓
4. Language Detection
   ↓
5. Parser Selection (registry.py)
   ↓
6. Code Parsing (python.py, javascript.py)
   ↓
7. Chunk Generation (indexer.py)
   ↓
8. Embedding Generation (embeddings.py)
   ↓
9. Vector Storage (database.py)
```

### Search Flow
```
1. CLI Command (search.py)
   ↓
2. Query Processing
   ↓
3. Embedding Generation (embeddings.py)
   ↓
4. Vector Search (database.py)
   ↓
5. Result Ranking (search.py)
   ↓
6. Output Formatting (output.py)
```

### Watch Flow
```
1. CLI Command (watch.py)
   ↓
2. File System Monitor (watcher.py)
   ↓
3. Change Detection
   ↓
4. Incremental Update (indexer.py)
   ↓
5. Database Update (database.py)
```

---

## 📦 Module Dependencies

### Core Dependencies
```python
# External packages
chromadb              # Vector database
sentence-transformers # Text embeddings
tree-sitter          # Code parsing
tree-sitter-languages # Language grammars
typer                # CLI framework
rich                 # Terminal formatting
pydantic             # Data validation
watchdog             # File monitoring
aiofiles             # Async file operations
```

### Internal Dependencies
```python
# Module relationships
cli.commands → core.* → parsers.*
core.indexer → parsers.registry
core.search → core.database
core.watcher → core.indexer
config.settings ← all modules
```

---

## 🎯 Key Design Patterns

### 1. Abstract Base Classes
```python
# parsers/base.py
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> List[CodeChunk]:
        pass
```

### 2. Registry Pattern
```python
# parsers/registry.py
class ParserRegistry:
    def register(self, language: str, parser: BaseParser):
        self._parsers[language] = parser
    
    def get_parser(self, language: str) -> BaseParser:
        return self._parsers.get(language)
```

### 3. Dependency Injection
```python
# core/indexer.py
class SemanticIndexer:
    def __init__(
        self,
        database: VectorDatabase,
        embeddings: EmbeddingGenerator,
        parser_registry: ParserRegistry
    ):
        self.database = database
        self.embeddings = embeddings
        self.parsers = parser_registry
```

### 4. Async/Await Pattern
```python
# core/indexer.py
async def index_files(self, files: List[Path]) -> None:
    tasks = [self._index_file(file) for file in files]
    await asyncio.gather(*tasks)
```

---

## 🔧 Extension Points

### Adding New Languages
1. Create parser in `parsers/new_language.py`
2. Inherit from `BaseParser`
3. Register in `parsers/__init__.py`
4. Add file extension mapping

### Adding New Commands
1. Create command in `cli/commands/new_command.py`
2. Use Typer decorators
3. Import in `cli/main.py`
4. Add to main app

### Adding New Database Backends
1. Create implementation in `core/new_database.py`
2. Inherit from `VectorDatabase`
3. Implement abstract methods
4. Add configuration options

---

## 📊 File Size Guidelines

### Small Files (< 100 lines)
- Configuration files
- Simple data models
- Utility functions

### Medium Files (100-500 lines)
- Command implementations
- Parser implementations
- Core algorithms

### Large Files (500+ lines)
- Main entry points
- Complex algorithms
- Test files

---

## 🧪 Testing Structure

```
tests/
├── __init__.py
├── conftest.py                     # Pytest configuration
├── unit/                           # Unit tests
│   ├── test_parsers.py
│   ├── test_indexer.py
│   ├── test_search.py
│   └── test_database.py
├── integration/                    # Integration tests
│   ├── test_cli.py
│   ├── test_workflow.py
│   └── test_end_to_end.py
└── fixtures/                       # Test data
    ├── sample_code/
    └── expected_results/
```

---

## 📝 Configuration Files

### Project Configuration
- **`pyproject.toml`** - Python project metadata, dependencies, tools
- **`uv.lock`** - Exact dependency versions
- **`.pre-commit-config.yaml`** - Git hooks configuration

### Development Configuration
- **`.gitignore`** - Git ignore patterns
- **`scripts/`** - Development automation scripts

### Runtime Configuration
- **`.mcp-vector-search/config.yaml`** - User settings
- **`.mcp-vector-search/db/`** - Vector database storage

---

## 🔍 Code Organization Principles

### 1. Separation of Concerns
- CLI layer handles user interaction
- Core layer handles business logic
- Parser layer handles language-specific code

### 2. Single Responsibility
- Each module has one clear purpose
- Functions do one thing well
- Classes represent single concepts

### 3. Dependency Direction
- Dependencies flow inward (CLI → Core → Parsers)
- No circular dependencies
- Abstract interfaces for decoupling

### 4. Testability
- Pure functions where possible
- Dependency injection for testing
- Clear interfaces for mocking

---

## 📈 Future Structure Considerations

### Planned Additions
```
src/mcp_vector_search/
├── mcp/                            # MCP server implementation
│   ├── server.py
│   └── protocol.py
├── plugins/                        # Plugin system
│   ├── __init__.py
│   └── base.py
└── integrations/                   # IDE integrations
    ├── vscode/
    └── jetbrains/
```

### Scalability Considerations
- Plugin architecture for extensibility
- Microservice-ready design
- Database abstraction for scaling
- Async-first for performance

---

## 🔗 Related Documentation

- **[CLAUDE.md](../CLAUDE.md)** - Documentation index
- **[docs/developer/API.md](developer/API.md)** - Internal API reference
- **[docs/developer/CONTRIBUTING.md](developer/CONTRIBUTING.md)** - Contribution guidelines
- **[DEVELOPMENT.md](../DEVELOPMENT.md)** - Development workflow
