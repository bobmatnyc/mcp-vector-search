# Architecture Documentation

System architecture, design decisions, and technical deep dives for MCP Vector Search.

## 🏗️ Architecture Overview

MCP Vector Search follows a **layered, async-first architecture** designed for:
- **Performance**: Fast indexing and sub-second search
- **Extensibility**: Easy to add new languages and features
- **Maintainability**: Clear separation of concerns
- **Scalability**: Efficient resource usage

## 📚 Architecture Documentation

### Core Architecture

#### [Architecture Overview](overview.md)
High-level system design and architecture principles.

**Topics**: System layers, design philosophy, component interaction, technology stack

### Workflows

#### [Indexing Workflow](indexing-workflow.md)
Deep dive into how code indexing works.

**Topics**: File discovery, parsing, chunking, embedding generation, vector storage, incremental updates

### Performance

#### [Performance Architecture](performance.md)
Performance optimizations and architectural decisions.

**Topics**: Connection pooling, caching strategies, async operations, embedding optimization, database tuning

### Design Decisions

#### [Design Decisions](design-decisions.md)
Architecture Decision Records (ADRs) and key design choices.

**Topics**: Technology choices, design patterns, trade-offs, evolution of architecture

## 🎯 Architecture Layers

### Layer 1: CLI Interface
**Location**: `src/mcp_vector_search/cli/`

User interaction and command routing using Typer framework.

**Components**:
- Command routing and validation
- Rich terminal output
- Error handling and help text
- "Did you mean" suggestions

### Layer 2: MCP Server
**Location**: `src/mcp_vector_search/mcp/`

Model Context Protocol server for AI tool integration.

**Components**:
- MCP protocol implementation
- Tool definitions and handlers
- Context management
- AI-optimized responses

### Layer 3: Core Engine
**Location**: `src/mcp_vector_search/core/`

Business logic and semantic processing.

**Components**:
- Semantic indexer (code chunking)
- Search engine (vector similarity)
- Project manager
- File watcher
- Embedding generator
- Database abstraction

### Layer 4: Parser System
**Location**: `src/mcp_vector_search/parsers/`

Language-specific code parsing and analysis.

**Components**:
- Parser registry (plugin system)
- Base parser interface
- Language-specific parsers (Python, JavaScript, TypeScript, etc.)
- AST analysis
- Fallback regex parsing

### Layer 5: Database Layer
**Location**: `src/mcp_vector_search/core/database.py`

Vector database abstraction with connection pooling.

**Components**:
- ChromaDB integration
- Connection pooling (13.6% performance boost)
- Query optimization
- Metadata management

### Layer 6: Utilities
**Location**: `src/mcp_vector_search/config/`, utilities

Configuration, timing, and helper functions.

**Components**:
- Configuration management
- Timing utilities
- Logging and diagnostics
- File utilities

## 🔄 Data Flow

### Indexing Flow
```
User Command
    ↓
CLI Layer (parse args, validate)
    ↓
Project Manager (discover files)
    ↓
Parser Registry (select parser)
    ↓
Language Parser (extract code chunks)
    ↓
Semantic Indexer (generate embeddings)
    ↓
Vector Database (store vectors + metadata)
```

### Search Flow
```
User Query
    ↓
CLI Layer (parse query, options)
    ↓
Search Engine (generate query embedding)
    ↓
Vector Database (similarity search)
    ↓
Search Engine (rank and filter results)
    ↓
CLI Layer (format and display)
```

### File Watching Flow
```
File System Event
    ↓
File Watcher (detect changes)
    ↓
Change Analysis (determine action)
    ↓
Semantic Indexer (update/delete chunks)
    ↓
Vector Database (apply changes)
```

## 🎨 Design Patterns

### Registry Pattern
Used for parser system - allows dynamic registration of language parsers.

### Dependency Injection
Core components receive dependencies via constructor for testability.

### Async/Await
Async operations throughout for non-blocking I/O.

### Connection Pooling
Database connection reuse for performance.

### Strategy Pattern
Different search strategies (semantic, similar, contextual).

## 🔧 Technology Stack

### Core Technologies
- **Python 3.11+**: Modern Python with async support
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings
- **Tree-sitter**: AST parsing
- **Typer**: CLI framework
- **Rich**: Terminal formatting

### Supporting Libraries
- **Pydantic**: Data validation
- **Watchdog**: File system monitoring
- **aiofiles**: Async file I/O
- **pytest**: Testing framework

## 📊 Performance Characteristics

- **Indexing Speed**: ~1000 files/minute (typical Python project)
- **Search Latency**: <100ms for most queries
- **Memory Usage**: ~50MB baseline + ~1MB per 1000 code chunks
- **Storage**: ~1KB per code chunk (compressed embeddings)

See [Performance Architecture](performance.md) for details.

## 🔗 Related Documentation

- **[Development](../development/README.md)** - Development guides and API reference
- **[Reference](../reference/README.md)** - Technical reference
- **[Advanced Topics](../advanced/README.md)** - Performance tuning

---

**[← Back to Documentation Index](../index.md)**
