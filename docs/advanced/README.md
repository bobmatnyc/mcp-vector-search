# Advanced Topics

Advanced guides for power users, performance tuning, troubleshooting, and extending MCP Vector Search.

## 🎯 Advanced Guides

### Performance & Optimization

#### [Performance Tuning](performance-tuning.md)
Optimize MCP Vector Search for your specific use case.

**Topics**: Configuration tuning, indexing optimization, search optimization, memory management, database tuning

### Models & Configuration

#### [Embedding Models](embedding-models.md)
Choose and configure the right embedding model for your needs.

**Topics**: Available models, model selection criteria, code-specific models, custom models, model performance comparison

### Troubleshooting

#### [Troubleshooting Guide](troubleshooting.md)
Common issues, solutions, and debugging techniques.

**Topics**: Installation issues, indexing problems, search quality, performance issues, MCP integration, file watching

### Extensibility

#### [Extending MCP Vector Search](extending.md)
Add new languages, parsers, and customize behavior.

**Topics**: Adding language support, custom parsers, custom embeddings, plugin development, integration patterns

## 🔧 When to Use These Guides

### Performance Tuning
Use when:
- Search is slow for your codebase
- Indexing takes too long
- Memory usage is high
- You have specific performance requirements

**[→ Read Performance Tuning Guide](performance-tuning.md)**

### Embedding Models
Use when:
- Default search results aren't accurate enough
- You work with specialized code (domain-specific)
- You want to optimize for speed or quality
- You need multilingual support

**[→ Read Embedding Models Guide](embedding-models.md)**

### Troubleshooting
Use when:
- Something isn't working as expected
- You encounter errors
- Search results are poor quality
- Integration isn't working

**[→ Read Troubleshooting Guide](troubleshooting.md)**

### Extending
Use when:
- You want to add a new programming language
- You need custom parsing logic
- You want to integrate with other tools
- You want to contribute new features

**[→ Read Extending Guide](extending.md)**

## 💡 Advanced Topics by Use Case

### Large Codebases (10,000+ files)
1. [Performance Tuning](performance-tuning.md) - Optimize indexing and search
2. [Embedding Models](embedding-models.md) - Choose efficient models
3. Enable connection pooling and caching

### Specialized Domains
1. [Embedding Models](embedding-models.md) - Use domain-specific models
2. [Performance Tuning](performance-tuning.md) - Tune similarity thresholds
3. Consider custom training for embeddings

### Multi-Language Projects
1. [Supported Languages](../reference/supported-languages.md) - Check language support
2. [Extending](extending.md) - Add missing languages
3. [Configuration](../getting-started/configuration.md) - Configure file extensions

### CI/CD Integration
1. [Performance Tuning](performance-tuning.md) - Optimize for automation
2. [Troubleshooting](troubleshooting.md) - Debug CI/CD issues
3. [Development Setup](../development/setup.md) - Automated testing

### Custom Integrations
1. [Extending](extending.md) - Integration patterns
2. [API Reference](../development/api.md) - Use internal APIs
3. [MCP Integration](../guides/mcp-integration.md) - MCP protocol

## 🔬 Advanced Features

### Connection Pooling
Automatic 13.6% performance improvement with zero configuration.

```python
from mcp_vector_search.core.database import PooledChromaVectorDatabase

database = PooledChromaVectorDatabase(
    max_connections=10,
    min_connections=2,
    max_idle_time=300.0
)
```

### Semi-Automatic Reindexing
Multiple strategies without daemon processes:
- Search-triggered checks
- Git hooks
- Scheduled tasks
- Manual checks
- Periodic checker

See [Performance Tuning](performance-tuning.md) for details.

### Custom Embedding Models
Use specialized models for better results:

```bash
mcp-vector-search config set embedding_model microsoft/codebert-base
```

See [Embedding Models](embedding-models.md) for options.

### Advanced Search Techniques
- Similarity search (find similar code)
- Context-aware search
- Filtered search by file type
- Threshold tuning

See [Searching Guide](../guides/searching.md) for techniques.

## 🔗 Related Documentation

- **[Guides](../guides/README.md)** - User guides for common tasks
- **[Reference](../reference/README.md)** - Technical reference
- **[Development](../development/README.md)** - For contributors
- **[Architecture](../architecture/README.md)** - System architecture

## 🆘 Need More Help?

- **Questions**: [GitHub Discussions](https://github.com/bobmatnyc/mcp-vector-search/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/bobmatnyc/mcp-vector-search/discussions)

---

**[← Back to Documentation Index](../index.md)**
