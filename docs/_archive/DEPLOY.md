# Deployment Guide

## 📦 Installation Methods

### PyPI Installation (Recommended)
```bash
# Install latest stable version
pip install mcp-vector-search

# Install specific version
pip install mcp-vector-search==0.0.3

# Upgrade to latest
pip install mcp-vector-search --upgrade
```

### UV Package Manager
```bash
# Add to project
uv add mcp-vector-search

# Install globally
uv tool install mcp-vector-search
```

### From Source
```bash
# Clone repository
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search

# Install with UV
uv sync && uv pip install -e .

# Or with pip
pip install -e .
```

---

## 🖥️ System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Memory**: 512MB RAM
- **Storage**: 100MB free space
- **OS**: macOS, Linux, Windows

### Recommended Requirements
- **Python**: 3.12+
- **Memory**: 2GB RAM (for large codebases)
- **Storage**: 1GB free space
- **CPU**: Multi-core for faster indexing

### Dependencies
- **ChromaDB**: Vector database (auto-installed)
- **Sentence Transformers**: Embeddings (auto-installed)
- **Tree-sitter**: Code parsing (auto-installed)
- **Rich**: Terminal output (auto-installed)

---

## 🚀 Quick Start Deployment

### 1. Install Package
```bash
pip install mcp-vector-search
```

### 2. Verify Installation
```bash
mcp-vector-search version
mcp-vector-search --help
```

### 3. Initialize Project
```bash
cd /path/to/your/project
mcp-vector-search init
```

### 4. Index Codebase
```bash
mcp-vector-search index
```

### 5. Start Searching
```bash
mcp-vector-search search "authentication logic"
```

---

## 🏢 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install mcp-vector-search
RUN pip install mcp-vector-search

# Set working directory
WORKDIR /workspace

# Copy your codebase
COPY . .

# Initialize and index
RUN mcp-vector-search init && mcp-vector-search index

# Default command
CMD ["mcp-vector-search", "search"]
```

### CI/CD Integration
```yaml
# .github/workflows/search-index.yml
name: Update Search Index
on:
  push:
    branches: [main]

jobs:
  update-index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install mcp-vector-search
        run: pip install mcp-vector-search
      
      - name: Update search index
        run: |
          mcp-vector-search init
          mcp-vector-search index
      
      - name: Cache index
        uses: actions/cache@v3
        with:
          path: .mcp-vector-search/
          key: search-index-${{ github.sha }}
```

### Server Deployment
```bash
# Install on server
pip install mcp-vector-search

# Set up systemd service (optional)
sudo tee /etc/systemd/system/mcp-vector-search.service << EOF
[Unit]
Description=MCP Vector Search Watcher
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/usr/local/bin/mcp-vector-search watch
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable mcp-vector-search
sudo systemctl start mcp-vector-search
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Custom embedding model
export MCP_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Optional: Custom database path
export MCP_DB_PATH="/custom/path/.mcp-vector-search"

# Optional: Logging level
export MCP_LOG_LEVEL="INFO"
```

### Configuration File
```yaml
# .mcp-vector-search/config.yaml
database:
  path: ".mcp-vector-search/db"
  collection_name: "code_chunks"

embedding:
  model: "all-MiniLM-L6-v2"
  batch_size: 32

indexing:
  chunk_size: 1000
  overlap: 200
  exclude_patterns:
    - "*.pyc"
    - "node_modules/"
    - ".git/"

search:
  max_results: 20
  similarity_threshold: 0.7
```

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install mcp-vector-search -v
```

#### Permission Errors
```bash
# Install for user only
pip install mcp-vector-search --user

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mcp-vector-search
```

#### Memory Issues
```bash
# Reduce batch size for large codebases
mcp-vector-search config set indexing.batch_size 16

# Index incrementally
mcp-vector-search index --incremental
```

#### Tree-sitter Issues
```bash
# Force regex fallback if Tree-sitter fails
mcp-vector-search config set parsing.force_regex true

# Check parser status
mcp-vector-search doctor
```

### Performance Optimization

#### Large Codebases
```bash
# Use parallel processing
mcp-vector-search index --parallel

# Exclude unnecessary files
mcp-vector-search config set indexing.exclude_patterns '["*.min.js", "dist/", "build/"]'

# Adjust chunk size
mcp-vector-search config set indexing.chunk_size 2000
```

#### Memory Usage
```bash
# Monitor memory usage
mcp-vector-search status --memory

# Reduce embedding dimensions (if supported)
mcp-vector-search config set embedding.dimensions 384
```

---

## 📊 Monitoring

### Health Checks
```bash
# Check system status
mcp-vector-search doctor

# Verify database integrity
mcp-vector-search status --detailed

# Test search functionality
mcp-vector-search search "test query" --dry-run
```

### Logging
```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG
mcp-vector-search index

# Log to file
mcp-vector-search index 2>&1 | tee indexing.log
```

### Metrics
```bash
# Show indexing statistics
mcp-vector-search status

# Performance metrics
mcp-vector-search status --performance
```

---

## 🔄 Updates

### Upgrading
```bash
# Check current version
mcp-vector-search version

# Upgrade to latest
pip install mcp-vector-search --upgrade

# Verify upgrade
mcp-vector-search version
```

### Migration
```bash
# Backup existing index
cp -r .mcp-vector-search .mcp-vector-search.backup

# Re-index after major updates
mcp-vector-search index --rebuild
```

---

## 🆘 Support

### Getting Help
- **Documentation**: [CLAUDE.md](../CLAUDE.md)
- **Issues**: [GitHub Issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bobmatnyc/mcp-vector-search/discussions)

### Reporting Issues
Include the following information:
- Operating system and version
- Python version
- mcp-vector-search version
- Error messages and logs
- Steps to reproduce
