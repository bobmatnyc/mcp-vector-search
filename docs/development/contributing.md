# Contributing to MCP Vector Search

## 🎯 Welcome Contributors!

Thank you for your interest in contributing to MCP Vector Search! This guide will help you get started with contributing to the project.

---

## 🚀 Quick Start

### 1. Fork & Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/mcp-vector-search.git
cd mcp-vector-search
```

### 2. Set Up Development Environment
```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and development tools
uv sync

# Install in development mode
uv pip install -e .

# Install pre-commit hooks
uv run pre-commit install
```

### 3. Verify Setup
```bash
# Run the development test suite
./scripts/dev-test.sh

# Test CLI functionality
uv run mcp-vector-search version
```

---

## 🔄 Development Workflow

### Branch Strategy
- **`main`** - Stable release branch
- **`develop`** - Development integration branch (if needed)
- **`feature/your-feature`** - Feature branches
- **`fix/issue-description`** - Bug fix branches
- **`docs/topic`** - Documentation updates

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/semantic-improvements
   ```

2. **Make Changes**
   - Follow the [code style guidelines](LINTING.md)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Changes**
   ```bash
   # Run development tests
   ./scripts/dev-test.sh
   
   # Test local deployment
   ./scripts/deploy-test.sh
   ```

4. **Commit Changes**
   ```bash
   # Pre-commit hooks will run automatically
   git add .
   git commit -m "feat: improve semantic search accuracy"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/semantic-improvements
   # Create pull request on GitHub
   ```

---

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples
```bash
# Feature addition
git commit -m "feat(parsers): add Go language support"

# Bug fix
git commit -m "fix(search): handle empty query gracefully"

# Documentation
git commit -m "docs: update API reference for new search options"

# Breaking change
git commit -m "feat!: change search API to async/await"
```

---

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
├── fixtures/               # Test data
└── conftest.py            # Pytest configuration
```

### Writing Tests

#### Unit Tests
```python
# tests/unit/test_parsers.py
import pytest
from mcp_vector_search.parsers.python import PythonParser

def test_python_parser_extracts_functions():
    """Test that Python parser correctly extracts functions."""
    parser = PythonParser()
    code = """
def hello_world():
    print("Hello, world!")
    """
    
    chunks = parser.parse(code)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.FUNCTION
    assert "hello_world" in chunks[0].content
```

#### Integration Tests
```python
# tests/integration/test_indexing.py
import pytest
from pathlib import Path
from mcp_vector_search.core.indexer import SemanticIndexer

@pytest.mark.asyncio
async def test_full_indexing_workflow(tmp_path, mock_database):
    """Test complete indexing workflow."""
    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("def test(): pass")
    
    # Index file
    indexer = SemanticIndexer(mock_database, mock_embeddings, parser_registry)
    result = await indexer.index_files([test_file])
    
    assert result.chunks_created > 0
    assert result.errors == []
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/mcp_vector_search

# Run specific test file
uv run pytest tests/unit/test_parsers.py

# Run tests matching pattern
uv run pytest -k "test_python"
```

---

## 📚 Documentation Guidelines

### Types of Documentation

1. **Code Documentation**
   - Docstrings for all public functions/classes
   - Type hints for all parameters and returns
   - Inline comments for complex logic

2. **User Documentation**
   - README.md updates for new features
   - CLI help text updates
   - Usage examples

3. **Developer Documentation**
   - API documentation for new modules
   - Architecture decisions
   - Migration guides for breaking changes

### Docstring Style
```python
def search_similar(
    query: str,
    limit: int = 10,
    threshold: float = 0.7,
) -> List[SearchResult]:
    """Search for code similar to the given query.
    
    This function performs semantic search using vector embeddings
    to find code chunks that are semantically similar to the query.
    
    Args:
        query: The search query string. Can be natural language
            or code snippets.
        limit: Maximum number of results to return. Must be > 0.
        threshold: Minimum similarity score (0.0 to 1.0). Results
            below this threshold will be filtered out.
    
    Returns:
        List of SearchResult objects sorted by similarity score
        in descending order.
    
    Raises:
        ValueError: If limit <= 0 or threshold not in [0.0, 1.0].
        DatabaseError: If vector database is unavailable.
    
    Example:
        >>> results = await search_similar("authentication logic")
        >>> for result in results:
        ...     print(f"{result.similarity_score:.3f}: {result.chunk.file_path}")
    """
```

---

## 🎨 Code Style Guidelines

### Python Style
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Maximum line length: 88 characters

### Type Hints
```python
# Good: Clear type hints
async def process_files(
    files: List[Path],
    parser: BaseParser,
    batch_size: int = 10,
) -> Dict[str, List[CodeChunk]]:
    """Process files with proper typing."""

# Bad: No type hints
async def process_files(files, parser, batch_size=10):
    """Process files without typing."""
```

### Error Handling
```python
# Good: Specific exceptions with context
try:
    result = await dangerous_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise ProcessingError("Cannot complete operation") from e

# Bad: Bare except
try:
    result = dangerous_operation()
except:
    pass
```

### Async/Await
```python
# Good: Proper async usage
async def index_files(self, files: List[Path]) -> IndexingResult:
    tasks = [self._index_file(file) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return self._process_results(results)

# Bad: Blocking in async function
async def index_files(self, files: List[Path]) -> IndexingResult:
    results = []
    for file in files:  # Sequential, not concurrent
        result = await self._index_file(file)
        results.append(result)
    return results
```

---

## 🐛 Bug Reports

### Before Reporting
1. Check existing [issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
2. Try the latest version
3. Reproduce with minimal example

### Bug Report Template
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.11.5]
- mcp-vector-search: [e.g., 0.0.3]

**Additional Context**
- Error logs
- Sample code/files
- Screenshots if applicable
```

---

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe your use case and why this feature would be valuable.

**Proposed Solution**
If you have ideas for implementation, describe them here.

**Alternatives Considered**
Any alternative solutions or workarounds you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

---

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- GitHub contributors list
- Release notes
- Documentation credits

### Types of Contributions
- Code contributions
- Documentation improvements
- Bug reports and testing
- Feature suggestions
- Community support

---

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussion
- **Pull Request Reviews**: Code review and feedback

### Response Times
- **Bug reports**: Within 48 hours
- **Feature requests**: Within 1 week
- **Pull requests**: Within 3-5 days

---

## 📋 Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (run `./scripts/dev-test.sh`)
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] PR description explains changes clearly
- [ ] Breaking changes are documented

### PR Template
```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

---

## 🎉 Thank You!

Your contributions make MCP Vector Search better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other users, every contribution is valuable and appreciated!

For questions about contributing, feel free to open a [discussion](https://github.com/bobmatnyc/mcp-vector-search/discussions) or reach out through GitHub issues.
