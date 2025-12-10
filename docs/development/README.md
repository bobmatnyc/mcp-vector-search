# Development Documentation

Documentation for contributors, maintainers, and developers working on MCP Vector Search.

## 🚀 Sprint Planning (NEW - Structural Code Analysis Project)

**Active Project**: [Structural Code Analysis](../projects/structural-code-analysis.md)
**GitHub Project**: https://github.com/users/bobmatnyc/projects/13

| Document | Purpose | Audience |
|----------|---------|----------|
| **[Sprint Quickstart](./sprint-quickstart.md)** | Get started with issue #2 immediately | Developers starting work |
| **[Sprint Plan Summary](./sprint-plan-summary.md)** | Quick reference for sprint meetings | Team leads, PMs |
| **[Sprint Plan (Full)](./sprint-plan.md)** | Comprehensive sprint breakdown | Project managers |
| **[Sprint Board](./sprint-board.md)** | Visual tracking board | All team members |
| **[Dependency Graph](./dependency-graph.txt)** | Issue dependencies visualization | Architects, tech leads |

**Quick Navigation**:
- 👉 **Just starting?** [Sprint Quickstart](./sprint-quickstart.md)
- 👉 **Planning a sprint?** [Sprint Plan Summary](./sprint-plan-summary.md)
- 👉 **Tracking progress?** [Sprint Board](./sprint-board.md)

---

## 🎯 Quick Links for Developers

- **[Development Setup](setup.md)** - Get your development environment ready
- **[Contributing Guide](contributing.md)** - How to contribute to the project
- **[Architecture Guide](architecture.md)** - Technical architecture deep dive
- **[API Reference](api.md)** - Internal API documentation

## 📚 Documentation Sections

### Getting Started with Development

#### [Development Setup](setup.md)
Set up your development environment and workflow.

**Topics**: Prerequisites, installation, dev tools, testing, development commands

#### [Contributing Guide](contributing.md)
Learn how to contribute to MCP Vector Search.

**Topics**: Contribution workflow, code standards, pull requests, code review, community guidelines

### Technical Documentation

#### [Architecture Guide](architecture.md)
Comprehensive technical architecture documentation.

**Topics**: System design, layer architecture, module deep dive, design patterns, data flow

#### [API Reference](api.md)
Internal API documentation for core modules.

**Topics**: Core modules, parser API, database API, CLI API, utilities

#### [Testing Guide](testing.md)
Testing strategies, guidelines, and best practices.

**Topics**: Test structure, unit tests, integration tests, test coverage, testing tools, test data

#### [Code Quality](code-quality.md)
Code quality standards and tools.

**Topics**: Linting (ruff), formatting (black), type checking (mypy), pre-commit hooks, CI/CD

### Project Management

#### [Project Organization](project-organization.md)
File organization standards and conventions.

**Topics**: Directory structure, file placement rules, naming conventions, migration guide

#### [Versioning & Releases](versioning.md)
Version management and release process.

**Topics**: Semantic versioning, version workflow, release process, changelog management, publishing

## 🛠️ Developer Workflows

### I want to...

**Set up my development environment**
→ [Development Setup](setup.md)

**Make my first contribution**
→ [Contributing Guide](contributing.md)

**Understand the codebase**
→ [Architecture Guide](architecture.md)

**Use internal APIs**
→ [API Reference](api.md)

**Write tests**
→ [Testing Guide](testing.md)

**Ensure code quality**
→ [Code Quality](code-quality.md)

**Follow file organization**
→ [Project Organization](project-organization.md)

**Release a new version**
→ [Versioning & Releases](versioning.md)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│           CLI Layer (Typer)                 │
├─────────────────────────────────────────────┤
│         MCP Server (Protocol)               │
├─────────────────────────────────────────────┤
│        Core Engine (Business Logic)         │
├─────────────────────────────────────────────┤
│      Parser System (Language Support)       │
├─────────────────────────────────────────────┤
│     Database Layer (ChromaDB + Pooling)     │
├─────────────────────────────────────────────┤
│        Utilities (Config, Timing, etc.)     │
└─────────────────────────────────────────────┘
```

See [Architecture Guide](architecture.md) for details.

## 📏 Development Standards

### Code Standards
- **Python 3.11+**: Modern Python with type hints
- **Type Safety**: Full mypy coverage
- **Code Style**: Black formatter + Ruff linter
- **Testing**: Pytest with good coverage
- **Documentation**: Comprehensive docstrings

### Git Standards
- **Commits**: Conventional commits format
- **Branches**: Feature branches from main
- **PRs**: Required reviews before merge
- **CI/CD**: Automated testing and linting

See [Contributing Guide](contributing.md) for complete standards.

## 🧪 Testing

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── e2e/            # End-to-end tests
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test
uv run pytest tests/unit/test_search.py
```

See [Testing Guide](testing.md) for complete testing documentation.

## 🔧 Development Tools

### Essential Tools
- **uv** - Fast Python package manager
- **pytest** - Testing framework
- **ruff** - Fast Python linter
- **black** - Code formatter
- **mypy** - Static type checker
- **pre-commit** - Git hooks

### Development Commands
```bash
# Install dev dependencies
uv sync

# Run from source
./dev-mcp <command>

# Run tests
uv run pytest

# Lint code
uv run ruff check

# Format code
uv run black src/

# Type check
uv run mypy src/
```

See [Development Setup](setup.md) for details.

## 🔗 Related Documentation

- **[Architecture](../architecture/README.md)** - System architecture and design
- **[Reference](../reference/README.md)** - Technical reference
- **[Advanced Topics](../advanced/README.md)** - Performance and extensions

## 💬 Community

- **Discussions**: [GitHub Discussions](https://github.com/bobmatnyc/mcp-vector-search/discussions)
- **Issues**: [GitHub Issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
- **Pull Requests**: [Contributing Guide](contributing.md)

---

**[← Back to Documentation Index](../index.md)**
