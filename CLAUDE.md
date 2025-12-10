# MCP Vector Search - AI Assistant Instructions

## Quick Reference

| Item | Value |
|------|-------|
| **Project** | CLI-first semantic code search with MCP integration |
| **Language** | Python 3.11+ |
| **Stack** | FastAPI, ChromaDB, Sentence Transformers, Tree-sitter |
| **Package Manager** | uv (dev), pip/PyPI (distribution) |

## Active Projects

| Project | Status | Docs |
|---------|--------|------|
| [Structural Code Analysis](docs/projects/structural-code-analysis.md) | Phase 1/5 | [Design](docs/research/mcp-vector-search-structural-analysis-design.md) |

**GitHub Project**: https://github.com/users/bobmatnyc/projects/13

## Root Directory Policy

**Allowed**: `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `LICENSE`, `pyproject.toml`, `Makefile`, `.gitignore`

**Prohibited**: Test files, temp files, data files, logs, build artifacts

## Directory Structure

```
src/mcp_vector_search/     # Source code
tests/{unit,integration,e2e,manual}/  # Tests
docs/{projects,research,development,guides,reference}/  # Documentation
scripts/                   # Build/utility scripts
examples/                  # Usage examples
.mcp-vector-search/        # Runtime data (gitignored)
```

## Commands

```bash
# Development
uv sync                    # Install deps
uv run pytest              # Run tests
uv run black . && uv run ruff check .  # Format & lint
uv run mypy src/           # Type check

# CLI
mcp-vector-search init     # Initialize
mcp-vector-search index    # Index codebase
mcp-vector-search search "query"  # Search
mcp-vector-search chat     # Interactive chat

# Release
make pre-publish           # Quality gate (required)
make release-pypi          # Publish to PyPI
```

## Code Standards

- **Format**: Black
- **Lint**: Ruff
- **Types**: Mypy (strict)
- **Security**: Bandit
- **Tests**: Pytest, 80% coverage minimum
- **Docs**: Docstrings + type hints required

## Architecture

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Indexer | `core/indexer.py` | Parse code, generate embeddings |
| Search | `core/search.py` | Semantic + hybrid search |
| MCP Server | `mcp/server.py` | Claude Desktop integration |
| CLI | `cli/` | Command-line interface |

### Design Principles

- **Async-first**: Use async/await for I/O
- **Type-safe**: Full type hints, strict mypy
- **Modular**: Clear separation of concerns
- **Testable**: Dependency injection

## PR Workflow

```
📋 Backlog → 🎯 Ready → 🔧 In Progress → 👀 In Review → ✅ Done
```

**Branch naming**: `feature/<issue>-<description>` (e.g., `feature/2-metric-dataclasses`)

**Commit format**: `<type>(<scope>): <description>` (Conventional Commits)

## Memory Integration

Uses KuzuMemory for context management:
- `kuzu-memory enhance <prompt>` - Add project context
- `kuzu-memory recall <query>` - Query memories
- `kuzu-memory learn <content>` - Store learnings

## Key Links

| Resource | URL |
|----------|-----|
| GitHub Project | https://github.com/users/bobmatnyc/projects/13 |
| Milestones | https://github.com/bobmatnyc/mcp-vector-search/milestones |
| Issues | https://github.com/bobmatnyc/mcp-vector-search/issues |
| PR Workflow | [docs/development/pr-workflow-guide.md](docs/development/pr-workflow-guide.md) |

## Pre-Commit Checklist

- [ ] `make pre-publish` passes
- [ ] Tests added/updated
- [ ] Docs updated
- [ ] CHANGELOG.md updated
- [ ] No files in project root

---

*Last Updated: December 9, 2024*
