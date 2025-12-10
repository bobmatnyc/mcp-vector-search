# HyperDev December 2025: AI-Powered Code Search & Analysis

> **Building the Future of Developer Tooling with Ticket-Driven Development**

## Introduction

Welcome to the December 2025 edition of HyperDev, where we explore the cutting edge of AI-powered development tools. This month, we're diving deep into **mcp-vector-search**—a CLI-first semantic code search tool that's evolving into a comprehensive code intelligence platform.

What makes this project special isn't just the technology—it's how we're building it. We're using **Ticket-Driven Development (TkDD)** with Claude MPM, demonstrating how AI agents can orchestrate complex software projects from design to deployment.

**Follow along**: https://github.com/users/bobmatnyc/projects/13

---

## What is mcp-vector-search?

**mcp-vector-search** is a CLI tool that brings semantic understanding to code search. Unlike traditional grep or regex-based search, it understands what your code *means*, not just what it *contains*.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Find code by meaning, not just keywords |
| **Hybrid Search** | Combine semantic + keyword for precision |
| **MCP Integration** | Native Claude Desktop integration |
| **D3.js Visualization** | Interactive code relationship graphs |
| **AI Chat** | Natural language code Q&A |

### Quick Start

```bash
# Install
pipx install mcp-vector-search

# Index your codebase
mcp-vector-search index

# Search semantically
mcp-vector-search search "authentication middleware"

# Chat with your code
mcp-vector-search chat "how does the login flow work?"
```

---

## New AI Features (v0.15-v0.16)

### 1. Agentic Chat Mode

The `chat` command now supports **dual-intent mode**—automatically detecting whether you're asking a question or searching for code:

```bash
# Question mode (AI explains)
mcp-vector-search chat "how does caching work in this project?"

# Find mode (returns code)
mcp-vector-search chat "find the database connection pool"
```

**Key features**:
- `--think` flag for complex reasoning (uses advanced models)
- `--files` filter to scope results
- Context-aware responses with code citations

### 2. Multi-Provider AI Support

Choose your AI provider based on needs:

| Provider | Use Case | Flag |
|----------|----------|------|
| OpenAI GPT-4o | Complex reasoning | `--think` |
| Claude Sonnet | Balanced performance | default |
| Ollama (local) | Privacy-first | `--provider ollama` |

### 3. MCP Server Integration

Seamless integration with Claude Desktop and other MCP-compatible tools:

```bash
# Auto-detect and configure
mcp-vector-search setup

# Manual platform selection
mcp-vector-search setup --platform claude_desktop
```

Exposed MCP tools:
- `search_code` - Semantic code search
- `search_similar` - Find similar code patterns
- `search_context` - Contextual code discovery
- `get_project_status` - Index health check
- `index_project` - Trigger reindexing

---

## Roadmap: Structural Code Analysis

We're building a comprehensive code analysis system, and we're doing it in public. Here's what's coming:

### Phase 1: Core Metrics (Dec 10-23, 2024)

**Goal**: Add structural metrics to every indexed code chunk

- **Cognitive Complexity**: SonarQube-compatible complexity scoring
- **Cyclomatic Complexity**: Path counting for test coverage
- **Nesting Depth**: Identify deeply nested code
- **Parameter Count**: Flag functions with too many params

```bash
# Coming soon
mcp-vector-search analyze --quick
```

### Phase 2: Quality Gates (Dec 24-30, 2024)

**Goal**: CI/CD integration for code quality

- Configurable thresholds (strict/standard/relaxed)
- SARIF output for GitHub Actions
- `--fail-on-smell` for quality gates
- Diff-aware analysis (only check changed files)

```bash
# Coming soon
mcp-vector-search analyze --fail-on-smell --output sarif
```

### Phase 3: Cross-File Analysis (Dec 31 - Jan 6, 2025)

**Goal**: Understand code relationships across files

- **Coupling Metrics**: Afferent/efferent coupling
- **Instability Index**: Identify fragile modules
- **Circular Dependencies**: Detect import cycles
- **Trend Tracking**: SQLite-based historical analysis

### Phase 4: Visualization Export (Jan 7-13, 2025)

**Goal**: Beautiful dashboards for code health

- JSON export for custom dashboards
- Standalone HTML reports
- Halstead metrics for complexity
- Technical debt estimation

### Phase 5: Search Integration (Jan 20 - Feb 3, 2025)

**Goal**: Quality-aware search results

- Filter by complexity: `--max-complexity 15`
- Exclude smelly code: `--no-smells`
- Quality-weighted ranking
- Expose analysis as MCP tools

---

## Ticket-Driven Development (TkDD)

This project is built using **Ticket-Driven Development**—a methodology where AI agents orchestrate work through tickets, not just code.

### What is TkDD?

TkDD treats tickets as the primary unit of work:

1. **Ticket as Contract**: Every feature starts with a well-defined ticket
2. **Continuous Updates**: Agents update tickets throughout the work lifecycle
3. **Evidence-Based Completion**: Tickets close with proof, not promises
4. **Dependency Tracking**: Issues explicitly declare what they block/are blocked by

### How We're Using It

Our [GitHub Project](https://github.com/users/bobmatnyc/projects/13) demonstrates TkDD in action:

```
📋 Backlog → 🎯 Ready → 🔧 In Progress → 👀 In Review → ✅ Done
```

**Key practices**:

1. **Epic → Issue → PR**: Every feature flows through this hierarchy
2. **Dependency Mapping**: Issues declare blockers in their body
3. **Milestone Tracking**: 5 milestones map to 5 release versions
4. **Roadmap View**: GitHub Projects roadmap shows timeline

### Example: Issue #2

```markdown
## Create metric dataclasses and interfaces

**Blocked by:** None (can start immediately)
**Blocks:** #3, #4, #5, #6, #7, #8, #9

## Tasks
- [ ] Create ChunkMetrics dataclass
- [ ] Create FileMetrics dataclass
- [ ] Create MetricCollector ABC
...
```

When work starts:
1. Ticket transitions to "In Progress"
2. Branch created: `feature/2-metric-dataclasses`
3. Commits reference: `Refs #2`
4. PR closes ticket: `Closes #2`

---

## Follow Along

We're building this in public. Here's how to follow:

### GitHub Resources

| Resource | URL |
|----------|-----|
| **Project Board** | https://github.com/users/bobmatnyc/projects/13 |
| **Roadmap View** | https://github.com/users/bobmatnyc/projects/13/views/1 |
| **Milestones** | https://github.com/bobmatnyc/mcp-vector-search/milestones |
| **Design Doc** | [structural-analysis-design.md](../research/mcp-vector-search-structural-analysis-design.md) |

### Try It Yourself

```bash
# Install the current version
pipx install mcp-vector-search

# Index a codebase
cd your-project
mcp-vector-search init
mcp-vector-search index

# Search semantically
mcp-vector-search search "error handling patterns"

# Chat about your code
mcp-vector-search chat "explain the main architecture"
```

### Contribute

The project is open source. To contribute:

1. Check the [project board](https://github.com/users/bobmatnyc/projects/13) for ready issues
2. Follow the [PR workflow guide](../development/pr-workflow-guide.md)
3. Start with issues that have no blockers

---

## Technical Deep Dive: Cognitive Complexity

One of the key metrics we're implementing is **Cognitive Complexity**—SonarQube's algorithm for measuring code understandability.

### Why Cognitive Complexity?

Unlike cyclomatic complexity (which just counts paths), cognitive complexity measures how hard code is to *understand*:

```python
# Cyclomatic: 11, Cognitive: 11 (both high, but different reasons)
def process_items(items):
    for item in items:           # +1
        if item.valid:           # +1 (+ nesting)
            if item.priority:    # +1 (+ nesting)
                handle_priority(item)
            elif item.urgent:    # +1 (+ nesting)
                handle_urgent(item)
            else:
                handle_normal(item)
        else:
            log_invalid(item)
```

### The Algorithm

```
+1 for each: if, elif, else, for, while, try, catch, switch/match, ternary
+1 additional per nesting level for control structures
Boolean operators (and, or) add +1 but no nesting penalty
```

### Grades

| Grade | Score | Interpretation |
|-------|-------|----------------|
| A | ≤5 | Easy to understand |
| B | ≤10 | Moderate complexity |
| C | ≤15 | Consider refactoring |
| D | ≤25 | Needs refactoring |
| F | >25 | Critical - split this function |

---

## Architecture Preview

Here's the module structure we're building:

```
src/mcp_vector_search/
├── analysis/
│   ├── metrics.py              # Dataclasses
│   ├── collectors/
│   │   ├── base.py             # MetricCollector ABC
│   │   ├── complexity.py       # Cognitive/cyclomatic
│   │   ├── coupling.py         # Afferent/efferent
│   │   └── smells.py           # Code smell detection
│   ├── reporters/
│   │   ├── console.py          # Rich terminal output
│   │   ├── sarif.py            # CI/CD integration
│   │   └── html.py             # Standalone reports
│   └── thresholds.py           # Configurable limits
```

The key insight: **metrics are computed during existing Tree-sitter traversal**, adding near-zero overhead to indexing.

---

## Conclusion

mcp-vector-search is evolving from a search tool into a code intelligence platform. By building with TkDD and Claude MPM, we're demonstrating how AI can orchestrate complex software projects while maintaining quality and traceability.

**Key takeaways**:

1. **Semantic search** changes how developers find code
2. **Structural analysis** adds quality signals to search results
3. **TkDD** brings accountability to AI-assisted development
4. **Open development** lets you follow and contribute

The future of developer tooling is here. Come build it with us.

---

**Links**:
- GitHub: https://github.com/bobmatnyc/mcp-vector-search
- PyPI: https://pypi.org/project/mcp-vector-search/
- Project Board: https://github.com/users/bobmatnyc/projects/13

**Author**: Robert Matsuoka (@bobmatnyc)
**Date**: December 2025
