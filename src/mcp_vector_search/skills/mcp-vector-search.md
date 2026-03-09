---
name: mcp-vector-search
description: Semantic code search, knowledge graph, and analysis tools for this project
version: 1.0.0
author: mcp-vector-search
---

# MCP Vector Search — Project Skill

## Overview

MCP Vector Search (MVS) provides semantic code search, knowledge graph traversal, static analysis, and documentation generation for this project. It indexes the codebase into a vector database and builds a typed knowledge graph so Claude can find code by meaning, trace call chains, detect smells, and review pull requests with full context.

## Quick Reference

### Search

| Command | Use case |
|---------|----------|
| `mvs search "<query>"` | Semantic search across indexed code |
| `mvs search "<query>" --limit N` | Limit result count |
| `mvs search "<query>" --lang python` | Filter by language |
| `mvs search "<query>" --format json` | JSON output for scripting |
| `mvs search "<query>" --git-blame` | Enrich results with last-author info |

### Knowledge Graph

| Command | Use case |
|---------|----------|
| `mvs kg build` | Build/rebuild full knowledge graph |
| `mvs kg build --incremental` | Only rebuild changed files (hash-based) |
| `mvs kg status` | Entity and edge counts |
| `mvs kg trace <entity> --depth N` | N-hop call chain (tree/flat/json) |
| `mvs kg trace <entity> --direction incoming` | Find callers of an entity |
| `mvs kg query "<cypher>"` | Raw Kuzu query |
| `mvs kg history <entity>` | What commit last touched this entity |
| `mvs kg callers-at <entity> <commit>` | Callers as of a specific git commit |
| `mvs kg ancestor <sha1> <sha2>` | Check commit ordering |

### Analysis

| Command | Use case |
|---------|----------|
| `mvs analyze` | Full complexity + code smells report |
| `mvs analyze dead-code` | Find unreachable functions |
| `mvs analyze --complexity` | Cyclomatic complexity hotspots |
| `mvs check-circular` | Detect circular import cycles |

### Wiki / Story

| Command | Use case |
|---------|----------|
| `mvs wiki generate` | Generate markdown wiki from codebase |
| `mvs wiki publish` | Push wiki to GitHub Wiki (.wiki.git) |
| `mvs story generate` | Generate codebase narrative |

## CLI Commands

### Search
```bash
mvs search "<query>"                      # semantic search
mvs search "<query>" --limit N            # limit results
mvs search "<query>" --format json        # JSON output for scripting
mvs search "<query>" --lang python        # filter by language
mvs search "<query>" --git-blame          # enrich with last-author info
```

### Indexing
```bash
mvs index                  # incremental index (only changed files)
mvs index --force          # full reindex
mvs index --no-kg          # skip knowledge graph build
mvs status                 # show index stats
```

### Knowledge Graph
```bash
mvs kg build                              # build/rebuild KG
mvs kg build --incremental               # only rebuild changed files (hash-based)
mvs kg status                            # entity and edge counts
mvs kg trace <entity> --depth N          # N-hop call chain (tree/flat/json)
mvs kg trace <entity> --direction incoming   # find callers
mvs kg history <entity>                  # what commit last touched this entity
mvs kg callers-at <entity> <commit>      # callers as of a git commit
mvs kg ancestor <sha1> <sha2>           # check commit ordering
mvs kg query "<cypher>"                  # raw Kuzu query
```

### Analysis
```bash
mvs analyze                # full complexity + code smells report
mvs analyze dead-code      # find unreachable functions
mvs analyze --complexity   # cyclomatic complexity hotspots
```

### Wiki
```bash
mvs wiki generate          # generate markdown wiki from codebase
mvs wiki publish           # push wiki to GitHub Wiki (.wiki.git)
```

### Other
```bash
mvs visualize              # launch D3.js graph visualization server
mvs story generate         # generate codebase narrative
mvs doctor                 # health check
mvs migrate                # run schema/skill migrations
```

## MCP Tools (Claude Desktop / Claude Code)

When this project's MCP server is running, Claude has direct access to these tools.

### Search Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_code` | `query, limit, language, file_pattern` | Semantic search across indexed code |
| `search_similar` | `file_path, limit` | Find files similar to a given file |
| `search_context` | `query, limit` | Search with surrounding context |

### Knowledge Graph Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `kg_build` | `incremental` | Build or incrementally update KG |
| `kg_stats` | — | Node/edge counts by type |
| `kg_query` | `cypher` | Raw Kuzu query |
| `kg_ontology` | — | Describe entity types and relationships |
| `trace_execution_flow` | `entry_point, depth, direction` | Call chain traversal (max depth 8) |
| `kg_ia` | `question` | Natural-language KG question answering |

### Analysis Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `analyze_project` | — | Full analysis report |
| `analyze_file` | `file_path` | Single-file analysis |
| `find_smells` | — | Code smell detection |
| `get_complexity_hotspots` | `limit` | Most complex functions |
| `check_circular_dependencies` | — | Detect import cycles |
| `interpret_analysis` | `report` | AI interpretation of analysis results |

### Review Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `review_repository` | — | Whole-repo review |
| `review_pull_request` | `pr_url` | PR review with diff context |
| `code_review` | `file_path` | Single-file review |

### Documentation Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `wiki_generate` | `topic` | Generate wiki page for a topic |
| `story_generate` | — | Narrative codebase summary |
| `save_report` | `content, filename` | Persist analysis to disk |

### Project Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_project_status` | — | Index health and stats |
| `index_project` | `force` | Trigger reindexing |
| `embed_chunks` | — | Embed pending chunks |

## Common Workflows

### First-time setup
```bash
mvs setup          # auto-detect project, index, configure MCP platforms
mvs kg build       # build knowledge graph
mvs doctor         # verify everything is healthy
```

### Finding code by meaning
```bash
mvs search "authentication middleware"
mvs search "database connection pooling" --lang python --limit 10
mvs search "retry with exponential backoff" --git-blame
```

### Understanding call flows
```bash
mvs kg trace SemanticIndexer --depth 3
mvs kg trace run_phase1_chunking --direction incoming
mvs kg trace UserService --depth 2 --format json
```

### After modifying files
```bash
mvs index                    # re-index changed files only
mvs kg build --incremental   # update KG for changed files only
```

### Code review prep
```bash
mvs analyze
mvs analyze dead-code
mvs kg trace <changed_function> --depth 2 --direction incoming
mvs review_pull_request <pr_url>
```

### Exploring architecture
```bash
mvs kg query "MATCH (f:Function)-[:CALLS]->(g:Function) WHERE f.file_path CONTAINS 'handlers' RETURN f.name, g.name LIMIT 20"
mvs kg ontology
mvs story generate
```

### Generating docs
```bash
mvs wiki generate
mvs wiki publish
```

## Configuration

Project config lives at `.mcp-vector-search/config.json`. Key fields:

| Field | Description |
|-------|-------------|
| `embedding_model` | sentence-transformers model name |
| `similarity_threshold` | search relevance cutoff (0.0–1.0) |
| `file_extensions` | indexed file types |
| `kg_enabled` | whether to build knowledge graph |
| `watch_enabled` | auto-reindex on file change |

## Notes

- The KG `--incremental` flag uses SHA-256 file hashes, not mtimes — safe to use always
- `trace_execution_flow` depth is capped at 8 to prevent graph explosion
- `commit_sha` on entities reflects state at last `mvs kg build` time (not full git history)
- `search_code` returns cosine-similarity ranked results; adjust `similarity_threshold` if results are too broad/narrow
- Large projects: use `--limit` to keep search fast; `visualize` may need `--code-only`
- Run `mvs migrate` after upgrading mcp-vector-search to apply schema and skill updates
