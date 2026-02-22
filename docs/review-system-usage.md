# AI-Powered Code Review System - Usage Guide

## Overview

The review system provides automated code analysis using:
- **Vector Search**: Find relevant code chunks matching review queries
- **Knowledge Graph**: Understand code relationships and data flow
- **LLM Analysis**: Deep analysis using specialized prompts for each review type

## Architecture

```
User Request
    ↓
ReviewEngine.run_review()
    ↓
1. Vector Search (gather relevant code chunks)
    ↓
2. Knowledge Graph (gather relationships)
    ↓
3. Context Formatting (code + KG → LLM-ready format)
    ↓
4. LLM Analysis (specialized prompts per review type)
    ↓
5. JSON Parsing (extract structured findings)
    ↓
ReviewResult (findings, summary, metadata)
```

## Review Types

### 1. Security Review (`ReviewType.SECURITY`)

Focuses on OWASP Top 10 and common vulnerabilities:
- Injection flaws (SQL, Command, LDAP)
- Authentication & access control
- Cryptographic issues
- Input validation
- Security misconfigurations

**CWE IDs included** for recognized vulnerability types.

### 2. Architecture Review (`ReviewType.ARCHITECTURE`)

Focuses on SOLID principles and design patterns:
- SOLID principles violations
- Coupling & cohesion issues
- Circular dependencies
- God classes
- Dependency management

### 3. Performance Review (`ReviewType.PERFORMANCE`)

Focuses on efficiency and optimization:
- N+1 query problems
- Algorithmic complexity (O(n²) or worse)
- Blocking I/O in async contexts
- Memory leaks
- Missing caching opportunities

## Usage Example

```python
from pathlib import Path
from mcp_vector_search.analysis.review import ReviewEngine, ReviewType
from mcp_vector_search.core.search import SemanticSearchEngine
from mcp_vector_search.core.knowledge_graph import KnowledgeGraph
from mcp_vector_search.core.llm_client import LLMClient
from mcp_vector_search.core.database import VectorDatabase

# Initialize infrastructure
project_root = Path("/path/to/project")
database = VectorDatabase(index_path)
search_engine = SemanticSearchEngine(database, project_root)

# Initialize knowledge graph (optional, graceful degradation)
kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
knowledge_graph = KnowledgeGraph(kg_path) if kg_path.exists() else None

# Initialize LLM client (auto-detects provider from env vars)
llm_client = LLMClient(project_root=project_root)

# Create review engine
engine = ReviewEngine(
    search_engine=search_engine,
    knowledge_graph=knowledge_graph,
    llm_client=llm_client,
    project_root=project_root
)

# Run security review
result = await engine.run_review(
    review_type=ReviewType.SECURITY,
    scope="src/auth",  # Optional: review specific path
    max_chunks=30      # Max code chunks to analyze
)

# Process findings
print(f"Review Summary: {result.summary}")
print(f"Found {len(result.findings)} issues")
print(f"Used {result.context_chunks_used} code chunks")
print(f"Used {result.kg_relationships_used} KG relationships")

for finding in result.findings:
    print(f"\n{finding.severity.upper()}: {finding.title}")
    print(f"  File: {finding.file_path}:{finding.start_line}-{finding.end_line}")
    print(f"  Category: {finding.category}")
    print(f"  Confidence: {finding.confidence:.2f}")
    if finding.cwe_id:
        print(f"  CWE: {finding.cwe_id}")
    print(f"  Recommendation: {finding.recommendation}")
```

## Output Format

### ReviewResult

```python
@dataclass
class ReviewResult:
    review_type: ReviewType           # Type of review performed
    findings: list[ReviewFinding]     # List of findings
    summary: str                      # Summary (e.g., "Found 5 security issues: 2 CRITICAL, 3 HIGH")
    scope: str                        # What was reviewed (path or "entire project")
    context_chunks_used: int          # Number of code chunks analyzed
    kg_relationships_used: int        # Number of KG relationships used
    model_used: str                   # LLM model name
    duration_seconds: float           # Review duration
```

### ReviewFinding

```python
@dataclass
class ReviewFinding:
    title: str                        # Short title (e.g., "SQL Injection in user_query()")
    description: str                  # Detailed explanation
    severity: Severity                # critical, high, medium, low, info
    file_path: str                    # File containing the issue
    start_line: int                   # Start line of issue
    end_line: int                     # End line of issue
    category: str                     # Category (e.g., "SQL Injection", "God Class")
    recommendation: str               # Actionable fix recommendation
    confidence: float                 # Confidence level (0.0-1.0)
    cwe_id: str | None                # CWE ID (for security findings)
    code_snippet: str | None          # Relevant code snippet
    related_files: list[str]          # Related files to review
```

## Targeted Search Queries

The review engine uses specialized search queries per review type:

### Security Queries
- "authentication login password credential"
- "sql query database execute"
- "user input validation sanitize"
- "file upload path traversal"
- "encryption hash token secret key"
- "permission authorization access control"

### Architecture Queries
- "class definition inheritance"
- "import module dependency"
- "configuration settings"
- "error handling exception"
- "interface abstract base class"

### Performance Queries
- "database query loop iteration"
- "cache memory buffer pool"
- "async await concurrent parallel"
- "file read write io operation"
- "sort search algorithm complexity"

## Graceful Degradation

The system gracefully handles missing components:

1. **No Knowledge Graph**: Continues with vector search only
2. **No Search Results**: Returns empty result with explanation
3. **LLM Parsing Errors**: Logs error and returns empty findings

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: AWS Bedrock credentials

The LLM client auto-detects the provider based on available credentials.

### Max Chunks

Default: 30 code chunks per review

Higher values provide more context but increase LLM costs and latency.

## Performance Considerations

- **Vector Search**: Fast (milliseconds per query)
- **Knowledge Graph**: Fast (single-digit milliseconds per entity)
- **LLM Analysis**: Slow (1-10 seconds depending on context size)

**Typical review duration**: 5-15 seconds for 30 chunks.

## Next Steps (Phase 2-4)

This is **Phase 1** of Issue #89. Future phases will add:

- **Phase 2**: CLI command (`analyze review --type security`)
- **Phase 3**: Chat integration (tool calling)
- **Phase 4**: SARIF export for CI/CD integration
