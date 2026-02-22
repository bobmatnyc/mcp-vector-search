# Multi-Language PR Review Example

This document demonstrates the multi-language support in the PR/MR review system.

## Overview

The PR review system now automatically detects languages from file extensions and provides language-specific standards, idioms, anti-patterns, and security concerns.

## Supported Languages

- **Python** (.py, .pyw, .pyi)
- **TypeScript** (.ts, .tsx, .mts, .cts)
- **JavaScript** (.js, .jsx, .mjs, .cjs)
- **Java** (.java)
- **C#** (.cs, .csx)
- **Ruby** (.rb, .rake, .gemspec)
- **Go** (.go)
- **Rust** (.rs)
- **PHP** (.php, .phtml)
- **Swift** (.swift)
- **Kotlin** (.kt, .kts)
- **Scala** (.scala, .sc)

## How It Works

### 1. Language Detection

When you submit a PR, the system:
- Scans all file patches for file extensions
- Matches extensions to language profiles
- Extracts language-specific standards

### 2. Language Context Injection

For each detected language, the review includes:

**Idioms to enforce:**
- Language-specific best practices (e.g., "Use type hints for Python", "Prefer const over let in TypeScript")

**Anti-patterns to flag:**
- Common mistakes (e.g., "Mutable default arguments in Python", "Using any type in TypeScript")

**Security concerns:**
- Language-specific vulnerabilities (e.g., "SQL injection via string formatting", "XSS via innerHTML")

### 3. Config File Discovery

The system also discovers language-specific config files:
- **Python**: `pyproject.toml`, `.flake8`, `mypy.ini`, `ruff.toml`
- **TypeScript**: `tsconfig.json`, `.eslintrc.js`, `.eslintrc.json`
- **Ruby**: `.rubocop.yml`
- **Java**: `checkstyle.xml`, `pom.xml`
- **Go**: `.golangci.yml`
- **Rust**: `Cargo.toml`, `clippy.toml`
- **PHP**: `.php-cs-fixer.php`, `phpstan.neon`
- **Swift**: `.swiftlint.yml`
- **Kotlin**: `detekt.yml`
- **Scala**: `scalafmt.conf`

## Example: Mixed-Language PR

### PR with Python + TypeScript Changes

**Files Changed:**
- `backend/api.py` (Python)
- `frontend/components/UserList.tsx` (TypeScript)

**Generated Review Context:**

```markdown
## Language-Specific Standards

## Python Standards

**Idioms to enforce:**
- Use type hints for all function signatures (PEP 484)
- Follow PEP 8 style guide (snake_case, 79-88 char lines)
- Use dataclasses or Pydantic for data models
- Prefer context managers (with) for resource management
- Use f-strings for string formatting (Python 3.6+)

**Anti-patterns to flag:**
- Mutable default arguments (def f(x=[]))
- Catching bare except: without specifying exception type
- Using == None instead of is None
- String concatenation in loops (use join() instead)

**Security concerns:**
- SQL injection via string formatting in queries
- Command injection via subprocess/os.system with user input
- Unsafe deserialization (pickle.loads on untrusted data)
- Hardcoded secrets/credentials in code

## TypeScript Standards

**Idioms to enforce:**
- Use strict TypeScript (strict: true in tsconfig)
- Prefer interface over type for object shapes
- Use readonly for immutable properties
- Prefer optional chaining (?.) over null checks
- Use nullish coalescing (??) over || for null/undefined

**Anti-patterns to flag:**
- Using any type (disables type safety)
- Non-null assertions (!) without justification
- Implicit any in function parameters
- Mutating arrays/objects passed as parameters

**Security concerns:**
- XSS via innerHTML/dangerouslySetInnerHTML
- Prototype pollution vulnerabilities
- Insecure eval() usage
- Hardcoded API keys in client code
```

## Example Review Comments

The LLM will now provide language-aware comments:

### Python File (`backend/api.py`)

```json
{
  "file_path": "backend/api.py",
  "line_number": 42,
  "comment": "SQL injection vulnerability: user input is concatenated directly into query without parameterization. This violates Python security best practices.",
  "severity": "critical",
  "category": "security",
  "suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
  "is_blocking": true
}
```

### TypeScript File (`frontend/components/UserList.tsx`)

```json
{
  "file_path": "frontend/components/UserList.tsx",
  "line_number": 15,
  "comment": "Using 'any' type disables TypeScript's type safety. TypeScript idiom: prefer 'unknown' for unsafe values and use type guards.",
  "severity": "medium",
  "category": "quality",
  "suggestion": "Change type from 'any' to 'unknown' and add type guard: if (typeof data === 'object' && data !== null) { ... }",
  "is_blocking": false
}
```

## Usage

### CLI

```bash
# Review PR between branches (auto-detects languages)
uv run mcp-vector-search pr-review --base main --head feature-branch

# Review with custom instructions (language context still applied)
uv run mcp-vector-search pr-review --base main --head feature-branch --instructions path/to/custom.yaml
```

### Programmatic

```python
from pathlib import Path
from mcp_vector_search.analysis.review import PRReviewEngine
from mcp_vector_search.core.search import SemanticSearchEngine
from mcp_vector_search.core.llm_client import LLMClient

# Initialize components
search_engine = SemanticSearchEngine(...)
llm_client = LLMClient(...)
project_root = Path("/path/to/project")

# Create review engine
engine = PRReviewEngine(
    search_engine=search_engine,
    knowledge_graph=None,  # Optional
    llm_client=llm_client,
    project_root=project_root
)

# Review PR (language context automatically injected)
result = await engine.review_from_git(
    base_ref="main",
    head_ref="feature-branch"
)

# Print results
print(result.summary)
for comment in result.comments:
    print(f"{comment.severity}: {comment.comment}")
```

## Configuration

### Custom Language Standards

You can override or extend language standards in `.mcp-vector-search/review-instructions.yaml`:

```yaml
language_standards:
  - "Python: Use Pydantic v2 for all data models"
  - "TypeScript: Use React hooks instead of class components"
  - "All languages: Maximum function length 50 lines"

scope_standards:
  - "All API endpoints must have rate limiting"
  - "All database queries must have timeouts"
```

The language-specific context is **additive** — custom instructions are merged with auto-detected language standards.

## Benefits

### 1. Language-Aware Reviews
- Python code is reviewed against PEP 8 and Python-specific idioms
- TypeScript code is reviewed against strict typing standards
- Java code is reviewed against SOLID principles and Java conventions

### 2. Mixed-Language PRs
- Reviews handle PRs touching multiple languages gracefully
- Each language gets appropriate standards applied
- No manual configuration needed

### 3. Security
- Language-specific security vulnerabilities are highlighted
- SQL injection patterns differ by language (Python string formatting vs Java concatenation)
- XSS concerns in frontend code, command injection in backend code

### 4. Consistency
- Teams working across multiple languages get consistent review quality
- Standards are enforced uniformly across the codebase
- New developers learn language-specific best practices through reviews

## Architecture

### File Structure

```
src/mcp_vector_search/analysis/review/
├── language_profiles.py       # Language definitions (NEW)
├── instructions.py            # Config file discovery (UPDATED)
├── pr_engine.py               # PR review engine (UPDATED)
├── prompts.py                 # Review prompts (UPDATED)
└── pr_models.py               # Data models
```

### Key Components

1. **LanguageProfile**: Dataclass defining standards for each language
2. **detect_languages()**: Detects languages from file extensions
3. **get_languages_in_pr()**: Returns unique language profiles in PR
4. **_build_language_context()**: Formats language context for LLM prompt
5. **InstructionsLoader**: Discovers language-specific config files

## Testing

Run tests to verify multi-language support:

```bash
# Run all review tests
uv run pytest tests/unit/test_language_profiles.py -v

# Run specific test
uv run pytest tests/unit/test_language_profiles.py::TestLanguageProfiles::test_detect_languages_multi_language -v
```

## Future Enhancements

Potential improvements:
- Add more languages (Perl, Haskell, Elixir, etc.)
- Extract actual rules from config files (e.g., parse `.rubocop.yml` for specific cops)
- Language-specific complexity metrics (Python: McCabe complexity, TypeScript: type coverage)
- Framework-specific standards (Django, Rails, React, Angular)
