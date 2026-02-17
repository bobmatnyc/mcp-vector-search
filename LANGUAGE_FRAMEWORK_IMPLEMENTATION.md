# ProgrammingLanguage and ProgrammingFramework Entities

## Overview

This implementation adds **ProgrammingLanguage** and **ProgrammingFramework** entities to the Knowledge Graph, enabling language and framework detection from codebases and configuration files.

## New Entity Types

### 1. ProgrammingLanguage

Represents programming languages detected in the codebase.

**Properties:**
- `id` (str): Unique identifier (e.g., `"lang:python"`)
- `name` (str): Language name (e.g., `"Python"`, `"JavaScript"`)
- `version` (str): Version if detected (e.g., `"3.12"`, `"ES2022"`)
- `file_extensions` (str): Comma-separated list of extensions (e.g., `".py,.pyi"`)

**Supported Languages:**
- Python (`.py`, `.pyi`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Rust (`.rs`)
- Go (`.go`)
- Java (`.java`)
- Ruby (`.rb`)
- PHP (`.php`)
- C (`.c`, `.h`)
- C++ (`.cpp`, `.cc`, `.cxx`, `.hpp`)
- C# (`.cs`)
- Swift (`.swift`)
- Kotlin (`.kt`)
- Scala (`.scala`)
- R (`.r`)
- Objective-C (`.m`)
- Lua (`.lua`)
- Perl (`.pl`)
- Shell (`.sh`, `.bash`, `.zsh`)

### 2. ProgrammingFramework

Represents frameworks detected from config files.

**Properties:**
- `id` (str): Unique identifier (e.g., `"framework:fastapi"`)
- `name` (str): Framework display name (e.g., `"FastAPI"`, `"React"`)
- `version` (str): Version from package manifest
- `language_id` (str): Reference to ProgrammingLanguage node
- `category` (str): Framework category (web, testing, orm, cli, etc.)

**Supported Frameworks:**

**Python (from pyproject.toml, requirements.txt):**
- FastAPI (web)
- Django (web)
- Flask (web)
- pytest (testing)
- unittest (testing)
- SQLAlchemy (orm)
- Pydantic (validation)
- NumPy (scientific)
- Pandas (data)
- Requests (http)
- aiohttp (http)
- Click (cli)
- Typer (cli)

**JavaScript/TypeScript (from package.json):**
- React (web)
- Vue (web)
- Angular (web)
- Express (web)
- Next.js (web)
- Nuxt.js (web)
- Jest (testing)
- Mocha (testing)
- Vitest (testing)
- Axios (http)

**Rust (from Cargo.toml):**
- Actix Web (web)
- Rocket (web)
- Tokio (async)
- Serde (serialization)

**Go (from go.mod):**
- Gin (web)
- Echo (web)
- Fiber (web)
- GORM (orm)

## New Relationships

### 1. WRITTEN_IN (CodeEntity → ProgrammingLanguage)

Links code entities to their programming language based on file extension.

**Example:**
```cypher
MATCH (e:CodeEntity {name: "search_code"})-[:WRITTEN_IN]->(l:ProgrammingLanguage)
RETURN e.name, l.name
// Result: "search_code", "python"
```

### 2. USES_FRAMEWORK (Project → ProgrammingFramework)

Links projects to frameworks detected in configuration files.

**Example:**
```cypher
MATCH (p:Project)-[:USES_FRAMEWORK]->(f:ProgrammingFramework)
RETURN p.name, f.name, f.category
// Result: "mcp-vector-search", "FastAPI", "web"
```

### 3. FRAMEWORK_FOR (ProgrammingFramework → ProgrammingLanguage)

Links frameworks to their target programming languages.

**Example:**
```cypher
MATCH (f:ProgrammingFramework {name: "FastAPI"})-[:FRAMEWORK_FOR]->(l:ProgrammingLanguage)
RETURN f.name, l.name
// Result: "FastAPI", "python"
```

## Detection Logic

### Language Detection

Languages are automatically detected from file extensions when building the Knowledge Graph:

1. **Scan CodeEntity nodes** for file paths
2. **Extract file extensions** (e.g., `.py`, `.js`, `.rs`)
3. **Map extensions to languages** using predefined mapping
4. **Create ProgrammingLanguage nodes** for detected languages
5. **Create WRITTEN_IN relationships** for each CodeEntity

### Framework Detection

Frameworks are detected from configuration files:

#### Python

**pyproject.toml:**
```toml
[project]
dependencies = [
    "fastapi>=0.100.0",
    "sqlalchemy>=2.0.0"
]

[tool.poetry.dependencies]
pytest = "^7.0"
```

**requirements.txt:**
```
fastapi>=0.100.0
pytest==7.4.0
sqlalchemy>=2.0.0
```

#### JavaScript/TypeScript

**package.json:**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "express": "^4.18.0"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
```

#### Rust

**Cargo.toml:**
```toml
[dependencies]
actix-web = "4.0"
tokio = { version = "1.0", features = ["full"] }
```

#### Go

**go.mod:**
```
module example.com/myapp

require (
    github.com/gin-gonic/gin v1.9.0
    gorm.io/gorm v1.25.0
)
```

## Usage Examples

### Query Languages in Codebase

```cypher
MATCH (l:ProgrammingLanguage)
RETURN l.name, l.file_extensions
ORDER BY l.name
```

### Find Frameworks by Category

```cypher
MATCH (f:ProgrammingFramework {category: "web"})
RETURN f.name, f.version
ORDER BY f.name
```

### Find Code Written in Specific Language

```cypher
MATCH (e:CodeEntity)-[:WRITTEN_IN]->(l:ProgrammingLanguage {name: "Python"})
RETURN e.name, e.entity_type, e.file_path
LIMIT 10
```

### Find Frameworks for a Language

```cypher
MATCH (f:ProgrammingFramework)-[:FRAMEWORK_FOR]->(l:ProgrammingLanguage {name: "Python"})
RETURN f.name, f.category, f.version
ORDER BY f.category, f.name
```

### Project Technology Stack

```cypher
MATCH (p:Project)-[:USES_FRAMEWORK]->(f:ProgrammingFramework)-[:FRAMEWORK_FOR]->(l:ProgrammingLanguage)
RETURN p.name,
       collect(DISTINCT l.name) AS languages,
       collect(f.name + ' (' + f.category + ')') AS frameworks
```

### Language Ecosystem Analysis

```cypher
// Count code entities per language
MATCH (e:CodeEntity)-[:WRITTEN_IN]->(l:ProgrammingLanguage)
RETURN l.name, count(e) AS entity_count
ORDER BY entity_count DESC
```

## CLI Integration

The new entities are automatically displayed in the `kg status` command:

```bash
mcp-vector-search kg status
```

**Output:**
```
Knowledge Graph Status

Nodes (125 total)
├── Code Entities          95
│   ├── Functions          45
│   ├── Classes            30
│   ├── Modules            15
│   └── Files               5
├── Languages               3  # NEW
├── Frameworks              5  # NEW
├── Doc Sections           15
├── Tags                   10
└── Persons                 5

Relationships (450 total)
├── Code Structure         250
│   ├── Calls              120
│   ├── Imports             80
│   ├── Inherits            30
│   └── Contains            20
├── Language/Framework      98  # NEW
│   ├── Written_In          90  # NEW
│   ├── Uses_Framework       5  # NEW
│   └── Framework_For        3  # NEW
└── ...
```

## Files Modified

### 1. `src/mcp_vector_search/core/knowledge_graph.py`

**Added:**
- `ProgrammingLanguage` dataclass
- `ProgrammingFramework` dataclass
- Schema creation for `ProgrammingLanguage` and `ProgrammingFramework` node tables
- Schema creation for `WRITTEN_IN`, `USES_FRAMEWORK`, `FRAMEWORK_FOR` relationship tables
- `add_programming_language()` method
- `add_programming_framework()` method
- `add_written_in_relationship()` method
- `add_uses_framework_relationship()` method
- `add_framework_for_relationship()` method
- Updated `get_detailed_stats()` to include language and framework counts

### 2. `src/mcp_vector_search/core/kg_builder.py`

**Added:**
- Imports for `ProgrammingLanguage` and `ProgrammingFramework`
- `_detect_language_from_extension()` method
- `_get_extensions_for_language()` method
- `_extract_languages_and_frameworks()` method
- `_detect_python_frameworks()` method
- `_detect_javascript_frameworks()` method
- `_detect_rust_frameworks()` method
- `_detect_go_frameworks()` method
- Integration in `build_from_database()` to extract languages, frameworks, and create relationships

### 3. `src/mcp_vector_search/cli/commands/kg.py`

**Added:**
- Language and Framework node display in `kg status` command
- Language/Framework relationships section in relationship tree

## Testing

### Test Script

```bash
python test_language_framework.py
```

### Test Results

```
✓ KG initialized
✓ Added Python language
✓ Added JavaScript language
✓ Added FastAPI framework
✓ Added React framework
✓ Created FRAMEWORK_FOR relationship (FastAPI → Python)
✓ Created FRAMEWORK_FOR relationship (React → JavaScript)

Querying languages...
  - Python (3.12)
  - JavaScript (ES2022)

Querying frameworks...
  - FastAPI [web] (v0.100.0)
  - React [web] (v18.2.0)

Querying FRAMEWORK_FOR relationships...
  - FastAPI → Python
  - React → JavaScript

Getting detailed stats...
  Languages: 2
  Frameworks: 2
  FRAMEWORK_FOR relationships: 2

✓ Test completed successfully!
```

## Build Integration

Languages and frameworks are automatically extracted during `kg build`:

```bash
mcp-vector-search kg build
```

**Output:**
```
Building Knowledge Graph
Project: /path/to/project

Phase 1: Scanning chunks... [████████████████████] 100%
Phase 2: Extracting entities... [████████████████████] 100%
Phase 3: Building relations... [████████████████████] 100%

Extracting work entities from git...
✓ Extracted 5 person entities from git
✓ Extracted project: mcp-vector-search

Extracting programming languages and frameworks...
✓ Detected 3 programming languages
✓ Detected 5 frameworks

Creating WRITTEN_IN relationships...
✓ Created 90 WRITTEN_IN relationships

Knowledge Graph Statistics
┌─────────────────┬───────┐
│ Metric          │ Count │
├─────────────────┼───────┤
│ Code Entities   │    95 │
│ Languages       │     3 │
│ Frameworks      │     5 │
│ Written_In      │    90 │
│ Uses_Framework  │     5 │
│ Framework_For   │     3 │
└─────────────────┴───────┘

✓ Knowledge graph built successfully!
```

## Future Enhancements

### Potential Improvements

1. **Version Detection**: Parse versions from config files more robustly
2. **Language Version Detection**: Detect Python version from pyproject.toml or .python-version
3. **More Frameworks**: Add support for additional frameworks (e.g., Spring Boot, Laravel, etc.)
4. **Framework Dependencies**: Track framework-to-framework dependencies
5. **Language Interop**: Detect FFI/interop relationships (e.g., Python calling Rust via PyO3)
6. **Package Manager Detection**: Detect npm, pip, cargo, go modules versions
7. **Build Tool Detection**: Detect webpack, rollup, gradle, maven, etc.

### Example Future Queries

```cypher
// Multi-language projects
MATCH (p:Project)-[:PART_OF]-(e:CodeEntity)-[:WRITTEN_IN]->(l:ProgrammingLanguage)
WITH p, count(DISTINCT l) AS lang_count
WHERE lang_count > 1
RETURN p.name, lang_count
ORDER BY lang_count DESC

// Framework popularity across projects
MATCH (p:Project)-[:USES_FRAMEWORK]->(f:ProgrammingFramework)
RETURN f.name, f.category, count(p) AS project_count
ORDER BY project_count DESC

// Language ecosystem health
MATCH (l:ProgrammingLanguage)<-[:FRAMEWORK_FOR]-(f:ProgrammingFramework)
RETURN l.name,
       count(f) AS framework_count,
       collect(f.name) AS frameworks
ORDER BY framework_count DESC
```

## Conclusion

This implementation provides comprehensive language and framework tracking in the Knowledge Graph, enabling:

- **Language Analysis**: Understand which languages are used in the codebase
- **Framework Detection**: Automatically detect frameworks from config files
- **Technology Stack Visibility**: Query the complete tech stack for projects
- **Code Organization**: Link code entities to their programming languages
- **Ecosystem Insights**: Analyze framework usage and language ecosystems

The system is extensible and can be enhanced with additional language support, framework detection, and relationship types as needed.
