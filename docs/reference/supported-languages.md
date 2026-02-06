# Supported Languages

MCP Vector Search provides comprehensive language support with AST-based parsing for deep code understanding.

## Language Support Matrix

### Fully Supported (AST Parsing)

Languages with full Tree-sitter AST parsing support:

| Language | Extensions | Parser Features | Documentation Extraction | Code Elements |
|----------|-----------|----------------|-------------------------|---------------|
| **Python** | `.py`, `.pyw`, `.pyi` | Full AST | Docstrings (`"""`, `'''`) | Functions, classes, methods, async functions |
| **JavaScript** | `.js`, `.jsx`, `.mjs`, `.cjs` | Full AST | JSDoc comments (`/** */`) | Functions, classes, methods, arrow functions |
| **TypeScript** | `.ts`, `.tsx`, `.mts`, `.cts` | Full AST | TSDoc comments (`/** */`) | Functions, classes, methods, interfaces, types |
| **Dart/Flutter** | `.dart` | Full AST | Dart doc comments (`///`) | Functions, classes, methods, constructors |
| **PHP** | `.php`, `.phtml` | Full AST | PHPDoc comments (`/** */`) | Functions, classes, methods |
| **Ruby** | `.rb`, `.rake`, `.gemspec` | Full AST | YARD comments (`#`) | Methods, classes, modules |
| **Java** | `.java` | Full AST | JavaDoc comments (`/** */`) | Classes, methods, constructors |
| **Go** | `.go` | Full AST | Go doc comments (`//`) | Functions, methods, structs |
| **Rust** | `.rs` | Full AST | Rust doc comments (`///`, `//!`) | Functions, structs, impl blocks, traits |
| **C#** | `.cs` | Full AST | XML doc comments (`///`) | Classes, interfaces, structs, enums, methods, properties, constructors |

### Markup & Text

| Language | Extensions | Features |
|----------|-----------|----------|
| **HTML** | `.html`, `.htm` | Tag-based parsing, semantic sections |
| **Markdown** | `.md`, `.markdown` | Header-based chunking, code blocks |
| **Text** | All other extensions | Line-based parsing, fallback for unrecognized types |

## Language-Specific Features

### C# Parser

The C# parser provides comprehensive support for .NET development:

**Code Elements:**
- Classes, interfaces, structs, enums
- Methods, constructors, properties
- Namespace declarations

**Documentation:**
- XML documentation comments (`///`)
- Multi-line XML doc extraction
- Summary, param, returns tags

**Attributes:**
- Extracts C# attributes (e.g., `[HttpGet]`, `[Authorize]`, `[JsonProperty]`)
- Stores attributes as decorators for searchability

**Metrics:**
- Cyclomatic complexity calculation for methods
- Parameter and return type extraction

**Fallback:**
- Regex-based parsing when Tree-sitter unavailable
- Graceful degradation to ensure basic functionality

**Example C# Code:**
```csharp
/// <summary>
/// Handles user authentication requests.
/// </summary>
/// <param name="username">The user's username</param>
/// <returns>Authentication result</returns>
[HttpPost]
[Authorize]
public async Task<AuthResult> AuthenticateAsync(string username)
{
    // Method implementation
}
```

**Extracted Metadata:**
- Function name: `AuthenticateAsync`
- Docstring: XML doc comment text
- Decorators: `["[HttpPost]", "[Authorize]"]`
- Parameters: `[{"name": "username", "type": "string"}]`
- Return type: `Task<AuthResult>`
- Complexity score: Calculated cyclomatic complexity

### Python Parser

**Code Elements:**
- Functions, classes, methods
- Async functions and methods
- Decorators, type hints

**Documentation:**
- Triple-quoted docstrings (`"""`, `'''`)
- Google, NumPy, reStructuredText formats

**Metrics:**
- Cyclomatic complexity
- Parameter and return type extraction from annotations

### JavaScript/TypeScript Parser

**Code Elements:**
- Functions, arrow functions, classes, methods
- Interfaces, types (TypeScript)
- Decorators (TypeScript)

**Documentation:**
- JSDoc/TSDoc comments (`/** */`)
- Type annotations (TypeScript)

**Metrics:**
- Cyclomatic complexity
- Parameter types, return types

### Dart/Flutter Parser

**Code Elements:**
- Functions, classes, methods
- Constructors (named and unnamed)
- Widgets and state classes

**Documentation:**
- Dart doc comments (`///`)
- Multi-line documentation extraction

**Metrics:**
- Complexity scoring
- Parameter and return type extraction

## Adding New Languages

MCP Vector Search uses a plugin architecture for language support. To add a new language:

1. **Create Parser**: Extend `BaseParser` class
2. **Implement Tree-sitter**: Use `tree-sitter-language-pack` for AST parsing
3. **Extract Chunks**: Implement `parse_content()` to extract code elements
4. **Register Parser**: Add to `ParserRegistry` in `registry.py`
5. **Configure Extensions**: Map file extensions to parser

**Example:**
```python
from .base import BaseParser

class MyLanguageParser(BaseParser):
    def __init__(self):
        super().__init__("my_language")
        # Initialize Tree-sitter parser

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        # Extract code chunks from content
        pass
```

## Fallback Parsing

For unsupported file types, MCP Vector Search provides fallback parsing:

- **Text Parser**: Line-based chunking for text files
- **Regex Fallback**: Pattern-based parsing when Tree-sitter unavailable
- **Graceful Degradation**: Ensures basic search functionality for all file types

## Performance Considerations

**AST Parsing:**
- First-class support for Tree-sitter AST parsing
- Fast parallel processing for large codebases
- Caching of parsed AST nodes

**Complexity Calculation:**
- Cyclomatic complexity for supported languages
- Conditional branching analysis
- Loop and exception handling detection

**Memory Efficiency:**
- Streaming file reading for large files
- Chunked processing to limit memory usage
- Lazy loading of parser modules

## Configuration

Control language support via configuration:

```bash
# Set file extensions to index
mcp-vector-search config set file_extensions '["py", "js", "ts", "cs"]'

# Auto-detect extensions from project
mcp-vector-search setup  # Automatically detects all supported languages
```

## Language Detection

MCP Vector Search automatically detects languages based on:

1. **File Extension**: Primary detection method
2. **Tree-sitter Parser**: Validates syntax for detected language
3. **Fallback**: Defaults to text parser if language not recognized

---

**[← Back to Reference Index](README.md)**
