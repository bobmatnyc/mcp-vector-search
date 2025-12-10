# Research: `analyze --quick` CLI Command Implementation

**Date**: 2024-12-10
**Issue**: #10 - Implement `analyze --quick` CLI command
**Related**: #11 - Console reporter for analysis results
**Researcher**: Claude (Research Agent)

## Executive Summary

This research provides a comprehensive analysis of implementing the `analyze --quick` CLI command for mcp-vector-search. The command will perform structural code analysis using existing metric collectors and output results to the console.

**Key Findings:**
- CLI uses Typer framework with modular command organization
- Analysis collectors already integrated into indexer (5 collectors available)
- `visualize` command provides pattern for subdirectory organization
- No console reporter exists yet (#11) - needs implementation
- All infrastructure exists; needs CLI wrapper + console reporter

**Recommended Approach:**
- Single-file implementation: `/src/mcp_vector_search/cli/commands/analyze.py`
- Integrate with existing collectors via indexer
- Rich console output using existing `output.py` utilities
- Support `--quick` flag for fast analysis (subset of files/metrics)

---

## 1. Existing CLI Patterns

### 1.1 Command Registration Pattern

**Location**: `/src/mcp_vector_search/cli/main.py`

The main CLI uses Typer's `app.add_typer()` for command groups:

```python
# Create main app
app = create_enhanced_typer(
    name="mcp-vector-search",
    help="CLI-first semantic code search with MCP integration",
)

# Import command modules
from .commands.search import search_app
from .commands.status import main as status_main
from .commands.visualize import app as visualize_app

# Register commands
app.add_typer(search_app, name="search", help="🔍 Search code semantically")
app.command("status", help="📊 Show project status")(status_main)
app.add_typer(visualize_app, name="visualize", help="📊 Visualize code relationships")
```

**Pattern for analyze command:**
```python
from .commands.analyze import analyze_app
app.add_typer(analyze_app, name="analyze", help="📈 Analyze code complexity")
```

### 1.2 Command Module Structure

**Simple Command** (`status.py` pattern):
```python
"""Status command for MCP Vector Search CLI."""
import typer
from ...core.project import ProjectManager
from ..output import console, print_error

# Create Typer app (optional for single command)
status_app = typer.Typer(help="Show project status")

@status_app.command()  # or direct function for app.command()
def main(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(...),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """📊 Show project status and statistics."""
    # Implementation
```

**Complex Command with Subcommands** (`search.py` pattern):
```python
"""Search command with subcommands."""
import typer

search_app = create_enhanced_typer(
    help="🔍 Search code semantically",
    invoke_without_command=True,
)

@search_app.callback(invoke_without_command=True)
def search_main(ctx: typer.Context, query: str | None = typer.Argument(None)) -> None:
    """Main search command."""
    if ctx.invoked_subcommand is None:
        # Default behavior
        run_search(query)

@search_app.command("interactive")
def interactive_search(ctx: typer.Context) -> None:
    """Interactive search mode."""
    pass

@search_app.command("history")
def show_history(ctx: typer.Context) -> None:
    """Show search history."""
    pass
```

**Subdirectory Organization** (`visualize/cli.py` pattern):
```
cli/commands/visualize/
├── __init__.py           # Exports 'app'
├── cli.py                # Main command definitions
├── graph_builder.py      # Logic module
├── server.py             # Server logic
└── exporters/
    ├── __init__.py
    ├── json_exporter.py
    └── html_exporter.py
```

### 1.3 Common CLI Options

**Standard Options Pattern:**
```python
project_root: Path | None = typer.Option(
    None,
    "--project-root",
    "-p",
    help="Project root directory (auto-detected if not specified)",
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    rich_help_panel="🔧 Global Options",
)

verbose: bool = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Show detailed information",
    rich_help_panel="📊 Display Options",
)

json_output: bool = typer.Option(
    False,
    "--json",
    help="Output results in JSON format",
    rich_help_panel="📊 Display Options",
)
```

### 1.4 Output Utilities

**Location**: `/src/mcp_vector_search/cli/output.py`

Available utilities:
```python
from ..output import (
    console,              # Rich Console instance
    print_error,          # Print error messages
    print_info,           # Print info messages
    print_tip,            # Print helpful tips
    print_json,           # Print JSON output
    print_search_results, # Format search results
)

# Rich formatting
console.print("[bold blue]Project Analysis[/bold blue]")
console.print(f"  Total Files: {total_files}")
console.print("[green]✓ Analysis complete[/green]")
```

---

## 2. Analysis Module Integration

### 2.1 Available Collectors

**Location**: `/src/mcp_vector_search/analysis/collectors/`

**Collector Interface** (`base.py`):
```python
class MetricCollector(ABC):
    """Abstract base class for metric collectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for collector."""
        pass

    @abstractmethod
    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process AST node during traversal."""
        pass

    @abstractmethod
    def finalize_function(self, node: Node, context: CollectorContext) -> dict[str, Any]:
        """Return metrics for completed function."""
        pass

    def reset(self) -> None:
        """Reset collector state for next function."""
        pass
```

**Available Collectors** (from `complexity.py`):

1. **CognitiveComplexityCollector**: Measures how hard code is to understand
2. **CyclomaticComplexityCollector**: Counts independent execution paths
3. **NestingDepthCollector**: Tracks maximum nesting level
4. **ParameterCountCollector**: Counts function parameters
5. **MethodCountCollector**: Counts methods in classes

**Multi-language Support:**
- Python, JavaScript, TypeScript, Java, C++, Go, Rust, PHP
- Node type mappings in `LANGUAGE_NODE_TYPES` dictionary

### 2.2 Metric Dataclasses

**Location**: `/src/mcp_vector_search/analysis/metrics.py`

**ChunkMetrics** (function/method level):
```python
@dataclass
class ChunkMetrics:
    cognitive_complexity: int = 0
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0
    lines_of_code: int = 0
    smells: list[str] = field(default_factory=list)
    complexity_grade: str = field(init=False, default="A")  # A-F scale

    def to_metadata(self) -> dict[str, Any]:
        """Flatten for ChromaDB storage."""
        pass
```

**FileMetrics** (file level):
```python
@dataclass
class FileMetrics:
    file_path: str
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    method_count: int = 0

    # Aggregated complexity
    total_complexity: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0

    chunks: list[ChunkMetrics] = field(default_factory=list)

    @property
    def health_score(self) -> float:
        """0.0-1.0 health score based on metrics."""
        pass
```

**ProjectMetrics** (project level):
```python
@dataclass
class ProjectMetrics:
    project_root: str
    analyzed_at: datetime
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0

    files: dict[str, FileMetrics] = field(default_factory=dict)

    avg_file_complexity: float = 0.0
    hotspots: list[str] = field(default_factory=list)  # Top 10 complex files

    def get_hotspots(self, limit: int = 10) -> list[FileMetrics]:
        """Return top N most complex files."""
        pass

    def to_summary(self) -> dict[str, Any]:
        """Generate summary for reporting."""
        pass
```

### 2.3 Indexer Integration

**Location**: `/src/mcp_vector_search/core/indexer.py`

The indexer already integrates collectors:

```python
class SemanticIndexer:
    def __init__(
        self,
        database: ChromaVectorDatabase,
        project_root: Path,
        config: ProjectConfig | None = None,
        collectors: list[MetricCollector] | None = None,
    ):
        # Initialize collectors (defaults to all complexity collectors)
        self.collectors = (
            collectors if collectors is not None else self._default_collectors()
        )

    def _default_collectors(self) -> list[MetricCollector]:
        """Return default collectors."""
        return [
            CognitiveComplexityCollector(),
            CyclomaticComplexityCollector(),
            NestingDepthCollector(),
            ParameterCountCollector(),
            MethodCountCollector(),
        ]

    def _collect_metrics(
        self, chunk: CodeChunk, source_code: bytes, language: str
    ) -> ChunkMetrics | None:
        """Collect metrics for a chunk using collectors."""
        # Implementation runs all collectors on AST
        pass
```

**During Indexing:**
```python
# In index_file() method:
if self.collectors:
    chunk_metrics = self._collect_metrics(chunk, source_code, language)
    # Metrics stored in ChromaDB metadata
```

**Key Insight**: Collectors run during indexing and metrics stored in ChromaDB metadata.

---

## 3. Requirements for Analyze Command

### 3.1 Functional Requirements

**Core Functionality:**
1. Analyze code complexity for current project or specified path
2. Support `--quick` flag for fast analysis (subset of metrics/files)
3. Output metrics to console in human-readable format
4. Support JSON output for programmatic use
5. Integrate with existing metric collectors

**Analysis Modes:**
- **Full Analysis** (default): Analyze all files, all metrics, detailed output
- **Quick Analysis** (`--quick`): Sample analysis for rapid feedback
  - Analyze subset of files (e.g., top 20% largest, random sample)
  - Core metrics only (cognitive complexity, cyclomatic complexity)
  - Summary statistics only (no per-file details)

### 3.2 Command Interface Design

**Proposed Command Structure:**

```bash
# Basic analysis (full)
mcp-vector-search analyze

# Quick analysis (fast mode)
mcp-vector-search analyze --quick

# Analyze specific directory
mcp-vector-search analyze --path src/

# Filter by language
mcp-vector-search analyze --language python

# JSON output
mcp-vector-search analyze --json

# Limit to top N hotspots
mcp-vector-search analyze --top 10

# Verbose output with per-file details
mcp-vector-search analyze --verbose

# Combined options
mcp-vector-search analyze --quick --language typescript --top 5
```

**Command Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project-root`, `-p` | Path | cwd | Project root directory |
| `--quick`, `-q` | Flag | False | Fast analysis mode (sample) |
| `--path` | Path | None | Analyze specific path (file or directory) |
| `--language`, `-l` | str | None | Filter by language (python, javascript, etc.) |
| `--top`, `-t` | int | 10 | Number of hotspots to show |
| `--threshold` | int | 10 | Complexity threshold for warnings (default: 10) |
| `--json` | Flag | False | Output JSON format |
| `--verbose`, `-v` | Flag | False | Show per-file details |
| `--export` | Path | None | Export full report to file |

### 3.3 Output Format Requirements

**Console Output** (feeds into #11 - console reporter):

```
📈 Code Complexity Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Summary
  Root: /Users/masa/Projects/mcp-vector-search
  Analyzed: 42 files (3,456 lines)
  Functions: 128 | Classes: 15

Complexity Distribution
  Grade A (0-5):    85 functions (66%)  ████████████████████
  Grade B (6-10):   32 functions (25%)  ████████
  Grade C (11-20):  10 functions (8%)   ██
  Grade D (21-30):   1 function (1%)
  Grade F (31+):     0 functions (0%)

Quality Metrics
  Average Complexity: 4.2 (Good)
  Health Score: 0.87 (Excellent)
  Files Needing Attention: 3

🔥 Complexity Hotspots (Top 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. src/core/indexer.py            Avg: 12.5 (C)  Max: 24 (D)
  2. src/cli/commands/search.py     Avg: 9.8 (B)   Max: 18 (C)
  3. src/parsers/python_parser.py   Avg: 8.3 (B)   Max: 15 (C)
  4. src/core/database.py           Avg: 7.1 (B)   Max: 12 (C)
  5. src/mcp/server.py              Avg: 6.8 (B)   Max: 10 (B)

💡 Recommendations
  • Refactor 1 function with grade D (complexity > 20)
  • Review 3 files with health score < 0.7
  • Consider splitting src/core/indexer.py (12.5 avg complexity)

✓ Analysis complete (42 files in 1.2s)
```

**Quick Mode Output** (subset):

```
📈 Quick Analysis (Sample: 10 files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project Summary
  Sampled: 10 of 42 files (24%)
  Estimated Functions: ~128
  Estimated Avg Complexity: 4.5

🔥 Top Hotspots (Sampled)
  1. src/core/indexer.py            Avg: 12.5 (C)
  2. src/cli/commands/search.py     Avg: 9.8 (B)
  3. src/parsers/python_parser.py   Avg: 8.3 (B)

💡 Run 'mcp-vector-search analyze' for full analysis
```

**JSON Output Format:**

```json
{
  "project_root": "/Users/masa/Projects/mcp-vector-search",
  "analyzed_at": "2024-12-10T10:30:00Z",
  "mode": "full",
  "summary": {
    "total_files": 42,
    "total_lines": 3456,
    "total_functions": 128,
    "total_classes": 15,
    "avg_complexity": 4.2,
    "health_score": 0.87
  },
  "distribution": {
    "A": 85,
    "B": 32,
    "C": 10,
    "D": 1,
    "F": 0
  },
  "hotspots": [
    {
      "file": "src/core/indexer.py",
      "avg_complexity": 12.5,
      "max_complexity": 24,
      "grade": "C",
      "health_score": 0.65,
      "functions": 15
    }
  ],
  "recommendations": [
    "Refactor 1 function with grade D (complexity > 20)",
    "Review 3 files with health score < 0.7"
  ]
}
```

---

## 4. Implementation Recommendations

### 4.1 File Organization

**Recommended: Single File Implementation**

```
src/mcp_vector_search/cli/commands/analyze.py
```

**Rationale:**
- Command is focused and self-contained
- No need for subdirectory complexity (unlike `visualize` with exporters/server)
- Easier maintenance and testing
- Can refactor to subdirectory later if needed

**Alternative: Subdirectory (if reporter grows complex)**

```
src/mcp_vector_search/cli/commands/analyze/
├── __init__.py           # Exports analyze_app
├── cli.py                # Command definitions
├── reporter.py           # Console reporter (#11)
└── analyzer.py           # Analysis logic
```

### 4.2 Command Structure

**Proposed Implementation** (`analyze.py`):

```python
"""Analyze command for code complexity analysis."""

import asyncio
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ...analysis.collectors import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
)
from ...analysis.metrics import ProjectMetrics, FileMetrics
from ...core.project import ProjectManager
from ..output import console, print_error, print_info, print_json

# Create analyze app
analyze_app = typer.Typer(
    help="📈 Analyze code complexity and quality metrics",
    invoke_without_command=True,
)


@analyze_app.callback(invoke_without_command=True)
def analyze_main(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        rich_help_panel="🔧 Global Options",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        "-q",
        help="Quick analysis mode (sample subset of files)",
        rich_help_panel="⚡ Analysis Options",
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Analyze specific path (file or directory)",
        exists=True,
        rich_help_panel="🔍 Filters",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Filter by language (python, javascript, typescript, etc.)",
        rich_help_panel="🔍 Filters",
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-t",
        help="Number of hotspots to show",
        min=1,
        max=50,
        rich_help_panel="📊 Display Options",
    ),
    threshold: int = typer.Option(
        10,
        "--threshold",
        help="Complexity threshold for warnings",
        min=1,
        rich_help_panel="⚠️ Thresholds",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed per-file metrics",
        rich_help_panel="📊 Display Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
        rich_help_panel="📊 Display Options",
    ),
    export: Path | None = typer.Option(
        None,
        "--export",
        help="Export full report to file",
        rich_help_panel="💾 Export",
    ),
) -> None:
    """📈 Analyze code complexity and quality metrics.

    Performs structural analysis of your codebase using complexity metrics
    to identify hotspots and provide quality recommendations.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Full analysis:[/green]
        $ mcp-vector-search analyze

    [green]Quick analysis (fast mode):[/green]
        $ mcp-vector-search analyze --quick

    [green]Analyze specific directory:[/green]
        $ mcp-vector-search analyze --path src/

    [bold cyan]Advanced Options:[/bold cyan]

    [green]Filter by language:[/green]
        $ mcp-vector-search analyze --language python

    [green]Show top hotspots:[/green]
        $ mcp-vector-search analyze --top 5

    [green]Export to JSON:[/green]
        $ mcp-vector-search analyze --json > analysis.json

    [dim]💡 Tip: Use --quick for rapid feedback during development.[/dim]
    """
    try:
        root = project_root or Path.cwd()

        asyncio.run(
            run_analysis(
                project_root=root,
                quick=quick,
                path=path,
                language=language,
                top=top,
                threshold=threshold,
                verbose=verbose,
                json_output=json_output,
                export=export,
            )
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(1)


async def run_analysis(
    project_root: Path,
    quick: bool = False,
    path: Path | None = None,
    language: str | None = None,
    top: int = 10,
    threshold: int = 10,
    verbose: bool = False,
    json_output: bool = False,
    export: Path | None = None,
) -> None:
    """Run code analysis."""
    # Implementation here
    # 1. Load project configuration
    # 2. Analyze files (quick mode: sample, full mode: all)
    # 3. Collect metrics using collectors
    # 4. Aggregate to ProjectMetrics
    # 5. Output using console reporter or JSON
    pass


def print_analysis_results(
    metrics: ProjectMetrics,
    top: int = 10,
    threshold: int = 10,
    verbose: bool = False,
) -> None:
    """Print analysis results to console (implements #11 reporter)."""
    # Implementation of console reporter
    # - Summary section
    # - Distribution chart
    # - Hotspots table
    # - Recommendations
    pass


if __name__ == "__main__":
    analyze_app()
```

### 4.3 Integration with Existing Systems

**Leverage Indexer's Collector Integration:**

```python
from ...core.indexer import SemanticIndexer
from ...analysis.collectors import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
)

# Quick mode: use subset of collectors
if quick:
    collectors = [
        CognitiveComplexityCollector(),
        CyclomaticComplexityCollector(),
    ]
else:
    # Full mode: use all default collectors
    collectors = None  # Uses indexer defaults

# Create indexer with collectors
indexer = SemanticIndexer(
    database=database,
    project_root=project_root,
    config=config,
    collectors=collectors,
)

# Metrics already collected during indexing
# Retrieve from ChromaDB metadata
```

**Alternative: Direct Analysis (without re-indexing):**

```python
from ...parsers.registry import get_parser_registry
from ...analysis.collectors import CollectorContext

# Parse file and run collectors directly
parser_registry = get_parser_registry()
parser = parser_registry.get_parser(language)

source_code = file_path.read_bytes()
chunks = parser.parse_file(file_path, source_code)

# Run collectors on each chunk
for chunk in chunks:
    context = CollectorContext(
        file_path=str(file_path),
        source_code=source_code,
        language=language,
    )

    for collector in collectors:
        collector.collect_node(chunk.node, context, depth=0)
        metrics = collector.finalize_function(chunk.node, context)
        collector.reset()
```

### 4.4 Quick Mode Implementation

**Strategy 1: Random Sampling**
- Select random 10-20% of files
- Analyze with all collectors
- Extrapolate project-wide metrics

**Strategy 2: Smart Sampling**
- Select largest files (top 20% by LOC)
- Select most recently modified
- Analyze core directories only (src/, lib/)

**Strategy 3: Reduced Metrics** (Recommended)
- Analyze all files
- Use only 2 collectors (cognitive + cyclomatic)
- Skip detailed per-function metrics

```python
if quick:
    # Reduced collectors for speed
    collectors = [
        CognitiveComplexityCollector(),
        CyclomaticComplexityCollector(),
    ]
    # Sample files (20%)
    sample_size = max(1, int(len(all_files) * 0.2))
    files_to_analyze = random.sample(all_files, sample_size)
else:
    # Full analysis
    collectors = None  # All default collectors
    files_to_analyze = all_files
```

### 4.5 Console Reporter Design (#11)

**Reporter Interface:**

```python
class ConsoleReporter:
    """Console reporter for analysis results."""

    def __init__(self, console: Console):
        self.console = console

    def print_summary(self, metrics: ProjectMetrics) -> None:
        """Print project summary section."""
        self.console.print("[bold blue]Project Summary[/bold blue]")
        # ...

    def print_distribution(self, metrics: ProjectMetrics) -> None:
        """Print complexity distribution chart."""
        # Use Rich Table or Panel
        pass

    def print_hotspots(
        self,
        metrics: ProjectMetrics,
        top: int = 10
    ) -> None:
        """Print complexity hotspots table."""
        # Use Rich Table
        pass

    def print_recommendations(
        self,
        metrics: ProjectMetrics,
        threshold: int = 10
    ) -> None:
        """Print actionable recommendations."""
        # Analyze metrics and suggest improvements
        pass
```

---

## 5. Best Practices and Design Patterns

### 5.1 CLI Design Best Practices

**From Research** ([Typer Best Practices](https://www.projectrules.ai/rules/typer), [CLI Design Guide](https://realpython.com/python-typer-cli/)):

1. **Stateless Commands**: Design commands to be stateless, pass all data as arguments
2. **Rich Help Messages**: Provide detailed descriptions with examples
3. **Default Values**: Offer sensible defaults to reduce user input
4. **Progressive Disclosure**: Simple by default, power through options
5. **Validation**: Validate input early with clear error messages
6. **Graceful Error Handling**: Use try/except with informative messages
7. **Async Support**: Use `asyncio.run()` for async operations

### 5.2 Typer Patterns from Codebase

**Rich Help Panels:**
```python
verbose: bool = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Show detailed information",
    rich_help_panel="📊 Display Options",  # Groups related options
)
```

**Context Sharing:**
```python
@app.callback()
def main(ctx: typer.Context, project_root: Path | None = None):
    ctx.ensure_object(dict)
    ctx.obj["project_root"] = project_root

@app.command()
def command(ctx: typer.Context):
    root = ctx.obj.get("project_root") or Path.cwd()
```

**Async Execution:**
```python
def command_main(...) -> None:
    """Sync wrapper for async logic."""
    try:
        asyncio.run(run_command_async(...))
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print_error(f"Command failed: {e}")
        raise typer.Exit(1)
```

### 5.3 Output Formatting Patterns

**Rich Table Example** (from status.py pattern):
```python
from rich.table import Table

table = Table(title="🔥 Complexity Hotspots")
table.add_column("Rank", style="cyan", no_wrap=True)
table.add_column("File", style="magenta")
table.add_column("Avg", justify="right", style="yellow")
table.add_column("Max", justify="right", style="red")
table.add_column("Grade", justify="center")

for i, file_metrics in enumerate(hotspots, 1):
    table.add_row(
        str(i),
        file_metrics.file_path,
        f"{file_metrics.avg_complexity:.1f}",
        str(file_metrics.max_complexity),
        file_metrics.grade,
    )

console.print(table)
```

---

## 6. Testing Considerations

### 6.1 Test Structure

**Test File**: `/tests/unit/test_cli_analyze.py`

```python
"""Tests for analyze CLI command."""

import pytest
from typer.testing import CliRunner
from mcp_vector_search.cli.main import app

runner = CliRunner()

def test_analyze_basic():
    """Test basic analyze command."""
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 0
    assert "Project Summary" in result.stdout

def test_analyze_quick():
    """Test quick mode."""
    result = runner.invoke(app, ["analyze", "--quick"])
    assert result.exit_code == 0
    assert "Quick Analysis" in result.stdout

def test_analyze_json_output():
    """Test JSON output format."""
    result = runner.invoke(app, ["analyze", "--json"])
    assert result.exit_code == 0
    # Validate JSON structure
```

### 6.2 Integration Tests

**Test File**: `/tests/integration/test_analyze_integration.py`

```python
"""Integration tests for analyze command."""

def test_analyze_real_project(tmp_path):
    """Test analysis on real project."""
    # Create test project structure
    # Run analyze command
    # Verify metrics collected
```

---

## 7. Implementation Checklist

### Phase 1: Basic Command Structure
- [ ] Create `/src/mcp_vector_search/cli/commands/analyze.py`
- [ ] Define `analyze_app` with Typer
- [ ] Implement `analyze_main()` callback with all options
- [ ] Register command in `main.py`
- [ ] Add basic help documentation

### Phase 2: Analysis Logic
- [ ] Implement `run_analysis()` async function
- [ ] Integrate with `ProjectManager` for config loading
- [ ] Implement file discovery and filtering (language, path)
- [ ] Integrate collectors (use indexer pattern or direct)
- [ ] Aggregate metrics into `ProjectMetrics`

### Phase 3: Console Reporter (#11)
- [ ] Implement `print_summary()` for project overview
- [ ] Implement `print_distribution()` for grade distribution
- [ ] Implement `print_hotspots()` with Rich table
- [ ] Implement `print_recommendations()` with actionable tips
- [ ] Add progress indicators for long analyses

### Phase 4: Quick Mode
- [ ] Implement file sampling strategy
- [ ] Implement collector subset selection
- [ ] Add "quick analysis" banner to output
- [ ] Add tip to run full analysis

### Phase 5: Output Formats
- [ ] Implement JSON output format
- [ ] Implement export to file (--export option)
- [ ] Add verbose mode with per-file details

### Phase 6: Testing
- [ ] Unit tests for CLI command
- [ ] Integration tests with real projects
- [ ] Test all output formats (console, JSON, export)
- [ ] Test error handling

### Phase 7: Documentation
- [ ] Update CLI help with examples
- [ ] Add to README.md
- [ ] Create user guide in docs/
- [ ] Add to CHANGELOG.md

---

## 8. Dependencies

### Existing Dependencies (Already Available)
- ✅ Typer - CLI framework
- ✅ Rich - Console formatting
- ✅ Loguru - Logging
- ✅ Analysis collectors - Metric collection
- ✅ Metric dataclasses - Data structures
- ✅ ProjectManager - Configuration

### New Dependencies (None Required)
- All infrastructure exists
- No new external dependencies needed

---

## 9. Related Issues and PRs

**Related Issues:**
- #10 - Implement `analyze --quick` CLI command (this research)
- #11 - Console reporter for analysis results (implementation target)
- #9 - ChromaDB metadata schema extension (metrics storage)

**Related Milestones:**
- Structural Code Analysis (Phase 1) - Current milestone

---

## 10. Next Steps

### Immediate Actions
1. **Create analyze.py**: Implement basic command structure
2. **Implement Console Reporter**: Build reporter for #11
3. **Integrate Collectors**: Use existing collector infrastructure
4. **Add Tests**: Unit + integration tests

### Follow-up Work
1. **Export Formats**: Add markdown/CSV export
2. **Trend Analysis**: Track metrics over time
3. **CI Integration**: Add to GitHub Actions
4. **VS Code Extension**: Integrate with editor

---

## Appendix A: Code Examples

### Example 1: Basic Analyze Command

```python
"""Minimal analyze command implementation."""

import asyncio
from pathlib import Path
import typer
from ...core.project import ProjectManager
from ...analysis.metrics import ProjectMetrics
from ..output import console, print_error

analyze_app = typer.Typer(help="📈 Analyze code complexity")

@analyze_app.callback(invoke_without_command=True)
def analyze_main(
    ctx: typer.Context,
    quick: bool = typer.Option(False, "--quick", "-q"),
) -> None:
    """Analyze code complexity."""
    asyncio.run(run_analysis(Path.cwd(), quick))

async def run_analysis(project_root: Path, quick: bool) -> None:
    """Run analysis."""
    # Load project
    pm = ProjectManager(project_root)
    if not pm.is_initialized():
        print_error("Project not initialized")
        return

    # Analyze (placeholder)
    console.print("[bold blue]📈 Code Analysis[/bold blue]")
    console.print(f"  Mode: {'Quick' if quick else 'Full'}")
    console.print("[green]✓ Analysis complete[/green]")
```

### Example 2: Console Reporter

```python
"""Console reporter for analysis results."""

from rich.console import Console
from rich.table import Table
from ...analysis.metrics import ProjectMetrics

class AnalysisReporter:
    """Reporter for analysis results."""

    def __init__(self, console: Console):
        self.console = console

    def report(self, metrics: ProjectMetrics, top: int = 10) -> None:
        """Print full analysis report."""
        self._print_header()
        self._print_summary(metrics)
        self._print_distribution(metrics)
        self._print_hotspots(metrics, top)
        self._print_recommendations(metrics)

    def _print_header(self) -> None:
        """Print report header."""
        self.console.print("\n[bold blue]📈 Code Complexity Analysis[/bold blue]")
        self.console.print("━" * 60)

    def _print_summary(self, metrics: ProjectMetrics) -> None:
        """Print project summary."""
        self.console.print("\n[bold]Project Summary[/bold]")
        self.console.print(f"  Root: {metrics.project_root}")
        self.console.print(f"  Files: {metrics.total_files}")
        self.console.print(f"  Lines: {metrics.total_lines:,}")
        self.console.print(f"  Functions: {metrics.total_functions}")
        self.console.print(f"  Classes: {metrics.total_classes}")

    def _print_hotspots(self, metrics: ProjectMetrics, top: int) -> None:
        """Print complexity hotspots."""
        hotspots = metrics.get_hotspots(limit=top)

        table = Table(title=f"\n🔥 Complexity Hotspots (Top {top})")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("File", style="magenta")
        table.add_column("Avg", justify="right", style="yellow", width=8)
        table.add_column("Max", justify="right", style="red", width=8)

        for i, fm in enumerate(hotspots, 1):
            table.add_row(
                str(i),
                fm.file_path,
                f"{fm.avg_complexity:.1f}",
                str(fm.max_complexity),
            )

        self.console.print(table)
```

---

## Appendix B: File Paths Reference

### CLI Files
- `/src/mcp_vector_search/cli/main.py` - Main CLI app
- `/src/mcp_vector_search/cli/commands/status.py` - Status command (pattern)
- `/src/mcp_vector_search/cli/commands/search.py` - Search command (pattern)
- `/src/mcp_vector_search/cli/commands/visualize/cli.py` - Subdirectory pattern
- `/src/mcp_vector_search/cli/output.py` - Output utilities

### Analysis Files
- `/src/mcp_vector_search/analysis/collectors/base.py` - Collector interface
- `/src/mcp_vector_search/analysis/collectors/complexity.py` - Complexity collectors
- `/src/mcp_vector_search/analysis/metrics.py` - Metric dataclasses
- `/src/mcp_vector_search/core/indexer.py` - Indexer with collector integration

---

## Sources

- [7 Python CLI Libraries for Building Professional Command-Line Tools](https://dev.to/aaravjoshi/7-python-cli-libraries-for-building-professional-command-line-tools-2024-guide-7ad)
- [Typer CLI Best Practices and Coding Standards](https://www.projectrules.ai/rules/typer)
- [Build a Command-Line To-Do App With Python and Typer](https://realpython.com/python-typer-cli/)
- [Building Powerful CLIs with Click and Typer](https://procodebase.com/article/building-powerful-command-line-interfaces-with-click-and-typer-in-python)

---

**Research Complete**: 2024-12-10
**Next**: Proceed with implementation following recommendations
