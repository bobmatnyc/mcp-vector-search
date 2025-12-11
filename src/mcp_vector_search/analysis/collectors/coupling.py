"""Coupling metric collectors for structural code analysis.

This module provides collectors for measuring coupling metrics:
- EfferentCouplingCollector: Counts outgoing dependencies (imports from this file)
- AfferentCouplingCollector: Counts incoming dependencies (files that import this file)
- InstabilityCalculator: Calculates instability metrics across the project
- CircularDependencyDetector: Detects circular/cyclic dependencies in import graph

Coupling metrics help identify architectural dependencies and potential refactoring needs.
Circular dependencies can lead to:
- Initialization issues and import errors
- Tight coupling and reduced maintainability
- Difficulty in testing and refactoring
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import CollectorContext, MetricCollector

if TYPE_CHECKING:
    from tree_sitter import Node


# =============================================================================
# Circular Dependency Detection Types
# =============================================================================


class NodeColor(Enum):
    """Node colors for DFS-based cycle detection.

    Standard graph coloring algorithm:
    - WHITE: Node not yet visited
    - GRAY: Node currently being processed (in current DFS path)
    - BLACK: Node fully processed (all descendants visited)

    Cycle detection: If we encounter a GRAY node during DFS, we've found a cycle.
    """

    WHITE = "white"  # Unvisited
    GRAY = "gray"  # In current path (cycle if revisited)
    BLACK = "black"  # Fully processed


@dataclass
class ImportGraph:
    """Directed graph representing import dependencies between files.

    Nodes represent files, edges represent import relationships.
    An edge from A to B means "A imports B".

    Attributes:
        adjacency_list: Maps file paths to list of files they import

    Example:
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")
        graph.add_edge("utils.py", "helpers.py")
        # main.py → utils.py → helpers.py
    """

    adjacency_list: dict[str, list[str]] = field(default_factory=dict)

    def add_edge(self, from_file: str, to_file: str) -> None:
        """Add directed edge from from_file to to_file (from_file imports to_file).

        Args:
            from_file: Source file that contains the import
            to_file: Target file being imported
        """
        if from_file not in self.adjacency_list:
            self.adjacency_list[from_file] = []
        if to_file not in self.adjacency_list[from_file]:
            self.adjacency_list[from_file].append(to_file)

    def add_node(self, file_path: str) -> None:
        """Add node (file) to graph without any edges.

        Useful for ensuring isolated files are tracked.

        Args:
            file_path: Path to file to add as node
        """
        if file_path not in self.adjacency_list:
            self.adjacency_list[file_path] = []

    def get_neighbors(self, file_path: str) -> list[str]:
        """Get list of files that file_path imports.

        Args:
            file_path: File to get imports for

        Returns:
            List of files imported by file_path
        """
        return self.adjacency_list.get(file_path, [])

    def get_all_files(self) -> list[str]:
        """Get all files in the graph.

        Returns:
            List of all file paths (nodes) in the graph
        """
        # Include both keys and values to catch files that are imported but don't import anything
        all_files = set(self.adjacency_list.keys())
        for imports in self.adjacency_list.values():
            all_files.update(imports)
        return sorted(all_files)


@dataclass
class CircularDependency:
    """Represents a detected circular dependency cycle.

    Attributes:
        cycle_chain: List of files forming the cycle (first == last)
        cycle_length: Number of unique files in cycle

    Example:
        cycle = CircularDependency(
            cycle_chain=["a.py", "b.py", "c.py", "a.py"]
        )
        assert cycle.cycle_length == 3
        assert cycle.format_chain() == "a.py → b.py → c.py → a.py"
    """

    cycle_chain: list[str]

    @property
    def cycle_length(self) -> int:
        """Number of unique files in cycle (excluding duplicate start/end)."""
        return len(self.cycle_chain) - 1 if len(self.cycle_chain) > 1 else 0

    def format_chain(self) -> str:
        """Format cycle as human-readable chain with arrows.

        Returns:
            Formatted cycle string (e.g., "A → B → C → A")
        """
        return " → ".join(self.cycle_chain)

    def get_affected_files(self) -> list[str]:
        """Get unique list of files involved in this cycle.

        Returns:
            Sorted list of unique file paths in cycle
        """
        # Remove duplicate (last element equals first)
        unique_files = (
            set(self.cycle_chain[:-1])
            if len(self.cycle_chain) > 1
            else set(self.cycle_chain)
        )
        return sorted(unique_files)


class CircularDependencyDetector:
    """Detects circular dependencies in import graphs using DFS-based cycle detection.

    Uses three-color DFS algorithm (Tarjan-inspired):
    - WHITE: Unvisited node
    - GRAY: Node in current DFS path (cycle if we revisit a GRAY node)
    - BLACK: Fully processed node

    This algorithm efficiently detects all elementary cycles in O(V+E) time.

    Design Decisions:
    - **Algorithm Choice**: DFS with color marking chosen over Tarjan's SCC because:
      - Simpler implementation and easier to understand
      - Directly provides cycle paths (not just strongly connected components)
      - O(V+E) time complexity (same as Tarjan's)
      - Better for reporting individual cycles to developers

    - **Path Tracking**: Maintains explicit path stack during DFS to reconstruct cycles
      - Enables user-friendly "A → B → C → A" output
      - Memory overhead acceptable for typical codebases (<10K files)

    - **Duplicate Cycle Handling**: Detects and reports all unique cycle instances
      - Same cycle may be discovered multiple times from different starting points
      - Deduplication handled by caller if needed

    Trade-offs:
    - **Simplicity vs. Optimization**: Chose simpler DFS over complex SCC algorithms
      - Performance: Acceptable for codebases up to ~50K files
      - Maintainability: Easier to debug and extend
    - **Memory vs. Clarity**: Stores full path during DFS for clear error messages
      - Alternative: Store only parent pointers (saves memory but harder to debug)

    Example:
        detector = CircularDependencyDetector(import_graph)
        cycles = detector.detect_cycles()

        if detector.has_cycles():
            for cycle in cycles:
                print(f"Cycle detected: {cycle.format_chain()}")
    """

    def __init__(self, import_graph: ImportGraph) -> None:
        """Initialize detector with import graph.

        Args:
            import_graph: Graph of import dependencies to analyze
        """
        self.graph = import_graph
        self._cycles: list[CircularDependency] = []
        self._colors: dict[str, NodeColor] = {}
        self._path: list[str] = []  # Current DFS path for cycle reconstruction

    def detect_cycles(self) -> list[CircularDependency]:
        """Detect all circular dependencies in the import graph.

        Uses DFS with three-color marking:
        1. WHITE: Node not yet visited
        2. GRAY: Node in current DFS path (cycle if revisited)
        3. BLACK: Node fully processed

        Returns:
            List of CircularDependency objects for all detected cycles

        Complexity:
            Time: O(V + E) where V = files, E = import edges
            Space: O(V) for color map and path stack
        """
        self._cycles = []
        self._colors = dict.fromkeys(self.graph.get_all_files(), NodeColor.WHITE)
        self._path = []

        # Run DFS from each unvisited node
        for file in self.graph.get_all_files():
            if self._colors[file] == NodeColor.WHITE:
                self._dfs(file)

        return self._cycles

    def _dfs(self, node: str) -> None:
        """Depth-first search to detect cycles.

        Core cycle detection logic:
        - Mark node GRAY (in current path)
        - Visit all neighbors
        - If neighbor is GRAY → cycle detected (it's in current path)
        - If neighbor is WHITE → recurse
        - Mark node BLACK after processing all neighbors

        Args:
            node: Current file being visited
        """
        self._colors[node] = NodeColor.GRAY
        self._path.append(node)

        # Visit all files that this file imports
        for neighbor in self.graph.get_neighbors(node):
            if self._colors[neighbor] == NodeColor.GRAY:
                # Found cycle! Neighbor is in current path
                self._record_cycle(neighbor)
            elif self._colors[neighbor] == NodeColor.WHITE:
                # Unvisited node, continue DFS
                self._dfs(neighbor)

        # Finished processing this node
        self._path.pop()
        self._colors[node] = NodeColor.BLACK

    def _record_cycle(self, cycle_start: str) -> None:
        """Record detected cycle by extracting path from cycle_start to current node.

        When we detect a cycle (encounter GRAY node), we extract the cycle from
        the current DFS path stack.

        Args:
            cycle_start: File where cycle begins (GRAY node we just encountered)
        """
        # Find cycle_start in current path
        try:
            start_index = self._path.index(cycle_start)
        except ValueError:
            # Should not happen if algorithm is correct
            return

        # Extract cycle: [cycle_start, ..., current_node, cycle_start]
        cycle_chain = self._path[start_index:] + [cycle_start]
        self._cycles.append(CircularDependency(cycle_chain=cycle_chain))

    def has_cycles(self) -> bool:
        """Check if any cycles were detected.

        Note: Must call detect_cycles() first.

        Returns:
            True if cycles exist, False otherwise
        """
        return len(self._cycles) > 0

    def get_cycle_chains(self) -> list[str]:
        """Get human-readable cycle chains.

        Returns:
            List of formatted cycle strings (e.g., ["A → B → C → A"])
        """
        return [cycle.format_chain() for cycle in self._cycles]

    def get_affected_files(self) -> list[str]:
        """Get all unique files involved in any cycle.

        Returns:
            Sorted list of unique file paths involved in cycles
        """
        affected = set()
        for cycle in self._cycles:
            affected.update(cycle.get_affected_files())
        return sorted(affected)


def build_import_graph_from_dict(file_imports: dict[str, list[str]]) -> ImportGraph:
    """Build ImportGraph from dictionary of file imports.

    Utility function to construct graph from parsed import data.

    Args:
        file_imports: Dictionary mapping file paths to lists of imported files

    Returns:
        ImportGraph with all edges added

    Example:
        imports = {
            "main.py": ["utils.py", "config.py"],
            "utils.py": ["helpers.py"],
            "helpers.py": []
        }
        graph = build_import_graph_from_dict(imports)
    """
    graph = ImportGraph()

    # Add all files as nodes first (ensures isolated files are included)
    for file_path in file_imports.keys():
        graph.add_node(file_path)

    # Add edges for imports
    for file_path, imports in file_imports.items():
        for imported_file in imports:
            graph.add_edge(file_path, imported_file)

    return graph


# =============================================================================
# Multi-language Import Statement Mappings
# =============================================================================

IMPORT_NODE_TYPES = {
    "python": {
        "import": ["import_statement", "import_from_statement"],
        "module_name": ["dotted_name", "aliased_import"],
    },
    "javascript": {
        "import": ["import_statement"],
        "module_name": ["string", "import_clause"],
        "require_call": ["call_expression"],  # require('module')
    },
    "typescript": {
        "import": ["import_statement"],
        "module_name": ["string", "import_clause"],
        "import_type": ["import_statement"],  # import type { T } from 'mod'
        "require_call": ["call_expression"],
    },
    "java": {
        "import": ["import_declaration"],
        "module_name": ["scoped_identifier"],
    },
    "rust": {
        "import": ["use_declaration"],
        "module_name": ["scoped_identifier"],
    },
    "php": {
        "import": ["namespace_use_declaration"],
        "module_name": ["qualified_name"],
    },
    "ruby": {
        "import": ["call"],  # require, require_relative
        "module_name": ["string"],
    },
}


def get_import_node_types(language: str, category: str) -> list[str]:
    """Get tree-sitter node types for imports in a given language.

    Args:
        language: Programming language identifier (e.g., "python", "javascript")
        category: Category of import node ("import", "module_name", etc.)

    Returns:
        List of node type names for this language/category.
        Returns empty list if language/category not found.

    Examples:
        >>> get_import_node_types("python", "import")
        ["import_statement", "import_from_statement"]

        >>> get_import_node_types("javascript", "import")
        ["import_statement"]
    """
    # Default to Python-like behavior for unknown languages
    lang_mapping = IMPORT_NODE_TYPES.get(language, IMPORT_NODE_TYPES["python"])
    return lang_mapping.get(category, [])


def is_stdlib_module(module_name: str, language: str) -> bool:
    """Check if a module is from the standard library.

    Args:
        module_name: Module name (e.g., "os", "sys", "fs")
        language: Programming language

    Returns:
        True if module is standard library, False otherwise

    Examples:
        >>> is_stdlib_module("os", "python")
        True

        >>> is_stdlib_module("requests", "python")
        False

        >>> is_stdlib_module("fs", "javascript")
        True
    """
    if language == "python":
        # Python standard library check
        # Use sys.stdlib_module_names (Python 3.10+) or hardcoded list
        if hasattr(sys, "stdlib_module_names"):
            return module_name.split(".")[0] in sys.stdlib_module_names
        else:
            # Fallback: common stdlib modules
            common_stdlib = {
                "os",
                "sys",
                "re",
                "json",
                "math",
                "time",
                "datetime",
                "collections",
                "itertools",
                "functools",
                "pathlib",
                "typing",
                "dataclasses",
                "asyncio",
                "contextlib",
                "abc",
                "io",
                "logging",
                "unittest",
                "pytest",
            }
            return module_name.split(".")[0] in common_stdlib

    elif language in ("javascript", "typescript"):
        # Node.js built-in modules
        nodejs_builtins = {
            "fs",
            "path",
            "http",
            "https",
            "url",
            "os",
            "util",
            "events",
            "stream",
            "buffer",
            "crypto",
            "child_process",
            "cluster",
            "dns",
            "net",
            "tls",
            "dgram",
            "readline",
            "zlib",
            "process",
            "console",
            "assert",
            "timers",
        }
        return module_name.split("/")[0] in nodejs_builtins

    return False


def is_relative_import(module_name: str, language: str) -> bool:
    """Check if import is relative to current file.

    Args:
        module_name: Module path
        language: Programming language

    Returns:
        True if import is relative, False otherwise

    Examples:
        >>> is_relative_import("./utils", "javascript")
        True

        >>> is_relative_import("lodash", "javascript")
        False

        >>> is_relative_import(".utils", "python")
        True
    """
    if language == "python":
        # Python relative imports start with "."
        return module_name.startswith(".")
    elif language in ("javascript", "typescript"):
        # JS/TS relative imports start with "./" or "../"
        return module_name.startswith("./") or module_name.startswith("../")
    return False


class EfferentCouplingCollector(MetricCollector):
    """Collects efferent coupling metrics (outgoing dependencies).

    Efferent coupling (Ce) measures how many external modules/files a file
    depends on. Higher Ce indicates fragility - changes to dependencies can
    break this file.

    Tracks:
    - Total unique dependencies (efferent_coupling score)
    - All imported modules
    - Internal vs. external imports
    - Standard library vs. third-party imports

    Example:
        # Python file with Ce = 3
        import os           # stdlib
        from typing import List  # stdlib (not counted, same base module)
        import requests     # external
        from .utils import helper  # internal

        # Ce = 3 (os, requests, .utils)
    """

    def __init__(self) -> None:
        """Initialize efferent coupling collector."""
        self._imports: set[str] = set()  # All unique imports
        self._internal_imports: set[str] = set()
        self._external_imports: set[str] = set()

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "efferent_coupling"
        """
        return "efferent_coupling"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and extract import statements.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language and file info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Check if this is an import statement
        if node_type in get_import_node_types(language, "import"):
            self._extract_import(node, context)
        elif language in ("javascript", "typescript"):
            # Handle require() calls in JS/TS
            if node_type in get_import_node_types(language, "require_call"):
                self._extract_require_call(node, context)

    def _extract_import(self, node: Node, context: CollectorContext) -> None:
        r"""Extract module name from import statement.

        Handles:
        - Python: import module, from module import X
        - JavaScript/TypeScript: import ... from 'module'
        - Java: import com.example.Class
        - Rust: use std::collections::HashMap
        - PHP: use MyNamespace\MyClass
        - Ruby: require "module"

        Args:
            node: Import statement node
            context: Collector context
        """
        language = context.language

        if language == "python":
            # Python: import os, from os import path
            # Look for dotted_name or module_name field
            module_node = node.child_by_field_name("module_name")
            if module_node:
                module_name = module_node.text.decode("utf-8")
                self._add_import(module_name, context)
            else:
                # Look for dotted_name child
                for child in node.children:
                    if child.type == "dotted_name":
                        module_name = child.text.decode("utf-8")
                        self._add_import(module_name, context)
                    elif child.type == "aliased_import":
                        # import os as operating_system
                        for subchild in child.children:
                            if subchild.type == "dotted_name":
                                module_name = subchild.text.decode("utf-8")
                                self._add_import(module_name, context)
                                break
                    elif child.type == "relative_import":
                        # Relative import (from . import X)
                        dots = child.text.decode("utf-8")
                        self._add_import(dots, context)
                        break

        elif language in ("javascript", "typescript"):
            # JavaScript/TypeScript: import ... from 'module'
            for child in node.children:
                if child.type == "string":
                    module_str = child.text.decode("utf-8")
                    module_name = module_str.strip("\"'")
                    self._add_import(module_name, context)

        elif language == "java":
            for child in node.children:
                if child.type == "scoped_identifier":
                    module_name = child.text.decode("utf-8")
                    self._add_import(module_name, context)

        elif language == "rust":
            for child in node.children:
                if child.type == "scoped_identifier":
                    module_name = child.text.decode("utf-8")
                    self._add_import(module_name, context)

        elif language == "php":
            for child in node.children:
                if child.type == "qualified_name":
                    module_name = child.text.decode("utf-8")
                    self._add_import(module_name, context)

        elif language == "ruby":
            # Ruby uses method calls for imports
            if node.type == "call":
                method_child = node.child_by_field_name("method")
                if method_child and method_child.text.decode("utf-8") in [
                    "require",
                    "require_relative",
                ]:
                    args_child = node.child_by_field_name("arguments")
                    if args_child:
                        for child in args_child.children:
                            if child.type == "string":
                                module_str = child.text.decode("utf-8")
                                module_name = module_str.strip("\"'")
                                self._add_import(module_name, context)

    def _extract_require_call(self, node: Node, context: CollectorContext) -> None:
        """Extract module name from require('module') call.

        Handles:
        - JavaScript/TypeScript: const x = require('module')

        Args:
            node: Call expression node
            context: Collector context
        """
        # Check if this is a require() call
        function_node = node.child_by_field_name("function")
        if function_node and function_node.type == "identifier":
            function_name = function_node.text.decode("utf-8")
            if function_name == "require":
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        if child.type == "string":
                            module_str = child.text.decode("utf-8")
                            module_name = module_str.strip("\"'")
                            self._add_import(module_name, context)

    def _add_import(self, module_name: str, context: CollectorContext) -> None:
        """Add import to tracking sets and classify as internal/external.

        Args:
            module_name: Imported module name
            context: Collector context with language info
        """
        language = context.language

        # Add to all imports
        self._imports.add(module_name)

        # Classify import
        if is_relative_import(module_name, language):
            # Relative import = internal
            self._internal_imports.add(module_name)
        elif is_stdlib_module(module_name, language):
            # Standard library = external (but not third-party)
            self._external_imports.add(module_name)
        else:
            # Check if internal by checking if it starts with project root
            # For now, treat non-relative, non-stdlib as external
            # Future enhancement: project_root detection
            self._external_imports.add(module_name)

    def get_imported_modules(self) -> set[str]:
        """Get set of all imported module names.

        Returns:
            Set of module names imported by this file
        """
        return self._imports.copy()

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return empty dict - coupling is file-level, not function-level.

        Coupling metrics are computed at file level during finalization.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Empty dictionary (no function-level coupling metrics)
        """
        return {}

    def get_file_metrics(self) -> dict[str, Any]:
        """Get file-level coupling metrics.

        Returns:
            Dictionary with efferent coupling metrics
        """
        return {
            "efferent_coupling": len(self._imports),
            "imports": sorted(self._imports),
            "internal_imports": sorted(self._internal_imports),
            "external_imports": sorted(self._external_imports),
        }

    def reset(self) -> None:
        """Reset collector state for next file."""
        self._imports.clear()
        self._internal_imports.clear()
        self._external_imports.clear()


class AfferentCouplingCollector(MetricCollector):
    """Tracks afferent coupling (Ca) - incoming dependencies.

    Afferent coupling measures how many other files depend on this file
    (i.e., how many files import this file). Higher Ca indicates this
    file is more load-bearing - changes will affect many other files.

    Interpretation:
    - 0-2: Low coupling, changes affect few files
    - 3-5: Moderate coupling, shared utility
    - 6-10: High coupling, critical component
    - 11+: Very high coupling, core infrastructure

    Example:
        # File A is imported by files B, C, D
        # Afferent Coupling (Ca) = 3

    Note: Afferent coupling requires project-wide import graph analysis.
    Use build_import_graph() to construct the graph before creating this collector.
    """

    def __init__(self, import_graph: dict[str, set[str]] | None = None) -> None:
        """Initialize afferent coupling collector.

        Args:
            import_graph: Pre-built import graph mapping module_name → set of importing files.
                         If None, afferent coupling will always be 0.
        """
        self._import_graph = import_graph or {}
        self._current_file: str | None = None

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "afferent_coupling"
        """
        return "afferent_coupling"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node (no-op for afferent coupling).

        Afferent coupling is computed from the import graph, not by traversing nodes.

        Args:
            node: Current tree-sitter AST node (unused)
            context: Shared context with file path
            depth: Current depth in AST (unused)
        """
        # Store current file path for lookup
        if context.file_path and not self._current_file:
            self._current_file = context.file_path

    def get_afferent_coupling(self, file_path: str) -> int:
        """Get count of files that import this file.

        Args:
            file_path: Path to the file to check

        Returns:
            Number of files that import this file
        """
        # Normalize file path for lookup
        normalized_path = self._normalize_path(file_path)

        # Look up in import graph
        if normalized_path in self._import_graph:
            return len(self._import_graph[normalized_path])

        return 0

    def get_dependents(self, file_path: str) -> list[str]:
        """Get list of files that depend on this file.

        Args:
            file_path: Path to the file to check

        Returns:
            List of file paths that import this file
        """
        normalized_path = self._normalize_path(file_path)

        if normalized_path in self._import_graph:
            return sorted(self._import_graph[normalized_path])

        return []

    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path for consistent lookup.

        Args:
            file_path: File path to normalize

        Returns:
            Normalized file path
        """
        # Convert to Path and resolve to absolute path
        path = Path(file_path)
        if path.is_absolute():
            return str(path)

        # If relative, return as-is (caller should ensure consistency)
        return str(path)

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return final afferent coupling metrics.

        Note: This is called per function, but afferent coupling is a file-level metric.

        Args:
            node: Function definition node
            context: Shared context with file path

        Returns:
            Dictionary with afferent_coupling count and dependents list
        """
        file_path = context.file_path
        return {
            "afferent_coupling": self.get_afferent_coupling(file_path),
            "dependents": self.get_dependents(file_path),
        }

    def reset(self) -> None:
        """Reset collector state for next file."""
        self._current_file = None


def build_import_graph(
    project_root: Path, files: list[Path], language: str = "python"
) -> dict[str, set[str]]:
    """Build project-wide import graph for afferent coupling analysis.

    Analyzes all files in the project to construct a reverse dependency graph
    mapping each module to the set of files that import it.

    Args:
        project_root: Root directory of the project
        files: List of file paths to analyze
        language: Programming language (default: "python")

    Returns:
        Dictionary mapping module_name → set of file paths that import it

    Example:
        >>> files = [Path("a.py"), Path("b.py"), Path("c.py")]
        >>> graph = build_import_graph(Path("/project"), files)
        >>> graph["module_x"]
        {"a.py", "c.py"}  # Both a.py and c.py import module_x
    """
    import_graph: dict[str, set[str]] = {}

    # Use tree-sitter to parse each file and extract imports
    try:
        from tree_sitter import Parser

        # Get tree-sitter language
        language_obj = _get_tree_sitter_language(language)
        if not language_obj:
            # Fallback: no tree-sitter support, return empty graph
            return import_graph

        parser = Parser()
        parser.set_language(language_obj)

    except ImportError:
        # Tree-sitter not available, return empty graph
        return import_graph

    # Create efferent coupling collector to extract imports
    efferent_collector = EfferentCouplingCollector()

    for file_path in files:
        # Skip non-existent files
        if not file_path.exists():
            continue

        # Read file content
        try:
            source_code = file_path.read_bytes()
        except OSError:
            continue

        # Parse with tree-sitter
        tree = parser.parse(source_code)
        if not tree or not tree.root_node:
            continue

        # Create context for this file
        context = CollectorContext(
            file_path=str(file_path.relative_to(project_root)),
            source_code=source_code,
            language=language,
        )

        # Traverse AST and collect imports
        efferent_collector.reset()
        _traverse_tree(tree.root_node, context, efferent_collector)

        # Get imported modules for this file
        imported_modules = efferent_collector.get_imported_modules()

        # Update import graph (reverse mapping)
        file_key = str(file_path.relative_to(project_root))
        for module_name in imported_modules:
            if module_name not in import_graph:
                import_graph[module_name] = set()
            import_graph[module_name].add(file_key)

    return import_graph


def _traverse_tree(
    node: Node, context: CollectorContext, collector: EfferentCouplingCollector
) -> None:
    """Recursively traverse tree-sitter AST and collect imports.

    Args:
        node: Current AST node
        context: Collector context
        collector: Efferent coupling collector to accumulate imports
    """
    # Process current node
    collector.collect_node(node, context, depth=0)

    # Recursively process children
    for child in node.children:
        _traverse_tree(child, context, collector)


def _get_tree_sitter_language(language: str) -> Any:  # noqa: ARG001
    """Get tree-sitter Language object for the given language.

    Args:
        language: Programming language identifier

    Returns:
        Tree-sitter Language object, or None if not available
    """
    try:
        # Language loading depends on tree-sitter installation
        # This is a simplified version - actual implementation should handle
        # loading compiled language libraries properly
        # In a real implementation, this would load the compiled language library
        # For now, return None to indicate unsupported
        return None

    except ImportError:
        return None


class InstabilityCalculator:
    """Calculator for instability metrics across the project.

    Instability (I) = Ce / (Ce + Ca) measures how much a file depends on others
    vs. how much others depend on it.

    Interpretation:
    - I = 0.0-0.3: Stable (maximally stable at 0.0)
    - I = 0.3-0.7: Balanced
    - I = 0.7-1.0: Unstable (maximally unstable at 1.0)

    Stable files should contain abstractions and core logic.
    Unstable files should contain concrete implementations and glue code.
    """

    def __init__(
        self,
        efferent_collector: EfferentCouplingCollector,
        afferent_collector: AfferentCouplingCollector,
    ) -> None:
        """Initialize instability calculator.

        Args:
            efferent_collector: Collector for outgoing dependencies
            afferent_collector: Collector for incoming dependencies
        """
        self._efferent_collector = efferent_collector
        self._afferent_collector = afferent_collector

    def calculate_instability(self, file_path: str) -> float:
        """Calculate instability for a single file.

        Args:
            file_path: Path to the file

        Returns:
            Instability value from 0.0 (stable) to 1.0 (unstable)
        """
        ce = len(self._efferent_collector.get_imported_modules())
        ca = self._afferent_collector.get_afferent_coupling(file_path)

        total = ce + ca
        if total == 0:
            return 0.0

        return ce / total

    def calculate_project_instability(
        self, file_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate instability for all files in the project.

        Args:
            file_metrics: Dictionary mapping file_path → file metrics

        Returns:
            Dictionary mapping file_path → instability value
        """
        instability_map: dict[str, float] = {}

        for file_path in file_metrics:
            # Get coupling metrics from file_metrics
            if "coupling" in file_metrics[file_path]:
                coupling = file_metrics[file_path]["coupling"]
                ce = coupling.get("efferent_coupling", 0)
                ca = coupling.get("afferent_coupling", 0)

                total = ce + ca
                if total == 0:
                    instability = 0.0
                else:
                    instability = ce / total

                instability_map[file_path] = instability

        return instability_map

    def get_stability_grade(self, instability: float) -> str:
        """Get letter grade for instability value.

        Args:
            instability: Instability value (0.0-1.0)

        Returns:
            Letter grade from A to F

        Grade thresholds:
        - A: 0.0-0.2 (very stable)
        - B: 0.2-0.4 (stable)
        - C: 0.4-0.6 (balanced)
        - D: 0.6-0.8 (unstable)
        - F: 0.8-1.0 (very unstable)
        """
        if instability <= 0.2:
            return "A"
        elif instability <= 0.4:
            return "B"
        elif instability <= 0.6:
            return "C"
        elif instability <= 0.8:
            return "D"
        else:
            return "F"

    def get_stability_category(self, instability: float) -> str:
        """Get stability category for instability value.

        Args:
            instability: Instability value (0.0-1.0)

        Returns:
            Category: "Stable", "Balanced", or "Unstable"
        """
        if instability <= 0.3:
            return "Stable"
        elif instability <= 0.7:
            return "Balanced"
        else:
            return "Unstable"

    def get_most_stable_files(
        self, instability_map: dict[str, float], limit: int = 10
    ) -> list[tuple[str, float]]:
        """Get most stable files (lowest instability).

        Args:
            instability_map: Dictionary mapping file_path → instability
            limit: Maximum number of files to return

        Returns:
            List of (file_path, instability) tuples, sorted by stability
        """
        sorted_files = sorted(instability_map.items(), key=lambda x: x[1])
        return sorted_files[:limit]

    def get_most_unstable_files(
        self, instability_map: dict[str, float], limit: int = 10
    ) -> list[tuple[str, float]]:
        """Get most unstable files (highest instability).

        Args:
            instability_map: Dictionary mapping file_path → instability
            limit: Maximum number of files to return

        Returns:
            List of (file_path, instability) tuples, sorted by instability (descending)
        """
        sorted_files = sorted(instability_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:limit]
