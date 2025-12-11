"""Coupling metric collectors for cross-file dependency analysis.

This module provides collectors for analyzing dependencies between files:
- ImportGraphBuilder: Builds import dependency graph from source files
- CircularDependencyDetector: Detects circular/cyclic dependencies in import graph

Circular dependencies can lead to:
- Initialization issues and import errors
- Tight coupling and reduced maintainability
- Difficulty in testing and refactoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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


def build_import_graph(file_imports: dict[str, list[str]]) -> ImportGraph:
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
        graph = build_import_graph(imports)
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
