"""Unit tests for coupling metric collectors (import graph and circular dependencies)."""

from mcp_vector_search.analysis.collectors.coupling import (
    CircularDependency,
    CircularDependencyDetector,
    ImportGraph,
    NodeColor,
    build_import_graph,
)


class TestImportGraph:
    """Test ImportGraph data structure."""

    def test_empty_graph(self):
        """Test empty graph initialization."""
        graph = ImportGraph()
        assert graph.adjacency_list == {}
        assert graph.get_all_files() == []

    def test_add_node(self):
        """Test adding nodes without edges."""
        graph = ImportGraph()
        graph.add_node("file1.py")
        graph.add_node("file2.py")

        assert "file1.py" in graph.adjacency_list
        assert "file2.py" in graph.adjacency_list
        assert graph.get_neighbors("file1.py") == []
        assert graph.get_neighbors("file2.py") == []

    def test_add_edge(self):
        """Test adding directed edges."""
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")

        assert "main.py" in graph.adjacency_list
        assert "utils.py" in graph.get_neighbors("main.py")

    def test_add_edge_creates_node_if_not_exists(self):
        """Test that add_edge creates source node if it doesn't exist."""
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")

        assert "main.py" in graph.adjacency_list

    def test_add_duplicate_edge(self):
        """Test that duplicate edges are not added twice."""
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")
        graph.add_edge("main.py", "utils.py")  # Duplicate

        neighbors = graph.get_neighbors("main.py")
        assert neighbors.count("utils.py") == 1

    def test_get_neighbors_nonexistent_node(self):
        """Test getting neighbors of non-existent node returns empty list."""
        graph = ImportGraph()
        assert graph.get_neighbors("nonexistent.py") == []

    def test_get_all_files_includes_imported_files(self):
        """Test that get_all_files includes files that are imported but don't import anything."""
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")
        # utils.py is imported but has no entry in adjacency_list as a key

        all_files = graph.get_all_files()
        assert "main.py" in all_files
        assert "utils.py" in all_files

    def test_complex_graph(self):
        """Test building a complex graph with multiple edges."""
        graph = ImportGraph()
        graph.add_edge("main.py", "utils.py")
        graph.add_edge("main.py", "config.py")
        graph.add_edge("utils.py", "helpers.py")
        graph.add_edge("config.py", "helpers.py")

        assert len(graph.get_neighbors("main.py")) == 2
        assert "utils.py" in graph.get_neighbors("main.py")
        assert "config.py" in graph.get_neighbors("main.py")
        assert "helpers.py" in graph.get_neighbors("utils.py")


class TestCircularDependency:
    """Test CircularDependency dataclass."""

    def test_simple_cycle(self):
        """Test simple 2-node cycle (A → B → A)."""
        cycle = CircularDependency(cycle_chain=["a.py", "b.py", "a.py"])

        assert cycle.cycle_length == 2
        assert cycle.format_chain() == "a.py → b.py → a.py"
        assert set(cycle.get_affected_files()) == {"a.py", "b.py"}

    def test_complex_cycle(self):
        """Test complex 3-node cycle (A → B → C → A)."""
        cycle = CircularDependency(cycle_chain=["a.py", "b.py", "c.py", "a.py"])

        assert cycle.cycle_length == 3
        assert cycle.format_chain() == "a.py → b.py → c.py → a.py"
        assert set(cycle.get_affected_files()) == {"a.py", "b.py", "c.py"}

    def test_self_cycle(self):
        """Test self-import cycle (A → A)."""
        cycle = CircularDependency(cycle_chain=["a.py", "a.py"])

        assert cycle.cycle_length == 1
        assert cycle.format_chain() == "a.py → a.py"
        assert cycle.get_affected_files() == ["a.py"]

    def test_empty_cycle(self):
        """Test empty cycle edge case."""
        cycle = CircularDependency(cycle_chain=[])

        assert cycle.cycle_length == 0
        assert cycle.format_chain() == ""
        assert cycle.get_affected_files() == []

    def test_get_affected_files_sorted(self):
        """Test that affected files are returned sorted."""
        cycle = CircularDependency(cycle_chain=["z.py", "a.py", "m.py", "z.py"])

        affected = cycle.get_affected_files()
        assert affected == sorted(affected)


class TestCircularDependencyDetector:
    """Test CircularDependencyDetector cycle detection algorithm."""

    def test_no_cycles_empty_graph(self):
        """Test empty graph has no cycles."""
        graph = ImportGraph()
        detector = CircularDependencyDetector(graph)

        cycles = detector.detect_cycles()

        assert cycles == []
        assert not detector.has_cycles()
        assert detector.get_cycle_chains() == []
        assert detector.get_affected_files() == []

    def test_no_cycles_acyclic_graph(self):
        """Test acyclic graph (DAG) has no cycles."""
        # Graph: A → B → C (linear chain)
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "c.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert cycles == []
        assert not detector.has_cycles()

    def test_no_cycles_tree_structure(self):
        """Test tree structure (no cycles)."""
        # Graph:
        #     A
        #    / \
        #   B   C
        #  / \
        # D   E
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("a.py", "c.py")
        graph.add_edge("b.py", "d.py")
        graph.add_edge("b.py", "e.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert cycles == []
        assert not detector.has_cycles()

    def test_simple_cycle_two_nodes(self):
        """Test simple 2-node cycle (A ↔ B)."""
        # Graph: A → B → A
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

        cycle = cycles[0]
        assert cycle.cycle_length == 2

        # Cycle can be detected from either starting point
        # So it could be [a.py, b.py, a.py] or [b.py, a.py, b.py]
        affected = cycle.get_affected_files()
        assert set(affected) == {"a.py", "b.py"}

    def test_simple_cycle_three_nodes(self):
        """Test simple 3-node cycle (A → B → C → A)."""
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "c.py")
        graph.add_edge("c.py", "a.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

        cycle = cycles[0]
        assert cycle.cycle_length == 3
        assert set(cycle.get_affected_files()) == {"a.py", "b.py", "c.py"}

    def test_complex_cycle_four_nodes(self):
        """Test complex 4-node cycle (A → B → C → D → A)."""
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "c.py")
        graph.add_edge("c.py", "d.py")
        graph.add_edge("d.py", "a.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

        cycle = cycles[0]
        assert cycle.cycle_length == 4
        assert set(cycle.get_affected_files()) == {"a.py", "b.py", "c.py", "d.py"}

    def test_self_import(self):
        """Test self-import (A imports A)."""
        graph = ImportGraph()
        graph.add_edge("a.py", "a.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

        cycle = cycles[0]
        assert cycle.cycle_length == 1
        assert cycle.get_affected_files() == ["a.py"]
        assert cycle.format_chain() == "a.py → a.py"

    def test_multiple_independent_cycles(self):
        """Test graph with multiple independent cycles."""
        # Graph:
        # Cycle 1: A → B → A
        # Cycle 2: C → D → C
        # No connection between cycles
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")
        graph.add_edge("c.py", "d.py")
        graph.add_edge("d.py", "c.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 2
        assert detector.has_cycles()

        # Check that we found both cycles
        all_affected = detector.get_affected_files()
        assert set(all_affected) == {"a.py", "b.py", "c.py", "d.py"}

    def test_nested_cycles(self):
        """Test graph with nested cycles."""
        # Graph:
        # Outer cycle: A → B → C → A
        # Inner cycle: B → D → B
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "c.py")
        graph.add_edge("c.py", "a.py")
        graph.add_edge("b.py", "d.py")
        graph.add_edge("d.py", "b.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        # Should detect both cycles
        assert len(cycles) >= 2
        assert detector.has_cycles()

        all_affected = detector.get_affected_files()
        assert "a.py" in all_affected
        assert "b.py" in all_affected
        assert "c.py" in all_affected
        assert "d.py" in all_affected

    def test_cycle_with_acyclic_branches(self):
        """Test graph with cycle and acyclic branches."""
        # Graph:
        # Cycle: A → B → A
        # Acyclic branches: A → C, B → D
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")
        graph.add_edge("a.py", "c.py")
        graph.add_edge("b.py", "d.py")

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        # Should only detect the A-B cycle
        assert len(cycles) == 1
        assert detector.has_cycles()

        affected = detector.get_affected_files()
        assert set(affected) == {"a.py", "b.py"}

        # c.py and d.py should not be in affected files
        assert "c.py" not in affected
        assert "d.py" not in affected

    def test_get_cycle_chains(self):
        """Test get_cycle_chains returns formatted strings."""
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")

        detector = CircularDependencyDetector(graph)
        detector.detect_cycles()

        chains = detector.get_cycle_chains()
        assert len(chains) == 1
        assert " → " in chains[0]

    def test_diamond_with_cycle(self):
        """Test diamond shape with cycle at bottom."""
        # Graph:
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        #     ↑ (D → B creates cycle)
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("a.py", "c.py")
        graph.add_edge("b.py", "d.py")
        graph.add_edge("c.py", "d.py")
        graph.add_edge("d.py", "b.py")  # Creates cycle

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

        affected = detector.get_affected_files()
        assert "b.py" in affected
        assert "d.py" in affected

    def test_detect_cycles_can_be_called_multiple_times(self):
        """Test that detect_cycles() can be safely called multiple times."""
        graph = ImportGraph()
        graph.add_edge("a.py", "b.py")
        graph.add_edge("b.py", "a.py")

        detector = CircularDependencyDetector(graph)

        cycles1 = detector.detect_cycles()
        cycles2 = detector.detect_cycles()

        assert len(cycles1) == len(cycles2)
        assert detector.has_cycles()


class TestBuildImportGraph:
    """Test build_import_graph utility function."""

    def test_empty_imports(self):
        """Test building graph from empty imports dict."""
        file_imports = {}
        graph = build_import_graph(file_imports)

        assert graph.get_all_files() == []

    def test_simple_imports(self):
        """Test building graph from simple imports."""
        file_imports = {
            "main.py": ["utils.py", "config.py"],
            "utils.py": ["helpers.py"],
            "config.py": [],
            "helpers.py": [],
        }

        graph = build_import_graph(file_imports)

        assert "main.py" in graph.get_all_files()
        assert "utils.py" in graph.get_all_files()
        assert "config.py" in graph.get_all_files()
        assert "helpers.py" in graph.get_all_files()

        assert set(graph.get_neighbors("main.py")) == {"utils.py", "config.py"}
        assert graph.get_neighbors("utils.py") == ["helpers.py"]
        assert graph.get_neighbors("config.py") == []
        assert graph.get_neighbors("helpers.py") == []

    def test_circular_imports(self):
        """Test building graph with circular imports."""
        file_imports = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["a.py"],
        }

        graph = build_import_graph(file_imports)

        detector = CircularDependencyDetector(graph)
        cycles = detector.detect_cycles()

        assert len(cycles) == 1
        assert detector.has_cycles()

    def test_isolated_files_included(self):
        """Test that isolated files (no imports) are included as nodes."""
        file_imports = {
            "main.py": ["utils.py"],
            "isolated.py": [],  # No imports, should still be in graph
        }

        graph = build_import_graph(file_imports)

        assert "isolated.py" in graph.get_all_files()
        assert graph.get_neighbors("isolated.py") == []


class TestNodeColor:
    """Test NodeColor enum."""

    def test_enum_values(self):
        """Test that NodeColor has expected values."""
        assert NodeColor.WHITE.value == "white"
        assert NodeColor.GRAY.value == "gray"
        assert NodeColor.BLACK.value == "black"

    def test_enum_members(self):
        """Test that NodeColor has all expected members."""
        members = list(NodeColor)
        assert len(members) == 3
        assert NodeColor.WHITE in members
        assert NodeColor.GRAY in members
        assert NodeColor.BLACK in members
