"""Unit tests for coupling metric collectors."""

from pathlib import Path

import pytest

from mcp_vector_search.analysis.collectors.base import CollectorContext
from mcp_vector_search.analysis.collectors.coupling import (
    AfferentCouplingCollector,
    CircularDependency,
    CircularDependencyDetector,
    EfferentCouplingCollector,
    ImportGraph,
    NodeColor,
    build_import_graph,
    build_import_graph_from_dict,
    get_import_node_types,
    is_relative_import,
    is_stdlib_module,
)
from mcp_vector_search.analysis.metrics import CouplingMetrics


class TestGetImportNodeTypes:
    """Test import node type lookup helper."""

    def test_python_import_types(self):
        """Test Python import node type mappings."""
        result = get_import_node_types("python", "import")
        assert "import_statement" in result
        assert "import_from_statement" in result

    def test_javascript_import_types(self):
        """Test JavaScript import node type mappings."""
        result = get_import_node_types("javascript", "import")
        assert "import_statement" in result

    def test_typescript_import_types(self):
        """Test TypeScript import node type mappings."""
        result = get_import_node_types("typescript", "import")
        assert "import_statement" in result

    def test_java_import_types(self):
        """Test Java import node type mappings."""
        result = get_import_node_types("java", "import")
        assert "import_declaration" in result

    def test_rust_import_types(self):
        """Test Rust import node type mappings."""
        result = get_import_node_types("rust", "import")
        assert "use_declaration" in result

    def test_unknown_language_fallback(self):
        """Test unknown language falls back to Python-like behavior."""
        result = get_import_node_types("unknown_lang", "import")
        assert "import_statement" in result
        assert "import_from_statement" in result

    def test_unknown_category(self):
        """Test unknown category returns empty list."""
        assert get_import_node_types("python", "nonexistent") == []


class TestImportHelpers:
    """Test helper functions for import classification."""

    def test_is_stdlib_module_python(self):
        """Test Python stdlib detection."""
        # Common stdlib modules
        assert is_stdlib_module("os", "python") is True
        assert is_stdlib_module("sys", "python") is True
        assert is_stdlib_module("json", "python") is True
        assert is_stdlib_module("typing", "python") is True

        # Third-party modules
        assert is_stdlib_module("requests", "python") is False
        assert is_stdlib_module("numpy", "python") is False
        assert is_stdlib_module("flask", "python") is False

    def test_is_stdlib_module_javascript(self):
        """Test Node.js built-in module detection."""
        # Node.js built-ins
        assert is_stdlib_module("fs", "javascript") is True
        assert is_stdlib_module("path", "javascript") is True
        assert is_stdlib_module("http", "javascript") is True

        # Third-party modules
        assert is_stdlib_module("express", "javascript") is False
        assert is_stdlib_module("lodash", "javascript") is False
        assert is_stdlib_module("react", "javascript") is False

    def test_is_relative_import_python(self):
        """Test Python relative import detection."""
        assert is_relative_import(".utils", "python") is True
        assert is_relative_import("..models", "python") is True
        assert is_relative_import("...config", "python") is True

        assert is_relative_import("os", "python") is False
        assert is_relative_import("mypackage", "python") is False

    def test_is_relative_import_javascript(self):
        """Test JavaScript relative import detection."""
        assert is_relative_import("./utils", "javascript") is True
        assert is_relative_import("../models", "javascript") is True
        assert is_relative_import("../../config", "javascript") is True

        assert is_relative_import("fs", "javascript") is False
        assert is_relative_import("lodash", "javascript") is False
        assert is_relative_import("@types/node", "javascript") is False


class MockNode:
    """Mock tree-sitter node for testing."""

    def __init__(self, node_type: str, text: str = "", children=None):
        self.type = node_type
        self._text = text.encode("utf-8")
        self.children = children or []
        self.is_named = True

    @property
    def text(self) -> bytes:
        """Return node text as bytes."""
        return self._text

    def child_by_field_name(self, field: str):
        """Mock field-based child lookup."""
        return None


class TestEfferentCouplingCollector:
    """Test EfferentCouplingCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = EfferentCouplingCollector()
        assert collector.name == "efferent_coupling"
        assert len(collector._imports) == 0

    def test_no_imports(self):
        """Test file with no imports has Ce = 0."""
        collector = EfferentCouplingCollector()

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 0
        assert result["imports"] == []

    def test_python_single_import(self):
        """Test Python single import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"import os", language="python"
        )

        # Mock import statement: import os
        dotted_name = MockNode("dotted_name", text="os")
        import_node = MockNode("import_statement", children=[dotted_name])
        collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 1
        assert "os" in result["imports"]

    def test_python_from_import(self):
        """Test Python from...import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"from collections import deque",
            language="python",
        )

        # Mock import: from collections import deque
        dotted_name = MockNode("dotted_name", text="collections")
        import_node = MockNode("import_from_statement", children=[dotted_name])
        collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 1
        assert "collections" in result["imports"]

    def test_python_import_with_alias(self):
        """Test Python import with alias (as keyword)."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"import numpy as np", language="python"
        )

        # Mock import: import numpy as np
        dotted_name = MockNode("dotted_name", text="numpy")
        aliased_import = MockNode(
            "aliased_import", text="numpy as np", children=[dotted_name]
        )
        import_node = MockNode("import_statement", children=[aliased_import])
        collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 1
        assert "numpy" in result["imports"]

    def test_multiple_imports(self):
        """Test multiple import statements increase Ce."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(file_path="test.py", source_code=b"", language="python")

        # Simulate multiple imports
        imports = ["os", "sys", "json", "pathlib"]
        for module_name in imports:
            dotted_name = MockNode("dotted_name", text=module_name)
            import_node = MockNode("import_statement", children=[dotted_name])
            collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 4
        assert set(result["imports"]) == set(imports)

    def test_duplicate_imports_counted_once(self):
        """Test duplicate imports are only counted once."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(file_path="test.py", source_code=b"", language="python")

        # Import same module twice
        for _ in range(2):
            dotted_name = MockNode("dotted_name", text="os")
            import_node = MockNode("import_statement", children=[dotted_name])
            collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 1
        assert result["imports"] == ["os"]

    def test_javascript_import(self):
        """Test JavaScript import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.js",
            source_code=b'import React from "react"',
            language="javascript",
        )

        # Mock: import React from "react"
        string_node = MockNode("string", text='"react"')
        import_node = MockNode("import_statement", children=[string_node])
        collector.collect_node(import_node, ctx, 0)

        result = collector.get_file_metrics()
        assert result["efferent_coupling"] == 1
        assert "react" in result["imports"]

    def test_get_imported_modules(self):
        """Test get_imported_modules returns copy of set."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(file_path="test.py", source_code=b"", language="python")

        # Add import
        dotted_name = MockNode("dotted_name", text="os")
        import_node = MockNode("import_statement", children=[dotted_name])
        collector.collect_node(import_node, ctx, 0)

        # Get imported modules
        modules = collector.get_imported_modules()
        assert "os" in modules

        # Modify returned set should not affect collector
        modules.add("sys")
        assert "sys" not in collector._imports

    def test_reset(self):
        """Test reset clears imported modules."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(file_path="test.py", source_code=b"", language="python")

        # Add imports
        dotted_name = MockNode("dotted_name", text="os")
        import_node = MockNode("import_statement", children=[dotted_name])
        collector.collect_node(import_node, ctx, 0)
        assert len(collector._imports) > 0

        # Reset
        collector.reset()
        assert len(collector._imports) == 0


class TestAfferentCouplingCollector:
    """Test AfferentCouplingCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = AfferentCouplingCollector()
        assert collector.name == "afferent_coupling"
        assert collector._import_graph == {}

    def test_initialization_with_graph(self):
        """Test collector with pre-built import graph."""
        import_graph = {
            "module_a": {"file1.py", "file2.py"},
            "module_b": {"file1.py"},
        }
        collector = AfferentCouplingCollector(import_graph=import_graph)
        assert collector._import_graph == import_graph

    def test_no_dependents(self):
        """Test file with no dependents has Ca = 0."""
        collector = AfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["afferent_coupling"] == 0
        assert result["dependents"] == []

    def test_single_dependent(self):
        """Test file with one dependent has Ca = 1."""
        import_graph = {"test.py": {"dependent1.py"}}
        collector = AfferentCouplingCollector(import_graph=import_graph)
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["afferent_coupling"] == 1
        assert result["dependents"] == ["dependent1.py"]

    def test_multiple_dependents(self):
        """Test file with multiple dependents."""
        import_graph = {"test.py": {"dep1.py", "dep2.py", "dep3.py"}}
        collector = AfferentCouplingCollector(import_graph=import_graph)
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["afferent_coupling"] == 3
        assert set(result["dependents"]) == {"dep1.py", "dep2.py", "dep3.py"}

    def test_get_afferent_coupling(self):
        """Test get_afferent_coupling method."""
        import_graph = {
            "module_a.py": {"file1.py", "file2.py"},
            "module_b.py": {"file1.py"},
        }
        collector = AfferentCouplingCollector(import_graph=import_graph)

        assert collector.get_afferent_coupling("module_a.py") == 2
        assert collector.get_afferent_coupling("module_b.py") == 1
        assert collector.get_afferent_coupling("module_c.py") == 0  # Not in graph

    def test_get_dependents(self):
        """Test get_dependents method."""
        import_graph = {"module_a.py": {"file1.py", "file2.py"}}
        collector = AfferentCouplingCollector(import_graph=import_graph)

        dependents = collector.get_dependents("module_a.py")
        assert set(dependents) == {"file1.py", "file2.py"}
        assert dependents == sorted(dependents)  # Should be sorted

    def test_get_dependents_empty(self):
        """Test get_dependents returns empty list for unknown file."""
        collector = AfferentCouplingCollector()
        dependents = collector.get_dependents("unknown.py")
        assert dependents == []

    def test_reset(self):
        """Test reset clears current file."""
        import_graph = {"test.py": {"dep.py"}}
        collector = AfferentCouplingCollector(import_graph=import_graph)
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Process node to set current file
        collector.collect_node(MockNode("function_definition"), ctx, 0)
        assert collector._current_file == "test.py"

        # Reset
        collector.reset()
        assert collector._current_file is None


class TestBuildImportGraph:
    """Test build_import_graph function."""

    def test_empty_file_list(self):
        """Test with empty file list returns empty graph."""
        project_root = Path("/project")
        graph = build_import_graph(project_root, [], language="python")
        assert graph == {}

    def test_nonexistent_files(self):
        """Test with nonexistent files returns empty graph."""
        project_root = Path("/project")
        files = [Path("/project/nonexistent.py")]
        graph = build_import_graph(project_root, files, language="python")
        assert graph == {}

    @pytest.mark.skip(
        reason="Requires tree-sitter setup and real file system integration"
    )
    def test_build_graph_from_files(self, tmp_path):
        """Test building import graph from actual files."""
        # Create temporary project structure
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create file A that imports B and C
        file_a = project_root / "a.py"
        file_a.write_text("import b\nfrom c import func")

        # Create file B that imports C
        file_b = project_root / "b.py"
        file_b.write_text("import c")

        # Create file C (no imports)
        file_c = project_root / "c.py"
        file_c.write_text("def func(): pass")

        # Build import graph
        files = [file_a, file_b, file_c]
        graph = build_import_graph(project_root, files, language="python")

        # Expected: b is imported by a, c is imported by a and b
        assert "b" in graph
        assert "a.py" in graph["b"]
        assert "c" in graph
        assert "a.py" in graph["c"]
        assert "b.py" in graph["c"]


class TestCouplingMetrics:
    """Test CouplingMetrics dataclass."""

    def test_initialization(self):
        """Test CouplingMetrics initializes with defaults."""
        metrics = CouplingMetrics()
        assert metrics.efferent_coupling == 0
        assert metrics.afferent_coupling == 0
        assert metrics.imports == []
        assert metrics.internal_imports == []
        assert metrics.external_imports == []
        assert metrics.dependents == []

    def test_instability_zero_coupling(self):
        """Test instability is 0.0 when no coupling."""
        metrics = CouplingMetrics(efferent_coupling=0, afferent_coupling=0)
        assert metrics.instability == 0.0

    def test_instability_maximally_stable(self):
        """Test instability is 0.0 for maximally stable (many incoming, no outgoing)."""
        metrics = CouplingMetrics(efferent_coupling=0, afferent_coupling=10)
        assert metrics.instability == 0.0

    def test_instability_maximally_unstable(self):
        """Test instability is 1.0 for maximally unstable (many outgoing, no incoming)."""
        metrics = CouplingMetrics(efferent_coupling=10, afferent_coupling=0)
        assert metrics.instability == 1.0

    def test_instability_balanced(self):
        """Test instability is 0.5 for balanced coupling."""
        metrics = CouplingMetrics(efferent_coupling=5, afferent_coupling=5)
        assert metrics.instability == 0.5

    def test_instability_calculation(self):
        """Test instability calculation with various values."""
        # Ce = 3, Ca = 7 -> Instability = 3 / (3 + 7) = 0.3
        metrics = CouplingMetrics(efferent_coupling=3, afferent_coupling=7)
        assert metrics.instability == 0.3

        # Ce = 8, Ca = 2 -> Instability = 8 / (8 + 2) = 0.8
        metrics = CouplingMetrics(efferent_coupling=8, afferent_coupling=2)
        assert metrics.instability == 0.8

    def test_with_module_lists(self):
        """Test CouplingMetrics with module and dependent lists."""
        metrics = CouplingMetrics(
            efferent_coupling=2,
            afferent_coupling=3,
            imports=["module_a", "module_b"],
            internal_imports=[".utils"],
            external_imports=["requests"],
            dependents=["dep1.py", "dep2.py", "dep3.py"],
        )
        assert metrics.efferent_coupling == 2
        assert metrics.afferent_coupling == 3
        assert len(metrics.imports) == 2
        assert len(metrics.internal_imports) == 1
        assert len(metrics.external_imports) == 1
        assert len(metrics.dependents) == 3
        assert metrics.instability == 0.4  # 2 / (2 + 3)


class TestCollectorIntegration:
    """Test efferent and afferent collectors working together."""

    def test_collectors_independent(self):
        """Test both collectors can run independently."""
        efferent = EfferentCouplingCollector()
        import_graph = {"test.py": {"dep.py"}}
        afferent = AfferentCouplingCollector(import_graph=import_graph)

        ctx = CollectorContext(
            file_path="test.py", source_code=b"import os", language="python"
        )

        # Simulate import
        dotted_name = MockNode("dotted_name", text="os")
        import_node = MockNode("import_statement", children=[dotted_name])

        # Process with both collectors
        efferent.collect_node(import_node, ctx, 0)
        afferent.collect_node(import_node, ctx, 0)

        # Get results
        efferent_result = efferent.get_file_metrics()
        afferent_result = afferent.finalize_function(
            MockNode("function_definition"), ctx
        )

        # Verify independence
        assert efferent_result["efferent_coupling"] == 1
        assert afferent_result["afferent_coupling"] == 1
        assert "os" in efferent_result["imports"]
        assert "dep.py" in afferent_result["dependents"]

    def test_combined_coupling_metrics(self):
        """Test creating CouplingMetrics from both collectors."""
        # Setup collectors
        efferent = EfferentCouplingCollector()
        import_graph = {"test.py": {"dep1.py", "dep2.py"}}
        afferent = AfferentCouplingCollector(import_graph=import_graph)

        ctx = CollectorContext(file_path="test.py", source_code=b"", language="python")

        # Add imports to efferent
        for module in ["os", "sys", "json"]:
            dotted_name = MockNode("dotted_name", text=module)
            import_node = MockNode("import_statement", children=[dotted_name])
            efferent.collect_node(import_node, ctx, 0)

        # Get results
        efferent_result = efferent.get_file_metrics()
        afferent_result = afferent.finalize_function(
            MockNode("function_definition"), ctx
        )

        # Create CouplingMetrics
        coupling = CouplingMetrics(
            efferent_coupling=efferent_result["efferent_coupling"],
            afferent_coupling=afferent_result["afferent_coupling"],
            imports=efferent_result["imports"],
            dependents=afferent_result["dependents"],
        )

        # Verify combined metrics
        assert coupling.efferent_coupling == 3  # os, sys, json
        assert coupling.afferent_coupling == 2  # dep1, dep2
        assert len(coupling.imports) == 3
        assert len(coupling.dependents) == 2
        assert coupling.instability == 0.6  # 3 / (3 + 2)


# =============================================================================
# Circular Dependency Detection Tests
# =============================================================================


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
        """Test that get_all_files includes files that are imported but don't import."""
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


class TestBuildImportGraphFromDict:
    """Test build_import_graph_from_dict utility function."""

    def test_empty_imports(self):
        """Test building graph from empty imports dict."""
        file_imports: dict[str, list[str]] = {}
        graph = build_import_graph_from_dict(file_imports)

        assert graph.get_all_files() == []

    def test_simple_imports(self):
        """Test building graph from simple imports."""
        file_imports = {
            "main.py": ["utils.py", "config.py"],
            "utils.py": ["helpers.py"],
            "config.py": [],
            "helpers.py": [],
        }

        graph = build_import_graph_from_dict(file_imports)

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

        graph = build_import_graph_from_dict(file_imports)

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

        graph = build_import_graph_from_dict(file_imports)

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
