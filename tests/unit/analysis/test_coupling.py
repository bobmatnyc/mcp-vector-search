"""Unit tests for coupling metric collectors."""

import pytest

from mcp_vector_search.analysis import (
    CollectorContext,
    CouplingMetrics,
    EfferentCouplingCollector,
)
from mcp_vector_search.analysis.collectors.coupling import (
    get_import_node_types,
    is_relative_import,
    is_stdlib_module,
)


class TestImportHelpers:
    """Test helper functions for import classification."""

    def test_get_import_node_types_python(self):
        """Test Python import node type mappings."""
        assert get_import_node_types("python", "import_statement") == [
            "import_statement"
        ]
        assert get_import_node_types("python", "import_from") == [
            "import_from_statement"
        ]

    def test_get_import_node_types_javascript(self):
        """Test JavaScript import node type mappings."""
        assert get_import_node_types("javascript", "import_statement") == [
            "import_statement"
        ]
        assert get_import_node_types("javascript", "require_call") == [
            "call_expression"
        ]

    def test_get_import_node_types_typescript(self):
        """Test TypeScript import node type mappings."""
        assert get_import_node_types("typescript", "import_statement") == [
            "import_statement"
        ]
        assert get_import_node_types("typescript", "import_type") == [
            "import_statement"
        ]

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

    def __init__(self, node_type: str, text: bytes = b"", children=None):
        self.type = node_type
        self.text = text
        self.children = children or []

    def child_by_field_name(self, field: str):
        """Mock field-based child lookup."""
        return None


class TestCouplingMetrics:
    """Test CouplingMetrics dataclass."""

    def test_initialization(self):
        """Test CouplingMetrics initializes with defaults."""
        metrics = CouplingMetrics()
        assert metrics.efferent_coupling == 0
        assert metrics.imports == []
        assert metrics.internal_imports == []
        assert metrics.external_imports == []

    def test_with_values(self):
        """Test CouplingMetrics with values."""
        metrics = CouplingMetrics(
            efferent_coupling=3,
            imports=["os", "sys", "requests"],
            internal_imports=["./utils"],
            external_imports=["os", "sys", "requests"],
        )
        assert metrics.efferent_coupling == 3
        assert len(metrics.imports) == 3
        assert len(metrics.internal_imports) == 1
        assert len(metrics.external_imports) == 3


class TestEfferentCouplingCollector:
    """Test EfferentCouplingCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = EfferentCouplingCollector()
        assert collector.name == "efferent_coupling"
        assert len(collector._imports) == 0
        assert len(collector._internal_imports) == 0
        assert len(collector._external_imports) == 0

    def test_python_simple_import(self):
        """Test Python simple import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"import os", language="python"
        )

        # Mock: import os
        import_node = MockNode(
            "import_statement",
            children=[MockNode("dotted_name", text=b"os")],
        )

        collector.collect_node(import_node, ctx, 0)

        # Process dotted_name child
        for child in import_node.children:
            collector.collect_node(child, ctx, 1)

        # Should have collected "os" import
        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "os" in metrics["imports"]
        assert "os" in metrics["external_imports"]

    def test_python_from_import(self):
        """Test Python from...import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"from os.path import join",
            language="python",
        )

        # Create mock node with child_by_field_name support
        module_node = MockNode("dotted_name", text=b"os.path")
        import_node = MockNode(
            "import_from_statement",
            children=[module_node],
        )

        # Override child_by_field_name to return module_node
        import_node.child_by_field_name = lambda field: (
            module_node if field == "module_name" else None
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "os.path" in metrics["imports"]

    def test_python_relative_import(self):
        """Test Python relative import."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"from .utils import helper",
            language="python",
        )

        # Mock: from .utils import helper
        relative_node = MockNode("relative_import", text=b".")
        import_node = MockNode(
            "import_from_statement",
            children=[relative_node],
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "." in metrics["imports"]
        assert "." in metrics["internal_imports"]

    def test_python_multiple_imports(self):
        """Test multiple Python imports."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"import os\nimport sys\nfrom typing import List",
            language="python",
        )

        # Simulate processing three imports
        # Import 1: import os
        collector._add_import("os", ctx)

        # Import 2: import sys
        collector._add_import("sys", ctx)

        # Import 3: from typing import List
        collector._add_import("typing", ctx)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 3
        assert set(metrics["imports"]) == {"os", "sys", "typing"}
        assert all(
            imp in metrics["external_imports"] for imp in ["os", "sys", "typing"]
        )

    def test_python_duplicate_imports(self):
        """Test that duplicate imports are counted once."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"import os\nimport os",
            language="python",
        )

        # Add same import twice
        collector._add_import("os", ctx)
        collector._add_import("os", ctx)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1  # Should count only once
        assert metrics["imports"] == ["os"]

    def test_javascript_import_statement(self):
        """Test JavaScript import statement."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.js",
            source_code=b"import fs from 'fs'",
            language="javascript",
        )

        # Mock: import fs from 'fs'
        string_node = MockNode("string", text=b"'fs'")
        import_node = MockNode(
            "import_statement",
            children=[string_node],
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "fs" in metrics["imports"]
        assert "fs" in metrics["external_imports"]

    def test_javascript_relative_import(self):
        """Test JavaScript relative import."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.js",
            source_code=b"import utils from './utils'",
            language="javascript",
        )

        # Mock: import utils from './utils'
        string_node = MockNode("string", text=b"'./utils'")
        import_node = MockNode(
            "import_statement",
            children=[string_node],
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "./utils" in metrics["imports"]
        assert "./utils" in metrics["internal_imports"]

    def test_javascript_require_call(self):
        """Test JavaScript require() call."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.js",
            source_code=b"const fs = require('fs')",
            language="javascript",
        )

        # Mock: require('fs')
        string_node = MockNode("string", text=b"'fs'")
        args_node = MockNode("arguments", children=[string_node])
        function_node = MockNode("identifier", text=b"require")
        call_node = MockNode("call_expression", children=[function_node, args_node])

        # Override child_by_field_name
        call_node.child_by_field_name = lambda field: (
            function_node
            if field == "function"
            else args_node
            if field == "arguments"
            else None
        )

        collector.collect_node(call_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "fs" in metrics["imports"]
        assert "fs" in metrics["external_imports"]

    def test_typescript_import_type(self):
        """Test TypeScript type-only import."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.ts",
            source_code=b"import type { User } from './types'",
            language="typescript",
        )

        # Mock: import type { User } from './types'
        string_node = MockNode("string", text=b"'./types'")
        import_node = MockNode(
            "import_statement",
            children=[string_node],
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "./types" in metrics["imports"]
        assert "./types" in metrics["internal_imports"]

    def test_mixed_internal_external(self):
        """Test classification of internal vs external imports."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"",
            language="python",
        )

        # Add various imports
        collector._add_import("os", ctx)  # stdlib -> external
        collector._add_import("requests", ctx)  # third-party -> external
        collector._add_import(".utils", ctx)  # relative -> internal
        collector._add_import("..models", ctx)  # relative -> internal

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 4
        assert set(metrics["internal_imports"]) == {".utils", "..models"}
        assert set(metrics["external_imports"]) == {"os", "requests"}

    def test_no_imports(self):
        """Test file with no imports (Ce = 0)."""
        collector = EfferentCouplingCollector()
        # Create context but don't use it - collector tracks no imports
        CollectorContext(
            file_path="test.py",
            source_code=b"def foo(): pass",
            language="python",
        )

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 0
        assert metrics["imports"] == []
        assert metrics["internal_imports"] == []
        assert metrics["external_imports"] == []

    def test_finalize_function_returns_empty(self):
        """Test finalize_function returns empty dict (coupling is file-level)."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"",
            language="python",
        )

        mock_node = MockNode("function_definition")
        result = collector.finalize_function(mock_node, ctx)

        assert result == {}

    def test_reset(self):
        """Test reset clears all state."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"",
            language="python",
        )

        # Add imports
        collector._add_import("os", ctx)
        collector._add_import(".utils", ctx)

        assert len(collector._imports) == 2

        # Reset
        collector.reset()

        assert len(collector._imports) == 0
        assert len(collector._internal_imports) == 0
        assert len(collector._external_imports) == 0

    def test_sorted_output(self):
        """Test that imports are returned sorted."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"",
            language="python",
        )

        # Add imports in random order
        collector._add_import("sys", ctx)
        collector._add_import("os", ctx)
        collector._add_import("json", ctx)

        metrics = collector.get_file_metrics()
        assert metrics["imports"] == ["json", "os", "sys"]  # Alphabetically sorted

    def test_python_aliased_import(self):
        """Test Python aliased import (import os as operating_system)."""
        collector = EfferentCouplingCollector()
        ctx = CollectorContext(
            file_path="test.py",
            source_code=b"import os as operating_system",
            language="python",
        )

        # Mock: import os as operating_system
        dotted_name = MockNode("dotted_name", text=b"os")
        aliased_import = MockNode(
            "aliased_import",
            children=[dotted_name],
        )
        import_node = MockNode(
            "import_statement",
            children=[aliased_import],
        )

        collector.collect_node(import_node, ctx, 0)

        metrics = collector.get_file_metrics()
        assert metrics["efferent_coupling"] == 1
        assert "os" in metrics["imports"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
