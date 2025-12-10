"""Unit tests for metric collector base classes."""

import pytest

from mcp_vector_search.analysis import CollectorContext, MetricCollector


class TestCollectorContext:
    """Test CollectorContext dataclass."""

    def test_initialization_required_fields(self):
        """Test context initialization with required fields."""
        ctx = CollectorContext(
            file_path="/test/path/test.py",
            source_code=b"def foo(): pass",
            language="python",
        )

        assert ctx.file_path == "/test/path/test.py"
        assert ctx.source_code == b"def foo(): pass"
        assert ctx.language == "python"

    def test_initialization_optional_fields(self):
        """Test optional fields have correct defaults."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        assert ctx.current_function is None
        assert ctx.current_class is None
        assert ctx.nesting_stack == []

    def test_source_code_bytes_type(self):
        """Test source_code accepts bytes (required by tree-sitter)."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"def foo(): pass", language="python"
        )

        assert isinstance(ctx.source_code, bytes)

    def test_current_function_mutation(self):
        """Test current_function can be modified during traversal."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        assert ctx.current_function is None

        ctx.current_function = "my_function"
        assert ctx.current_function == "my_function"

        ctx.current_function = None
        assert ctx.current_function is None

    def test_current_class_mutation(self):
        """Test current_class can be modified during traversal."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        assert ctx.current_class is None

        ctx.current_class = "MyClass"
        assert ctx.current_class == "MyClass"

        ctx.current_class = None
        assert ctx.current_class is None

    def test_nesting_stack_operations(self):
        """Test nesting_stack supports stack operations."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        assert ctx.nesting_stack == []

        # Test push
        ctx.nesting_stack.append("function")
        assert len(ctx.nesting_stack) == 1
        assert ctx.nesting_stack[-1] == "function"

        ctx.nesting_stack.append("if_statement")
        assert len(ctx.nesting_stack) == 2
        assert ctx.nesting_stack[-1] == "if_statement"

        # Test pop
        popped = ctx.nesting_stack.pop()
        assert popped == "if_statement"
        assert len(ctx.nesting_stack) == 1

        # Test clear
        ctx.nesting_stack.clear()
        assert ctx.nesting_stack == []

    def test_nesting_depth_calculation(self):
        """Test nesting depth can be calculated from stack."""
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Depth 0
        assert len(ctx.nesting_stack) == 0

        # Depth 1
        ctx.nesting_stack.append("function")
        assert len(ctx.nesting_stack) == 1

        # Depth 2
        ctx.nesting_stack.append("if")
        assert len(ctx.nesting_stack) == 2

        # Depth 3
        ctx.nesting_stack.append("for")
        assert len(ctx.nesting_stack) == 3

    def test_multiple_contexts_independent(self):
        """Test multiple contexts don't share state."""
        ctx1 = CollectorContext(
            file_path="file1.py", source_code=b"code1", language="python"
        )
        ctx2 = CollectorContext(
            file_path="file2.py", source_code=b"code2", language="javascript"
        )

        ctx1.current_function = "func1"
        ctx1.nesting_stack.append("if")

        assert ctx2.current_function is None
        assert ctx2.nesting_stack == []

    def test_supported_languages(self):
        """Test context supports various language identifiers."""
        languages = ["python", "javascript", "typescript", "java", "go", "rust"]

        for lang in languages:
            ctx = CollectorContext(
                file_path=f"test.{lang}", source_code=b"code", language=lang
            )
            assert ctx.language == lang


class TestMetricCollector:
    """Test MetricCollector ABC."""

    def test_is_abstract(self):
        """Verify MetricCollector cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MetricCollector()  # type: ignore

    def test_abstract_methods_required(self):
        """Verify all abstract methods must be implemented."""

        # Missing all abstract methods
        class IncompleteCollector(MetricCollector):
            pass

        with pytest.raises(TypeError):
            IncompleteCollector()  # type: ignore

        # Missing collect_node
        class MissingCollectNode(MetricCollector):
            @property
            def name(self) -> str:
                return "test"

            def finalize_function(self, node, context):
                return {}

        with pytest.raises(TypeError):
            MissingCollectNode()  # type: ignore

        # Missing finalize_function
        class MissingFinalize(MetricCollector):
            @property
            def name(self) -> str:
                return "test"

            def collect_node(self, node, context, depth):
                pass

        with pytest.raises(TypeError):
            MissingFinalize()  # type: ignore

    def test_concrete_implementation(self):
        """Test concrete implementation with all abstract methods."""

        class ConcreteCollector(MetricCollector):
            def __init__(self):
                self._count = 0

            @property
            def name(self) -> str:
                return "test_collector"

            def collect_node(self, node, context, depth):
                self._count += 1

            def finalize_function(self, node, context):
                return {"count": self._count}

        # Should instantiate successfully
        collector = ConcreteCollector()
        assert collector.name == "test_collector"

        # Test collect_node
        collector.collect_node(None, None, 0)  # type: ignore
        assert collector._count == 1

        # Test finalize_function
        result = collector.finalize_function(None, None)  # type: ignore
        assert result == {"count": 1}

    def test_reset_default_implementation(self):
        """Test reset() has default no-op implementation."""

        class MinimalCollector(MetricCollector):
            @property
            def name(self) -> str:
                return "minimal"

            def collect_node(self, node, context, depth):
                pass

            def finalize_function(self, node, context):
                return {}

        collector = MinimalCollector()

        # reset() should exist and do nothing
        result = collector.reset()
        assert result is None

    def test_reset_custom_implementation(self):
        """Test reset() can be overridden for stateful collectors."""

        class StatefulCollector(MetricCollector):
            def __init__(self):
                self._complexity = 0
                self._nesting_stack = []

            @property
            def name(self) -> str:
                return "stateful"

            def collect_node(self, node, context, depth):
                self._complexity += 1
                self._nesting_stack.append(depth)

            def finalize_function(self, node, context):
                return {
                    "complexity": self._complexity,
                    "depth": len(self._nesting_stack),
                }

            def reset(self):
                self._complexity = 0
                self._nesting_stack.clear()

        collector = StatefulCollector()

        # Accumulate state
        collector.collect_node(None, None, 1)  # type: ignore
        collector.collect_node(None, None, 2)  # type: ignore
        assert collector._complexity == 2
        assert len(collector._nesting_stack) == 2

        # Reset should clear state
        collector.reset()
        assert collector._complexity == 0
        assert collector._nesting_stack == []

    def test_collector_workflow(self):
        """Test typical collector lifecycle workflow."""

        class WorkflowCollector(MetricCollector):
            def __init__(self):
                self._if_count = 0
                self._for_count = 0

            @property
            def name(self) -> str:
                return "workflow"

            def collect_node(self, node, context, depth):
                # Simulate counting different node types
                if hasattr(node, "type"):
                    if node.type == "if_statement":
                        self._if_count += 1
                    elif node.type == "for_statement":
                        self._for_count += 1

            def finalize_function(self, node, context):
                return {"if_count": self._if_count, "for_count": self._for_count}

            def reset(self):
                self._if_count = 0
                self._for_count = 0

        collector = WorkflowCollector()

        # Simulate first function
        class MockNode:
            def __init__(self, node_type):
                self.type = node_type

        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("if_statement"), ctx, 1)
        collector.collect_node(MockNode("if_statement"), ctx, 2)
        collector.collect_node(MockNode("for_statement"), ctx, 2)

        result1 = collector.finalize_function(None, ctx)  # type: ignore
        assert result1 == {"if_count": 2, "for_count": 1}

        # Reset for second function
        collector.reset()

        collector.collect_node(MockNode("for_statement"), ctx, 1)
        result2 = collector.finalize_function(None, ctx)  # type: ignore
        assert result2 == {"if_count": 0, "for_count": 1}

    def test_multiple_collectors(self):
        """Test multiple collectors can coexist independently."""

        class CounterCollector(MetricCollector):
            def __init__(self):
                self._count = 0

            @property
            def name(self) -> str:
                return "counter"

            def collect_node(self, node, context, depth):
                self._count += 1

            def finalize_function(self, node, context):
                return {"count": self._count}

        class DepthCollector(MetricCollector):
            def __init__(self):
                self._max_depth = 0

            @property
            def name(self) -> str:
                return "depth"

            def collect_node(self, node, context, depth):
                self._max_depth = max(self._max_depth, depth)

            def finalize_function(self, node, context):
                return {"max_depth": self._max_depth}

        counter = CounterCollector()
        depth = DepthCollector()

        # Simulate traversal
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        counter.collect_node(None, ctx, 1)  # type: ignore
        counter.collect_node(None, ctx, 2)  # type: ignore
        depth.collect_node(None, ctx, 1)  # type: ignore
        depth.collect_node(None, ctx, 3)  # type: ignore

        # Each collector maintains independent state
        assert counter.finalize_function(None, ctx) == {"count": 2}  # type: ignore
        assert depth.finalize_function(None, ctx) == {"max_depth": 3}  # type: ignore

    def test_collector_name_property(self):
        """Test name property is accessible."""

        class NamedCollector(MetricCollector):
            @property
            def name(self) -> str:
                return "my_custom_collector"

            def collect_node(self, node, context, depth):
                pass

            def finalize_function(self, node, context):
                return {}

        collector = NamedCollector()
        assert collector.name == "my_custom_collector"
        assert isinstance(collector.name, str)
