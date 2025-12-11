"""Unit tests for Halstead complexity metrics collector."""

import math

import pytest

from mcp_vector_search.analysis.collectors.base import CollectorContext
from mcp_vector_search.analysis.collectors.halstead import (
    HalsteadCollector,
    HalsteadMetrics,
)


class MockNode:
    """Mock tree-sitter node for testing."""

    def __init__(self, node_type: str, text: str = "", children=None):
        """Initialize mock node.

        Args:
            node_type: Type of the node (e.g., 'function_definition')
            text: Text content of the node
            children: List of child nodes
        """
        self.type = node_type
        self.text = text.encode("utf-8") if text else b""
        self.children = children or []
        self.is_named = True


@pytest.fixture
def collector():
    """Create a fresh HalsteadCollector instance."""
    return HalsteadCollector()


@pytest.fixture
def context():
    """Create a basic CollectorContext."""
    return CollectorContext(
        file_path="test.py",
        source_code=b"def test(): pass",
        language="python",
    )


class TestHalsteadMetrics:
    """Test HalsteadMetrics dataclass and calculations."""

    def test_from_counts_basic(self):
        """Test basic metric calculation from counts."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        assert metrics.distinct_operators == 6
        assert metrics.distinct_operands == 3
        assert metrics.total_operators == 10
        assert metrics.total_operands == 8
        assert metrics.vocabulary == 9  # 6 + 3
        assert metrics.length == 18  # 10 + 8

    def test_from_counts_volume_calculation(self):
        """Test volume calculation: V = N × log₂(n)."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        # V = 18 × log₂(9) ≈ 18 × 3.17 ≈ 57.06
        expected_volume = 18 * math.log2(9)
        assert abs(metrics.volume - expected_volume) < 0.01

    def test_from_counts_difficulty_calculation(self):
        """Test difficulty calculation: D = (n1/2) × (N2/n2)."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        # D = (6/2) × (8/3) = 3 × 2.67 = 8
        expected_difficulty = (6 / 2) * (8 / 3)
        assert abs(metrics.difficulty - expected_difficulty) < 0.01

    def test_from_counts_effort_calculation(self):
        """Test effort calculation: E = D × V."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        expected_difficulty = (6 / 2) * (8 / 3)
        expected_volume = 18 * math.log2(9)
        expected_effort = expected_difficulty * expected_volume

        assert abs(metrics.effort - expected_effort) < 0.01

    def test_from_counts_time_calculation(self):
        """Test time calculation: T = E / 18."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        expected_time = metrics.effort / 18
        assert abs(metrics.time_seconds - expected_time) < 0.01

    def test_from_counts_bugs_calculation(self):
        """Test bugs calculation: B = V / 3000."""
        metrics = HalsteadMetrics.from_counts(n1=6, n2=3, N1=10, N2=8)

        expected_bugs = metrics.volume / 3000
        assert abs(metrics.estimated_bugs - expected_bugs) < 0.01

    def test_from_counts_zero_vocabulary(self):
        """Test edge case: zero vocabulary (no operators or operands)."""
        metrics = HalsteadMetrics.from_counts(n1=0, n2=0, N1=0, N2=0)

        assert metrics.vocabulary == 0
        assert metrics.length == 0
        assert metrics.volume == 0.0
        assert metrics.difficulty == 0.0
        assert metrics.effort == 0.0
        assert metrics.time_seconds == 0.0
        assert metrics.estimated_bugs == 0.0

    def test_from_counts_zero_operands(self):
        """Test edge case: zero operands (only operators)."""
        metrics = HalsteadMetrics.from_counts(n1=5, n2=0, N1=10, N2=0)

        assert metrics.vocabulary == 5
        assert metrics.length == 10
        assert metrics.volume > 0
        assert metrics.difficulty == 0.0  # Division by zero protection
        assert metrics.effort == 0.0

    def test_from_counts_single_operator_operand(self):
        """Test minimal code: one operator, one operand."""
        metrics = HalsteadMetrics.from_counts(n1=1, n2=1, N1=1, N2=1)

        assert metrics.vocabulary == 2
        assert metrics.length == 2
        # V = 2 × log₂(2) = 2 × 1 = 2
        assert abs(metrics.volume - 2.0) < 0.01

    def test_formulas_consistency(self):
        """Test that all formulas are consistent with each other."""
        n1, n2, N1, N2 = 10, 5, 20, 15  # noqa: N806 - Halstead notation
        metrics = HalsteadMetrics.from_counts(n1, n2, N1, N2)

        # Verify derived formulas
        assert metrics.vocabulary == n1 + n2
        assert metrics.length == N1 + N2
        assert abs(metrics.volume - (N1 + N2) * math.log2(n1 + n2)) < 0.01
        assert abs(metrics.difficulty - (n1 / 2) * (N2 / n2)) < 0.01
        assert abs(metrics.effort - metrics.difficulty * metrics.volume) < 0.01
        assert abs(metrics.time_seconds - metrics.effort / 18) < 0.01
        assert abs(metrics.estimated_bugs - metrics.volume / 3000) < 0.01


class TestHalsteadCollector:
    """Test HalsteadCollector with mock AST nodes."""

    def test_collector_initialization(self, collector):
        """Test collector starts with zero counts."""
        assert collector.name == "halstead"
        assert len(collector._operators) == 0
        assert len(collector._operands) == 0
        assert collector._total_operators == 0
        assert collector._total_operands == 0

    def test_simple_addition(self, collector, context):
        """Test simple addition: a + b."""
        # Create AST: binary_operator node with + operator
        node = MockNode("binary_operator", "+")

        collector.collect_node(node, context, depth=0)

        # Should count + as an operator
        assert "+" in collector._operators
        assert collector._total_operators == 1

    def test_identifier_as_operand(self, collector, context):
        """Test identifier is counted as operand."""
        node = MockNode("identifier", "variable_name")

        collector.collect_node(node, context, depth=0)

        # Should count identifier as operand
        assert "variable_name" in collector._operands
        assert collector._total_operands == 1

    def test_literal_as_operand(self, collector, context):
        """Test literals are counted as operands."""
        # Test integer literal
        int_node = MockNode("integer", "42")
        collector.collect_node(int_node, context, depth=0)
        assert "42" in collector._operands

        # Test float literal
        float_node = MockNode("float", "3.14")
        collector.collect_node(float_node, context, depth=0)
        assert "3.14" in collector._operands

        # Test string literal
        str_node = MockNode("string", '"hello"')
        collector.collect_node(str_node, context, depth=0)
        assert '"hello"' in collector._operands

    def test_boolean_literals(self, collector, context):
        """Test boolean literals are counted as operands."""
        true_node = MockNode("true", "True")
        false_node = MockNode("false", "False")

        collector.collect_node(true_node, context, depth=0)
        collector.collect_node(false_node, context, depth=0)

        assert "True" in collector._operands
        assert "False" in collector._operands

    def test_none_literal(self, collector, context):
        """Test None literal is counted as operand."""
        none_node = MockNode("none", "None")

        collector.collect_node(none_node, context, depth=0)

        assert "None" in collector._operands

    def test_control_flow_as_operators(self, collector, context):
        """Test control flow statements are counted as operators."""
        if_node = MockNode("if_statement", "if")
        for_node = MockNode("for_statement", "for")
        while_node = MockNode("while_statement", "while")

        collector.collect_node(if_node, context, depth=0)
        collector.collect_node(for_node, context, depth=0)
        collector.collect_node(while_node, context, depth=0)

        # Should count all control flow as operators
        assert collector._total_operators == 3

    def test_function_definition_as_operator(self, collector, context):
        """Test function definition is counted as operator."""
        func_node = MockNode("function_definition", "def")

        collector.collect_node(func_node, context, depth=0)

        assert collector._total_operators >= 1

    def test_assignment_as_operator(self, collector, context):
        """Test assignment is counted as operator."""
        assign_node = MockNode("assignment", "=")

        collector.collect_node(assign_node, context, depth=0)

        assert collector._total_operators >= 1

    def test_comparison_operators(self, collector, context):
        """Test comparison operators are counted."""
        operators = [
            MockNode("comparison_operator", "=="),
            MockNode("comparison_operator", "!="),
            MockNode("comparison_operator", "<"),
            MockNode("comparison_operator", ">"),
        ]

        for op in operators:
            collector.collect_node(op, context, depth=0)

        assert collector._total_operators == 4

    def test_logical_operators(self, collector, context):
        """Test logical operators are counted."""
        and_node = MockNode("boolean_operator", "and")
        or_node = MockNode("boolean_operator", "or")
        not_node = MockNode("unary_operator", "not")

        collector.collect_node(and_node, context, depth=0)
        collector.collect_node(or_node, context, depth=0)
        collector.collect_node(not_node, context, depth=0)

        assert collector._total_operators == 3

    def test_distinct_vs_total_counts(self, collector, context):
        """Test that distinct and total counts are tracked separately."""
        # Add same operator multiple times
        node1 = MockNode("binary_operator", "+")
        node2 = MockNode("binary_operator", "+")
        node3 = MockNode("binary_operator", "-")

        collector.collect_node(node1, context, depth=0)
        collector.collect_node(node2, context, depth=0)
        collector.collect_node(node3, context, depth=0)

        # Distinct: + and - = 2
        assert len(collector._operators) == 2
        # Total: 3 occurrences
        assert collector._total_operators == 3

    def test_operand_repetition(self, collector, context):
        """Test operand repetition tracking."""
        # Use same variable multiple times
        var1 = MockNode("identifier", "x")
        var2 = MockNode("identifier", "x")
        var3 = MockNode("identifier", "y")

        collector.collect_node(var1, context, depth=0)
        collector.collect_node(var2, context, depth=0)
        collector.collect_node(var3, context, depth=0)

        # Distinct: x and y = 2
        assert len(collector._operands) == 2
        # Total: 3 occurrences
        assert collector._total_operands == 3

    def test_finalize_function(self, collector, context):
        """Test finalize_function returns all expected metrics."""
        # Add some operators and operands
        collector.collect_node(MockNode("function_definition", "def"), context, 0)
        collector.collect_node(MockNode("identifier", "func"), context, 0)
        collector.collect_node(MockNode("return_statement", "return"), context, 0)
        collector.collect_node(MockNode("integer", "42"), context, 0)

        result = collector.finalize_function(MockNode("function_definition"), context)

        # Check all expected keys are present
        assert "halstead_volume" in result
        assert "halstead_difficulty" in result
        assert "halstead_effort" in result
        assert "halstead_bugs" in result
        assert "halstead_n1" in result
        assert "halstead_n2" in result
        assert "halstead_N1" in result
        assert "halstead_N2" in result

        # Check values are reasonable
        assert result["halstead_volume"] >= 0
        assert result["halstead_n1"] > 0  # Has operators
        assert result["halstead_n2"] > 0  # Has operands

    def test_reset_clears_state(self, collector, context):
        """Test that reset() clears all state."""
        # Add some data
        collector.collect_node(MockNode("binary_operator", "+"), context, 0)
        collector.collect_node(MockNode("identifier", "x"), context, 0)

        assert len(collector._operators) > 0
        assert len(collector._operands) > 0

        # Reset
        collector.reset()

        # All state should be cleared
        assert len(collector._operators) == 0
        assert len(collector._operands) == 0
        assert collector._total_operators == 0
        assert collector._total_operands == 0

    def test_empty_function_metrics(self, collector, context):
        """Test metrics for function with no operators or operands."""
        result = collector.finalize_function(MockNode("function_definition"), context)

        # Should handle empty case gracefully
        assert result["halstead_volume"] == 0.0
        assert result["halstead_difficulty"] == 0.0
        assert result["halstead_effort"] == 0.0
        assert result["halstead_bugs"] == 0.0

    def test_complex_expression(self, collector, context):
        """Test complex expression with multiple operators and operands."""
        # Simulate: result = (a + b) * (c - d)
        nodes = [
            MockNode("assignment", "="),
            MockNode("identifier", "result"),
            MockNode("binary_operator", "*"),
            MockNode("binary_operator", "+"),
            MockNode("identifier", "a"),
            MockNode("identifier", "b"),
            MockNode("binary_operator", "-"),
            MockNode("identifier", "c"),
            MockNode("identifier", "d"),
        ]

        for node in nodes:
            collector.collect_node(node, context, depth=0)

        result = collector.finalize_function(MockNode("function_definition"), context)

        # Operators: =, *, +, - = 4 distinct
        assert result["halstead_n1"] == 4
        # Operands: result, a, b, c, d = 5 distinct
        assert result["halstead_n2"] == 5
        # Total operators: 4
        assert result["halstead_N1"] == 4
        # Total operands: 5
        assert result["halstead_N2"] == 5

        # Volume should be positive
        assert result["halstead_volume"] > 0

    def test_keyword_not_counted_as_operand(self, collector, context):
        """Test that Python keywords are not counted as operands."""
        # These should be operators, not operands
        keywords = [
            MockNode("identifier", "if"),
            MockNode("identifier", "for"),
            MockNode("identifier", "while"),
            MockNode("identifier", "def"),
        ]

        for kw in keywords:
            collector.collect_node(kw, context, depth=0)

        # Keywords should not be in operands
        assert "if" not in collector._operands
        assert "for" not in collector._operands
        assert "while" not in collector._operands
        assert "def" not in collector._operands

    def test_realistic_function_metrics(self, collector, context):
        """Test realistic metrics for a typical function."""
        # Simulate: def add(a, b): return a + b
        nodes = [
            MockNode("function_definition", "def"),
            MockNode("identifier", "add"),
            MockNode("identifier", "a"),
            MockNode("identifier", "b"),
            MockNode("return_statement", "return"),
            MockNode("identifier", "a"),
            MockNode("binary_operator", "+"),
            MockNode("identifier", "b"),
        ]

        for node in nodes:
            collector.collect_node(node, context, depth=0)

        result = collector.finalize_function(MockNode("function_definition"), context)

        # Should have reasonable metrics
        assert result["halstead_n1"] >= 3  # def, return, +
        assert result["halstead_n2"] >= 3  # add, a, b
        assert result["halstead_volume"] > 0
        assert result["halstead_difficulty"] > 0

        # Simple function should have low bug estimate
        assert result["halstead_bugs"] < 0.1

    def test_volume_scales_with_complexity(self, collector, context):
        """Test that volume increases with code complexity."""
        # Simple code
        simple_collector = HalsteadCollector()
        simple_nodes = [
            MockNode("return_statement", "return"),
            MockNode("integer", "1"),
        ]
        for node in simple_nodes:
            simple_collector.collect_node(node, context, 0)
        simple_result = simple_collector.finalize_function(
            MockNode("function_definition"), context
        )

        # Complex code
        complex_collector = HalsteadCollector()
        complex_nodes = [
            MockNode("if_statement", "if"),
            MockNode("identifier", "x"),
            MockNode("comparison_operator", ">"),
            MockNode("integer", "0"),
            MockNode("return_statement", "return"),
            MockNode("identifier", "x"),
            MockNode("binary_operator", "+"),
            MockNode("identifier", "y"),
            MockNode("else_clause", "else"),
            MockNode("return_statement", "return"),
            MockNode("integer", "0"),
        ]
        for node in complex_nodes:
            complex_collector.collect_node(node, context, 0)
        complex_result = complex_collector.finalize_function(
            MockNode("function_definition"), context
        )

        # Complex code should have higher volume
        assert complex_result["halstead_volume"] > simple_result["halstead_volume"]

    def test_get_node_text(self, collector):
        """Test _get_node_text helper method."""
        node = MockNode("identifier", "test_name")
        source = b"test_name"

        text = collector._get_node_text(node, source)
        assert text == "test_name"

    def test_is_operator_detection(self, collector):
        """Test _is_operator method."""
        # Should detect operators
        assert collector._is_operator("binary_operator", "+", "python")
        assert collector._is_operator("if_statement", "if", "python")
        assert collector._is_operator("function_definition", "def", "python")

        # Should not detect operands
        assert not collector._is_operator("identifier", "variable", "python")
        assert not collector._is_operator("integer", "42", "python")

    def test_is_operand_detection(self, collector):
        """Test _is_operand method."""
        # Should detect operands
        assert collector._is_operand("identifier", "variable", "python")
        assert collector._is_operand("integer", "42", "python")
        assert collector._is_operand("string", '"hello"', "python")

        # Should not detect operators
        assert not collector._is_operand("binary_operator", "+", "python")
        assert not collector._is_operand("if_statement", "if", "python")

        # Keywords should not be operands
        assert not collector._is_operand("identifier", "if", "python")
        assert not collector._is_operand("identifier", "for", "python")
