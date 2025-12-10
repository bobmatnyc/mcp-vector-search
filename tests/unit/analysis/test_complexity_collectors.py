"""Unit tests for complexity metric collectors."""

from mcp_vector_search.analysis import (
    CognitiveComplexityCollector,
    CollectorContext,
    CyclomaticComplexityCollector,
    MethodCountCollector,
    NestingDepthCollector,
    ParameterCountCollector,
)
from mcp_vector_search.analysis.collectors.complexity import get_node_types


class TestGetNodeTypes:
    """Test multi-language node type lookup helper."""

    def test_python_node_types(self):
        """Test Python node type mappings."""
        assert get_node_types("python", "function_def") == ["function_definition"]
        assert get_node_types("python", "if_statement") == ["if_statement"]
        assert get_node_types("python", "boolean_and") == ["and"]
        assert get_node_types("python", "boolean_or") == ["or"]

    def test_javascript_node_types(self):
        """Test JavaScript node type mappings."""
        assert "function_declaration" in get_node_types("javascript", "function_def")
        assert "arrow_function" in get_node_types("javascript", "function_def")
        assert get_node_types("javascript", "boolean_and") == ["&&"]
        assert get_node_types("javascript", "boolean_or") == ["||"]

    def test_typescript_node_types(self):
        """Test TypeScript node type mappings."""
        assert "method_signature" in get_node_types("typescript", "function_def")
        assert get_node_types("typescript", "if_statement") == ["if_statement"]

    def test_java_node_types(self):
        """Test Java node type mappings."""
        assert "method_declaration" in get_node_types("java", "function_def")
        assert "class_declaration" in get_node_types("java", "class_def")
        assert get_node_types("java", "boolean_and") == ["&&"]

    def test_rust_node_types(self):
        """Test Rust node type mappings."""
        assert get_node_types("rust", "function_def") == ["function_item"]
        assert get_node_types("rust", "match_statement") == ["match_expression"]
        assert get_node_types("rust", "case_clause") == ["match_arm"]

    def test_php_node_types(self):
        """Test PHP node type mappings."""
        assert "function_definition" in get_node_types("php", "function_def")
        assert "trait_declaration" in get_node_types("php", "class_def")
        assert "and" in get_node_types("php", "boolean_and")
        assert "&&" in get_node_types("php", "boolean_and")

    def test_ruby_node_types(self):
        """Test Ruby node type mappings."""
        assert "method" in get_node_types("ruby", "function_def")
        assert "module" in get_node_types("ruby", "class_def")
        assert get_node_types("ruby", "elif_clause") == ["elsif"]

    def test_unknown_language_fallback(self):
        """Test unknown language falls back to Python-like behavior."""
        # Unknown language should fall back to Python
        result = get_node_types("unknown_lang", "function_def")
        assert result == ["function_definition"]

    def test_unknown_category(self):
        """Test unknown category returns empty list."""
        assert get_node_types("python", "nonexistent_category") == []
        assert get_node_types("javascript", "made_up_node") == []


class MockNode:
    """Mock tree-sitter node for testing."""

    def __init__(self, node_type: str, children=None):
        self.type = node_type
        self.children = children or []
        self.is_named = True

    def child_by_field_name(self, field: str):
        """Mock field-based child lookup."""
        return None


class TestCognitiveComplexityCollector:
    """Test CognitiveComplexityCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = CognitiveComplexityCollector()
        assert collector.name == "cognitive_complexity"
        assert collector._complexity == 0
        assert collector._nesting_level == 0

    def test_simple_if_statement(self):
        """Test if statement adds +1 complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Single if statement at nesting level 0
        node = MockNode("if_statement")
        collector.collect_node(node, ctx, 0)

        result = collector.finalize_function(node, ctx)
        assert result["cognitive_complexity"] == 1

    def test_nested_if_statements(self):
        """Test nested if statements increase complexity with nesting penalty."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # First if at level 0: +1
        collector.collect_node(MockNode("if_statement"), ctx, 0)
        assert collector._complexity == 1
        assert collector._nesting_level == 1

        # Second if at level 1: +1 (statement) +1 (nesting) = +2
        collector.collect_node(MockNode("if_statement"), ctx, 1)
        assert collector._complexity == 3

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 3

    def test_elif_clause(self):
        """Test elif clause adds complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("elif_clause"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_for_loop(self):
        """Test for loop adds complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("for_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_while_loop(self):
        """Test while loop adds complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("while_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_break_statement(self):
        """Test break statement adds +1 complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("break_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_continue_statement(self):
        """Test continue statement adds +1 complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("continue_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_boolean_operators(self):
        """Test boolean operators add complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Test 'and'
        collector.collect_node(MockNode("and"), ctx, 0)
        assert collector._complexity == 1

        # Test 'or'
        collector.collect_node(MockNode("or"), ctx, 0)
        assert collector._complexity == 2

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 2

    def test_javascript_boolean_operators(self):
        """Test JavaScript && and || operators."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.js", source_code=b"code", language="javascript"
        )

        # JavaScript uses && and ||
        collector.collect_node(MockNode("&&"), ctx, 0)
        collector.collect_node(MockNode("||"), ctx, 0)

        result = collector.finalize_function(MockNode("function_declaration"), ctx)
        assert result["cognitive_complexity"] == 2

    def test_match_statement(self):
        """Test match/switch statement adds complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("match_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_case_clause(self):
        """Test case clause adds +1 complexity."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("case_clause"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cognitive_complexity"] == 1

    def test_reset(self):
        """Test reset clears collector state."""
        collector = CognitiveComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Accumulate state
        collector.collect_node(MockNode("if_statement"), ctx, 0)
        collector.collect_node(MockNode("for_statement"), ctx, 1)
        assert collector._complexity > 0

        # Reset
        collector.reset()
        assert collector._complexity == 0
        assert collector._nesting_level == 0
        assert collector._current_function_name is None


class TestCyclomaticComplexityCollector:
    """Test CyclomaticComplexityCollector."""

    def test_initialization(self):
        """Test collector initializes with baseline complexity of 1."""
        collector = CyclomaticComplexityCollector()
        assert collector.name == "cyclomatic_complexity"
        assert collector._complexity == 1

    def test_baseline_complexity(self):
        """Test baseline complexity is 1 for simple function."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # No decision points
        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 1

    def test_if_statement(self):
        """Test if statement adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("if_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 2

    def test_elif_clause(self):
        """Test elif clause adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("if_statement"), ctx, 0)
        collector.collect_node(MockNode("elif_clause"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 3

    def test_for_loop(self):
        """Test for loop adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("for_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 2

    def test_while_loop(self):
        """Test while loop adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("while_statement"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 2

    def test_except_clause(self):
        """Test except/catch clause adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("except_clause"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 2

    def test_ternary_expression(self):
        """Test ternary expression adds +1 complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("conditional_expression"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["cyclomatic_complexity"] == 2

    def test_boolean_operators(self):
        """Test boolean operators add complexity."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        collector.collect_node(MockNode("and"), ctx, 0)
        collector.collect_node(MockNode("or"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        # 1 (baseline) + 1 (and) + 1 (or) = 3
        assert result["cyclomatic_complexity"] == 3

    def test_complex_function(self):
        """Test complex function with multiple decision points."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Simulate function with: if, elif, for, while, try-except
        collector.collect_node(MockNode("if_statement"), ctx, 0)
        collector.collect_node(MockNode("elif_clause"), ctx, 0)
        collector.collect_node(MockNode("for_statement"), ctx, 1)
        collector.collect_node(MockNode("while_statement"), ctx, 2)
        collector.collect_node(MockNode("except_clause"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        # 1 (baseline) + 1 (if) + 1 (elif) + 1 (for) + 1 (while) + 1 (except) = 6
        assert result["cyclomatic_complexity"] == 6

    def test_reset(self):
        """Test reset returns complexity to baseline."""
        collector = CyclomaticComplexityCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Accumulate complexity
        collector.collect_node(MockNode("if_statement"), ctx, 0)
        collector.collect_node(MockNode("for_statement"), ctx, 0)
        assert collector._complexity == 3

        # Reset
        collector.reset()
        assert collector._complexity == 1


class TestNestingDepthCollector:
    """Test NestingDepthCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = NestingDepthCollector()
        assert collector.name == "nesting_depth"
        assert collector._max_depth == 0
        assert collector._current_depth == 0

    def test_no_nesting(self):
        """Test function with no nesting has depth 0."""
        collector = NestingDepthCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["max_nesting_depth"] == 0

    def test_single_if_statement(self):
        """Test single if statement creates depth 1."""
        collector = NestingDepthCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Simulate entering if statement
        ctx.nesting_stack.append("if")
        collector.collect_node(MockNode("if_statement"), ctx, 1)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["max_nesting_depth"] == 1

    def test_nested_structures(self):
        """Test nested structures increase depth."""
        collector = NestingDepthCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Simulate: if -> for -> while
        ctx.nesting_stack.append("if")
        collector.collect_node(MockNode("if_statement"), ctx, 1)

        ctx.nesting_stack.append("for")
        collector.collect_node(MockNode("for_statement"), ctx, 2)

        ctx.nesting_stack.append("while")
        collector.collect_node(MockNode("while_statement"), ctx, 3)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["max_nesting_depth"] == 3

    def test_max_depth_tracking(self):
        """Test max depth is tracked correctly."""
        collector = NestingDepthCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Enter depth 1
        ctx.nesting_stack.append("if")
        collector.collect_node(MockNode("if_statement"), ctx, 1)

        # Enter depth 2
        ctx.nesting_stack.append("for")
        collector.collect_node(MockNode("for_statement"), ctx, 2)

        # Exit depth 2, back to depth 1
        ctx.nesting_stack.pop()

        # Enter depth 2 again
        ctx.nesting_stack.append("while")
        collector.collect_node(MockNode("while_statement"), ctx, 2)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        # Max should be 2, not higher
        assert result["max_nesting_depth"] == 2

    def test_reset(self):
        """Test reset clears depth tracking."""
        collector = NestingDepthCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Accumulate depth
        ctx.nesting_stack.append("if")
        collector.collect_node(MockNode("if_statement"), ctx, 1)
        assert collector._max_depth > 0

        # Reset
        collector.reset()
        assert collector._max_depth == 0
        assert collector._current_depth == 0


class TestParameterCountCollector:
    """Test ParameterCountCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = ParameterCountCollector()
        assert collector.name == "parameter_count"
        assert collector._parameter_count == 0

    def test_no_parameters(self):
        """Test function with no parameters."""
        collector = ParameterCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Empty parameter list
        params_node = MockNode("parameters", children=[])
        collector.collect_node(params_node, ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["parameter_count"] == 0

    def test_single_parameter(self):
        """Test function with single parameter."""
        collector = ParameterCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Parameter list with one identifier
        param = MockNode("identifier")
        params_node = MockNode("parameters", children=[param])
        collector.collect_node(params_node, ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["parameter_count"] == 1

    def test_multiple_parameters(self):
        """Test function with multiple parameters."""
        collector = ParameterCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Parameter list with three identifiers
        param1 = MockNode("identifier")
        param2 = MockNode("identifier")
        param3 = MockNode("identifier")
        params_node = MockNode("parameters", children=[param1, param2, param3])
        collector.collect_node(params_node, ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["parameter_count"] == 3

    def test_ignores_punctuation(self):
        """Test parameter counting ignores commas and parentheses."""
        collector = ParameterCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Parameter list with commas and parentheses
        param1 = MockNode("identifier")
        comma = MockNode(",")
        comma.is_named = False
        param2 = MockNode("identifier")
        params_node = MockNode("parameters", children=[param1, comma, param2])
        collector.collect_node(params_node, ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["parameter_count"] == 2

    def test_reset(self):
        """Test reset clears parameter count."""
        collector = ParameterCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Count parameters
        params_node = MockNode(
            "parameters",
            children=[MockNode("identifier"), MockNode("identifier")],
        )
        collector.collect_node(params_node, ctx, 0)
        assert collector._parameter_count == 2

        # Reset
        collector.reset()
        assert collector._parameter_count == 0


class TestMethodCountCollector:
    """Test MethodCountCollector."""

    def test_initialization(self):
        """Test collector initializes correctly."""
        collector = MethodCountCollector()
        assert collector.name == "method_count"
        assert collector._method_count == 0
        assert collector._inside_class is False

    def test_no_methods(self):
        """Test class with no methods returns 0."""
        collector = MethodCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        result = collector.finalize_function(MockNode("class_definition"), ctx)
        assert result["method_count"] == 0

    def test_single_method(self):
        """Test class with single method."""
        collector = MethodCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )

        # Enter class
        collector.collect_node(MockNode("class_definition"), ctx, 0)
        assert collector._inside_class is True

        # Count method
        collector.collect_node(MockNode("function_definition"), ctx, 1)

        result = collector.finalize_function(MockNode("class_definition"), ctx)
        assert result["method_count"] == 1

    def test_multiple_methods(self):
        """Test class with multiple methods."""
        collector = MethodCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )
        ctx.current_class = "MyClass"

        # Count methods
        collector.collect_node(MockNode("function_definition"), ctx, 1)
        collector.collect_node(MockNode("function_definition"), ctx, 1)
        collector.collect_node(MockNode("function_definition"), ctx, 1)

        result = collector.finalize_function(MockNode("class_definition"), ctx)
        assert result["method_count"] == 3

    def test_top_level_function_not_counted(self):
        """Test top-level functions are not counted as methods."""
        collector = MethodCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )
        # No class context

        # This is a top-level function, not a method
        collector.collect_node(MockNode("function_definition"), ctx, 0)

        result = collector.finalize_function(MockNode("function_definition"), ctx)
        assert result["method_count"] == 0

    def test_reset(self):
        """Test reset clears method count."""
        collector = MethodCountCollector()
        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )
        ctx.current_class = "MyClass"

        # Count methods
        collector.collect_node(MockNode("function_definition"), ctx, 1)
        assert collector._method_count == 1

        # Reset
        collector.reset()
        assert collector._method_count == 0
        assert collector._inside_class is False


class TestCollectorIntegration:
    """Test multiple collectors working together."""

    def test_all_collectors_independent(self):
        """Test all collectors can run independently without interference."""
        cognitive = CognitiveComplexityCollector()
        cyclomatic = CyclomaticComplexityCollector()
        nesting = NestingDepthCollector()
        params = ParameterCountCollector()
        methods = MethodCountCollector()

        ctx = CollectorContext(
            file_path="test.py", source_code=b"code", language="python"
        )
        ctx.current_class = "TestClass"

        # Simulate complex function
        if_node = MockNode("if_statement")
        for_node = MockNode("for_statement")
        params_node = MockNode("parameters", children=[MockNode("identifier")])

        # Process with all collectors
        for collector in [cognitive, cyclomatic, nesting, params, methods]:
            collector.collect_node(if_node, ctx, 0)
            collector.collect_node(for_node, ctx, 1)
            collector.collect_node(params_node, ctx, 0)

        # Each collector should produce independent results
        cog_result = cognitive.finalize_function(MockNode("function_definition"), ctx)
        cyc_result = cyclomatic.finalize_function(MockNode("function_definition"), ctx)
        nest_result = nesting.finalize_function(MockNode("function_definition"), ctx)
        param_result = params.finalize_function(MockNode("function_definition"), ctx)
        method_result = methods.finalize_function(MockNode("function_definition"), ctx)

        # Verify all collectors returned values
        assert "cognitive_complexity" in cog_result
        assert "cyclomatic_complexity" in cyc_result
        assert "max_nesting_depth" in nest_result
        assert "parameter_count" in param_result
        assert "method_count" in method_result

        # Verify independence - modifying one doesn't affect others
        cognitive.reset()
        assert cognitive._complexity == 0
        assert cyclomatic._complexity > 1  # Should still have accumulated value
