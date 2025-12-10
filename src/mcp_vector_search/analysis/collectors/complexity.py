"""Complexity metric collectors for structural code analysis.

This module provides collectors for various complexity metrics:
- CognitiveComplexityCollector: Measures how hard code is to understand
- CyclomaticComplexityCollector: Counts independent execution paths
- NestingDepthCollector: Tracks maximum nesting level
- ParameterCountCollector: Counts function parameters
- MethodCountCollector: Counts methods in classes

All collectors support multiple languages via tree-sitter node type mappings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import CollectorContext, MetricCollector

if TYPE_CHECKING:
    from tree_sitter import Node


# Multi-language node type mappings
# Maps logical categories to language-specific tree-sitter node types
LANGUAGE_NODE_TYPES = {
    "python": {
        "function_def": ["function_definition"],
        "class_def": ["class_definition"],
        "if_statement": ["if_statement"],
        "elif_clause": ["elif_clause"],
        "else_clause": ["else_clause"],
        "for_loop": ["for_statement"],
        "while_loop": ["while_statement"],
        "try_statement": ["try_statement"],
        "except_clause": ["except_clause"],
        "with_statement": ["with_statement"],
        "match_statement": ["match_statement"],
        "case_clause": ["case_clause"],
        "ternary": ["conditional_expression"],
        "boolean_and": ["and"],
        "boolean_or": ["or"],
        "parameters": ["parameters"],
        "break": ["break_statement"],
        "continue": ["continue_statement"],
        "return": ["return_statement"],
    },
    "javascript": {
        "function_def": [
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
        ],
        "class_def": ["class_declaration", "class"],
        "if_statement": ["if_statement"],
        "elif_clause": [],  # JS uses else if, not elif
        "else_clause": ["else_clause"],
        "for_loop": ["for_statement", "for_in_statement"],
        "while_loop": ["while_statement"],
        "try_statement": ["try_statement"],
        "except_clause": ["catch_clause"],
        "with_statement": [],
        "match_statement": ["switch_statement"],
        "case_clause": ["switch_case"],
        "ternary": ["ternary_expression"],
        "boolean_and": ["&&"],
        "boolean_or": ["||"],
        "parameters": ["formal_parameters"],
        "break": ["break_statement"],
        "continue": ["continue_statement"],
        "return": ["return_statement"],
    },
    "typescript": {
        "function_def": [
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "method_signature",
        ],
        "class_def": ["class_declaration", "class"],
        "if_statement": ["if_statement"],
        "elif_clause": [],
        "else_clause": ["else_clause"],
        "for_loop": ["for_statement", "for_in_statement"],
        "while_loop": ["while_statement"],
        "try_statement": ["try_statement"],
        "except_clause": ["catch_clause"],
        "with_statement": [],
        "match_statement": ["switch_statement"],
        "case_clause": ["switch_case"],
        "ternary": ["ternary_expression"],
        "boolean_and": ["&&"],
        "boolean_or": ["||"],
        "parameters": ["formal_parameters"],
        "break": ["break_statement"],
        "continue": ["continue_statement"],
        "return": ["return_statement"],
    },
    "java": {
        "function_def": ["method_declaration", "constructor_declaration"],
        "class_def": ["class_declaration", "interface_declaration"],
        "if_statement": ["if_statement"],
        "elif_clause": [],  # Java uses else if
        "else_clause": ["else"],
        "for_loop": ["for_statement", "enhanced_for_statement"],
        "while_loop": ["while_statement"],
        "try_statement": ["try_statement"],
        "except_clause": ["catch_clause"],
        "with_statement": ["try_with_resources_statement"],
        "match_statement": ["switch_expression", "switch_statement"],
        "case_clause": ["switch_label"],
        "ternary": ["ternary_expression"],
        "boolean_and": ["&&"],
        "boolean_or": ["||"],
        "parameters": ["formal_parameters"],
        "break": ["break_statement"],
        "continue": ["continue_statement"],
        "return": ["return_statement"],
    },
    "rust": {
        "function_def": ["function_item"],
        "class_def": ["struct_item", "impl_item", "trait_item"],
        "if_statement": ["if_expression"],
        "elif_clause": [],  # Rust uses else if
        "else_clause": ["else_clause"],
        "for_loop": ["for_expression"],
        "while_loop": ["while_expression"],
        "try_statement": [],  # Rust uses Result/Option, not try
        "except_clause": [],
        "with_statement": [],
        "match_statement": ["match_expression"],
        "case_clause": ["match_arm"],
        "ternary": [],  # Rust uses if-else expressions
        "boolean_and": ["&&"],
        "boolean_or": ["||"],
        "parameters": ["parameters"],
        "break": ["break_expression"],
        "continue": ["continue_expression"],
        "return": ["return_expression"],
    },
    "php": {
        "function_def": ["function_definition", "method_declaration"],
        "class_def": [
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        ],
        "if_statement": ["if_statement"],
        "elif_clause": ["else_if_clause"],
        "else_clause": ["else_clause"],
        "for_loop": ["for_statement", "foreach_statement"],
        "while_loop": ["while_statement"],
        "try_statement": ["try_statement"],
        "except_clause": ["catch_clause"],
        "with_statement": [],
        "match_statement": ["switch_statement", "match_expression"],
        "case_clause": ["case_statement", "match_arm"],
        "ternary": ["conditional_expression"],
        "boolean_and": ["and", "&&"],
        "boolean_or": ["or", "||"],
        "parameters": ["formal_parameters"],
        "break": ["break_statement"],
        "continue": ["continue_statement"],
        "return": ["return_statement"],
    },
    "ruby": {
        "function_def": ["method", "singleton_method"],
        "class_def": ["class", "module"],
        "if_statement": ["if", "unless"],
        "elif_clause": ["elsif"],
        "else_clause": ["else"],
        "for_loop": ["for"],
        "while_loop": ["while", "until"],
        "try_statement": ["begin"],
        "except_clause": ["rescue"],
        "with_statement": [],
        "match_statement": ["case"],
        "case_clause": ["when"],
        "ternary": ["conditional"],
        "boolean_and": ["and", "&&"],
        "boolean_or": ["or", "||"],
        "parameters": ["method_parameters"],
        "break": ["break"],
        "continue": ["next"],
        "return": ["return"],
    },
}


def get_node_types(language: str, category: str) -> list[str]:
    """Get tree-sitter node types for a given language and category.

    Provides language-agnostic access to node types by mapping logical
    categories (e.g., "if_statement", "function_def") to language-specific
    tree-sitter node type names.

    Args:
        language: Programming language identifier (e.g., "python", "javascript")
        category: Logical category of node (e.g., "if_statement", "for_loop")

    Returns:
        List of node type names for this language/category combination.
        Returns empty list if language/category not found.

    Examples:
        >>> get_node_types("python", "function_def")
        ["function_definition"]

        >>> get_node_types("javascript", "function_def")
        ["function_declaration", "function", "arrow_function", "method_definition"]

        >>> get_node_types("unknown_lang", "if_statement")
        []  # Falls back to Python-like behavior
    """
    # Default to Python-like node types for unknown languages
    lang_mapping = LANGUAGE_NODE_TYPES.get(language, LANGUAGE_NODE_TYPES["python"])
    return lang_mapping.get(category, [])


class CognitiveComplexityCollector(MetricCollector):
    """Tracks cognitive complexity - how hard code is to understand.

    Cognitive complexity measures the difficulty of understanding code flow
    by penalizing nested control structures and complex boolean logic.

    Scoring Rules:
    - +1 for each: if, elif, else, for, while, catch/except, ternary
    - +1 for each nesting level (nested if inside if gets +2 total)
    - +1 for each: break, continue, goto
    - +1 for boolean operators: and, or, &&, ||
    - +1 for recursion (function calling itself)

    The final score indicates code readability:
    - 0-5: Excellent (Grade A)
    - 6-10: Good (Grade B)
    - 11-20: Acceptable (Grade C)
    - 21-30: Needs improvement (Grade D)
    - 31+: Refactor recommended (Grade F)

    Example:
        Simple function (complexity = 0):
            def add(a, b):
                return a + b

        Nested conditionals (complexity = 4):
            def process(x):  # +0 (function entry)
                if x > 0:     # +1 (if) +0 (nesting level 0)
                    if x > 10:  # +1 (if) +1 (nesting level 1) = +2
                        return x
                return 0      # Total: 4
    """

    def __init__(self) -> None:
        """Initialize cognitive complexity collector."""
        self._complexity = 0
        self._nesting_level = 0
        self._current_function_name: str | None = None

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "cognitive_complexity"
        """
        return "cognitive_complexity"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and update cognitive complexity.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language and scope info
            depth: Current depth in AST (unused, we track logical nesting)
        """
        language = context.language
        node_type = node.type

        # Control flow statements (+1 + nesting level)
        control_flow_categories = [
            "if_statement",
            "elif_clause",
            "else_clause",
            "for_loop",
            "while_loop",
            "except_clause",
            "ternary",
        ]

        for category in control_flow_categories:
            if node_type in get_node_types(language, category):
                # +1 for statement itself, +nesting_level for being nested
                self._complexity += 1 + self._nesting_level
                break

        # Match/switch statements (+1 + nesting level)
        if node_type in get_node_types(language, "match_statement"):
            self._complexity += 1 + self._nesting_level

        # Case clauses (+1, no nesting penalty)
        if node_type in get_node_types(language, "case_clause"):
            self._complexity += 1

        # Jump statements (+1)
        jump_categories = ["break", "continue"]
        for category in jump_categories:
            if node_type in get_node_types(language, category):
                self._complexity += 1
                break

        # Boolean operators (+1 per operator)
        if node_type in get_node_types(
            language, "boolean_and"
        ) or node_type in get_node_types(language, "boolean_or"):
            self._complexity += 1

        # Track nesting level changes
        nesting_categories = [
            "if_statement",
            "for_loop",
            "while_loop",
            "try_statement",
            "match_statement",
        ]

        for category in nesting_categories:
            if node_type in get_node_types(language, category):
                self._nesting_level += 1
                break

        # Track function name for recursion detection
        if node_type in get_node_types(language, "function_def"):
            # Extract function name if available
            if hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    self._current_function_name = name_node.text.decode("utf-8")

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return final cognitive complexity for completed function.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Dictionary with cognitive_complexity metric
        """
        # TODO: Detect recursion by analyzing function calls
        # This would require looking at all identifier nodes and checking
        # if any match self._current_function_name
        return {"cognitive_complexity": self._complexity}

    def reset(self) -> None:
        """Reset collector state for next function."""
        self._complexity = 0
        self._nesting_level = 0
        self._current_function_name = None


class CyclomaticComplexityCollector(MetricCollector):
    """Tracks cyclomatic complexity - number of independent execution paths.

    Cyclomatic complexity measures the number of linearly independent paths
    through code. Higher values indicate more test cases needed for coverage.

    Scoring Rules:
    - Start with complexity = 1 (single straight-through path)
    - +1 for each: if, elif, for, while, case/match, catch/except
    - +1 for each boolean operator: and, or, &&, ||
    - +1 for each ternary expression

    Interpretation:
    - 1-4: Simple, low risk
    - 5-7: Moderate complexity
    - 8-10: Complex, higher risk
    - 11+: Very complex, difficult to test

    Example:
        def check_value(x):
            # complexity = 1 (baseline)
            if x > 0:     # +1 = 2
                return "positive"
            elif x < 0:   # +1 = 3
                return "negative"
            else:
                return "zero"
            # Total: 3 (three independent paths)
    """

    def __init__(self) -> None:
        """Initialize cyclomatic complexity collector."""
        self._complexity = 1  # Start at 1 (baseline path)

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "cyclomatic_complexity"
        """
        return "cyclomatic_complexity"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and update cyclomatic complexity.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Decision points (+1 each)
        decision_categories = [
            "if_statement",
            "elif_clause",
            "for_loop",
            "while_loop",
            "except_clause",
            "case_clause",
            "ternary",
        ]

        for category in decision_categories:
            if node_type in get_node_types(language, category):
                self._complexity += 1
                break

        # Boolean operators (+1 each)
        if node_type in get_node_types(
            language, "boolean_and"
        ) or node_type in get_node_types(language, "boolean_or"):
            self._complexity += 1

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return final cyclomatic complexity for completed function.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Dictionary with cyclomatic_complexity metric
        """
        return {"cyclomatic_complexity": self._complexity}

    def reset(self) -> None:
        """Reset collector state for next function."""
        self._complexity = 1  # Reset to baseline


class NestingDepthCollector(MetricCollector):
    """Tracks maximum nesting depth of control structures.

    Nesting depth measures how deeply control structures are nested.
    Deep nesting (>3 levels) indicates code that is hard to read and maintain.

    Tracked Structures:
    - Functions/methods
    - If/elif/else blocks
    - For/while loops
    - Try/catch/except blocks
    - With/using statements
    - Match/switch statements

    Interpretation:
    - 0-1: Flat, easy to read
    - 2-3: Acceptable nesting
    - 4-5: High nesting, consider refactoring
    - 6+: Excessive nesting, refactor recommended

    Example:
        def process():         # depth 0 (not counted in nesting)
            if condition:      # depth 1
                for item in items:  # depth 2
                    while busy:     # depth 3
                        if ready:   # depth 4 (max_nesting = 4)
                            process()
    """

    def __init__(self) -> None:
        """Initialize nesting depth collector."""
        self._max_depth = 0
        self._current_depth = 0

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "nesting_depth"
        """
        return "nesting_depth"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and track nesting depth changes.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language info
            depth: Current depth in AST
        """
        # Use context nesting stack for accurate tracking
        # The traversal engine should manage this stack
        if context.nesting_stack:
            stack_depth = len(context.nesting_stack)
            self._max_depth = max(self._max_depth, stack_depth)

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return maximum nesting depth for completed function.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Dictionary with max_nesting_depth metric
        """
        return {"max_nesting_depth": self._max_depth}

    def reset(self) -> None:
        """Reset collector state for next function."""
        self._max_depth = 0
        self._current_depth = 0


class ParameterCountCollector(MetricCollector):
    """Counts function parameters.

    Parameter count indicates function complexity and potential coupling.
    Functions with many parameters are harder to understand and test.

    Interpretation:
    - 0-2: Ideal, easy to understand
    - 3-4: Acceptable
    - 5-6: Consider refactoring
    - 7+: Too many parameters, refactor recommended

    Recommendations for high parameter counts:
    - Introduce parameter objects
    - Use builder pattern
    - Split into smaller functions

    Example:
        def simple(x, y):  # parameter_count = 2
            return x + y

        def complex(a, b, c, d, e, f, g):  # parameter_count = 7 (too many!)
            pass
    """

    def __init__(self) -> None:
        """Initialize parameter count collector."""
        self._parameter_count = 0

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "parameter_count"
        """
        return "parameter_count"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and count parameters.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Look for parameter list nodes
        if node_type in get_node_types(language, "parameters"):
            # Count child nodes that are parameters
            # Different languages have different parameter node structures
            # Python: named nodes are parameters
            # JavaScript: formal_parameter nodes
            # Java: formal_parameter nodes
            self._parameter_count = self._count_parameters(node, language)

    def _count_parameters(self, params_node: Node, language: str) -> int:
        """Count parameters in a parameter list node.

        Args:
            params_node: Parameter list node
            language: Programming language

        Returns:
            Number of parameters
        """
        count = 0

        # Iterate through child nodes
        for child in params_node.children:
            # Skip punctuation and keywords (commas, parentheses, etc.)
            if child.type in (",", "(", ")", "self", "cls"):
                continue

            # Language-specific parameter node types
            param_types = {
                "python": ["identifier", "typed_parameter", "default_parameter"],
                "javascript": ["formal_parameter", "identifier"],
                "typescript": ["required_parameter", "optional_parameter"],
                "java": ["formal_parameter"],
                "rust": ["parameter"],
                "php": ["simple_parameter", "variadic_parameter"],
                "ruby": ["identifier", "optional_parameter"],
            }

            lang_params = param_types.get(language, ["identifier"])

            # Check if this child is a parameter node
            if child.type in lang_params or child.is_named:
                count += 1

        return count

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return parameter count for completed function.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Dictionary with parameter_count metric
        """
        return {"parameter_count": self._parameter_count}

    def reset(self) -> None:
        """Reset collector state for next function."""
        self._parameter_count = 0


class MethodCountCollector(MetricCollector):
    """Counts methods in classes.

    Method count indicates class complexity and potential violation of
    Single Responsibility Principle. Classes with many methods may need
    to be split into smaller, focused classes.

    Interpretation:
    - 0-5: Focused class, good
    - 6-10: Moderate complexity
    - 11-15: High complexity, consider refactoring
    - 16+: Too many responsibilities, split class

    Only counts methods inside classes, not top-level functions.

    Example:
        class Simple:  # method_count = 2
            def __init__(self): pass
            def process(self): pass

        class Complex:  # method_count = 12 (too many!)
            def method1(self): pass
            def method2(self): pass
            # ... 10 more methods
    """

    def __init__(self) -> None:
        """Initialize method count collector."""
        self._method_count = 0
        self._inside_class = False

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "method_count"
        """
        return "method_count"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and count methods in classes.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language and class info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Track when we enter/exit a class
        if node_type in get_node_types(language, "class_def"):
            self._inside_class = True

        # Count function definitions inside classes
        # Use context.current_class as the primary indicator
        if context.current_class and node_type in get_node_types(
            language, "function_def"
        ):
            self._method_count += 1
        elif self._inside_class and node_type in get_node_types(
            language, "function_def"
        ):
            # Fallback to _inside_class flag if context.current_class not set
            self._method_count += 1

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return method count for completed class.

        Note: This is called when exiting a function, but for classes
        we need to track this differently. For now, return the count
        accumulated so far.

        Args:
            node: Function/class definition node
            context: Shared context

        Returns:
            Dictionary with method_count metric (0 for functions, count for classes)
        """
        # Only return method count if we're in a class
        if context.current_class or self._inside_class:
            return {"method_count": self._method_count}

        # For regular functions, method_count is 0
        return {"method_count": 0}

    def reset(self) -> None:
        """Reset collector state for next class/function."""
        self._method_count = 0
        self._inside_class = False
