"""Halstead complexity metrics collector for structural code analysis.

This module implements Halstead complexity measures, which quantify code
complexity based on the number and frequency of operators and operands.

Halstead Metrics:
- n1: Number of distinct operators
- n2: Number of distinct operands
- N1: Total number of operators
- N2: Total number of operands
- Vocabulary (n): n1 + n2
- Length (N): N1 + N2
- Volume (V): N × log₂(n) - Information content in bits
- Difficulty (D): (n1/2) × (N2/n2) - How hard to understand
- Effort (E): D × V - Mental effort required
- Time (T): E / 18 - Estimated programming time in seconds
- Bugs (B): V / 3000 - Estimated number of bugs

References:
    Halstead, Maurice H. (1977). Elements of Software Science.
    https://en.wikipedia.org/wiki/Halstead_complexity_measures
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import CollectorContext, MetricCollector

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics for a code unit.

    Attributes:
        distinct_operators: Number of distinct operators (n1)
        distinct_operands: Number of distinct operands (n2)
        total_operators: Total number of operators (N1)
        total_operands: Total number of operands (N2)
        vocabulary: Program vocabulary (n = n1 + n2)
        length: Program length (N = N1 + N2)
        volume: Information content in bits (V = N × log₂(n))
        difficulty: How hard to understand (D = (n1/2) × (N2/n2))
        effort: Mental effort required (E = D × V)
        time_seconds: Estimated programming time (T = E / 18)
        estimated_bugs: Estimated number of bugs (B = V / 3000)

    Example:
        Simple function:
        >>> def add(a, b):
        ...     return a + b

        Operators: def, return, +, (, ), , ≈ 6 distinct
        Operands: add, a, b ≈ 3 distinct

        metrics = HalsteadMetrics.from_counts(
            n1=6, n2=3, N1=6, N2=6
        )
        # Volume ≈ 12 × log₂(9) ≈ 38 bits
    """

    # Raw counts
    distinct_operators: int  # n1
    distinct_operands: int  # n2
    total_operators: int  # N1
    total_operands: int  # N2

    # Derived metrics
    vocabulary: int  # n = n1 + n2
    length: int  # N = N1 + N2
    volume: float  # V = N × log₂(n)
    difficulty: float  # D = (n1/2) × (N2/n2)
    effort: float  # E = D × V
    time_seconds: float  # T = E / 18
    estimated_bugs: float  # B = V / 3000

    @classmethod
    def from_counts(
        cls,
        n1: int,
        n2: int,
        N1: int,  # noqa: N803 - Halstead notation uses uppercase N1, N2
        N2: int,  # noqa: N803 - Halstead notation uses uppercase N1, N2
    ) -> HalsteadMetrics:
        """Calculate all Halstead metrics from raw operator/operand counts.

        Args:
            n1: Number of distinct operators
            n2: Number of distinct operands
            N1: Total number of operators
            N2: Total number of operands

        Returns:
            HalsteadMetrics with all derived metrics calculated

        Example:
            >>> metrics = HalsteadMetrics.from_counts(6, 3, 6, 6)
            >>> metrics.vocabulary
            9
            >>> metrics.volume > 0
            True
        """
        vocabulary = n1 + n2
        length = N1 + N2

        # Handle edge cases to avoid division by zero or log(0)
        if vocabulary == 0 or length == 0:
            return cls(
                distinct_operators=n1,
                distinct_operands=n2,
                total_operators=N1,
                total_operands=N2,
                vocabulary=vocabulary,
                length=length,
                volume=0.0,
                difficulty=0.0,
                effort=0.0,
                time_seconds=0.0,
                estimated_bugs=0.0,
            )

        # Calculate derived metrics
        volume = length * math.log2(vocabulary)

        # Difficulty = (n1/2) × (N2/n2)
        # Avoid division by zero
        if n2 == 0:
            difficulty = 0.0
        else:
            difficulty = (n1 / 2) * (N2 / n2)

        effort = difficulty * volume
        time_seconds = effort / 18  # Stroud number (psychological moments per second)
        estimated_bugs = volume / 3000  # Empirical constant

        return cls(
            distinct_operators=n1,
            distinct_operands=n2,
            total_operators=N1,
            total_operands=N2,
            vocabulary=vocabulary,
            length=length,
            volume=volume,
            difficulty=difficulty,
            effort=effort,
            time_seconds=time_seconds,
            estimated_bugs=estimated_bugs,
        )


# Language-specific operator and operand definitions
# Maps programming language to sets of operators and node type categories

PYTHON_OPERATORS = {
    # Binary operators
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "@",
    # Comparison
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    # Logical
    "and",
    "or",
    "not",
    "is",
    "in",
    # Bitwise
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
    # Assignment
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "%=",
    "**=",
    "&=",
    "|=",
    "^=",
    ">>=",
    "<<=",
    "@=",
    # Control keywords
    "if",
    "else",
    "elif",
    "for",
    "while",
    "with",
    "try",
    "except",
    "finally",
    "raise",
    "def",
    "class",
    "lambda",
    "return",
    "yield",
    "yield from",
    "import",
    "from",
    "as",
    "assert",
    "pass",
    "break",
    "continue",
    "global",
    "nonlocal",
    "del",
    # Access/call operators
    ".",
    "[",
    "]",
    "(",
    ")",
    ",",
    ":",
    "->",
}

PYTHON_OPERATOR_NODE_TYPES = {
    "binary_operator",
    "unary_operator",
    "boolean_operator",
    "comparison_operator",
    "assignment",
    "augmented_assignment",
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "while_statement",
    "function_definition",
    "class_definition",
    "lambda",
    "return_statement",
    "yield",
    "import_statement",
    "import_from_statement",
    "try_statement",
    "except_clause",
    "finally_clause",
    "raise_statement",
    "with_statement",
    "assert_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "del_statement",
    "call",
    "subscript",
    "attribute",
    "global_statement",
    "nonlocal_statement",
}

PYTHON_OPERAND_NODE_TYPES = {
    "identifier",
    "integer",
    "float",
    "string",
    "true",
    "false",
    "none",
    "concatenated_string",
    "formatted_string",
}


class HalsteadCollector(MetricCollector):
    """Collects Halstead complexity metrics using tree-sitter AST traversal.

    This collector analyzes code to count operators and operands, then
    calculates Halstead's software complexity metrics. These metrics
    provide insights into code volume, difficulty, and estimated bugs.

    Operators are language constructs that perform operations:
    - Arithmetic: +, -, *, /, etc.
    - Comparison: ==, !=, <, >, etc.
    - Logical: and, or, not
    - Control flow: if, for, while, try, etc.
    - Definitions: def, class, lambda
    - Access: ., [], ()

    Operands are the data being operated on:
    - Variable names (identifiers)
    - Literals (numbers, strings, booleans)
    - Constants (None, True, False)

    Example:
        >>> code = '''
        ... def calculate(x, y):
        ...     if x > 0:
        ...         return x + y
        ...     return 0
        ... '''
        >>> # Operators: def, if, >, return (×2), +
        >>> # Operands: calculate, x (×3), y, 0 (×2)
        >>> # Result: High volume indicates complex logic

    Performance:
        Target: <2ms per file
        Scales linearly with AST node count
    """

    def __init__(self) -> None:
        """Initialize Halstead metrics collector."""
        self._operators: set[str] = set()
        self._operands: set[str] = set()
        self._total_operators = 0
        self._total_operands = 0

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "halstead"
        """
        return "halstead"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process AST node and count operators/operands.

        Classifies each node as an operator, operand, or neither, and
        updates the running counts accordingly.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language and source info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Get node text for operator/operand identification
        node_text = self._get_node_text(node, context.source_code)

        # Check if node is an operator
        if self._is_operator(node_type, node_text, language):
            self._operators.add(node_text)
            self._total_operators += 1
        # Check if node is an operand
        elif self._is_operand(node_type, node_text, language):
            self._operands.add(node_text)
            self._total_operands += 1

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return final Halstead metrics for completed function.

        Calculates all derived Halstead metrics from the accumulated
        operator and operand counts.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Dictionary with Halstead metrics:
            - halstead_volume: Information content in bits
            - halstead_difficulty: How hard to understand
            - halstead_effort: Mental effort required
            - halstead_bugs: Estimated number of bugs
            - halstead_n1: Distinct operators count
            - halstead_n2: Distinct operands count
            - halstead_N1: Total operators count
            - halstead_N2: Total operands count
        """
        metrics = HalsteadMetrics.from_counts(
            n1=len(self._operators),
            n2=len(self._operands),
            N1=self._total_operators,
            N2=self._total_operands,
        )

        return {
            "halstead_volume": metrics.volume,
            "halstead_difficulty": metrics.difficulty,
            "halstead_effort": metrics.effort,
            "halstead_bugs": metrics.estimated_bugs,
            "halstead_n1": metrics.distinct_operators,
            "halstead_n2": metrics.distinct_operands,
            "halstead_N1": metrics.total_operators,
            "halstead_N2": metrics.total_operands,
        }

    def reset(self) -> None:
        """Reset collector state for next function."""
        self._operators.clear()
        self._operands.clear()
        self._total_operators = 0
        self._total_operands = 0

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """Extract text content from a tree-sitter node.

        Args:
            node: Tree-sitter AST node
            source: Raw source code as bytes

        Returns:
            Decoded text content of the node
        """
        return node.text.decode("utf-8") if node.text else ""

    def _is_operator(self, node_type: str, node_text: str, language: str) -> bool:
        """Check if node represents an operator.

        An operator is a language construct that performs an operation.
        This includes arithmetic operators, control flow keywords, etc.

        Args:
            node_type: Tree-sitter node type
            node_text: Text content of the node
            language: Programming language identifier

        Returns:
            True if node is an operator, False otherwise
        """
        # Language-specific operator detection
        if language == "python":
            # Check if node type is an operator type
            if node_type in PYTHON_OPERATOR_NODE_TYPES:
                return True

            # Check if node text matches known operators
            if node_text in PYTHON_OPERATORS:
                return True

        # Default: not an operator
        return False

    def _is_operand(self, node_type: str, node_text: str, language: str) -> bool:
        """Check if node represents an operand.

        An operand is data being operated on: variables, literals, constants.

        Args:
            node_type: Tree-sitter node type
            node_text: Text content of the node
            language: Programming language identifier

        Returns:
            True if node is an operand, False otherwise
        """
        # Language-specific operand detection
        if language == "python":
            # Exclude Python keywords that aren't literals
            python_keywords = {
                "if",
                "else",
                "elif",
                "for",
                "while",
                "with",
                "try",
                "except",
                "finally",
                "raise",
                "def",
                "class",
                "lambda",
                "return",
                "yield",
                "import",
                "from",
                "as",
                "assert",
                "pass",
                "break",
                "continue",
                "global",
                "nonlocal",
                "del",
                "and",
                "or",
                "not",
                "is",
                "in",
            }

            # Identifiers are operands if not keywords
            if node_type == "identifier" and node_text in python_keywords:
                return False

            # Check if node type is an operand type
            if node_type in PYTHON_OPERAND_NODE_TYPES:
                return True

        # Default: not an operand
        return False
