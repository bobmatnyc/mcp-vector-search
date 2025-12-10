"""Base collector interface for metric collection during AST traversal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class CollectorContext:
    """Shared context passed to all collectors during AST traversal.

    Provides state management for tracking the current position and scope
    during tree-sitter AST traversal. All collectors receive the same
    context instance and can read/modify it during traversal.

    Attributes:
        file_path: Path to the file being analyzed
        source_code: Raw source code as bytes (required by tree-sitter)
        language: Programming language identifier (e.g., "python", "javascript")
        current_function: Name of the function currently being analyzed
        current_class: Name of the class currently being analyzed
        nesting_stack: Stack tracking nested scopes (for depth calculation)

    Example:
        context = CollectorContext(
            file_path="/path/to/file.py",
            source_code=b"def foo(): pass",
            language="python"
        )

        # During traversal
        context.current_function = "foo"
        context.nesting_stack.append("function")
        depth = len(context.nesting_stack)
    """

    file_path: str
    source_code: bytes
    language: str

    # Accumulator for current function being analyzed
    current_function: str | None = None
    current_class: str | None = None

    # Stack for tracking nesting
    nesting_stack: list[str] = field(default_factory=list)


class MetricCollector(ABC):
    """Abstract base class for metric collectors.

    Collectors implement the visitor pattern for AST traversal. Each collector
    is responsible for tracking specific metrics (e.g., complexity, nesting)
    during tree-sitter node traversal.

    Lifecycle:
    1. collect_node() called for each AST node during traversal
    2. Collector accumulates state during traversal
    3. finalize_function() called when exiting a function/method
    4. reset() called to prepare for next function

    Subclasses must implement:
    - name: Unique identifier for the collector
    - collect_node(): Process individual AST nodes
    - finalize_function(): Return final metrics for completed function

    Example Implementation:
        class ComplexityCollector(MetricCollector):
            def __init__(self):
                self._complexity = 0

            @property
            def name(self) -> str:
                return "complexity"

            def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
                if node.type in ("if_statement", "while_statement"):
                    self._complexity += 1

            def finalize_function(self, node: Node, context: CollectorContext) -> dict[str, Any]:
                return {"cognitive_complexity": self._complexity}

            def reset(self) -> None:
                self._complexity = 0
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this collector.

        Returns:
            Collector name (e.g., "complexity", "nesting", "parameters")
        """
        pass

    @abstractmethod
    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Called for each AST node during traversal.

        Collectors accumulate state here by examining node properties
        and updating internal counters/state.

        Args:
            node: Current tree-sitter AST node being visited
            context: Shared context with file info and current scope
            depth: Current nesting depth in the AST

        Example:
            def collect_node(self, node, context, depth):
                if node.type == "if_statement":
                    self._if_count += 1
                elif node.type == "for_statement":
                    self._loop_count += 1
        """
        pass

    @abstractmethod
    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Called when exiting a function/method.

        Returns final metrics for this function. The returned dictionary
        should contain metric names as keys and their values.

        Args:
            node: The function/method definition node
            context: Shared context with file info and scope

        Returns:
            Dictionary of metric names to values
            Example: {"cognitive_complexity": 5, "max_nesting": 3}

        Example:
            def finalize_function(self, node, context):
                return {
                    "cognitive_complexity": self._complexity,
                    "nesting_depth": self._max_depth
                }
        """
        pass

    def reset(self) -> None:
        """Reset collector state for next function.

        Called after finalize_function() to prepare the collector
        for analyzing the next function/method.

        Default implementation does nothing. Override if collector
        maintains state that needs clearing.

        Example:
            def reset(self):
                self._complexity = 0
                self._nesting_stack.clear()
                self._max_depth = 0
        """
        return  # Default no-op implementation - subclasses override if needed
