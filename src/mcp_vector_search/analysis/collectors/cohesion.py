"""LCOM4 cohesion metric collector.

LCOM4 (Lack of Cohesion of Methods version 4) measures class cohesion
by counting connected components in the method-attribute graph.

A class is cohesive when its methods work together using shared attributes.
LCOM4 counts how many disconnected groups of methods exist:
- LCOM4 = 1: Perfect cohesion (all methods connected)
- LCOM4 > 1: Poor cohesion (class should potentially be split)

Example:
    # Cohesive class (LCOM4 = 1)
    class GoodClass:
        def method_a(self):
            return self.x + self.y

        def method_b(self):
            return self.x * self.y  # Shares x, y with method_a

    # Incohesive class (LCOM4 = 2)
    class BadClass:
        def method_a(self):
            return self.x + self.y  # Group 1

        def method_c(self):
            return self.z + self.w  # Group 2 (no shared attributes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass
class MethodAttributeAccess:
    """Tracks which attributes a method accesses.

    Attributes:
        method_name: Name of the method
        attributes: Set of instance attributes accessed (e.g., {"x", "y"})
    """

    method_name: str
    attributes: set[str] = field(default_factory=set)


@dataclass
class ClassCohesion:
    """LCOM4 result for a single class.

    Attributes:
        class_name: Name of the class
        lcom4: Number of connected components (1=cohesive, >1=incohesive)
        method_count: Total number of methods in class
        attribute_count: Total number of instance attributes accessed
        method_attributes: Detailed mapping of method names to their attributes
    """

    class_name: str
    lcom4: int
    method_count: int
    attribute_count: int
    method_attributes: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class FileCohesion:
    """Cohesion metrics for all classes in a file.

    Attributes:
        file_path: Path to the analyzed file
        classes: List of per-class cohesion results
        avg_lcom4: Average LCOM4 across all classes
        max_lcom4: Maximum LCOM4 value (worst cohesion)
    """

    file_path: Path
    classes: list[ClassCohesion] = field(default_factory=list)
    avg_lcom4: float = 0.0
    max_lcom4: int = 0


class UnionFind:
    """Union-Find data structure for connected components.

    Efficiently tracks and merges disjoint sets to count connected
    components in the method-attribute graph.

    Example:
        uf = UnionFind(["method_a", "method_b", "method_c"])
        uf.union("method_a", "method_b")  # Connect a and b
        uf.count_components()  # Returns 2 (groups: {a,b}, {c})
    """

    def __init__(self, items: list[str]) -> None:
        """Initialize union-find with independent items.

        Args:
            items: List of method names to track
        """
        self.parent = {item: item for item in items}
        self.rank = dict.fromkeys(items, 0)

    def find(self, item: str) -> str:
        """Find root of item's set with path compression.

        Args:
            item: Method name to find root for

        Returns:
            Root of the set containing item
        """
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        """Merge sets containing item1 and item2.

        Uses union by rank for efficiency.

        Args:
            item1: First method name
            item2: Second method name
        """
        root1, root2 = self.find(item1), self.find(item2)
        if root1 != root2:
            # Union by rank: attach smaller tree under larger
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def count_components(self) -> int:
        """Count number of connected components.

        Returns:
            Number of disjoint sets (LCOM4 value)
        """
        return len({self.find(item) for item in self.parent})


class LCOM4Calculator:
    """Calculate LCOM4 cohesion metric for Python classes.

    Algorithm:
    1. For each class, extract methods and their attribute accesses
    2. Build undirected graph: nodes=methods, edges=shared attributes
    3. Count connected components using Union-Find
    4. LCOM4 = number of components (1=cohesive, >1=potentially split)

    Example:
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(
            Path("my_file.py"),
            "class MyClass:\\n    def foo(self): return self.x\\n"
        )
        print(f"LCOM4: {result.classes[0].lcom4}")
    """

    def __init__(self) -> None:
        """Initialize LCOM4 calculator with tree-sitter parser."""
        self._parser = None
        self._language = None
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Python."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("python")
            self._parser = get_parser("python")
            logger.debug("Python Tree-sitter parser initialized for LCOM4")
        except Exception as e:
            logger.warning(f"Tree-sitter initialization failed: {e}")
            self._parser = None
            self._language = None

    def calculate_file_cohesion(self, file_path: Path, content: str) -> FileCohesion:
        """Calculate LCOM4 for all classes in a file.

        Args:
            file_path: Path to the file (for reporting)
            content: Source code content as string

        Returns:
            FileCohesion with per-class LCOM4 results
        """
        if not self._parser:
            logger.warning("Tree-sitter parser not available, returning empty result")
            return FileCohesion(file_path=file_path)

        tree = self._parser.parse(bytes(content, "utf8"))
        classes = self._find_classes(tree.root_node)

        class_cohesions = []
        for class_node in classes:
            cohesion = self._calculate_class_cohesion(class_node, content)
            if cohesion:
                class_cohesions.append(cohesion)

        # Calculate aggregate metrics
        if class_cohesions:
            avg_lcom4 = sum(c.lcom4 for c in class_cohesions) / len(class_cohesions)
            max_lcom4 = max(c.lcom4 for c in class_cohesions)
        else:
            avg_lcom4 = 0.0
            max_lcom4 = 0

        return FileCohesion(
            file_path=file_path,
            classes=class_cohesions,
            avg_lcom4=avg_lcom4,
            max_lcom4=max_lcom4,
        )

    def _find_classes(self, root: Node) -> list[Node]:
        """Find all class definitions in the AST.

        Args:
            root: Root AST node

        Returns:
            List of class_definition nodes
        """
        classes = []

        def visit(node: Node) -> None:
            if node.type == "class_definition":
                classes.append(node)
            for child in node.children:
                visit(child)

        visit(root)
        return classes

    def _calculate_class_cohesion(
        self, class_node: Node, content: str
    ) -> ClassCohesion | None:
        """Calculate LCOM4 for a single class.

        Args:
            class_node: AST node for class definition
            content: Source code (for extracting text)

        Returns:
            ClassCohesion result, or None if class has no methods
        """
        class_name = self._get_class_name(class_node, content)
        methods = self._extract_methods(class_node)

        if not methods:
            logger.debug(f"Class {class_name} has no methods, skipping LCOM4")
            return None

        # Extract attribute accesses for each method
        method_attributes: dict[str, set[str]] = {}
        for method_node in methods:
            method_name = self._get_method_name(method_node, content)
            # Skip special methods that don't access self
            if self._is_static_or_class_method(method_node):
                continue

            attributes = self._find_attribute_accesses(method_node, content)
            if method_name and attributes:
                method_attributes[method_name] = attributes

        # Handle edge cases
        if not method_attributes:
            # No methods with attribute accesses
            lcom4 = len(methods) if methods else 0
            return ClassCohesion(
                class_name=class_name,
                lcom4=lcom4,
                method_count=len(methods),
                attribute_count=0,
                method_attributes={},
            )

        # Calculate LCOM4 using connected components
        lcom4 = self._calculate_lcom4(method_attributes)

        # Count unique attributes
        all_attributes = set()
        for attrs in method_attributes.values():
            all_attributes.update(attrs)

        return ClassCohesion(
            class_name=class_name,
            lcom4=lcom4,
            method_count=len(methods),
            attribute_count=len(all_attributes),
            method_attributes=method_attributes,
        )

    def _get_class_name(self, class_node: Node, content: str) -> str:
        """Extract class name from class definition node.

        Args:
            class_node: Class definition AST node
            content: Source code

        Returns:
            Class name or "UnknownClass"
        """
        name_node = class_node.child_by_field_name("name")
        if name_node:
            return content[name_node.start_byte : name_node.end_byte]
        return "UnknownClass"

    def _extract_methods(self, class_node: Node) -> list[Node]:
        """Extract method nodes from a class.

        Args:
            class_node: Class definition AST node

        Returns:
            List of function_definition nodes that are methods
        """
        methods = []
        body = class_node.child_by_field_name("body")
        if not body:
            return methods

        for child in body.children:
            if child.type == "function_definition":
                methods.append(child)
            elif child.type == "decorated_definition":
                # Decorated methods: @decorator\ndef method(...)
                # Find the function_definition inside
                for subchild in child.children:
                    if subchild.type == "function_definition":
                        methods.append(subchild)
                        break

        return methods

    def _get_method_name(self, method_node: Node, content: str) -> str | None:
        """Extract method name from function definition.

        Args:
            method_node: Function definition AST node
            content: Source code

        Returns:
            Method name or None
        """
        name_node = method_node.child_by_field_name("name")
        if name_node:
            return content[name_node.start_byte : name_node.end_byte]
        return None

    def _is_static_or_class_method(self, method_node: Node) -> bool:
        """Check if method is @staticmethod or @classmethod.

        Args:
            method_node: Function definition AST node

        Returns:
            True if method is static or class method
        """
        # Check if parent is decorated_definition (for decorated methods)
        parent = method_node.parent
        if parent and parent.type == "decorated_definition":
            # Look for decorators in parent's children
            for child in parent.children:
                if child.type == "decorator":
                    decorator_text = child.text.decode("utf-8")
                    if (
                        "@staticmethod" in decorator_text
                        or "@classmethod" in decorator_text
                    ):
                        return True

        # Also check direct children (in case structure is different)
        for child in method_node.children:
            if child.type == "decorator":
                decorator_text = child.text.decode("utf-8")
                if (
                    "@staticmethod" in decorator_text
                    or "@classmethod" in decorator_text
                ):
                    return True

        return False

    def _find_attribute_accesses(self, method_node: Node, content: str) -> set[str]:
        """Find all self.attribute accesses in a method.

        Args:
            method_node: Function definition AST node
            content: Source code

        Returns:
            Set of attribute names accessed via self
        """
        attributes = set()

        def visit(node: Node) -> None:
            # Look for attribute access: self.attribute
            if node.type == "attribute":
                # Check if object is 'self'
                obj_node = node.child_by_field_name("object")
                if obj_node and obj_node.type == "identifier":
                    obj_name = content[obj_node.start_byte : obj_node.end_byte]
                    if obj_name == "self":
                        # Extract attribute name
                        attr_node = node.child_by_field_name("attribute")
                        if attr_node:
                            attr_name = content[
                                attr_node.start_byte : attr_node.end_byte
                            ]
                            attributes.add(attr_name)

            for child in node.children:
                visit(child)

        visit(method_node)
        return attributes

    def _calculate_lcom4(self, method_attributes: dict[str, set[str]]) -> int:
        """Calculate LCOM4 using connected components.

        Uses Union-Find to efficiently count connected components
        in the method-attribute graph.

        Args:
            method_attributes: Mapping of method names to their attributes

        Returns:
            LCOM4 value (number of connected components)
        """
        if not method_attributes:
            return 0

        methods = list(method_attributes.keys())

        # Edge case: single method
        if len(methods) == 1:
            return 1

        # Initialize union-find
        uf = UnionFind(methods)

        # Connect methods that share attributes
        methods_list = list(methods)
        for i, method1 in enumerate(methods_list):
            for method2 in methods_list[i + 1 :]:
                # Check if methods share any attributes
                shared = method_attributes[method1] & method_attributes[method2]
                if shared:
                    uf.union(method1, method2)

        return uf.count_components()
