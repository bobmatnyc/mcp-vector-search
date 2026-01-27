"""Entry point detection for dead code analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class EntryPointType(str, Enum):
    """Types of entry points in Python codebases.

    Attributes:
        MAIN: if __name__ == "__main__" blocks
        CLI: CLI decorators (@click.command, @app.command)
        ROUTE: HTTP route decorators (@app.get, @router.post)
        TEST: pytest test functions and fixtures
        EXPORT: __all__ exports in __init__.py files
        PUBLIC: Public module-level functions (optional)
        CUSTOM: User-specified entry points
    """

    MAIN = "MAIN"
    CLI = "CLI"
    ROUTE = "ROUTE"
    TEST = "TEST"
    EXPORT = "EXPORT"
    PUBLIC = "PUBLIC"
    CUSTOM = "CUSTOM"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


@dataclass
class EntryPoint:
    """Represents a detected entry point in the codebase.

    Attributes:
        name: Function or symbol name
        file_path: Path to the file containing the entry point
        line_number: Line number where entry point is defined
        type: Type of entry point (MAIN, CLI, ROUTE, etc.)
        confidence: Confidence level (0.0-1.0) that this is a true entry point
    """

    name: str
    file_path: str
    line_number: int
    type: EntryPointType
    confidence: float = 1.0

    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.type.value}] {self.name} at {self.file_path}:{self.line_number} (confidence: {self.confidence:.2f})"


class EntryPointDetector:
    """Detects entry points in Python code using AST analysis.

    Entry points are functions/methods that serve as starting points for code execution:
    - Main blocks (if __name__ == "__main__")
    - CLI commands (@click.command, @typer.command)
    - HTTP routes (@app.get, @router.post)
    - Test functions (test_*, @pytest.fixture)
    - Module exports (__all__ lists)
    - Optionally, public module-level functions

    Example:
        detector = EntryPointDetector(include_public=True)
        entry_points = detector.detect_from_directory(Path("src/"))
        for ep in entry_points:
            print(f"{ep.type}: {ep.name} at {ep.file_path}:{ep.line_number}")
    """

    def __init__(self, include_public: bool = False) -> None:
        """Initialize entry point detector.

        Args:
            include_public: If True, include public module-level functions as entry points
        """
        self.include_public = include_public

        # CLI decorator patterns (click, typer, etc.)
        self._cli_decorators = {
            "command",
            "group",
            "click.command",
            "click.group",
            "app.command",
            "typer.command",
        }

        # HTTP route decorator patterns (FastAPI, Flask, etc.)
        self._route_decorators = {
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "options",
            "head",
            "route",
            "websocket",
            "api_route",
            # Flask patterns
            "app.route",
            "app.get",
            "app.post",
        }

        # Test patterns
        self._test_patterns = {"test_", "pytest.fixture", "fixture"}

    def detect_from_file(self, file_path: Path, code: str) -> list[EntryPoint]:
        """Detect entry points in a single file.

        Args:
            file_path: Path to the Python file
            code: Source code content

        Returns:
            List of detected entry points
        """
        entry_points: list[EntryPoint] = []

        try:
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing {file_path}: {e}")
            return entry_points

        # Convert file path to string for consistent storage
        file_path_str = str(file_path)

        # Detect different entry point types
        entry_points.extend(self._detect_main_block(tree, file_path_str))
        entry_points.extend(self._detect_cli_commands(tree, file_path_str))
        entry_points.extend(self._detect_routes(tree, file_path_str))
        entry_points.extend(self._detect_tests(tree, file_path_str))
        entry_points.extend(self._detect_exports(tree, file_path_str))

        if self.include_public:
            entry_points.extend(self._detect_public_functions(tree, file_path_str))

        return entry_points

    def detect_from_directory(self, root_path: Path) -> list[EntryPoint]:
        """Detect entry points in all Python files under a directory.

        Args:
            root_path: Root directory to search

        Returns:
            List of all detected entry points
        """
        all_entry_points: list[EntryPoint] = []

        # Find all Python files
        python_files = list(root_path.rglob("*.py"))

        for file_path in python_files:
            try:
                code = file_path.read_text(encoding="utf-8")
                entry_points = self.detect_from_file(file_path, code)
                all_entry_points.extend(entry_points)
            except Exception as e:
                logger.debug(f"Failed to process {file_path}: {e}")
                continue

        logger.info(
            f"Detected {len(all_entry_points)} entry points in {len(python_files)} files"
        )
        return all_entry_points

    def _detect_main_block(self, tree: ast.AST, file_path: str) -> list[EntryPoint]:
        """Detect if __name__ == '__main__' blocks.

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of entry points found in main blocks
        """
        entry_points: list[EntryPoint] = []

        for node in ast.walk(tree):
            # Look for if statements with __name__ == "__main__" condition
            if isinstance(node, ast.If):
                # Check if condition is a comparison
                if isinstance(node.test, ast.Compare):
                    # Check for __name__ on left side
                    is_name_check = (
                        isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                    )

                    # Check for "__main__" on right side
                    is_main_check = any(
                        isinstance(comp, ast.Constant) and comp.value == "__main__"
                        for comp in node.test.comparators
                    )

                    if is_name_check and is_main_check:
                        # Found a main block - extract function calls within it
                        for stmt in node.body:
                            if isinstance(stmt, ast.Expr) and isinstance(
                                stmt.value, ast.Call
                            ):
                                # Extract function name from call
                                func_name = self._extract_function_name(stmt.value.func)
                                if func_name:
                                    entry_points.append(
                                        EntryPoint(
                                            name=func_name,
                                            file_path=file_path,
                                            line_number=getattr(stmt, "lineno", 0),
                                            type=EntryPointType.MAIN,
                                            confidence=1.0,
                                        )
                                    )

        return entry_points

    def _detect_cli_commands(self, tree: ast.AST, file_path: str) -> list[EntryPoint]:
        """Detect CLI decorator patterns (click, typer).

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of CLI entry points
        """
        entry_points: list[EntryPoint] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check decorators
                for decorator in node.decorator_list:
                    decorator_name = self._extract_decorator_name(decorator)
                    if decorator_name in self._cli_decorators:
                        entry_points.append(
                            EntryPoint(
                                name=node.name,
                                file_path=file_path,
                                line_number=node.lineno,
                                type=EntryPointType.CLI,
                                confidence=1.0,
                            )
                        )
                        break

        return entry_points

    def _detect_routes(self, tree: ast.AST, file_path: str) -> list[EntryPoint]:
        """Detect FastAPI/Flask route decorators.

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of route entry points
        """
        entry_points: list[EntryPoint] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check decorators
                for decorator in node.decorator_list:
                    decorator_name = self._extract_decorator_name(decorator)
                    if decorator_name in self._route_decorators:
                        entry_points.append(
                            EntryPoint(
                                name=node.name,
                                file_path=file_path,
                                line_number=node.lineno,
                                type=EntryPointType.ROUTE,
                                confidence=1.0,
                            )
                        )
                        break

        return entry_points

    def _detect_tests(self, tree: ast.AST, file_path: str) -> list[EntryPoint]:
        """Detect pytest test functions and fixtures.

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of test entry points
        """
        entry_points: list[EntryPoint] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function name starts with test_
                if node.name.startswith("test_"):
                    entry_points.append(
                        EntryPoint(
                            name=node.name,
                            file_path=file_path,
                            line_number=node.lineno,
                            type=EntryPointType.TEST,
                            confidence=1.0,
                        )
                    )
                    continue

                # Check for pytest.fixture decorator
                for decorator in node.decorator_list:
                    decorator_name = self._extract_decorator_name(decorator)
                    if "fixture" in decorator_name:
                        entry_points.append(
                            EntryPoint(
                                name=node.name,
                                file_path=file_path,
                                line_number=node.lineno,
                                type=EntryPointType.TEST,
                                confidence=1.0,
                            )
                        )
                        break

        return entry_points

    def _detect_exports(self, tree: ast.AST, file_path: str) -> list[EntryPoint]:
        """Detect __all__ exports in __init__.py files.

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of export entry points
        """
        entry_points: list[EntryPoint] = []

        # Only check __init__.py files
        if not file_path.endswith("__init__.py"):
            return entry_points

        for node in ast.walk(tree):
            # Look for __all__ = [...] assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        # Extract list of exported names
                        if isinstance(node.value, ast.List | ast.Tuple):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    entry_points.append(
                                        EntryPoint(
                                            name=elt.value,
                                            file_path=file_path,
                                            line_number=getattr(node, "lineno", 0),
                                            type=EntryPointType.EXPORT,
                                            confidence=1.0,
                                        )
                                    )

        return entry_points

    def _detect_public_functions(
        self, tree: ast.AST, file_path: str
    ) -> list[EntryPoint]:
        """Detect public module-level functions (optional).

        Public functions are module-level functions that don't start with underscore.

        Args:
            tree: AST tree
            file_path: Path to file

        Returns:
            List of public function entry points
        """
        entry_points: list[EntryPoint] = []

        # Only look at module-level function definitions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Public if doesn't start with _
                if not node.name.startswith("_"):
                    entry_points.append(
                        EntryPoint(
                            name=node.name,
                            file_path=file_path,
                            line_number=node.lineno,
                            type=EntryPointType.PUBLIC,
                            confidence=0.7,  # Lower confidence - might not be called externally
                        )
                    )

        return entry_points

    def _extract_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name from AST node.

        Handles:
        - Simple names: @command
        - Attributes: @click.command
        - Calls: @app.command()

        Args:
            decorator: AST decorator node

        Returns:
            Decorator name as string
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            # Handle obj.attr patterns
            if isinstance(decorator.value, ast.Name):
                return f"{decorator.value.id}.{decorator.attr}"
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            # Handle decorator calls like @app.command()
            return self._extract_decorator_name(decorator.func)
        return ""

    def _extract_function_name(self, node: ast.expr) -> str:
        """Extract function name from call node.

        Args:
            node: AST expression node

        Returns:
            Function name or empty string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""
