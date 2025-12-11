"""Coupling metric collectors for dependency analysis.

This module provides collectors for efferent coupling (Ce) - the count of
external modules/files a file depends on (outgoing dependencies).

Higher efferent coupling indicates fragility - changes to dependencies can
break this file.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from .base import CollectorContext, MetricCollector

if TYPE_CHECKING:
    from tree_sitter import Node


# Multi-language import node type mappings
IMPORT_NODE_TYPES = {
    "python": {
        "import_statement": ["import_statement"],
        "import_from": ["import_from_statement"],
    },
    "javascript": {
        "import_statement": ["import_statement"],
        "require_call": ["call_expression"],  # require('module')
    },
    "typescript": {
        "import_statement": ["import_statement"],
        "import_type": ["import_statement"],  # import type { T } from 'mod'
        "require_call": ["call_expression"],
    },
}


def get_import_node_types(language: str, category: str) -> list[str]:
    """Get tree-sitter node types for import statements.

    Args:
        language: Programming language (e.g., "python", "javascript")
        category: Import category (e.g., "import_statement", "import_from")

    Returns:
        List of node type names for this language/category combination.
        Returns empty list if language/category not found.

    Examples:
        >>> get_import_node_types("python", "import_statement")
        ["import_statement"]

        >>> get_import_node_types("javascript", "import_statement")
        ["import_statement"]
    """
    lang_mapping = IMPORT_NODE_TYPES.get(language, IMPORT_NODE_TYPES["python"])
    return lang_mapping.get(category, [])


def is_stdlib_module(module_name: str, language: str) -> bool:
    """Check if a module is from the standard library.

    Args:
        module_name: Module name (e.g., "os", "sys", "fs")
        language: Programming language

    Returns:
        True if module is standard library, False otherwise

    Examples:
        >>> is_stdlib_module("os", "python")
        True

        >>> is_stdlib_module("requests", "python")
        False

        >>> is_stdlib_module("fs", "javascript")
        True
    """
    if language == "python":
        # Python standard library check
        # Use sys.stdlib_module_names (Python 3.10+) or hardcoded list
        if hasattr(sys, "stdlib_module_names"):
            return module_name.split(".")[0] in sys.stdlib_module_names
        else:
            # Fallback: common stdlib modules
            common_stdlib = {
                "os",
                "sys",
                "re",
                "json",
                "math",
                "time",
                "datetime",
                "collections",
                "itertools",
                "functools",
                "pathlib",
                "typing",
                "dataclasses",
                "asyncio",
                "contextlib",
                "abc",
                "io",
                "logging",
                "unittest",
                "pytest",
            }
            return module_name.split(".")[0] in common_stdlib

    elif language in ("javascript", "typescript"):
        # Node.js built-in modules
        nodejs_builtins = {
            "fs",
            "path",
            "http",
            "https",
            "url",
            "os",
            "util",
            "events",
            "stream",
            "buffer",
            "crypto",
            "child_process",
            "cluster",
            "dns",
            "net",
            "tls",
            "dgram",
            "readline",
            "zlib",
            "process",
            "console",
            "assert",
            "timers",
        }
        return module_name.split("/")[0] in nodejs_builtins

    return False


def is_relative_import(module_name: str, language: str) -> bool:
    """Check if import is relative to current file.

    Args:
        module_name: Module path
        language: Programming language

    Returns:
        True if import is relative, False otherwise

    Examples:
        >>> is_relative_import("./utils", "javascript")
        True

        >>> is_relative_import("lodash", "javascript")
        False

        >>> is_relative_import(".utils", "python")
        True
    """
    if language == "python":
        # Python relative imports start with "."
        return module_name.startswith(".")
    elif language in ("javascript", "typescript"):
        # JS/TS relative imports start with "./" or "../"
        return module_name.startswith("./") or module_name.startswith("../")
    return False


class EfferentCouplingCollector(MetricCollector):
    """Collects efferent coupling metrics (outgoing dependencies).

    Efferent coupling (Ce) measures how many external modules/files a file
    depends on. Higher Ce indicates fragility - changes to dependencies can
    break this file.

    Tracks:
    - Total unique dependencies (efferent_coupling score)
    - All imported modules
    - Internal vs. external imports
    - Standard library vs. third-party imports

    Example:
        # Python file with Ce = 3
        import os           # stdlib
        from typing import List  # stdlib (not counted, same base module)
        import requests     # external
        from .utils import helper  # internal

        # Ce = 3 (os, requests, .utils)
    """

    def __init__(self) -> None:
        """Initialize efferent coupling collector."""
        self._imports: set[str] = set()  # All unique imports
        self._internal_imports: set[str] = set()
        self._external_imports: set[str] = set()

    @property
    def name(self) -> str:
        """Return collector identifier.

        Returns:
            Collector name "efferent_coupling"
        """
        return "efferent_coupling"

    def collect_node(self, node: Node, context: CollectorContext, depth: int) -> None:
        """Process node and extract import statements.

        Args:
            node: Current tree-sitter AST node
            context: Shared context with language and file info
            depth: Current depth in AST (unused)
        """
        language = context.language
        node_type = node.type

        # Check if this is an import statement
        if node_type in get_import_node_types(language, "import_statement"):
            self._extract_import(node, context)
        elif node_type in get_import_node_types(language, "import_from"):
            self._extract_import_from(node, context)
        elif language in ("javascript", "typescript"):
            # Handle require() calls in JS/TS
            self._extract_require_call(node, context)

    def _extract_import(self, node: Node, context: CollectorContext) -> None:
        """Extract module name from import statement.

        Handles:
        - Python: import module, import module as alias
        - JavaScript/TypeScript: import ... from 'module'

        Args:
            node: Import statement node
            context: Collector context
        """
        language = context.language

        if language == "python":
            # Python: import os, import os.path, import os as operating_system
            # Look for dotted_name child
            for child in node.children:
                if child.type == "dotted_name":
                    module_name = child.text.decode("utf-8")
                    self._add_import(module_name, context)
                elif child.type == "aliased_import":
                    # import os as operating_system
                    # Get the name field (first child)
                    for subchild in child.children:
                        if subchild.type == "dotted_name":
                            module_name = subchild.text.decode("utf-8")
                            self._add_import(module_name, context)
                            break

        elif language in ("javascript", "typescript"):
            # JavaScript/TypeScript: import ... from 'module'
            # Find the string literal containing module path
            for child in node.children:
                if child.type == "string":
                    # Extract module name from string (remove quotes)
                    module_str = child.text.decode("utf-8")
                    module_name = module_str.strip("\"'")
                    self._add_import(module_name, context)

    def _extract_import_from(self, node: Node, context: CollectorContext) -> None:
        """Extract module name from 'from X import Y' statement.

        Handles:
        - Python: from module import X, from .relative import Y

        Args:
            node: Import from statement node
            context: Collector context
        """
        language = context.language

        if language == "python":
            # Python: from module import X
            # Look for module_name field
            module_node = node.child_by_field_name("module_name")
            if module_node:
                module_name = module_node.text.decode("utf-8")
                self._add_import(module_name, context)
            else:
                # Check for relative imports (from . import X)
                for child in node.children:
                    if child.type == "relative_import":
                        # Relative import detected
                        dots = child.text.decode("utf-8")
                        self._add_import(dots, context)
                        break
                    elif child.type == "dotted_name":
                        # Absolute import
                        module_name = child.text.decode("utf-8")
                        self._add_import(module_name, context)
                        break

    def _extract_require_call(self, node: Node, context: CollectorContext) -> None:
        """Extract module name from require('module') call.

        Handles:
        - JavaScript/TypeScript: const x = require('module')

        Args:
            node: Call expression node
            context: Collector context
        """
        language = context.language

        if language not in ("javascript", "typescript"):
            return

        # Check if this is a require() call
        # Structure: call_expression with function=identifier("require")
        function_node = node.child_by_field_name("function")
        if function_node and function_node.type == "identifier":
            function_name = function_node.text.decode("utf-8")
            if function_name == "require":
                # Get arguments
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        if child.type == "string":
                            module_str = child.text.decode("utf-8")
                            module_name = module_str.strip("\"'")
                            self._add_import(module_name, context)

    def _add_import(self, module_name: str, context: CollectorContext) -> None:
        """Add import to tracking sets and classify as internal/external.

        Args:
            module_name: Imported module name
            context: Collector context with language info
        """
        language = context.language

        # Add to all imports
        self._imports.add(module_name)

        # Classify import
        if is_relative_import(module_name, language):
            # Relative import = internal
            self._internal_imports.add(module_name)
        elif is_stdlib_module(module_name, language):
            # Standard library = external (but not third-party)
            self._external_imports.add(module_name)
        else:
            # Check if internal by checking if it starts with project root
            # For now, treat non-relative, non-stdlib as external
            # Future enhancement: project_root detection
            self._external_imports.add(module_name)

    def finalize_function(
        self, node: Node, context: CollectorContext
    ) -> dict[str, Any]:
        """Return empty dict - coupling is file-level, not function-level.

        Coupling metrics are computed at file level during finalization.

        Args:
            node: Function definition node
            context: Shared context

        Returns:
            Empty dictionary (no function-level coupling metrics)
        """
        return {}

    def get_file_metrics(self) -> dict[str, Any]:
        """Get file-level coupling metrics.

        Returns:
            Dictionary with efferent coupling metrics
        """
        return {
            "efferent_coupling": len(self._imports),
            "imports": sorted(self._imports),
            "internal_imports": sorted(self._internal_imports),
            "external_imports": sorted(self._external_imports),
        }

    def reset(self) -> None:
        """Reset collector state for next file."""
        self._imports.clear()
        self._internal_imports.clear()
        self._external_imports.clear()
