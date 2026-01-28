"""Python parser using Tree-sitter for MCP Vector Search."""

from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser
from .python_helpers import (
    ClassExtractor,
    FunctionExtractor,
    ModuleExtractor,
    RegexFallbackParser,
)


class PythonParser(BaseParser):
    """Python parser using Tree-sitter for AST-based code analysis.

    This parser coordinates multiple helper classes to extract different
    types of code elements (functions, classes, modules) from Python source.
    """

    def __init__(self) -> None:
        """Initialize Python parser."""
        super().__init__("python")
        self._parser = None
        self._language = None
        self._initialize_parser()

        # Initialize extractors
        self._function_extractor = FunctionExtractor(self)
        self._class_extractor = ClassExtractor(self)
        self._module_extractor = ModuleExtractor(self)
        self._fallback_parser = RegexFallbackParser(self)

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Python."""
        try:
            # Try the tree-sitter-language-pack package (maintained alternative)
            from tree_sitter_language_pack import get_language, get_parser

            # Get the language and parser objects
            self._language = get_language("python")
            self._parser = get_parser("python")

            logger.debug(
                "Python Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}")

        try:
            # Fallback to manual tree-sitter setup (requires language binaries)
            # This would require language binaries to be available
            # For now, we'll skip this and rely on fallback parsing
            logger.debug("Manual tree-sitter setup not implemented yet")
            self._parser = None
            self._language = None
        except Exception as e:
            logger.debug(f"Manual tree-sitter setup failed: {e}")
            self._parser = None
            self._language = None

        logger.info(
            "Using fallback regex-based parsing for Python (Tree-sitter unavailable)"
        )

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Python file and extract code chunks.

        Args:
            file_path: Path to the Python file to parse

        Returns:
            List of extracted code chunks
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Python content and extract code chunks.

        Args:
            content: Python source code
            file_path: Path to source file (for metadata)

        Returns:
            List of extracted code chunks
        """
        if not content.strip():
            return []

        # If Tree-sitter is not available, fall back to simple parsing
        if not self._parser:
            return await self._fallback_parser.parse(content, file_path)

        try:
            # Parse with Tree-sitter
            tree = self._parser.parse(content.encode("utf-8"))
            return self._extract_chunks_from_tree(tree, content, file_path)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            return await self._fallback_parser.parse(content, file_path)

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from Tree-sitter AST.

        Args:
            tree: Parsed tree-sitter tree
            content: Source code content
            file_path: Path to source file

        Returns:
            List of extracted code chunks
        """
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes.

            Args:
                node: Tree-sitter node to visit
                current_class: Current class context (for methods)
            """
            node_type = node.type

            if node_type == "function_definition":
                chunks.extend(
                    self._function_extractor.extract(
                        node, lines, file_path, current_class
                    )
                )
            elif node_type == "class_definition":
                class_chunks = self._class_extractor.extract(node, lines, file_path)
                chunks.extend(class_chunks)

                # Visit class methods with class context
                from .python_helpers.metadata_extractor import MetadataExtractor

                class_name = MetadataExtractor.get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name)
            elif node_type == "module":
                # Extract module-level code
                module_chunk = self._module_extractor.extract(node, lines, file_path)
                if module_chunk:
                    chunks.append(module_chunk)

                # Visit all children
                for child in node.children:
                    visit_node(child)
            else:
                # Visit children for other node types
                for child in node.children:
                    visit_node(child, current_class)

        # Start traversal from root
        visit_node(tree.root_node)

        # If no specific chunks found, create a single chunk for the whole file
        if not chunks:
            chunks.append(
                self._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                )
            )

        return chunks

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions.

        Returns:
            List of supported file extensions
        """
        return [".py", ".pyw"]
