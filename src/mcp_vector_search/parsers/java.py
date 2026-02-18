"""Java parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class JavaParser(BaseParser):
    """Java parser with tree-sitter AST support and fallback regex parsing."""

    def __init__(self) -> None:
        """Initialize Java parser with lazy grammar loading."""
        super().__init__("java")
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialized = False

    def _ensure_parser_initialized(self) -> None:
        """Ensure tree-sitter parser is initialized (lazy loading)."""
        if not self._initialized:
            self._initialize_parser()
            self._initialized = True

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Java."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("java")
            self._parser = get_parser("java")

            logger.debug(
                "Java Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            self._use_tree_sitter = True
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}, using regex fallback")
            self._use_tree_sitter = False

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Java file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Java content and extract code chunks."""
        if not content.strip():
            return []

        # Lazy load parser on first use
        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                tree = self._parser.parse(content.encode("utf-8"))
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                return await self._regex_parse(content, file_path)
        else:
            return await self._regex_parse(content, file_path)

    def parse_file_sync(self, file_path: Path) -> list[CodeChunk]:
        """Parse file synchronously (optimized for multiprocessing workers)."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self._parse_content_sync(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    def _parse_content_sync(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse content synchronously without async overhead."""
        if not content.strip():
            return []

        # Lazy load parser on first use
        self._ensure_parser_initialized()

        if self._use_tree_sitter:
            try:
                tree = self._parser.parse(content.encode("utf-8"))
                return self._extract_chunks_from_tree(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                import asyncio

                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._regex_parse(content, file_path)
                    )
                finally:
                    loop.close()
        else:
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._regex_parse(content, file_path))
            finally:
                loop.close()

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from Java AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            # Check if this node type should be extracted
            extracted = False

            if node_type == "method_declaration":
                chunks.extend(
                    self._extract_method(node, lines, file_path, current_class)
                )
                extracted = True
            elif node_type == "constructor_declaration":
                chunks.extend(
                    self._extract_constructor(node, lines, file_path, current_class)
                )
                extracted = True
            elif node_type == "class_declaration":
                class_chunks = self._extract_class(node, lines, file_path)
                chunks.extend(class_chunks)

                # Visit class members
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name)
                extracted = True
            elif node_type == "interface_declaration":
                interface_chunks = self._extract_interface(node, lines, file_path)
                chunks.extend(interface_chunks)

                # Visit interface members
                interface_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, interface_name)
                extracted = True
            elif node_type == "enum_declaration":
                enum_chunks = self._extract_enum(node, lines, file_path)
                chunks.extend(enum_chunks)
                extracted = True

            # If not extracted as a specific chunk, visit children
            if not extracted:
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

    def _extract_method(
        self, node, lines: list[str], file_path: Path, current_class: str | None
    ) -> list[CodeChunk]:
        """Extract method declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        method_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring (Javadoc comment before method)
        docstring = self._extract_javadoc(node, lines)

        # Extract annotations
        annotations = self._extract_annotations(node)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "java")

        # Extract parameters
        parameters = self._extract_parameters(node)

        # Extract return type
        return_type = self._extract_return_type(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=current_class,
            docstring=docstring,
            complexity_score=complexity,
            decorators=annotations,
            parameters=parameters,
            return_type=return_type,
        )

        return [chunk]

    def _extract_constructor(
        self, node, lines: list[str], file_path: Path, current_class: str | None
    ) -> list[CodeChunk]:
        """Extract constructor declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        constructor_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring (Javadoc comment before constructor)
        docstring = self._extract_javadoc(node, lines)

        # Extract annotations
        annotations = self._extract_annotations(node)

        # Extract parameters
        parameters = self._extract_parameters(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="constructor",
            function_name=constructor_name,
            class_name=current_class,
            docstring=docstring,
            decorators=annotations,
            parameters=parameters,
        )

        return [chunk]

    def _extract_class(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract class declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        class_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring (Javadoc comment before class)
        docstring = self._extract_javadoc(node, lines)

        # Extract annotations
        annotations = self._extract_annotations(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            decorators=annotations,
        )

        return [chunk]

    def _extract_interface(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract interface declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        interface_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring (Javadoc comment before interface)
        docstring = self._extract_javadoc(node, lines)

        # Extract annotations
        annotations = self._extract_annotations(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="interface",
            class_name=interface_name,
            docstring=docstring,
            decorators=annotations,
        )

        return [chunk]

    def _extract_enum(self, node, lines: list[str], file_path: Path) -> list[CodeChunk]:
        """Extract enum declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        enum_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring (Javadoc comment before enum)
        docstring = self._extract_javadoc(node, lines)

        # Extract annotations
        annotations = self._extract_annotations(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="enum",
            class_name=enum_name,
            docstring=docstring,
            decorators=annotations,
        )

        return [chunk]

    def _get_node_name(self, node) -> str:
        """Extract name from AST node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return "unknown"

    def _extract_javadoc(self, node, lines: list[str]) -> str | None:
        """Extract Javadoc comment before node."""
        # Look for a comment node just before this node
        parent = node.parent
        if not parent:
            return None

        # Find the index of current node in parent's children
        try:
            node_index = parent.children.index(node)
            if node_index > 0:
                prev_node = parent.children[node_index - 1]
                if prev_node.type in ["comment", "block_comment"]:
                    comment_text = prev_node.text.decode("utf-8")
                    # Check if it's a Javadoc comment (starts with /**)
                    if comment_text.strip().startswith("/**"):
                        return comment_text.strip()
        except (ValueError, IndexError):
            pass

        return None

    def _extract_annotations(self, node) -> list[str]:
        """Extract annotations from node."""
        annotations = []

        # Annotations are typically in a modifiers node
        for child in node.children:
            if child.type == "modifiers":
                # Look for annotation nodes within modifiers
                for modifier_child in child.children:
                    if modifier_child.type in ["annotation", "marker_annotation"]:
                        annotation_text = modifier_child.text.decode("utf-8")
                        annotations.append(annotation_text)
            elif child.type in ["annotation", "marker_annotation"]:
                # Direct annotation (less common but possible)
                annotation_text = child.text.decode("utf-8")
                annotations.append(annotation_text)

        return annotations

    def _extract_parameters(self, node) -> list[dict]:
        """Extract parameters from method/constructor node."""
        parameters = []

        # Find formal_parameters node
        for child in node.children:
            if child.type == "formal_parameters":
                for param_child in child.children:
                    if param_child.type == "formal_parameter":
                        param_info = self._parse_parameter(param_child)
                        if param_info:
                            parameters.append(param_info)

        return parameters

    def _parse_parameter(self, param_node) -> dict | None:
        """Parse a single parameter node."""
        param_type = None
        param_name = None

        for child in param_node.children:
            if child.type in [
                "type_identifier",
                "integral_type",
                "floating_point_type",
                "boolean_type",
                "generic_type",
                "array_type",
            ]:
                param_type = child.text.decode("utf-8")
            elif child.type == "identifier":
                param_name = child.text.decode("utf-8")

        if param_name:
            return {
                "name": param_name,
                "type": param_type or "unknown",
            }

        return None

    def _extract_return_type(self, node) -> str | None:
        """Extract return type from method node."""
        for child in node.children:
            if child.type in [
                "type_identifier",
                "integral_type",
                "floating_point_type",
                "boolean_type",
                "void_type",
                "generic_type",
                "array_type",
            ]:
                return child.text.decode("utf-8")
        return None

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback regex-based parsing when tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Java patterns
        method_patterns = [
            re.compile(
                r"^\s*(?:public|private|protected|static|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{",
                re.MULTILINE,
            ),  # Method declarations
        ]

        class_patterns = [
            re.compile(
                r"^\s*(?:public|private|protected)?\s*(?:static|final)?\s*(?:abstract)?\s*class\s+(\w+)",
                re.MULTILINE,
            ),  # class Name
            re.compile(
                r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)", re.MULTILINE
            ),  # interface Name
            re.compile(
                r"^\s*(?:public|private|protected)?\s*enum\s+(\w+)", re.MULTILINE
            ),  # enum Name
        ]

        import_pattern = re.compile(r"^\s*import\s+.*", re.MULTILINE)

        # Extract imports
        imports = []
        for match in import_pattern.finditer(content):
            import_line = match.group(0).strip()
            imports.append(import_line)

        # Extract classes/interfaces/enums
        for pattern in class_patterns:
            for match in pattern.finditer(content):
                class_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the class (matching braces)
                class_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if class_content:
                    chunk = self._create_chunk(
                        content=class_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="class",
                        class_name=class_name,
                    )
                    chunks.append(chunk)

        # Extract methods
        for pattern in method_patterns:
            for match in pattern.finditer(content):
                method_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the method (matching braces)
                method_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if method_content:
                    chunk = self._create_chunk(
                        content=method_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="method",
                        function_name=method_name,
                    )
                    chunks.append(chunk)

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

    def _extract_block(
        self, content: str, start_pos: int, lines: list[str]
    ) -> tuple[str, int]:
        """Extract a code block (matching braces) starting from position."""
        # Find the opening brace
        brace_start = content.find("{", start_pos)
        if brace_start == -1:
            return None, 0

        # Count braces to find matching closing brace
        brace_count = 0
        pos = brace_start
        in_string = False
        in_char = False
        escape_next = False

        while pos < len(content):
            char = content[pos]

            # Handle string/char literals and escape sequences
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"' and not in_char:
                in_string = not in_string
            elif char == "'" and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        block_content = content[start_pos : pos + 1]
                        end_line = content[: pos + 1].count("\n") + 1
                        return block_content, end_line

            pos += 1

        return None, 0

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".java"]

    def _calculate_complexity(self, node, language: str | None = None) -> float:
        """Calculate cyclomatic complexity for Java code."""
        if language is None:
            language = "java"

        if not hasattr(node, "children"):
            return 1.0

        complexity = 1.0  # Base complexity

        # Java decision node types
        decision_nodes = {
            "if_statement",
            "while_statement",
            "for_statement",
            "enhanced_for_statement",  # for-each loop
            "switch_expression",
            "catch_clause",
            "conditional_expression",  # ternary operator
            "binary_expression",  # && and || operators
        }

        def count_decision_points(n):
            nonlocal complexity
            if hasattr(n, "type") and n.type in decision_nodes:
                complexity += 1
            if hasattr(n, "children"):
                for child in n.children:
                    count_decision_points(child)

        count_decision_points(node)
        return complexity
