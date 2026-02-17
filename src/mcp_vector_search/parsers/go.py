"""Go parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class GoParser(BaseParser):
    """Go parser with tree-sitter AST support and fallback regex parsing."""

    def __init__(self) -> None:
        """Initialize Go parser."""
        super().__init__("go")
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialized = False

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Go."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("go")
            self._parser = get_parser("go")

            logger.debug(
                "Go Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            self._use_tree_sitter = True
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}, using regex fallback")
            self._use_tree_sitter = False

    def _ensure_parser_initialized(self) -> None:
        """Ensure tree-sitter parser is initialized (lazy loading)."""
        if not self._initialized:
            self._initialize_parser()
            self._initialized = True

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Go file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Go content and extract code chunks."""
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

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from Go AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_type=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            # Check if this node type should be extracted
            extracted = False

            if node_type == "function_declaration":
                chunks.extend(
                    self._extract_function(node, lines, file_path, current_type)
                )
                extracted = True
            elif node_type == "method_declaration":
                chunks.extend(
                    self._extract_method(node, lines, file_path, current_type)
                )
                extracted = True
            elif node_type == "type_declaration":
                # Check if it's a struct, interface, or type alias
                type_chunks = self._extract_type_declaration(node, lines, file_path)
                chunks.extend(type_chunks)

                # Visit type members
                type_name = self._get_type_name(node)
                for child in node.children:
                    visit_node(child, type_name)
                extracted = True

            # If not extracted as a specific chunk, visit children
            if not extracted:
                for child in node.children:
                    visit_node(child, current_type)

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

    def _extract_function(
        self, node, lines: list[str], file_path: Path, current_type: str | None
    ) -> list[CodeChunk]:
        """Extract function declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        function_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "go")

        # Extract parameters and return type
        parameters = self._extract_parameters(node)
        return_type = self._extract_return_type(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            return_type=return_type,
        )

        return [chunk]

    def _extract_method(
        self, node, lines: list[str], file_path: Path, current_type: str | None
    ) -> list[CodeChunk]:
        """Extract method declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        method_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract receiver type
        receiver_type = self._extract_receiver_type(node)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "go")

        # Extract parameters and return type
        parameters = self._extract_parameters(node)
        return_type = self._extract_return_type(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=receiver_type or current_type,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            return_type=return_type,
        )

        return [chunk]

    def _extract_type_declaration(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract type declaration from AST node."""
        chunks = []

        # Find type_spec nodes
        for child in node.children:
            if child.type == "type_spec":
                chunk = self._extract_type_spec(child, lines, file_path)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _extract_type_spec(
        self, node, lines: list[str], file_path: Path
    ) -> CodeChunk | None:
        """Extract a single type specification."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        type_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Determine type kind (struct, interface, etc.)
        chunk_type = "class"  # Default
        for child in node.children:
            if child.type == "struct_type":
                chunk_type = "struct"
            elif child.type == "interface_type":
                chunk_type = "interface"

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        return self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            class_name=type_name,
            docstring=docstring,
        )

    def _get_node_name(self, node) -> str:
        """Extract name from AST node."""
        for child in node.children:
            if child.type in ["identifier", "type_identifier", "field_identifier"]:
                return child.text.decode("utf-8")
        return "unknown"

    def _get_type_name(self, node) -> str | None:
        """Extract type name from type_declaration node."""
        for child in node.children:
            if child.type == "type_spec":
                return self._get_node_name(child)
        return None

    def _extract_receiver_type(self, node) -> str | None:
        """Extract receiver type from method declaration."""
        for child in node.children:
            if child.type == "parameter_list":
                # First parameter_list is the receiver
                for param_child in child.children:
                    if param_child.type in [
                        "parameter_declaration",
                        "type_identifier",
                        "pointer_type",
                    ]:
                        return param_child.text.decode("utf-8").strip("()")
        return None

    def _extract_doc_comment(self, node, lines: list[str]) -> str | None:
        """Extract doc comment before node."""
        # Look for a comment node just before this node
        parent = node.parent
        if not parent:
            return None

        # Find the index of current node in parent's children
        try:
            node_index = parent.children.index(node)
            if node_index > 0:
                prev_node = parent.children[node_index - 1]
                if prev_node.type == "comment":
                    comment_text = prev_node.text.decode("utf-8")
                    # Go doc comments start with //
                    if comment_text.strip().startswith("//"):
                        return comment_text.strip()
        except (ValueError, IndexError):
            pass

        return None

    def _extract_parameters(self, node) -> list[dict]:
        """Extract parameters from function/method node."""
        parameters = []

        # Find parameter_list nodes (skip first one if it's a method receiver)
        param_lists = [
            child for child in node.children if child.type == "parameter_list"
        ]

        # For methods, skip the first parameter_list (receiver)
        is_method = node.type == "method_declaration"
        start_index = 1 if is_method and len(param_lists) > 1 else 0

        for param_list in param_lists[start_index:]:
            for param_child in param_list.children:
                if param_child.type == "parameter_declaration":
                    param_info = self._parse_parameter(param_child)
                    if param_info:
                        parameters.append(param_info)

        return parameters

    def _parse_parameter(self, param_node) -> dict | None:
        """Parse a single parameter node."""
        param_names = []
        param_type = None

        for child in param_node.children:
            if child.type == "identifier":
                param_names.append(child.text.decode("utf-8"))
            elif child.type in [
                "type_identifier",
                "pointer_type",
                "slice_type",
                "array_type",
                "map_type",
                "interface_type",
                "struct_type",
            ]:
                param_type = child.text.decode("utf-8")

        # Go allows multiple names with same type: (a, b int)
        if param_names and param_type:
            return {
                "name": ", ".join(param_names),
                "type": param_type,
            }
        elif param_type:
            # Unnamed parameter
            return {
                "name": "_",
                "type": param_type,
            }

        return None

    def _extract_return_type(self, node) -> str | None:
        """Extract return type from function/method node."""
        # Look for parameter_list that represents return values
        # In Go, return values come after parameters
        param_lists = [
            child for child in node.children if child.type == "parameter_list"
        ]

        # For functions, the last parameter_list is the return type
        # For methods, skip the receiver (first one)
        is_method = node.type == "method_declaration"

        if is_method and len(param_lists) >= 3:
            # method: receiver, params, return
            return_list = param_lists[2]
        elif not is_method and len(param_lists) >= 2:
            # function: params, return
            return_list = param_lists[1]
        else:
            return None

        # Extract return type text
        return_text = return_list.text.decode("utf-8").strip("()")
        return return_text if return_text else None

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback regex-based parsing when tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Go patterns
        function_patterns = [
            re.compile(
                r"^\s*func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\)|[\w\[\]\*]+)?\s*\{",
                re.MULTILINE,
            ),  # Function declarations
        ]

        method_patterns = [
            re.compile(
                r"^\s*func\s+\([^)]+\)\s*(\w+)\s*\([^)]*\)\s*(?:\([^)]*\)|[\w\[\]\*]+)?\s*\{",
                re.MULTILINE,
            ),  # Method declarations with receiver
        ]

        type_patterns = [
            re.compile(r"^\s*type\s+(\w+)\s+struct\s*\{", re.MULTILINE),  # struct
            re.compile(r"^\s*type\s+(\w+)\s+interface\s*\{", re.MULTILINE),  # interface
        ]

        # Extract types (structs, interfaces)
        for pattern in type_patterns:
            for match in pattern.finditer(content):
                type_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the type (matching braces)
                type_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if type_content:
                    chunk_type = (
                        "struct" if "struct" in pattern.pattern else "interface"
                    )
                    chunk = self._create_chunk(
                        content=type_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=chunk_type,
                        class_name=type_name,
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

        # Extract functions
        for pattern in function_patterns:
            for match in pattern.finditer(content):
                function_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the function (matching braces)
                function_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if function_content:
                    chunk = self._create_chunk(
                        content=function_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="function",
                        function_name=function_name,
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
        in_rune = False
        in_raw_string = False
        escape_next = False

        while pos < len(content):
            char = content[pos]

            # Handle raw strings (backticks)
            if char == "`":
                in_raw_string = not in_raw_string
            elif in_raw_string:
                pos += 1
                continue

            # Handle string/rune literals and escape sequences
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"' and not in_rune:
                in_string = not in_string
            elif char == "'" and not in_string:
                in_rune = not in_rune
            elif not in_string and not in_rune:
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

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".go"]

    def _calculate_complexity(self, node, language: str | None = None) -> float:
        """Calculate cyclomatic complexity for Go code."""
        if language is None:
            language = "go"

        if not hasattr(node, "children"):
            return 1.0

        complexity = 1.0  # Base complexity

        # Go decision node types
        decision_nodes = {
            "if_statement",
            "for_statement",
            "switch_statement",
            "type_switch_statement",
            "select_statement",
            "case_clause",
            "default_case",
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
