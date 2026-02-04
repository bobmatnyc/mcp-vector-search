"""Rust parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class RustParser(BaseParser):
    """Rust parser with tree-sitter AST support and fallback regex parsing."""

    def __init__(self) -> None:
        """Initialize Rust parser."""
        super().__init__("rust")
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Rust."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("rust")
            self._parser = get_parser("rust")

            logger.debug(
                "Rust Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            self._use_tree_sitter = True
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}, using regex fallback")
            self._use_tree_sitter = False

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Rust file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Rust content and extract code chunks."""
        if not content.strip():
            return []

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
        """Extract code chunks from Rust AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_impl=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            # Check if this node type should be extracted
            extracted = False

            if node_type == "function_item":
                chunks.extend(
                    self._extract_function(node, lines, file_path, current_impl)
                )
                extracted = True
            elif node_type == "impl_item":
                # Extract impl block and its methods
                impl_chunks = self._extract_impl_block(node, lines, file_path)
                chunks.extend(impl_chunks)

                # Visit impl members
                impl_type = self._get_impl_type(node)
                for child in node.children:
                    visit_node(child, impl_type)
                extracted = True
            elif node_type == "struct_item":
                chunks.extend(self._extract_struct(node, lines, file_path))
                extracted = True
            elif node_type == "enum_item":
                chunks.extend(self._extract_enum(node, lines, file_path))
                extracted = True
            elif node_type == "trait_item":
                chunks.extend(self._extract_trait(node, lines, file_path))
                extracted = True
            elif node_type == "mod_item":
                chunks.extend(self._extract_module(node, lines, file_path))
                extracted = True

            # If not extracted as a specific chunk, visit children
            if not extracted:
                for child in node.children:
                    visit_node(child, current_impl)

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
        self, node, lines: list[str], file_path: Path, current_impl: str | None
    ) -> list[CodeChunk]:
        """Extract function declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        function_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Extract attributes
        attributes = self._extract_attributes(node)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "rust")

        # Extract parameters and return type
        parameters = self._extract_parameters(node)
        return_type = self._extract_return_type(node)

        # Determine if it's a method (inside impl block)
        chunk_type = "method" if current_impl else "function"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            function_name=function_name,
            class_name=current_impl,
            docstring=docstring,
            complexity_score=complexity,
            decorators=attributes,
            parameters=parameters,
            return_type=return_type,
        )

        return [chunk]

    def _extract_impl_block(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract impl block from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        impl_type = self._get_impl_type(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="impl",
            class_name=impl_type,
            docstring=docstring,
        )

        return [chunk]

    def _extract_struct(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract struct declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        struct_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Extract attributes
        attributes = self._extract_attributes(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="struct",
            class_name=struct_name,
            docstring=docstring,
            decorators=attributes,
        )

        return [chunk]

    def _extract_enum(self, node, lines: list[str], file_path: Path) -> list[CodeChunk]:
        """Extract enum declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        enum_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Extract attributes
        attributes = self._extract_attributes(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="enum",
            class_name=enum_name,
            docstring=docstring,
            decorators=attributes,
        )

        return [chunk]

    def _extract_trait(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract trait declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        trait_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        # Extract attributes
        attributes = self._extract_attributes(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="trait",
            class_name=trait_name,
            docstring=docstring,
            decorators=attributes,
        )

        return [chunk]

    def _extract_module(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract module declaration from AST node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        module_name = self._get_node_name(node)
        content = self._get_line_range(lines, start_line, end_line)

        # Extract doc comment
        docstring = self._extract_doc_comment(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="module",
            class_name=module_name,
            docstring=docstring,
        )

        return [chunk]

    def _get_node_name(self, node) -> str:
        """Extract name from AST node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
            elif child.type == "type_identifier":
                return child.text.decode("utf-8")
        return "unknown"

    def _get_impl_type(self, node) -> str:
        """Extract type name from impl block."""
        for child in node.children:
            if child.type in ["type_identifier", "generic_type"]:
                return child.text.decode("utf-8")
        return "unknown"

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
                if prev_node.type in ["line_comment", "block_comment"]:
                    comment_text = prev_node.text.decode("utf-8")
                    # Rust doc comments start with /// or //!
                    if comment_text.strip().startswith(("///", "//!")):
                        return comment_text.strip()
        except (ValueError, IndexError):
            pass

        return None

    def _extract_attributes(self, node) -> list[str]:
        """Extract attributes from node."""
        attributes = []

        # Attributes are siblings before the node in Rust
        parent = node.parent
        if parent:
            try:
                node_index = parent.children.index(node)
                # Look at previous siblings for attribute_item nodes
                for i in range(node_index - 1, -1, -1):
                    sibling = parent.children[i]
                    if sibling.type == "attribute_item":
                        attribute_text = sibling.text.decode("utf-8")
                        attributes.insert(
                            0, attribute_text
                        )  # Insert at beginning to preserve order
                    elif sibling.type not in ["\n", " "]:
                        # Stop when we hit a non-attribute, non-whitespace node
                        break
            except (ValueError, IndexError):
                pass

        return attributes

    def _extract_parameters(self, node) -> list[dict]:
        """Extract parameters from function node."""
        parameters = []

        # Find parameters node
        for child in node.children:
            if child.type == "parameters":
                for param_child in child.children:
                    if param_child.type == "parameter":
                        param_info = self._parse_parameter(param_child)
                        if param_info:
                            parameters.append(param_info)
                    elif param_child.type == "self_parameter":
                        # Handle self parameter
                        parameters.append(
                            {
                                "name": param_child.text.decode("utf-8"),
                                "type": "Self",
                            }
                        )

        return parameters

    def _parse_parameter(self, param_node) -> dict | None:
        """Parse a single parameter node."""
        param_name = None
        param_type = None

        for child in param_node.children:
            if child.type == "identifier":
                param_name = child.text.decode("utf-8")
            elif child.type in [
                "type_identifier",
                "reference_type",
                "generic_type",
                "array_type",
                "tuple_type",
                "pointer_type",
            ]:
                param_type = child.text.decode("utf-8")

        if param_name:
            return {
                "name": param_name,
                "type": param_type or "unknown",
            }

        return None

    def _extract_return_type(self, node) -> str | None:
        """Extract return type from function node."""
        for child in node.children:
            if child.type in [
                "type_identifier",
                "reference_type",
                "generic_type",
                "array_type",
                "tuple_type",
                "pointer_type",
            ]:
                # Look for the return type after the arrow
                parent_text = node.text.decode("utf-8")
                if "->" in parent_text:
                    # Extract everything after ->
                    return_part = parent_text.split("->")[1].strip()
                    # Remove the function body
                    if "{" in return_part:
                        return_part = return_part[: return_part.index("{")].strip()
                    return return_part
        return None

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback regex-based parsing when tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Rust patterns
        function_patterns = [
            re.compile(
                r"^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+)?fn\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)",
                re.MULTILINE,
            ),  # Function declarations
        ]

        impl_patterns = [
            re.compile(
                r"^\s*impl(?:<[^>]+>)?\s+(\w+)(?:<[^>]+>)?\s*\{", re.MULTILINE
            ),  # impl blocks
        ]

        struct_patterns = [
            re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),  # struct
        ]

        enum_patterns = [
            re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)", re.MULTILINE),  # enum
        ]

        trait_patterns = [
            re.compile(r"^\s*(?:pub\s+)?trait\s+(\w+)", re.MULTILINE),  # trait
        ]

        # Extract structs
        for pattern in struct_patterns:
            for match in pattern.finditer(content):
                struct_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the struct
                struct_content, end_line = self._extract_struct_or_enum(
                    content, match.start(), lines
                )

                if struct_content:
                    chunk = self._create_chunk(
                        content=struct_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="struct",
                        class_name=struct_name,
                    )
                    chunks.append(chunk)

        # Extract enums
        for pattern in enum_patterns:
            for match in pattern.finditer(content):
                enum_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the enum
                enum_content, end_line = self._extract_struct_or_enum(
                    content, match.start(), lines
                )

                if enum_content:
                    chunk = self._create_chunk(
                        content=enum_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="enum",
                        class_name=enum_name,
                    )
                    chunks.append(chunk)

        # Extract traits
        for pattern in trait_patterns:
            for match in pattern.finditer(content):
                trait_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the trait
                trait_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if trait_content:
                    chunk = self._create_chunk(
                        content=trait_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="trait",
                        class_name=trait_name,
                    )
                    chunks.append(chunk)

        # Extract impl blocks
        for pattern in impl_patterns:
            for match in pattern.finditer(content):
                impl_type = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the impl block
                impl_content, end_line = self._extract_block(
                    content, match.start(), lines
                )

                if impl_content:
                    chunk = self._create_chunk(
                        content=impl_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="impl",
                        class_name=impl_type,
                    )
                    chunks.append(chunk)

        # Extract functions
        for pattern in function_patterns:
            for match in pattern.finditer(content):
                function_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find the end of the function
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

    def _extract_struct_or_enum(
        self, content: str, start_pos: int, lines: list[str]
    ) -> tuple[str, int]:
        """Extract struct or enum declaration (may not have braces)."""
        # Find either semicolon (tuple struct) or opening brace (regular struct/enum)
        semicolon_pos = content.find(";", start_pos)
        brace_pos = content.find("{", start_pos)

        # If semicolon comes first, it's a tuple struct/enum
        if semicolon_pos != -1 and (brace_pos == -1 or semicolon_pos < brace_pos):
            end_line = content[: semicolon_pos + 1].count("\n") + 1
            return content[start_pos : semicolon_pos + 1], end_line

        # Otherwise, extract block with braces
        if brace_pos != -1:
            return self._extract_block(content, start_pos, lines)

        return None, 0

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

            # Handle raw strings (r#"..."#)
            if pos + 2 < len(content) and content[pos : pos + 2] == "r#":
                # Find the closing #"
                end_marker = content.find('#"', pos + 2)
                if end_marker != -1:
                    pos = end_marker + 2
                    continue

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
        return [".rs"]

    def _calculate_complexity(self, node, language: str | None = None) -> float:
        """Calculate cyclomatic complexity for Rust code."""
        if language is None:
            language = "rust"

        if not hasattr(node, "children"):
            return 1.0

        complexity = 1.0  # Base complexity

        # Rust decision node types
        decision_nodes = {
            "if_expression",
            "match_expression",
            "while_expression",
            "for_expression",
            "loop_expression",
            "match_arm",
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
