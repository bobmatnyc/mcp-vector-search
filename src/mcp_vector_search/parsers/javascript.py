"""JavaScript/TypeScript parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class JavaScriptParser(BaseParser):
    """JavaScript parser with tree-sitter AST support and fallback regex parsing."""

    def __init__(self, language: str = "javascript") -> None:
        """Initialize JavaScript parser."""
        super().__init__(language)
        self._parser = None
        self._language = None
        self._use_tree_sitter = False
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for JavaScript."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("javascript")
            self._parser = get_parser("javascript")

            logger.debug(
                "JavaScript Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            self._use_tree_sitter = True
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}, using regex fallback")
            self._use_tree_sitter = False

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a JavaScript/TypeScript file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse JavaScript/TypeScript content and extract code chunks."""
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
        """Extract code chunks from JavaScript AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            # Check if this node type should be extracted
            extracted = False

            if node_type == "function_declaration":
                chunks.extend(
                    self._extract_function(node, lines, file_path, current_class)
                )
                extracted = True
            elif node_type == "arrow_function":
                chunks.extend(
                    self._extract_arrow_function(node, lines, file_path, current_class)
                )
                extracted = True
            elif node_type == "class_declaration":
                class_chunks = self._extract_class(node, lines, file_path)
                chunks.extend(class_chunks)

                # Visit class methods
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name)
                extracted = True
            elif node_type == "method_definition":
                chunks.extend(
                    self._extract_method(node, lines, file_path, current_class)
                )
                extracted = True
            elif node_type == "lexical_declaration":
                # const/let declarations might be arrow functions
                extracted_chunks = self._extract_variable_function(
                    node, lines, file_path, current_class
                )
                if extracted_chunks:
                    chunks.extend(extracted_chunks)
                    extracted = True

            # Only recurse into children if we didn't extract this node
            # This prevents double-extraction of arrow functions in variable declarations
            if not extracted and hasattr(node, "children"):
                for child in node.children:
                    visit_node(child, current_class)

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
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract function declaration from AST."""
        function_name = self._get_node_name(node)
        if not function_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            chunk_depth=2 if class_name else 1,
        )
        return [chunk]

    def _extract_arrow_function(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract arrow function from AST."""
        # Arrow functions often don't have explicit names, try to get from parent
        parent = getattr(node, "parent", None)
        function_name = None

        if parent and parent.type == "variable_declarator":
            function_name = self._get_node_name(parent)

        if not function_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            chunk_depth=2 if class_name else 1,
        )
        return [chunk]

    def _extract_variable_function(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract function from variable declaration (const func = ...)."""
        chunks = []

        for child in node.children:
            if child.type == "variable_declarator":
                # Check if it's a function assignment
                for subchild in child.children:
                    if subchild.type in ("arrow_function", "function"):
                        func_name = self._get_node_name(child)
                        if func_name:
                            start_line = child.start_point[0] + 1
                            end_line = child.end_point[0] + 1

                            content = child.text.decode()
                            docstring = self._extract_jsdoc_from_node(child, lines)

                            # Calculate complexity
                            complexity = self._calculate_complexity(
                                subchild, "javascript"
                            )

                            # Extract parameters
                            parameters = self._extract_js_parameters(subchild)

                            chunk = self._create_chunk(
                                content=content,
                                file_path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                chunk_type="function",
                                function_name=func_name,
                                class_name=class_name,
                                docstring=docstring,
                                complexity_score=complexity,
                                parameters=parameters,
                                chunk_depth=2 if class_name else 1,
                            )
                            chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract class declaration from AST."""
        class_name = self._get_node_name(node)
        if not class_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            chunk_depth=1,
        )
        return [chunk]

    def _extract_method(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract method definition from class."""
        method_name = self._get_node_name(node)
        if not method_name:
            return []

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = node.text.decode()
        docstring = self._extract_jsdoc_from_node(node, lines)

        # Calculate complexity
        complexity = self._calculate_complexity(node, "javascript")

        # Extract parameters
        parameters = self._extract_js_parameters(node)

        # Check for decorators (TypeScript)
        decorators = self._extract_decorators_from_node(node)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            parameters=parameters,
            decorators=decorators,
            chunk_depth=2,
        )
        return [chunk]

    def _get_node_name(self, node) -> str | None:
        """Extract name from a named node."""
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return child.text.decode("utf-8")
        return None

    def _get_node_text(self, node) -> str:
        """Get text content of a node."""
        if hasattr(node, "text"):
            return node.text.decode("utf-8")
        return ""

    def _extract_js_parameters(self, node) -> list[dict]:
        """Extract function parameters from JavaScript/TypeScript AST."""
        parameters = []

        for child in node.children:
            if child.type == "formal_parameters":
                for param_node in child.children:
                    if param_node.type in (
                        "identifier",
                        "required_parameter",
                        "optional_parameter",
                        "rest_parameter",
                    ):
                        param_info = {"name": None, "type": None, "default": None}

                        # Extract parameter details
                        if param_node.type == "identifier":
                            param_info["name"] = self._get_node_text(param_node)
                        else:
                            # TypeScript typed parameters
                            for subchild in param_node.children:
                                if subchild.type == "identifier":
                                    param_info["name"] = self._get_node_text(subchild)
                                elif subchild.type == "type_annotation":
                                    param_info["type"] = self._get_node_text(subchild)
                                elif (
                                    "default" in subchild.type
                                    or subchild.type == "number"
                                ):
                                    param_info["default"] = self._get_node_text(
                                        subchild
                                    )

                        if param_info["name"] and param_info["name"] not in (
                            "(",
                            ")",
                            ",",
                            "...",
                        ):
                            # Clean up rest parameters
                            if param_info["name"].startswith("..."):
                                param_info["name"] = param_info["name"][3:]
                                param_info["rest"] = True
                            parameters.append(param_info)

        return parameters

    def _extract_decorators_from_node(self, node) -> list[str]:
        """Extract decorators from TypeScript node."""
        decorators = []

        for child in node.children:
            if child.type == "decorator":
                decorators.append(self._get_node_text(child))

        return decorators

    def _extract_jsdoc_from_node(self, node, lines: list[str]) -> str | None:
        """Extract JSDoc comment from before a node."""
        start_line = node.start_point[0]
        return self._extract_jsdoc(lines, start_line + 1)

    async def _regex_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse JavaScript/TypeScript using regex patterns."""
        chunks = []
        lines = self._split_into_lines(content)

        # JavaScript/TypeScript patterns
        function_patterns = [
            re.compile(r"^\s*function\s+(\w+)\s*\(", re.MULTILINE),  # function name()
            re.compile(
                r"^\s*const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{", re.MULTILINE
            ),  # const name = () => {
            re.compile(
                r"^\s*const\s+(\w+)\s*=\s*function\s*\(", re.MULTILINE
            ),  # const name = function(
            re.compile(
                r"^\s*(\w+)\s*:\s*function\s*\(", re.MULTILINE
            ),  # name: function(
            re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*{", re.MULTILINE),  # name() { (method)
            re.compile(
                r"^\s*async\s+function\s+(\w+)\s*\(", re.MULTILINE
            ),  # async function name()
            re.compile(
                r"^\s*async\s+(\w+)\s*\([^)]*\)\s*{", re.MULTILINE
            ),  # async name() {
        ]

        class_patterns = [
            re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),  # class Name
            re.compile(
                r"^\s*export\s+class\s+(\w+)", re.MULTILINE
            ),  # export class Name
            re.compile(
                r"^\s*export\s+default\s+class\s+(\w+)", re.MULTILINE
            ),  # export default class Name
        ]

        interface_patterns = [
            re.compile(
                r"^\s*interface\s+(\w+)", re.MULTILINE
            ),  # interface Name (TypeScript)
            re.compile(
                r"^\s*export\s+interface\s+(\w+)", re.MULTILINE
            ),  # export interface Name
        ]

        import_pattern = re.compile(r"^\s*(import|export).*", re.MULTILINE)

        # Extract imports
        imports = []
        for match in import_pattern.finditer(content):
            import_line = match.group(0).strip()
            imports.append(import_line)

        # Extract functions
        for pattern in function_patterns:
            for match in pattern.finditer(content):
                function_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find end of function
                end_line = self._find_block_end(lines, start_line, "{", "}")

                func_content = self._get_line_range(lines, start_line, end_line)

                if func_content.strip():
                    # Extract JSDoc comment
                    jsdoc = self._extract_jsdoc(lines, start_line)

                    chunk = self._create_chunk(
                        content=func_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="function",
                        function_name=function_name,
                        docstring=jsdoc,
                    )
                    chunk.imports = imports
                    chunks.append(chunk)

        # Extract classes
        for pattern in class_patterns:
            for match in pattern.finditer(content):
                class_name = match.group(1)
                start_line = content[: match.start()].count("\n") + 1

                # Find end of class
                end_line = self._find_block_end(lines, start_line, "{", "}")

                class_content = self._get_line_range(lines, start_line, end_line)

                if class_content.strip():
                    # Extract JSDoc comment
                    jsdoc = self._extract_jsdoc(lines, start_line)

                    chunk = self._create_chunk(
                        content=class_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="class",
                        class_name=class_name,
                        docstring=jsdoc,
                    )
                    chunk.imports = imports
                    chunks.append(chunk)

        # Extract interfaces (TypeScript)
        if self.language == "typescript":
            for pattern in interface_patterns:
                for match in pattern.finditer(content):
                    interface_name = match.group(1)
                    start_line = content[: match.start()].count("\n") + 1

                    # Find end of interface
                    end_line = self._find_block_end(lines, start_line, "{", "}")

                    interface_content = self._get_line_range(
                        lines, start_line, end_line
                    )

                    if interface_content.strip():
                        # Extract JSDoc comment
                        jsdoc = self._extract_jsdoc(lines, start_line)

                        chunk = self._create_chunk(
                            content=interface_content,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type="interface",
                            class_name=interface_name,  # Use class_name field for interface
                            docstring=jsdoc,
                        )
                        chunk.imports = imports
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

    def _find_block_end(
        self, lines: list[str], start_line: int, open_char: str, close_char: str
    ) -> int:
        """Find the end of a block by matching braces."""
        if start_line > len(lines):
            return len(lines)

        brace_count = 0
        found_opening = False

        for i in range(start_line - 1, len(lines)):
            line = lines[i]

            for char in line:
                if char == open_char:
                    brace_count += 1
                    found_opening = True
                elif char == close_char:
                    brace_count -= 1

                    if found_opening and brace_count == 0:
                        return i + 1  # Return 1-based line number

        return len(lines)

    def _extract_jsdoc(self, lines: list[str], start_line: int) -> str | None:
        """Extract JSDoc comment before a function/class."""
        if start_line <= 1:
            return None

        # Look backwards for JSDoc comment
        for i in range(start_line - 2, max(-1, start_line - 10), -1):
            line = lines[i].strip()

            if line.endswith("*/"):
                # Found end of JSDoc, collect the comment
                jsdoc_lines = []
                for j in range(i, -1, -1):
                    comment_line = lines[j].strip()
                    jsdoc_lines.insert(0, comment_line)

                    if comment_line.startswith("/**"):
                        # Found start of JSDoc
                        # Clean up the comment
                        cleaned_lines = []
                        for line in jsdoc_lines:
                            # Remove /** */ and * prefixes
                            cleaned = (
                                line.replace("/**", "")
                                .replace("*/", "")
                                .replace("*", "")
                                .strip()
                            )
                            if cleaned:
                                cleaned_lines.append(cleaned)

                        return " ".join(cleaned_lines) if cleaned_lines else None

            # If we hit non-comment code, stop looking
            elif line and not line.startswith("//") and not line.startswith("*"):
                break

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        if self.language == "typescript":
            return [".ts", ".tsx"]
        else:
            return [".js", ".jsx", ".mjs"]


class TypeScriptParser(JavaScriptParser):
    """TypeScript parser extending JavaScript parser."""

    def __init__(self) -> None:
        """Initialize TypeScript parser."""
        super().__init__("typescript")

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for TypeScript."""
        try:
            from tree_sitter_language_pack import get_language, get_parser

            self._language = get_language("typescript")
            self._parser = get_parser("typescript")

            logger.debug(
                "TypeScript Tree-sitter parser initialized via tree-sitter-language-pack"
            )
            self._use_tree_sitter = True
            return
        except Exception as e:
            logger.debug(f"tree-sitter-language-pack failed: {e}, using regex fallback")
            self._use_tree_sitter = False
