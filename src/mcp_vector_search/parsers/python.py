"""Python parser using Tree-sitter for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class PythonParser(BaseParser):
    """Python parser using Tree-sitter for AST-based code analysis."""

    def __init__(self) -> None:
        """Initialize Python parser."""
        super().__init__("python")
        self._parser = None
        self._language = None
        self._initialize_parser()

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
        """Parse a Python file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Python content and extract code chunks."""
        if not content.strip():
            return []

        # If Tree-sitter is not available, fall back to simple parsing
        if not self._parser:
            return await self._fallback_parse(content, file_path)

        try:
            # Parse with Tree-sitter
            tree = self._parser.parse(content.encode("utf-8"))
            return self._extract_chunks_from_tree(tree, content, file_path)
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
            return await self._fallback_parse(content, file_path)

    def _extract_chunks_from_tree(
        self, tree, content: str, file_path: Path
    ) -> list[CodeChunk]:
        """Extract code chunks from Tree-sitter AST."""
        chunks = []
        lines = self._split_into_lines(content)

        def visit_node(node, current_class=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            if node_type == "function_definition":
                chunks.extend(
                    self._extract_function(node, lines, file_path, current_class)
                )
            elif node_type == "class_definition":
                class_chunks = self._extract_class(node, lines, file_path)
                chunks.extend(class_chunks)

                # Visit class methods with class context
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name)
            elif node_type == "module":
                # Extract module-level code
                module_chunk = self._extract_module_chunk(node, lines, file_path)
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

    def _extract_function(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract function definition as a chunk."""
        chunks = []

        function_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get function content
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring if present
        docstring = self._extract_docstring(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=docstring,
        )
        chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract class definition as a chunk."""
        chunks = []

        class_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get class content
        content = self._get_line_range(lines, start_line, end_line)

        # Extract docstring if present
        docstring = self._extract_docstring(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
        )
        chunks.append(chunk)

        return chunks

    def _extract_module_chunk(
        self, node, lines: list[str], file_path: Path
    ) -> CodeChunk | None:
        """Extract module-level code (imports, constants, etc.)."""
        # Look for module-level statements (not inside functions/classes)
        module_lines = []

        for child in node.children:
            if child.type in ["import_statement", "import_from_statement"]:
                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1
                import_content = self._get_line_range(lines, start_line, end_line)
                module_lines.append(import_content.strip())

        if module_lines:
            content = "\n".join(module_lines)
            return self._create_chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(module_lines),
                chunk_type="imports",
            )

        return None

    def _get_node_name(self, node) -> str | None:
        """Extract name from a named node (function, class, etc.)."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _extract_docstring(self, node, lines: list[str]) -> str | None:
        """Extract docstring from a function or class node."""
        # Look for string literal as first statement in body
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr_child in stmt.children:
                            if expr_child.type == "string":
                                # Extract string content
                                start_line = expr_child.start_point[0] + 1
                                end_line = expr_child.end_point[0] + 1
                                docstring = self._get_line_range(
                                    lines, start_line, end_line
                                )
                                # Clean up docstring (remove quotes)
                                return self._clean_docstring(docstring)
        return None

    def _clean_docstring(self, docstring: str) -> str:
        """Clean up extracted docstring."""
        # Remove triple quotes and clean whitespace
        cleaned = re.sub(r'^["\']{{3}}|["\']{{3}}$', "", docstring.strip())
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned.strip())
        return cleaned.strip()

    async def _fallback_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing using regex when Tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Enhanced regex patterns
        function_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
        class_pattern = re.compile(r"^\s*class\s+(\w+)\s*[:\(]", re.MULTILINE)
        import_pattern = re.compile(r"^\s*(from\s+\S+\s+)?import\s+(.+)", re.MULTILINE)

        # Extract imports first
        imports = []
        for match in import_pattern.finditer(content):
            import_line = match.group(0).strip()
            imports.append(import_line)

        # Find functions
        for match in function_pattern.finditer(content):
            function_name = match.group(1)
            # Find the actual line with 'def' by looking for it in the match
            match_text = match.group(0)
            def_pos_in_match = match_text.find("def")
            actual_def_pos = match.start() + def_pos_in_match
            start_line = content[:actual_def_pos].count("\n") + 1

            # Find end of function (simple heuristic)
            end_line = self._find_function_end(lines, start_line)

            func_content = self._get_line_range(lines, start_line, end_line)

            if func_content.strip():  # Only add if content is not empty
                # Extract docstring using regex
                docstring = self._extract_docstring_regex(func_content)

                chunk = self._create_chunk(
                    content=func_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="function",
                    function_name=function_name,
                    docstring=docstring,
                )
                chunk.imports = imports  # Add imports to chunk
                chunks.append(chunk)

        # Find classes
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            # Find the actual line with 'class' by looking for it in the match
            match_text = match.group(0)
            class_pos_in_match = match_text.find("class")
            actual_class_pos = match.start() + class_pos_in_match
            start_line = content[:actual_class_pos].count("\n") + 1

            # Find end of class (simple heuristic)
            end_line = self._find_class_end(lines, start_line)

            class_content = self._get_line_range(lines, start_line, end_line)

            if class_content.strip():  # Only add if content is not empty
                # Extract class docstring
                docstring = self._extract_docstring_regex(class_content)

                chunk = self._create_chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    class_name=class_name,
                    docstring=docstring,
                )
                chunk.imports = imports  # Add imports to chunk
                chunks.append(chunk)

        # If no functions or classes found, create chunks for the whole file
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

    def _find_function_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a function using indentation."""
        if start_line > len(lines):
            return len(lines)

        # Get initial indentation of the def line
        start_idx = start_line - 1
        if start_idx >= len(lines):
            return len(lines)

        def_line = lines[start_idx]
        def_indent = len(def_line) - len(def_line.lstrip())

        # Find end by looking for line with indentation <= def indentation
        # Start from the line after the def line
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= def_indent:
                    return i  # Return 1-based line number (i is 0-based index)

        # If we reach here, the function goes to the end of the file
        return len(lines)

    def _find_class_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a class using indentation."""
        return self._find_function_end(lines, start_line)

    def _extract_docstring_regex(self, content: str) -> str | None:
        """Extract docstring using regex patterns."""
        # Look for triple-quoted strings at the beginning of the content
        # after the def/class line
        lines = content.splitlines()
        if len(lines) < 2:
            return None

        # Skip the def/class line and look for docstring in subsequent lines
        for i in range(1, min(len(lines), 5)):  # Check first few lines
            line = lines[i].strip()
            if not line:
                continue

            # Check for triple-quoted docstrings
            if line.startswith('"""') or line.startswith("'''"):
                quote_type = line[:3]

                # Single-line docstring
                if line.endswith(quote_type) and len(line) > 6:
                    return line[3:-3].strip()

                # Multi-line docstring
                docstring_lines = [line[3:]]
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line.endswith(quote_type):
                        docstring_lines.append(next_line[:-3])
                        break
                    docstring_lines.append(next_line)

                return " ".join(docstring_lines).strip()

            # If we hit non-docstring code, stop looking
            if line and not line.startswith("#"):
                break

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".py", ".pyw"]
