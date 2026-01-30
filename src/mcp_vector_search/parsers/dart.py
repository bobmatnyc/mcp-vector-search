"""Dart/Flutter parser using Tree-sitter for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class DartParser(BaseParser):
    """Dart/Flutter parser using Tree-sitter for AST-based code analysis."""

    def __init__(self) -> None:
        """Initialize Dart parser."""
        super().__init__("dart")
        self._parser = None
        self._language = None
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Dart."""
        try:
            # Try the tree-sitter-language-pack package (maintained alternative)
            from tree_sitter_language_pack import get_language, get_parser

            # Get the language and parser objects
            self._language = get_language("dart")
            self._parser = get_parser("dart")

            logger.debug(
                "Dart Tree-sitter parser initialized via tree-sitter-language-pack"
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
            "Using fallback regex-based parsing for Dart (Tree-sitter unavailable)"
        )

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Dart file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Dart content and extract code chunks."""
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

            if node_type in ["function_signature", "method_signature"]:
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
            elif node_type == "constructor_signature":
                chunks.extend(
                    self._extract_constructor(node, lines, file_path, current_class)
                )
            elif node_type == "mixin_declaration":
                chunks.extend(self._extract_mixin(node, lines, file_path))
            elif node_type == "program":
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
        content = node.text.decode()

        # Extract dartdoc if present
        dartdoc = self._extract_dartdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=function_name,
            class_name=class_name,
            docstring=dartdoc,
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
        content = node.text.decode()

        # Extract dartdoc if present
        dartdoc = self._extract_dartdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=dartdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_constructor(
        self, node, lines: list[str], file_path: Path, class_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract constructor definition as a chunk."""
        chunks = []

        constructor_name = self._get_node_name(node) or class_name
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get constructor content
        content = node.text.decode()

        # Extract dartdoc if present
        dartdoc = self._extract_dartdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="constructor",
            function_name=constructor_name,
            class_name=class_name,
            docstring=dartdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_mixin(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract mixin definition as a chunk."""
        chunks = []

        mixin_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get mixin content
        content = node.text.decode()

        # Extract dartdoc if present
        dartdoc = self._extract_dartdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="mixin",
            class_name=mixin_name,
            docstring=dartdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_module_chunk(
        self, node, lines: list[str], file_path: Path
    ) -> CodeChunk | None:
        """Extract module-level code (imports, exports, etc.)."""
        # Look for module-level statements (not inside functions/classes)
        module_lines = []

        for child in node.children:
            if child.type in ["import_or_export", "library_name"]:
                import_content = child.text.decode()
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

    def _extract_dartdoc(self, node, lines: list[str]) -> str | None:
        """Extract dartdoc from a function or class node."""
        # Look for documentation_comment node before the definition
        start_line = node.start_point[0]

        # Check a few lines before the node for dartdoc comments
        dartdoc_lines = []
        for i in range(max(0, start_line - 10), start_line):
            line = lines[i].strip()
            if line.startswith("///"):
                dartdoc_lines.append(line[3:].strip())
            elif line and not line.startswith("//"):
                # Stop if we hit non-comment code
                dartdoc_lines = []

        if dartdoc_lines:
            return " ".join(dartdoc_lines)

        return None

    async def _fallback_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing using regex when Tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Enhanced regex patterns for Dart
        # Match: class WidgetName extends StatelessWidget/StatefulWidget
        widget_pattern = re.compile(
            r"^\s*class\s+(\w+)\s+extends\s+(StatelessWidget|StatefulWidget)",
            re.MULTILINE,
        )
        # Match: class ClassName
        class_pattern = re.compile(r"^\s*class\s+(\w+)\s*[{<]", re.MULTILINE)
        # Match: Future<Type> funcName( or Type funcName( or void funcName(
        function_pattern = re.compile(
            r"^\s*(?:Future<[\w<>]+>|void|[\w<>]+)\s+(\w+)\s*\(", re.MULTILINE
        )
        # Match: import 'package:...' or import "..."
        import_pattern = re.compile(r"^\s*import\s+['\"](.+?)['\"]", re.MULTILINE)
        # Match: mixin MixinName
        mixin_pattern = re.compile(r"^\s*mixin\s+(\w+)", re.MULTILINE)

        # Extract imports first
        imports = []
        for match in import_pattern.finditer(content):
            import_line = match.group(0).strip()
            imports.append(import_line)

        # Find Widget classes (high priority for Flutter)
        for match in widget_pattern.finditer(content):
            class_name = match.group(1)
            widget_type = match.group(2)

            # Find the actual line with 'class' by looking for it in the match
            match_text = match.group(0)
            class_pos_in_match = match_text.find("class")
            actual_class_pos = match.start() + class_pos_in_match
            start_line = content[:actual_class_pos].count("\n") + 1

            # Find end of class (simple heuristic)
            end_line = self._find_class_end(lines, start_line)

            class_content = self._get_line_range(lines, start_line, end_line)

            if class_content.strip():
                # Extract dartdoc using regex
                dartdoc = self._extract_dartdoc_regex(lines, start_line)

                chunk = self._create_chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="widget",
                    class_name=f"{class_name} ({widget_type})",
                    docstring=dartdoc,
                )
                chunk.imports = imports
                chunks.append(chunk)

                # Extract build method separately for Flutter widgets
                build_method = self._extract_build_method(class_content, start_line)
                if build_method:
                    chunks.append(build_method)

        # Find regular classes (not already captured as widgets)
        widget_class_names = {
            match.group(1) for match in widget_pattern.finditer(content)
        }
        for match in class_pattern.finditer(content):
            class_name = match.group(1)

            # Skip if already captured as widget
            if class_name in widget_class_names:
                continue

            # Find the actual line with 'class'
            match_text = match.group(0)
            class_pos_in_match = match_text.find("class")
            actual_class_pos = match.start() + class_pos_in_match
            start_line = content[:actual_class_pos].count("\n") + 1

            # Find end of class
            end_line = self._find_class_end(lines, start_line)

            class_content = self._get_line_range(lines, start_line, end_line)

            if class_content.strip():
                # Extract dartdoc
                dartdoc = self._extract_dartdoc_regex(lines, start_line)

                chunk = self._create_chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    class_name=class_name,
                    docstring=dartdoc,
                )
                chunk.imports = imports
                chunks.append(chunk)

        # Find mixins
        for match in mixin_pattern.finditer(content):
            mixin_name = match.group(1)

            match_text = match.group(0)
            mixin_pos_in_match = match_text.find("mixin")
            actual_mixin_pos = match.start() + mixin_pos_in_match
            start_line = content[:actual_mixin_pos].count("\n") + 1

            # Find end of mixin
            end_line = self._find_class_end(lines, start_line)

            mixin_content = self._get_line_range(lines, start_line, end_line)

            if mixin_content.strip():
                dartdoc = self._extract_dartdoc_regex(lines, start_line)

                chunk = self._create_chunk(
                    content=mixin_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="mixin",
                    class_name=mixin_name,
                    docstring=dartdoc,
                )
                chunk.imports = imports
                chunks.append(chunk)

        # Find functions (including async functions)
        for match in function_pattern.finditer(content):
            function_name = match.group(1)

            # Skip constructor-like patterns (same name as class)
            if function_name and function_name[0].isupper():
                # Check if it's a class name (constructor)
                if any(function_name == chunk.class_name for chunk in chunks):
                    continue

            # Find the actual line
            match_text = match.group(0)
            # Look for the function name position
            func_name_pos = match_text.rfind(function_name)
            if func_name_pos == -1:
                continue
            actual_func_pos = match.start() + func_name_pos
            start_line = content[:actual_func_pos].count("\n") + 1

            # Find end of function
            end_line = self._find_function_end(lines, start_line)

            func_content = self._get_line_range(lines, start_line, end_line)

            if func_content.strip():
                # Extract dartdoc
                dartdoc = self._extract_dartdoc_regex(lines, start_line)

                chunk = self._create_chunk(
                    content=func_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="function",
                    function_name=function_name,
                    docstring=dartdoc,
                )
                chunk.imports = imports
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

    def _extract_build_method(
        self, class_content: str, class_start_line: int
    ) -> CodeChunk | None:
        """Extract build() method from a Widget class."""
        # Look for Widget build(BuildContext context)
        build_pattern = re.compile(r"^\s*Widget\s+build\s*\(", re.MULTILINE)
        match = build_pattern.search(class_content)

        if not match:
            return None

        # Calculate line number within class
        lines_before = class_content[: match.start()].count("\n")
        class_start_line + lines_before

        # Find end of build method
        class_lines = class_content.splitlines(keepends=True)
        build_start_idx = lines_before

        # Simple heuristic: find matching braces
        self._find_method_end(class_lines, build_start_idx)

        return None  # Simplified - build method is already in class chunk

    def _find_function_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a function using brace matching."""
        if start_line > len(lines):
            return len(lines)

        start_idx = start_line - 1
        if start_idx >= len(lines):
            return len(lines)

        # For Dart, we need to count braces
        brace_count = 0
        found_opening_brace = False

        for i in range(start_idx, len(lines)):
            line = lines[i]

            for char in line:
                if char == "{":
                    brace_count += 1
                    found_opening_brace = True
                elif char == "}":
                    brace_count -= 1
                    if found_opening_brace and brace_count == 0:
                        return i + 1  # Return 1-based line number

            # Handle single-line functions (arrow functions)
            if "=>" in line and ";" in line and not found_opening_brace:
                return i + 1

        return len(lines)

    def _find_class_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a class using brace matching."""
        return self._find_function_end(lines, start_line)

    def _find_method_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a method using brace matching."""
        brace_count = 0
        found_opening_brace = False

        for i in range(start_idx, len(lines)):
            line = lines[i]

            for char in line:
                if char == "{":
                    brace_count += 1
                    found_opening_brace = True
                elif char == "}":
                    brace_count -= 1
                    if found_opening_brace and brace_count == 0:
                        return i + 1

        return len(lines)

    def _extract_dartdoc_regex(self, lines: list[str], start_line: int) -> str | None:
        """Extract dartdoc using regex patterns."""
        # Look for /// comments before the definition
        dartdoc_lines = []

        # Check a few lines before the start_line
        for i in range(max(0, start_line - 15), start_line - 1):
            if i >= len(lines):
                continue

            line = lines[i].strip()
            if line.startswith("///"):
                dartdoc_lines.append(line[3:].strip())
            elif line and not line.startswith("//") and dartdoc_lines:
                # If we hit non-comment code after finding dartdoc, stop
                break
            elif line and not line.startswith("//") and not dartdoc_lines:
                # Reset if we hit code before finding dartdoc
                dartdoc_lines = []

        if dartdoc_lines:
            return " ".join(dartdoc_lines)

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".dart"]
