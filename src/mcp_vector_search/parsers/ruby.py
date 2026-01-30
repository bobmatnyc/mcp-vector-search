"""Ruby parser using Tree-sitter for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class RubyParser(BaseParser):
    """Ruby parser using Tree-sitter for AST-based code analysis."""

    def __init__(self) -> None:
        """Initialize Ruby parser."""
        super().__init__("ruby")
        self._parser = None
        self._language = None
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for Ruby."""
        try:
            # Try the tree-sitter-language-pack package (maintained alternative)
            from tree_sitter_language_pack import get_language, get_parser

            # Get the language and parser objects
            self._language = get_language("ruby")
            self._parser = get_parser("ruby")

            logger.debug(
                "Ruby Tree-sitter parser initialized via tree-sitter-language-pack"
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
            "Using fallback regex-based parsing for Ruby (Tree-sitter unavailable)"
        )

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a Ruby file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse Ruby content and extract code chunks."""
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

        def visit_node(node, current_class=None, current_module=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            if node_type == "method":
                chunks.extend(
                    self._extract_method(
                        node, lines, file_path, current_class, current_module
                    )
                )
            elif node_type == "singleton_method":
                chunks.extend(
                    self._extract_class_method(
                        node, lines, file_path, current_class, current_module
                    )
                )
            elif node_type == "class":
                class_chunks = self._extract_class(
                    node, lines, file_path, current_module
                )
                chunks.extend(class_chunks)

                # Visit class methods with class context
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name, current_module)
            elif node_type == "module":
                module_chunks = self._extract_module(node, lines, file_path)
                chunks.extend(module_chunks)

                # Visit module contents
                module_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, current_class, module_name)
            elif node_type == "program":
                # Extract module-level code
                module_chunk = self._extract_module_level_chunk(node, lines, file_path)
                if module_chunk:
                    chunks.append(module_chunk)

                # Visit all children
                for child in node.children:
                    visit_node(child)
            else:
                # Visit children for other node types
                for child in node.children:
                    visit_node(child, current_class, current_module)

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
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        module_name: str | None = None,
    ) -> list[CodeChunk]:
        """Extract instance method definition as a chunk."""
        chunks = []

        method_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get method content
        content = node.text.decode()

        # Extract RDoc if present
        rdoc = self._extract_rdoc(node, lines)

        # Build full qualified name
        full_class_name = self._build_qualified_name(module_name, class_name)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=full_class_name,
            docstring=rdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_class_method(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        module_name: str | None = None,
    ) -> list[CodeChunk]:
        """Extract class method (singleton method) as a chunk."""
        chunks = []

        method_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get method content
        content = node.text.decode()

        # Extract RDoc if present
        rdoc = self._extract_rdoc(node, lines)

        # Build full qualified name
        full_class_name = self._build_qualified_name(module_name, class_name)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class_method",
            function_name=f"self.{method_name}",
            class_name=full_class_name,
            docstring=rdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node, lines: list[str], file_path: Path, module_name: str | None = None
    ) -> list[CodeChunk]:
        """Extract class definition as a chunk."""
        chunks = []

        class_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get class content
        content = node.text.decode()

        # Extract RDoc if present
        rdoc = self._extract_rdoc(node, lines)

        # Build full qualified name
        full_class_name = self._build_qualified_name(module_name, class_name)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=full_class_name,
            docstring=rdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_module(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract module definition as a chunk."""
        chunks = []

        module_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get module content
        content = node.text.decode()

        # Extract RDoc if present
        rdoc = self._extract_rdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="module",
            class_name=module_name,
            docstring=rdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_module_level_chunk(
        self, node, lines: list[str], file_path: Path
    ) -> CodeChunk | None:
        """Extract module-level code (requires, constants, etc.)."""
        # Look for module-level statements (not inside functions/classes)
        module_lines = []

        for child in node.children:
            if child.type in ["call"]:
                # Check if it's a require/require_relative
                child_text = child.text.decode("utf-8")
                if child_text.startswith("require") or "require_relative" in child_text:
                    require_content = child.text.decode()
                    module_lines.append(require_content.strip())

        if module_lines:
            content = "\n".join(module_lines)
            return self._create_chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(module_lines),
                chunk_type="requires",
            )

        return None

    def _get_node_name(self, node) -> str | None:
        """Extract name from a named node (method, class, module, etc.)."""
        for child in node.children:
            if child.type in [
                "identifier",
                "constant",
                "instance_variable",
                "class_variable",
            ]:
                return child.text.decode("utf-8")
        return None

    def _extract_rdoc(self, node, lines: list[str]) -> str | None:
        """Extract RDoc from a method or class node."""
        # Look for comment nodes before the definition
        start_line = node.start_point[0]

        # Check a few lines before the node for # comments
        rdoc_lines = []
        for i in range(max(0, start_line - 15), start_line):
            line = lines[i].strip()
            if line.startswith("#"):
                # Remove # and strip whitespace
                rdoc_lines.append(line[1:].strip())
            elif line and not rdoc_lines:
                # Reset if we hit non-comment code before finding rdoc
                continue
            elif line and rdoc_lines:
                # Stop if we hit non-comment code after finding rdoc
                break

        if rdoc_lines:
            return " ".join(rdoc_lines)

        # Check for =begin...=end block comments
        for i in range(max(0, start_line - 20), start_line):
            line = lines[i].strip()
            if line == "=begin":
                # Found start of block comment
                block_lines = []
                for j in range(i + 1, min(len(lines), start_line)):
                    block_line = lines[j].strip()
                    if block_line == "=end":
                        break
                    block_lines.append(block_line)
                if block_lines:
                    return " ".join(block_lines)

        return None

    def _build_qualified_name(
        self, module_name: str | None, class_name: str | None
    ) -> str | None:
        """Build a fully qualified name from module and class names."""
        if module_name and class_name:
            return f"{module_name}::{class_name}"
        return class_name or module_name

    async def _fallback_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing using regex when Tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Enhanced regex patterns for Ruby
        module_pattern = re.compile(r"^\s*module\s+(\w+(?:::\w+)*)", re.MULTILINE)
        class_pattern = re.compile(
            r"^\s*class\s+(\w+)(?:\s+<\s+(\w+(?:::\w+)*))?", re.MULTILINE
        )
        method_pattern = re.compile(r"^\s*def\s+(self\.)?(\w+[?!]?)", re.MULTILINE)
        attr_pattern = re.compile(
            r"^\s*attr_(accessor|reader|writer)\s+:(\w+)(?:\s*,\s*:(\w+))*",
            re.MULTILINE,
        )
        require_pattern = re.compile(
            r"^\s*(require|require_relative)\s+['\"](.+?)['\"]", re.MULTILINE
        )

        # Extract requires first
        requires = []
        for match in require_pattern.finditer(content):
            require_line = match.group(0).strip()
            requires.append(require_line)

        # Find modules
        modules = {}
        for match in module_pattern.finditer(content):
            module_name = match.group(1)
            match_text = match.group(0)
            module_pos_in_match = match_text.find("module")
            actual_module_pos = match.start() + module_pos_in_match
            start_line = content[:actual_module_pos].count("\n") + 1

            # Find end of module
            end_line = self._find_block_end(lines, start_line)

            module_content = self._get_line_range(lines, start_line, end_line)

            if module_content.strip():
                # Extract RDoc using regex
                rdoc = self._extract_rdoc_regex(lines, start_line)

                chunk = self._create_chunk(
                    content=module_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="module",
                    class_name=module_name,
                    docstring=rdoc,
                )
                chunks.append(chunk)
                modules[module_name] = (start_line, end_line)

        # Find classes
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            # superclass = match.group(2)  # Could be used for inheritance info

            match_text = match.group(0)
            class_pos_in_match = match_text.find("class")
            actual_class_pos = match.start() + class_pos_in_match
            start_line = content[:actual_class_pos].count("\n") + 1

            # Find end of class
            end_line = self._find_block_end(lines, start_line)

            class_content = self._get_line_range(lines, start_line, end_line)

            if class_content.strip():
                # Extract RDoc
                rdoc = self._extract_rdoc_regex(lines, start_line)

                # Determine if class is inside a module
                module_name = self._find_containing_module(start_line, modules)
                full_class_name = self._build_qualified_name(module_name, class_name)

                chunk = self._create_chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    class_name=full_class_name,
                    docstring=rdoc,
                )
                chunks.append(chunk)

        # Find methods
        classes_and_modules = {}
        for chunk in chunks:
            if chunk.class_name:
                classes_and_modules[chunk.class_name] = (
                    chunk.start_line,
                    chunk.end_line,
                )

        for match in method_pattern.finditer(content):
            is_class_method = match.group(1) is not None
            method_name = match.group(2)

            match_text = match.group(0)
            def_pos_in_match = match_text.find("def")
            actual_def_pos = match.start() + def_pos_in_match
            start_line = content[:actual_def_pos].count("\n") + 1

            # Find end of method
            end_line = self._find_method_end(lines, start_line)

            method_content = self._get_line_range(lines, start_line, end_line)

            if method_content.strip():
                # Extract RDoc
                rdoc = self._extract_rdoc_regex(lines, start_line)

                # Find containing class/module
                containing_class = self._find_containing_class(
                    start_line, classes_and_modules
                )

                # Format method name
                if is_class_method:
                    method_name = f"self.{method_name}"

                chunk = self._create_chunk(
                    content=method_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class_method" if is_class_method else "method",
                    function_name=method_name,
                    class_name=containing_class,
                    docstring=rdoc,
                )
                chunks.append(chunk)

        # Find attr_accessor/reader/writer
        for match in attr_pattern.finditer(content):
            attr_type = match.group(1)
            attr_name = match.group(2)

            match_text = match.group(0)
            start_line = content[: match.start()].count("\n") + 1
            end_line = start_line

            attr_content = match.group(0)

            # Find containing class/module
            containing_class = self._find_containing_class(
                start_line, classes_and_modules
            )

            chunk = self._create_chunk(
                content=attr_content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type="attribute",
                function_name=f"attr_{attr_type} :{attr_name}",
                class_name=containing_class,
            )
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

    def _find_block_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a block (module/class) using 'end' keyword matching."""
        if start_line > len(lines):
            return len(lines)

        start_idx = start_line - 1
        if start_idx >= len(lines):
            return len(lines)

        # Count nested blocks
        block_count = 0
        keywords_start = [
            "module",
            "class",
            "def",
            "do",
            "begin",
            "case",
            "if",
            "unless",
            "while",
            "until",
            "for",
        ]

        for i in range(start_idx, len(lines)):
            line = lines[i].strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Check for block-starting keywords
            for keyword in keywords_start:
                # Use word boundaries to avoid matching substrings
                if re.search(rf"\b{keyword}\b", line):
                    block_count += 1
                    break

            # Check for 'end' keyword
            if re.search(r"\bend\b", line):
                block_count -= 1
                if block_count == 0:
                    return i + 1  # Return 1-based line number

        return len(lines)

    def _find_method_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a method using 'end' keyword matching."""
        return self._find_block_end(lines, start_line)

    def _find_containing_module(
        self, line_number: int, modules: dict[str, tuple[int, int]]
    ) -> str | None:
        """Find the module containing a given line number."""
        for module_name, (start, end) in modules.items():
            if start < line_number < end:
                return module_name
        return None

    def _find_containing_class(
        self, line_number: int, classes_and_modules: dict[str, tuple[int, int]]
    ) -> str | None:
        """Find the class/module containing a given line number."""
        # Find the most specific (innermost) containing class
        containing = None
        smallest_range = float("inf")

        for name, (start, end) in classes_and_modules.items():
            if start < line_number < end:
                range_size = end - start
                if range_size < smallest_range:
                    smallest_range = range_size
                    containing = name

        return containing

    def _extract_rdoc_regex(self, lines: list[str], start_line: int) -> str | None:
        """Extract RDoc using regex patterns."""
        # Look for # comments before the definition
        rdoc_lines = []

        # Check lines before the start_line
        for i in range(max(0, start_line - 15), start_line - 1):
            if i >= len(lines):
                continue

            line = lines[i].strip()
            if line.startswith("#"):
                rdoc_lines.append(line[1:].strip())
            elif line and rdoc_lines:
                # If we hit non-comment code after finding rdoc, stop
                break
            elif line and not rdoc_lines:
                # Reset if we hit code before finding rdoc
                rdoc_lines = []

        if rdoc_lines:
            return " ".join(rdoc_lines)

        # Check for =begin...=end block comments
        for i in range(max(0, start_line - 20), start_line - 1):
            if i >= len(lines):
                continue

            line = lines[i].strip()
            if line == "=begin":
                # Found start of block comment
                block_lines = []
                for j in range(i + 1, min(len(lines), start_line - 1)):
                    block_line = lines[j].strip()
                    if block_line == "=end":
                        break
                    block_lines.append(block_line)
                if block_lines:
                    return " ".join(block_lines)

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".rb", ".rake", ".gemspec"]
