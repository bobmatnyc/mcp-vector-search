"""PHP parser using Tree-sitter for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class PHPParser(BaseParser):
    """PHP parser using Tree-sitter for AST-based code analysis."""

    def __init__(self) -> None:
        """Initialize PHP parser."""
        super().__init__("php")
        self._parser = None
        self._language = None
        self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize Tree-sitter parser for PHP."""
        try:
            # Try the tree-sitter-language-pack package (maintained alternative)
            from tree_sitter_language_pack import get_language, get_parser

            # Get the language and parser objects
            self._language = get_language("php")
            self._parser = get_parser("php")

            logger.debug(
                "PHP Tree-sitter parser initialized via tree-sitter-language-pack"
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
            "Using fallback regex-based parsing for PHP (Tree-sitter unavailable)"
        )

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a PHP file and extract code chunks."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse PHP content and extract code chunks."""
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

        def visit_node(node, current_class=None, current_namespace=None):
            """Recursively visit AST nodes."""
            node_type = node.type

            if node_type == "function_definition":
                chunks.extend(
                    self._extract_function(
                        node, lines, file_path, current_class, current_namespace
                    )
                )
            elif node_type == "class_declaration":
                class_chunks = self._extract_class(
                    node, lines, file_path, current_namespace
                )
                chunks.extend(class_chunks)

                # Visit class methods with class context
                class_name = self._get_node_name(node)
                for child in node.children:
                    visit_node(child, class_name, current_namespace)
            elif node_type == "interface_declaration":
                chunks.extend(
                    self._extract_interface(node, lines, file_path, current_namespace)
                )
            elif node_type == "trait_declaration":
                chunks.extend(
                    self._extract_trait(node, lines, file_path, current_namespace)
                )
            elif node_type == "method_declaration":
                chunks.extend(
                    self._extract_method(
                        node, lines, file_path, current_class, current_namespace
                    )
                )
            elif node_type == "namespace_definition":
                namespace_name = self._get_namespace_name(node)
                # Visit children with namespace context
                for child in node.children:
                    visit_node(child, current_class, namespace_name)
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
                    visit_node(child, current_class, current_namespace)

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
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        namespace: str | None = None,
    ) -> list[CodeChunk]:
        """Extract function definition as a chunk."""
        chunks = []

        function_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get function content
        content = node.text.decode()

        # Extract PHPDoc if present
        phpdoc = self._extract_phpdoc(node, lines)

        # Build fully qualified function name
        full_name = function_name
        if namespace and not class_name:
            full_name = f"{namespace}\\{function_name}"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            function_name=full_name,
            class_name=class_name,
            docstring=phpdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_method(
        self,
        node,
        lines: list[str],
        file_path: Path,
        class_name: str | None = None,
        namespace: str | None = None,
    ) -> list[CodeChunk]:
        """Extract method definition as a chunk."""
        chunks = []

        method_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get method content
        content = node.text.decode()

        # Extract PHPDoc if present
        phpdoc = self._extract_phpdoc(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            function_name=method_name,
            class_name=class_name,
            docstring=phpdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node, lines: list[str], file_path: Path, namespace: str | None = None
    ) -> list[CodeChunk]:
        """Extract class definition as a chunk."""
        chunks = []

        class_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get class content
        content = node.text.decode()

        # Extract PHPDoc if present
        phpdoc = self._extract_phpdoc(node, lines)

        # Build fully qualified class name
        full_class_name = class_name
        if namespace:
            full_class_name = f"{namespace}\\{class_name}"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=full_class_name,
            docstring=phpdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_interface(
        self, node, lines: list[str], file_path: Path, namespace: str | None = None
    ) -> list[CodeChunk]:
        """Extract interface definition as a chunk."""
        chunks = []

        interface_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get interface content
        content = node.text.decode()

        # Extract PHPDoc if present
        phpdoc = self._extract_phpdoc(node, lines)

        # Build fully qualified interface name
        full_interface_name = interface_name
        if namespace:
            full_interface_name = f"{namespace}\\{interface_name}"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="interface",
            class_name=full_interface_name,
            docstring=phpdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_trait(
        self, node, lines: list[str], file_path: Path, namespace: str | None = None
    ) -> list[CodeChunk]:
        """Extract trait definition as a chunk."""
        chunks = []

        trait_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get trait content
        content = node.text.decode()

        # Extract PHPDoc if present
        phpdoc = self._extract_phpdoc(node, lines)

        # Build fully qualified trait name
        full_trait_name = trait_name
        if namespace:
            full_trait_name = f"{namespace}\\{trait_name}"

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="trait",
            class_name=full_trait_name,
            docstring=phpdoc,
        )
        chunks.append(chunk)

        return chunks

    def _extract_module_chunk(
        self, node, lines: list[str], file_path: Path
    ) -> CodeChunk | None:
        """Extract module-level code (use statements, requires, etc.)."""
        # Look for module-level statements (not inside functions/classes)
        module_lines = []

        for child in node.children:
            if child.type in ["namespace_use_declaration", "namespace_definition"]:
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
            if child.type in ["name", "identifier"]:
                return child.text.decode("utf-8")
        return None

    def _get_namespace_name(self, node) -> str | None:
        """Extract namespace name from namespace definition."""
        for child in node.children:
            if child.type == "namespace_name":
                return child.text.decode("utf-8")
        return None

    def _extract_phpdoc(self, node, lines: list[str]) -> str | None:
        """Extract PHPDoc from a function or class node."""
        # Look for comment node before the definition
        start_line = node.start_point[0]

        # Check a few lines before the node for PHPDoc comments
        phpdoc_lines = []
        in_phpdoc = False

        for i in range(max(0, start_line - 20), start_line):
            if i >= len(lines):
                continue

            line = lines[i].strip()

            if line.startswith("/**"):
                in_phpdoc = True
                # Extract content after /**
                content = line[3:].strip()
                if content and content != "*":
                    phpdoc_lines.append(content)
            elif in_phpdoc and line.startswith("*/"):
                in_phpdoc = False
                break
            elif in_phpdoc:
                # Remove leading * and whitespace
                content = line.lstrip("*").strip()
                if content:
                    phpdoc_lines.append(content)
            elif line and not line.startswith("//") and not in_phpdoc:
                # Reset if we hit non-comment code
                phpdoc_lines = []

        if phpdoc_lines:
            return " ".join(phpdoc_lines)

        return None

    async def _fallback_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing using regex when Tree-sitter is not available."""
        chunks = []
        lines = self._split_into_lines(content)

        # Enhanced regex patterns for PHP
        namespace_pattern = re.compile(r"^\s*namespace\s+([\w\\]+)", re.MULTILINE)
        class_pattern = re.compile(
            r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)", re.MULTILINE
        )
        interface_pattern = re.compile(r"^\s*interface\s+(\w+)", re.MULTILINE)
        trait_pattern = re.compile(r"^\s*trait\s+(\w+)", re.MULTILINE)
        function_pattern = re.compile(
            r"^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?function\s+(\w+)\s*\(",
            re.MULTILINE,
        )
        use_pattern = re.compile(r"^\s*use\s+([\w\\]+)", re.MULTILINE)

        # Extract namespace (there should be only one)
        current_namespace = None
        namespace_match = namespace_pattern.search(content)
        if namespace_match:
            current_namespace = namespace_match.group(1)

        # Extract use statements
        use_statements = []
        for match in use_pattern.finditer(content):
            use_line = match.group(0).strip()
            use_statements.append(use_line)

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

            if class_content.strip():
                # Extract PHPDoc using regex
                phpdoc = self._extract_phpdoc_regex(lines, start_line)

                # Build fully qualified class name
                full_class_name = class_name
                if current_namespace:
                    full_class_name = f"{current_namespace}\\{class_name}"

                chunk = self._create_chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    class_name=full_class_name,
                    docstring=phpdoc,
                )
                chunk.imports = use_statements
                chunks.append(chunk)

        # Find interfaces
        for match in interface_pattern.finditer(content):
            interface_name = match.group(1)

            match_text = match.group(0)
            interface_pos_in_match = match_text.find("interface")
            actual_interface_pos = match.start() + interface_pos_in_match
            start_line = content[:actual_interface_pos].count("\n") + 1

            # Find end of interface
            end_line = self._find_class_end(lines, start_line)

            interface_content = self._get_line_range(lines, start_line, end_line)

            if interface_content.strip():
                phpdoc = self._extract_phpdoc_regex(lines, start_line)

                full_interface_name = interface_name
                if current_namespace:
                    full_interface_name = f"{current_namespace}\\{interface_name}"

                chunk = self._create_chunk(
                    content=interface_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="interface",
                    class_name=full_interface_name,
                    docstring=phpdoc,
                )
                chunk.imports = use_statements
                chunks.append(chunk)

        # Find traits
        for match in trait_pattern.finditer(content):
            trait_name = match.group(1)

            match_text = match.group(0)
            trait_pos_in_match = match_text.find("trait")
            actual_trait_pos = match.start() + trait_pos_in_match
            start_line = content[:actual_trait_pos].count("\n") + 1

            # Find end of trait
            end_line = self._find_class_end(lines, start_line)

            trait_content = self._get_line_range(lines, start_line, end_line)

            if trait_content.strip():
                phpdoc = self._extract_phpdoc_regex(lines, start_line)

                full_trait_name = trait_name
                if current_namespace:
                    full_trait_name = f"{current_namespace}\\{trait_name}"

                chunk = self._create_chunk(
                    content=trait_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="trait",
                    class_name=full_trait_name,
                    docstring=phpdoc,
                )
                chunk.imports = use_statements
                chunks.append(chunk)

        # Find functions (excluding methods inside classes)
        class_regions = [(chunk.start_line, chunk.end_line) for chunk in chunks]

        for match in function_pattern.finditer(content):
            function_name = match.group(1)

            # Skip magic methods and constructors
            if function_name.startswith("__"):
                continue

            # Find the actual line
            match_text = match.group(0)
            func_pos_in_match = match_text.find("function")
            actual_func_pos = match.start() + func_pos_in_match
            start_line = content[:actual_func_pos].count("\n") + 1

            # Skip if this function is inside a class
            is_inside_class = any(
                start <= start_line <= end for start, end in class_regions
            )
            if is_inside_class:
                continue

            # Find end of function
            end_line = self._find_function_end(lines, start_line)

            func_content = self._get_line_range(lines, start_line, end_line)

            if func_content.strip():
                # Extract PHPDoc
                phpdoc = self._extract_phpdoc_regex(lines, start_line)

                # Build fully qualified function name
                full_function_name = function_name
                if current_namespace:
                    full_function_name = f"{current_namespace}\\{function_name}"

                chunk = self._create_chunk(
                    content=func_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="function",
                    function_name=full_function_name,
                    docstring=phpdoc,
                )
                chunk.imports = use_statements
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
        """Find the end line of a function using brace matching."""
        if start_line > len(lines):
            return len(lines)

        start_idx = start_line - 1
        if start_idx >= len(lines):
            return len(lines)

        # For PHP, we need to count braces
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

        return len(lines)

    def _find_class_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a class using brace matching."""
        return self._find_function_end(lines, start_line)

    def _extract_phpdoc_regex(self, lines: list[str], start_line: int) -> str | None:
        """Extract PHPDoc using regex patterns."""
        # Look for /** ... */ comments before the definition
        phpdoc_lines = []
        in_phpdoc = False

        # Check a few lines before the start_line
        for i in range(max(0, start_line - 20), start_line - 1):
            if i >= len(lines):
                continue

            line = lines[i].strip()

            if line.startswith("/**"):
                in_phpdoc = True
                # Extract content after /**
                content = line[3:].strip()
                if content and content not in ("*", "*/"):
                    phpdoc_lines.append(content)

                # Check for single-line PHPDoc
                if line.endswith("*/") and len(line) > 5:
                    # Single line PHPDoc
                    content = line[3:-2].strip()
                    if content and content != "*":
                        return content
                    in_phpdoc = False
            elif in_phpdoc and line.endswith("*/"):
                # End of multi-line PHPDoc
                content = line[:-2].lstrip("*").strip()
                if content:
                    phpdoc_lines.append(content)
                in_phpdoc = False
                break
            elif in_phpdoc:
                # Inside PHPDoc - remove leading * and whitespace
                content = line.lstrip("*").strip()
                if content:
                    phpdoc_lines.append(content)
            elif line and not line.startswith("//") and not in_phpdoc and phpdoc_lines:
                # If we hit non-comment code after finding PHPDoc, we're done
                break
            elif line and not line.startswith("//") and not in_phpdoc:
                # Reset if we hit code before finding PHPDoc
                phpdoc_lines = []

        if phpdoc_lines:
            return " ".join(phpdoc_lines)

        return None

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".php", ".phtml"]
