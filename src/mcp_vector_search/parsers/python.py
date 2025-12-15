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

        # Enhancement 1: Calculate complexity
        complexity = self._calculate_complexity(node, "python")

        # Enhancement 4: Extract decorators
        decorators = self._extract_decorators(node, lines)

        # Enhancement 4: Extract parameters
        parameters = self._extract_parameters(node)

        # Enhancement 4: Extract return type
        return_type = self._extract_return_type(node)

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
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            chunk_depth=2 if class_name else 1,
        )
        chunks.append(chunk)

        return chunks

    def _extract_class_skeleton(self, node, lines: list[str], file_path: Path) -> str:
        """Extract class skeleton with method signatures only (no method bodies).

        This reduces redundancy since method chunks contain full implementations.
        """
        skeleton_lines = []

        # Find the class body block
        class_block = None
        for child in node.children:
            if child.type == "block":
                class_block = child
                break

        if not class_block:
            # No block found, return full class content
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            return self._get_line_range(lines, start_line, end_line)

        # Add class definition line(s) and decorators (everything before the block)
        # but NOT the block's opening line (to avoid duplicating the docstring)
        class_start = node.start_point[0]
        block_start = class_block.start_point[0]

        for line_idx in range(class_start, block_start):
            if line_idx < len(lines):
                line = lines[line_idx].rstrip()
                # Add the line, ensuring we get the colon on the class definition
                skeleton_lines.append(line)

        # Add the colon line if it wasn't already added
        if skeleton_lines and not skeleton_lines[-1].rstrip().endswith(":"):
            # The class definition might span multiple lines
            # Find and add up to the colon
            for line_idx in range(class_start, block_start + 1):
                if line_idx < len(lines):
                    line = lines[line_idx].rstrip()
                    if line not in [s.rstrip() for s in skeleton_lines]:
                        skeleton_lines.append(line)
                    if line.endswith(":"):
                        break

        # Process class body - add class variables and method signatures
        indent = "    "  # Standard Python indent
        docstring_added = False

        for stmt in class_block.children:
            if stmt.type == "expression_statement":
                # Check if it's a docstring (first statement after class def)
                for expr_child in stmt.children:
                    if expr_child.type == "string":
                        # Add docstring only once
                        if not docstring_added:
                            doc_start = stmt.start_point[0]
                            doc_end = stmt.end_point[0]
                            for line_idx in range(doc_start, doc_end + 1):
                                if line_idx < len(lines):
                                    skeleton_lines.append(lines[line_idx].rstrip())
                            docstring_added = True
                        break
                else:
                    # Not a docstring - could be a class variable assignment
                    # Add it to the skeleton
                    stmt_start = stmt.start_point[0]
                    stmt_end = stmt.end_point[0]
                    for line_idx in range(stmt_start, stmt_end + 1):
                        if line_idx < len(lines):
                            skeleton_lines.append(lines[line_idx].rstrip())

            elif stmt.type in ("assignment", "annotated_assignment"):
                # Class variable - add it
                stmt_start = stmt.start_point[0]
                stmt_end = stmt.end_point[0]
                for line_idx in range(stmt_start, stmt_end + 1):
                    if line_idx < len(lines):
                        skeleton_lines.append(lines[line_idx].rstrip())

            elif stmt.type == "function_definition":
                # Method - add only the signature (no body)
                _ = self._get_node_name(stmt)  # Not used, but validates method

                # Add decorators
                for deco_child in stmt.children:
                    if deco_child.type == "decorator":
                        deco_line = deco_child.start_point[0]
                        if deco_line < len(lines):
                            skeleton_lines.append(lines[deco_line].rstrip())

                # Add the def line (with parameters and return type)
                def_line_start = stmt.start_point[0]

                # Find where the actual body starts (after the colon)
                # We want everything up to and including the colon
                for child in stmt.children:
                    if child.type == "block":
                        # The block starts after the colon
                        # Get lines up to the colon
                        block_line = child.start_point[0]
                        for line_idx in range(def_line_start, block_line + 1):
                            if line_idx < len(lines):
                                line = lines[line_idx].rstrip()
                                skeleton_lines.append(line)
                                # Stop if we've added the colon line
                                if ":" in line:
                                    break

                        # Check if there's a docstring in the method
                        for block_child in child.children:
                            if block_child.type == "expression_statement":
                                for expr_child in block_child.children:
                                    if expr_child.type == "string":
                                        # Add method docstring
                                        doc_start = block_child.start_point[0]
                                        doc_end = block_child.end_point[0]
                                        for line_idx in range(doc_start, doc_end + 1):
                                            if line_idx < len(lines):
                                                skeleton_lines.append(
                                                    lines[line_idx].rstrip()
                                                )
                                        break
                                break

                        # Add placeholder for method body
                        skeleton_lines.append(f"{indent}{indent}...")
                        skeleton_lines.append("")  # Blank line between methods
                        break

        return "\n".join(skeleton_lines)

    def _extract_class(
        self, node, lines: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Extract class definition as a chunk (skeleton only, no method bodies)."""
        chunks = []

        class_name = self._get_node_name(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Get class skeleton (without method bodies)
        content = self._extract_class_skeleton(node, lines, file_path)

        # Extract docstring if present
        docstring = self._extract_docstring(node, lines)

        # Enhancement 1: Calculate complexity (for the entire class)
        complexity = self._calculate_complexity(node, "python")

        # Enhancement 4: Extract decorators
        decorators = self._extract_decorators(node, lines)

        chunk = self._create_chunk(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            class_name=class_name,
            docstring=docstring,
            complexity_score=complexity,
            decorators=decorators,
            chunk_depth=1,
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
                # Extract class skeleton (method signatures only)
                skeleton_content = self._extract_class_skeleton_regex(
                    class_content, start_line, lines
                )

                # Extract class docstring
                docstring = self._extract_docstring_regex(skeleton_content)

                chunk = self._create_chunk(
                    content=skeleton_content,
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

    def _extract_class_skeleton_regex(
        self, class_content: str, start_line: int, all_lines: list[str]
    ) -> str:
        """Extract class skeleton using regex (fallback when tree-sitter unavailable).

        Returns class with method signatures only, no method bodies.
        """
        lines = class_content.splitlines()
        skeleton_lines = []
        i = 0

        # Get class definition line(s)
        while i < len(lines):
            line = lines[i]
            skeleton_lines.append(line)
            # Stop at the colon that ends the class definition
            if line.rstrip().endswith(":"):
                i += 1
                break
            i += 1

        # Track indentation level
        class_indent = None
        if skeleton_lines:
            first_line = skeleton_lines[0]
            class_indent = len(first_line) - len(first_line.lstrip())

        # Process class body
        in_method = False
        method_indent = None

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                # Keep blank lines if not in a method body
                if not in_method:
                    skeleton_lines.append(line)
                i += 1
                continue

            # Calculate indentation
            current_indent = len(line) - len(line.lstrip())

            # Check if we're back at class level or beyond
            if class_indent is not None and current_indent <= class_indent and stripped:
                # End of class
                break

            # Check if this is a method definition
            if re.match(r"^\s*(async\s+)?def\s+\w+", line):
                in_method = True
                method_indent = current_indent

                # Add any decorators before this method
                # (look backwards for @ lines)
                j = i - 1
                decorator_lines = []
                while j >= 0:
                    prev_line = lines[j]
                    if prev_line.strip().startswith("@"):
                        decorator_lines.insert(0, prev_line)
                        j -= 1
                    elif prev_line.strip():
                        break
                    else:
                        j -= 1

                # Remove decorators if we already added them
                if decorator_lines:
                    # Check if they're not already in skeleton_lines
                    for dec in decorator_lines:
                        if dec not in skeleton_lines[-len(decorator_lines) :]:
                            skeleton_lines.append(dec)

                # Add method signature line
                skeleton_lines.append(line)

                # Check if there's a docstring on next lines
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()

                    if not next_stripped:
                        j += 1
                        continue

                    # Check for docstring
                    if next_stripped.startswith('"""') or next_stripped.startswith(
                        "'''"
                    ):
                        quote_type = next_stripped[:3]
                        # Add docstring
                        skeleton_lines.append(next_line)
                        if not (
                            next_stripped.endswith(quote_type)
                            and len(next_stripped) > 6
                        ):
                            # Multi-line docstring
                            j += 1
                            while j < len(lines):
                                doc_line = lines[j]
                                skeleton_lines.append(doc_line)
                                if doc_line.strip().endswith(quote_type):
                                    j += 1
                                    break
                                j += 1
                        else:
                            j += 1
                        break
                    else:
                        break

                # Add placeholder for method body
                if method_indent is not None:
                    skeleton_lines.append(" " * (method_indent + 4) + "...")
                else:
                    skeleton_lines.append("        ...")

                i += 1
                continue

            # Check if we're still in a method
            if in_method:
                if method_indent is not None and current_indent <= method_indent:
                    # End of method
                    in_method = False
                    # Don't skip this line, process it in next iteration
                    continue
                else:
                    # Inside method body - skip it
                    i += 1
                    continue

            # Class-level statement (not a method)
            # This could be a class variable, docstring, etc.
            if current_indent > (class_indent or 0):
                skeleton_lines.append(line)

            i += 1

        return "\n".join(skeleton_lines)

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

    def _extract_decorators(self, node, lines: list[str]) -> list[str]:
        """Extract decorator names from function/class node."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                # Get decorator text (includes @ symbol)
                dec_text = self._get_node_text(child).strip()
                decorators.append(dec_text)
        return decorators

    def _extract_parameters(self, node) -> list[dict]:
        """Extract function parameters with type annotations."""
        parameters = []
        for child in node.children:
            if child.type == "parameters":
                for param_node in child.children:
                    if param_node.type in (
                        "identifier",
                        "typed_parameter",
                        "default_parameter",
                    ):
                        param_info = {"name": None, "type": None, "default": None}

                        # Extract parameter name
                        if param_node.type == "identifier":
                            param_info["name"] = self._get_node_text(param_node)
                        else:
                            # For typed or default parameters, find the identifier
                            for subchild in param_node.children:
                                if subchild.type == "identifier":
                                    param_info["name"] = self._get_node_text(subchild)
                                elif subchild.type == "type":
                                    param_info["type"] = self._get_node_text(subchild)
                                elif "default" in subchild.type:
                                    param_info["default"] = self._get_node_text(
                                        subchild
                                    )

                        if param_info["name"] and param_info["name"] not in (
                            "self",
                            "cls",
                            "(",
                            ")",
                            ",",
                        ):
                            parameters.append(param_info)
        return parameters

    def _extract_return_type(self, node) -> str | None:
        """Extract return type annotation from function."""
        for child in node.children:
            if child.type == "type":
                return self._get_node_text(child)
        return None

    def _get_node_text(self, node) -> str:
        """Get text content of a node."""
        if hasattr(node, "text"):
            return node.text.decode("utf-8")
        return ""

    def get_supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        return [".py", ".pyw"]
