"""Class skeleton generation for Python classes."""

import re


class ClassSkeletonGenerator:
    """Generates class skeletons with method signatures but no method bodies."""

    @staticmethod
    def generate_from_node(node, lines: list[str]) -> str:
        """Extract class skeleton from tree-sitter node with method signatures only.

        This reduces redundancy since method chunks contain full implementations.

        Args:
            node: Tree-sitter class definition node
            lines: Source code lines

        Returns:
            Class skeleton with method signatures
        """
        class_block = ClassSkeletonGenerator._find_class_block(node)

        if not class_block:
            # No block found, return full class content
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            return ClassSkeletonGenerator._get_line_range(lines, start_line, end_line)

        skeleton_lines = []
        ClassSkeletonGenerator._add_class_header(
            node, class_block, lines, skeleton_lines
        )
        ClassSkeletonGenerator._process_class_body(class_block, lines, skeleton_lines)

        return "\n".join(skeleton_lines)

    @staticmethod
    def _find_class_block(node):
        """Find the class body block node.

        Args:
            node: Tree-sitter class definition node

        Returns:
            Block node or None
        """
        for child in node.children:
            if child.type == "block":
                return child
        return None

    @staticmethod
    def _add_class_header(
        node, class_block, lines: list[str], skeleton_lines: list[str]
    ):
        """Add class definition and decorators to skeleton.

        Args:
            node: Tree-sitter class definition node
            class_block: The class body block node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        class_start = node.start_point[0]
        block_start = class_block.start_point[0]

        # Add class definition line(s) and decorators
        for line_idx in range(class_start, block_start):
            if line_idx < len(lines):
                line = lines[line_idx].rstrip()
                skeleton_lines.append(line)

        # Ensure the colon line is included
        if skeleton_lines and not skeleton_lines[-1].rstrip().endswith(":"):
            for line_idx in range(class_start, block_start + 1):
                if line_idx < len(lines):
                    line = lines[line_idx].rstrip()
                    if line not in [s.rstrip() for s in skeleton_lines]:
                        skeleton_lines.append(line)
                    if line.endswith(":"):
                        break

    @staticmethod
    def _process_class_body(class_block, lines: list[str], skeleton_lines: list[str]):
        """Process class body statements, adding variables and method signatures.

        Args:
            class_block: The class body block node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        indent = "    "  # Standard Python indent
        docstring_added = False

        for stmt in class_block.children:
            if stmt.type == "expression_statement":
                docstring_added = ClassSkeletonGenerator._handle_expression_statement(
                    stmt, lines, skeleton_lines, docstring_added
                )
            elif stmt.type in ("assignment", "annotated_assignment"):
                ClassSkeletonGenerator._add_class_variable(stmt, lines, skeleton_lines)
            elif stmt.type == "function_definition":
                ClassSkeletonGenerator._add_method_signature(
                    stmt, lines, skeleton_lines, indent
                )

    @staticmethod
    def _handle_expression_statement(
        stmt, lines: list[str], skeleton_lines: list[str], docstring_added: bool
    ) -> bool:
        """Handle expression statements (docstrings and class variables).

        Args:
            stmt: Expression statement node
            lines: Source code lines
            skeleton_lines: Output list to append to
            docstring_added: Whether docstring has already been added

        Returns:
            Updated docstring_added flag
        """
        for expr_child in stmt.children:
            if expr_child.type == "string":
                if not docstring_added:
                    ClassSkeletonGenerator._add_line_range_from_node(
                        stmt, lines, skeleton_lines
                    )
                    return True
                break
        else:
            # Not a docstring - could be a class variable assignment
            ClassSkeletonGenerator._add_line_range_from_node(
                stmt, lines, skeleton_lines
            )

        return docstring_added

    @staticmethod
    def _add_class_variable(stmt, lines: list[str], skeleton_lines: list[str]):
        """Add class variable assignment to skeleton.

        Args:
            stmt: Assignment node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        ClassSkeletonGenerator._add_line_range_from_node(stmt, lines, skeleton_lines)

    @staticmethod
    def _add_method_signature(
        stmt, lines: list[str], skeleton_lines: list[str], indent: str
    ):
        """Add method signature with decorators and docstring to skeleton.

        Args:
            stmt: Function definition node
            lines: Source code lines
            skeleton_lines: Output list to append to
            indent: Indentation string
        """
        # Add decorators
        ClassSkeletonGenerator._add_decorators(stmt, lines, skeleton_lines)

        # Add the def line (with parameters and return type)
        def_line_start = stmt.start_point[0]

        # Find where the actual body starts (after the colon)
        for child in stmt.children:
            if child.type == "block":
                block_line = child.start_point[0]
                ClassSkeletonGenerator._add_signature_lines(
                    def_line_start, block_line, lines, skeleton_lines
                )
                ClassSkeletonGenerator._add_method_docstring(
                    child, lines, skeleton_lines
                )

                # Add placeholder for method body
                skeleton_lines.append(f"{indent}{indent}...")
                skeleton_lines.append("")  # Blank line between methods
                break

    @staticmethod
    def _add_decorators(stmt, lines: list[str], skeleton_lines: list[str]):
        """Add method decorators to skeleton.

        Args:
            stmt: Function definition node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        for deco_child in stmt.children:
            if deco_child.type == "decorator":
                deco_line = deco_child.start_point[0]
                if deco_line < len(lines):
                    skeleton_lines.append(lines[deco_line].rstrip())

    @staticmethod
    def _add_signature_lines(
        def_line_start: int,
        block_line: int,
        lines: list[str],
        skeleton_lines: list[str],
    ):
        """Add method signature lines up to and including the colon.

        Args:
            def_line_start: Starting line of method definition
            block_line: Line where method block starts
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        for line_idx in range(def_line_start, block_line + 1):
            if line_idx < len(lines):
                line = lines[line_idx].rstrip()
                skeleton_lines.append(line)
                if ":" in line:
                    break

    @staticmethod
    def _add_method_docstring(block_child, lines: list[str], skeleton_lines: list[str]):
        """Add method docstring if present.

        Args:
            block_child: Method block node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        for block_stmt in block_child.children:
            if block_stmt.type == "expression_statement":
                for expr_child in block_stmt.children:
                    if expr_child.type == "string":
                        ClassSkeletonGenerator._add_line_range_from_node(
                            block_stmt, lines, skeleton_lines
                        )
                        return
                break

    @staticmethod
    def _add_line_range_from_node(node, lines: list[str], skeleton_lines: list[str]):
        """Add lines for a given node's range.

        Args:
            node: Tree-sitter node
            lines: Source code lines
            skeleton_lines: Output list to append to
        """
        start = node.start_point[0]
        end = node.end_point[0]
        for line_idx in range(start, end + 1):
            if line_idx < len(lines):
                skeleton_lines.append(lines[line_idx].rstrip())

    @staticmethod
    def generate_with_regex(
        class_content: str, start_line: int, all_lines: list[str]
    ) -> str:
        """Extract class skeleton using regex (fallback when tree-sitter unavailable).

        Returns class with method signatures only, no method bodies.

        Args:
            class_content: Full class content
            start_line: Starting line number (unused but kept for API consistency)
            all_lines: All source lines (unused but kept for API consistency)

        Returns:
            Class skeleton with method signatures
        """
        lines = class_content.splitlines()
        skeleton_lines = []

        # Get class definition line(s)
        i = ClassSkeletonGenerator._add_class_definition(lines, skeleton_lines)

        # Track indentation level
        class_indent = ClassSkeletonGenerator._get_class_indent(skeleton_lines)

        # Process class body
        ClassSkeletonGenerator._process_class_body_regex(
            lines, skeleton_lines, i, class_indent
        )

        return "\n".join(skeleton_lines)

    @staticmethod
    def _add_class_definition(lines: list[str], skeleton_lines: list[str]) -> int:
        """Add class definition lines to skeleton and return next index.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to

        Returns:
            Index of next line to process
        """
        i = 0
        while i < len(lines):
            line = lines[i]
            skeleton_lines.append(line)
            if line.rstrip().endswith(":"):
                i += 1
                break
            i += 1
        return i

    @staticmethod
    def _get_class_indent(skeleton_lines: list[str]) -> int | None:
        """Get the indentation level of the class.

        Args:
            skeleton_lines: Lines with class definition

        Returns:
            Indentation level or None
        """
        if skeleton_lines:
            first_line = skeleton_lines[0]
            return len(first_line) - len(first_line.lstrip())
        return None

    @staticmethod
    def _process_class_body_regex(
        lines: list[str], skeleton_lines: list[str], i: int, class_indent: int | None
    ):
        """Process class body using regex patterns.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to
            i: Current line index
            class_indent: Class indentation level
        """
        in_method = False
        method_indent = None

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                if not in_method:
                    skeleton_lines.append(line)
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())

            # Check if we're back at class level or beyond
            if class_indent is not None and current_indent <= class_indent and stripped:
                break

            # Check if this is a method definition
            if re.match(r"^\s*(async\s+)?def\s+\w+", line):
                in_method, method_indent = ClassSkeletonGenerator._handle_method_def(
                    lines, skeleton_lines, i, current_indent
                )
                i += 1
                continue

            # Check if we're still in a method
            if in_method:
                in_method = ClassSkeletonGenerator._check_method_end(
                    current_indent, method_indent
                )
                if not in_method:
                    continue
                i += 1
                continue

            # Class-level statement (not a method)
            if current_indent > (class_indent or 0):
                skeleton_lines.append(line)

            i += 1

    @staticmethod
    def _handle_method_def(
        lines: list[str], skeleton_lines: list[str], i: int, current_indent: int
    ) -> tuple[bool, int]:
        """Handle method definition, adding decorators and signature.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to
            i: Current line index
            current_indent: Current indentation level

        Returns:
            Tuple of (in_method flag, method_indent)
        """
        line = lines[i]

        # Add any decorators before this method
        ClassSkeletonGenerator._add_preceding_decorators(lines, skeleton_lines, i)

        # Add method signature line
        skeleton_lines.append(line)

        # Check if there's a docstring
        ClassSkeletonGenerator._add_regex_docstring(lines, skeleton_lines, i)

        # Add placeholder for method body
        skeleton_lines.append(" " * (current_indent + 4) + "...")

        return True, current_indent

    @staticmethod
    def _add_preceding_decorators(lines: list[str], skeleton_lines: list[str], i: int):
        """Add decorators that precede the method definition.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to
            i: Current method line index
        """
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

        # Add decorators if not already present
        if decorator_lines:
            for dec in decorator_lines:
                if dec not in skeleton_lines[-len(decorator_lines) :]:
                    skeleton_lines.append(dec)

    @staticmethod
    def _add_regex_docstring(lines: list[str], skeleton_lines: list[str], i: int):
        """Add method docstring if present after method definition.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to
            i: Current method line index
        """
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            next_stripped = next_line.strip()

            if not next_stripped:
                j += 1
                continue

            # Check for docstring
            if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                quote_type = next_stripped[:3]
                skeleton_lines.append(next_line)

                # Check if it's a multi-line docstring
                if not (next_stripped.endswith(quote_type) and len(next_stripped) > 6):
                    ClassSkeletonGenerator._add_multiline_docstring(
                        lines, skeleton_lines, j, quote_type
                    )
                break
            else:
                break

    @staticmethod
    def _add_multiline_docstring(
        lines: list[str], skeleton_lines: list[str], start_j: int, quote_type: str
    ):
        """Add remaining lines of multi-line docstring.

        Args:
            lines: Source code lines
            skeleton_lines: Output list to append to
            start_j: Starting index in lines
            quote_type: Quote type (triple quote string)
        """
        j = start_j + 1
        while j < len(lines):
            doc_line = lines[j]
            skeleton_lines.append(doc_line)
            if doc_line.strip().endswith(quote_type):
                break
            j += 1

    @staticmethod
    def _check_method_end(current_indent: int, method_indent: int | None) -> bool:
        """Check if we've exited the method body.

        Args:
            current_indent: Current line indentation
            method_indent: Method definition indentation

        Returns:
            True if still in method, False if exited
        """
        if method_indent is not None and current_indent <= method_indent:
            return False
        return True

    @staticmethod
    def _get_line_range(lines: list[str], start_line: int, end_line: int) -> str:
        """Extract a range of lines from content.

        Args:
            lines: List of lines
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)

        Returns:
            Content for the specified line range
        """
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return "\n".join(lines[start_idx:end_idx])
