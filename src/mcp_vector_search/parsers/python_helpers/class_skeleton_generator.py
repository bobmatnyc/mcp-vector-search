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
            return ClassSkeletonGenerator._get_line_range(lines, start_line, end_line)

        # Add class definition line(s) and decorators (everything before the block)
        class_start = node.start_point[0]
        block_start = class_block.start_point[0]

        for line_idx in range(class_start, block_start):
            if line_idx < len(lines):
                line = lines[line_idx].rstrip()
                skeleton_lines.append(line)

        # Add the colon line if it wasn't already added
        if skeleton_lines and not skeleton_lines[-1].rstrip().endswith(":"):
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
                # Add decorators
                for deco_child in stmt.children:
                    if deco_child.type == "decorator":
                        deco_line = deco_child.start_point[0]
                        if deco_line < len(lines):
                            skeleton_lines.append(lines[deco_line].rstrip())

                # Add the def line (with parameters and return type)
                def_line_start = stmt.start_point[0]

                # Find where the actual body starts (after the colon)
                for child in stmt.children:
                    if child.type == "block":
                        block_line = child.start_point[0]
                        for line_idx in range(def_line_start, block_line + 1):
                            if line_idx < len(lines):
                                line = lines[line_idx].rstrip()
                                skeleton_lines.append(line)
                                if ":" in line:
                                    break

                        # Check if there's a docstring in the method
                        for block_child in child.children:
                            if block_child.type == "expression_statement":
                                for expr_child in block_child.children:
                                    if expr_child.type == "string":
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
        i = 0

        # Get class definition line(s)
        while i < len(lines):
            line = lines[i]
            skeleton_lines.append(line)
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
                in_method = True
                method_indent = current_indent

                # Add any decorators before this method (look backwards)
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
                    in_method = False
                    continue
                else:
                    # Inside method body - skip it
                    i += 1
                    continue

            # Class-level statement (not a method)
            if current_indent > (class_indent or 0):
                skeleton_lines.append(line)

            i += 1

        return "\n".join(skeleton_lines)

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
