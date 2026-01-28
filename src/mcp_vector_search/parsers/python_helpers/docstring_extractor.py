"""Docstring extraction for Python code."""

import re


class DocstringExtractor:
    """Handles docstring extraction from Python code using tree-sitter or regex."""

    @staticmethod
    def extract_from_node(node, lines: list[str]) -> str | None:
        """Extract docstring from a tree-sitter function or class node.

        Args:
            node: Tree-sitter AST node
            lines: Source code lines

        Returns:
            Cleaned docstring text or None
        """
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
                                docstring = DocstringExtractor._get_line_range(
                                    lines, start_line, end_line
                                )
                                # Clean up docstring (remove quotes)
                                return DocstringExtractor._clean_docstring(docstring)
        return None

    @staticmethod
    def extract_with_regex(content: str) -> str | None:
        """Extract docstring using regex patterns.

        Args:
            content: Code content to extract docstring from

        Returns:
            Extracted docstring or None
        """
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

    @staticmethod
    def _clean_docstring(docstring: str) -> str:
        """Clean up extracted docstring.

        Args:
            docstring: Raw docstring text

        Returns:
            Cleaned docstring
        """
        # Remove triple quotes and clean whitespace
        cleaned = re.sub(r'^["\']{{3}}|["\']{{3}}$', "", docstring.strip())
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned.strip())
        return cleaned.strip()

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
        # Convert to 0-based indexing
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        return "\n".join(lines[start_idx:end_idx])
