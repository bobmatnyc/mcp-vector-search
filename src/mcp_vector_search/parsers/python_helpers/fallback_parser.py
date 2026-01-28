"""Regex-based fallback parser for Python when tree-sitter is unavailable."""

import re
from pathlib import Path

from ...core.models import CodeChunk
from .class_skeleton_generator import ClassSkeletonGenerator
from .docstring_extractor import DocstringExtractor


class RegexFallbackParser:
    """Fallback parsing using regex when Tree-sitter is not available."""

    def __init__(self, base_parser):
        """Initialize with reference to base parser.

        Args:
            base_parser: Reference to the PythonParser instance for shared utilities
        """
        self.base_parser = base_parser

    async def parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing using regex when Tree-sitter is not available.

        Args:
            content: Python source code
            file_path: Path to source file

        Returns:
            List of extracted code chunks
        """
        chunks = []
        lines = self.base_parser._split_into_lines(content)

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
            match_text = match.group(0)
            def_pos_in_match = match_text.find("def")
            actual_def_pos = match.start() + def_pos_in_match
            start_line = content[:actual_def_pos].count("\n") + 1

            # Find end of function (simple heuristic)
            end_line = self._find_function_end(lines, start_line)

            func_content = self.base_parser._get_line_range(lines, start_line, end_line)

            if func_content.strip():
                # Extract docstring using regex
                docstring = DocstringExtractor.extract_with_regex(func_content)

                chunk = self.base_parser._create_chunk(
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
            match_text = match.group(0)
            class_pos_in_match = match_text.find("class")
            actual_class_pos = match.start() + class_pos_in_match
            start_line = content[:actual_class_pos].count("\n") + 1

            # Find end of class (simple heuristic)
            end_line = self._find_class_end(lines, start_line)

            class_content = self.base_parser._get_line_range(
                lines, start_line, end_line
            )

            if class_content.strip():
                # Extract class skeleton (method signatures only)
                skeleton_content = ClassSkeletonGenerator.generate_with_regex(
                    class_content, start_line, lines
                )

                # Extract class docstring
                docstring = DocstringExtractor.extract_with_regex(skeleton_content)

                chunk = self.base_parser._create_chunk(
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
                self.base_parser._create_chunk(
                    content=content,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                )
            )

        return chunks

    def _find_function_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a function using indentation.

        Args:
            lines: Source code lines
            start_line: Starting line number (1-based)

        Returns:
            End line number (1-based)
        """
        if start_line > len(lines):
            return len(lines)

        # Get initial indentation of the def line
        start_idx = start_line - 1
        if start_idx >= len(lines):
            return len(lines)

        def_line = lines[start_idx]
        def_indent = len(def_line) - len(def_line.lstrip())

        # Find end by looking for line with indentation <= def indentation
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= def_indent:
                    return i

        # If we reach here, the function goes to the end of the file
        return len(lines)

    def _find_class_end(self, lines: list[str], start_line: int) -> int:
        """Find the end line of a class using indentation.

        Args:
            lines: Source code lines
            start_line: Starting line number (1-based)

        Returns:
            End line number (1-based)
        """
        return self._find_function_end(lines, start_line)
