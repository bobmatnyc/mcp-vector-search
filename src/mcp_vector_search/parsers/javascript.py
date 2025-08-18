"""JavaScript/TypeScript parser for MCP Vector Search."""

import re
from pathlib import Path

from loguru import logger

from ..core.models import CodeChunk
from .base import BaseParser


class JavaScriptParser(BaseParser):
    """JavaScript/TypeScript parser with fallback regex-based parsing."""

    def __init__(self, language: str = "javascript") -> None:
        """Initialize JavaScript parser."""
        super().__init__(language)

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

        return await self._regex_parse(content, file_path)

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
