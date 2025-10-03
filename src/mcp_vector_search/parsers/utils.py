"""Shared utilities for language parsers.

This module contains common functionality used across multiple parsers
to reduce code duplication and improve maintainability.
"""

from pathlib import Path
from re import Pattern

from ..config.constants import DEFAULT_CHUNK_SIZE
from ..core.models import CodeChunk


def split_into_lines(content: str) -> list[str]:
    """Split content into lines, handling different line endings.

    Args:
        content: Text content to split

    Returns:
        List of lines with line endings preserved
    """
    # Handle different line endings and preserve them
    return content.splitlines(keepends=True)


def get_line_range(lines: list[str], start_line: int, end_line: int) -> str:
    """Get content from a range of lines.

    Args:
        lines: List of lines
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)

    Returns:
        Joined content from the line range
    """
    # Convert to 0-indexed
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)

    return "".join(lines[start_idx:end_idx])


def find_block_end(lines: list[str], start_line: int, indent_char: str = " ") -> int:
    """Find the end of a code block based on indentation.

    This is a simple heuristic that looks for the next line with equal or
    lower indentation level than the starting line.

    Args:
        lines: List of lines
        start_line: Starting line number (1-indexed)
        indent_char: Character used for indentation (space or tab)

    Returns:
        End line number (1-indexed)
    """
    if start_line > len(lines):
        return len(lines)

    # Get indentation of starting line
    start_idx = start_line - 1
    start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

    # Find next line with same or lower indentation
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.strip():  # Skip empty lines
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= start_indent:
                return i  # Return 0-indexed position, will be used as end_line

    return len(lines)


def create_simple_chunks(
    content: str, file_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> list[CodeChunk]:
    """Create simple line-based chunks from content.

    This is a fallback chunking strategy when more sophisticated
    parsing is not available.

    Args:
        content: File content
        file_path: Path to source file
        chunk_size: Number of lines per chunk

    Returns:
        List of code chunks
    """
    lines = split_into_lines(content)
    chunks = []

    for i in range(0, len(lines), chunk_size):
        start_line = i + 1
        end_line = min(i + chunk_size, len(lines))

        chunk_content = get_line_range(lines, start_line, end_line)

        if chunk_content.strip():
            chunk = CodeChunk(
                content=chunk_content,
                start_line=start_line,
                end_line=end_line,
                file_path=str(file_path),
                chunk_type="block",
                metadata={"source": "simple_chunking"},
            )
            chunks.append(chunk)

    return chunks


def extract_docstring(lines: list[str], start_line: int) -> str | None:
    """Extract docstring/comment block starting from a given line.

    Supports Python docstrings (triple quotes), JavaDoc (/** */),
    and hash-based comments (# or //).

    Args:
        lines: List of lines
        start_line: Line number to start looking (1-indexed)

    Returns:
        Docstring content or None if not found
    """
    if start_line > len(lines):
        return None

    start_idx = start_line - 1

    # Check for Python-style docstring
    triple_double = '"""'
    triple_single = "'''"
    for quote in [triple_double, triple_single]:
        if quote in lines[start_idx]:
            # Multi-line docstring
            docstring_lines = []
            in_docstring = False

            for line in lines[start_idx:]:
                if quote in line:
                    if in_docstring:
                        # End of docstring
                        docstring_lines.append(line[: line.index(quote) + 3])
                        break
                    else:
                        # Start of docstring
                        in_docstring = True
                        docstring_lines.append(line)
                        if line.count(quote) >= 2:
                            # Single-line docstring
                            break
                elif in_docstring:
                    docstring_lines.append(line)

            if docstring_lines:
                return "".join(docstring_lines).strip()

    # Check for JavaDoc-style comment
    if start_idx > 0 and "/**" in lines[start_idx - 1]:
        comment_lines = []
        for i in range(start_idx - 1, -1, -1):
            comment_lines.insert(0, lines[i])
            if "/**" in lines[i]:
                break

        for i in range(start_idx, len(lines)):
            if "*/" in lines[i]:
                comment_lines.append(lines[i])
                break
            comment_lines.append(lines[i])

        return "".join(comment_lines).strip()

    # Check for hash/slash comments on previous lines
    comment_lines = []
    for i in range(start_idx - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith("#") or line.startswith("//"):
            comment_lines.insert(0, lines[i])
        elif line:
            break

    if comment_lines:
        return "".join(comment_lines).strip()

    return None


def extract_imports_with_pattern(
    content: str, pattern: Pattern[str], chunk_type: str = "import"
) -> list[str]:
    """Extract import/require/use statements using a regex pattern.

    Args:
        content: Source code content
        pattern: Compiled regex pattern to match imports
        chunk_type: Type of import (import, require, use, etc.)

    Returns:
        List of import statements
    """
    imports = []
    for match in pattern.finditer(content):
        import_line = match.group(0).strip()
        imports.append(import_line)
    return imports


def find_code_blocks_with_patterns(
    content: str, lines: list[str], patterns: dict[str, Pattern[str]], file_path: Path
) -> list[CodeChunk]:
    """Find code blocks (functions, classes, etc.) using regex patterns.

    This is a generic fallback parser that can be configured with different
    patterns for different languages.

    Args:
        content: Source code content
        lines: Pre-split lines
        patterns: Dictionary mapping block types to compiled regex patterns
        file_path: Path to source file

    Returns:
        List of code chunks
    """
    chunks = []

    for block_type, pattern in patterns.items():
        for match in pattern.finditer(content):
            # Extract the name from the first capturing group
            name = match.group(1) if match.groups() else "unknown"

            # Find line number
            match_pos = match.start()
            start_line = content[:match_pos].count("\n") + 1

            # Find end of block using indentation
            end_line = find_block_end(lines, start_line)

            # Get block content
            block_content = get_line_range(lines, start_line, end_line)

            if block_content.strip():
                # Extract docstring
                docstring = extract_docstring(lines, start_line + 1)

                chunk = CodeChunk(
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                    file_path=str(file_path),
                    chunk_type=block_type,
                    metadata={
                        "name": name,
                        "docstring": docstring,
                        "source": "regex_fallback",
                    },
                )
                chunks.append(chunk)

    return chunks
