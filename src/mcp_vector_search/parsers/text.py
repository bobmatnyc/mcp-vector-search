"""Text file parser for MCP Vector Search."""

from pathlib import Path

from ..config.constants import TEXT_CHUNK_SIZE
from ..core.models import CodeChunk
from .base import BaseParser


class TextParser(BaseParser):
    """Parser for plain text and markdown files (.txt, .md, .markdown)."""

    def __init__(self) -> None:
        """Initialize text parser."""
        super().__init__("text")

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a text file and extract chunks.

        Args:
            file_path: Path to the text file

        Returns:
            List of text chunks
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception:
            # Return empty list if file can't be read
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse text content into semantic chunks.

        Uses paragraph-based chunking for better semantic coherence.
        Falls back to line-based chunking for non-paragraph text.

        Args:
            content: Text content to parse
            file_path: Path to the source file

        Returns:
            List of text chunks
        """
        if not content.strip():
            return []

        chunks = []
        lines = content.splitlines(keepends=True)

        # Try paragraph-based chunking first
        paragraphs = self._extract_paragraphs(content)

        if paragraphs:
            # Use paragraph-based chunking
            for para_info in paragraphs:
                chunk = self._create_chunk(
                    content=para_info["content"],
                    file_path=file_path,
                    start_line=para_info["start_line"],
                    end_line=para_info["end_line"],
                    chunk_type="text",
                )
                chunks.append(chunk)
        else:
            # Fall back to line-based chunking for non-paragraph text
            # Use smaller chunks for text files (30 lines instead of 50)
            chunk_size = TEXT_CHUNK_SIZE
            for i in range(0, len(lines), chunk_size):
                start_line = i + 1
                end_line = min(i + chunk_size, len(lines))

                chunk_content = "".join(lines[i:end_line])

                if chunk_content.strip():
                    chunk = self._create_chunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="text",
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_paragraphs(self, content: str) -> list[dict]:
        """Extract paragraphs from text content.

        A paragraph is defined as one or more non-empty lines
        separated by empty lines.

        Args:
            content: Text content

        Returns:
            List of paragraph info dictionaries
        """
        lines = content.splitlines(keepends=True)
        paragraphs = []
        current_para = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            if line.strip():
                if not current_para:
                    start_line = i
                current_para.append(line)
            else:
                if current_para:
                    # End of paragraph
                    para_content = "".join(current_para)
                    if len(para_content.strip()) > 20:  # Minimum paragraph size
                        paragraphs.append(
                            {
                                "content": para_content,
                                "start_line": start_line,
                                "end_line": i - 1,
                            }
                        )
                    current_para = []

        # Handle last paragraph if exists
        if current_para:
            para_content = "".join(current_para)
            if len(para_content.strip()) > 20:
                paragraphs.append(
                    {
                        "content": para_content,
                        "start_line": start_line,
                        "end_line": len(lines),
                    }
                )

        # If we have very few paragraphs, merge small ones
        if paragraphs:
            merged = self._merge_small_paragraphs(paragraphs)
            return merged

        return []

    def _merge_small_paragraphs(
        self, paragraphs: list[dict], target_size: int = 200
    ) -> list[dict]:
        """Merge small paragraphs to create more substantial chunks.

        Args:
            paragraphs: List of paragraph dictionaries
            target_size: Target size for merged paragraphs in characters

        Returns:
            List of merged paragraph dictionaries
        """
        merged = []
        current_merge = None

        for para in paragraphs:
            para_len = len(para["content"])

            if current_merge is None:
                current_merge = para.copy()
            elif len(current_merge["content"]) + para_len < target_size * 2:
                # Merge with current
                current_merge["content"] += "\n" + para["content"]
                current_merge["end_line"] = para["end_line"]
            else:
                # Start new merge
                if len(current_merge["content"].strip()) > 20:
                    merged.append(current_merge)
                current_merge = para.copy()

        # Add last merge
        if current_merge and len(current_merge["content"].strip()) > 20:
            merged.append(current_merge)

        return merged

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".txt", ".md", ".markdown"]
