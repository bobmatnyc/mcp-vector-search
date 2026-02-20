"""Text file parser for MCP Vector Search."""

import re
from pathlib import Path

import yaml

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
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception:
            # Return empty list if file can't be read
            return []

    def _extract_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter from markdown content.

        Args:
            content: Markdown content

        Returns:
            Dictionary of frontmatter data
        """
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            try:
                return yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                return {}
        return {}

    def _extract_tags_from_frontmatter(self, frontmatter: dict) -> list[str]:
        """Extract tags from frontmatter fields.

        Looks for tags, categories, keywords, and labels fields.

        Args:
            frontmatter: Dictionary of frontmatter data

        Returns:
            List of deduplicated tags
        """
        tags = []
        for field in ["tags", "categories", "keywords", "labels"]:
            val = frontmatter.get(field, [])
            if isinstance(val, str):
                # Handle comma-separated strings
                tags.extend([t.strip() for t in val.split(",") if t.strip()])
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, list):
                        # Handle nested lists
                        tags.extend(str(t) for t in item if t)
                    else:
                        # Handle individual items
                        if item:
                            tags.append(str(item))
        # Deduplicate while preserving order
        return list(dict.fromkeys(tags))

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

        # Extract frontmatter tags for markdown files
        tags = []
        if str(file_path).endswith((".md", ".markdown")):
            frontmatter = self._extract_frontmatter(content)
            tags = self._extract_tags_from_frontmatter(frontmatter)

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
                chunk.tags = tags
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
                    chunk.tags = tags
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

    def parse_file_sync(self, file_path: Path) -> list[CodeChunk]:
        """Parse file synchronously (optimized for multiprocessing workers)."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Text parsing is already synchronous, just call the internal method
            if not content.strip():
                return []

            # Extract frontmatter tags for markdown files
            tags = []
            if str(file_path).endswith((".md", ".markdown")):
                frontmatter = self._extract_frontmatter(content)
                tags = self._extract_tags_from_frontmatter(frontmatter)

            # Try paragraph-based chunking first
            paragraphs = content.split("\n\n")

            # Filter out empty paragraphs
            valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]

            if len(valid_paragraphs) >= 2:
                # Use paragraph-based chunking
                chunks = self._chunk_by_paragraphs_sync(valid_paragraphs, file_path)
            else:
                # Fallback to line-based chunking
                chunks = self._chunk_by_lines_sync(content, file_path)

            # Set tags on all chunks
            for chunk in chunks:
                chunk.tags = tags

            return chunks

        except Exception:
            return []

    def _chunk_by_paragraphs_sync(
        self, paragraphs: list[str], file_path: Path
    ) -> list[CodeChunk]:
        """Chunk by paragraphs (synchronous)."""
        chunks = []
        current_chunk_text = []
        current_chunk_size = 0
        start_line = 1

        for para in paragraphs:
            para_size = len(para)

            # If paragraph alone exceeds max size, chunk it separately
            if para_size > TEXT_CHUNK_SIZE * 2:
                # Save current chunk if exists
                if current_chunk_text:
                    chunks.append(
                        self._create_chunk(
                            content="\n\n".join(current_chunk_text),
                            file_path=file_path,
                            start_line=start_line,
                            end_line=start_line
                            + sum(text.count("\n") for text in current_chunk_text),
                            chunk_type="text",
                        )
                    )
                    current_chunk_text = []
                    current_chunk_size = 0

                # Chunk large paragraph by lines
                large_para_chunks = self._chunk_by_lines_sync(para, file_path)
                chunks.extend(large_para_chunks)
                start_line += para.count("\n") + 2  # +2 for paragraph breaks

            # If adding this paragraph exceeds size, save current chunk
            elif (
                current_chunk_size + para_size > TEXT_CHUNK_SIZE and current_chunk_text
            ):
                chunks.append(
                    self._create_chunk(
                        content="\n\n".join(current_chunk_text),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=start_line
                        + sum(text.count("\n") for text in current_chunk_text),
                        chunk_type="text",
                    )
                )
                current_chunk_text = [para]
                current_chunk_size = para_size
                start_line += sum(text.count("\n") for text in current_chunk_text) + 2

            # Add paragraph to current chunk
            else:
                current_chunk_text.append(para)
                current_chunk_size += para_size

        # Add remaining chunk
        if current_chunk_text:
            chunks.append(
                self._create_chunk(
                    content="\n\n".join(current_chunk_text),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line
                    + sum(text.count("\n") for text in current_chunk_text),
                    chunk_type="text",
                )
            )

        return chunks

    def _chunk_by_lines_sync(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback to line-based chunking (synchronous)."""
        lines = self._split_into_lines(content)
        chunks = []

        # Simple line-based chunking
        chunk_size = TEXT_CHUNK_SIZE  # Use TEXT_CHUNK_SIZE as lines per chunk

        for i in range(0, len(lines), chunk_size):
            start_line = i + 1
            end_line = min(i + chunk_size, len(lines))

            chunk_content = self._get_line_range(lines, start_line, end_line)

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

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".txt", ".md", ".markdown"]
