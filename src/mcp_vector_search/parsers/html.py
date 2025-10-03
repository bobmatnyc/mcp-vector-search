"""HTML parser for MCP Vector Search."""

from html.parser import HTMLParser as BaseHTMLParser
from pathlib import Path

from ..core.models import CodeChunk
from .base import BaseParser


class HTMLContentParser(BaseHTMLParser):
    """HTML parser for extracting semantic content from HTML documents.

    Extracts meaningful content from semantic HTML tags while ignoring
    scripts, styles, and other non-content elements.
    """

    def __init__(self) -> None:
        """Initialize the HTML content parser."""
        super().__init__()
        self.sections: list[dict] = []
        self.current_section: dict | None = None
        self.current_tag: str | None = None
        self.current_attrs: dict = {}
        self.tag_stack: list[tuple[str, dict]] = []
        self.ignore_content = False
        self.line_number = 1

        # Semantic tags that define sections
        self.section_tags = {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "main",
            "aside",
            "header",
            "footer",
            "nav",
        }
        # Tags to ignore completely
        self.ignore_tags = {"script", "style", "noscript"}
        # Inline text tags
        self.inline_tags = {
            "span",
            "a",
            "strong",
            "em",
            "b",
            "i",
            "code",
            "pre",
            "small",
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening HTML tags.

        Args:
            tag: Tag name
            attrs: List of (name, value) tuples for tag attributes
        """
        tag = tag.lower()
        attr_dict = {k: v for k, v in attrs if v is not None}

        # Track tag stack
        self.tag_stack.append((tag, attr_dict))

        # Ignore script/style content
        if tag in self.ignore_tags:
            self.ignore_content = True
            return

        # Start new section for semantic tags
        if tag in self.section_tags:
            # Save current section if exists
            if self.current_section and self.current_section.get("content", "").strip():
                self._finalize_current_section()

            # Start new section
            tag_id = attr_dict.get("id", "")
            tag_class = attr_dict.get("class", "")

            # Create tag identifier
            tag_info = tag
            if tag_id:
                tag_info = f"{tag}#{tag_id}"
            elif tag_class:
                tag_info = f"{tag}.{tag_class.split()[0]}"

            self.current_section = {
                "tag": tag,
                "tag_info": tag_info,
                "content": "",
                "start_line": self.getpos()[0],
                "attrs": attr_dict,
            }

        # Handle paragraph tags
        if tag == "p":
            if not self.current_section:
                # Create implicit section for orphan paragraph
                self.current_section = {
                    "tag": "p",
                    "tag_info": "p",
                    "content": "",
                    "start_line": self.getpos()[0],
                    "attrs": {},
                }

        self.current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        """Handle closing HTML tags.

        Args:
            tag: Tag name
        """
        tag = tag.lower()

        # Pop from stack
        if self.tag_stack and self.tag_stack[-1][0] == tag:
            self.tag_stack.pop()

        # Stop ignoring content after script/style
        if tag in self.ignore_tags:
            self.ignore_content = False
            return

        # Finalize section for semantic tags
        if tag in self.section_tags or tag == "p":
            if self.current_section and self.current_section.get("content", "").strip():
                self.current_section["end_line"] = self.getpos()[0]
                self._finalize_current_section()

    def handle_data(self, data: str) -> None:
        """Handle text content between tags.

        Args:
            data: Text content
        """
        if self.ignore_content or not data.strip():
            return

        # Add content to current section
        if self.current_section is not None:
            # Add spacing between inline elements
            if self.current_section["content"] and not self.current_section[
                "content"
            ].endswith(" "):
                self.current_section["content"] += " "
            self.current_section["content"] += data.strip()

    def _finalize_current_section(self) -> None:
        """Finalize and save the current section."""
        if not self.current_section:
            return

        content = self.current_section["content"].strip()

        # Only save sections with meaningful content (min 20 chars)
        if len(content) >= 20:
            self.sections.append(
                {
                    "tag": self.current_section["tag"],
                    "tag_info": self.current_section["tag_info"],
                    "content": content,
                    "start_line": self.current_section["start_line"],
                    "end_line": self.current_section.get(
                        "end_line", self.current_section["start_line"]
                    ),
                    "attrs": self.current_section["attrs"],
                }
            )

        self.current_section = None

    def get_sections(self) -> list[dict]:
        """Get all extracted sections.

        Returns:
            List of section dictionaries
        """
        # Finalize any remaining section
        if self.current_section and self.current_section.get("content", "").strip():
            self._finalize_current_section()

        return self.sections


class HTMLParser(BaseParser):
    """Parser for HTML files (.html, .htm).

    Extracts semantic content from HTML documents by parsing
    heading hierarchy, sections, articles, and paragraphs.
    """

    def __init__(self) -> None:
        """Initialize HTML parser."""
        super().__init__("html")

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse an HTML file and extract semantic chunks.

        Args:
            file_path: Path to the HTML file

        Returns:
            List of semantic content chunks
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return await self.parse_content(content, file_path)
        except Exception:
            # Return empty list if file can't be read
            return []

    async def parse_content(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Parse HTML content into semantic chunks.

        Extracts content from semantic HTML tags (headings, sections, articles)
        while ignoring scripts, styles, and other non-content elements.

        Args:
            content: HTML content to parse
            file_path: Path to the source file

        Returns:
            List of semantic content chunks
        """
        if not content.strip():
            return []

        # Parse HTML content
        parser = HTMLContentParser()
        try:
            parser.feed(content)
        except Exception:
            # If parsing fails, fall back to simple text extraction
            return await self._fallback_parse(content, file_path)

        sections = parser.get_sections()

        if not sections:
            # No semantic sections found, try fallback
            return await self._fallback_parse(content, file_path)

        # Convert sections to chunks
        chunks = []

        # Merge small sections for better semantic coherence
        merged_sections = self._merge_small_sections(sections)

        for section in merged_sections:
            chunk_type = self._get_chunk_type(section["tag"])

            # Create descriptive metadata
            metadata = {
                "chunk_type": chunk_type,
                "function_name": section["tag_info"],  # Use tag_info as identifier
            }

            # Add class name for sections with specific IDs
            if section["attrs"].get("id"):
                metadata["class_name"] = section["attrs"]["id"]

            chunk = self._create_chunk(
                content=section["content"],
                file_path=file_path,
                start_line=section["start_line"],
                end_line=section["end_line"],
                **metadata,
            )
            chunks.append(chunk)

        return chunks

    def _get_chunk_type(self, tag: str) -> str:
        """Determine chunk type based on HTML tag.

        Args:
            tag: HTML tag name

        Returns:
            Chunk type string
        """
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return "heading"
        elif tag in {"section", "article", "main"}:
            return "section"
        elif tag == "p":
            return "paragraph"
        else:
            return "content"

    def _merge_small_sections(
        self, sections: list[dict], target_size: int = 200, max_size: int = 500
    ) -> list[dict]:
        """Merge small sections to create optimal chunk sizes.

        Args:
            sections: List of section dictionaries
            target_size: Target minimum size for chunks in characters
            max_size: Maximum size for chunks in characters

        Returns:
            List of merged section dictionaries
        """
        if not sections:
            return []

        merged = []
        current_merge: dict | None = None

        for section in sections:
            section_len = len(section["content"])

            # Always start new section for h1 tags
            if section["tag"] == "h1":
                if current_merge:
                    merged.append(current_merge)
                current_merge = section.copy()
                continue

            if current_merge is None:
                current_merge = section.copy()
            elif len(current_merge["content"]) + section_len < max_size:
                # Merge with current if under max size
                current_merge["content"] += "\n\n" + section["content"]
                current_merge["end_line"] = section["end_line"]

                # Update tag_info to reflect merged content
                if current_merge["tag_info"] != section["tag_info"]:
                    current_merge["tag_info"] = (
                        f"{current_merge['tag_info']}+{section['tag_info']}"
                    )
            else:
                # Start new section if max size would be exceeded
                if len(current_merge["content"]) >= target_size:
                    merged.append(current_merge)
                current_merge = section.copy()

        # Add last section
        if current_merge and len(current_merge["content"]) >= 20:
            merged.append(current_merge)

        return merged

    async def _fallback_parse(self, content: str, file_path: Path) -> list[CodeChunk]:
        """Fallback parsing for malformed HTML.

        Strips HTML tags and creates simple text chunks.

        Args:
            content: HTML content
            file_path: Path to source file

        Returns:
            List of text chunks
        """
        # Simple HTML tag removal
        import re

        # Remove script and style tags with content
        content = re.sub(
            r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE
        )
        content = re.sub(
            r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        # Split into chunks
        lines = content.split(".")
        chunks = []
        current_chunk = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            chunk_text = ". ".join(current_chunk)

            if len(chunk_text) >= 200 or i == len(lines):
                if chunk_text.strip():
                    chunk = self._create_chunk(
                        content=chunk_text.strip(),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=i,
                        chunk_type="text",
                    )
                    chunks.append(chunk)
                    current_chunk = []
                    start_line = i + 1

        return chunks

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".html", ".htm"]
