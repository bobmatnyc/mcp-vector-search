"""MCP handlers for wiki generation functionality."""

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..core.chunks_backend import ChunksBackend
from ..core.llm_client import LLMClient
from ..core.project import ProjectManager
from ..core.wiki import WikiGenerator


class WikiHandlers:
    """MCP handlers for wiki generation operations."""

    def __init__(self, project_root: Path):
        """Initialize wiki handlers.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    async def handle_wiki_generate(self, args: dict[str, Any]) -> CallToolResult:
        """Handle wiki_generate tool call.

        Args:
            args: Tool arguments containing:
                - force (bool): Force regeneration
                - ttl_hours (int): Cache TTL in hours
                - no_llm (bool): Skip LLM semantic grouping
                - format (str): Output format ('json' or 'summary')

        Returns:
            CallToolResult with wiki ontology
        """
        try:
            force = args.get("force", False)
            ttl_hours = args.get("ttl_hours", 24)
            no_llm = args.get("no_llm", False)
            output_format = args.get("format", "summary")

            # Load project config
            project_manager = ProjectManager(self.project_root)
            config = project_manager.load_config()

            # Initialize chunks backend
            chunks_backend = ChunksBackend(config.index_path)
            await chunks_backend.initialize()

            # Initialize LLM client (if not skipping)
            llm_client = None
            if not no_llm:
                try:
                    llm_client = LLMClient()
                except ValueError as e:
                    logger.warning(f"LLM client initialization failed: {e}")
                    no_llm = True  # Fallback to flat ontology

            # Update cache TTL
            ttl_hours = int(os.getenv("MCP_WIKI_CACHE_TTL_HOURS", str(ttl_hours)))

            # Initialize wiki generator
            generator = WikiGenerator(
                project_root=self.project_root,
                chunks_backend=chunks_backend,
                llm_client=llm_client,
            )
            generator.cache.ttl_hours = ttl_hours

            # Generate ontology
            logger.info("Generating wiki ontology via MCP...")
            ontology = await generator.generate(force=force, use_llm=not no_llm)

            # Cleanup
            await chunks_backend.close()

            # Format response based on output_format
            if output_format == "json":
                # Return full JSON ontology
                content = json.dumps(ontology.to_dict(), indent=2)
            else:
                # Return summary
                summary_lines = [
                    "# Codebase Wiki Ontology\n",
                    f"**Total Concepts:** {ontology.total_concepts}",
                    f"**Total Chunks:** {ontology.total_chunks}",
                    f"**Root Categories:** {len(ontology.root_categories)}",
                    f"**Generated:** {ontology.generated_at}",
                    f"**Cache TTL:** {ontology.ttl_hours} hours\n",
                    "## Root Categories\n",
                ]

                for cat_id in ontology.root_categories[:10]:  # Limit to top 10
                    concept = ontology.concepts[cat_id]
                    summary_lines.append(
                        f"### {concept.name} ({len(concept.children)} concepts, "
                        f"{concept.frequency} references)"
                    )
                    if concept.description:
                        summary_lines.append(f"{concept.description}\n")

                content = "\n".join(summary_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=content)],
                isError=False,
            )

        except Exception as e:
            logger.error(f"Wiki generation failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Wiki generation failed: {str(e)}",
                    )
                ],
                isError=True,
            )
