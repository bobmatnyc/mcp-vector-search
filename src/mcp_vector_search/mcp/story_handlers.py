"""Story generation handlers for MCP vector search server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent


class StoryHandlers:
    """Handlers for story generation MCP tools."""

    def __init__(self, project_root: Path):
        """Initialize story handlers.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    async def handle_story_generate(self, args: dict[str, Any]) -> CallToolResult:
        """Handle story_generate tool call.

        Args:
            args: Tool call arguments containing project_path, format, max_commits, etc.

        Returns:
            CallToolResult with story generation results or error
        """
        # Parse arguments
        project_path_str = args.get("project_path")
        format = args.get("format", "json")
        max_commits = args.get("max_commits", 200)
        max_issues = args.get("max_issues", 100)
        use_llm = args.get("use_llm", True)
        model = args.get("model")

        # Determine project root
        if project_path_str:
            project_root = Path(project_path_str).resolve()
        else:
            project_root = self.project_root

        if not project_root.exists():
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Project path does not exist: {project_root}"
                    )
                ],
                isError=True,
            )

        try:
            # Import story generator
            from ..story import StoryGenerator

            # Create generator
            generator = StoryGenerator(
                project_root=project_root,
                max_commits=max_commits,
                max_issues=max_issues,
                use_llm=use_llm,
                model=model,
            )

            # Generate story
            logger.info(f"Generating story for {project_root}")
            story = await generator.generate()

            # Render based on format
            if format == "json":
                # Return JSON directly
                story_json = story.model_dump_json(indent=2)
                return CallToolResult(
                    content=[TextContent(type="text", text=story_json)]
                )

            elif format in ("markdown", "html", "all"):
                # Render to files
                output_paths = generator.render(story, format=format)

                # Build response with file paths and summary
                response_lines = [
                    f"Story generated successfully for: {story.metadata.project_name}",
                    "",
                    "Summary:",
                    f"  • Commits analyzed: {story.metadata.total_commits}",
                    f"  • Contributors: {story.metadata.total_contributors}",
                    f"  • Files tracked: {story.metadata.total_files}",
                    f"  • Generation time: {story.metadata.generation_time_seconds:.1f}s",
                    "",
                ]

                if story.metadata.has_semantic_analysis:
                    response_lines.extend(
                        [
                            f"  • Semantic clusters: {len(story.analysis.clusters)}",
                            f"  • Tech stack items: {len(story.analysis.tech_stack)}",
                        ]
                    )

                if story.metadata.has_llm_narrative:
                    response_lines.append(
                        f"  • Narrative acts: {len(story.narrative.acts)}"
                    )

                response_lines.extend(
                    [
                        "",
                        "Files generated:",
                    ]
                )

                for path in output_paths:
                    size_kb = path.stat().st_size / 1024
                    response_lines.append(f"  • {path} ({size_kb:.1f} KB)")

                return CallToolResult(
                    content=[TextContent(type="text", text="\n".join(response_lines))]
                )

            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Invalid format: {format}. Must be 'json', 'markdown', 'html', or 'all'",
                        )
                    ],
                    isError=True,
                )

        except Exception as e:
            logger.error(f"Story generation failed: {e}", exc_info=True)
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Story generation failed: {str(e)}")
                ],
                isError=True,
            )
