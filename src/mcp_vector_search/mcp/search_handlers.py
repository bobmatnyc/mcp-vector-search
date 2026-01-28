"""Search operation handlers for MCP vector search server."""

from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent

from ..core.search import SemanticSearchEngine


class SearchHandlers:
    """Handlers for search-related MCP tool operations."""

    def __init__(self, search_engine: SemanticSearchEngine, project_root: Path):
        """Initialize search handlers.

        Args:
            search_engine: Semantic search engine instance
            project_root: Project root directory
        """
        self.search_engine = search_engine
        self.project_root = project_root

    async def handle_search_code(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_code tool call.

        Args:
            args: Tool call arguments containing query, filters, etc.

        Returns:
            CallToolResult with search results or error
        """
        query = args.get("query", "")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)
        file_extensions = args.get("file_extensions")
        language = args.get("language")
        function_name = args.get("function_name")
        class_name = args.get("class_name")
        files = args.get("files")

        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="Query parameter is required")],
                isError=True,
            )

        # Build filters
        filters = {}
        if file_extensions:
            filters["file_extension"] = {"$in": file_extensions}
        if language:
            filters["language"] = language
        if function_name:
            filters["function_name"] = function_name
        if class_name:
            filters["class_name"] = class_name
        if files:
            filters["file_pattern"] = files

        # Perform search
        results = await self.search_engine.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

        # Format results
        response_text = self._format_search_results(results, query)
        return CallToolResult(content=[TextContent(type="text", text=response_text)])

    async def handle_search_similar(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_similar tool call.

        Args:
            args: Tool call arguments containing file_path, etc.

        Returns:
            CallToolResult with similar code results or error
        """
        file_path = args.get("file_path", "")
        function_name = args.get("function_name")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)

        if not file_path:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            # Convert to Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                file_path_obj = self.project_root / file_path_obj

            if not file_path_obj.exists():
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"File not found: {file_path}")
                    ],
                    isError=True,
                )

            # Run similar search
            results = await self.search_engine.search_similar(
                file_path=file_path_obj,
                function_name=function_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"No similar code found for {file_path}"
                        )
                    ]
                )

            response_text = self._format_similar_results(results, file_path)
            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Similar search failed: {str(e)}")
                ],
                isError=True,
            )

    async def handle_search_context(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_context tool call.

        Args:
            args: Tool call arguments containing description, focus_areas, etc.

        Returns:
            CallToolResult with contextual search results or error
        """
        description = args.get("description", "")
        focus_areas = args.get("focus_areas")
        limit = args.get("limit", 10)

        if not description:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="description parameter is required")
                ],
                isError=True,
            )

        try:
            # Perform context search
            results = await self.search_engine.search_by_context(
                context_description=description, focus_areas=focus_areas, limit=limit
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"No contextually relevant code found for: {description}",
                        )
                    ]
                )

            response_text = self._format_context_results(
                results, description, focus_areas
            )
            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Context search failed: {str(e)}")
                ],
                isError=True,
            )

    def _format_search_results(self, results: list, query: str) -> str:
        """Format search results for display.

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Formatted text response
        """
        if not results:
            return f"No results found for query: '{query}'"

        response_lines = [f"Found {len(results)} results for query: '{query}'\n"]

        for i, result in enumerate(results, 1):
            response_lines.append(
                f"## Result {i} (Score: {result.similarity_score:.3f})"
            )
            response_lines.append(f"**File:** {result.file_path}")
            if result.function_name:
                response_lines.append(f"**Function:** {result.function_name}")
            if result.class_name:
                response_lines.append(f"**Class:** {result.class_name}")
            response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
            response_lines.append("**Code:**")
            response_lines.append("```" + (result.language or ""))
            response_lines.append(result.content)
            response_lines.append("```\n")

        return "\n".join(response_lines)

    def _format_similar_results(self, results: list, file_path: str) -> str:
        """Format similar code results for display.

        Args:
            results: List of similar code results
            file_path: Original file path

        Returns:
            Formatted text response
        """
        response_lines = [
            f"Found {len(results)} similar code snippets for {file_path}\n"
        ]

        for i, result in enumerate(results, 1):
            response_lines.append(
                f"## Result {i} (Score: {result.similarity_score:.3f})"
            )
            response_lines.append(f"**File:** {result.file_path}")
            if result.function_name:
                response_lines.append(f"**Function:** {result.function_name}")
            if result.class_name:
                response_lines.append(f"**Class:** {result.class_name}")
            response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
            response_lines.append("**Code:**")
            response_lines.append("```" + (result.language or ""))
            # Show more of the content for similar search
            content_preview = (
                result.content[:500] if len(result.content) > 500 else result.content
            )
            response_lines.append(
                content_preview + ("..." if len(result.content) > 500 else "")
            )
            response_lines.append("```\n")

        return "\n".join(response_lines)

    def _format_context_results(
        self, results: list, description: str, focus_areas: list[str] | None
    ) -> str:
        """Format contextual search results for display.

        Args:
            results: List of contextual search results
            description: Original context description
            focus_areas: Optional focus areas

        Returns:
            Formatted text response
        """
        response_lines = [f"Found {len(results)} contextually relevant code snippets"]
        if focus_areas:
            response_lines[0] += f" (focus: {', '.join(focus_areas)})"
        response_lines[0] += f" for: {description}\n"

        for i, result in enumerate(results, 1):
            response_lines.append(
                f"## Result {i} (Score: {result.similarity_score:.3f})"
            )
            response_lines.append(f"**File:** {result.file_path}")
            if result.function_name:
                response_lines.append(f"**Function:** {result.function_name}")
            if result.class_name:
                response_lines.append(f"**Class:** {result.class_name}")
            response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
            response_lines.append("**Code:**")
            response_lines.append("```" + (result.language or ""))
            # Show more of the content for context search
            content_preview = (
                result.content[:500] if len(result.content) > 500 else result.content
            )
            response_lines.append(
                content_preview + ("..." if len(result.content) > 500 else "")
            )
            response_lines.append("```\n")

        return "\n".join(response_lines)
