"""Review operation handlers for MCP vector search server."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..analysis.review import ReviewEngine, ReviewType
from ..analysis.review.pr_engine import PRReviewEngine
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError, SearchError
from ..core.factory import create_database
from ..core.git import GitError, GitManager, GitNotAvailableError, GitNotRepoError
from ..core.llm_client import LLMClient
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine


class ReviewHandlers:
    """Handlers for code review-related MCP tool operations."""

    def __init__(
        self,
        project_root: Path,
        search_engine: SemanticSearchEngine | None = None,
        llm_client: LLMClient | None = None,
    ):
        """Initialize review handlers.

        Args:
            project_root: Project root directory
            search_engine: Optional pre-initialized search engine
            llm_client: Optional pre-initialized LLM client
        """
        self.project_root = project_root
        self.search_engine = search_engine
        self.llm_client = llm_client

    async def handle_review_repository(self, args: dict[str, Any]) -> CallToolResult:
        """Handle review_repository tool call.

        Args:
            args: Tool call arguments containing review_type, scope, max_chunks, changed_only

        Returns:
            CallToolResult with review findings or error
        """
        try:
            # Extract arguments
            review_type_str = args.get("review_type", "security")
            scope = args.get("scope")
            max_chunks = args.get("max_chunks", 30)
            changed_only = args.get("changed_only", False)

            # Validate review type
            try:
                review_type = ReviewType(review_type_str.lower())
            except ValueError:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Invalid review type: {review_type_str}. Must be: security, architecture, or performance",
                        )
                    ],
                    isError=True,
                )

            # Check if project is initialized
            project_manager = ProjectManager(self.project_root)
            if not project_manager.is_initialized():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Project not initialized at {self.project_root}. Run index_project first.",
                        )
                    ],
                    isError=True,
                )

            config = project_manager.load_config()

            # Initialize search engine if not provided
            search_engine = self.search_engine
            if not search_engine:
                embedding_function, _ = create_embedding_function(
                    config.embedding_model
                )
                database = create_database(
                    persist_directory=config.index_path / "lance",
                    embedding_function=embedding_function,
                    collection_name="vectors",
                )
                await database.initialize()

                search_engine = SemanticSearchEngine(
                    database=database,
                    project_root=self.project_root,
                    similarity_threshold=config.similarity_threshold,
                )

            # Initialize LLM client if not provided
            llm_client = self.llm_client
            if not llm_client:
                llm_client = LLMClient()

            # Try to load knowledge graph (optional)
            knowledge_graph = None
            try:
                from ..core.knowledge_graph import KnowledgeGraph

                kg_path = config.index_path / "kg.db"
                if kg_path.exists():
                    knowledge_graph = KnowledgeGraph(kg_path)
            except Exception as e:
                logger.debug(f"Knowledge graph not available: {e}")

            # Get changed files if requested
            file_filter = None
            if changed_only:
                try:
                    git_manager = GitManager(self.project_root)
                    changed_files = git_manager.get_changed_files(
                        include_untracked=True
                    )
                    file_filter = [str(f) for f in changed_files]

                    if not file_filter:
                        return CallToolResult(
                            content=[
                                TextContent(
                                    type="text",
                                    text="No changed files found. Nothing to review.",
                                )
                            ],
                            isError=False,
                        )

                    logger.info(f"Reviewing {len(file_filter)} changed files")

                except (GitNotAvailableError, GitNotRepoError, GitError) as e:
                    logger.warning(f"Git error: {e}, falling back to full review")
                    file_filter = None

            # Create review engine
            review_engine = ReviewEngine(
                search_engine=search_engine,
                knowledge_graph=knowledge_graph,
                llm_client=llm_client,
                project_root=self.project_root,
            )

            # Run review
            result = await review_engine.run_review(
                review_type=review_type,
                scope=scope,
                max_chunks=max_chunks,
                file_filter=file_filter,
            )

            # Format response as JSON
            response_data = {
                "status": "success",
                "review_type": result.review_type.value,
                "scope": result.scope,
                "findings": [
                    {
                        "title": f.title,
                        "description": f.description,
                        "severity": f.severity.value,
                        "file_path": f.file_path,
                        "start_line": f.start_line,
                        "end_line": f.end_line,
                        "category": f.category,
                        "recommendation": f.recommendation,
                        "confidence": f.confidence,
                        "cwe_id": f.cwe_id,
                        "code_snippet": f.code_snippet,
                        "related_files": f.related_files,
                    }
                    for f in result.findings
                ],
                "summary": result.summary,
                "context_chunks_used": result.context_chunks_used,
                "kg_relationships_used": result.kg_relationships_used,
                "model_used": result.model_used,
                "duration_seconds": result.duration_seconds,
            }

            return CallToolResult(
                content=[
                    TextContent(type="text", text=json.dumps(response_data, indent=2))
                ],
                isError=False,
            )

        except ProjectNotFoundError as e:
            return CallToolResult(
                content=[TextContent(type="text", text=str(e))],
                isError=True,
            )
        except SearchError as e:
            logger.error(f"Review failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Review analysis failed: {e}")],
                isError=True,
            )
        except Exception as e:
            logger.error(f"Review handler error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unexpected error: {e}")],
                isError=True,
            )

    async def handle_review_pull_request(self, args: dict[str, Any]) -> CallToolResult:
        """Handle review_pull_request tool call.

        Args:
            args: Tool call arguments containing base_ref, head_ref, custom_instructions

        Returns:
            CallToolResult with PR review or error
        """
        try:
            # Extract arguments
            base_ref = args.get("base_ref", "main")
            head_ref = args.get("head_ref", "HEAD")
            custom_instructions = args.get("custom_instructions")

            # Check if project is initialized
            project_manager = ProjectManager(self.project_root)
            if not project_manager.is_initialized():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Project not initialized at {self.project_root}. Run index_project first.",
                        )
                    ],
                    isError=True,
                )

            config = project_manager.load_config()

            # Initialize search engine if not provided
            search_engine = self.search_engine
            if not search_engine:
                embedding_function, _ = create_embedding_function(
                    config.embedding_model
                )
                database = create_database(
                    persist_directory=config.index_path / "lance",
                    embedding_function=embedding_function,
                    collection_name="vectors",
                )
                await database.initialize()

                search_engine = SemanticSearchEngine(
                    database=database,
                    project_root=self.project_root,
                    similarity_threshold=config.similarity_threshold,
                )

            # Initialize LLM client if not provided
            llm_client = self.llm_client
            if not llm_client:
                llm_client = LLMClient()

            # Try to load knowledge graph (optional)
            knowledge_graph = None
            try:
                from ..core.knowledge_graph import KnowledgeGraph

                kg_path = config.index_path / "kg.db"
                if kg_path.exists():
                    knowledge_graph = KnowledgeGraph(kg_path)
            except Exception as e:
                logger.debug(f"Knowledge graph not available: {e}")

            # Create PR review engine
            pr_engine = PRReviewEngine(
                search_engine=search_engine,
                knowledge_graph=knowledge_graph,
                llm_client=llm_client,
                project_root=self.project_root,
            )

            # Run review
            result = await pr_engine.review_from_git(
                base_ref=base_ref,
                head_ref=head_ref,
                custom_instructions=custom_instructions,
            )

            # Format response as JSON
            response_data = {
                "status": "success",
                "summary": result.summary,
                "verdict": result.verdict,
                "overall_score": result.overall_score,
                "comments": [
                    {
                        "file_path": c.file_path,
                        "line_number": c.line_number,
                        "comment": c.comment,
                        "severity": c.severity.value,
                        "category": c.category,
                        "suggestion": c.suggestion,
                        "is_blocking": c.is_blocking,
                    }
                    for c in result.comments
                ],
                "blocking_issues": result.blocking_issues,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
                "context_files_used": result.context_files_used,
                "kg_relationships_used": result.kg_relationships_used,
                "review_instructions_applied": result.review_instructions_applied,
                "model_used": result.model_used,
                "duration_seconds": result.duration_seconds,
            }

            return CallToolResult(
                content=[
                    TextContent(type="text", text=json.dumps(response_data, indent=2))
                ],
                isError=False,
            )

        except (GitNotAvailableError, GitNotRepoError, GitError) as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Git error: {e}. Ensure the project is a git repository and refs exist.",
                    )
                ],
                isError=True,
            )
        except ProjectNotFoundError as e:
            return CallToolResult(
                content=[TextContent(type="text", text=str(e))],
                isError=True,
            )
        except SearchError as e:
            logger.error(f"PR review failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"PR review analysis failed: {e}")
                ],
                isError=True,
            )
        except Exception as e:
            logger.error(f"PR review handler error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unexpected error: {e}")],
                isError=True,
            )
