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

    async def handle_code_review(self, args: dict[str, Any]) -> CallToolResult:
        """Handle code_review tool call for local pre-push review.

        Args:
            args: Tool call arguments containing include_trivial, focus, max_files, max_chunks, skip_files

        Returns:
            CallToolResult with local code review findings or error
        """
        try:
            # Extract arguments
            include_trivial = args.get("include_trivial", False)
            focus = args.get("focus")
            max_files = args.get("max_files", 20)
            max_chunks = args.get("max_chunks", 30)
            skip_files = args.get("skip_files", [])

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

            # Check if it's a git repository and get staged changes
            try:
                git_manager = GitManager(self.project_root)
                staged_diff = git_manager.get_staged_diff()

                if not staged_diff:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="No staged changes found. Stage your changes with 'git add' before review.",
                            )
                        ],
                        isError=False,
                    )

            except (GitNotAvailableError, GitNotRepoError, GitError) as e:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Git error: {e}. Ensure the project is a git repository with staged changes.",
                        )
                    ],
                    isError=True,
                )

            # Analyze changes to determine if substantial
            changes_analysis = self._analyze_staged_changes(
                git_manager, skip_files, max_files
            )

            if not changes_analysis["is_substantial"] and not include_trivial:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "success",
                                    "is_substantial": False,
                                    "summary": "Changes are trivial - safe to push",
                                    "changes_summary": changes_analysis["summary"],
                                    "push_recommendation": "Safe to push",
                                    "skipped_reason": f"Changes below threshold: {changes_analysis['reason']}",
                                },
                                indent=2,
                            ),
                        )
                    ],
                    isError=False,
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

            # Create review engine focusing on staged files
            review_engine = ReviewEngine(
                search_engine=search_engine,
                knowledge_graph=knowledge_graph,
                llm_client=llm_client,
                project_root=self.project_root,
            )

            # Get list of changed files for review scope
            changed_files = changes_analysis["files"]

            # Run focused review on changed files
            # Use multiple review types if no specific focus
            review_types = [focus] if focus else ["security", "quality", "performance"]
            all_findings = []
            total_context_chunks = 0
            total_kg_relationships = 0

            for review_type_str in review_types:
                try:
                    if review_type_str == "quality":
                        review_type_str = "architecture"  # Map quality to architecture

                    review_type = ReviewType(review_type_str.lower())

                    result = await review_engine.run_review(
                        review_type=review_type,
                        scope=None,  # Will use file_filter instead
                        max_chunks=max_chunks,
                        file_filter=changed_files,
                    )

                    all_findings.extend(result.findings)
                    total_context_chunks += result.context_chunks_used
                    total_kg_relationships += result.kg_relationships_used

                except ValueError:
                    # Skip invalid review types
                    continue
                except Exception as e:
                    logger.warning(f"Review type {review_type_str} failed: {e}")
                    continue

            # Analyze findings and provide verdict
            blocking_issues = sum(
                1 for f in all_findings if f.severity.value in ["critical", "high"]
            )
            warnings = sum(1 for f in all_findings if f.severity.value == "medium")
            suggestions = sum(
                1 for f in all_findings if f.severity.value in ["low", "info"]
            )

            verdict = "clean"
            push_recommendation = "Safe to push"

            if blocking_issues > 0:
                verdict = "blocking_issues"
                push_recommendation = "Fix critical/high severity issues before pushing"
            elif warnings > 0:
                verdict = "warnings"
                push_recommendation = "Consider addressing warnings before pushing"
            elif suggestions > 0:
                verdict = "suggestions"
                push_recommendation = "Safe to push with suggested improvements"

            # Format response as JSON
            response_data = {
                "status": "success",
                "is_substantial": changes_analysis["is_substantial"],
                "summary": f"Found {blocking_issues} critical issues, {warnings} warnings, and {suggestions} suggestions",
                "verdict": verdict,
                "changes_summary": changes_analysis["summary"],
                "findings": [
                    {
                        "file_path": f.file_path,
                        "line_number": f.start_line,
                        "title": f.title,
                        "description": f.description,
                        "severity": f.severity.value,
                        "category": f.category,
                        "recommendation": f.recommendation,
                        "code_suggestion": f.code_snippet,  # Use code snippet as suggestion
                        "is_blocking": f.severity.value in ["critical", "high"],
                    }
                    for f in all_findings
                ],
                "blocking_issues": blocking_issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "push_recommendation": push_recommendation,
                "context_chunks_used": total_context_chunks,
                "kg_relationships_used": total_kg_relationships,
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
            logger.error(f"Local code review failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Review analysis failed: {e}")],
                isError=True,
            )
        except Exception as e:
            logger.error(f"Code review handler error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unexpected error: {e}")],
                isError=True,
            )

    def _analyze_staged_changes(
        self, git_manager: GitManager, skip_files: list[str], max_files: int
    ) -> dict[str, Any]:
        """Analyze staged changes to determine if they are substantial.

        Args:
            git_manager: GitManager instance
            skip_files: File patterns to skip
            max_files: Maximum number of files to analyze

        Returns:
            Dictionary with analysis results
        """
        import fnmatch

        # Get staged changes
        staged_files = git_manager.get_staged_files()

        # Filter out skipped files
        filtered_files = []
        for file_path in staged_files:
            skip = False
            for pattern in skip_files:
                if fnmatch.fnmatch(str(file_path), pattern):
                    skip = True
                    break
            if not skip:
                filtered_files.append(str(file_path))

        # Limit files
        if len(filtered_files) > max_files:
            filtered_files = filtered_files[:max_files]

        # Get diff stats
        try:
            diff_stats = git_manager.get_staged_diff_stats()
        except Exception:
            diff_stats = {
                "files_changed": len(filtered_files),
                "insertions": 0,
                "deletions": 0,
            }

        lines_changed = diff_stats.get("insertions", 0) + diff_stats.get("deletions", 0)
        files_changed = len(filtered_files)

        # Research-backed substantial commit criteria
        is_substantial = False
        reason = ""

        if lines_changed > 20:
            is_substantial = True
            reason = f"{lines_changed} lines changed"
        elif files_changed > 2:
            is_substantial = True
            reason = f"{files_changed} files modified"
        elif any(self._is_security_sensitive_file(f) for f in filtered_files):
            is_substantial = True
            reason = "security-sensitive files modified"
        elif any(self._has_structural_changes(f, git_manager) for f in filtered_files):
            is_substantial = True
            reason = "structural changes detected"
        else:
            reason = f"only {lines_changed} lines in {files_changed} files"

        return {
            "is_substantial": is_substantial,
            "reason": reason,
            "files": filtered_files,
            "summary": {
                "files_changed": files_changed,
                "lines_added": diff_stats.get("insertions", 0),
                "lines_deleted": diff_stats.get("deletions", 0),
                "total_lines_changed": lines_changed,
            },
        }

    def _is_security_sensitive_file(self, file_path: str) -> bool:
        """Check if file is in security-sensitive area."""
        sensitive_patterns = [
            "*auth*",
            "*login*",
            "*password*",
            "*security*",
            "*crypto*",
            "*database*",
            "*db*",
            "*sql*",
            "*query*",
            "*session*",
            "*validation*",
            "*sanitiz*",
            "*permission*",
            "*role*",
        ]

        import fnmatch

        file_lower = file_path.lower()
        return any(
            fnmatch.fnmatch(file_lower, pattern) for pattern in sensitive_patterns
        )

    def _has_structural_changes(self, file_path: str, git_manager: GitManager) -> bool:
        """Check if file has structural changes (new functions, classes, imports)."""
        try:
            diff = git_manager.get_file_diff(file_path, staged=True)
            if not diff:
                return False

            # Simple heuristics for structural changes
            structural_indicators = [
                "+def ",
                "+class ",
                "+import ",
                "+from ",
                "+async def",
                "+@",
                "+interface",
                "+extends",
                "+function",
                "+const ",
                "+let ",
                "+var ",
            ]

            return any(indicator in diff for indicator in structural_indicators)
        except Exception:
            return False
