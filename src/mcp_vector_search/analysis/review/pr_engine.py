"""PR/MR review engine with context-aware analysis.

This module implements the PRReviewEngine class which performs intelligent
pull request reviews using vector search, knowledge graph, and LLM analysis.

Key Features:
    - Context-aware: Uses vector search to find related code (callers, similar patterns)
    - Knowledge graph: Identifies dependencies and impact analysis
    - Customizable: Applies user-defined review instructions
    - Structured output: Returns actionable, categorized comments

Design Philosophy:
    - Gather rich context before asking LLM (no blind reviews)
    - For each changed file, find: callers, similar code, dependencies, tests
    - Apply custom instructions from user config
    - Return structured, actionable feedback with severity levels
"""

import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import SearchError
from ...core.git import GitManager
from ...core.knowledge_graph import KnowledgeGraph
from ...core.llm_client import LLMClient
from ...core.search import SemanticSearchEngine
from .cache import ReviewCache
from .instructions import InstructionsLoader
from .language_profiles import get_languages_in_pr
from .models import Severity
from .pr_models import (
    PRContext,
    PRFilePatch,
    PRReviewComment,
    PRReviewResult,
)

# PR review LLM prompt template
PR_REVIEW_PROMPT = """You are reviewing a pull request using the full codebase as context.

## Pull Request Information

**Title**: {pr_title}
**Description**: {pr_description}
**Base Branch**: {base_branch}
**Head Branch**: {head_branch}

## Language-Specific Standards

{language_context}

## Review Instructions

{review_instructions}

## Changed Files and Context

For each changed file, I've provided:
1. **The diff** (what changed)
2. **Callers** from the knowledge graph (who depends on this code)
3. **Similar patterns** in the codebase (for consistency checking)
4. **Existing tests** (to identify test gaps)

{file_contexts}

## Your Task

Review each changed file/function and provide:
- **Inline comments** for specific issues (with file path and line number)
- **Overall PR-level comments** for cross-cutting concerns

For each issue found, provide:
- `file_path`: Path to file (null for overall PR comments)
- `line_number`: Line number (null for file-level or overall comments)
- `comment`: Clear, actionable feedback
- `severity`: "critical", "high", "medium", "low", or "info"
- `category`: "security", "quality", "style", "performance", "logic", or "architecture"
- `suggestion`: Optional code fix (null if no specific fix)
- `is_blocking`: true for critical/high severity issues that should block merge

## Severity Guidelines

- **critical**: Security vulnerabilities, data loss risks, correctness bugs
- **high**: Major bugs, breaking changes without migration path
- **medium**: Code quality issues, missing error handling, test gaps
- **low**: Style inconsistencies, minor improvements
- **info**: Informational comments, suggestions, best practices

## Focus Areas

- **Security**: SQL injection, XSS, hardcoded secrets, auth bypass
- **Correctness**: Logic errors, edge cases, race conditions
- **Quality**: Error handling, input validation, test coverage
- **Style**: Consistency with similar code patterns in the codebase
- **Architecture**: Separation of concerns, dependency direction

## Output Format

Return ONLY a JSON array of comments (no additional text):

```json
[
  {{
    "file_path": "src/auth.py",
    "line_number": 42,
    "comment": "SQL injection vulnerability: user input is concatenated directly into query without parameterization.",
    "severity": "critical",
    "category": "security",
    "suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
    "is_blocking": true
  }},
  {{
    "file_path": null,
    "line_number": null,
    "comment": "Overall PR looks good, but consider adding integration tests for the auth flow.",
    "severity": "info",
    "category": "quality",
    "suggestion": null,
    "is_blocking": false
  }}
]
```

Analyze the PR and return your review:"""


class PRReviewEngine:
    """Reviews pull requests using full codebase context.

    Uses vector search to find related code, knowledge graph for impact analysis,
    and LLM for intelligent review with custom instructions.

    Example:
        >>> engine = PRReviewEngine(search_engine, kg, llm_client, project_root)
        >>> result = await engine.review_from_git(base_ref="main", head_ref="HEAD")
        >>> print(result.summary)
        >>> for comment in result.comments:
        ...     print(f"{comment.severity}: {comment.comment}")
    """

    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        knowledge_graph: KnowledgeGraph | None,
        llm_client: LLMClient,
        project_root: Path,
    ):
        """Initialize PR review engine.

        Args:
            search_engine: Semantic search engine for finding related code
            knowledge_graph: Optional KG for relationship queries
            llm_client: LLM client for code analysis
            project_root: Project root directory
        """
        self.search_engine = search_engine
        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client
        self.project_root = project_root
        self.instructions_loader = InstructionsLoader(project_root)
        self.cache = ReviewCache(project_root)

    async def review_pr(
        self,
        pr_context: PRContext,
        review_types: list[str] | None = None,
        custom_instructions: str | None = None,
    ) -> PRReviewResult:
        """Review a pull request with full codebase context.

        Pipeline:
        1. For each changed file, gather context via vector search and KG
        2. Apply custom review instructions
        3. Call LLM with rich context
        4. Parse structured comments
        5. Generate verdict and summary

        Args:
            pr_context: Full PR context with all file patches
            review_types: Types of review to focus on (default: ["security", "quality", "style"])
            custom_instructions: Override loaded instructions

        Returns:
            PRReviewResult with structured comments and verdict
        """
        start_time = time.time()
        logger.info(f"Starting PR review: {pr_context.title}")

        # Load or use custom instructions
        if custom_instructions:
            instructions_text = custom_instructions
            instructions_name = "custom (override)"
        else:
            instructions_text = self.instructions_loader.load()
            instructions_text = self.instructions_loader.format_for_prompt()
            instructions_name = "project standards"

        # Gather context for each changed file
        file_contexts = await self._gather_file_contexts(pr_context.patches)
        context_files_used = sum(len(fc["related_files"]) for fc in file_contexts)
        kg_relationships_used = sum(len(fc["kg_relationships"]) for fc in file_contexts)

        # Build language-specific context
        language_context = await self._build_language_context(pr_context.patches)

        # Format file contexts for prompt
        file_contexts_text = self._format_file_contexts(file_contexts)

        # Build LLM prompt
        prompt = PR_REVIEW_PROMPT.format(
            pr_title=pr_context.title,
            pr_description=pr_context.description or "No description provided",
            base_branch=pr_context.base_branch,
            head_branch=pr_context.head_branch,
            language_context=language_context,
            review_instructions=instructions_text,
            file_contexts=file_contexts_text,
        )

        # Call LLM
        comments = await self._analyze_with_llm(prompt)

        # Generate verdict and summary
        verdict, overall_score = self._generate_verdict(comments)
        summary = self._generate_summary(comments)

        # Count by severity
        blocking_issues = sum(1 for c in comments if c.is_blocking)
        warnings = sum(
            1
            for c in comments
            if c.severity in [Severity.MEDIUM, Severity.HIGH] and not c.is_blocking
        )
        suggestions = sum(
            1 for c in comments if c.severity in [Severity.LOW, Severity.INFO]
        )

        duration = time.time() - start_time

        return PRReviewResult(
            summary=summary,
            verdict=verdict,
            overall_score=overall_score,
            comments=comments,
            blocking_issues=blocking_issues,
            warnings=warnings,
            suggestions=suggestions,
            context_files_used=context_files_used,
            kg_relationships_used=kg_relationships_used,
            review_instructions_applied=instructions_name,
            model_used=self.llm_client.model,
            duration_seconds=duration,
        )

    async def review_from_git(
        self,
        base_ref: str = "main",
        head_ref: str = "HEAD",
        review_types: list[str] | None = None,
        custom_instructions: str | None = None,
    ) -> PRReviewResult:
        """Review changes between two git refs.

        Args:
            base_ref: Base branch/commit (default: "main")
            head_ref: Head branch/commit (default: "HEAD")
            review_types: Types of review to focus on
            custom_instructions: Override loaded instructions

        Returns:
            PRReviewResult with structured review
        """
        git_manager = GitManager(self.project_root)

        # Get diff
        diff_text = git_manager.get_diff(base_ref, head_ref)

        if not diff_text.strip():
            logger.info("No changes detected between refs")
            return PRReviewResult(
                summary="No changes to review",
                verdict="approve",
                overall_score=1.0,
                comments=[],
                blocking_issues=0,
                warnings=0,
                suggestions=0,
                context_files_used=0,
                kg_relationships_used=0,
                review_instructions_applied=None,
                model_used=self.llm_client.model,
                duration_seconds=0.0,
            )

        # Parse diff into PRContext
        pr_context = self._parse_git_diff(diff_text, base_ref, head_ref, git_manager)

        # Run review
        return await self.review_pr(pr_context, review_types, custom_instructions)

    def _parse_git_diff(
        self, diff_text: str, base_ref: str, head_ref: str, git_manager: GitManager
    ) -> PRContext:
        """Parse git diff into PRContext.

        Args:
            diff_text: Unified diff text
            base_ref: Base reference
            head_ref: Head reference
            git_manager: GitManager for retrieving file contents

        Returns:
            PRContext with parsed patches
        """
        patches: list[PRFilePatch] = []
        stats = git_manager.parse_diff_stats(diff_text)

        # Split diff by file
        file_diffs = re.split(r"^diff --git ", diff_text, flags=re.MULTILINE)

        for file_diff in file_diffs[1:]:  # Skip first empty split
            # Parse file paths from header
            lines = file_diff.splitlines()
            if not lines:
                continue

            # First line: a/old_path b/new_path
            header = lines[0]
            parts = header.split()
            if len(parts) < 2:
                continue

            old_path = parts[0][2:]  # Remove "a/" prefix
            new_path = parts[1][2:]  # Remove "b/" prefix

            # Detect file status
            is_new_file = "new file mode" in file_diff
            is_deleted = "deleted file mode" in file_diff
            is_renamed = old_path != new_path

            # Get file contents at refs
            old_content = (
                None if is_new_file else git_manager.get_file_at_ref(old_path, base_ref)
            )
            new_content = (
                None if is_deleted else git_manager.get_file_at_ref(new_path, head_ref)
            )

            # Get stats
            additions, deletions = stats.get(new_path, (0, 0))

            patch = PRFilePatch(
                file_path=new_path,
                old_content=old_content,
                new_content=new_content,
                diff_text="diff --git " + file_diff,  # Reconstruct full diff
                additions=additions,
                deletions=deletions,
                is_new_file=is_new_file,
                is_deleted=is_deleted,
                is_renamed=is_renamed,
                old_path=old_path if is_renamed else None,
            )

            patches.append(patch)

        # Get current branch for head_branch
        current_branch = git_manager.get_current_branch() or head_ref

        return PRContext(
            title=f"Changes from {base_ref} to {head_ref}",
            description=None,
            base_branch=base_ref,
            head_branch=current_branch,
            patches=patches,
        )

    async def _gather_file_contexts(
        self, patches: list[PRFilePatch]
    ) -> list[dict[str, Any]]:
        """Gather rich context for each changed file.

        For each file, finds:
        - Related files (callers, dependencies)
        - Similar code patterns
        - Knowledge graph relationships
        - Existing tests

        Args:
            patches: List of file patches

        Returns:
            List of context dictionaries
        """
        contexts = []

        for patch in patches:
            if patch.is_deleted:
                # Skip deleted files (no context needed)
                continue

            context = {
                "file_path": patch.file_path,
                "diff": patch.diff_text,
                "additions": patch.additions,
                "deletions": patch.deletions,
                "related_files": [],
                "kg_relationships": [],
            }

            # Search for related code
            try:
                # Search for similar patterns
                search_results = await self.search_engine.search(
                    query=f"code similar to {patch.file_path}",
                    limit=5,
                    similarity_threshold=0.5,
                    include_context=True,
                )

                context["related_files"] = [
                    {
                        "path": r.file_path,
                        "function": r.function_name,
                        "similarity": r.similarity_score,
                    }
                    for r in search_results
                ]

            except SearchError as e:
                logger.debug(f"Search failed for {patch.file_path}: {e}")

            # Gather KG relationships (if available)
            if self.knowledge_graph:
                try:
                    # Extract function names from diff
                    function_names = self._extract_function_names(patch.diff_text)

                    for func_name in function_names:
                        related = await self.knowledge_graph.find_related(
                            func_name, max_hops=1
                        )
                        if related:
                            context["kg_relationships"].append(
                                {
                                    "entity": func_name,
                                    "related": related[:5],  # Limit to 5
                                }
                            )

                except Exception as e:
                    logger.debug(f"KG query failed for {patch.file_path}: {e}")

            contexts.append(context)

        return contexts

    def _extract_function_names(self, diff_text: str) -> list[str]:
        """Extract function names from diff text.

        Args:
            diff_text: Unified diff text

        Returns:
            List of function names found in diff
        """
        function_names = []

        # Simple regex patterns for common languages
        patterns = [
            r"def\s+(\w+)\s*\(",  # Python
            r"function\s+(\w+)\s*\(",  # JavaScript
            r"func\s+(\w+)\s*\(",  # Go
            r"fn\s+(\w+)\s*\(",  # Rust
            r"(public|private|protected)?\s*\w+\s+(\w+)\s*\(",  # Java/C++
        ]

        for line in diff_text.splitlines():
            if line.startswith("+"):  # Only look at added lines
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        # Handle tuple results from patterns with groups
                        for match in matches:
                            if isinstance(match, tuple):
                                function_names.append(match[-1])  # Last group
                            else:
                                function_names.append(match)

        return list(set(function_names))  # Deduplicate

    async def _build_language_context(self, patches: list[PRFilePatch]) -> str:
        """Build language-specific context for the review prompt.

        Extracts language profiles for all languages present in the PR
        and formats their idioms, anti-patterns, and security concerns
        for injection into the review prompt.

        Args:
            patches: List of file patches in the PR

        Returns:
            Formatted language context string (empty if no languages detected)

        Example:
            >>> context = await engine._build_language_context(patches)
            >>> print(context)
            ## Python Standards
            **Idioms to enforce:**
            - Use type hints for all function signatures (PEP 484)
            ...
        """
        languages = get_languages_in_pr(patches)

        if not languages:
            return ""

        context_parts = []
        for profile in languages:
            context_parts.append(f"## {profile.name} Standards")
            context_parts.append("")

            # Show top idioms
            if profile.idioms:
                context_parts.append("**Idioms to enforce:**")
                for idiom in profile.idioms[:5]:  # Limit to top 5
                    context_parts.append(f"- {idiom}")
                context_parts.append("")

            # Show top anti-patterns
            if profile.anti_patterns:
                context_parts.append("**Anti-patterns to flag:**")
                for ap in profile.anti_patterns[:4]:  # Limit to top 4
                    context_parts.append(f"- {ap}")
                context_parts.append("")

            # Show top security concerns
            if profile.security_patterns:
                context_parts.append("**Security concerns:**")
                for sp in profile.security_patterns[:4]:  # Limit to top 4
                    context_parts.append(f"- {sp}")
                context_parts.append("")

        return "\n".join(context_parts)

    def _format_file_contexts(self, file_contexts: list[dict[str, Any]]) -> str:
        """Format file contexts for LLM prompt.

        Args:
            file_contexts: List of context dictionaries

        Returns:
            Formatted text
        """
        sections = []

        for ctx in file_contexts:
            sections.append(f"### File: {ctx['file_path']}")
            sections.append(f"**Changes**: +{ctx['additions']} -{ctx['deletions']}")
            sections.append("")

            # Show diff
            sections.append("**Diff:**")
            sections.append("```diff")
            # Truncate very large diffs
            diff_lines = ctx["diff"].splitlines()
            if len(diff_lines) > 100:
                sections.extend(diff_lines[:50])
                sections.append("... (diff truncated) ...")
                sections.extend(diff_lines[-50:])
            else:
                sections.append(ctx["diff"])
            sections.append("```")
            sections.append("")

            # Show related files
            if ctx["related_files"]:
                sections.append("**Similar Code Patterns:**")
                for rf in ctx["related_files"]:
                    sections.append(
                        f"- {rf['path']}:{rf.get('function', 'N/A')} "
                        f"(similarity: {rf['similarity']:.2f})"
                    )
                sections.append("")

            # Show KG relationships
            if ctx["kg_relationships"]:
                sections.append("**Knowledge Graph Relationships:**")
                for kg in ctx["kg_relationships"]:
                    sections.append(f"- Entity: `{kg['entity']}`")
                    for rel in kg["related"]:
                        rel_type = rel.get("relationship", "related_to")
                        rel_name = rel.get("name", "unknown")
                        sections.append(f"  - {rel_type}: `{rel_name}`")
                sections.append("")

            sections.append("---")
            sections.append("")

        return "\n".join(sections)

    async def _analyze_with_llm(self, prompt: str) -> list[PRReviewComment]:
        """Call LLM and parse structured comments.

        Args:
            prompt: Full review prompt

        Returns:
            List of PRReviewComment objects
        """
        try:
            messages = [{"role": "system", "content": prompt}]

            response = await self.llm_client._chat_completion(messages)

            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # Parse JSON comments
            comments = self._parse_comments_json(content)

            logger.info(f"LLM returned {len(comments)} review comments")

            return comments

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise SearchError(f"PR review analysis failed: {e}") from e

    def _parse_comments_json(self, llm_response: str) -> list[PRReviewComment]:
        """Parse JSON comments from LLM response.

        Args:
            llm_response: Raw LLM response

        Returns:
            List of PRReviewComment objects
        """
        # Extract JSON from markdown code blocks
        json_match = re.search(r"```json\s*\n(.*?)\n```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try without markdown
            json_match = re.search(r"```\s*\n(.*?)\n```", llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume entire response is JSON
                json_str = llm_response.strip()

        try:
            comments_data = json.loads(json_str)

            if not isinstance(comments_data, list):
                logger.warning("LLM returned non-list JSON, wrapping in list")
                comments_data = [comments_data]

            comments = []
            for item in comments_data:
                try:
                    comment = PRReviewComment(
                        file_path=item.get("file_path"),
                        line_number=item.get("line_number"),
                        comment=item["comment"],
                        severity=Severity(item["severity"].lower()),
                        category=item["category"],
                        suggestion=item.get("suggestion"),
                        is_blocking=item["is_blocking"],
                    )
                    comments.append(comment)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse comment: {e}, item: {item}")
                    continue

            return comments

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"LLM response: {llm_response[:500]}")
            return []

    def _generate_verdict(self, comments: list[PRReviewComment]) -> tuple[str, float]:
        """Generate verdict and overall score from comments.

        Args:
            comments: List of review comments

        Returns:
            Tuple of (verdict, overall_score)
                verdict: "approve", "request_changes", or "comment"
                overall_score: 0.0-1.0
        """
        if not comments:
            return ("approve", 1.0)

        # Count blocking issues
        blocking_count = sum(1 for c in comments if c.is_blocking)

        if blocking_count > 0:
            verdict = "request_changes"
        elif any(c.severity in [Severity.MEDIUM, Severity.HIGH] for c in comments):
            verdict = "comment"  # Has suggestions but not blocking
        else:
            verdict = "approve"  # Only low/info severity

        # Calculate score (0.0 = worst, 1.0 = best)
        # Deduct points for each severity level
        severity_weights = {
            Severity.CRITICAL: 0.25,
            Severity.HIGH: 0.15,
            Severity.MEDIUM: 0.10,
            Severity.LOW: 0.05,
            Severity.INFO: 0.01,
        }

        deductions = sum(severity_weights.get(c.severity, 0.0) for c in comments)
        overall_score = max(0.0, 1.0 - deductions)

        return (verdict, overall_score)

    def _generate_summary(self, comments: list[PRReviewComment]) -> str:
        """Generate review summary from comments.

        Args:
            comments: List of review comments

        Returns:
            Summary string
        """
        if not comments:
            return "No issues found. Code looks good!"

        # Count by severity
        severity_counts = {}
        for comment in comments:
            severity_counts[comment.severity] = (
                severity_counts.get(comment.severity, 0) + 1
            )

        # Build summary
        parts = [f"Found {len(comments)} issue(s):"]

        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                parts.append(f"  - {severity.value.upper()}: {count}")

        return "\n".join(parts)
