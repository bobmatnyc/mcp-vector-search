"""Code story narrative generator.

Orchestrates the full pipeline: extract → analyze → synthesize → render.
Produces a StoryIndex JSON artifact consumed by multiple renderers.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .models import (
    ContributorGraphEdge,
    ContributorGraphNode,
    StoryAnalysis,
    StoryExtraction,
    StoryIndex,
    StoryMetadata,
    StoryNarrative,
    StoryVisualization,
    TimelineEvent,
    TimelineEventType,
)

logger = logging.getLogger(__name__)


class StoryGenerator:
    """Main facade for story generation.

    Orchestrates: extract → analyze → synthesize → render.
    Each phase is optional and degrades gracefully.
    """

    def __init__(
        self,
        project_root: Path,
        max_commits: int = 200,
        max_issues: int = 100,
        use_llm: bool = True,
        model: str | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.max_commits = max_commits
        self.max_issues = max_issues
        self.use_llm = use_llm
        self.model = model

    async def generate(self) -> StoryIndex:
        """Run the full pipeline and return a StoryIndex."""
        start_time = time.time()

        # Phase 1: Extract
        extraction = await self.extract()

        # Phase 2: Analyze (enriches with vector search - graceful degradation)
        analysis = await self.analyze(extraction)

        # Phase 3: Synthesize (LLM narrative - optional)
        narrative = StoryNarrative()
        if self.use_llm:
            try:
                narrative = await self.synthesize(extraction, analysis)
            except Exception as e:
                logger.warning(
                    f"LLM synthesis failed, continuing without narrative: {e}"
                )

        # Phase 4: Build visualization data
        visualization = self._build_visualization(extraction, analysis)

        # Build metadata
        metadata = self._build_metadata(extraction, analysis, narrative, start_time)

        return StoryIndex(
            metadata=metadata,
            extraction=extraction,
            analysis=analysis,
            narrative=narrative,
            visualization=visualization,
        )

    async def extract(self) -> StoryExtraction:
        """Extract git/GitHub data."""
        from .extractor import StoryExtractor

        extractor = StoryExtractor(self.project_root)
        extraction = await extractor.extract_all()

        # Limit commits if requested
        if self.max_commits and len(extraction.commits) > self.max_commits:
            extraction.commits = extraction.commits[: self.max_commits]

        # Limit issues if requested
        if self.max_issues and len(extraction.issues) > self.max_issues:
            extraction.issues = extraction.issues[: self.max_issues]

        return extraction

    async def analyze(self, extraction: StoryExtraction) -> StoryAnalysis:
        """Run semantic analysis."""
        from .analyzer import StoryAnalyzer

        analyzer = StoryAnalyzer(self.project_root)
        return await analyzer.analyze(extraction)

    async def synthesize(
        self, extraction: StoryExtraction, analysis: StoryAnalysis
    ) -> StoryNarrative:
        """Generate LLM narrative."""
        from ..core.llm_client import LLMClient
        from .synthesizer import StorySynthesizer

        llm = LLMClient(model=self.model)
        synth = StorySynthesizer(llm)
        return await synth.synthesize(extraction, analysis)

    def _build_visualization(
        self, extraction: StoryExtraction, analysis: StoryAnalysis
    ) -> StoryVisualization:
        """Build pre-computed visualization data from extraction + analysis."""
        # Build timeline events (top commits by file changes, issues, PRs)
        timeline_events: list[TimelineEvent] = []

        # Add top commits (by file changes)
        top_commits = sorted(
            extraction.commits, key=lambda c: c.files_changed, reverse=True
        )[:10]

        for commit in top_commits:
            importance = min(commit.files_changed / 20.0, 1.0)  # Normalize to 0-1
            timeline_events.append(
                TimelineEvent(
                    date=commit.date,
                    event_type=TimelineEventType.COMMIT,
                    title=commit.message[:60],
                    description=f"{commit.files_changed} files changed by {commit.author_name}",
                    importance=importance,
                    metadata={
                        "hash": commit.short_hash,
                        "author": commit.author_name,
                        "files_changed": commit.files_changed,
                        "insertions": commit.insertions,
                        "deletions": commit.deletions,
                    },
                )
            )

        # Add issue events (open/close)
        for issue in extraction.issues[:10]:
            if issue.created_at:
                timeline_events.append(
                    TimelineEvent(
                        date=issue.created_at,
                        event_type=TimelineEventType.ISSUE,
                        title=f"Issue #{issue.number}: {issue.title}",
                        description=f"Opened - {', '.join(issue.labels) if issue.labels else 'no labels'}",
                        importance=0.6,
                        metadata={
                            "number": issue.number,
                            "state": issue.state,
                            "labels": issue.labels,
                        },
                    )
                )

            if issue.closed_at:
                timeline_events.append(
                    TimelineEvent(
                        date=issue.closed_at,
                        event_type=TimelineEventType.ISSUE,
                        title=f"Issue #{issue.number} closed",
                        description=issue.title,
                        importance=0.5,
                        metadata={"number": issue.number},
                    )
                )

        # Add PR events (merge)
        for pr in extraction.pull_requests[:10]:
            if pr.merged_at:
                importance = min(pr.files_changed / 15.0, 1.0)
                timeline_events.append(
                    TimelineEvent(
                        date=pr.merged_at,
                        event_type=TimelineEventType.PR,
                        title=f"PR #{pr.number}: {pr.title}",
                        description=f"{pr.files_changed} files, +{pr.additions}/-{pr.deletions}",
                        importance=importance,
                        metadata={
                            "number": pr.number,
                            "files_changed": pr.files_changed,
                            "additions": pr.additions,
                            "deletions": pr.deletions,
                        },
                    )
                )

        # Sort timeline by date
        timeline_events.sort(key=lambda e: e.date)

        # Build commit density (date -> count)
        commit_density: dict[str, int] = defaultdict(int)
        for commit in extraction.commits:
            date_str = commit.date.strftime("%Y-%m-%d")
            commit_density[date_str] += 1

        # Build contributor graph
        contributor_nodes: list[ContributorGraphNode] = []
        contributor_edges: list[ContributorGraphEdge] = []

        # Track files per contributor
        contributor_files: dict[str, set[str]] = defaultdict(set)
        for commit in extraction.commits:
            contributor_files[commit.author_name].update(commit.files)

        # Create nodes
        for contributor_info in extraction.contributors:
            contributor_nodes.append(
                ContributorGraphNode(
                    name=contributor_info.name,
                    commit_count=contributor_info.commit_count,
                    files_touched=len(contributor_files[contributor_info.name]),
                )
            )

        # Create edges (contributors who worked on same files)
        contributor_names = [c.name for c in extraction.contributors]
        for i, name1 in enumerate(contributor_names):
            for name2 in contributor_names[i + 1 :]:
                shared_files = len(contributor_files[name1] & contributor_files[name2])
                if shared_files > 0:
                    weight = shared_files / max(
                        len(contributor_files[name1]), len(contributor_files[name2])
                    )
                    contributor_edges.append(
                        ContributorGraphEdge(
                            source=name1,
                            target=name2,
                            shared_files=shared_files,
                            weight=weight,
                        )
                    )

        # Build activity heatmap (contributor -> {date: count})
        activity_heatmap: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for commit in extraction.commits:
            date_str = commit.date.strftime("%Y-%m-%d")
            activity_heatmap[commit.author_name][date_str] += 1

        # Convert defaultdicts to regular dicts for serialization
        activity_heatmap_serialized = {
            contributor: dict(dates) for contributor, dates in activity_heatmap.items()
        }

        return StoryVisualization(
            timeline_events=timeline_events,
            contributor_nodes=contributor_nodes,
            contributor_edges=contributor_edges,
            commit_density=dict(commit_density),
            activity_heatmap=activity_heatmap_serialized,
        )

    def _build_metadata(
        self,
        extraction: StoryExtraction,
        analysis: StoryAnalysis,
        narrative: StoryNarrative,
        start_time: float,
    ) -> StoryMetadata:
        """Build metadata from pipeline results."""
        from .. import __version__

        # Get git info
        git_remote = ""
        git_branch = ""
        git_commit = ""
        project_name = self.project_root.name

        try:
            import subprocess

            # Get remote URL
            try:
                result = subprocess.run(  # nosec B607 - git is a trusted binary
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                git_remote = result.stdout.strip()

                # Extract project name from remote URL
                if git_remote:
                    # Handle both SSH and HTTPS URLs
                    # git@github.com:user/repo.git -> repo
                    # https://github.com/user/repo.git -> repo
                    if git_remote.endswith(".git"):
                        project_name = git_remote.rsplit("/", 1)[-1][:-4]
                    else:
                        project_name = git_remote.rsplit("/", 1)[-1]
            except Exception as e:
                logger.debug(f"Could not get git remote: {e}")

            # Get branch
            try:
                result = subprocess.run(  # nosec B607 - git is a trusted binary
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                git_branch = result.stdout.strip()
            except Exception as e:
                logger.debug(f"Could not get git branch: {e}")

            # Get HEAD commit
            try:
                result = subprocess.run(  # nosec B607 - git is a trusted binary
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                git_commit = result.stdout.strip()[:8]
            except Exception as e:
                logger.debug(f"Could not get git commit: {e}")

        except Exception as e:
            logger.debug(f"Git info extraction failed: {e}")

        # Get date range from commits
        date_range_start = None
        date_range_end = None
        if extraction.commits:
            dates = [c.date for c in extraction.commits]
            date_range_start = min(dates)
            date_range_end = max(dates)

        # Count unique files from commits
        unique_files = set()
        for commit in extraction.commits:
            unique_files.update(commit.files)

        return StoryMetadata(
            version="1.0.0",
            generated_at=datetime.utcnow(),
            generator=f"mcp-vector-search v{__version__}",
            project_name=project_name,
            project_root=str(self.project_root),
            git_remote=git_remote,
            git_branch=git_branch,
            git_commit=git_commit,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            total_commits=len(extraction.commits),
            total_contributors=len(extraction.contributors),
            total_files=len(unique_files),
            generation_time_seconds=time.time() - start_time,
            llm_model=self.model,
            has_semantic_analysis=bool(analysis.clusters or analysis.tech_stack),
            has_llm_narrative=bool(narrative.title or narrative.acts),
        )

    def render(
        self, story: StoryIndex, format: str = "all", output_dir: Path | None = None
    ) -> list[Path]:
        """Render story to specified format(s). Returns list of output file paths."""
        if output_dir is None:
            output_dir = self.project_root
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        formats = ["json", "markdown", "html"] if format == "all" else [format]

        for fmt in formats:
            if fmt == "json":
                from .renderers.json_renderer import render_json

                path = output_dir / "story-index.json"
                render_json(story, path)
                outputs.append(path)
            elif fmt == "markdown":
                from .renderers.markdown_renderer import render_markdown

                path = output_dir / "STORY.md"
                render_markdown(story, path)
                outputs.append(path)
            elif fmt == "html":
                from .renderers.html_renderer import render_html

                path = output_dir / "story.html"
                render_html(story, path)
                outputs.append(path)

        return outputs


__all__ = ["StoryGenerator", "StoryIndex"]
