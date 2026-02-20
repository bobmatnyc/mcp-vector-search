"""Markdown renderer for StoryIndex."""

from __future__ import annotations

from pathlib import Path

from ..models import ConfidenceLevel, StoryIndex


def render_markdown(story: StoryIndex, output_path: Path | None = None) -> str:
    """Render StoryIndex as rich markdown document.

    Args:
        story: The StoryIndex to render
        output_path: If provided, write to this file

    Returns:
        Markdown string
    """
    lines: list[str] = []

    # === Header ===
    if story.narrative.title:
        lines.append(f"# {story.narrative.title}")
    else:
        project_name = story.metadata.project_name or "Project"
        lines.append(f"# {project_name} - Code Story")

    if story.narrative.subtitle:
        lines.append(f"\n*{story.narrative.subtitle}*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # === Metadata ===
    lines.append("## Project Overview")
    lines.append("")

    if story.metadata.project_name:
        lines.append(f"**Project**: {story.metadata.project_name}")

    if story.metadata.git_remote:
        lines.append(f"**Repository**: {story.metadata.git_remote}")

    if story.metadata.git_branch:
        lines.append(f"**Branch**: {story.metadata.git_branch}")

    # Date range
    if story.metadata.date_range_start and story.metadata.date_range_end:
        start_str = story.metadata.date_range_start.strftime("%B %Y")
        end_str = story.metadata.date_range_end.strftime("%B %Y")
        lines.append(f"**Timeline**: {start_str} - {end_str}")

    # Statistics
    lines.append("")
    lines.append("### Statistics")
    lines.append("")
    lines.append(f"- **Commits**: {story.metadata.total_commits:,}")
    lines.append(f"- **Contributors**: {story.metadata.total_contributors}")
    lines.append(f"- **Files**: {story.metadata.total_files:,}")

    if story.metadata.generated_at:
        gen_date = story.metadata.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"- **Generated**: {gen_date}")

    lines.append("")

    # === Table of Contents ===
    lines.append("## Table of Contents")
    lines.append("")

    toc_items = ["Executive Summary"]

    if story.narrative.acts:
        toc_items.extend(
            [f"Act {act.number}: {act.title}" for act in story.narrative.acts]
        )

    if story.narrative.themes:
        toc_items.append("Themes")

    if story.extraction.contributors:
        toc_items.append("Key Contributors")

    if story.analysis.tech_stack:
        toc_items.append("Technology Stack")

    if story.narrative.roads_not_taken:
        toc_items.append("Roads Not Taken")

    if story.visualization.timeline_events:
        toc_items.append("Timeline")

    if story.narrative.conclusion:
        toc_items.append("Conclusion")

    toc_items.extend(["Sources", "Metadata"])

    for i, item in enumerate(toc_items, 1):
        # Create anchor link
        anchor = item.lower().replace(" ", "-").replace(":", "")
        lines.append(f"{i}. [{item}](#{anchor})")

    lines.append("")

    # === Executive Summary ===
    lines.append("## Executive Summary")
    lines.append("")

    if story.narrative.executive_summary:
        lines.append(story.narrative.executive_summary)
    elif story.extraction.commits:
        # Generate basic summary from extraction data
        lines.append(_generate_basic_summary(story))
    else:
        lines.append("*No summary available.*")

    lines.append("")

    # === Narrative Acts ===
    if story.narrative.acts:
        for act in story.narrative.acts:
            lines.append(f"## Act {act.number}: {act.title}")
            lines.append("")

            if act.date_range:
                lines.append(f"*{act.date_range}*")
                lines.append("")

            if act.content:
                lines.append(act.content)
            else:
                lines.append("*No narrative content available.*")

            lines.append("")

            # Evidence citations
            if act.evidence:
                lines.append("**Evidence:**")
                lines.append("")
                for evidence in act.evidence[:10]:  # Limit to 10
                    lines.append(f"- `{evidence}`")
                lines.append("")

    # === Themes ===
    if story.narrative.themes:
        lines.append("## Themes")
        lines.append("")

        for theme in story.narrative.themes:
            confidence_badge = _confidence_badge(theme.confidence)
            lines.append(f"### {theme.name} {confidence_badge}")
            lines.append("")

            if theme.description:
                lines.append(theme.description)
                lines.append("")

            if theme.examples:
                lines.append("**Examples:**")
                lines.append("")
                for example in theme.examples[:5]:  # Limit to 5
                    lines.append(f"- {example}")
                lines.append("")

    # === Key Contributors ===
    if story.extraction.contributors:
        lines.append("## Key Contributors")
        lines.append("")

        # Sort by commit count
        sorted_contributors = sorted(
            story.extraction.contributors,
            key=lambda c: c.commit_count,
            reverse=True,
        )[:10]  # Top 10

        lines.append("| Contributor | Commits | First Commit | Last Commit |")
        lines.append("|-------------|---------|--------------|-------------|")

        for contributor in sorted_contributors:
            first = (
                contributor.first_commit.strftime("%Y-%m-%d")
                if contributor.first_commit
                else "N/A"
            )
            last = (
                contributor.last_commit.strftime("%Y-%m-%d")
                if contributor.last_commit
                else "N/A"
            )
            lines.append(
                f"| {contributor.name} | {contributor.commit_count} | {first} | {last} |"
            )

        lines.append("")

    # === Technology Stack ===
    if story.analysis.tech_stack:
        lines.append("## Technology Stack")
        lines.append("")

        # Group by category
        by_category: dict[str, list[str]] = {}
        for tech in story.analysis.tech_stack:
            category = tech.category or "Other"
            if category not in by_category:
                by_category[category] = []

            tech_name = tech.name
            if tech.version:
                tech_name += f" ({tech.version})"

            by_category[category].append(tech_name)

        for category, techs in sorted(by_category.items()):
            lines.append(f"### {category.title()}")
            lines.append("")
            for tech in sorted(techs):
                lines.append(f"- {tech}")
            lines.append("")

    elif story.analysis.language_distribution:
        lines.append("## Technology Stack")
        lines.append("")
        lines.append("### Languages")
        lines.append("")

        # Sort by line count
        sorted_langs = sorted(
            story.analysis.language_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        total_lines = sum(story.analysis.language_distribution.values())

        for lang, count in sorted_langs:
            percentage = (count / total_lines * 100) if total_lines > 0 else 0
            lines.append(f"- **{lang}**: {count:,} lines ({percentage:.1f}%)")

        lines.append("")

    # === Architectural Patterns ===
    if story.analysis.architectural_patterns:
        lines.append("## Architectural Patterns")
        lines.append("")

        for pattern in story.analysis.architectural_patterns:
            confidence_badge = _confidence_badge(pattern.confidence)
            lines.append(f"### {pattern.name} {confidence_badge}")
            lines.append("")

            if pattern.description:
                lines.append(pattern.description)
                lines.append("")

            if pattern.evidence:
                lines.append("**Evidence:**")
                lines.append("")
                for evidence in pattern.evidence[:5]:
                    lines.append(f"- {evidence}")
                lines.append("")

    # === Roads Not Taken ===
    if story.narrative.roads_not_taken:
        lines.append("## Roads Not Taken")
        lines.append("")

        for road in story.narrative.roads_not_taken:
            lines.append(f"### {road.title}")
            lines.append("")

            if road.description:
                lines.append(road.description)
                lines.append("")

            if road.evidence:
                lines.append("**Evidence:**")
                lines.append("")
                for evidence in road.evidence[:3]:
                    lines.append(f"- {evidence}")
                lines.append("")

    # === Timeline ===
    if story.visualization.timeline_events:
        lines.append("## Timeline")
        lines.append("")
        lines.append("Key events in the project's history:")
        lines.append("")

        # Sort by date
        sorted_events = sorted(
            story.visualization.timeline_events,
            key=lambda e: e.date,
        )

        # Filter to top N by importance
        important_events = sorted(
            sorted_events,
            key=lambda e: e.importance,
            reverse=True,
        )[:20]

        # Re-sort by date
        important_events.sort(key=lambda e: e.date)

        for event in important_events:
            date_str = event.date.strftime("%Y-%m-%d")
            importance_stars = "⭐" * min(int(event.importance * 5), 5)
            lines.append(f"- **{date_str}** {importance_stars} - {event.title}")

            if event.description:
                lines.append(f"  - {event.description}")

        lines.append("")

    # === Conclusion ===
    if story.narrative.conclusion:
        lines.append("## Conclusion")
        lines.append("")
        lines.append(story.narrative.conclusion)
        lines.append("")

    # === Confidence Assessment ===
    if story.narrative.confidence_assessment:
        lines.append("### Confidence Assessment")
        lines.append("")
        lines.append(story.narrative.confidence_assessment)
        lines.append("")

    # === Sources ===
    lines.append("## Sources")
    lines.append("")

    if story.extraction.commits:
        lines.append(f"- **Commits analyzed**: {len(story.extraction.commits):,}")

    if story.extraction.issues:
        lines.append(f"- **Issues tracked**: {len(story.extraction.issues)}")

    if story.extraction.pull_requests:
        lines.append(
            f"- **Pull requests reviewed**: {len(story.extraction.pull_requests)}"
        )

    if story.extraction.docs:
        lines.append(f"- **Documentation files**: {len(story.extraction.docs)}")
        for doc in story.extraction.docs[:5]:
            lines.append(f"  - {doc.path}")

    lines.append("")

    # === Metadata Footer ===
    lines.append("## Metadata")
    lines.append("")
    lines.append(
        f"- **Generator**: {story.metadata.generator} v{story.metadata.version}"
    )

    if story.metadata.llm_model:
        lines.append(f"- **LLM Model**: {story.metadata.llm_model}")

    lines.append(
        f"- **Semantic Analysis**: {'Yes' if story.metadata.has_semantic_analysis else 'No'}"
    )
    lines.append(
        f"- **LLM Narrative**: {'Yes' if story.metadata.has_llm_narrative else 'No'}"
    )

    if story.metadata.generation_time_seconds > 0:
        lines.append(
            f"- **Generation Time**: {story.metadata.generation_time_seconds:.2f}s"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by {story.metadata.generator}*")
    lines.append("")

    # Join and write
    markdown = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    return markdown


def _confidence_badge(confidence: ConfidenceLevel) -> str:
    """Generate confidence badge for markdown.

    Args:
        confidence: Confidence level

    Returns:
        Badge string
    """
    badges = {
        ConfidenceLevel.HIGH: "🟢",
        ConfidenceLevel.MEDIUM: "🟡",
        ConfidenceLevel.LOW: "🔴",
    }
    return badges.get(confidence, "⚪")


def _generate_basic_summary(story: StoryIndex) -> str:
    """Generate basic summary from extraction data when no LLM narrative exists.

    Args:
        story: StoryIndex with extraction data

    Returns:
        Basic summary text
    """
    lines: list[str] = []

    # Project overview
    project_name = story.metadata.project_name or "This project"
    lines.append(
        f"{project_name} has evolved through {story.metadata.total_commits:,} commits "
        f"by {story.metadata.total_contributors} contributors."
    )

    # Timeframe
    if story.metadata.date_range_start and story.metadata.date_range_end:
        start = story.metadata.date_range_start.strftime("%B %Y")
        end = story.metadata.date_range_end.strftime("%B %Y")
        lines.append(f"Development spans from {start} to {end}.")

    # Top contributors
    if story.extraction.contributors:
        top_contributors = sorted(
            story.extraction.contributors,
            key=lambda c: c.commit_count,
            reverse=True,
        )[:3]

        names = ", ".join(c.name for c in top_contributors)
        lines.append(f"Key contributors include {names}.")

    # Activity summary
    if story.extraction.commits:
        total_insertions = sum(c.insertions for c in story.extraction.commits)
        total_deletions = sum(c.deletions for c in story.extraction.commits)

        lines.append(
            f"The codebase has seen {total_insertions:,} insertions and "
            f"{total_deletions:,} deletions across {story.metadata.total_files:,} files."
        )

    return " ".join(lines)
