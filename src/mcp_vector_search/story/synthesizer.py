"""LLM-powered narrative synthesizer.

Uses LLMClient to generate compelling narratives from extraction + analysis data.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..core.llm_client import LLMClient
from .models import (
    ConfidenceLevel,
    NarrativeAct,
    RoadNotTaken,
    StoryAnalysis,
    StoryExtraction,
    StoryNarrative,
    Theme,
)
from .recipe import NARRATIVE_RECIPE

logger = logging.getLogger(__name__)


class StorySynthesizer:
    """LLM-powered narrative synthesizer.

    Takes raw extraction data and semantic analysis, then uses an LLM
    to generate a compelling three-act narrative with themes and insights.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize synthesizer with LLM client.

        Args:
            llm_client: Configured LLM client (OpenAI, OpenRouter, or Bedrock)
        """
        self.llm = llm_client

    async def synthesize(
        self, extraction: StoryExtraction, analysis: StoryAnalysis
    ) -> StoryNarrative:
        """Generate full narrative from extraction + analysis data.

        Args:
            extraction: Raw extracted data from git/GitHub
            analysis: Semantic analysis from vector search

        Returns:
            StoryNarrative with LLM-generated content
        """
        try:
            logger.info("Building context for LLM narrative generation")

            # Build context string
            context = self._build_context(extraction, analysis)

            # Format recipe with context
            prompt = NARRATIVE_RECIPE.format(
                stats_summary=context["stats_summary"],
                tech_context=context["tech_context"],
                evolution_summary=context["evolution_summary"],
                code_themes=context["code_themes"],
            )

            logger.info(f"Sending prompt to LLM ({self.llm.provider})")

            # Send to LLM
            messages = [{"role": "user", "content": prompt}]

            response = await self.llm._chat_completion(messages)

            # Extract content from response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            if not content:
                raise ValueError("Empty response from LLM")

            logger.info("Parsing LLM response into narrative structure")

            # Parse response into StoryNarrative
            narrative = self._parse_narrative_response(content)

            logger.info(
                f"Narrative generated: {len(narrative.acts)} acts, "
                f"{len(narrative.themes)} themes"
            )

            return narrative

        except Exception as e:
            logger.error(f"Narrative synthesis failed: {e}", exc_info=True)

            # Return empty narrative on failure
            return StoryNarrative(
                title="Story Generation Failed",
                subtitle="Unable to generate narrative due to error",
                executive_summary=f"Error during synthesis: {str(e)}",
                acts=[],
                themes=[],
                roads_not_taken=[],
                conclusion="",
                confidence_assessment="No narrative generated due to synthesis error",
            )

    def _build_context(
        self, extraction: StoryExtraction, analysis: StoryAnalysis
    ) -> dict[str, str]:
        """Build context string for LLM.

        Args:
            extraction: Raw extracted data
            analysis: Semantic analysis results

        Returns:
            Dictionary with formatted context sections
        """
        # Stats summary
        stats_parts = []
        stats_parts.append(f"Total commits: {len(extraction.commits)}")
        stats_parts.append(f"Contributors: {len(extraction.contributors)}")

        if extraction.issues:
            stats_parts.append(f"Issues: {len(extraction.issues)}")
        if extraction.pull_requests:
            stats_parts.append(f"Pull requests: {len(extraction.pull_requests)}")

        # Date range
        if extraction.commits:
            sorted_commits = sorted(extraction.commits, key=lambda c: c.date)
            first_commit = sorted_commits[0].date.strftime("%Y-%m-%d")
            last_commit = sorted_commits[-1].date.strftime("%Y-%m-%d")
            stats_parts.append(f"Date range: {first_commit} to {last_commit}")

        stats_summary = "\n".join(f"- {part}" for part in stats_parts)

        # Tech context
        tech_parts = []

        if analysis.tech_stack:
            tech_parts.append("Technologies:")
            for tech in analysis.tech_stack[:10]:  # Top 10
                evidence = ", ".join(tech.evidence[:2])  # First 2 pieces of evidence
                tech_parts.append(f"  - {tech.name} ({tech.category}): {evidence}")

        if analysis.language_distribution:
            tech_parts.append("\nLanguage Distribution:")
            sorted_langs = sorted(
                analysis.language_distribution.items(), key=lambda x: x[1], reverse=True
            )
            for lang, count in sorted_langs[:5]:  # Top 5
                tech_parts.append(f"  - {lang}: {count} chunks")

        if analysis.architectural_patterns:
            tech_parts.append("\nArchitectural Patterns:")
            for pattern in analysis.architectural_patterns:
                tech_parts.append(
                    f"  - {pattern.name} (confidence: {pattern.confidence})"
                )

        tech_context = (
            "\n".join(tech_parts) if tech_parts else "No technical analysis available"
        )

        # Evolution summary
        evolution_parts = []

        if analysis.evolution_phases:
            evolution_parts.append("Evolution Phases:")
            for phase in analysis.evolution_phases:
                date_range = (
                    f"{phase.date_start.strftime('%Y-%m-%d')} to "
                    f"{phase.date_end.strftime('%Y-%m-%d')}"
                )
                evolution_parts.append(f"\n{phase.name} ({date_range}):")
                evolution_parts.append(f"  - {phase.commit_count} commits")

                if phase.dominant_areas:
                    areas = ", ".join(phase.dominant_areas[:3])
                    evolution_parts.append(f"  - Focus areas: {areas}")

                if phase.key_commits:
                    commits_str = ", ".join(phase.key_commits[:3])
                    evolution_parts.append(f"  - Key commits: {commits_str}")

        # Add recent commits summary
        if extraction.commits:
            evolution_parts.append("\nRecent Commits:")
            sorted_commits = sorted(
                extraction.commits, key=lambda c: c.date, reverse=True
            )

            for commit in sorted_commits[:10]:  # Last 10 commits
                date_str = commit.date.strftime("%Y-%m-%d")
                evolution_parts.append(
                    f"  - {commit.short_hash} ({date_str}): {commit.message[:60]}"
                )

        evolution_summary = (
            "\n".join(evolution_parts)
            if evolution_parts
            else "No evolution data available"
        )

        # Code themes
        theme_parts = []

        if analysis.clusters:
            theme_parts.append("Semantic Code Themes:")
            for cluster in analysis.clusters:
                theme_parts.append(
                    f"\n{cluster.name} (confidence: {cluster.confidence}):"
                )
                theme_parts.append(f"  - Query: {cluster.query}")
                theme_parts.append(f"  - Files: {len(cluster.files)}")

                if cluster.files:
                    files_str = ", ".join(cluster.files[:3])
                    theme_parts.append(f"  - Examples: {files_str}")

                if cluster.code_snippets:
                    theme_parts.append("  - Code samples:")
                    for snippet in cluster.code_snippets[:2]:
                        theme_parts.append(f"    * {snippet}")

        code_themes = (
            "\n".join(theme_parts)
            if theme_parts
            else "No code theme analysis available"
        )

        return {
            "stats_summary": stats_summary,
            "tech_context": tech_context,
            "evolution_summary": evolution_summary,
            "code_themes": code_themes,
        }

    def _parse_narrative_response(self, response: str) -> StoryNarrative:
        """Parse LLM response into StoryNarrative model.

        Tries JSON parsing first, then falls back to markdown parsing.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed StoryNarrative
        """
        # Try JSON parsing first (most reliable)
        try:
            # Clean response (remove markdown code fences if present)
            cleaned = response.strip()

            # Remove markdown code fence if present
            if cleaned.startswith("```"):
                # Find end of first line (language identifier)
                first_newline = cleaned.find("\n")
                if first_newline > 0:
                    cleaned = cleaned[first_newline + 1 :]

                # Remove closing fence
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

            cleaned = cleaned.strip()

            # Parse as JSON
            data = json.loads(cleaned)

            return self._parse_narrative_dict(data)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting markdown parsing.")

            # Fallback to markdown parsing
            return self._parse_narrative_markdown(response)

    def _parse_narrative_dict(self, data: dict[str, Any]) -> StoryNarrative:
        """Parse narrative from JSON dict.

        Args:
            data: Parsed JSON dictionary

        Returns:
            StoryNarrative instance
        """
        # Parse acts
        acts = []
        for act_data in data.get("acts", []):
            act = NarrativeAct(
                number=act_data.get("number", 1),
                title=act_data.get("title", ""),
                date_range=act_data.get("date_range", ""),
                content=act_data.get("content", ""),
                evidence=act_data.get("evidence", []),
                key_commits=act_data.get("key_commits", []),
            )
            acts.append(act)

        # Parse themes
        themes = []
        for theme_data in data.get("themes", []):
            # Map confidence string to enum
            confidence_str = theme_data.get("confidence", "medium").lower()
            try:
                confidence = ConfidenceLevel(confidence_str)
            except ValueError:
                confidence = ConfidenceLevel.MEDIUM

            theme = Theme(
                name=theme_data.get("name", ""),
                description=theme_data.get("description", ""),
                confidence=confidence,
                evidence_count=theme_data.get("evidence_count", 0),
                examples=theme_data.get("examples", []),
            )
            themes.append(theme)

        # Parse roads not taken
        roads = []
        for road_data in data.get("roads_not_taken", []):
            road = RoadNotTaken(
                title=road_data.get("title", ""),
                description=road_data.get("description", ""),
                evidence=road_data.get("evidence", []),
            )
            roads.append(road)

        return StoryNarrative(
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            executive_summary=data.get("executive_summary", ""),
            acts=acts,
            themes=themes,
            roads_not_taken=roads,
            conclusion=data.get("conclusion", ""),
            confidence_assessment=data.get("confidence_assessment", ""),
        )

    def _parse_narrative_markdown(self, response: str) -> StoryNarrative:
        """Parse narrative from markdown-formatted response.

        Fallback parser for when LLM doesn't return JSON.

        Args:
            response: Raw markdown response

        Returns:
            Best-effort StoryNarrative
        """
        logger.info("Using markdown fallback parser")

        # Extract title (first # heading)
        title_match = re.search(r"^#\s+(.+)$", response, re.MULTILINE)
        title = title_match.group(1) if title_match else "Project Story"

        # Extract subtitle (first line after title or second # heading)
        subtitle_match = re.search(r"^##\s+(.+)$", response, re.MULTILINE)
        subtitle = subtitle_match.group(1) if subtitle_match else ""

        # Try to extract executive summary (text before first Act heading)
        act_pattern = r"##\s+Act\s+\d+"
        first_act_match = re.search(act_pattern, response, re.IGNORECASE)

        executive_summary = ""
        if first_act_match:
            # Get text between title/subtitle and first act
            intro_text = response[: first_act_match.start()].strip()
            # Remove title and subtitle
            intro_lines = intro_text.split("\n")
            summary_lines = [
                line
                for line in intro_lines
                if not line.startswith("#") and line.strip()
            ]
            executive_summary = "\n".join(summary_lines[:10])  # First 10 lines

        # Extract acts
        acts = self._extract_acts_from_markdown(response)

        # Create basic narrative
        return StoryNarrative(
            title=title,
            subtitle=subtitle,
            executive_summary=executive_summary or "Summary not available",
            acts=acts,
            themes=[],  # Difficult to extract from markdown
            roads_not_taken=[],  # Difficult to extract from markdown
            conclusion="",
            confidence_assessment="Parsed from markdown format (may be incomplete)",
        )

    def _extract_acts_from_markdown(self, text: str) -> list[NarrativeAct]:
        """Extract acts from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of NarrativeAct instances
        """
        acts = []

        # Find all Act headings
        act_pattern = r"##\s+Act\s+(\d+):?\s*(.+?)$"
        act_matches = list(re.finditer(act_pattern, text, re.IGNORECASE | re.MULTILINE))

        for i, match in enumerate(act_matches):
            act_num = int(match.group(1))
            act_title = match.group(2).strip()

            # Extract content between this act and next act (or end of text)
            start_pos = match.end()
            end_pos = (
                act_matches[i + 1].start() if i + 1 < len(act_matches) else len(text)
            )

            content = text[start_pos:end_pos].strip()

            # Try to extract date range from first line of content
            date_range = ""
            first_line = content.split("\n")[0] if content else ""
            date_match = re.search(r"\((.*?)\)", first_line)
            if date_match:
                date_range = date_match.group(1)

            act = NarrativeAct(
                number=act_num,
                title=act_title,
                date_range=date_range,
                content=content[:500],  # Limit content length
                evidence=[],
                key_commits=[],
            )
            acts.append(act)

        return acts
