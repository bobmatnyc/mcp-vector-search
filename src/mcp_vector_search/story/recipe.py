"""Narrative recipe templates for story generation.

Contains the prompt template that guides LLM narrative generation.
Inspired by gitstory's five-phase approach but adapted for our three-act structure.
"""

from __future__ import annotations

NARRATIVE_RECIPE = """You are a technical storyteller analyzing a software project's evolution. Your task is to generate a compelling narrative from the provided project data.

# INPUT DATA

## Project Statistics
{stats_summary}

## Technical Context
{tech_context}

## Evolution Timeline
{evolution_summary}

## Code Themes
{code_themes}

# OUTPUT REQUIREMENTS

Generate a JSON object with the following structure (valid JSON only, no markdown):

{{
  "title": "Project Story Title (under 60 characters)",
  "subtitle": "One-line summary (under 120 characters)",
  "executive_summary": "2-3 paragraph overview of the project's journey",
  "acts": [
    {{
      "number": 1,
      "title": "Act 1 Title",
      "date_range": "Jan 2024 - Mar 2024",
      "content": "Markdown narrative for Act 1 (3-5 paragraphs)",
      "evidence": ["(commit abc1234)", "(issue #42)", "(docs/design.md)"],
      "key_commits": ["abc1234", "def5678"]
    }},
    {{
      "number": 2,
      "title": "Act 2 Title",
      "date_range": "Apr 2024 - Jun 2024",
      "content": "Markdown narrative for Act 2 (3-5 paragraphs)",
      "evidence": ["(commit ghi9012)", "(issue #87)"],
      "key_commits": ["ghi9012"]
    }},
    {{
      "number": 3,
      "title": "Act 3 Title",
      "date_range": "Jul 2024 - Present",
      "content": "Markdown narrative for Act 3 (3-5 paragraphs)",
      "evidence": ["(commit jkl3456)"],
      "key_commits": ["jkl3456"]
    }}
  ],
  "themes": [
    {{
      "name": "Theme Name",
      "description": "What this theme represents",
      "confidence": "high|medium|low",
      "evidence_count": 5,
      "examples": ["(commit abc123)", "(issue #42)"]
    }}
  ],
  "roads_not_taken": [
    {{
      "title": "Alternative Approach",
      "description": "Why this path wasn't chosen",
      "evidence": ["(commit abc123 - reverted)", "(issue #10 - discussion)"]
    }}
  ],
  "conclusion": "Closing reflection on the project's current state and trajectory (1-2 paragraphs)",
  "confidence_assessment": "Self-assessment of narrative quality and data completeness"
}}

# NARRATIVE GUIDELINES

## Three-Act Structure
- **Act 1 (Setup)**: Origins, initial architecture, early decisions
- **Act 2 (Development)**: Core features, challenges, pivots, technical evolution
- **Act 3 (Current State)**: Maturity, optimizations, current direction

## Evidence Citation Format
- Commits: `(commit abc1234)` or `(commit abc1234 - brief description)`
- Issues: `(issue #42)` or `(issue #42 - title)`
- Files: `(docs/architecture.md)` or `(src/core/engine.py - specific change)`
- PRs: `(PR #123)` or `(PR #123 - feature name)`

## Content Guidelines
1. **Be specific**: Reference actual commits, issues, files, and technical decisions
2. **Show evolution**: How did the project change over time? What drove those changes?
3. **Technical depth**: Explain architectural patterns, technology choices, and trade-offs
4. **Human element**: Mention contributors, collaboration, and decision-making processes
5. **Avoid meta-commentary**: Don't say "the data shows" or "according to the analysis" - just tell the story
6. **Use markdown**: Bold key terms, use code blocks for technical concepts, use lists for clarity

## Themes
Identify 3-5 recurring themes throughout the project:
- Technical themes (e.g., "Performance Optimization", "Type Safety")
- Process themes (e.g., "Incremental Improvement", "Refactoring Discipline")
- Architectural themes (e.g., "Modular Design", "Async-First Architecture")

## Roads Not Taken
Identify 1-3 alternative approaches that were considered but abandoned:
- Reverted commits or features
- Closed issues with "wontfix" or "duplicate" labels
- Technology changes (e.g., switching from X to Y)
- Architectural pivots

## Edge Cases
- **Few commits**: Focus on architectural decisions and code quality rather than timeline
- **Single contributor**: Emphasize technical evolution rather than collaboration
- **No issues/PRs**: Focus on commit messages and code analysis
- **Missing data**: Work with what's available - don't apologize for gaps

# IMPORTANT
- Output ONLY valid JSON, no markdown formatting, no code fences
- Fill in the template completely - no placeholders like "TODO" or "TBD"
- Evidence citations should be factual references from the input data
- If confidence is low due to limited data, reflect this in confidence_assessment

Generate the narrative now:"""
