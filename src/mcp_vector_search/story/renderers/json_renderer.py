"""JSON renderer for StoryIndex."""

from __future__ import annotations

from pathlib import Path

from ..models import StoryIndex


def render_json(story: StoryIndex, output_path: Path | None = None) -> str:
    """Render StoryIndex as formatted JSON.

    Args:
        story: The StoryIndex to render
        output_path: If provided, write to this file

    Returns:
        JSON string
    """
    json_str = story.model_dump_json(indent=2)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")

    return json_str
