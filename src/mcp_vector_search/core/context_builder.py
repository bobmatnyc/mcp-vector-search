"""Context builder for enriched embeddings.

Prepends metadata header (file path, language, class/function context, imports,
docstring) to chunk content before embedding.  The stored ``content`` field in
the database is left completely untouched — only the text sent to the embedding
model is enriched.

Research shows that contextual metadata prepending improves semantic retrieval
quality by 35–49% by giving the model more signal about where and what a chunk
is, rather than treating every snippet as context-free code.

Token budget note
-----------------
MiniLM has a hard 256-token limit.  A compact pipe-separated header like::

    File: src/foo.py | Lang: python | Class: MyClass | Fn: my_method | Uses: os, re | Desc: Does X

is typically 15–25 tokens, leaving 230–240 tokens for the original code — well
within the budget.  We therefore use ``|`` separators (not newlines) for the
header and insert a single ``---\\n`` divider before the code.
"""

from __future__ import annotations

from typing import Any


def build_contextual_text(chunk: Any) -> str:
    """Return context-enriched text for embedding.

    The function accepts both ``CodeChunk`` dataclass instances and plain
    ``dict`` objects (as produced by the two-phase pipeline in ``indexer.py``).

    Args:
        chunk: Either a :class:`~mcp_vector_search.core.models.CodeChunk`
            dataclass or a plain ``dict`` produced during the indexing pipeline.
            In either case the following fields are consulted (missing/falsy
            fields are silently skipped):
            ``file_path``, ``language``, ``class_name``, ``function_name``,
            ``imports``, ``docstring``, ``content``.

    Returns:
        A string of the form::

            File: ... | Lang: ... | Class: ... | Fn: ... | Uses: ... | Desc: ...
            ---
            <original content>

        If no metadata is available the original content is returned unchanged.
    """
    # Support both dict and dataclass/object access patterns
    if isinstance(chunk, dict):
        file_path = chunk.get("file_path") or ""
        language = chunk.get("language") or ""
        class_name = chunk.get("class_name") or ""
        function_name = chunk.get("function_name") or ""
        imports_raw = chunk.get("imports") or []
        docstring = chunk.get("docstring") or ""
        content = chunk.get("content") or ""
    else:
        # CodeChunk dataclass (or any object with attribute access)
        file_path = str(getattr(chunk, "file_path", "") or "")
        language = getattr(chunk, "language", "") or ""
        class_name = getattr(chunk, "class_name", "") or ""
        function_name = getattr(chunk, "function_name", "") or ""
        imports_raw = getattr(chunk, "imports", []) or []
        docstring = getattr(chunk, "docstring", "") or ""
        content = getattr(chunk, "content", "") or ""

    parts: list[str] = []

    # File path — use just the filename component to save tokens.
    # A short relative path (e.g. "src/foo.py") is already compact.
    # Skip degenerate paths like ".", "", or "/".
    if file_path:
        # Normalise Path objects to string
        fp_str = str(file_path)
        # Skip paths that contain no useful information
        if fp_str not in (".", "/", ""):
            # If the path contains many segments, only keep the last two for brevity
            segments = fp_str.replace("\\", "/").split("/")
            short_path = "/".join(segments[-2:]) if len(segments) > 2 else fp_str
            parts.append(f"File: {short_path}")

    # Language (skip generic/unknown values)
    if language and language not in ("text", "unknown", ""):
        parts.append(f"Lang: {language}")

    # Class context (helps distinguish methods from standalone functions)
    if class_name:
        parts.append(f"Class: {class_name}")

    # Function / method name
    if function_name:
        parts.append(f"Fn: {function_name}")

    # Import sources — a comma-separated summary of what the chunk depends on.
    # ``imports_raw`` can be:
    #   * list[dict] with 'source' key  (CodeChunk after parsing)
    #   * list[str] of JSON strings like '{"source": "os", "statement": "import os"}'
    #   * list[str] of plain module names
    #   * A single JSON-encoded string (legacy storage format from chunks.lance)
    import json as _json

    sources: list[str] = []
    if isinstance(imports_raw, str):
        # Stored as a single JSON-encoded list (legacy lancedb_backend format)
        try:
            decoded = _json.loads(imports_raw)
            if isinstance(decoded, list):
                imports_raw = decoded
            else:
                imports_raw = []
        except (ValueError, TypeError):
            imports_raw = []

    for imp in imports_raw:
        if isinstance(imp, dict):
            src = imp.get("source", "")
            if src:
                sources.append(src)
        elif isinstance(imp, str):
            # May be a JSON-encoded dict or a plain module name
            stripped = imp.strip()
            if stripped.startswith("{"):
                try:
                    decoded_imp = _json.loads(stripped)
                    src = decoded_imp.get("source", "")
                    if src:
                        sources.append(src)
                    continue
                except (ValueError, TypeError):
                    pass
            # Plain module name
            if stripped:
                sources.append(stripped)

    if sources:
        # Cap at 10 sources to keep the header compact
        sources_str = ", ".join(sources[:10])
        parts.append(f"Uses: {sources_str}")

    # Docstring summary — truncated to 200 chars to stay within token budget
    if docstring:
        doc_summary = docstring.strip()
        if len(doc_summary) > 200:
            doc_summary = doc_summary[:200].rstrip() + "..."
        if doc_summary:
            parts.append(f"Desc: {doc_summary}")

    if parts:
        header = " | ".join(parts)
        return f"{header}\n---\n{content}"

    return content
