"""Unit tests for context_builder.build_contextual_text()."""

import json
from pathlib import Path

from mcp_vector_search.core.context_builder import build_contextual_text
from mcp_vector_search.core.models import CodeChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(**kwargs) -> CodeChunk:
    """Create a minimal CodeChunk with sensible defaults."""
    defaults = {
        "content": "def foo(): pass",
        "file_path": Path("src/utils.py"),
        "start_line": 1,
        "end_line": 1,
        "language": "python",
    }
    defaults.update(kwargs)
    return CodeChunk(**defaults)


# ---------------------------------------------------------------------------
# Tests: CodeChunk dataclass input
# ---------------------------------------------------------------------------


class TestBuildContextualTextCodeChunk:
    """Tests with CodeChunk dataclass input."""

    def test_returns_original_content_unchanged_with_no_metadata(self):
        """Content-only chunk (no metadata) returns content unchanged."""
        chunk = CodeChunk(
            content="x = 1",
            file_path=Path(""),  # empty path
            start_line=1,
            end_line=1,
            language="",  # blank language
        )
        result = build_contextual_text(chunk)
        # No metadata parts should be produced — result equals content
        assert result == "x = 1"

    def test_prepends_file_lang_fn_header(self):
        """Function in a file gets a compact header prepended."""
        chunk = _make_chunk(
            content="def add(a, b):\n    return a + b",
            file_path=Path("src/math_utils.py"),
            language="python",
            function_name="add",
        )
        result = build_contextual_text(chunk)
        assert result.startswith("File:")
        assert "Lang: python" in result
        assert "Fn: add" in result
        assert "---" in result
        # Original content appears after separator
        assert "def add(a, b):" in result

    def test_class_context_included(self):
        """Method chunk includes Class: field in header."""
        chunk = _make_chunk(
            content="def compute(self): pass",
            class_name="Calculator",
            function_name="compute",
        )
        result = build_contextual_text(chunk)
        assert "Class: Calculator" in result
        assert "Fn: compute" in result

    def test_docstring_truncated_at_200_chars(self):
        """Long docstrings are truncated to 200 chars."""
        long_doc = "A" * 300
        chunk = _make_chunk(docstring=long_doc)
        result = build_contextual_text(chunk)
        # The Desc field should appear but be truncated
        assert "Desc:" in result
        assert "A" * 201 not in result  # Truncated before 201

    def test_short_docstring_included_verbatim(self):
        """Short docstrings appear verbatim in the header."""
        chunk = _make_chunk(docstring="Compute the sum of two numbers.")
        result = build_contextual_text(chunk)
        assert "Desc: Compute the sum of two numbers." in result

    def test_imports_list_of_dicts_extracts_sources(self):
        """Imports as list[dict] with 'source' key are summarized."""
        chunk = _make_chunk(
            imports=[
                {"source": "os", "statement": "import os"},
                {"source": "re", "statement": "import re"},
            ]
        )
        result = build_contextual_text(chunk)
        assert "Uses: os, re" in result

    def test_imports_capped_at_10_sources(self):
        """More than 10 import sources are capped at 10."""
        imports = [
            {"source": f"mod{i}", "statement": f"import mod{i}"} for i in range(15)
        ]
        chunk = _make_chunk(imports=imports)
        result = build_contextual_text(chunk)
        # mod10 through mod14 should not appear
        assert "mod14" not in result
        assert "mod0" in result

    def test_text_language_skipped(self):
        """Language 'text' is not included in the header."""
        chunk = _make_chunk(language="text")
        result = build_contextual_text(chunk)
        assert "Lang:" not in result

    def test_file_path_shortened_to_two_segments(self):
        """Deep paths are shortened to last two segments."""
        chunk = _make_chunk(
            file_path=Path("a/b/c/d/e/src/utils.py"),
        )
        result = build_contextual_text(chunk)
        assert "File: src/utils.py" in result
        # Should not contain the long prefix
        assert "a/b/c" not in result

    def test_original_content_not_modified(self):
        """The original chunk.content attribute is not mutated."""
        chunk = _make_chunk(
            content="original code",
            function_name="foo",
        )
        _ = build_contextual_text(chunk)
        assert chunk.content == "original code"

    def test_separator_newline_between_header_and_code(self):
        """Header and code are separated by '---\\n'."""
        chunk = _make_chunk(function_name="bar")
        result = build_contextual_text(chunk)
        assert "\n---\n" in result


# ---------------------------------------------------------------------------
# Tests: dict input (two-phase pipeline format)
# ---------------------------------------------------------------------------


class TestBuildContextualTextDict:
    """Tests with plain dict input (as produced by indexer.py pipeline)."""

    def test_basic_dict_with_all_fields(self):
        """Dict with all fields produces enriched text."""
        chunk_dict = {
            "chunk_id": "abc123",
            "file_path": "src/core/engine.py",
            "language": "python",
            "class_name": "Engine",
            "function_name": "run",
            "docstring": "Run the engine.",
            "imports": [
                json.dumps({"source": "asyncio", "statement": "import asyncio"})
            ],
            "content": "def run(self): ...",
        }
        result = build_contextual_text(chunk_dict)
        assert "File:" in result
        assert "Lang: python" in result
        assert "Class: Engine" in result
        assert "Fn: run" in result
        assert "Uses: asyncio" in result
        assert "Desc: Run the engine." in result
        assert "def run(self): ..." in result

    def test_dict_imports_as_json_encoded_strings(self):
        """Imports stored as JSON-encoded strings (indexer.py format) are parsed."""
        chunk_dict = {
            "file_path": "app.py",
            "language": "python",
            "content": "x = 1",
            "imports": [
                json.dumps({"source": "os", "statement": "import os"}),
                json.dumps({"source": "sys", "statement": "import sys"}),
            ],
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, sys" in result

    def test_dict_imports_as_plain_strings(self):
        """Imports as plain module-name strings are included as-is."""
        chunk_dict = {
            "file_path": "app.py",
            "language": "python",
            "content": "pass",
            "imports": ["os", "re", "sys"],
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, re, sys" in result

    def test_dict_missing_optional_fields_graceful(self):
        """Dict missing optional fields (class_name, etc.) does not raise."""
        chunk_dict = {
            "file_path": "minimal.py",
            "content": "pass",
        }
        result = build_contextual_text(chunk_dict)
        # Should at least include the file
        assert "File: minimal.py" in result
        assert "pass" in result

    def test_dict_empty_content_returns_empty_with_header(self):
        """Empty content still produces a header when metadata is present."""
        chunk_dict = {
            "file_path": "empty.py",
            "language": "python",
            "content": "",
        }
        result = build_contextual_text(chunk_dict)
        assert "File: empty.py" in result
        assert "---" in result

    def test_dict_no_metadata_returns_content_only(self):
        """Dict with no meaningful metadata returns content unchanged."""
        chunk_dict = {"content": "raw code"}
        result = build_contextual_text(chunk_dict)
        assert result == "raw code"

    def test_dict_imports_as_single_json_list_string(self):
        """Imports stored as a single JSON-encoded list string (lancedb legacy) are parsed."""
        imports_json = json.dumps(
            [
                {"source": "os", "statement": "import os"},
                {"source": "re", "statement": "import re"},
            ]
        )
        chunk_dict = {
            "file_path": "foo.py",
            "language": "python",
            "content": "pass",
            "imports": imports_json,  # Single JSON string (legacy format)
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, re" in result


# ---------------------------------------------------------------------------
# Tests: Edge cases and token-budget constraints
# ---------------------------------------------------------------------------


class TestContextBuilderEdgeCases:
    """Edge cases and token budget constraints."""

    def test_none_class_name_not_included(self):
        """None class_name does not produce 'Class: None' in header."""
        chunk = _make_chunk(class_name=None)
        result = build_contextual_text(chunk)
        assert "Class:" not in result

    def test_none_function_name_not_included(self):
        """None function_name does not produce 'Fn: None' in header."""
        chunk = _make_chunk(function_name=None)
        result = build_contextual_text(chunk)
        assert "Fn:" not in result

    def test_empty_imports_list_no_uses_field(self):
        """Empty imports list does not produce 'Uses:' in header."""
        chunk = _make_chunk(imports=[])
        result = build_contextual_text(chunk)
        assert "Uses:" not in result

    def test_header_uses_pipe_separator(self):
        """Header parts are separated by ' | ' (compact, not newlines)."""
        chunk = _make_chunk(
            language="python",
            class_name="Foo",
            function_name="bar",
        )
        result = build_contextual_text(chunk)
        header_line = result.split("\n---\n")[0]
        assert " | " in header_line
        # Header should be a single line (no internal newlines)
        assert "\n" not in header_line

    def test_full_contextual_text_format(self):
        """Comprehensive test of the exact output format."""
        chunk = _make_chunk(
            content="return self.value",
            file_path=Path("src/models/user.py"),
            language="python",
            class_name="User",
            function_name="get_value",
            docstring="Return the user value.",
            imports=[{"source": "uuid", "statement": "import uuid"}],
        )
        result = build_contextual_text(chunk)
        expected_header = (
            "File: models/user.py | Lang: python | Class: User | Fn: get_value | "
            "Uses: uuid | Desc: Return the user value."
        )
        assert result == f"{expected_header}\n---\nreturn self.value"
