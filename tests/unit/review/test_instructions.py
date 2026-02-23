"""Unit tests for InstructionsLoader."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.analysis.review.instructions import (
    DEFAULT_INSTRUCTIONS,
    InstructionsLoader,
    ReviewInstructions,
)


@pytest.fixture
def tmp_project_root():
    """Create a temporary project root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_format_for_prompt_has_defaults(tmp_project_root):
    """Test format_for_prompt returns non-empty string with defaults."""
    loader = InstructionsLoader(tmp_project_root)
    text = loader.format_for_prompt()

    assert isinstance(text, str)
    assert len(text) > 0
    # Should contain some default content
    assert "Language Standards" in text or "Standards" in text


def test_discover_and_load_returns_review_instructions(tmp_project_root):
    """Test discover_and_load returns ReviewInstructions object."""
    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.discover_and_load()

    assert isinstance(instructions, ReviewInstructions)
    assert isinstance(instructions.language_standards, list)
    assert isinstance(instructions.scope_standards, list)
    assert isinstance(instructions.style_preferences, list)
    assert isinstance(instructions.custom_review_focus, list)
    assert isinstance(instructions.sources_found, list)


def test_extract_from_editorconfig(tmp_project_root):
    """Test _extract_from_editorconfig parses indent_style and max_line_length."""
    editorconfig = tmp_project_root / ".editorconfig"
    editorconfig.write_text(
        "[*.py]\nindent_style = space\nindent_size = 4\nmax_line_length = 120\n"
    )

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_editorconfig()

    assert isinstance(rules, list)
    # Should extract max_line_length for Python
    assert any("120" in r for r in rules)
    # Should extract indent style for Python
    assert any("space" in r.lower() for r in rules)


def test_extract_from_editorconfig_missing_file(tmp_project_root):
    """Test _extract_from_editorconfig handles missing file gracefully."""
    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_editorconfig()

    # Should return empty list, not raise error
    assert rules == []


def test_extract_from_pyproject(tmp_project_root):
    """Test _extract_from_pyproject extracts ruff line-length."""
    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text('[tool.ruff]\nline-length = 88\ntarget-version = "py311"\n')

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_pyproject()

    assert isinstance(rules, list)
    assert any("88" in r for r in rules)
    assert any("py311" in r.lower() or "3.11" in r for r in rules)


def test_extract_from_pyproject_missing_file(tmp_project_root):
    """Test _extract_from_pyproject handles missing file gracefully."""
    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_pyproject()

    # Should return empty list, not raise error
    assert rules == []


def test_load_from_file_yaml(tmp_project_root):
    """Test loading instructions from YAML file works."""
    config_dir = tmp_project_root / ".mcp-vector-search"
    config_dir.mkdir()
    config_file = config_dir / "review-instructions.yaml"
    config_file.write_text(
        """language_standards:
  - "Use type hints"
  - "Follow PEP 8"

scope_standards:
  - "No hardcoded secrets"

style_preferences:
  - "Prefer composition"

custom_review_focus:
  - "Check error handling"
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.load()

    assert isinstance(instructions, dict)
    assert "language_standards" in instructions
    assert "Use type hints" in instructions["language_standards"]


def test_discover_and_load_with_yaml_config(tmp_project_root):
    """Test discover_and_load uses custom YAML config when present."""
    config_dir = tmp_project_root / ".mcp-vector-search"
    config_dir.mkdir()
    config_file = config_dir / "review-instructions.yaml"
    config_file.write_text(
        """language_standards:
  - "Custom rule 1"

scope_standards:
  - "Custom security rule"
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.discover_and_load()

    assert instructions.has_custom_config is True
    assert ".mcp-vector-search/review-instructions.yaml" in instructions.sources_found
    # Should include custom rules
    assert any("Custom rule 1" in r for r in instructions.language_standards)


def test_sources_found_tracks_files(tmp_project_root):
    """Test sources_found list tracks which files were used."""
    # Create editorconfig
    editorconfig = tmp_project_root / ".editorconfig"
    editorconfig.write_text("[*.py]\nmax_line_length = 100\n")

    # Create pyproject.toml
    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text("[tool.ruff]\nline-length = 88\n")

    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.discover_and_load()

    # Should track both sources
    assert ".editorconfig" in instructions.sources_found
    assert "pyproject.toml" in instructions.sources_found


def test_to_prompt_text_formatting(tmp_project_root):
    """Test ReviewInstructions.to_prompt_text() formats correctly."""
    instructions = ReviewInstructions(
        language_standards=["Rule 1", "Rule 2"],
        scope_standards=["Security rule"],
        style_preferences=["Style rule"],
        custom_review_focus=["Focus area"],
        sources_found=["pyproject.toml"],
        has_custom_config=False,
    )

    text = instructions.to_prompt_text()

    assert isinstance(text, str)
    assert "Language Standards" in text
    assert "Rule 1" in text
    assert "Security rule" in text
    assert "pyproject.toml" in text


def test_extract_from_markdown(tmp_project_root):
    """Test _extract_from_markdown extracts bullet points from markdown."""
    loader = InstructionsLoader(tmp_project_root)
    content = """# Code Style Guide

## Coding Standards

- Use snake_case for variables and functions
- Maximum line length is 88 characters
- All functions must have docstrings

## Other Section

Not relevant content here.
"""

    rules = loader._extract_from_markdown(content, "CONTRIBUTING.md")

    assert isinstance(rules, list)
    assert len(rules) > 0
    # Should extract rules from relevant sections
    assert any("snake_case" in r for r in rules)
    assert any("88" in r for r in rules)


def test_format_editorconfig_section_python(tmp_project_root):
    """Test _format_editorconfig_section formats Python rules."""
    loader = InstructionsLoader(tmp_project_root)

    config = {"indent_style": "space", "indent_size": "4", "max_line_length": "120"}

    rules = loader._format_editorconfig_section("*.py", config)

    assert isinstance(rules, list)
    assert any("Python files" in r for r in rules)
    assert any("120" in r for r in rules)


def test_format_editorconfig_section_generic(tmp_project_root):
    """Test _format_editorconfig_section skips generic rules without max_line_length."""
    loader = InstructionsLoader(tmp_project_root)

    config = {"indent_style": "space", "indent_size": "4"}

    rules = loader._format_editorconfig_section("*", config)

    # Should skip generic rules without specific constraints
    assert rules == []


def test_get_category(tmp_project_root):
    """Test get_category returns rules for specific category."""
    loader = InstructionsLoader(tmp_project_root)
    loader.load()

    security_rules = loader.get_category("scope_standards")

    assert isinstance(security_rules, list)
    # Should have some default security rules
    assert len(security_rules) > 0


def test_add_instruction(tmp_project_root):
    """Test add_instruction adds custom instruction at runtime."""
    loader = InstructionsLoader(tmp_project_root)
    loader.load()

    loader.add_instruction("custom_review_focus", "Check for async/await usage")

    custom_rules = loader.get_category("custom_review_focus")
    assert "Check for async/await usage" in custom_rules


def test_create_example_config(tmp_project_root):
    """Test create_example_config creates example YAML file."""
    output_path = tmp_project_root / "example-config.yaml"

    InstructionsLoader.create_example_config(output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "language_standards:" in content
    assert "scope_standards:" in content


def test_default_instructions_structure():
    """Test DEFAULT_INSTRUCTIONS has expected structure."""
    assert isinstance(DEFAULT_INSTRUCTIONS, dict)
    assert "language_standards" in DEFAULT_INSTRUCTIONS
    assert "scope_standards" in DEFAULT_INSTRUCTIONS
    assert "style_preferences" in DEFAULT_INSTRUCTIONS
    assert "custom_review_focus" in DEFAULT_INSTRUCTIONS

    # Each category should be a list
    for _category, rules in DEFAULT_INSTRUCTIONS.items():
        assert isinstance(rules, list)
        assert len(rules) > 0


def test_discover_and_load_uses_defaults_when_no_files(tmp_project_root):
    """Test discover_and_load uses defaults when no config files found."""
    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.discover_and_load()

    # Should have some instructions (from defaults)
    assert len(instructions.language_standards) > 0
    assert len(instructions.scope_standards) > 0


def test_extract_from_pyproject_with_ruff_lint(tmp_project_root):
    """Test _extract_from_pyproject extracts ruff.lint settings."""
    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text(
        """[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_pyproject()

    assert isinstance(rules, list)
    # Should mention enabled checks
    assert any("pycodestyle" in r.lower() or "error" in r.lower() for r in rules)


def test_extract_from_pyproject_with_mypy(tmp_project_root):
    """Test _extract_from_pyproject extracts mypy settings."""
    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text(
        """[tool.mypy]
disallow_untyped_defs = true
strict_equality = true
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_pyproject()

    assert isinstance(rules, list)
    assert any("type hints" in r.lower() or "type check" in r.lower() for r in rules)


def test_extract_from_pyproject_with_black(tmp_project_root):
    """Test _extract_from_pyproject extracts black settings."""
    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text(
        """[tool.black]
line-length = 100
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_pyproject()

    assert isinstance(rules, list)
    assert any("100" in r for r in rules)


def test_extract_from_language_configs(tmp_project_root):
    """Test _extract_from_language_configs detects language config files."""
    # Create a tsconfig.json
    tsconfig = tmp_project_root / "tsconfig.json"
    tsconfig.write_text('{"compilerOptions": {"strict": true}}')

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_from_language_configs()

    assert isinstance(rules, list)
    assert any("TypeScript" in r or "strict" in r for r in rules)


def test_extract_tsconfig_rules(tmp_project_root):
    """Test _extract_tsconfig_rules parses TypeScript config."""
    tsconfig = tmp_project_root / "tsconfig.json"
    tsconfig.write_text(
        """{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}"""
    )

    loader = InstructionsLoader(tmp_project_root)
    rules = loader._extract_tsconfig_rules(tsconfig, "TypeScript")

    assert isinstance(rules, list)
    assert any("strict" in r.lower() for r in rules)
    assert any("implicit any" in r.lower() for r in rules)


def test_discover_and_load_with_multiple_sources(tmp_project_root):
    """Test discover_and_load merges multiple sources."""
    # Create multiple config files
    editorconfig = tmp_project_root / ".editorconfig"
    editorconfig.write_text("[*.py]\nmax_line_length = 100\n")

    pyproject = tmp_project_root / "pyproject.toml"
    pyproject.write_text("[tool.ruff]\nline-length = 88\n")

    contributing = tmp_project_root / "CONTRIBUTING.md"
    contributing.write_text(
        """## Code Standards

- Use type hints for all functions
- Write comprehensive tests
"""
    )

    loader = InstructionsLoader(tmp_project_root)
    instructions = loader.discover_and_load()

    # Should have found multiple sources
    assert len(instructions.sources_found) >= 2
    # Should have extracted rules from different categories
    assert (
        len(instructions.style_preferences) > 0
        or len(instructions.custom_review_focus) > 0
    )


def test_review_instructions_immutable_defaults():
    """Test ReviewInstructions uses field default_factory for mutable fields."""
    # Create two instances
    inst1 = ReviewInstructions()
    inst2 = ReviewInstructions()

    # Add to one instance
    inst1.language_standards.append("test")

    # Should not affect the other instance
    assert "test" not in inst2.language_standards


def test_loader_handles_invalid_yaml_gracefully(tmp_project_root):
    """Test loader handles invalid YAML without crashing."""
    config_dir = tmp_project_root / ".mcp-vector-search"
    config_dir.mkdir()
    config_file = config_dir / "review-instructions.yaml"
    config_file.write_text("invalid: yaml: content: [[[")

    loader = InstructionsLoader(tmp_project_root)
    # Should not raise, fall back to defaults
    instructions = loader.discover_and_load()

    assert isinstance(instructions, ReviewInstructions)
