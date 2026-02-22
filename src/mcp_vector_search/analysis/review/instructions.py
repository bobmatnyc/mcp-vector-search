"""Review instructions loader for customizable PR/MR review rules.

This module provides the InstructionsLoader class for loading custom review
instructions from a YAML config file or auto-discovering standards from
repository files.

Config File Location:
    .mcp-vector-search/review-instructions.yaml (explicit override)

Auto-Discovered Files (in priority order):
    1. .mcp-vector-search/review-instructions.yaml
    2. .github/PULL_REQUEST_TEMPLATE.md
    3. CONTRIBUTING.md or docs/CONTRIBUTING.md
    4. DEVELOPMENT.md or docs/DEVELOPMENT.md
    5. STYLE_GUIDE.md or docs/style*.md
    6. pyproject.toml (tool.ruff, tool.mypy, tool.black)
    7. .editorconfig

Design Philosophy:
    - User-defined review standards take precedence
    - Auto-discover standards from existing repository files
    - Graceful fallback to defaults if no standards found
    - YAML for human-readable configuration
    - Validate and sanitize user input
    - Support multiple instruction categories
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed, review instructions will use defaults only")


# Default review instructions (fallback when no config file)
DEFAULT_INSTRUCTIONS = {
    "language_standards": [
        "Follow language-specific naming conventions (e.g., snake_case for Python, camelCase for JavaScript)",
        "All public functions and classes must have docstrings/comments",
        "Use type hints/annotations where applicable",
        "Avoid deep nesting (max 3-4 levels)",
    ],
    "scope_standards": [
        "No hardcoded credentials, secrets, or API keys",
        "All database queries must use parameterized statements",
        "Error handling required for all external API calls",
        "Input validation required for all user-provided data",
    ],
    "style_preferences": [
        "Prefer composition over inheritance",
        "Functions should be small and focused (Single Responsibility)",
        "Avoid magic numbers and strings (use named constants)",
        "Comments should explain WHY, not WHAT",
    ],
    "custom_review_focus": [
        "Check for proper error handling and edge cases",
        "Verify logging is appropriate and not excessive",
        "Ensure backward compatibility is maintained",
    ],
}


@dataclass
class ReviewInstructions:
    """Structured review instructions with source attribution.

    Attributes:
        language_standards: Language-specific conventions and style rules
        scope_standards: Security, quality, and safety requirements
        style_preferences: Design patterns and architectural preferences
        custom_review_focus: Project-specific review priorities
        sources_found: List of files that were discovered and used
        has_custom_config: Whether explicit .mcp-vector-search/review-instructions.yaml exists
    """

    language_standards: list[str] = field(default_factory=list)
    scope_standards: list[str] = field(default_factory=list)
    style_preferences: list[str] = field(default_factory=list)
    custom_review_focus: list[str] = field(default_factory=list)
    sources_found: list[str] = field(default_factory=list)
    has_custom_config: bool = False

    def to_prompt_text(self) -> str:
        """Format all instructions for injection into LLM prompt.

        Returns:
            Formatted instructions string with source attribution

        Example:
            >>> instructions = ReviewInstructions(...)
            >>> print(instructions.to_prompt_text())
            ## Language Standards
            - Follow language-specific naming conventions...
            ...
            Sources: .editorconfig, pyproject.toml
        """
        sections = []

        # Build instruction sections
        categories = [
            ("Language Standards", self.language_standards),
            ("Scope Standards", self.scope_standards),
            ("Style Preferences", self.style_preferences),
            ("Custom Review Focus", self.custom_review_focus),
        ]

        for title, rules in categories:
            if rules:
                sections.append(f"## {title}\n")
                for rule in rules:
                    sections.append(f"- {rule}")
                sections.append("")  # Blank line

        # Add source attribution
        if self.sources_found:
            sources_text = ", ".join(self.sources_found)
            sections.append(f"\n*Standards discovered from: {sources_text}*")

        return "\n".join(sections)


class InstructionsLoader:
    """Loads custom review instructions from YAML config or auto-discovers from repo files.

    The loader searches for review instructions in multiple sources:
    1. Explicit config: .mcp-vector-search/review-instructions.yaml (highest priority)
    2. Repository standards files (auto-discovered in priority order)
    3. Defaults (fallback if nothing found)

    Auto-discovery checks these files in order:
        - .github/PULL_REQUEST_TEMPLATE.md
        - CONTRIBUTING.md or docs/CONTRIBUTING.md
        - DEVELOPMENT.md or docs/DEVELOPMENT.md
        - STYLE_GUIDE.md or docs/style*.md
        - pyproject.toml (tool.ruff, tool.mypy, tool.black)
        - .editorconfig

    Attributes:
        project_root: Root directory of the project
        config_path: Path to the review instructions YAML file
        instructions: Loaded instructions dictionary

    Example:
        >>> loader = InstructionsLoader(Path("/path/to/project"))
        >>> instructions = loader.discover_and_load()
        >>> print(instructions.to_prompt_text())
    """

    # Files to search for in order of priority (after explicit config)
    STANDARDS_FILES = [
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/pull_request_template.md",
        "CONTRIBUTING.md",
        "docs/CONTRIBUTING.md",
        "DEVELOPMENT.md",
        "docs/DEVELOPMENT.md",
        "DEVELOPMENT.rst",
        "docs/DEVELOPMENT.rst",
        "STYLE_GUIDE.md",
        "STYLE_GUIDE.rst",
        "docs/style-guide.md",
        "docs/style.md",
    ]

    def __init__(self, project_root: Path):
        """Initialize instructions loader.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.config_path = (
            project_root / ".mcp-vector-search" / "review-instructions.yaml"
        )
        self.instructions: dict[str, list[str]] = {}

    def discover_and_load(self) -> ReviewInstructions:
        """Auto-discover standards files and merge into unified instructions.

        Discovery priority:
        1. Explicit config (.mcp-vector-search/review-instructions.yaml)
        2. Repository standards files (CONTRIBUTING.md, DEVELOPMENT.md, etc.)
        3. Tool config files (pyproject.toml, .editorconfig)
        4. Defaults

        Returns:
            ReviewInstructions with merged standards and source attribution

        Example:
            >>> loader = InstructionsLoader(Path("/path/to/project"))
            >>> instructions = loader.discover_and_load()
            >>> print(f"Found sources: {instructions.sources_found}")
        """
        sources_found: list[str] = []
        merged_instructions: dict[str, list[str]] = {
            "language_standards": [],
            "scope_standards": [],
            "style_preferences": [],
            "custom_review_focus": [],
        }
        has_custom_config = False

        # 1. Check for explicit config (highest priority, overrides everything)
        if self.config_path.exists() and YAML_AVAILABLE:
            try:
                yaml_instructions = self._load_from_file()
                # Merge YAML config
                for category, rules in yaml_instructions.items():
                    if category in merged_instructions:
                        merged_instructions[category].extend(rules)
                    else:
                        merged_instructions[category] = rules
                sources_found.append(".mcp-vector-search/review-instructions.yaml")
                has_custom_config = True
                logger.info(
                    f"Loaded custom review instructions from {self.config_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load review instructions from {self.config_path}: {e}"
                )

        # 2. Auto-discover markdown standards files
        for file_name in self.STANDARDS_FILES:
            file_path = self.project_root / file_name
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    extracted = self._extract_from_markdown(content, file_name)
                    if extracted:
                        # Add to appropriate category (heuristic based on content)
                        merged_instructions["custom_review_focus"].extend(extracted)
                        sources_found.append(file_name)
                        logger.debug(
                            f"Extracted {len(extracted)} rules from {file_name}"
                        )
                except Exception as e:
                    logger.debug(f"Failed to read {file_name}: {e}")

        # 3. Extract from pyproject.toml
        pyproject_rules = self._extract_from_pyproject()
        if pyproject_rules:
            merged_instructions["style_preferences"].extend(pyproject_rules)
            sources_found.append("pyproject.toml")
            logger.debug(f"Extracted {len(pyproject_rules)} rules from pyproject.toml")

        # 4. Extract from .editorconfig
        editorconfig_rules = self._extract_from_editorconfig()
        if editorconfig_rules:
            merged_instructions["style_preferences"].extend(editorconfig_rules)
            sources_found.append(".editorconfig")
            logger.debug(
                f"Extracted {len(editorconfig_rules)} rules from .editorconfig"
            )

        # 5. Deduplicate rules within each category
        for category in merged_instructions:
            if merged_instructions[category]:
                # Remove exact duplicates while preserving order
                seen = set()
                deduplicated = []
                for rule in merged_instructions[category]:
                    if rule not in seen:
                        seen.add(rule)
                        deduplicated.append(rule)
                merged_instructions[category] = deduplicated

        # 6. Fallback to defaults if nothing found
        if not sources_found:
            logger.debug("No custom standards found, using defaults")
            for category, rules in DEFAULT_INSTRUCTIONS.items():
                merged_instructions[category] = rules

        # Log discovery summary
        if sources_found:
            logger.info(f"Discovered standards from: {', '.join(sources_found)}")

        return ReviewInstructions(
            language_standards=merged_instructions.get("language_standards", []),
            scope_standards=merged_instructions.get("scope_standards", []),
            style_preferences=merged_instructions.get("style_preferences", []),
            custom_review_focus=merged_instructions.get("custom_review_focus", []),
            sources_found=sources_found,
            has_custom_config=has_custom_config,
        )

    def _extract_from_markdown(self, content: str, source: str) -> list[str]:
        """Extract actionable standards from markdown files.

        Looks for sections with headings like "Code Style", "Standards",
        "Requirements", "Conventions" and extracts bullet points or numbered lists.

        Args:
            content: Markdown file content
            source: Source file name (for logging)

        Returns:
            List of extracted standards

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> content = "## Code Style\\n- Use snake_case\\n- Max line length 88"
            >>> rules = loader._extract_from_markdown(content, "CONTRIBUTING.md")
            >>> print(rules)
            ['Code Style: Use snake_case', 'Code Style: Max line length 88']
        """
        rules: list[str] = []

        # Keywords to identify relevant sections
        relevant_keywords = [
            "code style",
            "style guide",
            "standards",
            "conventions",
            "requirements",
            "guidelines",
            "best practices",
            "coding standards",
        ]

        # Split into sections by headers
        lines = content.split("\n")
        current_section = None
        in_relevant_section = False

        for line in lines:
            # Check for markdown headers (# Header or ## Header)
            header_match = re.match(r"^#{1,6}\s+(.+)$", line)
            if header_match:
                section_title = header_match.group(1).strip()
                # Check if this is a relevant section
                if any(
                    keyword in section_title.lower() for keyword in relevant_keywords
                ):
                    current_section = section_title
                    in_relevant_section = True
                else:
                    in_relevant_section = False
                continue

            # Extract bullet points or numbered lists
            if in_relevant_section:
                # Match bullet points (-, *, +) or numbered lists (1., 2.)
                bullet_match = re.match(r"^[\s]*[-*+]\s+(.+)$", line)
                numbered_match = re.match(r"^[\s]*\d+\.\s+(.+)$", line)

                if bullet_match:
                    rule_text = bullet_match.group(1).strip()
                    if rule_text and len(rule_text) > 10:  # Skip very short items
                        rules.append(f"{current_section}: {rule_text}")
                elif numbered_match:
                    rule_text = numbered_match.group(1).strip()
                    if rule_text and len(rule_text) > 10:
                        rules.append(f"{current_section}: {rule_text}")

        return rules

    def _extract_from_pyproject(self) -> list[str]:
        """Extract standards from pyproject.toml tool configuration.

        Looks for:
        - [tool.ruff] line-length, target-version
        - [tool.ruff.lint] select/ignore rules
        - [tool.mypy] configuration
        - [tool.black] line-length

        Returns:
            List of extracted standards

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> rules = loader._extract_from_pyproject()
            >>> print(rules)
            ['Maximum line length: 88', 'Target Python version: 3.11', ...]
        """
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            return []

        rules: list[str] = []

        try:
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)

            # Extract from [tool.ruff]
            if "tool" in config and "ruff" in config["tool"]:
                ruff_config = config["tool"]["ruff"]

                if "line-length" in ruff_config:
                    rules.append(f"Maximum line length: {ruff_config['line-length']}")

                if "target-version" in ruff_config:
                    target = ruff_config["target-version"]
                    rules.append(f"Target Python version: {target}")

                # Extract from [tool.ruff.lint]
                if "lint" in ruff_config:
                    lint_config = ruff_config["lint"]

                    if "select" in lint_config:
                        select_rules = lint_config["select"]
                        rule_descriptions = {
                            "E": "pycodestyle errors",
                            "W": "pycodestyle warnings",
                            "F": "pyflakes",
                            "I": "import sorting (isort)",
                            "B": "flake8-bugbear",
                            "C4": "flake8-comprehensions",
                            "UP": "pyupgrade (modern Python idioms)",
                            "N": "pep8-naming conventions",
                        }
                        for rule_code in select_rules:
                            if rule_code in rule_descriptions:
                                rules.append(
                                    f"Enable {rule_descriptions[rule_code]} checks"
                                )

            # Extract from [tool.mypy]
            if "tool" in config and "mypy" in config["tool"]:
                mypy_config = config["tool"]["mypy"]

                if mypy_config.get("check_untyped_defs"):
                    rules.append("Type check all function definitions")

                if mypy_config.get("disallow_untyped_defs"):
                    rules.append("All functions must have type hints")

                if mypy_config.get("strict_equality"):
                    rules.append("Use strict equality checks")

            # Extract from [tool.black]
            if "tool" in config and "black" in config["tool"]:
                black_config = config["tool"]["black"]

                if "line-length" in black_config:
                    rules.append(f"Maximum line length: {black_config['line-length']}")

        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml: {e}")

        return rules

    def _extract_from_editorconfig(self) -> list[str]:
        """Extract formatting standards from .editorconfig.

        Looks for:
        - indent_style (tabs/spaces)
        - indent_size
        - max_line_length
        - end_of_line
        - charset

        Returns:
            List of extracted standards

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> rules = loader._extract_from_editorconfig()
            >>> print(rules)
            ['Python files: Use spaces for indentation', 'Python files: Indent size 4', ...]
        """
        editorconfig_path = self.project_root / ".editorconfig"
        if not editorconfig_path.exists():
            return []

        rules: list[str] = []

        try:
            content = editorconfig_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            current_section = None
            section_config: dict[str, str] = {}

            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Check for section headers [*.py]
                section_match = re.match(r"^\[(.+)\]$", line)
                if section_match:
                    # Process previous section
                    if current_section and section_config:
                        section_rules = self._format_editorconfig_section(
                            current_section, section_config
                        )
                        rules.extend(section_rules)

                    # Start new section
                    current_section = section_match.group(1)
                    section_config = {}
                    continue

                # Parse key = value pairs
                if "=" in line:
                    key, value = line.split("=", 1)
                    section_config[key.strip()] = value.strip()

            # Process last section
            if current_section and section_config:
                section_rules = self._format_editorconfig_section(
                    current_section, section_config
                )
                rules.extend(section_rules)

        except Exception as e:
            logger.debug(f"Failed to parse .editorconfig: {e}")

        return rules

    def _format_editorconfig_section(
        self, section: str, config: dict[str, str]
    ) -> list[str]:
        """Format editorconfig section into readable rules.

        Prioritizes Python-specific rules and avoids generic duplicates.

        Args:
            section: Section pattern (e.g., "*.py", "*")
            config: Configuration key-value pairs

        Returns:
            List of formatted rules
        """
        rules: list[str] = []

        # Determine file type for prefix (prioritize Python)
        is_python = "*.py" in section
        is_yaml = "*.{yml,yaml}" in section or "*.yml" in section or "*.yaml" in section
        is_markdown = "*.md" in section
        is_all_files = section == "*"

        # Skip generic "all files" rules to avoid duplication
        # We only care about Python-specific rules for code review
        if is_all_files and not config.get("max_line_length"):
            # Skip generic charset/eol rules unless they have specific constraints
            return []

        if is_python:
            prefix = "Python files"
        elif is_yaml:
            prefix = "YAML files"
        elif is_markdown:
            prefix = "Markdown files"
        elif is_all_files:
            prefix = "All files"
        else:
            # Skip non-standard file types
            return []

        # Format each config item
        if "indent_style" in config:
            style = config["indent_style"]
            rules.append(f"{prefix}: Use {style} for indentation")

        if "indent_size" in config:
            size = config["indent_size"]
            rules.append(f"{prefix}: Indent size {size}")

        if "max_line_length" in config:
            length = config["max_line_length"]
            rules.append(f"{prefix}: Maximum line length {length}")

        if "end_of_line" in config and is_all_files:
            eol = config["end_of_line"]
            rules.append(f"Use {eol} line endings")

        if "charset" in config and is_all_files:
            charset = config["charset"]
            rules.append(f"Character encoding: {charset}")

        return rules

    def load(self) -> dict[str, list[str]]:
        """Load review instructions from config file or defaults.

        Returns:
            Dictionary mapping instruction category to list of rules

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> instructions = loader.load()
            >>> instructions["language_standards"]
            ['Follow language-specific naming conventions...', ...]
        """
        # Try to load from config file
        if self.config_path.exists() and YAML_AVAILABLE:
            try:
                self.instructions = self._load_from_file()
                logger.info(
                    f"Loaded custom review instructions from {self.config_path}"
                )
                return self.instructions
            except Exception as e:
                logger.warning(
                    f"Failed to load review instructions from {self.config_path}: {e}. "
                    "Using defaults."
                )

        # Fallback to defaults
        self.instructions = DEFAULT_INSTRUCTIONS.copy()
        logger.debug("Using default review instructions")
        return self.instructions

    def _load_from_file(self) -> dict[str, list[str]]:
        """Load and validate instructions from YAML file.

        Returns:
            Validated instructions dictionary

        Raises:
            ValueError: If file format is invalid
        """
        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Review instructions must be a YAML dictionary")

        # Validate structure (all values should be lists of strings)
        validated: dict[str, list[str]] = {}
        for category, rules in data.items():
            if not isinstance(rules, list):
                logger.warning(
                    f"Skipping invalid category '{category}': must be a list of strings"
                )
                continue

            # Filter to only string rules
            string_rules = [r for r in rules if isinstance(r, str)]
            if len(string_rules) != len(rules):
                logger.warning(
                    f"Category '{category}' has {len(rules) - len(string_rules)} "
                    "non-string rules that were skipped"
                )

            if string_rules:
                validated[category] = string_rules

        # Merge with defaults if some categories missing
        for category, default_rules in DEFAULT_INSTRUCTIONS.items():
            if category not in validated:
                validated[category] = default_rules
                logger.debug(f"Using default instructions for category '{category}'")

        return validated

    def format_for_prompt(self) -> str:
        """Format instructions as a text string for LLM prompt.

        Uses the new auto-discovery system. For backward compatibility,
        falls back to old load() method if needed.

        Returns:
            Formatted instructions string

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> print(loader.format_for_prompt())
            ## Language Standards
            - Follow language-specific naming conventions...
            - All public functions must have docstrings...
            ...
            *Standards discovered from: .editorconfig, pyproject.toml*
        """
        # Use new auto-discovery system
        instructions = self.discover_and_load()
        return instructions.to_prompt_text()

    def get_category(self, category: str) -> list[str]:
        """Get rules for a specific category.

        Args:
            category: Category name (e.g., "language_standards")

        Returns:
            List of rules for that category (empty list if not found)

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> loader.load()
            >>> security_rules = loader.get_category("scope_standards")
        """
        if not self.instructions:
            self.load()

        return self.instructions.get(category, [])

    def add_instruction(self, category: str, rule: str) -> None:
        """Add a custom instruction at runtime (not persisted to file).

        Args:
            category: Category name
            rule: Instruction rule to add

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> loader.load()
            >>> loader.add_instruction("custom_review_focus", "Check for async/await usage")
        """
        if not self.instructions:
            self.load()

        if category not in self.instructions:
            self.instructions[category] = []

        self.instructions[category].append(rule)
        logger.debug(f"Added runtime instruction to '{category}': {rule}")

    @staticmethod
    def create_example_config(output_path: Path) -> None:
        """Create an example review-instructions.yaml file.

        Args:
            output_path: Path where example config should be written

        Example:
            >>> InstructionsLoader.create_example_config(
            ...     Path(".mcp-vector-search/review-instructions.yaml.example")
            ... )
        """
        example_content = """# Custom Review Instructions for PR/MR Reviews
#
# This file defines custom review standards and guidelines that will be
# applied during automated code reviews. Customize these rules to match
# your team's coding standards and best practices.
#
# Categories:
#   - language_standards: Language-specific conventions and style
#   - scope_standards: Security, quality, and safety requirements
#   - style_preferences: Design patterns and architectural preferences
#   - custom_review_focus: Project-specific review priorities

language_standards:
  - "Use snake_case for all Python variables and functions"
  - "All public methods must have docstrings with Args/Returns sections"
  - "Maximum function length: 50 lines"
  - "Type hints required for all function signatures"
  - "Avoid deep nesting (maximum 3 levels)"

scope_standards:
  - "All database queries must use parameterized statements"
  - "Error handling required for all external API calls"
  - "No hardcoded credentials or API keys"
  - "Input validation required for all user-provided data"
  - "Logging must not expose sensitive information"

style_preferences:
  - "Prefer composition over inheritance"
  - "Use type hints for all function signatures"
  - "Functions should have single responsibility"
  - "Avoid magic numbers (use named constants)"
  - "Comments should explain WHY, not WHAT"

custom_review_focus:
  - "Pay special attention to authentication and authorization"
  - "Check for proper async/await usage in concurrent code"
  - "Verify backward compatibility with existing API"
  - "Ensure test coverage for all new functionality"
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(example_content)

        logger.info(f"Created example config at {output_path}")
