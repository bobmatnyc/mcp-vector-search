"""Review instructions loader for customizable PR/MR review rules.

This module provides the InstructionsLoader class for loading custom review
instructions from a YAML config file or falling back to sensible defaults.

Config File Location:
    .mcp-vector-search/review-instructions.yaml

Design Philosophy:
    - User-defined review standards take precedence
    - Graceful fallback to defaults if config missing
    - YAML for human-readable configuration
    - Validate and sanitize user input
    - Support multiple instruction categories
"""

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


class InstructionsLoader:
    """Loads custom review instructions from YAML config or uses defaults.

    The loader searches for review instructions in the project's
    .mcp-vector-search/review-instructions.yaml file. If not found,
    it falls back to sensible defaults.

    Attributes:
        project_root: Root directory of the project
        config_path: Path to the review instructions YAML file
        instructions: Loaded instructions dictionary

    Example:
        >>> loader = InstructionsLoader(Path("/path/to/project"))
        >>> instructions = loader.load()
        >>> formatted = loader.format_for_prompt()
        >>> print(formatted)
    """

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

        Returns:
            Formatted instructions string

        Example:
            >>> loader = InstructionsLoader(Path("."))
            >>> loader.load()
            >>> print(loader.format_for_prompt())
            ## Language Standards
            - Follow language-specific naming conventions...
            - All public functions must have docstrings...
            ...
        """
        if not self.instructions:
            self.load()

        sections = []

        for category, rules in self.instructions.items():
            # Convert snake_case to Title Case for display
            title = category.replace("_", " ").title()
            sections.append(f"## {title}\n")

            for rule in rules:
                sections.append(f"- {rule}")

            sections.append("")  # Blank line between sections

        return "\n".join(sections)

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
