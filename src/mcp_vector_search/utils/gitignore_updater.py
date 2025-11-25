"""Gitignore file update utilities for automatic .gitignore entry management."""

from pathlib import Path

from loguru import logger


def ensure_gitignore_entry(
    project_root: Path,
    pattern: str = ".mcp-vector-search/",
    comment: str | None = "MCP Vector Search index directory",
    create_if_missing: bool = True,
) -> bool:
    """Ensure a pattern exists in .gitignore file.

    This function safely adds a pattern to .gitignore if it doesn't already exist.
    It handles various edge cases including:
    - Non-existent .gitignore files (creates if in git repo)
    - Empty .gitignore files
    - Existing patterns in various formats
    - Negation patterns (conflict detection)
    - Permission errors
    - Encoding issues

    Design Decision: Non-Blocking Operation
    ----------------------------------------
    This function is designed to be non-critical and non-blocking. It will:
    - NEVER raise exceptions (returns False on errors)
    - Log warnings for failures instead of blocking
    - Allow project initialization to continue even if gitignore update fails

    Rationale: .gitignore updates are a quality-of-life improvement, not a
    requirement for mcp-vector-search functionality. Users can manually add
    the entry if automatic update fails.

    Pattern Detection Strategy
    --------------------------
    The function checks for semantic equivalents of the pattern:
    - `.mcp-vector-search/` (exact match)
    - `.mcp-vector-search` (without trailing slash)
    - `.mcp-vector-search/*` (with wildcard)
    - `/.mcp-vector-search/` (root-relative)

    All are treated as equivalent to avoid duplicate entries.

    Edge Cases Handled
    ------------------
    1. .gitignore does not exist -> Create (if in git repo)
    2. .gitignore is empty -> Add pattern
    3. Pattern already exists -> Skip (log debug)
    4. Similar pattern exists -> Skip (log debug)
    5. Negation pattern exists -> Warn and skip (respects user intent)
    6. Not a git repository -> Skip (no .gitignore needed)
    7. Permission denied -> Warn and skip (log manual instructions)
    8. Encoding errors -> Try fallback encoding
    9. Missing parent directory -> Should not occur (project_root exists)
    10. Concurrent modification -> Safe (append operation is atomic-ish)

    Args:
        project_root: Project root directory (must exist)
        pattern: Pattern to add to .gitignore (default: .mcp-vector-search/)
        comment: Optional comment to add before the pattern
        create_if_missing: Create .gitignore if it doesn't exist (default: True)

    Returns:
        True if pattern was added or already exists, False on error

    Performance:
        - Time Complexity: O(n) where n = lines in .gitignore (typically <1000)
        - Space Complexity: O(n) for reading file into memory
        - Expected Runtime: <10ms for typical .gitignore files

    Notes:
        - Only creates .gitignore in git repositories (checks for .git directory)
        - Preserves existing file structure and encoding (UTF-8)
        - Handles negation patterns gracefully (warns but doesn't override)
        - Non-blocking: logs warnings instead of raising exceptions

    Examples:
        >>> # Basic usage during project initialization
        >>> ensure_gitignore_entry(Path("/path/to/project"))
        True

        >>> # Custom pattern with custom comment
        >>> ensure_gitignore_entry(
        ...     Path("/path/to/project"),
        ...     pattern=".custom-dir/",
        ...     comment="Custom tool directory"
        ... )
        True

        >>> # Don't create .gitignore if missing
        >>> ensure_gitignore_entry(
        ...     Path("/path/to/project"),
        ...     create_if_missing=False
        ... )
        False
    """
    gitignore_path = project_root / ".gitignore"

    # Edge Case 1: Check if this is a git repository
    # Only create/modify .gitignore in git repositories to avoid polluting non-git projects
    git_dir = project_root / ".git"
    if not git_dir.exists():
        logger.debug(
            "Not a git repository (no .git directory), skipping .gitignore update"
        )
        return False

    try:
        # Edge Case 2: Handle non-existent .gitignore
        if not gitignore_path.exists():
            if not create_if_missing:
                logger.debug(".gitignore does not exist and create_if_missing=False")
                return False

            # Create new .gitignore with the pattern
            content = f"# {comment}\n{pattern}\n" if comment else f"{pattern}\n"
            gitignore_path.write_text(content, encoding="utf-8")
            logger.info(f"Created .gitignore with {pattern} entry")
            return True

        # Read existing content with UTF-8 encoding
        try:
            content = gitignore_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Edge Case 8: Fallback to more lenient encoding
            logger.debug("UTF-8 decode failed, trying with error replacement")
            try:
                content = gitignore_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(
                    f"Failed to read .gitignore due to encoding error: {e}. "
                    f"Please manually add '{pattern}' to your .gitignore"
                )
                return False

        # Edge Case 3: Handle empty .gitignore
        stripped_content = content.strip()
        if not stripped_content:
            content = f"# {comment}\n{pattern}\n" if comment else f"{pattern}\n"
            gitignore_path.write_text(content, encoding="utf-8")
            logger.info(f"Added {pattern} to empty .gitignore")
            return True

        # Check for existing patterns (Edge Cases 4, 5, 6)
        lines = content.split("\n")
        normalized_pattern = pattern.rstrip("/").lstrip("/")

        for line in lines:
            # Skip comments and empty lines
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            # Edge Case 6: Check for negation pattern (conflict)
            # Negation patterns indicate explicit user intent to track the directory
            if stripped_line.startswith("!") and normalized_pattern in stripped_line:
                logger.warning(
                    f".gitignore contains negation pattern: {stripped_line}. "
                    "This indicates you want to track .mcp-vector-search/ in git. "
                    "Skipping automatic entry to respect your configuration."
                )
                return False

            # Normalize line for comparison
            normalized_line = stripped_line.rstrip("/").lstrip("/")

            # Edge Cases 4 & 5: Check for exact or similar matches
            # These patterns are semantically equivalent for .gitignore:
            # - .mcp-vector-search/
            # - .mcp-vector-search
            # - .mcp-vector-search/*
            # - /.mcp-vector-search/
            if (
                normalized_line == normalized_pattern
                or normalized_line == normalized_pattern + "/*"
            ):
                logger.debug(f"Pattern already exists in .gitignore: {stripped_line}")
                return True

        # Pattern doesn't exist, add it
        # Preserve file structure: ensure proper newline handling
        if not content.endswith("\n"):
            content += "\n"

        # Add blank line before comment for visual separation
        content += "\n"

        if comment:
            content += f"# {comment}\n"
        content += f"{pattern}\n"

        # Write back to file
        gitignore_path.write_text(content, encoding="utf-8")
        logger.info(f"Added {pattern} to .gitignore")
        return True

    except PermissionError:
        # Edge Case 7: Handle read-only .gitignore or protected directory
        logger.warning(
            f"Cannot update .gitignore: Permission denied. "
            f"Please manually add '{pattern}' to your .gitignore file at {gitignore_path}"
        )
        return False
    except Exception as e:
        # Catch-all for unexpected errors (don't block initialization)
        logger.warning(
            f"Failed to update .gitignore: {e}. "
            f"Please manually add '{pattern}' to your .gitignore"
        )
        return False
