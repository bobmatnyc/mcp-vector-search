"""Contextual suggestion system for better user experience."""

import json
from pathlib import Path

from loguru import logger
from rich.console import Console


class ContextualSuggestionProvider:
    """Provides context-aware suggestions based on project state and user workflow."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the suggestion provider.

        Args:
            project_root: Root directory of the project (defaults to current directory)
        """
        self.project_root = project_root or Path.cwd()
        self.console = Console(stderr=True)

    def get_project_state(self) -> dict[str, bool]:
        """Analyze the current project state.

        Returns:
            Dictionary with boolean flags indicating project state
        """
        state = {
            "is_initialized": False,
            "has_index": False,
            "has_config": False,
            "has_recent_changes": False,
            "is_git_repo": False,
            "has_mcp_config": False,
        }

        try:
            # Check if project is initialized
            config_dir = self.project_root / ".mcp-vector-search"
            state["is_initialized"] = config_dir.exists()

            if state["is_initialized"]:
                # Check for config
                config_file = config_dir / "config.json"
                state["has_config"] = config_file.exists()

                # Check for index
                index_dir = config_dir / "chroma_db"
                state["has_index"] = index_dir.exists() and any(index_dir.iterdir())

            # Check if it's a git repo
            git_dir = self.project_root / ".git"
            state["is_git_repo"] = git_dir.exists()

            # Check for MCP configuration (Claude Desktop config)
            home = Path.home()
            claude_config = (
                home
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
            if claude_config.exists():
                try:
                    with open(claude_config) as f:
                        config_data = json.load(f)
                        mcp_servers = config_data.get("mcpServers", {})
                        state["has_mcp_config"] = "mcp-vector-search" in mcp_servers
                except (OSError, json.JSONDecodeError):
                    pass

            # TODO: Check for recent file changes (would need file system monitoring)
            # For now, we'll assume false
            state["has_recent_changes"] = False

        except Exception as e:
            # If we can't determine state, provide conservative defaults
            logger.debug(f"Failed to determine project state for suggestions: {e}")
            pass

        return state

    def get_workflow_suggestions(self, failed_command: str) -> list[dict[str, str]]:
        """Get workflow-based suggestions for a failed command.

        Args:
            failed_command: The command that failed

        Returns:
            List of suggestion dictionaries with 'command', 'reason', and 'priority'
        """
        suggestions = []
        state = self.get_project_state()

        # High priority suggestions based on project state
        if not state["is_initialized"]:
            suggestions.append(
                {
                    "command": "init",
                    "reason": "Project is not initialized for vector search",
                    "priority": "high",
                    "description": "Set up the project configuration and create necessary directories",
                }
            )
        elif not state["has_index"]:
            suggestions.append(
                {
                    "command": "index",
                    "reason": "No search index found - create one to enable searching",
                    "priority": "high",
                    "description": "Build the vector index for your codebase",
                }
            )

        # Context-specific suggestions based on the failed command
        if failed_command.lower() in ["search", "find", "query", "s", "f"]:
            if not state["has_index"]:
                suggestions.append(
                    {
                        "command": "index",
                        "reason": "Cannot search without an index",
                        "priority": "high",
                        "description": "Build the search index first",
                    }
                )
            else:
                suggestions.extend(
                    [
                        {
                            "command": "search",
                            "reason": "Correct command for semantic code search",
                            "priority": "high",
                            "description": "Search your codebase semantically",
                        },
                        {
                            "command": "interactive",
                            "reason": "Try interactive search for better experience",
                            "priority": "medium",
                            "description": "Start an interactive search session",
                        },
                    ]
                )

        elif failed_command.lower() in ["index", "build", "scan", "i", "b"]:
            suggestions.append(
                {
                    "command": "index",
                    "reason": "Correct command for building search index",
                    "priority": "high",
                    "description": "Index your codebase for semantic search",
                }
            )

        elif failed_command.lower() in ["status", "info", "stat", "st"]:
            suggestions.append(
                {
                    "command": "status",
                    "reason": "Show project status and statistics",
                    "priority": "high",
                    "description": "Display current project information",
                }
            )

        elif failed_command.lower() in ["config", "configure", "settings", "c"]:
            suggestions.append(
                {
                    "command": "config",
                    "reason": "Manage project configuration",
                    "priority": "high",
                    "description": "View or modify project settings",
                }
            )

        # MCP-related suggestions
        if not state["has_mcp_config"] and failed_command.lower() in [
            "mcp",
            "claude",
            "server",
        ]:
            suggestions.append(
                {
                    "command": "init-mcp",
                    "reason": "Set up Claude Code MCP integration",
                    "priority": "medium",
                    "description": "Configure MCP server for Claude Code integration",
                }
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = suggestion["command"]
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def get_next_steps(self) -> list[dict[str, str]]:
        """Get suggested next steps based on current project state.

        Returns:
            List of suggested next step dictionaries
        """
        state = self.get_project_state()
        next_steps = []

        if not state["is_initialized"]:
            next_steps.append(
                {
                    "command": "init",
                    "description": "Initialize the project for semantic search",
                    "priority": "high",
                }
            )
        elif not state["has_index"]:
            next_steps.append(
                {
                    "command": "index",
                    "description": "Build the search index for your codebase",
                    "priority": "high",
                }
            )
        else:
            # Project is ready for use
            next_steps.extend(
                [
                    {
                        "command": 'search "your query here"',
                        "description": "Search your codebase semantically",
                        "priority": "high",
                    },
                    {
                        "command": "status",
                        "description": "Check project statistics and index health",
                        "priority": "medium",
                    },
                ]
            )

            if state["has_recent_changes"]:
                next_steps.insert(
                    0,
                    {
                        "command": "index --force",
                        "description": "Update the index with recent changes",
                        "priority": "high",
                    },
                )

        if not state["has_mcp_config"] and state["is_initialized"]:
            next_steps.append(
                {
                    "command": "init-mcp",
                    "description": "Set up Claude Code integration",
                    "priority": "low",
                }
            )

        return next_steps

    def show_contextual_help(self, failed_command: str | None = None) -> None:
        """Show contextual help and suggestions.

        Args:
            failed_command: The command that failed (if any)
        """
        if failed_command:
            self.console.print(
                f"\n[yellow]Command '{failed_command}' not recognized.[/yellow]"
            )

            suggestions = self.get_workflow_suggestions(failed_command)
            if suggestions:
                self.console.print(
                    "\n[bold]Based on your project state, you might want to try:[/bold]"
                )

                for i, suggestion in enumerate(suggestions[:3], 1):  # Show top 3
                    priority_color = {
                        "high": "red",
                        "medium": "yellow",
                        "low": "dim",
                    }.get(suggestion["priority"], "white")

                    self.console.print(
                        f"  [{priority_color}]{i}.[/{priority_color}] "
                        f"[bold cyan]mcp-vector-search {suggestion['command']}[/bold cyan]"
                    )
                    self.console.print(f"     {suggestion['description']}")
                    if suggestion.get("reason"):
                        self.console.print(f"     [dim]({suggestion['reason']})[/dim]")
        else:
            # Show general next steps
            next_steps = self.get_next_steps()
            if next_steps:
                self.console.print("\n[bold]Suggested next steps:[/bold]")

                for i, step in enumerate(next_steps[:3], 1):
                    priority_color = {
                        "high": "green",
                        "medium": "yellow",
                        "low": "dim",
                    }.get(step["priority"], "white")

                    self.console.print(
                        f"  [{priority_color}]{i}.[/{priority_color}] "
                        f"[bold cyan]mcp-vector-search {step['command']}[/bold cyan]"
                    )
                    self.console.print(f"     {step['description']}")

    def get_command_completion_suggestions(self, partial_command: str) -> list[str]:
        """Get command completion suggestions for a partial command.

        Args:
            partial_command: Partial command string

        Returns:
            List of possible command completions
        """
        all_commands = [
            "search",
            "index",
            "status",
            "config",
            "init",
            "mcp",
            "doctor",
            "version",
            "watch",
            "auto-index",
            "history",
            "interactive",
            "demo",
            "install",
            "reset",
            "health",
        ]

        # Add common aliases and shortcuts
        all_commands.extend(["s", "i", "st", "c", "f", "find"])

        partial_lower = partial_command.lower()
        matches = [cmd for cmd in all_commands if cmd.startswith(partial_lower)]

        return sorted(matches)


def get_contextual_suggestions(
    project_root: Path | None = None, failed_command: str | None = None
) -> None:
    """Get and display contextual suggestions.

    Args:
        project_root: Root directory of the project
        failed_command: The command that failed
    """
    provider = ContextualSuggestionProvider(project_root)
    provider.show_contextual_help(failed_command)


def suggest_workflow_commands(project_root: Path | None = None) -> list[str]:
    """Get workflow command suggestions for the current project state.

    Args:
        project_root: Root directory of the project

    Returns:
        List of suggested commands in priority order
    """
    provider = ContextualSuggestionProvider(project_root)
    next_steps = provider.get_next_steps()

    return [step["command"] for step in next_steps]
