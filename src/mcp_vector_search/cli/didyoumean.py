"""Enhanced CLI with 'did you mean' functionality for better user experience."""

import difflib

import click
import typer
from click_didyoumean import DYMGroup


class EnhancedDidYouMeanTyper(typer.Typer):
    """Enhanced Typer class with advanced 'did you mean' functionality."""

    def __init__(self, *args, **kwargs):
        """Initialize with enhanced did-you-mean support."""
        # Extract Typer-specific kwargs
        typer_kwargs = {}
        click_kwargs = {}

        # Separate Typer and Click kwargs
        typer_specific = {
            "name",
            "help",
            "epilog",
            "short_help",
            "options_metavar",
            "add_completion",
            "context_settings",
            "callback",
            "invoke_without_command",
            "no_args_is_help",
            "subcommand_metavar",
            "chain",
            "result_callback",
            "deprecated",
            "rich_markup_mode",
            "rich_help_panel",
            "pretty_exceptions_enable",
            "pretty_exceptions_show_locals",
            "pretty_exceptions_short",
        }

        for key, value in kwargs.items():
            if key in typer_specific:
                typer_kwargs[key] = value
            else:
                click_kwargs[key] = value

        # Initialize Typer with its specific kwargs
        super().__init__(*args, **typer_kwargs)

        # Store click kwargs for later use
        self._click_kwargs = click_kwargs
        self.command_aliases = {}  # Store command aliases

    def __call__(self, *args, **kwargs):
        """Override call to use enhanced DYMGroup."""
        # Get the underlying click group
        click_group = super().__call__(*args, **kwargs)

        # If click_group is None (command already executed), return None
        # This happens after command execution completes successfully
        if click_group is None:
            return None

        # If click_group is an integer, it's an exit code from standalone_mode=False
        # Return it as-is to preserve exit code propagation
        if isinstance(click_group, int):
            return click_group

        # Create enhanced DYM group with original group's properties
        enhanced_group = EnhancedDidYouMeanGroup(
            name=click_group.name,
            commands=click_group.commands,
            callback=click_group.callback,
            params=click_group.params,
            help=click_group.help,
            epilog=click_group.epilog,
            short_help=click_group.short_help,
            add_help_option=click_group.add_help_option,
            context_settings=click_group.context_settings,
            invoke_without_command=click_group.invoke_without_command,
            no_args_is_help=click_group.no_args_is_help,
            subcommand_metavar=click_group.subcommand_metavar,
            chain=click_group.chain,
            result_callback=click_group.result_callback,
            deprecated=click_group.deprecated,
            **self._click_kwargs,
        )

        # Additional attributes that might be needed
        if hasattr(click_group, "options_metavar"):
            enhanced_group.options_metavar = click_group.options_metavar

        return enhanced_group

    def add_alias(self, command_name: str, alias: str) -> None:
        """Add an alias for a command.

        Args:
            command_name: The original command name
            alias: The alias to add
        """
        self.command_aliases[alias] = command_name


class EnhancedDidYouMeanGroup(DYMGroup):
    """Enhanced Click group with advanced 'did you mean' functionality."""

    def __init__(self, *args, **kwargs):
        """Initialize with better error messages and fuzzy matching."""
        super().__init__(*args, **kwargs)
        self.max_suggestions = 3  # Maximum number of suggestions to show

    def resolve_command(self, ctx: click.Context, args: list) -> tuple:
        """Resolve command with enhanced error handling and suggestions."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            # Enhanced error handling with better suggestions
            if "No such command" in str(e) and args:
                command_name = args[0]

                # Use our enhanced suggestion system
                add_common_suggestions(ctx, command_name)

                # Re-raise with original message (suggestions already printed)
                raise click.UsageError(str(e), ctx=ctx)
            raise

    def get_command(self, ctx: click.Context, cmd_name: str):
        """Get command with support for aliases and shortcuts."""
        # First try exact match
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv

        # Try common typo mappings
        suggestion = COMMON_TYPOS.get(cmd_name.lower())
        if suggestion and " " not in suggestion:  # Only single commands, not flags
            return super().get_command(ctx, suggestion)

        return None


def create_enhanced_typer(**kwargs) -> typer.Typer:
    """Create a Typer instance with enhanced 'did you mean' functionality."""
    # Set default values for better UX
    defaults = {
        "no_args_is_help": True,
        "add_completion": False,
        "rich_markup_mode": "rich",
    }

    # Merge with provided kwargs
    final_kwargs = {**defaults, **kwargs}

    # Create the enhanced Typer
    app = EnhancedDidYouMeanTyper(**final_kwargs)

    return app


def enhance_existing_typer(app: typer.Typer) -> typer.Typer:
    """Enhance an existing Typer app with advanced 'did you mean' functionality."""
    # This is a bit tricky since we need to modify the underlying Click group
    # We'll create a wrapper that intercepts the click group creation

    original_call = app.__call__

    def enhanced_call(*args, **kwargs):
        """Enhanced call that uses EnhancedDidYouMeanGroup."""
        click_group = original_call(*args, **kwargs)

        # If click_group is None (command already executed), return None
        # This happens after command execution completes successfully
        if click_group is None:
            return None

        # Create enhanced group
        enhanced_group = EnhancedDidYouMeanGroup(
            name=click_group.name,
            commands=click_group.commands,
            callback=click_group.callback,
            params=click_group.params,
            help=click_group.help,
            epilog=click_group.epilog,
            short_help=click_group.short_help,
            options_metavar=click_group.options_metavar,
            add_help_option=click_group.add_help_option,
            context_settings=click_group.context_settings,
            invoke_without_command=click_group.invoke_without_command,
            no_args_is_help=click_group.no_args_is_help,
            subcommand_metavar=click_group.subcommand_metavar,
            chain=click_group.chain,
            result_callback=click_group.result_callback,
            deprecated=click_group.deprecated,
        )

        return enhanced_group

    app.__call__ = enhanced_call
    return app


# Enhanced typo mapping with comprehensive variations
COMMON_TYPOS = {
    # Search command variations
    "serach": "search",
    "seach": "search",
    "searh": "search",
    "sarch": "search",
    "serch": "search",
    "searhc": "search",
    "find": "search",
    "query": "search",
    "lookup": "search",
    "grep": "search",
    "s": "search",  # Single letter shortcut
    "f": "search",  # Alternative shortcut for find
    # Chat command variations
    "cht": "chat",
    "caht": "chat",
    "chta": "chat",
    "ask": "chat",
    "question": "chat",
    "qa": "chat",
    "llm": "chat",
    "gpt": "chat",
    "explain": "chat",
    "answer": "chat",
    # Index command variations
    "indx": "index",
    "idx": "index",
    "indexx": "index",
    "indez": "index",
    "inedx": "index",
    "reindex": "index --force",
    "rebuild": "index --force",
    "refresh": "index --force",
    "scan": "index",
    "build": "index",
    "i": "index",  # Single letter shortcut
    "b": "index",  # Alternative shortcut for build
    # Status command variations
    "stat": "status",
    "stats": "status",
    "info": "status",
    "information": "status",
    "details": "status",
    "summary": "status",
    "overview": "status",
    "st": "status",  # Common abbreviation
    "status": "status",
    # Config command variations
    "conf": "config",
    "cfg": "config",
    "configure": "config",
    "configuration": "config",
    "setting": "config",
    "settings": "config",
    "preferences": "config",
    "prefs": "config",
    "c": "config",  # Single letter shortcut
    # Init command variations
    "initialize": "init",
    "setup": "init",
    "start": "init",
    "create": "init",
    "new": "init",
    "begin": "init",
    "initalize": "init",  # Common misspelling
    "initalise": "init",
    "initialise": "init",
    # Watch command variations
    "monitor": "watch",
    "observe": "watch",
    "track": "watch",
    "listen": "watch",
    "follow": "watch",
    "w": "watch",  # Single letter shortcut
    # Auto-index command variations
    "auto": "auto-index",
    "automatic": "auto-index",
    "autoindex": "auto-index",
    "auto_index": "auto-index",
    "ai": "auto-index",  # Abbreviation
    # MCP command variations
    "claude": "mcp",
    "server": "mcp",
    "protocol": "mcp",
    "model-context": "mcp",
    "context": "mcp",
    "m": "mcp",  # Single letter shortcut
    # Install command variations
    "deploy": "install",
    "add": "install",
    "instal": "install",  # Common typo
    "install": "install",
    # Demo command variations
    "example": "demo",
    "sample": "demo",
    "test": "demo",
    "try": "demo",
    "d": "demo",  # Single letter shortcut
    # Health/Doctor command variations
    "check": "doctor",
    "health": "doctor",
    "diagnose": "doctor",
    "verify": "doctor",
    "validate": "doctor",
    "repair": "doctor",
    "fix": "doctor",
    "dr": "doctor",  # Common abbreviation
    # Version command variations
    "ver": "version",
    "v": "version",
    "--version": "version",
    "-v": "version",
    # Help command variations
    "help": "--help",
    "h": "--help",
    "--help": "--help",
    "-h": "--help",
    "?": "--help",
    # History command variations
    "hist": "history",
    "log": "history",
    "logs": "history",
    "recent": "history",
    # Reset command variations
    "clear": "reset",
    "clean": "reset",
    "purge": "reset",
    "wipe": "reset",
    "remove": "reset",
    "delete": "reset",
    # Interactive command variations
    "interact": "interactive",
    "session": "interactive",
    "repl": "interactive",
    "console": "interactive",
    "ui": "interactive",
}

# Command descriptions and examples for better error messages
COMMAND_INFO = {
    "search": {
        "description": "Search for code patterns semantically",
        "examples": [
            'mcp-vector-search search "authentication function"',
            'mcp-vector-search search "error handling" --limit 5',
            'mcp-vector-search search --files "*.ts" "query"',
        ],
        "related": ["chat", "index", "status"],
    },
    "chat": {
        "description": "Ask AI questions about your code (requires API key)",
        "examples": [
            'mcp-vector-search chat "where is the database configured?"',
            'mcp-vector-search chat "how does authentication work?"',
            'mcp-vector-search chat --limit 3 "explain error handling"',
        ],
        "related": ["search", "status", "index"],
    },
    "index": {
        "description": "Index codebase for semantic search",
        "examples": [
            "mcp-vector-search index",
            "mcp-vector-search index --force",
            'mcp-vector-search index --include "*.py,*.js"',
        ],
        "related": ["auto-index", "watch", "reset"],
    },
    "status": {
        "description": "Show project status and statistics",
        "examples": ["mcp-vector-search status", "mcp-vector-search status --verbose"],
        "related": ["doctor", "history", "version"],
    },
    "config": {
        "description": "Manage project configuration",
        "examples": [
            "mcp-vector-search config show",
            "mcp-vector-search config set model all-MiniLM-L6-v2",
        ],
        "related": ["init", "status"],
    },
    "init": {
        "description": "Initialize project for semantic search",
        "examples": [
            "mcp-vector-search init",
            "mcp-vector-search init --model sentence-transformers/all-MiniLM-L6-v2",
        ],
        "related": ["config", "install", "index"],
    },
    "mcp": {
        "description": "Manage Claude Code MCP integration",
        "examples": ["mcp-vector-search mcp", "mcp-vector-search mcp test"],
        "related": ["init-mcp", "install"],
    },
    "doctor": {
        "description": "Check system dependencies and configuration",
        "examples": [
            "mcp-vector-search doctor",
        ],
        "related": ["status", "health"],
    },
    "version": {
        "description": "Show version information",
        "examples": ["mcp-vector-search version", "mcp-vector-search --version"],
        "related": ["status", "doctor"],
    },
}


def get_fuzzy_matches(
    command: str, available_commands: list[str], cutoff: float = 0.6
) -> list[tuple[str, float]]:
    """Get fuzzy matches for a command using difflib.

    Args:
        command: The command to match
        available_commands: List of available commands
        cutoff: Minimum similarity ratio (0.0 to 1.0)

    Returns:
        List of tuples (command, similarity_ratio) sorted by similarity
    """
    matches = []
    for cmd in available_commands:
        ratio = difflib.SequenceMatcher(None, command.lower(), cmd.lower()).ratio()
        if ratio >= cutoff:
            matches.append((cmd, ratio))

    # Sort by similarity ratio (highest first)
    return sorted(matches, key=lambda x: x[1], reverse=True)


def format_command_suggestion(command: str, show_examples: bool = True) -> str:
    """Format a command suggestion with description and examples.

    Args:
        command: The command to format
        show_examples: Whether to include usage examples

    Returns:
        Formatted suggestion string
    """
    if command in COMMAND_INFO:
        info = COMMAND_INFO[command]
        suggestion = f"  [bold cyan]{command}[/bold cyan] - {info['description']}"

        if show_examples and info["examples"]:
            suggestion += f"\n    Example: [dim]{info['examples'][0]}[/dim]"

        return suggestion
    else:
        return f"  [bold cyan]{command}[/bold cyan]"


def add_common_suggestions(ctx: click.Context, command_name: str) -> None:
    """Add enhanced command suggestions to error messages.

    Args:
        ctx: Click context
        command_name: The invalid command name that was entered
    """
    from rich.console import Console

    console = Console(stderr=True)

    # First, check for exact typo matches
    direct_suggestion = COMMON_TYPOS.get(command_name.lower())
    if direct_suggestion:
        console.print("\n[yellow]Did you mean:[/yellow]")
        console.print(format_command_suggestion(direct_suggestion.split()[0]))

        if "--" not in direct_suggestion:  # Don't show examples for flags
            console.print(
                f"\n[dim]Try: [bold]mcp-vector-search {direct_suggestion}[/bold][/dim]"
            )
        return

    # Get available commands from the context
    available_commands = []
    if hasattr(ctx, "command") and hasattr(ctx.command, "commands"):
        available_commands = list(ctx.command.commands.keys())

    if not available_commands:
        # Fallback to common commands
        available_commands = [
            "search",
            "index",
            "status",
            "config",
            "init",
            "mcp",
            "doctor",
            "version",
        ]

    # Get fuzzy matches
    fuzzy_matches = get_fuzzy_matches(command_name, available_commands, cutoff=0.4)

    if fuzzy_matches:
        console.print("\n[yellow]Did you mean one of these?[/yellow]")

        # Show up to 3 best matches
        for cmd, _ratio in fuzzy_matches[:3]:
            console.print(format_command_suggestion(cmd, show_examples=False))

        # Show example for the best match
        if fuzzy_matches:
            best_match = fuzzy_matches[0][0]
            if best_match in COMMAND_INFO and COMMAND_INFO[best_match]["examples"]:
                console.print(
                    f"\n[dim]Example: [bold]{COMMAND_INFO[best_match]['examples'][0]}[/bold][/dim]"
                )

    # Show related commands for context
    console.print(
        f"\n[dim]Available commands: {', '.join(sorted(available_commands))}[/dim]"
    )
    console.print(
        "[dim]Use [bold]mcp-vector-search --help[/bold] for more information[/dim]"
    )
