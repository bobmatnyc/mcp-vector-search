"""Enhanced CLI with 'did you mean' functionality for better user experience."""

import click
import typer
from click_didyoumean import DYMGroup
from typing import Any, Dict, Optional


class DidYouMeanTyper(typer.Typer):
    """Enhanced Typer class with 'did you mean' functionality."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with did-you-mean support."""
        # Extract Typer-specific kwargs
        typer_kwargs = {}
        click_kwargs = {}
        
        # Separate Typer and Click kwargs
        typer_specific = {
            'name', 'help', 'epilog', 'short_help', 'options_metavar',
            'add_completion', 'context_settings', 'callback', 'invoke_without_command',
            'no_args_is_help', 'subcommand_metavar', 'chain', 'result_callback',
            'deprecated', 'rich_markup_mode', 'rich_help_panel', 'pretty_exceptions_enable',
            'pretty_exceptions_show_locals', 'pretty_exceptions_short'
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
    
    def __call__(self, *args, **kwargs):
        """Override call to use DYMGroup."""
        # Get the underlying click group
        click_group = super().__call__(*args, **kwargs)
        
        # Create a new DYMGroup with the same properties
        dym_group = DYMGroup(
            name=click_group.name,
            commands=click_group.commands,
            **self._click_kwargs
        )
        
        # Copy all attributes from the original group
        for attr in dir(click_group):
            if not attr.startswith('_') and attr not in ['commands', 'name']:
                try:
                    setattr(dym_group, attr, getattr(click_group, attr))
                except (AttributeError, TypeError):
                    # Skip attributes that can't be set
                    pass
        
        return dym_group


class DidYouMeanGroup(DYMGroup):
    """Custom Click group with enhanced 'did you mean' functionality."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with better error messages."""
        super().__init__(*args, **kwargs)
    
    def resolve_command(self, ctx: click.Context, args: list) -> tuple:
        """Resolve command with enhanced error handling."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            # Enhance the error message with available commands
            if "No such command" in str(e):
                available_commands = list(self.commands.keys())
                if available_commands:
                    commands_list = ", ".join(f"'{cmd}'" for cmd in sorted(available_commands))
                    enhanced_msg = f"{str(e)}\n\nAvailable commands: {commands_list}"
                    raise click.UsageError(enhanced_msg, ctx=ctx)
            raise


def create_enhanced_typer(**kwargs) -> typer.Typer:
    """Create a Typer instance with 'did you mean' functionality."""
    # Set default values for better UX
    defaults = {
        'no_args_is_help': True,
        'add_completion': False,
        'rich_markup_mode': 'rich',
    }
    
    # Merge with provided kwargs
    final_kwargs = {**defaults, **kwargs}
    
    # Create the enhanced Typer
    app = DidYouMeanTyper(**final_kwargs)
    
    return app


def enhance_existing_typer(app: typer.Typer) -> typer.Typer:
    """Enhance an existing Typer app with 'did you mean' functionality."""
    # This is a bit tricky since we need to modify the underlying Click group
    # We'll create a wrapper that intercepts the click group creation
    
    original_call = app.__call__
    
    def enhanced_call(*args, **kwargs):
        """Enhanced call that uses DYMGroup."""
        click_group = original_call(*args, **kwargs)
        
        # Create enhanced group
        enhanced_group = DidYouMeanGroup(
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


def add_common_suggestions(ctx: click.Context, command_name: str) -> None:
    """Add common command suggestions to error messages."""
    common_typos = {
        'serach': 'search',
        'seach': 'search',
        'searh': 'search',
        'find': 'search',
        'indx': 'index',
        'idx': 'index',
        'reindex': 'index --force',
        'stat': 'status',
        'stats': 'status',
        'info': 'status',
        'conf': 'config',
        'cfg': 'config',
        'setting': 'config',
        'settings': 'config',
        'init': 'init',
        'initialize': 'init',
        'setup': 'init',
        'start': 'init',
        'watch': 'watch',
        'monitor': 'watch',
        'auto': 'auto-index',
        'automatic': 'auto-index',
        'mcp': 'mcp',
        'claude': 'mcp',
        'server': 'mcp',
        'install': 'install',
        'setup': 'install',
        'demo': 'demo',
        'example': 'demo',
        'test': 'mcp test',
        'check': 'status',
        'doctor': 'doctor',
        'health': 'doctor',
        'version': 'version',
        'ver': 'version',
        'help': '--help',
        'h': '--help',
    }
    
    if command_name.lower() in common_typos:
        suggestion = common_typos[command_name.lower()]
        click.echo(f"\nDid you mean: mcp-vector-search {suggestion}?", err=True)
