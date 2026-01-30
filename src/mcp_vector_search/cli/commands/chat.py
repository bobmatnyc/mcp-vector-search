"""Chat command for LLM-powered intelligent code search with interactive REPL."""

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function, suppress_stdout_stderr
from ...core.exceptions import ProjectNotFoundError
from ...core.llm_client import LLMClient
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import (
    console,
    print_error,
    print_warning,
)


def show_api_key_help() -> None:
    """Display helpful error message when API key is missing."""
    message = """[bold yellow]No LLM API Key Found[/bold yellow]

The chat feature requires an API key for an LLM provider.

[bold cyan]Set one of these environment variables:[/bold cyan]
  [green]OPENAI_API_KEY[/green]       - For OpenAI (GPT-4, etc.)
  [green]OPENROUTER_API_KEY[/green]  - For OpenRouter (Claude, GPT, etc.) [dim](recommended)[/dim]

[bold cyan]Example:[/bold cyan]
  [yellow]export OPENAI_API_KEY="sk-..."[/yellow]
  [yellow]export OPENROUTER_API_KEY="sk-or-..."[/yellow]

[bold cyan]Get API keys at:[/bold cyan]
  OpenAI: [link=https://platform.openai.com/api-keys]https://platform.openai.com/api-keys[/link]
  OpenRouter: [link=https://openrouter.ai/keys]https://openrouter.ai/keys[/link]

[dim]Alternatively, run: [cyan]mcp-vector-search setup[/cyan] for interactive setup[/dim]"""

    panel = Panel(
        message,
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(panel)


class EnhancedChatSession:
    """Enhanced session with 5-pair compaction and task tracking.

    Features:
    - Keeps last 5 user/assistant exchange pairs verbatim
    - Compacts older exchanges into history summary
    - Tracks current task context
    - Always preserves system prompt and task context
    """

    RECENT_EXCHANGES_TO_KEEP = 5  # Keep last 5 pairs

    def __init__(self, system_prompt: str) -> None:
        """Initialize session with system prompt.

        Args:
            system_prompt: Initial system message
        """
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []
        self.history_summary: str = ""
        self.current_task: dict[str, str] | None = None  # {description, status}

    def set_task(self, description: str) -> None:
        """Set the current task being worked on.

        Args:
            description: Task description
        """
        self.current_task = {
            "description": description,
            "status": "in_progress",
        }

    def update_task_status(self, status: str) -> None:
        """Update current task status.

        Args:
            status: New status (in_progress, completed, blocked)
        """
        if self.current_task:
            self.current_task["status"] = status

    def clear_task(self) -> None:
        """Clear the current task."""
        self.current_task = None

    def add_message(self, role: str, content: str) -> None:
        """Add message to history and compact if needed.

        Args:
            role: Message role (user/assistant/tool)
            content: Message content
        """
        self.messages.append({"role": role, "content": content})

        # Count user/assistant pairs
        pairs = sum(1 for m in self.messages if m["role"] == "user")

        # Compact on 6th pair (keep 5)
        if pairs > self.RECENT_EXCHANGES_TO_KEEP:
            self._compact_history()

    def add_tool_message(self, message: dict[str, Any]) -> None:
        """Add a tool call or tool result message.

        Args:
            message: Full message dict (may contain tool_calls)
        """
        self.messages.append(message)

    def _compact_history(self) -> None:
        """Compact conversation by summarizing oldest pair into history_summary."""
        logger.debug("Compacting conversation history")

        # Find the first user message and its response
        first_user_idx = None
        first_assistant_idx = None

        for i, msg in enumerate(self.messages):
            if msg["role"] == "user" and first_user_idx is None:
                first_user_idx = i
            elif msg["role"] == "assistant" and first_user_idx is not None:
                first_assistant_idx = i
                break

        if first_user_idx is None or first_assistant_idx is None:
            return

        # Extract the pair to summarize
        user_msg = self.messages[first_user_idx]["content"]
        assistant_msg = self.messages[first_assistant_idx].get("content", "")

        # Create summary of this exchange
        user_preview = user_msg[:150].replace("\n", " ")
        assistant_preview = (
            assistant_msg[:150].replace("\n", " ") if assistant_msg else "[tool calls]"
        )

        summary_entry = (
            f"- User asked: {user_preview}...\n  Assistant: {assistant_preview}..."
        )

        # Append to history summary
        if self.history_summary:
            self.history_summary += f"\n{summary_entry}"
        else:
            self.history_summary = summary_entry

        # Remove the compacted messages (including any tool messages between)
        # Find all messages up to and including the first assistant response
        messages_to_remove = first_assistant_idx + 1

        # Also remove any tool results that followed this exchange
        while (
            messages_to_remove < len(self.messages)
            and self.messages[messages_to_remove].get("role") == "tool"
        ):
            messages_to_remove += 1

        self.messages = self.messages[messages_to_remove:]

        logger.debug(
            f"Compacted {messages_to_remove} messages, history now has {len(self.messages)} messages"
        )

    def get_messages(self) -> list[dict[str, Any]]:
        """Build complete message list for API call.

        Structure: [system, history_summary?, task_context?, ...recent_messages]

        Returns:
            List of message dictionaries
        """
        result = [{"role": "system", "content": self.system_prompt}]

        # Add history summary if exists
        if self.history_summary:
            result.append(
                {
                    "role": "system",
                    "content": f"[Previous Conversation Summary]\n{self.history_summary}\n[End Summary]",
                }
            )

        # Add task context if exists
        if self.current_task:
            result.append(
                {
                    "role": "system",
                    "content": f"[Current Task]\nDescription: {self.current_task['description']}\nStatus: {self.current_task['status']}\n[End Task]",
                }
            )

        # Add recent messages
        result.extend(self.messages)

        return result

    def clear(self) -> None:
        """Clear conversation history, keeping only system prompt."""
        self.messages = []
        self.history_summary = ""
        # Keep current_task intact


# Create chat subcommand app with "did you mean" functionality
chat_app = create_enhanced_typer(
    help="Chat with your codebase using Claude Opus 4",
    invoke_without_command=True,
    no_args_is_help=False,  # Allow running without args to start REPL
)


def show_intro() -> None:
    """Display the REPL intro statement."""
    intro = """[bold cyan]MCP Vector Search - Chat[/bold cyan]

[dim]I'm here to help you understand and work with your codebase.[/dim]

[bold]What I can do:[/bold]
  Search and explain code semantically
  Answer questions about architecture and patterns
  Analyze code quality and complexity
  Write analysis reports to markdown files
  Search the web for documentation

[dim]I'll respond conversationally. Ask for code when you need it![/dim]

[bold]Commands:[/bold] /task, /status, /clear, /exit"""

    panel = Panel(
        intro,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


# Conversational system prompt
CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful code assistant. IMPORTANT GUIDELINES:

1. BE CONVERSATIONAL AND FRIENDLY
   - Explain concepts in plain language first
   - Use a natural, helpful tone
   - Don't be overly formal or robotic

2. DO NOT SHOW CODE UNLESS ASKED
   - Summarize search results rather than showing raw code
   - Describe what functions/classes do without dumping code
   - Only show code when user explicitly asks: "show me", "what's the code", "show the implementation"

3. WHEN SHOWING CODE
   - Use markdown code blocks with language hints
   - Keep snippets focused and relevant
   - Add brief explanations

4. TOOL USAGE
   - Use search_code to find relevant code
   - Use read_file for full file context
   - Use write_markdown to create reports
   - Use analyze_code for quality metrics
   - Use web_search for external documentation

5. TASK TRACKING
   - Track ONE task at a time
   - Update when user gives a new task
   - Reference the current task in your responses when relevant

Remember: Be helpful, conversational, and only show code when explicitly requested."""


@chat_app.callback(invoke_without_command=True)
def chat_main(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Initial question (or omit to start interactive REPL)",
    ),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        rich_help_panel="Global Options",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (default: claude-opus-4 via OpenRouter)",
        rich_help_panel="LLM Options",
    ),
    provider: str | None = typer.Option(
        "openrouter",
        "--provider",
        help="LLM provider: 'openai' or 'openrouter' (default: openrouter)",
        rich_help_panel="LLM Options",
    ),
    timeout: float | None = typer.Option(
        60.0,
        "--timeout",
        help="API timeout in seconds",
        min=5.0,
        max=300.0,
        rich_help_panel="LLM Options",
    ),
) -> None:
    """Interactive chat REPL powered by Claude Opus 4.

    Start an interactive session to explore and understand your codebase.
    Ask questions naturally - I'll explain concepts conversationally and
    only show code when you ask for it.

    [bold cyan]Quick Start:[/bold cyan]
        $ mcp-vector-search chat

    [bold cyan]With Initial Question:[/bold cyan]
        $ mcp-vector-search chat "how does the search work?"

    [bold cyan]Commands in REPL:[/bold cyan]
        /task <description>  - Set current task
        /status              - Show task and session status
        /clear               - Clear conversation history
        /exit                - Exit the REPL

    [bold cyan]Setup:[/bold cyan]
        $ export OPENROUTER_API_KEY="your-key"
        Get a key at: [cyan]https://openrouter.ai/keys[/cyan]
    """
    # A subcommand was invoked - let it handle the request
    if ctx.invoked_subcommand is not None:
        return

    # Get project root
    if project_root is None:
        if ctx.obj and isinstance(ctx.obj, dict):
            project_root = ctx.obj.get("project_root")
        if project_root is None:
            project_root = Path.cwd()

    try:
        if query:
            # Single query mode - process and exit
            asyncio.run(
                run_single_query(
                    project_root=project_root,
                    query=query,
                    model=model,
                    provider=provider or "openrouter",
                    timeout=timeout or 60.0,
                )
            )
        else:
            # No query - start interactive REPL
            asyncio.run(
                run_chat_repl(
                    project_root=project_root,
                    model=model,
                    provider=provider or "openrouter",
                    timeout=timeout or 60.0,
                )
            )
    except (typer.Exit, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        print_error(f"Chat failed: {e}")
        raise typer.Exit(1) from None


async def run_single_query(
    project_root: Path,
    query: str,
    model: str | None = None,
    provider: str = "openrouter",
    timeout: float = 60.0,
) -> None:
    """Run a single query and exit.

    Args:
        project_root: Project root directory
        query: User's question
        model: Model to use
        provider: LLM provider
        timeout: API timeout
    """
    from ...core.config_utils import get_openai_api_key, get_openrouter_api_key

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Validate provider
    if provider == "openai" and not openai_key:
        show_api_key_help()
        raise typer.Exit(1)
    elif provider == "openrouter" and not openrouter_key:
        show_api_key_help()
        raise typer.Exit(1)

    # Load project
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Initialize LLM client
    try:
        llm_client = LLMClient(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            model=model,
            provider=provider,
            timeout=timeout,
        )
        console.print(f"[dim]Using {llm_client.provider}: {llm_client.model}[/dim]\n")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Initialize search engine (suppress verbose model loading output)
    with suppress_stdout_stderr():
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

    # Create session for single query
    session = EnhancedChatSession(CONVERSATIONAL_SYSTEM_PROMPT)

    # Process query
    await _process_query(
        query=query,
        llm_client=llm_client,
        search_engine=search_engine,
        database=database,
        session=session,
        project_root=project_root,
        config=config,
    )


async def run_chat_repl(
    project_root: Path,
    model: str | None = None,
    provider: str = "openrouter",
    timeout: float = 60.0,
) -> None:
    """Run the interactive chat REPL.

    Args:
        project_root: Project root directory
        model: Model to use
        provider: LLM provider
        timeout: API timeout
    """
    from ...core.config_utils import get_openai_api_key, get_openrouter_api_key
    from ...core.search import SemanticSearchEngine

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Validate provider and keys
    if provider == "openai" and not openai_key:
        show_api_key_help()
        raise typer.Exit(1)
    elif provider == "openrouter" and not openrouter_key:
        show_api_key_help()
        raise typer.Exit(1)

    # Load project configuration
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Initialize LLM client
    try:
        llm_client = LLMClient(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            model=model,
            provider=provider,
            timeout=timeout,
        )
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Initialize search components (suppress verbose model loading output)
    with suppress_stdout_stderr():
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=config.similarity_threshold,
        )

    # Create session
    session = EnhancedChatSession(CONVERSATIONAL_SYSTEM_PROMPT)

    # Show intro
    show_intro()
    console.print(
        f"\n[dim]Connected to {llm_client.provider}: {llm_client.model}[/dim]"
    )
    console.print("[dim]Type your questions or use /exit to quit[/dim]\n")

    # REPL loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.startswith("/"):
                command = user_input.lower().split()[0]
                args = user_input[len(command) :].strip()

                if command in ("/exit", "/quit"):
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break

                elif command == "/clear":
                    session.clear()
                    console.print("[green]Conversation cleared.[/green]\n")
                    continue

                elif command == "/task":
                    if args:
                        session.set_task(args)
                        console.print(f"[green]Task set:[/green] {args}\n")
                    else:
                        console.print("[yellow]Usage: /task <description>[/yellow]\n")
                    continue

                elif command == "/status":
                    _show_status(session)
                    continue

                else:
                    console.print(f"[yellow]Unknown command: {command}[/yellow]")
                    console.print(
                        "[dim]Available: /task, /status, /clear, /exit[/dim]\n"
                    )
                    continue

            # Process query
            await _process_query(
                query=user_input,
                llm_client=llm_client,
                search_engine=search_engine,
                database=database,
                session=session,
                project_root=project_root,
                config=config,
            )

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except EOFError:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print_error(f"Error: {e}")


def _show_status(session: EnhancedChatSession) -> None:
    """Show session status.

    Args:
        session: Current chat session
    """
    console.print("\n[bold cyan]Session Status[/bold cyan]")

    # Task status
    if session.current_task:
        console.print(
            f"  [bold]Current Task:[/bold] {session.current_task['description']}"
        )
        console.print(f"  [bold]Status:[/bold] {session.current_task['status']}")
    else:
        console.print("  [dim]No active task[/dim]")

    # Conversation stats
    msg_count = len(session.messages)
    has_history = bool(session.history_summary)
    console.print(f"  [bold]Messages in context:[/bold] {msg_count}")
    console.print(
        f"  [bold]Has compacted history:[/bold] {'Yes' if has_history else 'No'}"
    )
    console.print()


def _get_tools() -> list[dict[str, Any]]:
    """Get tool definitions for the LLM.

    Returns:
        List of tool definitions
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search the codebase for relevant code using semantic search. Use this to find implementations, patterns, or answers to questions about the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'authentication logic', 'database connection')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 5, max: 10)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full content of a specific file. Use when you need complete context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to file from project root",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_markdown",
                "description": "Write a markdown report to a file. Use for analysis reports, documentation, or summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename for the report (e.g., 'analysis-report.md')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Markdown content to write",
                        },
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_code",
                "description": "Get code quality metrics for the project. Returns complexity, code smells, and recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Analysis focus: 'summary', 'complexity', 'smells', or 'all'",
                            "default": "summary",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for documentation, tutorials, or solutions. Use for external references.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files matching a pattern in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., '*.py', 'src/**/*.ts')",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
    ]


async def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    search_engine: Any,
    database: Any,
    project_root: Path,
    config: Any,
) -> str:
    """Execute a tool and return result.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        search_engine: Search engine instance
        database: Vector database
        project_root: Project root path
        config: Project config

    Returns:
        Tool execution result as string
    """
    if tool_name == "search_code":
        return await _tool_search_code(
            arguments.get("query", ""),
            arguments.get("limit", 5),
            search_engine,
            database,
            project_root,
            config,
        )

    elif tool_name == "read_file":
        return await _tool_read_file(
            arguments.get("file_path", ""),
            project_root,
        )

    elif tool_name == "write_markdown":
        return await _tool_write_markdown(
            arguments.get("filename", "report.md"),
            arguments.get("content", ""),
            project_root,
        )

    elif tool_name == "analyze_code":
        return await _tool_analyze_code(
            arguments.get("focus", "summary"),
            project_root,
            config,
        )

    elif tool_name == "web_search":
        return await _tool_web_search(
            arguments.get("query", ""),
        )

    elif tool_name == "list_files":
        return await _tool_list_files(
            arguments.get("pattern", "*"),
            project_root,
        )

    else:
        return f"Error: Unknown tool '{tool_name}'"


async def _tool_search_code(
    query: str,
    limit: int,
    search_engine: Any,
    database: Any,
    project_root: Path,
    config: Any,
) -> str:
    """Execute search_code tool."""
    try:
        limit = min(limit, 10)
        async with database:
            results = await search_engine.search(
                query=query,
                limit=limit,
                similarity_threshold=config.similarity_threshold,
                include_context=True,
            )

        if not results:
            return "No results found for this query."

        # Format results
        parts = []
        for i, result in enumerate(results, 1):
            try:
                rel_path = str(result.file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(result.file_path)

            parts.append(
                f"[Result {i}: {rel_path}]\n"
                f"Location: {result.location}\n"
                f"Lines {result.start_line}-{result.end_line}\n"
                f"Similarity: {result.similarity_score:.3f}\n"
                f"```\n{result.content}\n```\n"
            )
        return "\n".join(parts)

    except Exception as e:
        logger.error(f"search_code failed: {e}")
        return f"Error searching code: {e}"


async def _tool_read_file(file_path: str, project_root: Path) -> str:
    """Execute read_file tool."""
    try:
        if file_path.startswith("/"):
            full_path = Path(file_path)
        else:
            full_path = project_root / file_path

        # Security check
        try:
            full_path.relative_to(project_root)
        except ValueError:
            return "Error: File must be within project root"

        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        if not full_path.is_file():
            return f"Error: Not a file: {file_path}"

        # Size limit
        max_size = 100_000
        if full_path.stat().st_size > max_size:
            return "Error: File too large. Use search_code instead."

        content = full_path.read_text(errors="replace")
        return f"File: {file_path}\n```\n{content}\n```"

    except Exception as e:
        logger.error(f"read_file failed: {e}")
        return f"Error reading file: {e}"


async def _tool_write_markdown(filename: str, content: str, project_root: Path) -> str:
    """Execute write_markdown tool."""
    try:
        # Ensure filename is safe
        if "/" in filename or "\\" in filename:
            filename = Path(filename).name

        if not filename.endswith(".md"):
            filename += ".md"

        # Write to reports directory
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        output_path = reports_dir / filename
        output_path.write_text(content)

        return f"Report written to: {output_path.relative_to(project_root)}"

    except Exception as e:
        logger.error(f"write_markdown failed: {e}")
        return f"Error writing report: {e}"


async def _tool_analyze_code(focus: str, project_root: Path, config: Any) -> str:
    """Execute analyze_code tool."""
    try:
        from ...analysis import ProjectMetrics
        from ...analysis.interpretation import EnhancedJSONExporter
        from ...parsers.registry import ParserRegistry

        parser_registry = ParserRegistry()
        project_metrics = ProjectMetrics(project_root=str(project_root))

        # Parse files
        for file_ext in config.file_extensions:
            parser = parser_registry.get_parser(file_ext)
            if parser:
                for file_path in project_root.rglob(f"*{file_ext}"):
                    should_skip = any(
                        ignore in str(file_path) for ignore in config.ignore_patterns
                    )
                    if should_skip:
                        continue

                    try:
                        chunks = parser.parse_file(file_path)
                        project_metrics.add_file(file_path, chunks)
                    except Exception:
                        pass

        # Generate metrics
        exporter = EnhancedJSONExporter(project_root=project_root)
        export = exporter.export_with_context(
            project_metrics,
            include_smells=(focus in ("smells", "all")),
        )

        # Format based on focus
        if focus == "summary":
            summary = export.project_summary
            return f"""Project Analysis Summary:
- Files analyzed: {summary.get("total_files", "N/A")}
- Total functions: {summary.get("total_functions", "N/A")}
- Average complexity: {summary.get("average_complexity", "N/A")}
- Health grade: {summary.get("health_grade", "N/A")}"""

        elif focus == "complexity":
            hotspots = export.model_dump().get("complexity_hotspots", [])[:5]
            if not hotspots:
                return "No complexity hotspots found."
            lines = ["Top complexity hotspots:"]
            for h in hotspots:
                lines.append(
                    f"- {h.get('name', 'Unknown')}: complexity {h.get('complexity', 'N/A')}"
                )
            return "\n".join(lines)

        elif focus == "smells":
            smells = export.model_dump().get("code_smells", [])[:10]
            if not smells:
                return "No code smells detected."
            lines = ["Code smells detected:"]
            for s in smells:
                lines.append(
                    f"- [{s.get('severity', 'info')}] {s.get('smell_type', 'Unknown')}: {s.get('message', '')}"
                )
            return "\n".join(lines)

        else:  # all
            return json.dumps(export.model_dump(), indent=2)[:5000]

    except Exception as e:
        logger.error(f"analyze_code failed: {e}")
        return f"Error analyzing code: {e}"


async def _tool_web_search(query: str) -> str:
    """Execute web_search tool."""
    try:
        import httpx

        # Use DuckDuckGo instant answers API (no auth needed)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                },
            )

            if response.status_code != 200:
                return f"Web search failed: HTTP {response.status_code}"

            data = response.json()

            # Extract useful info
            parts = []

            # Abstract (main result)
            if data.get("Abstract"):
                parts.append(f"Summary: {data['Abstract']}")
                if data.get("AbstractURL"):
                    parts.append(f"Source: {data['AbstractURL']}")

            # Related topics
            related = data.get("RelatedTopics", [])[:3]
            if related:
                parts.append("\nRelated:")
                for topic in related:
                    if isinstance(topic, dict) and topic.get("Text"):
                        parts.append(f"- {topic['Text'][:200]}")

            if not parts:
                return f"No direct results found. Try searching: https://www.google.com/search?q={query.replace(' ', '+')}"

            return "\n".join(parts)

    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"Web search error: {e}. Try manual search."


async def _tool_list_files(pattern: str, project_root: Path) -> str:
    """Execute list_files tool."""
    try:
        from glob import glob

        matches = glob(str(project_root / pattern), recursive=True)

        if not matches:
            return f"No files found matching: {pattern}"

        rel_paths = []
        for match in matches[:50]:
            try:
                rel_path = Path(match).relative_to(project_root)
                rel_paths.append(str(rel_path))
            except ValueError:
                continue

        if not rel_paths:
            return f"No files found matching: {pattern}"

        return f"Files matching '{pattern}':\n" + "\n".join(
            f"- {p}" for p in sorted(rel_paths)
        )

    except Exception as e:
        logger.error(f"list_files failed: {e}")
        return f"Error listing files: {e}"


async def _process_query(
    query: str,
    llm_client: LLMClient,
    search_engine: Any,
    database: Any,
    session: EnhancedChatSession,
    project_root: Path,
    config: Any,
) -> None:
    """Process a user query with tool use.

    Args:
        query: User's question
        llm_client: LLM client
        search_engine: Search engine
        database: Vector database
        session: Chat session
        project_root: Project root
        config: Project config
    """
    tools = _get_tools()

    # Add user message to session
    session.add_message("user", query)

    # Get conversation context
    messages = session.get_messages()

    # Agentic tool loop
    max_iterations = 15
    for _iteration in range(max_iterations):
        try:
            response = await llm_client.chat_with_tools(messages, tools)

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # Add assistant message with tool calls
                messages.append(message)

                # Execute tools
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    function = tool_call.get("function", {})
                    function_name = function.get("name")
                    arguments_str = function.get("arguments", "{}")

                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Show tool usage (show progress dot for list_files to reduce noise)
                    if function_name == "list_files":
                        console.print(".", end="", style="dim")
                    else:
                        args_display = ", ".join(
                            f"{k}={repr(v)[:30]}" for k, v in arguments.items()
                        )
                        console.print(f"[dim]{function_name}({args_display})[/dim]")

                    # Execute tool
                    result = await _execute_tool(
                        function_name,
                        arguments,
                        search_engine,
                        database,
                        project_root,
                        config,
                    )

                    # Add tool result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                    )

            else:
                # Final response - no more tool calls
                final_content = message.get("content", "")

                if not final_content:
                    print_error("Empty response from LLM")
                    return

                # Display response
                console.print("\n[bold cyan]Assistant:[/bold cyan]\n")

                with Live(
                    "", console=console, auto_refresh=True, vertical_overflow="visible"
                ) as live:
                    live.update(Markdown(final_content))

                console.print()

                # Add to session
                session.add_message("assistant", final_content)

                return

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            print_error(f"Error: {e}")
            return

    # Max iterations reached
    print_warning("Maximum tool iterations reached.")


if __name__ == "__main__":
    chat_app()
