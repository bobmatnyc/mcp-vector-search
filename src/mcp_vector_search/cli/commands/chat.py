"""Chat command for LLM-powered intelligent code search."""

import asyncio
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError, SearchError
from ...core.llm_client import LLMClient
from ...core.project import ProjectManager
from ...core.search import SemanticSearchEngine
from ..didyoumean import create_enhanced_typer
from ..output import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)


def show_api_key_help() -> None:
    """Display helpful error message when API key is missing."""
    message = """[bold yellow]⚠️  No LLM API Key Found[/bold yellow]

The chat feature requires an API key for an LLM provider.

[bold cyan]Set one of these environment variables:[/bold cyan]
  • [green]OPENAI_API_KEY[/green]       - For OpenAI (GPT-4, etc.) [dim](recommended)[/dim]
  • [green]OPENROUTER_API_KEY[/green]  - For OpenRouter (Claude, GPT, etc.)

[bold cyan]Example:[/bold cyan]
  [yellow]export OPENAI_API_KEY="sk-..."[/yellow]
  [yellow]export OPENROUTER_API_KEY="sk-or-..."[/yellow]

[bold cyan]Get API keys at:[/bold cyan]
  • OpenAI: [link=https://platform.openai.com/api-keys]https://platform.openai.com/api-keys[/link]
  • OpenRouter: [link=https://openrouter.ai/keys]https://openrouter.ai/keys[/link]

[dim]Alternatively, run: [cyan]mcp-vector-search setup[/cyan] for interactive setup[/dim]"""

    panel = Panel(
        message,
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(panel)


class ChatSession:
    """Manages conversation history with automatic compaction.

    Keeps system prompt intact, compacts older messages when history grows large,
    and maintains recent exchanges for context.
    """

    # Threshold for compaction (estimated tokens, ~4 chars per token)
    COMPACTION_THRESHOLD = 8000 * 4  # ~32000 chars
    RECENT_EXCHANGES_TO_KEEP = 3  # Keep last N user/assistant pairs

    def __init__(self, system_prompt: str) -> None:
        """Initialize session with system prompt.

        Args:
            system_prompt: Initial system message
        """
        self.system_prompt = system_prompt
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_message(self, role: str, content: str) -> None:
        """Add message to history and compact if needed.

        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        self.messages.append({"role": role, "content": content})

        # Check if compaction needed
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        if total_chars > self.COMPACTION_THRESHOLD:
            self._compact_history()

    def _compact_history(self) -> None:
        """Compact conversation history by summarizing older exchanges.

        Strategy:
        1. Keep system prompt intact
        2. Summarize older exchanges into brief context
        3. Keep recent N exchanges verbatim
        """
        logger.debug("Compacting conversation history")

        # Separate system prompt and conversation
        system_msg = self.messages[0]
        conversation = self.messages[1:]

        # Keep recent exchanges (last N user/assistant pairs)
        recent_start = max(0, len(conversation) - (self.RECENT_EXCHANGES_TO_KEEP * 2))
        older_messages = conversation[:recent_start]
        recent_messages = conversation[recent_start:]

        # Summarize older messages
        if older_messages:
            summary_parts = []
            for msg in older_messages:
                role = msg["role"].capitalize()
                content_preview = msg["content"][:100].replace("\n", " ")
                summary_parts.append(f"{role}: {content_preview}...")

            summary = "\n".join(summary_parts)
            summary_msg = {
                "role": "system",
                "content": f"[Previous conversation summary]\n{summary}\n[End summary]",
            }

            # Rebuild messages: system + summary + recent
            self.messages = [system_msg, summary_msg] + recent_messages

            logger.debug(
                f"Compacted {len(older_messages)} old messages, kept {len(recent_messages)} recent"
            )

    def get_messages(self) -> list[dict[str, str]]:
        """Get current message history.

        Returns:
            List of message dictionaries
        """
        return self.messages.copy()

    def clear(self) -> None:
        """Clear conversation history, keeping only system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]


# Create chat subcommand app with "did you mean" functionality
chat_app = create_enhanced_typer(
    help="🤖 LLM-powered intelligent code search",
    invoke_without_command=True,
)


@chat_app.callback(invoke_without_command=True)
def chat_main(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Natural language query about your code",
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
        rich_help_panel="🔧 Global Options",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum number of results to return",
        min=1,
        max=20,
        rich_help_panel="📊 Result Options",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (defaults based on provider: gpt-4o-mini for OpenAI, claude-3-haiku for OpenRouter)",
        rich_help_panel="🤖 LLM Options",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="LLM provider to use: 'openai' or 'openrouter' (auto-detect if not specified)",
        rich_help_panel="🤖 LLM Options",
    ),
    timeout: float | None = typer.Option(
        30.0,
        "--timeout",
        help="API timeout in seconds",
        min=5.0,
        max=120.0,
        rich_help_panel="🤖 LLM Options",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results in JSON format",
        rich_help_panel="📊 Result Options",
    ),
    files: str | None = typer.Option(
        None,
        "--files",
        "-f",
        help="Filter by file glob patterns (e.g., '*.py', 'src/*.js'). Matches basename or relative path.",
        rich_help_panel="🔍 Filters",
    ),
    think: bool = typer.Option(
        False,
        "--think",
        "-t",
        help="Use advanced model for complex queries (gpt-4o / claude-sonnet-4). Better reasoning, higher cost.",
        rich_help_panel="🤖 LLM Options",
    ),
) -> None:
    """🤖 Ask questions about your code in natural language.

    Uses LLM (OpenAI or OpenRouter) to intelligently search your codebase and answer
    questions like "where is X defined?", "how does Y work?", etc.

    [bold cyan]Setup:[/bold cyan]

    [green]Option A - OpenAI (recommended):[/green]
        $ export OPENAI_API_KEY="your-key-here"
        Get a key at: [cyan]https://platform.openai.com/api-keys[/cyan]

    [green]Option B - OpenRouter:[/green]
        $ export OPENROUTER_API_KEY="your-key-here"
        Get a key at: [cyan]https://openrouter.ai/keys[/cyan]

    [dim]Provider is auto-detected. OpenAI is preferred if both keys are set.[/dim]

    [bold cyan]Examples:[/bold cyan]

    [green]Ask where a parameter is set:[/green]
        $ mcp-vector-search chat "where is similarity_threshold set?"

    [green]Ask how something works:[/green]
        $ mcp-vector-search chat "how does the indexing process work?"

    [green]Find implementation details:[/green]
        $ mcp-vector-search chat "show me the search ranking algorithm"

    [green]Force specific provider:[/green]
        $ mcp-vector-search chat "question" --provider openai
        $ mcp-vector-search chat "question" --provider openrouter

    [green]Use custom model:[/green]
        $ mcp-vector-search chat "question" --model gpt-4o
        $ mcp-vector-search chat "question" --model anthropic/claude-3.5-sonnet

    [bold cyan]Advanced:[/bold cyan]

    [green]Filter by file pattern:[/green]
        $ mcp-vector-search chat "how does validation work?" --files "*.py"
        $ mcp-vector-search chat "find React components" --files "src/*.tsx"

    [green]Limit results:[/green]
        $ mcp-vector-search chat "find auth code" --limit 3

    [green]Custom timeout:[/green]
        $ mcp-vector-search chat "complex question" --timeout 60

    [green]Use advanced model for complex queries:[/green]
        $ mcp-vector-search chat "explain the authentication flow" --think

    [dim]💡 Tip: Use --think for complex architectural questions. It uses gpt-4o or
    claude-sonnet-4 for better reasoning at higher cost.[/dim]
    """
    # If no query provided and no subcommand invoked, exit (show help)
    if query is None:
        if ctx.invoked_subcommand is None:
            # No query and no subcommand - show help
            raise typer.Exit()
        else:
            # A subcommand was invoked - let it handle the request
            return

    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()

        # Validate provider if specified
        if provider and provider not in ("openai", "openrouter"):
            print_error(
                f"Invalid provider: {provider}. Must be 'openai' or 'openrouter'"
            )
            raise typer.Exit(1)

        # Run the chat with intent detection and routing
        asyncio.run(
            run_chat_with_intent(
                project_root=project_root,
                query=query,
                limit=limit,
                model=model,
                provider=provider,
                timeout=timeout,
                json_output=json_output,
                files=files,
                think=think,
            )
        )

    except (typer.Exit, SystemExit):
        # Re-raise exit exceptions without printing additional error messages
        # The error message has already been shown to the user
        raise
    except Exception as e:
        #  Log real exceptions (not typer.Exit)
        if not isinstance(e, (typer.Exit, SystemExit)):
            logger.error(f"Chat failed: {e}")
            print_error(f"Chat failed: {e}")
        raise typer.Exit(
            1
        ) from None  # Suppress exception chain to avoid double-printing


async def run_chat_with_intent(
    project_root: Path,
    query: str,
    limit: int = 5,
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 30.0,
    json_output: bool = False,
    files: str | None = None,
    think: bool = False,
) -> None:
    """Route to appropriate chat mode based on detected intent.

    Args:
        project_root: Project root directory
        query: User's natural language query
        limit: Maximum results to return
        model: Model to use (optional)
        provider: LLM provider
        timeout: API timeout
        json_output: Whether to output JSON
        files: File pattern filter
        think: Use advanced model
    """
    # Initialize LLM client for intent detection
    from ...core.config_utils import (
        get_openai_api_key,
        get_openrouter_api_key,
        get_preferred_llm_provider,
    )

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Determine provider (same logic as before)
    if not provider:
        preferred_provider = get_preferred_llm_provider(config_dir)
        if preferred_provider == "openai" and openai_key:
            provider = "openai"
        elif preferred_provider == "openrouter" and openrouter_key:
            provider = "openrouter"
        elif openai_key:
            provider = "openai"
        elif openrouter_key:
            provider = "openrouter"
        else:
            console.print()  # Blank line for spacing
            show_api_key_help()
            raise typer.Exit(1)

    # Create temporary client for intent detection (use fast model)
    try:
        intent_client = LLMClient(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            provider=provider,
            timeout=timeout,
            think=False,  # Use fast model for intent detection
        )

        # Detect intent
        intent = await intent_client.detect_intent(query)

        # Show intent to user
        if intent == "find":
            console.print("\n[cyan]🔍 Intent: Find[/cyan] - Searching codebase\n")
            await run_chat_search(
                project_root=project_root,
                query=query,
                limit=limit,
                model=model,
                provider=provider,
                timeout=timeout,
                json_output=json_output,
                files=files,
                think=think,
            )
        else:
            # Answer mode - force think mode and enter interactive session
            console.print(
                "\n[cyan]💬 Intent: Answer[/cyan] - Entering interactive mode\n"
            )
            await run_chat_answer(
                project_root=project_root,
                initial_query=query,
                limit=limit,
                model=model,
                provider=provider,
                timeout=timeout,
                files=files,
            )

    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        # Default to find mode on error
        console.print("\n[yellow]⚠ Using default search mode[/yellow]\n")
        await run_chat_search(
            project_root=project_root,
            query=query,
            limit=limit,
            model=model,
            provider=provider,
            timeout=timeout,
            json_output=json_output,
            files=files,
            think=think,
        )


async def run_chat_answer(
    project_root: Path,
    initial_query: str,
    limit: int = 5,
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 30.0,
    files: str | None = None,
) -> None:
    """Run interactive answer mode with streaming responses.

    Args:
        project_root: Project root directory
        initial_query: Initial user question
        limit: Max search results for context
        model: Model to use (optional, defaults to advanced model)
        provider: LLM provider
        timeout: API timeout
        files: File pattern filter
    """
    from ...core.config_utils import get_openai_api_key, get_openrouter_api_key

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Load project configuration
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Initialize LLM client with advanced model (force think mode)
    try:
        llm_client = LLMClient(
            openai_api_key=openai_key,
            openrouter_api_key=openrouter_key,
            model=model,
            provider=provider,
            timeout=timeout,
            think=True,  # Always use advanced model for answer mode
        )
        provider_display = llm_client.provider.capitalize()
        model_info = f"{llm_client.model} [bold magenta](thinking mode)[/bold magenta]"
        print_success(f"Connected to {provider_display}: {model_info}")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Initialize search engine
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

    # Initialize session (cleared on startup)
    system_prompt = """You are a helpful code assistant analyzing a codebase. Answer questions based on provided code context.

Guidelines:
- Be concise but thorough
- Reference specific functions, classes, or files
- Use code examples when helpful
- If context is insufficient, say so
- Use markdown formatting"""

    session = ChatSession(system_prompt)

    # Process initial query
    await _process_answer_query(
        query=initial_query,
        llm_client=llm_client,
        search_engine=search_engine,
        database=database,
        session=session,
        project_root=project_root,
        limit=limit,
        files=files,
        config=config,
    )

    # Interactive loop
    console.print("\n[dim]Type your questions or '/exit' to quit[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # Check for exit command
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("\n[cyan]👋 Session ended.[/cyan]")
                break

            # Process query
            await _process_answer_query(
                query=user_input,
                llm_client=llm_client,
                search_engine=search_engine,
                database=database,
                session=session,
                project_root=project_root,
                limit=limit,
                files=files,
                config=config,
            )

        except KeyboardInterrupt:
            console.print("\n\n[cyan]👋 Session ended.[/cyan]")
            break
        except EOFError:
            console.print("\n\n[cyan]👋 Session ended.[/cyan]")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print_error(f"Error: {e}")


async def _process_answer_query(
    query: str,
    llm_client: LLMClient,
    search_engine: SemanticSearchEngine,
    database: ChromaVectorDatabase,
    session: ChatSession,
    project_root: Path,
    limit: int,
    files: str | None,
    config: Any,
) -> None:
    """Process a single answer query with agentic tool use.

    Args:
        query: User query
        llm_client: LLM client instance
        search_engine: Search engine instance
        database: Vector database
        session: Chat session
        project_root: Project root path
        limit: Max results
        files: File pattern filter
        config: Project config
    """
    # Define search tools for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search the codebase for relevant code snippets using semantic search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant code (e.g., 'authentication logic', 'database connection', 'error handling')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5, max: 10)",
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
                "description": "Read the full content of a specific file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to the file from project root",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in the codebase matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to match files (e.g., '*.py', 'src/**/*.ts', 'tests/')",
                        }
                    },
                    "required": ["pattern"],
                },
            },
        },
    ]

    # System prompt for tool use
    system_prompt = """You are a helpful code assistant with access to search tools. Use these tools to find and analyze code in the codebase.

Available tools:
- search_code: Search for relevant code using semantic search
- read_file: Read the full content of a specific file
- list_files: List files matching a pattern

Guidelines:
1. Use search_code to find relevant code snippets
2. Use read_file when you need to see the full file context
3. Use list_files to understand the project structure
4. Make multiple searches if needed to gather enough context
5. After gathering sufficient information, provide your analysis

Always base your answers on actual code from the tools. If you can't find relevant code, say so."""

    # Tool execution functions
    async def execute_search_code(query_str: str, limit_val: int = 5) -> str:
        """Execute search_code tool."""
        try:
            limit_val = min(limit_val, 10)  # Cap at 10
            async with database:
                results = await search_engine.search(
                    query=query_str,
                    limit=limit_val,
                    similarity_threshold=config.similarity_threshold,
                    include_context=True,
                )

                # Post-filter by file pattern if specified
                if files and results:
                    filtered_results = []
                    for result in results:
                        try:
                            rel_path = str(result.file_path.relative_to(project_root))
                        except ValueError:
                            rel_path = str(result.file_path)

                        if fnmatch(rel_path, files) or fnmatch(
                            os.path.basename(rel_path), files
                        ):
                            filtered_results.append(result)
                    results = filtered_results

            if not results:
                return "No results found for this query."

            # Format results
            result_parts = []
            for i, result in enumerate(results, 1):
                try:
                    rel_path = str(result.file_path.relative_to(project_root))
                except ValueError:
                    rel_path = str(result.file_path)

                result_parts.append(
                    f"[Result {i}: {rel_path}]\n"
                    f"Location: {result.location}\n"
                    f"Lines {result.start_line}-{result.end_line}\n"
                    f"Similarity: {result.similarity_score:.3f}\n"
                    f"```\n{result.content}\n```\n"
                )
            return "\n".join(result_parts)

        except Exception as e:
            logger.error(f"search_code tool failed: {e}")
            return f"Error searching code: {e}"

    async def execute_read_file(file_path: str) -> str:
        """Execute read_file tool."""
        try:
            # Normalize path
            if file_path.startswith("/"):
                full_path = Path(file_path)
            else:
                full_path = project_root / file_path

            # Security check: file must be within project
            try:
                full_path.relative_to(project_root)
            except ValueError:
                return f"Error: File must be within project root: {project_root}"

            if not full_path.exists():
                return f"Error: File not found: {file_path}"

            if not full_path.is_file():
                return f"Error: Not a file: {file_path}"

            # Read file with size limit
            max_size = 100_000  # 100KB
            file_size = full_path.stat().st_size
            if file_size > max_size:
                return f"Error: File too large ({file_size} bytes). Use search_code instead."

            content = full_path.read_text(errors="replace")
            return f"File: {file_path}\n```\n{content}\n```"

        except Exception as e:
            logger.error(f"read_file tool failed: {e}")
            return f"Error reading file: {e}"

    async def execute_list_files(pattern: str) -> str:
        """Execute list_files tool."""
        try:
            from glob import glob

            # Use glob to find matching files
            matches = glob(str(project_root / pattern), recursive=True)

            if not matches:
                return f"No files found matching pattern: {pattern}"

            # Get relative paths and limit results
            rel_paths = []
            for match in matches[:50]:  # Limit to 50 files
                try:
                    rel_path = Path(match).relative_to(project_root)
                    rel_paths.append(str(rel_path))
                except ValueError:
                    continue

            if not rel_paths:
                return f"No files found matching pattern: {pattern}"

            return f"Files matching '{pattern}':\n" + "\n".join(
                f"- {p}" for p in sorted(rel_paths)
            )

        except Exception as e:
            logger.error(f"list_files tool failed: {e}")
            return f"Error listing files: {e}"

    # Get conversation history
    conversation_history = session.get_messages()[1:]  # Skip system prompt

    # Build messages: system + history + current query
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})

    # Agentic loop
    max_iterations = 10
    for _iteration in range(max_iterations):
        try:
            response = await llm_client.chat_with_tools(messages, tools)

            # Extract message from response
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # Add assistant message with tool calls
                messages.append(message)

                # Execute each tool call
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    function = tool_call.get("function", {})
                    function_name = function.get("name")
                    arguments_str = function.get("arguments", "{}")

                    # Parse arguments
                    try:
                        import json

                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Display tool usage
                    console.print(
                        f"\n[dim]🔧 Using tool: {function_name}({', '.join(f'{k}={repr(v)}' for k, v in arguments.items())})[/dim]"
                    )

                    # Execute tool
                    if function_name == "search_code":
                        result = await execute_search_code(
                            arguments.get("query", ""),
                            arguments.get("limit", 5),
                        )
                        console.print(
                            f"[dim]   Found {len(result.split('[Result')) - 1} results[/dim]"
                        )
                    elif function_name == "read_file":
                        result = await execute_read_file(arguments.get("file_path", ""))
                        console.print("[dim]   Read file[/dim]")
                    elif function_name == "list_files":
                        result = await execute_list_files(arguments.get("pattern", ""))
                        console.print("[dim]   Listed files[/dim]")
                    else:
                        result = f"Error: Unknown tool: {function_name}"

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                    )

            else:
                # No tool calls - final response
                final_content = message.get("content", "")

                if not final_content:
                    print_error("LLM returned empty response")
                    return

                # Stream the final response
                console.print("\n[bold cyan]🤖 Assistant:[/bold cyan]\n")

                # Use Rich Live for rendering
                with Live("", console=console, auto_refresh=True) as live:
                    live.update(Markdown(final_content))

                # Add to session history
                session.add_message("user", query)
                session.add_message("assistant", final_content)

                return

        except Exception as e:
            logger.error(f"Tool execution loop failed: {e}")
            print_error(f"Error: {e}")
            return

    # Max iterations reached
    print_warning(
        "\n⚠ Maximum iterations reached. The assistant may not have gathered enough information."
    )


async def run_chat_search(
    project_root: Path,
    query: str,
    limit: int = 5,
    model: str | None = None,
    provider: str | None = None,
    timeout: float = 30.0,
    json_output: bool = False,
    files: str | None = None,
    think: bool = False,
) -> None:
    """Run LLM-powered chat search.

    Implementation Flow:
    1. Initialize LLM client and validate API key
    2. Generate 2-3 targeted search queries from natural language
    3. Execute each search query against vector database
    4. Have LLM analyze all results and select most relevant ones
    5. Display results with explanations

    Args:
        project_root: Project root directory
        query: Natural language query from user
        limit: Maximum number of results to return
        model: Model to use (optional, defaults based on provider)
        provider: LLM provider ('openai' or 'openrouter', auto-detect if None)
        timeout: API timeout in seconds
        json_output: Whether to output JSON format
        files: Optional glob pattern to filter files (e.g., '*.py', 'src/*.js')
        think: Use advanced "thinking" model for complex queries
    """
    # Check for API keys (environment variable or config file)
    from ...core.config_utils import (
        get_openai_api_key,
        get_openrouter_api_key,
        get_preferred_llm_provider,
    )

    config_dir = project_root / ".mcp-vector-search"
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)

    # Determine which provider to use
    if provider:
        # Explicit provider specified
        if provider == "openai" and not openai_key:
            print_error("OpenAI API key not found.")
            print_info("\n[bold]To use OpenAI:[/bold]")
            print_info(
                "1. Get an API key from [cyan]https://platform.openai.com/api-keys[/cyan]"
            )
            print_info("2. Set environment variable:")
            print_info("   [yellow]export OPENAI_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("Or run: [cyan]mcp-vector-search setup[/cyan]")
            raise typer.Exit(1)
        elif provider == "openrouter" and not openrouter_key:
            print_error("OpenRouter API key not found.")
            print_info("\n[bold]To use OpenRouter:[/bold]")
            print_info("1. Get an API key from [cyan]https://openrouter.ai/keys[/cyan]")
            print_info("2. Set environment variable:")
            print_info("   [yellow]export OPENROUTER_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("Or run: [cyan]mcp-vector-search setup[/cyan]")
            raise typer.Exit(1)
    else:
        # Auto-detect provider
        preferred_provider = get_preferred_llm_provider(config_dir)

        if preferred_provider == "openai" and openai_key:
            provider = "openai"
        elif preferred_provider == "openrouter" and openrouter_key:
            provider = "openrouter"
        elif openai_key:
            provider = "openai"
        elif openrouter_key:
            provider = "openrouter"
        else:
            console.print()  # Blank line for spacing
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
            think=think,
        )
        provider_display = llm_client.provider.capitalize()
        model_info = f"{llm_client.model}"
        if think:
            model_info += " [bold magenta](thinking mode)[/bold magenta]"
        print_success(f"Connected to {provider_display}: {model_info}")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Step 1: Generate search queries from natural language
    console.print(f"\n[cyan]💭 Analyzing query:[/cyan] [white]{query}[/white]")

    try:
        search_queries = await llm_client.generate_search_queries(query, limit=3)

        if not search_queries:
            print_error("Failed to generate search queries from your question.")
            raise typer.Exit(1)

        console.print(
            f"\n[cyan]🔍 Generated {len(search_queries)} search queries:[/cyan]"
        )
        for i, sq in enumerate(search_queries, 1):
            console.print(f"  {i}. [yellow]{sq}[/yellow]")

    except SearchError as e:
        print_error(f"Failed to generate queries: {e}")
        raise typer.Exit(1)

    # Step 2: Execute each search query
    console.print("\n[cyan]🔎 Searching codebase...[/cyan]")

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

    # Execute all searches
    search_results = {}
    total_results = 0

    try:
        async with database:
            for search_query in search_queries:
                results = await search_engine.search(
                    query=search_query,
                    limit=limit * 2,  # Get more results for LLM to analyze
                    similarity_threshold=config.similarity_threshold,
                    include_context=True,
                )

                # Post-filter results by file pattern if specified
                if files and results:
                    filtered_results = []
                    for result in results:
                        # Get relative path from project root
                        try:
                            rel_path = str(result.file_path.relative_to(project_root))
                        except ValueError:
                            # If file is outside project root, use absolute path
                            rel_path = str(result.file_path)

                        # Match against glob pattern (both full path and basename)
                        if fnmatch(rel_path, files) or fnmatch(
                            os.path.basename(rel_path), files
                        ):
                            filtered_results.append(result)
                    results = filtered_results

                search_results[search_query] = results
                total_results += len(results)

                console.print(
                    f"  • [yellow]{search_query}[/yellow]: {len(results)} results"
                )

    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)

    if total_results == 0:
        print_warning("\n⚠️  No results found for any search query.")
        print_info("\n[bold]Suggestions:[/bold]")
        print_info("  • Try rephrasing your question")
        print_info("  • Use more general terms")
        print_info(
            "  • Check if relevant files are indexed with [cyan]mcp-vector-search status[/cyan]"
        )
        raise typer.Exit(0)

    # Step 3: Have LLM analyze and rank results
    console.print(f"\n[cyan]🤖 Analyzing {total_results} results...[/cyan]")

    try:
        ranked_results = await llm_client.analyze_and_rank_results(
            original_query=query,
            search_results=search_results,
            top_n=limit,
        )

        if not ranked_results:
            print_warning("\n⚠️  LLM could not identify relevant results.")
            raise typer.Exit(0)

    except SearchError as e:
        print_error(f"Result analysis failed: {e}")
        # Fallback: show raw search results
        print_warning("\nShowing raw search results instead...")
        await _show_fallback_results(search_results, limit)
        raise typer.Exit(1)

    # Step 4: Display results with explanations
    if json_output:
        await _display_json_results(ranked_results)
    else:
        await _display_rich_results(ranked_results, query)


async def _display_rich_results(
    ranked_results: list[dict],
    original_query: str,
) -> None:
    """Display results in rich formatted output.

    Args:
        ranked_results: List of ranked results with explanations
        original_query: Original user query
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    console.print(
        f"\n[bold cyan]🎯 Top Results for:[/bold cyan] [white]{original_query}[/white]\n"
    )

    for i, item in enumerate(ranked_results, 1):
        result = item["result"]
        relevance = item["relevance"]
        explanation = item["explanation"]
        query = item["query"]

        # Determine relevance emoji and color
        if relevance == "High":
            relevance_emoji = "🟢"
            relevance_color = "green"
        elif relevance == "Medium":
            relevance_emoji = "🟡"
            relevance_color = "yellow"
        else:
            relevance_emoji = "🔴"
            relevance_color = "red"

        # Header with result number and file
        console.print(f"[bold]📍 Result {i} of {len(ranked_results)}[/bold]")
        console.print(
            f"[cyan]📂 {result.file_path.relative_to(result.file_path.parent.parent)}[/cyan]"
        )

        # Relevance and explanation
        console.print(
            f"\n{relevance_emoji} [bold {relevance_color}]Relevance: {relevance}[/bold {relevance_color}]"
        )
        console.print(f"[dim]Search query: {query}[/dim]")
        console.print(f"\n💡 [italic]{explanation}[/italic]\n")

        # Code snippet with syntax highlighting
        file_ext = result.file_path.suffix.lstrip(".")
        code_syntax = Syntax(
            result.content,
            lexer=file_ext or "python",
            theme="monokai",
            line_numbers=True,
            start_line=result.start_line,
        )

        panel = Panel(
            code_syntax,
            title=f"[bold]{result.function_name or result.class_name or 'Code'}[/bold]",
            border_style="cyan",
        )
        console.print(panel)

        # Metadata
        metadata = []
        if result.function_name:
            metadata.append(f"Function: [cyan]{result.function_name}[/cyan]")
        if result.class_name:
            metadata.append(f"Class: [cyan]{result.class_name}[/cyan]")
        metadata.append(f"Lines: [cyan]{result.start_line}-{result.end_line}[/cyan]")
        metadata.append(f"Similarity: [cyan]{result.similarity_score:.3f}[/cyan]")

        console.print("[dim]" + " | ".join(metadata) + "[/dim]")
        console.print()  # Blank line between results

    # Footer with tips
    console.print("[dim]─" * 80 + "[/dim]")
    console.print(
        "\n[dim]💡 Tip: Try different phrasings or add more specific terms for better results[/dim]"
    )


async def _display_json_results(ranked_results: list[dict]) -> None:
    """Display results in JSON format.

    Args:
        ranked_results: List of ranked results with explanations
    """
    from ..output import print_json

    json_data = []
    for item in ranked_results:
        result = item["result"]
        json_data.append(
            {
                "file": str(result.file_path),
                "start_line": result.start_line,
                "end_line": result.end_line,
                "function_name": result.function_name,
                "class_name": result.class_name,
                "content": result.content,
                "similarity_score": result.similarity_score,
                "relevance": item["relevance"],
                "explanation": item["explanation"],
                "search_query": item["query"],
            }
        )

    print_json(json_data, title="Chat Search Results")


async def _show_fallback_results(
    search_results: dict[str, list],
    limit: int,
) -> None:
    """Show fallback results when LLM analysis fails.

    Args:
        search_results: Dictionary of search queries to results
        limit: Number of results to show
    """
    from ..output import print_search_results

    # Flatten and deduplicate results
    all_results = []
    seen_files = set()

    for results in search_results.values():
        for result in results:
            file_key = (result.file_path, result.start_line)
            if file_key not in seen_files:
                all_results.append(result)
                seen_files.add(file_key)

    # Sort by similarity score
    all_results.sort(key=lambda r: r.similarity_score, reverse=True)

    # Show top N
    print_search_results(
        results=all_results[:limit],
        query="Combined search results",
        show_content=True,
    )


if __name__ == "__main__":
    chat_app()
