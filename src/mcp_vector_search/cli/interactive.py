"""Interactive search features for MCP Vector Search."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.models import SearchResult
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine
from .output import print_error, print_info, print_search_results, print_warning

console = Console()


class InteractiveSearchSession:
    """Interactive search session with filtering and refinement."""

    def __init__(self, project_root: Path):
        """Initialize interactive search session."""
        self.project_root = project_root
        self.project_manager = ProjectManager(project_root)
        self.search_engine: SemanticSearchEngine | None = None
        self.database: ChromaVectorDatabase | None = None
        self.last_results: list[SearchResult] = []
        self.search_history: list[str] = []

    async def start(self) -> None:
        """Start interactive search session."""
        if not self.project_manager.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first."
            )

        config = self.project_manager.load_config()

        # Setup database and search engine
        embedding_function, _ = create_embedding_function(config.embedding_model)
        self.database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        self.search_engine = SemanticSearchEngine(
            database=self.database,
            project_root=self.project_root,
            similarity_threshold=config.similarity_threshold,
        )

        await self.database.initialize()

        # Show welcome message
        self._show_welcome()

        # Main interactive loop
        try:
            await self._interactive_loop()
        finally:
            await self.database.close()

    def _show_welcome(self) -> None:
        """Show welcome message and help."""
        welcome_text = """
[bold blue]Interactive Search Session[/bold blue]

Available commands:
• [cyan]search <query>[/cyan] - Perform semantic search
• [cyan]filter[/cyan] - Filter current results
• [cyan]refine[/cyan] - Refine last search
• [cyan]history[/cyan] - Show search history
• [cyan]stats[/cyan] - Show result statistics
• [cyan]help[/cyan] - Show this help
• [cyan]quit[/cyan] - Exit session

Type your search query or command:
        """
        console.print(Panel(welcome_text.strip(), border_style="blue"))

    async def _interactive_loop(self) -> None:
        """Main interactive loop."""
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]Search[/bold cyan]").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == "help":
                    self._show_welcome()
                elif user_input.lower() == "history":
                    self._show_history()
                elif user_input.lower() == "stats":
                    self._show_stats()
                elif user_input.lower() == "filter":
                    await self._filter_results()
                elif user_input.lower() == "refine":
                    await self._refine_search()
                elif user_input.startswith("search "):
                    query = user_input[7:].strip()
                    await self._perform_search(query)
                else:
                    # Treat as search query
                    await self._perform_search(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                print_error(f"Error: {e}")

    async def _perform_search(self, query: str) -> None:
        """Perform a search and display results."""
        if not query:
            print_warning("Please provide a search query")
            return

        try:
            console.print(f"\n[dim]Searching for: {query}[/dim]")

            results = await self.search_engine.search(
                query=query,
                limit=20,  # Get more results for filtering
                include_context=True,
            )

            self.last_results = results
            self.search_history.append(query)

            if results:
                print_search_results(results[:10], query, show_content=False)

                if len(results) > 10:
                    console.print(
                        f"\n[dim]Showing top 10 of {len(results)} results. Use 'filter' to refine.[/dim]"
                    )

                # Show quick actions
                self._show_quick_actions()
            else:
                print_warning(f"No results found for: {query}")
                self._suggest_alternatives(query)

        except Exception as e:
            print_error(f"Search failed: {e}")

    def _show_quick_actions(self) -> None:
        """Show quick action options."""
        actions = [
            "[cyan]filter[/cyan] - Filter results",
            "[cyan]refine[/cyan] - Refine search",
            "[cyan]stats[/cyan] - Show statistics",
        ]
        console.print(f"\n[dim]Quick actions: {' | '.join(actions)}[/dim]")

    async def _filter_results(self) -> None:
        """Interactive result filtering."""
        if not self.last_results:
            print_warning("No results to filter. Perform a search first.")
            return

        console.print(f"\n[bold]Filtering {len(self.last_results)} results[/bold]")

        # Show available filter options
        available_languages = {r.language for r in self.last_results}
        available_files = {r.file_path.name for r in self.last_results}
        available_functions = {
            r.function_name for r in self.last_results if r.function_name
        }
        {r.class_name for r in self.last_results if r.class_name}

        # Language filter
        if len(available_languages) > 1:
            lang_choice = Prompt.ask(
                f"Filter by language? ({', '.join(sorted(available_languages))})",
                default="",
                show_default=False,
            )
            if lang_choice and lang_choice in available_languages:
                self.last_results = [
                    r for r in self.last_results if r.language == lang_choice
                ]
                console.print(
                    f"[green]Filtered to {len(self.last_results)} results with language: {lang_choice}[/green]"
                )

        # File filter
        if len(available_files) > 1 and len(self.last_results) > 1:
            file_pattern = Prompt.ask(
                "Filter by file name pattern (partial match)",
                default="",
                show_default=False,
            )
            if file_pattern:
                self.last_results = [
                    r
                    for r in self.last_results
                    if file_pattern.lower() in r.file_path.name.lower()
                ]
                console.print(
                    f"[green]Filtered to {len(self.last_results)} results matching: {file_pattern}[/green]"
                )

        # Function filter
        if available_functions and len(self.last_results) > 1:
            func_pattern = Prompt.ask(
                "Filter by function name pattern (partial match)",
                default="",
                show_default=False,
            )
            if func_pattern:
                self.last_results = [
                    r
                    for r in self.last_results
                    if r.function_name
                    and func_pattern.lower() in r.function_name.lower()
                ]
                console.print(
                    f"[green]Filtered to {len(self.last_results)} results with function matching: {func_pattern}[/green]"
                )

        # Similarity threshold filter
        min_similarity = Prompt.ask(
            "Minimum similarity threshold (0.0-1.0)", default="", show_default=False
        )
        if min_similarity:
            try:
                threshold = float(min_similarity)
                if 0.0 <= threshold <= 1.0:
                    self.last_results = [
                        r for r in self.last_results if r.similarity_score >= threshold
                    ]
                    console.print(
                        f"[green]Filtered to {len(self.last_results)} results with similarity >= {threshold}[/green]"
                    )
            except ValueError:
                print_warning("Invalid similarity threshold")

        # Show filtered results
        if self.last_results:
            print_search_results(
                self.last_results[:10], "Filtered Results", show_content=False
            )
        else:
            print_warning("No results match the filters")

    async def _refine_search(self) -> None:
        """Refine the last search with additional terms."""
        if not self.search_history:
            print_warning("No previous search to refine")
            return

        last_query = self.search_history[-1]
        console.print(f"[dim]Last search: {last_query}[/dim]")

        additional_terms = Prompt.ask("Add terms to refine search", default="")
        if additional_terms:
            refined_query = f"{last_query} {additional_terms}"
            await self._perform_search(refined_query)

    def _show_history(self) -> None:
        """Show search history."""
        if not self.search_history:
            print_info("No search history")
            return

        table = Table(title="Search History", show_header=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Query", style="white")

        for i, query in enumerate(self.search_history[-10:], 1):
            table.add_row(str(i), query)

        console.print(table)

    def _show_stats(self) -> None:
        """Show statistics for current results."""
        if not self.last_results:
            print_info("No results to analyze")
            return

        # Calculate statistics
        languages = {}
        files = {}
        avg_similarity = sum(r.similarity_score for r in self.last_results) / len(
            self.last_results
        )

        for result in self.last_results:
            languages[result.language] = languages.get(result.language, 0) + 1
            files[result.file_path.name] = files.get(result.file_path.name, 0) + 1

        # Create statistics table
        table = Table(title="Result Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Results", str(len(self.last_results)))
        table.add_row("Average Similarity", f"{avg_similarity:.1%}")
        table.add_row(
            "Languages",
            ", ".join(f"{lang}({count})" for lang, count in languages.items()),
        )
        table.add_row("Unique Files", str(len(files)))

        console.print(table)

    def _suggest_alternatives(self, query: str) -> None:
        """Suggest alternative search terms."""
        suggestions = []

        # Simple suggestions based on common patterns
        words = query.lower().split()
        for word in words:
            if word in ["auth", "authentication"]:
                suggestions.extend(["login", "user", "session", "token"])
            elif word in ["db", "database"]:
                suggestions.extend(["query", "model", "connection", "storage"])
            elif word in ["api"]:
                suggestions.extend(["endpoint", "request", "response", "handler"])
            elif word in ["test", "testing"]:
                suggestions.extend(["mock", "assert", "spec", "unit"])

        if suggestions:
            unique_suggestions = list(set(suggestions))[:5]
            console.print(
                f"[dim]Try these terms: {', '.join(unique_suggestions)}[/dim]"
            )


async def start_interactive_search(project_root: Path) -> None:
    """Start an interactive search session."""
    session = InteractiveSearchSession(project_root)
    await session.start()
