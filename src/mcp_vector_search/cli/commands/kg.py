"""Knowledge graph CLI commands."""

import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from ...core.chunks_backend import ChunksBackend
from ...core.factory import ComponentFactory
from ...core.kg_builder import KGBuilder
from ...core.knowledge_graph import KnowledgeGraph
from ...core.project import ProjectManager
from .visualize.server import find_free_port, start_visualization_server

console = Console()
kg_app = typer.Typer(name="kg", help="📊 Knowledge graph operations")


def _build_kg_impl(
    project_root: Path,
    force: bool,
    limit: int | None,
    skip_documents: bool,
    incremental: bool,
    verbose: bool,
) -> None:
    """Shared implementation for build/index commands.

    Args:
        project_root: Project root directory
        force: Force rebuild even if graph exists
        limit: Limit number of chunks to process (for testing)
        skip_documents: Skip expensive DOCUMENTS relationship extraction
        incremental: Only process new chunks not in previous build
        verbose: Enable verbose debug output
    """
    project_root = project_root.resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Building Knowledge Graph[/bold cyan]\nProject: {project_root}",
            border_style="cyan",
        )
    )

    async def _load_and_prepare_chunks():
        """Load chunks from database and save to temp file for subprocess.

        Returns:
            Tuple of (chunks_temp_path, files_to_delete_temp_path, current_hashes).
            - chunks_temp_path: path to JSON file with chunks to build into KG
            - files_to_delete_temp_path: path to JSON file with file paths whose KG
              entities should be deleted before building (None if not incremental or
              no deletions needed)
            - current_hashes: dict[file_path, sha256] for all currently indexed files
              (empty dict when incremental=False, used to update metadata after build)
        """
        import os
        import sys
        import tempfile
        from dataclasses import asdict

        # Initialize database to load chunks
        components = await ComponentFactory.create_standard_components(
            project_root, use_pooling=False
        )
        database = components.database

        temp_path = None
        files_to_delete_path = None
        current_hashes: dict[str, str] = {}

        async with database:
            # Check if index exists
            chunk_count = database.get_chunk_count()
            if chunk_count == 0:
                console.print(
                    "[red]✗[/red] No index found. Run 'mcp-vector-search index' first."
                )
                raise typer.Exit(1)

            if verbose:
                console.print(
                    f"[cyan]Loading {chunk_count} chunks from database...[/cyan]"
                )

            # --- Incremental: hash-based change detection ---
            files_to_process: set[str] | None = None  # None = process all
            files_to_delete: set[str] = set()

            if incremental:
                # Use ChunksBackend to get per-file hashes (file_hash lives there)
                lance_path = project_root / ".mcp-vector-search" / "lance"
                chunks_backend = ChunksBackend(lance_path)
                await chunks_backend.initialize()
                try:
                    current_hashes = await chunks_backend.get_all_indexed_file_hashes()
                finally:
                    # ChunksBackend has no async close; just release reference
                    pass

                # Compare against last KG build's stored hashes
                builder_for_diff = KGBuilder(
                    None,  # type: ignore[arg-type]  # only metadata path needed
                    project_root,
                )
                changed_files, new_files, deleted_files = (
                    builder_for_diff._get_changed_files(current_hashes)
                )

                files_to_process = changed_files | new_files
                files_to_delete = changed_files | deleted_files

                if verbose:
                    console.print(
                        f"[yellow]Incremental mode: {len(new_files)} new, "
                        f"{len(changed_files)} changed, "
                        f"{len(deleted_files)} deleted file(s)[/yellow]"
                    )

                if not files_to_process and not files_to_delete:
                    console.print(
                        "[green]✓ KG already up to date — no file changes detected[/green]"
                    )
                    raise typer.Exit(0)

                if verbose:
                    console.print(
                        f"[cyan]Processing {len(files_to_process)} file(s), "
                        f"deleting entities from {len(files_to_delete)} file(s)[/cyan]"
                    )

            # --- Load all chunks (filtered if incremental) ---
            chunks = []
            for batch in database.iter_chunks_batched(batch_size=5000):
                if files_to_process is not None:
                    # Only keep chunks belonging to files that changed or are new
                    batch = [c for c in batch if str(c.file_path) in files_to_process]
                chunks.extend(batch)

                # Apply limit if specified
                if limit and len(chunks) >= limit:
                    chunks = chunks[:limit]
                    break

            if incremental and len(chunks) == 0 and not files_to_delete:
                console.print(
                    "[green]✓ No chunks to process in incremental mode[/green]"
                )
                raise typer.Exit(0)

            if verbose:
                console.print(f"[green]✓ Loaded {len(chunks)} chunks[/green]")

            # --- Serialize chunks to temp JSON file ---
            if verbose:
                console.print("[dim]Serializing chunks to temp file...[/dim]")
                sys.stdout.flush()
            temp_fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="kg_chunks_")
            try:
                with open(temp_path, "w") as f:
                    chunks_data = [asdict(chunk) for chunk in chunks]
                    for chunk_dict in chunks_data:
                        chunk_dict["file_path"] = str(chunk_dict["file_path"])
                    json.dump(chunks_data, f)
                if verbose:
                    console.print(f"[dim]Saved chunks to {temp_path}[/dim]")
                    sys.stdout.flush()
            finally:
                os.close(temp_fd)

            # --- Write files-to-delete list (incremental only) ---
            if incremental and files_to_delete:
                del_fd, files_to_delete_path = tempfile.mkstemp(
                    suffix=".json", prefix="kg_delete_"
                )
                try:
                    with open(files_to_delete_path, "w") as f:
                        json.dump(list(files_to_delete), f)
                    if verbose:
                        console.print(
                            f"[dim]Saved {len(files_to_delete)} file(s) to delete at "
                            f"{files_to_delete_path}[/dim]"
                        )
                finally:
                    os.close(del_fd)

        # Return paths AFTER database is properly closed
        if verbose:
            console.print("[dim]Closing database...[/dim]")
            sys.stdout.flush()

        # CRITICAL: Explicitly close database and await cleanup
        await database.close()
        if verbose:
            console.print("[dim]Database closed explicitly[/dim]")
            sys.stdout.flush()

        return temp_path, files_to_delete_path, current_hashes

    # Load chunks in parent process (before spawning subprocess)
    if verbose:
        console.print("[cyan]Loading chunks in parent process...[/cyan]")
    chunks_file, files_to_delete_path, current_hashes = asyncio.run(
        _load_and_prepare_chunks()
    )
    if verbose:
        console.print(f"[green]✓ Chunks saved to {chunks_file}[/green]")
        if files_to_delete_path:
            console.print(
                f"[green]✓ Files-to-delete saved to {files_to_delete_path}[/green]"
            )

    # CRITICAL: Force cleanup of asyncio resources and background threads
    # LanceDB creates a persistent "LanceDBBackgroundEventLoop" daemon thread
    # that MUST be terminated before spawning subprocess (causes Kuzu segfaults)
    import gc
    import threading
    import time

    if verbose:
        console.print("[dim]Forcing cleanup of async resources...[/dim]")

    # Force garbage collection to cleanup lingering asyncio resources
    gc.collect()

    # Close all asyncio event loops
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            loop.close()
            if verbose:
                console.print("[dim]Closed asyncio event loop[/dim]")
    except RuntimeError:
        pass  # No event loop in current thread

    # Set new event loop policy to ensure clean state
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    # Give background threads time to terminate
    max_wait = 3.0  # Wait up to 3 seconds
    start_time = time.time()
    threads = threading.enumerate()
    if verbose:
        console.print(f"[dim]Initial threads: {[t.name for t in threads]}[/dim]")

    while len(threads) > 1 and (time.time() - start_time) < max_wait:
        time.sleep(0.2)
        gc.collect()  # Keep collecting
        threads = threading.enumerate()

    # Final check - only warn in verbose mode or if there are actual problematic threads
    threads = threading.enumerate()
    if len(threads) > 1:
        background = [t for t in threads if t != threading.main_thread()]
        if verbose:
            console.print(
                f"[yellow]⚠ Warning: {len(background)} background thread(s) still active:[/yellow]"
            )
            for t in background:
                console.print(f"  - {t.name} (daemon={t.daemon})")
            console.print()
            console.print(
                "[yellow]These daemon threads will be inherited by subprocess.[/yellow]"
            )
            console.print(
                "[yellow]Subprocess will fail with segfault if threads interfere with Kuzu.[/yellow]"
            )
            console.print()
    else:
        if verbose:
            console.print(
                "[green]✓ No background threads, safe to spawn subprocess[/green]"
            )

    # Run in completely isolated subprocess using subprocess.run()
    # This is necessary because Kuzu (Rust-based) segfaults with background threads
    # CRITICAL: We use subprocess.run() instead of multiprocessing.Process()
    # because multiprocessing can inherit module state including LanceDB threads
    import subprocess

    if verbose:
        console.print("[dim]Starting completely isolated subprocess...[/dim]")

    # Build command to execute the isolated subprocess script
    # CRITICAL: Use the SAME Python interpreter that's running mcp-vector-search
    # Not sys.executable, which might be a different tool's Python (like claude-mpm)
    import shutil

    # Find mcp-vector-search command and extract its Python interpreter
    mcp_cmd = shutil.which("mcp-vector-search")
    if mcp_cmd:
        # Read shebang from mcp-vector-search to get correct Python
        with open(mcp_cmd) as f:
            shebang = f.readline().strip()
            if shebang.startswith("#!"):
                python_executable = shebang[2:].strip()
            else:
                python_executable = sys.executable
    else:
        python_executable = sys.executable

    if verbose:
        console.print(f"[dim]Using Python: {python_executable}[/dim]")

    subprocess_script = Path(__file__).parent / "_kg_subprocess.py"
    cmd = [
        python_executable,
        str(subprocess_script),
        str(project_root.absolute()),
        chunks_file,
    ]
    if force:
        cmd.append("--force")
    if skip_documents:
        cmd.append("--skip-documents")
    if verbose:
        cmd.append("--verbose")
    if files_to_delete_path:
        cmd.extend(["--files-to-delete", files_to_delete_path])

    if verbose:
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    # Run subprocess with inherited stdout/stderr for live output
    result = subprocess.run(
        cmd,
        check=False,
        stdout=None,  # Inherit stdout (shows in console)
        stderr=None,  # Inherit stderr (shows in console)
    )

    # Clean up files-to-delete temp file (chunks file is cleaned by subprocess)
    if files_to_delete_path:
        try:
            Path(files_to_delete_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug(
                "Failed to clean up temp delete file %s: %s", files_to_delete_path, e
            )

    if verbose:
        console.print(
            f"[dim]Subprocess finished with exitcode: {result.returncode}[/dim]"
        )

    if result.returncode != 0:
        console.print(f"[red]✗ Build failed with exit code {result.returncode}[/red]")
        if result.returncode not in (0, 1):
            # Unexpected exit code — likely a native LanceDB/Kuzu crash
            # (e.g. SIGSEGV=11, OS-defined=120) rather than an application error.
            logger.error(
                "KG subprocess exited with unexpected code %d. "
                "This is typically a native LanceDB or Kuzu crash during initialization. "
                "To recover: rm -rf .mcp-vector-search/lance/*/_transactions/ && mvs index --force",
                result.returncode,
            )
        # Clean up temp file
        try:
            Path(chunks_file).unlink()
        except Exception as e:
            logger.debug("Failed to clean up temp chunks file %s: %s", chunks_file, e)
        raise typer.Exit(
            1
        )  # Always normalize to 1, never propagate raw signal/OS codes

    # --- Post-build: persist file hashes so next incremental run can diff ---
    if incremental and current_hashes:
        try:
            builder_for_meta = KGBuilder(
                None,  # type: ignore[arg-type]  # only metadata path needed
                project_root,
            )
            builder_for_meta.update_metadata_file_hashes(current_hashes)
            if verbose:
                console.print(
                    f"[dim]Updated kg_metadata.json with {len(current_hashes)} file hashes[/dim]"
                )
        except Exception as e:
            logger.warning("Failed to update KG metadata file hashes: %s", e)

    if verbose:
        console.print("[green]✓ Build completed successfully in subprocess[/green]")


@kg_app.command("build")
def build_kg(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Force rebuild even if graph exists"
    ),
    limit: int | None = typer.Option(
        None, help="Limit number of chunks to process (for testing)"
    ),
    skip_documents: bool = typer.Option(
        False,
        "--skip-documents",
        help="Skip expensive DOCUMENTS relationship extraction (faster build)",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Only process new chunks not in previous build (incremental build)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug output (shows detailed progress)",
    ),
):
    """Build knowledge graph from indexed chunks.

    [yellow]⚠ DEPRECATED:[/yellow] Use [cyan]'mvs index kg'[/cyan] instead.

    This command extracts entities and relationships from your indexed
    codebase and builds a queryable knowledge graph.

    Example:
        mcp-vector-search kg build
        mcp-vector-search kg build --force
        mcp-vector-search kg build --limit 100  # Test with 100 chunks
        mcp-vector-search kg build --skip-documents  # Faster build for large repos
        mcp-vector-search kg build --incremental  # Only process new chunks
    """
    console.print(
        "[yellow]⚠ 'mvs kg build' is deprecated. Use 'mvs index kg' instead.[/yellow]"
    )
    _build_kg_impl(project_root, force, limit, skip_documents, incremental, verbose)


@kg_app.command("query")
def query_kg(
    entity: str = typer.Argument(..., help="Entity name or ID to query"),
    hops: int = typer.Option(2, help="Maximum number of relationship hops"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Query relationships for an entity.

    Find entities related to the specified entity within N hops.

    Examples:
        mcp-vector-search kg query "SemanticSearchEngine"
        mcp-vector-search kg query "search" --hops 1
    """
    project_root = project_root.resolve()

    async def _query():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Check if graph exists
        stats = await kg.get_stats()
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            raise typer.Exit(1)

        # Try to find entity by name or ID
        # For simplicity, we'll search for entities matching the name
        related = await kg.find_related(entity, max_hops=hops)

        if not related:
            console.print(f"[yellow]No related entities found for '{entity}'[/yellow]")
            raise typer.Exit(0)

        # Display results
        table = Table(title=f"Entities related to '{entity}' (within {hops} hops)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("File", style="dim")

        for rel in related:
            table.add_row(rel["name"], rel["type"], rel["file_path"])

        console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_query())


@kg_app.command("stats")
def kg_stats(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show knowledge graph statistics.

    Display entity and relationship counts from the knowledge graph.

    Example:
        mcp-vector-search kg stats
    """
    project_root = project_root.resolve()

    async def _stats():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get statistics
        stats = await kg.get_stats()

        # Load metadata for gap detection
        metadata_path = (
            project_root / ".mcp-vector-search" / "knowledge_graph" / "kg_metadata.json"
        )
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.debug("Failed to load KG metadata from %s: %s", metadata_path, e)

        # Get current chunk count from database
        current_chunks = 0
        gap = 0
        try:
            components = await ComponentFactory.create_standard_components(
                project_root, use_pooling=False
            )
            database = components.database
            async with database:
                current_chunks = database.get_chunk_count()
                if metadata:
                    source_chunks = metadata.get("source_chunk_count", 0)
                    gap = current_chunks - source_chunks
            await database.close()
        except Exception as e:
            logger.debug("Failed to get chunk count from database: %s", e)

        # Display results
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Entities", f"[green]{stats['total_entities']:,}[/green]")
        if "code_entities" in stats:
            table.add_row(
                "  Code Entities", f"[green]{stats['code_entities']:,}[/green]"
            )
        if "doc_sections" in stats:
            table.add_row("  Doc Sections", f"[green]{stats['doc_sections']:,}[/green]")
        if "tags" in stats:
            table.add_row("  Tags", f"[green]{stats['tags']:,}[/green]")
        if "persons" in stats:
            table.add_row("  Persons", f"[green]{stats['persons']:,}[/green]")
        if "projects" in stats:
            table.add_row("  Projects", f"[green]{stats['projects']:,}[/green]")

        # Show chunk tracking info
        if metadata:
            table.add_row("", "")  # Separator
            table.add_row(
                "Source Chunks", f"[green]{metadata['source_chunk_count']:,}[/green]"
            )
            table.add_row("Current Chunks", f"[cyan]{current_chunks:,}[/cyan]")
            if gap > 0:
                table.add_row("Gap", f"[yellow]{gap:,} new[/yellow]")
            elif gap == 0:
                table.add_row("Gap", "[green]0 (up to date)[/green]")
            else:
                table.add_row("Gap", f"[red]{abs(gap):,} removed[/red]")

            # Show last build time
            if "last_build" in metadata:
                try:
                    last_build = datetime.fromisoformat(metadata["last_build"])
                    now = datetime.now(UTC)
                    delta = now - last_build
                    if delta.days > 0:
                        time_ago = f"{delta.days}d ago"
                    elif delta.seconds > 3600:
                        time_ago = f"{delta.seconds // 3600}h ago"
                    elif delta.seconds > 60:
                        time_ago = f"{delta.seconds // 60}m ago"
                    else:
                        time_ago = "just now"
                    table.add_row("Last Build", f"[dim]{time_ago}[/dim]")
                except Exception as e:
                    logger.debug("Failed to format build time: %s", e)

        table.add_row("", "")  # Separator
        table.add_row("Database Path", f"[dim]{stats['database_path']}[/dim]")

        # Add relationship counts
        if "relationships" in stats:
            table.add_row("", "")  # Separator
            for rel_type, count in stats["relationships"].items():
                table.add_row(f"  {rel_type.title()}", f"[green]{count:,}[/green]")

        console.print(table)

        # Check if KG is incomplete (entities exist but no relationships)
        has_relationships = await kg.has_relationships()
        total_entities = stats.get("total_entities", 0)

        if total_entities > 0 and not has_relationships:
            console.print()
            console.print(
                "[yellow]⚠️  Knowledge Graph is incomplete[/yellow] "
                "[dim](entities exist but no relationships)[/dim]"
            )
            console.print(
                "   [yellow]Run 'mcp-vector-search kg build --force' to rebuild[/yellow]"
            )

        # Show helpful message if gap exists
        if gap > 0:
            console.print(
                f"\n💡 [yellow]Run 'kg build --incremental' to add {gap:,} new chunks[/yellow]"
            )

        # Close connection
        await kg.close()

    asyncio.run(_stats())


@kg_app.command("ontology")
def kg_ontology(
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Filter by document category"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show section details"),
    ] = False,
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Browse the document ontology tree.

    Shows documents grouped by category (readme, guide, api_doc, etc.)
    with section counts, tags, and cross-references.

    Examples:
        mcp-vector-search kg ontology
        mcp-vector-search kg ontology --category guide
        mcp-vector-search kg ontology --verbose
    """
    project_root = project_root.resolve()

    # Category emoji map
    category_icons: dict[str, str] = {
        # Core documentation
        "readme": "📋",
        "guide": "📖",
        "tutorial": "🎓",
        "api_doc": "🔌",
        "design": "🎨",
        "spec": "📐",
        "research": "🔬",
        "changelog": "📝",
        # Development lifecycle
        "bugfix": "🐛",
        "performance": "⚡",
        "setup": "⚙️",
        "configuration": "🔧",
        "migration": "🔄",
        "upgrade_guide": "⬆️",
        "release_notes": "🚀",
        "deployment": "🚢",
        # Support and reference
        "troubleshooting": "🔍",
        "faq": "❓",
        "security": "🔐",
        "example": "💡",
        # Project management
        "contributing": "🤝",
        "license": "⚖️",
        "roadmap": "🗺️",
        "internal": "🏠",
        "report": "📊",
        "feature": "✨",
        "project": "📁",
        "test_doc": "🧪",
        # Fallback
        "other": "📄",
    }

    async def _ontology():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        try:
            # Get ontology data
            data = await kg.get_document_ontology(category=category)
        finally:
            await kg.close()

        total_docs = data["total_documents"]
        total_sections = data["total_sections"]
        total_xrefs = data["total_cross_references"]
        categories = data["categories"]

        if total_docs == 0:
            console.print()
            if category:
                console.print(
                    f"[yellow]No documents found in category '{category}'.[/yellow]"
                )
            else:
                console.print(
                    "[yellow]No documents found. Build the knowledge graph with document support first.[/yellow]"
                )
                console.print(
                    "  Run [cyan]'mvs kg build'[/cyan] or [cyan]'mvs index kg'[/cyan]"
                )
            return

        # Header panel
        header_parts = ["[bold cyan]Document Ontology[/bold cyan]"]
        header_parts.append(
            f"[dim]{total_docs} document{'s' if total_docs != 1 else ''}, "
            f"{total_sections} section{'s' if total_sections != 1 else ''}, "
            f"{total_xrefs} cross-reference{'s' if total_xrefs != 1 else ''}[/dim]"
        )
        console.print()
        console.print(
            Panel.fit(
                "\n".join(header_parts),
                border_style="cyan",
            )
        )
        console.print()

        # Build Rich tree per category
        for cat_name in sorted(categories.keys()):
            docs = categories[cat_name]
            if not docs:
                continue

            icon = category_icons.get(cat_name, "📄")
            cat_tree = Tree(
                f"{icon} [bold green]{cat_name}[/bold green] "
                f"[dim]({len(docs)} document{'s' if len(docs) != 1 else ''})[/dim]"
            )

            for doc in docs:
                file_path = doc["file_path"]
                title = doc.get("title") or ""
                word_count = doc.get("word_count") or 0
                section_count = doc.get("section_count") or 0
                tags = doc.get("tags") or []
                cross_refs = doc.get("cross_references") or []
                sections = doc.get("sections") or []

                # Truncate long paths for display
                display_path = file_path
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]

                # Build doc label
                doc_label_parts = [f"[cyan]{display_path}[/cyan]"]
                if title and title != display_path:
                    doc_label_parts.append(f'[dim]— "{title}"[/dim]')
                doc_label_parts.append(
                    f"[dim]({word_count:,} words, {section_count} sections)[/dim]"
                )
                doc_node = cat_tree.add(" ".join(doc_label_parts))

                # Tags line
                if tags:
                    tag_str = ", ".join(sorted(tags[:8]))
                    if len(tags) > 8:
                        tag_str += f" +{len(tags) - 8} more"
                    doc_node.add(f"[dim]Tags:[/dim] [magenta][{tag_str}][/magenta]")

                # Cross-references line
                if cross_refs:
                    ref_paths = [
                        r.get("title") or r.get("file_path", "") for r in cross_refs[:5]
                    ]
                    ref_str = ", ".join(ref_paths)
                    if len(cross_refs) > 5:
                        ref_str += f" +{len(cross_refs) - 5} more"
                    doc_node.add(f"[dim]→[/dim] [yellow]{ref_str}[/yellow]")

                # Sections (verbose mode only)
                if verbose and sections:
                    for sec in sections:
                        indent = "  " * max(0, (sec.get("level", 1) - 1))
                        sec_name = sec.get("name", "")
                        sec_line = sec.get("line", "")
                        level_marker = "#" * sec.get("level", 1)
                        line_info = f"[dim]:L{sec_line}[/dim]" if sec_line else ""
                        doc_node.add(
                            f"[dim]{indent}{level_marker}[/dim] "
                            f"[white]{sec_name}[/white]{line_info}"
                        )

            console.print(cat_tree)
            console.print()

    asyncio.run(_ontology())


@kg_app.command("ia")
def ia_tree(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show doc_category per document"
    ),
) -> None:
    """Show the Information Architecture tree of the knowledge graph.

    Documents are grouped into thematic IA categories:
    Orientation, Guides & Tutorials, Architecture & Design,
    API Reference, Operations, Lifecycle, Testing.

    Examples:
        mcp-vector-search kg ia
        mcp-vector-search kg ia --verbose
    """
    project_root = project_root.resolve()

    # IA group emoji map
    group_icons: dict[str, str] = {
        "Orientation": "🧭",
        "Guides & Tutorials": "📖",
        "Architecture & Design": "🏗️",
        "API Reference": "🔌",
        "Operations": "⚙️",
        "Lifecycle": "🔄",
        "Testing": "🧪",
        "Uncategorized": "📄",
    }

    async def _ia():
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        try:
            data = await kg.get_ia_tree()
        finally:
            await kg.close()

        ia_tree_data = data["ia_tree"]
        total_topics = data["total_topics"]
        total_docs = data["total_documents"]

        if total_docs == 0:
            console.print()
            console.print(
                "[yellow]No IA topics found. Build the knowledge graph first.[/yellow]"
            )
            console.print(
                "  Run [cyan]'mvs kg build'[/cyan] or [cyan]'mvs index kg'[/cyan]"
            )
            return

        console.print()
        console.print(
            Panel.fit(
                f"[bold cyan]Information Architecture[/bold cyan]  "
                f"[dim]{total_topics} group{'s' if total_topics != 1 else ''}, "
                f"{total_docs} document{'s' if total_docs != 1 else ''}[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        root = Tree("[bold white]Information Architecture[/bold white]")

        for group_name in sorted(ia_tree_data.keys()):
            group_data = ia_tree_data[group_name]
            docs = group_data["documents"]
            icon = group_icons.get(group_name, "📄")

            group_node = root.add(
                f"{icon} [bold green]{group_name}[/bold green] "
                f"[dim]({len(docs)} document{'s' if len(docs) != 1 else ''})[/dim]"
            )

            for doc in docs:
                file_path = doc["file_path"] or ""
                title = doc.get("title") or ""
                word_count = doc.get("word_count") or 0
                doc_category = doc.get("doc_category") or ""

                display_path = file_path
                if len(display_path) > 55:
                    display_path = "..." + display_path[-52:]

                label_parts = [f"[cyan]{display_path}[/cyan]"]
                if title and title != file_path:
                    label_parts.append(f'[dim]— "{title}"[/dim]')
                if word_count:
                    label_parts.append(f"[dim]({word_count:,} words)[/dim]")
                if verbose and doc_category:
                    label_parts.append(f"[dim][{doc_category}][/dim]")

                group_node.add(" ".join(label_parts))

        console.print(root)
        console.print()

    asyncio.run(_ia())


@kg_app.command("status")
def kg_status(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show detailed knowledge graph status with hierarchical breakdown.

    Display comprehensive status including:
    - Node counts by type (classes, functions, modules, files)
    - Relationship counts by category (code structure, documentation, metadata)
    - Warnings for incomplete builds

    Example:
        mcp-vector-search kg status
    """
    project_root = project_root.resolve()

    async def _status():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get detailed statistics
        stats = await kg.get_detailed_stats()

        # Check if graph exists
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            await kg.close()
            raise typer.Exit(1)

        # Display header
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]Knowledge Graph Status[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()

        # Build Nodes tree
        nodes_tree = Tree(
            f"[bold cyan]Nodes[/bold cyan] [dim]({stats['total_entities']:,} total)[/dim]"
        )

        # Code Entities subtree
        entity_types = stats.get("entity_types", {})
        total_code = stats.get("code_entities", 0)
        code_branch = nodes_tree.add(
            f"[green]Code Entities[/green] [dim]{total_code:,}[/dim]"
        )

        # Display entity types dynamically based on what exists in the database
        for entity_type, count in entity_types.items():
            # Pluralize entity type for display
            if entity_type.endswith("s"):
                display_name = entity_type.capitalize() + "es"
            else:
                display_name = entity_type.capitalize() + "s"
            code_branch.add(f"[dim]{display_name:<13}[/dim] {count:,}")

        # Doc Sections
        nodes_tree.add(
            f"[green]Doc Sections[/green] [dim]{stats.get('doc_sections', 0):,}[/dim]"
        )

        # Tags
        nodes_tree.add(
            f"[green]Tags[/green]         [dim]{stats.get('tags', 0):,}[/dim]"
        )

        # Persons
        nodes_tree.add(
            f"[green]Persons[/green]      [dim]{stats.get('persons', 0):,}[/dim]"
        )

        # Projects
        nodes_tree.add(
            f"[green]Projects[/green]     [dim]{stats.get('projects', 0):,}[/dim]"
        )

        # Repositories
        if stats.get("repositories", 0) > 0:
            nodes_tree.add(
                f"[green]Repositories[/green] [dim]{stats.get('repositories', 0):,}[/dim]"
            )

        # Branches
        if stats.get("branches", 0) > 0:
            nodes_tree.add(
                f"[green]Branches[/green]     [dim]{stats.get('branches', 0):,}[/dim]"
            )

        # Commits
        if stats.get("commits", 0) > 0:
            nodes_tree.add(
                f"[green]Commits[/green]      [dim]{stats.get('commits', 0):,}[/dim]"
            )

        # Languages
        if stats.get("languages", 0) > 0:
            nodes_tree.add(
                f"[green]Languages[/green]    [dim]{stats.get('languages', 0):,}[/dim]"
            )

        # Frameworks
        if stats.get("frameworks", 0) > 0:
            nodes_tree.add(
                f"[green]Frameworks[/green]   [dim]{stats.get('frameworks', 0):,}[/dim]"
            )

        console.print(nodes_tree)
        console.print()

        # Build Relationships tree
        relationships = stats.get("relationships", {})
        total_rels = sum(relationships.values())
        rels_tree = Tree(
            f"[bold cyan]Relationships[/bold cyan] [dim]({total_rels:,} total)[/dim]"
        )

        # Code Structure subtree
        code_rels = {
            "calls": relationships.get("calls", 0),
            "imports": relationships.get("imports", 0),
            "inherits": relationships.get("inherits", 0),
            "contains": relationships.get("contains", 0),
        }
        code_rels_total = sum(code_rels.values())
        code_structure_label = (
            f"[green]Code Structure[/green]       [dim]{code_rels_total:,}[/dim]"
            if code_rels_total > 0
            else f"[yellow]Code Structure[/yellow]       [dim]{code_rels_total:,}[/dim]  [yellow]⚠️[/yellow]"
        )
        code_rel_branch = rels_tree.add(code_structure_label)
        code_rel_branch.add(f"[dim]Calls[/dim]            {code_rels['calls']:,}")
        code_rel_branch.add(f"[dim]Imports[/dim]          {code_rels['imports']:,}")
        code_rel_branch.add(f"[dim]Inherits[/dim]         {code_rels['inherits']:,}")
        code_rel_branch.add(f"[dim]Contains[/dim]         {code_rels['contains']:,}")

        # Documentation subtree (DOCUMENTS relation removed - not implemented in subprocess mode)
        doc_rels = {
            "follows": relationships.get("follows", 0),
            "demonstrates": relationships.get("demonstrates", 0),
            "references": relationships.get("references", 0),
            "links_to": relationships.get("links_to", 0),
        }
        doc_rels_total = sum(doc_rels.values())
        doc_rel_branch = rels_tree.add(
            f"[green]Documentation[/green]    [dim]{doc_rels_total:,}[/dim]"
        )
        doc_rel_branch.add(f"[dim]Follows[/dim]        {doc_rels['follows']:,}")
        doc_rel_branch.add(f"[dim]Demonstrates[/dim]   {doc_rels['demonstrates']:,}")
        doc_rel_branch.add(f"[dim]References[/dim]     {doc_rels['references']:,}")
        doc_rel_branch.add(f"[dim]Links To[/dim]       {doc_rels['links_to']:,}")

        # Metadata subtree
        meta_rels = {
            "has_tag": relationships.get("has_tag", 0),
            "authored": relationships.get("authored", 0),
            "modified": relationships.get("modified", 0),
            "part_of": relationships.get("part_of", 0),
        }
        meta_rels_total = sum(meta_rels.values())
        meta_rel_branch = rels_tree.add(
            f"[green]Metadata[/green]             [dim]{meta_rels_total:,}[/dim]"
        )
        meta_rel_branch.add(f"[dim]Has_Tag[/dim]          {meta_rels['has_tag']:,}")
        meta_rel_branch.add(f"[dim]Authored[/dim]         {meta_rels['authored']:,}")
        meta_rel_branch.add(f"[dim]Modified[/dim]         {meta_rels['modified']:,}")
        meta_rel_branch.add(f"[dim]Part_Of[/dim]          {meta_rels['part_of']:,}")

        # Version Control subtree
        vc_rels = {
            "modifies": relationships.get("modifies", 0),
            "branched_from": relationships.get("branched_from", 0),
            "committed_to": relationships.get("committed_to", 0),
            "belongs_to": relationships.get("belongs_to", 0),
        }
        vc_rels_total = sum(vc_rels.values())
        if vc_rels_total > 0:
            vc_rel_branch = rels_tree.add(
                f"[green]Version Control[/green]      [dim]{vc_rels_total:,}[/dim]"
            )
            vc_rel_branch.add(f"[dim]Modifies[/dim]         {vc_rels['modifies']:,}")
            vc_rel_branch.add(
                f"[dim]Branched_From[/dim]    {vc_rels['branched_from']:,}"
            )
            vc_rel_branch.add(
                f"[dim]Committed_To[/dim]     {vc_rels['committed_to']:,}"
            )
            vc_rel_branch.add(f"[dim]Belongs_To[/dim]       {vc_rels['belongs_to']:,}")

        # Language/Framework subtree
        lang_rels = {
            "written_in": relationships.get("written_in", 0),
            "uses_framework": relationships.get("uses_framework", 0),
            "framework_for": relationships.get("framework_for", 0),
        }
        lang_rels_total = sum(lang_rels.values())
        if lang_rels_total > 0:
            lang_rel_branch = rels_tree.add(
                f"[green]Language/Framework[/green]   [dim]{lang_rels_total:,}[/dim]"
            )
            lang_rel_branch.add(
                f"[dim]Written_In[/dim]        {lang_rels['written_in']:,}"
            )
            lang_rel_branch.add(
                f"[dim]Uses_Framework[/dim]    {lang_rels['uses_framework']:,}"
            )
            lang_rel_branch.add(
                f"[dim]Framework_For[/dim]     {lang_rels['framework_for']:,}"
            )

        console.print(rels_tree)

        # Show warning if code relationships are missing
        if code_rels_total == 0:
            console.print()
            console.print("[yellow]⚠️  Code relationships not built[/yellow]")
            console.print(
                "   Run [cyan]'mcp-vector-search kg build --force'[/cyan] to rebuild"
            )

        console.print()

        # Build Cross-Entity Relationships tree
        cross_samples = await kg.get_cross_entity_samples(limit_per_type=2)

        if cross_samples:
            cross_tree = Tree("[bold cyan]Cross-Entity Relationships[/bold cyan]")

            # Map pattern names to relationship types for count lookup
            pattern_to_rel = {
                "DocSection → DOCUMENTS → CodeEntity": "documents",
                "DocSection → DEMONSTRATES → Tag": "demonstrates",
                "DocSection → REFERENCES → CodeEntity": "references",
                "Person → AUTHORED → CodeEntity": "authored",
                "Person → MODIFIED → CodeEntity": "modified",
                "DocSection → HAS_TAG → Tag": "has_tag",
                "CodeEntity → PART_OF → Project": "part_of",
            }

            for pattern_name, samples in cross_samples.items():
                # Get total count from stats
                rel_type = pattern_to_rel.get(pattern_name, "")
                total_count = relationships.get(rel_type, 0)

                # Create branch for this pattern with total count
                if total_count > len(samples):
                    pattern_label = (
                        f"[green]{pattern_name}[/green] "
                        f"[dim](showing {len(samples)} of {total_count:,})[/dim]"
                    )
                else:
                    pattern_label = (
                        f"[green]{pattern_name}[/green] "
                        f"[dim](showing {len(samples)})[/dim]"
                    )

                pattern_branch = cross_tree.add(pattern_label)

                # Add sample relationships
                for sample in samples:
                    pattern_branch.add(
                        f"[cyan]{sample['source']}[/cyan] "
                        f"[dim]→[/dim] "
                        f"[yellow]{sample['rel']}[/yellow] "
                        f"[dim]→[/dim] "
                        f"[magenta]{sample['target']}[/magenta]"
                    )

            console.print(cross_tree)
            console.print()

        # Close connection
        await kg.close()

    asyncio.run(_status())


@kg_app.command("calls")
def get_calls(
    function: str = typer.Argument(..., help="Function name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show call graph for a function.

    Display functions that call or are called by the specified function.

    Examples:
        mcp-vector-search kg calls "search"
        mcp-vector-search kg calls "SemanticSearchEngine.search"
    """
    project_root = project_root.resolve()

    async def _calls():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get call graph
        calls = await kg.get_call_graph(function)

        if not calls:
            console.print(
                f"[yellow]No call relationships found for '{function}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate calls vs called_by
        calls_out = [c for c in calls if c["direction"] == "calls"]
        calls_in = [c for c in calls if c["direction"] == "called_by"]

        # Display results
        if calls_out:
            table = Table(title=f"Functions called by '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_out:
                table.add_row(call["name"])

            console.print(table)

        if calls_in:
            table = Table(title=f"Functions that call '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_in:
                table.add_row(call["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_calls())


@kg_app.command("inherits")
def get_inheritance(
    class_name: str = typer.Argument(..., help="Class name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show inheritance tree for a class.

    Display parent and child classes in the inheritance hierarchy.

    Examples:
        mcp-vector-search kg inherits "BaseModel"
        mcp-vector-search kg inherits "SemanticSearchEngine"
    """
    project_root = project_root.resolve()

    async def _inherits():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get inheritance tree
        hierarchy = await kg.get_inheritance_tree(class_name)

        if not hierarchy:
            console.print(
                f"[yellow]No inheritance relationships found for '{class_name}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate parents vs children
        parents = [h for h in hierarchy if h["relation"] == "parent"]
        children = [h for h in hierarchy if h["relation"] == "child"]

        # Display results
        if parents:
            table = Table(title=f"Classes that '{class_name}' inherits from")
            table.add_column("Class", style="cyan")

            for parent in parents:
                table.add_row(parent["name"])

            console.print(table)

        if children:
            table = Table(title=f"Classes that inherit from '{class_name}'")
            table.add_column("Class", style="cyan")

            for child in children:
                table.add_row(child["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_inherits())


@kg_app.command("trace")
def kg_trace(
    entry_point: str = typer.Argument(..., help="Function or class name to trace from"),
    depth: int = typer.Option(
        3, "--depth", "-d", help="Max call chain depth (1–8)", min=1, max=8
    ),
    direction: str = typer.Option(
        "outgoing",
        "--direction",
        help="outgoing=what it calls, incoming=what calls it, both=full neighborhood",
    ),
    output_format: str = typer.Option(
        "tree",
        "--format",
        "-f",
        help="Output format: tree, flat, or json",
    ),
) -> None:
    """Trace execution flow from a function through the call graph.

    Shows the call chain from a function entry point, following CALLS
    relationships in the knowledge graph up to the specified depth.

    Requires 'mvs index kg' to have been run first.

    Examples:

      mvs kg trace handle_search_code

      mvs kg trace WikiPublisher --depth 4 --direction incoming

      mvs kg trace enrich_with_git_blame --format flat
    """
    if direction not in ("outgoing", "incoming", "both"):
        console.print(
            f"[red]Invalid direction '{direction}'. Use: outgoing, incoming, both[/red]"
        )
        raise typer.Exit(1)

    if output_format not in ("tree", "flat", "json"):
        console.print(
            f"[red]Invalid format '{output_format}'. Use: tree, flat, json[/red]"
        )
        raise typer.Exit(1)

    async def run():
        project_root = Path.cwd()
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"

        if not kg_path.exists():
            console.print(
                "[red]Knowledge graph not found.[/red] "
                "Run [cyan]mvs index kg[/cyan] to build it first."
            )
            raise typer.Exit(1)

        kg = KnowledgeGraph(kg_path)
        try:
            await kg.initialize()
        except Exception as e:
            console.print(f"[red]Failed to initialize knowledge graph: {e}[/red]")
            raise typer.Exit(1)

        with console.status(f"[bold blue]Tracing {entry_point}...[/bold blue]"):
            result = await kg.trace_execution_flow(
                entry_point=entry_point,
                depth=depth,
                direction=direction,
            )

        await kg.close()

        if result["entry"] is None:
            console.print(
                f"[yellow]No entity found matching '{entry_point}'.[/yellow] "
                "Try a more specific name."
            )
            raise typer.Exit(1)

        entry = result["entry"]
        nodes = result["nodes"]
        edges = result["edges"]

        if output_format == "json":
            # Raw JSON to stdout — pipe-friendly
            import sys

            json.dump(result, sys.stdout, indent=2, default=str)
            sys.stdout.write("\n")
            return

        if output_format == "flat":
            # Flat list — one entry per node, ordered by depth
            print(
                f"{entry['name']}  [{(entry.get('file_path') or '?').split('/src/')[-1]}]  (entry)"
            )
            for node in sorted(nodes, key=lambda n: (n["depth"], n["name"])):
                short = (node.get("file_path") or "?").split("/src/")[-1]
                print(f"{node['name']}  [{short}]  depth={node['depth']}")
            return

        # Tree format (default)
        short_entry_file = (entry.get("file_path") or "?").split("/src/")[-1]
        direction_label = {
            "outgoing": "calls",
            "incoming": "called by",
            "both": "related",
        }[direction]

        console.print()
        root_label = (
            f"[bold cyan]{entry['name']}[/bold cyan] "
            f"[dim]({entry.get('entity_type', 'function')})[/dim] "
            f"[dim green][{short_entry_file}][/dim green]"
        )
        tree = Tree(root_label)

        if not nodes:
            tree.add(f"[dim]No {direction_label} found[/dim]")
        else:
            # Build tree structure: group edges by depth, then by caller
            node_map = {n["id"]: n for n in nodes}
            node_map[entry["id"]] = entry

            def add_children(
                parent_tree: Tree, parent_id: str, current_depth: int, visited: set
            ):
                if current_depth > depth:
                    return
                child_edges = [e for e in edges if e["from_id"] == parent_id]
                for edge in child_edges:
                    child_id = edge["to_id"]
                    child = node_map.get(child_id)
                    if child is None:
                        continue
                    short_file = (child.get("file_path") or "?").split("/src/")[-1]
                    child_label = (
                        f"[cyan]{child['name']}[/cyan] "
                        f"[dim]({child.get('entity_type', '?')})[/dim] "
                        f"[dim green][{short_file}:{child.get('start_line', '?')}][/dim green]"
                    )
                    subtree = parent_tree.add(child_label)
                    if child_id not in visited:
                        visited.add(child_id)
                        add_children(subtree, child_id, current_depth + 1, visited)

            if direction in ("outgoing", "both"):
                add_children(tree, entry["id"], 1, {entry["id"]})
            else:
                # Incoming: show callers
                caller_edges = [e for e in edges if e["to_id"] == entry["id"]]
                for edge in caller_edges:
                    caller = node_map.get(edge["from_id"])
                    if caller:
                        short_file = (caller.get("file_path") or "?").split("/src/")[-1]
                        tree.add(
                            f"[cyan]{caller['name']}[/cyan] "
                            f"[dim green][{short_file}:{caller.get('start_line', '?')}][/dim green]"
                            f" [dim]calls this[/dim]"
                        )

        console.print(tree)
        console.print(
            f"\n[dim]Nodes: {result['total_nodes']} | "
            f"Edges: {len(edges)} | "
            f"Depth reached: {result['depth_reached']}/{depth}"
            + (" | [yellow]truncated[/yellow]" if result["truncated"] else "")
            + "[/dim]"
        )

    asyncio.run(run())


@kg_app.command("history")
def kg_history(
    entity_name: str = typer.Argument(..., help="Entity name to look up"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show the commit history recorded for a named entity.

    Returns the commit SHA(s) stored at last kg_build time for the entity.
    V1 note: reflects the most recent commit per file at kg_build time,
    not the full git log.

    Examples:
        mcp-vector-search kg history "MyClass"
        mcp-vector-search kg history "process_request"
    """
    project_root = project_root.resolve()

    async def _history() -> None:
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        history = await kg.get_entity_history(entity_name)
        await kg.close()

        if not history:
            console.print(
                f"[yellow]No entity named '{entity_name}' found in the knowledge graph.[/yellow]"
            )
            raise typer.Exit(0)

        table = Table(title=f"KG history: {entity_name}")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("File")
        table.add_column("Commit SHA", style="dim")

        for entry in history:
            table.add_row(
                entry["name"],
                entry["entity_type"],
                entry["file_path"] or "(none)",
                entry["commit_sha"] or "(none)",
            )

        console.print(table)
        console.print(
            "[dim]V1: reflects the most recent commit per file at kg_build time.[/dim]"
        )

    asyncio.run(_history())


@kg_app.command("callers-at")
def kg_callers_at(
    entity_name: str = typer.Argument(..., help="Callee entity name"),
    commit_sha: str = typer.Argument(..., help="Git commit SHA"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show what called entity_name as of a given commit.

    Returns callers whose stored commit_sha is an ancestor of (or equal to)
    the specified commit.  Requires the knowledge graph to be built first.

    Examples:
        mcp-vector-search kg callers-at "process_request" abc1234
        mcp-vector-search kg callers-at "MyClass.__init__" HEAD
    """
    project_root = project_root.resolve()

    async def _callers_at() -> None:
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        callers = await kg.get_callers_at_commit(entity_name, commit_sha, project_root)
        await kg.close()

        if not callers:
            console.print(
                f"[yellow]No callers found for '{entity_name}' at commit {commit_sha}.[/yellow]"
            )
            raise typer.Exit(0)

        table = Table(title=f"Callers of '{entity_name}' as of {commit_sha[:12]}")
        table.add_column("Caller", style="cyan")
        table.add_column("File")
        table.add_column("Caller commit SHA", style="dim")
        table.add_column("Callee", style="green")

        for c in callers:
            table.add_row(
                c["caller_name"],
                c["caller_file"] or "(none)",
                c["caller_commit_sha"] or "(none)",
                c["callee_name"],
            )

        console.print(table)

    asyncio.run(_callers_at())


@kg_app.command("ancestor")
def kg_ancestor(
    earlier: str = typer.Argument(..., help="Earlier (ancestor) commit SHA"),
    later: str = typer.Argument(..., help="Later (descendant) commit SHA"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Check if earlier commit is an ancestor of later commit.

    Prints 'yes' (exit 0) or 'no' (exit 1).

    Examples:
        mcp-vector-search kg ancestor abc1234 def5678
        mcp-vector-search kg ancestor HEAD~3 HEAD
    """
    from ...core.git_utils import is_ancestor_commit

    project_root = project_root.resolve()
    result = is_ancestor_commit(earlier, later, project_root)
    if result:
        console.print("yes")
        raise typer.Exit(0)
    else:
        console.print("no")
        raise typer.Exit(1)


@kg_app.command("visualize")
def visualize_kg(
    port: int = typer.Option(
        8502, "--port", "-p", help="Port for visualization server"
    ),
    output: Path = typer.Option(
        Path("kg-graph.json"),
        "--output",
        "-o",
        help="Output file for KG graph data",
    ),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Visualize knowledge graph with interactive D3.js force-directed layout.

    Opens a browser with an interactive visualization showing:
    - Nodes: Code entities (classes, functions, modules, files)
    - Edges: Relationships (calls, imports, inherits, contains)
    - Color-coded by entity type
    - Interactive: drag nodes, zoom, pan, click for details

    Examples:
        mcp-vector-search kg visualize
        mcp-vector-search kg visualize --port 3000
        mcp-vector-search kg visualize --output my-kg.json
    """
    project_root = project_root.resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Knowledge Graph Visualization[/bold cyan]\nProject: {project_root}",
            border_style="cyan",
        )
    )

    async def _export_kg():
        """Export KG data to JSON for visualization."""
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Check if graph exists
        stats = await kg.get_stats()
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            raise typer.Exit(1)

        # Export visualization data
        console.print("[cyan]Exporting knowledge graph data...[/cyan]")
        viz_data = await kg.get_initial_visualization_data(max_nodes=100)

        # Save to JSON
        with open(output, "w") as f:
            json.dump(viz_data, f, indent=2)

        console.print(
            f"[green]✓[/green] Exported {len(viz_data['nodes'])} nodes "
            f"and {len(viz_data['links'])} links to {output}"
        )

        # Close connection
        await kg.close()

        return viz_data

    # Export KG data
    asyncio.run(_export_kg())

    # Use specified port or find free one
    if port == 8502:  # Default port, try to find free one
        try:
            port = find_free_port(8502, 8599)
        except OSError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

    # Get visualization directory - use project-local storage
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        console.print(
            "[red]Project not initialized. Run 'mcp-vector-search init' first.[/red]"
        )
        raise typer.Exit(1)

    viz_dir = project_manager.project_root / ".mcp-vector-search" / "kg_visualization"

    if not viz_dir.exists():
        console.print(
            f"[yellow]Creating KG visualization directory at {viz_dir}...[/yellow]"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Create KG-specific HTML template
    html_file = viz_dir / "index.html"
    console.print("[yellow]Creating KG visualization HTML file...[/yellow]")
    _create_kg_html_template(html_file)

    # Copy KG graph data to visualization directory
    dest = viz_dir / "kg-graph.json"
    shutil.copy(output, dest)
    console.print(f"[green]✓[/green] Copied KG graph data to {dest}")

    # Start server
    console.print()
    start_visualization_server(port, viz_dir, auto_open=True)


def _create_kg_html_template(output_path: Path):
    """Create HTML template for KG visualization with D3.js force-directed graph.

    Args:
        output_path: Path to save HTML file
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }

        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }

        #controls h3 {
            margin-top: 0;
            color: #60a5fa;
            font-size: 16px;
        }

        #controls label {
            display: block;
            margin: 8px 0;
            cursor: pointer;
            user-select: none;
            font-size: 13px;
        }

        #controls input[type="checkbox"] {
            margin-right: 6px;
        }

        #info {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 400px;
            max-height: 600px;
            overflow-y: auto;
        }

        #info h3 {
            margin-top: 0;
            color: #60a5fa;
            font-size: 16px;
        }

        #info p {
            margin: 5px 0;
            font-size: 13px;
        }

        #info .label {
            color: #9ca3af;
            font-weight: 600;
        }

        #legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
        }

        #legend h4 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #60a5fa;
            font-size: 14px;
        }

        .legend-item {
            margin: 6px 0;
            display: flex;
            align-items: center;
            font-size: 12px;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
            display: inline-block;
        }

        #status {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(42, 42, 42, 0.95);
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 2000;
            display: none;
            color: #60a5fa;
            font-size: 14px;
        }

        svg {
            width: 100vw;
            height: 100vh;
            background: #1a1a1a;
        }

        .node {
            cursor: pointer;
            stroke: #fff;
            stroke-width: 1.5px;
        }

        .node:hover {
            stroke: #60a5fa;
            stroke-width: 3px;
        }

        .node.aggregate {
            stroke: #fbbf24;
            stroke-width: 2px;
        }

        .node.expanded {
            stroke: #10b981;
            stroke-width: 2px;
        }

        .link {
            stroke-opacity: 0.6;
        }

        .link:hover {
            stroke-opacity: 1;
            stroke-width: 3px;
        }

        .node-label {
            font-size: 10px;
            font-family: monospace;
            fill: #e0e0e0;
            pointer-events: none;
            text-anchor: middle;
        }
    </style>
</head>
<body>
    <div id="controls">
        <h3>Filters</h3>
        <h4 style="margin: 10px 0 5px 0; color: #9ca3af; font-size: 12px;">Code Relationships</h4>
        <label><input type="checkbox" id="show-calls" checked> Calls</label>
        <label><input type="checkbox" id="show-imports" checked> Imports</label>
        <label><input type="checkbox" id="show-inherits" checked> Inherits</label>
        <label><input type="checkbox" id="show-contains" checked> Contains</label>
        <h4 style="margin: 10px 0 5px 0; color: #9ca3af; font-size: 12px;">Documentation</h4>
        <label><input type="checkbox" id="show-follows" checked> Follows</label>
        <label><input type="checkbox" id="show-references" checked> References</label>
        <label><input type="checkbox" id="show-demonstrates" checked> Demonstrates</label>
        <label><input type="checkbox" id="show-has_tag" checked> Has Tag</label>
        <h4 style="margin: 10px 0 5px 0; color: #9ca3af; font-size: 12px;">Other</h4>
        <label><input type="checkbox" id="show-authored" checked> Authored</label>
        <label><input type="checkbox" id="show-part_of" checked> Part Of</label>
        <p style="margin-top: 15px; font-size: 11px; color: #9ca3af; border-top: 1px solid #374151; padding-top: 10px;">
            <strong>Tip:</strong> Double-click nodes to expand
        </p>
    </div>

    <div id="info" style="display: none;">
        <h3>Node Details</h3>
        <p><span class="label">Name:</span> <span id="info-name"></span></p>
        <p><span class="label">Type:</span> <span id="info-type"></span></p>
        <p><span class="label">File:</span> <span id="info-file"></span></p>
        <p><span class="label">Connections:</span> <span id="info-connections"></span></p>
        <p><span class="label">Status:</span> <span id="info-status"></span></p>
    </div>

    <div id="status">Loading...</div>

    <div id="legend">
        <h4>Node Types</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #6366f1;"></span>
            <span>File</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #8b5cf6;"></span>
            <span>Module</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #ec4899;"></span>
            <span>Class</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #10b981;"></span>
            <span>Function/Method</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #22d3ee;"></span>
            <span>Doc Section</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #a855f7;"></span>
            <span>Tag</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #f97316;"></span>
            <span>Person</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #14b8a6;"></span>
            <span>Project</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #fbbf24;"></span>
            <span>Aggregate (unexpanded)</span>
        </div>
        <h4 style="margin-top: 15px;">Relationships</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #f59e0b;"></span>
            <span>Calls</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #3b82f6;"></span>
            <span>Imports</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #ef4444;"></span>
            <span>Inherits</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #6b7280;"></span>
            <span>Contains</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #84cc16;"></span>
            <span>Follows</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #06b6d4;"></span>
            <span>References</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #a855f7;"></span>
            <span>Demonstrates</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #f472b6;"></span>
            <span>Has Tag</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #fb923c;"></span>
            <span>Authored</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #4ade80;"></span>
            <span>Part Of</span>
        </div>
    </div>

    <svg id="graph"></svg>

    <script>
        // Color schemes
        const nodeColors = {
            1: '#6366f1',  // file - indigo
            2: '#8b5cf6',  // module - purple
            3: '#ec4899',  // class - pink
            4: '#10b981',  // function/method - green
            5: '#22d3ee',  // doc_section - cyan
            6: '#a855f7',  // tag - purple
            7: '#f97316',  // person - orange
            8: '#14b8a6',  // project - teal
            9: '#fbbf24',  // aggregate - amber
            0: '#6b7280'   // unknown - gray
        };

        const linkColors = {
            'calls': '#f59e0b',        // orange
            'imports': '#3b82f6',      // blue
            'inherits': '#ef4444',     // red
            'contains': '#6b7280',     // gray
            'follows': '#84cc16',      // lime
            'references': '#06b6d4',   // cyan
            'demonstrates': '#a855f7', // purple
            'has_tag': '#f472b6',      // pink
            'authored': '#fb923c',     // orange
            'part_of': '#4ade80'       // green
        };

        // Load graph data
        fetch('/api/kg-graph')
            .then(response => response.json())
            .then(data => {
                initGraph(data.nodes, data.links);
            })
            .catch(error => {
                console.error('Error loading graph data:', error);
                alert('Failed to load knowledge graph data');
            });

        function initGraph(initialNodes, initialLinks) {
            const svg = d3.select('#graph');
            const width = window.innerWidth;
            const height = window.innerHeight;

            // Mutable graph state
            let nodes = [...initialNodes];
            let links = [...initialLinks];
            const expandedNodes = new Set();

            // Add zoom behavior
            const g = svg.append('g');
            svg.call(d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                }));

            // Create simulation
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links)
                    .id(d => d.id)
                    .distance(100))
                .force('charge', d3.forceManyBody()
                    .strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide()
                    .radius(30));

            // Create link group
            let linkGroup = g.append('g');
            let link = linkGroup.selectAll('line');

            // Create node group
            let nodeGroup = g.append('g');
            let node = nodeGroup.selectAll('circle');

            // Create label group
            let labelGroup = g.append('g');
            let label = labelGroup.selectAll('text');

            // Initial render
            updateGraph();

            // Drag functions
            function dragStarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragEnded(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            // Expand node on double-click
            async function expandNode(event, d) {
                event.stopPropagation();

                if (d.expandable === false) return;

                if (expandedNodes.has(d.id)) {
                    console.log('Node already expanded:', d.id);
                    return;
                }

                // Show loading indicator
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = `Expanding ${d.name}...`;
                statusDiv.style.display = 'block';

                try {
                    const response = await fetch(`/api/kg-expand/${encodeURIComponent(d.id)}`);
                    const data = await response.json();

                    if (data.error) {
                        console.error('Expand error:', data.error);
                        statusDiv.textContent = `Error: ${data.error}`;
                        setTimeout(() => statusDiv.style.display = 'none', 3000);
                        return;
                    }

                    if (data.nodes && data.nodes.length > 0) {
                        // Add new nodes
                        data.nodes.forEach(newNode => {
                            if (!nodes.find(n => n.id === newNode.id)) {
                                // Position near parent with some randomness
                                newNode.x = d.x + (Math.random() - 0.5) * 200;
                                newNode.y = d.y + (Math.random() - 0.5) * 200;
                                nodes.push(newNode);
                            }
                        });

                        // Add new links
                        data.links.forEach(newLink => {
                            const linkKey = `${newLink.source}-${newLink.target}`;
                            const reverseKey = `${newLink.target}-${newLink.source}`;
                            const exists = links.some(l => {
                                const lSource = typeof l.source === 'object' ? l.source.id : l.source;
                                const lTarget = typeof l.target === 'object' ? l.target.id : l.target;
                                const lKey = `${lSource}-${lTarget}`;
                                const lReverse = `${lTarget}-${lSource}`;
                                return lKey === linkKey || lReverse === reverseKey;
                            });
                            if (!exists) {
                                links.push(newLink);
                            }
                        });

                        // Update visualization
                        updateGraph();

                        expandedNodes.add(d.id);

                        // Mark node as expanded
                        d3.selectAll('.node')
                            .filter(n => n.id === d.id)
                            .classed('expanded', true);

                        statusDiv.textContent = `Expanded ${d.name}: +${data.nodes.length} nodes`;

                        // Show aggregation info if any
                        if (data.aggregations && data.aggregations.length > 0) {
                            const aggText = data.aggregations.map(a =>
                                `${a.type}: showing ${a.shown}/${a.total}`
                            ).join(', ');
                            console.log('Aggregated:', aggText);
                        }
                    } else {
                        statusDiv.textContent = `No neighbors found for ${d.name}`;
                    }

                    setTimeout(() => statusDiv.style.display = 'none', 2000);
                } catch (error) {
                    console.error('Expand failed:', error);
                    statusDiv.textContent = `Failed to expand: ${error.message}`;
                    setTimeout(() => statusDiv.style.display = 'none', 3000);
                }
            }

            function updateGraph() {
                // Update links
                link = link.data(links, d => {
                    const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                    const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                    return `${sourceId}-${targetId}`;
                });
                link.exit().remove();
                link = link.enter()
                    .append('line')
                    .attr('class', 'link')
                    .attr('stroke', d => linkColors[d.type] || '#6b7280')
                    .attr('stroke-width', 1)
                    .merge(link);

                // Update nodes
                node = node.data(nodes, d => d.id);
                node.exit().remove();
                const nodeEnter = node.enter()
                    .append('circle')
                    .attr('class', d => d.type === 'aggregate' ? 'node aggregate' : 'node')
                    .attr('r', d => d.type === 'aggregate' ? 12 : 8)
                    .attr('fill', d => nodeColors[d.group] || nodeColors[0])
                    .on('click', (event, d) => {
                        event.stopPropagation();
                        showInfo(d, links, nodes, node, link, label, expandedNodes);
                    })
                    .on('dblclick', expandNode)
                    .call(d3.drag()
                        .on('start', dragStarted)
                        .on('drag', dragged)
                        .on('end', dragEnded));
                node = nodeEnter.merge(node);

                // Update labels
                label = label.data(nodes, d => d.id);
                label.exit().remove();
                label = label.enter()
                    .append('text')
                    .attr('class', 'node-label')
                    .attr('dy', -12)
                    .text(d => d.name || d.id)
                    .merge(label);

                // Update simulation
                simulation.nodes(nodes);
                simulation.force('link').links(links);
                simulation.alpha(0.3).restart();

                // Update tick handler
                simulation.on('tick', () => {
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);

                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);

                    label
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                });
            }

            // Filter controls
            ['calls', 'imports', 'inherits', 'contains', 'follows', 'references',
             'demonstrates', 'has_tag', 'authored', 'part_of'].forEach(type => {
                const element = document.getElementById(`show-${type}`);
                if (element) {
                    element.addEventListener('change', (e) => {
                        filterLinks(link, e.target.checked, type);
                    });
                }
            });
        }

        function filterLinks(link, show, type) {
            link.style('display', d => {
                if (d.type === type) {
                    return show ? null : 'none';
                }
                return null;
            });
        }

        function showInfo(node, links, allNodes, nodeElements, linkElements, labelElements, expandedNodes) {
            const info = document.getElementById('info');
            document.getElementById('info-name').textContent = node.name || node.id;
            document.getElementById('info-type').textContent = node.type || 'unknown';
            document.getElementById('info-file').textContent = node.file_path || 'N/A';

            // Find connected node IDs
            const connectedNodes = new Set([node.id]);
            links.forEach(link => {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                if (sourceId === node.id) connectedNodes.add(targetId);
                if (targetId === node.id) connectedNodes.add(sourceId);
            });

            // Count connections
            document.getElementById('info-connections').textContent = connectedNodes.size - 1;

            // Show expansion status
            const statusText = expandedNodes && expandedNodes.has(node.id) ? 'Expanded' :
                              (node.expandable === false ? 'Not expandable' : 'Not expanded (double-click to expand)');
            document.getElementById('info-status').textContent = statusText;

            // Dim unconnected nodes
            nodeElements.style('opacity', d => connectedNodes.has(d.id) ? 1 : 0.1);
            labelElements.style('opacity', d => connectedNodes.has(d.id) ? 1 : 0.1);

            // Dim unconnected edges
            linkElements.style('opacity', d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                return (sourceId === node.id || targetId === node.id) ? 0.8 : 0.05;
            });

            info.style.display = 'block';
        }

        // Close info on background click and reset view
        const svg = d3.select('#graph');
        svg.on('click', (event) => {
            if (event.target === svg.node() || event.target.tagName === 'svg') {
                // Reset all opacities
                d3.selectAll('.node').style('opacity', 1);
                d3.selectAll('.node-label').style('opacity', 1);
                d3.selectAll('.link').style('opacity', 0.6);
                document.getElementById('info').style.display = 'none';
            }
        });
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    console.print(f"[green]✓[/green] Created KG visualization HTML at {output_path}")
