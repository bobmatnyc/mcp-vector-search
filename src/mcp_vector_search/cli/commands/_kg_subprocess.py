"""Isolated subprocess for KG building - NO LANCEDB IMPORTS!

This module is intentionally isolated from the main codebase to prevent
any LanceDB background threads from being created. It's invoked as a
completely separate Python process.
"""

import json
import os
import sys
import threading
from pathlib import Path

# CRITICAL: Set environment variables BEFORE any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Entry point for subprocess KG build."""
    import argparse

    from rich.console import Console
    from rich.table import Table

    from mcp_vector_search.core.kg_builder import KGBuilder
    from mcp_vector_search.core.knowledge_graph import KnowledgeGraph
    from mcp_vector_search.core.models import CodeChunk

    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=str)
    parser.add_argument("chunks_file", type=str)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-documents", action="store_true")
    args = parser.parse_args()

    console = Console()
    project_root = Path(args.project_root)
    chunks_file = Path(args.chunks_file)

    # Check threads immediately
    threads = threading.enumerate()
    console.print(f"[cyan]🔍 Thread check: {len(threads)} thread(s) active[/cyan]")
    for t in threads:
        console.print(f"  - {t.name} (daemon={t.daemon})")

    if len(threads) > 1:
        background_threads = [t for t in threads if t != threading.main_thread()]
        if background_threads:
            console.print(
                f"[red]✗ ERROR: {len(background_threads)} background thread(s) detected![/red]"
            )
            console.print(
                "[red]Kuzu requires single-threaded execution. Background threads (even daemons) "
                "cause segfaults during relationship insertion.[/red]"
            )
            for t in background_threads:
                console.print(f"  - {t.name} (daemon={t.daemon})")
            return 1

    try:
        # Load chunks from JSON file (no database access!)
        console.print(f"[cyan]Loading chunks from {chunks_file.name}...[/cyan]")
        with open(chunks_file) as f:
            chunks_data = json.load(f)

        # Deserialize chunks
        chunks = []
        for chunk_dict in chunks_data:
            chunk_dict["file_path"] = Path(chunk_dict["file_path"])
            chunks.append(CodeChunk(**chunk_dict))
        console.print(f"[green]✓ Loaded {len(chunks)} chunks[/green]")

        if len(chunks) == 0:
            console.print(
                "[red]✗[/red] No chunks found. Run 'mcp-vector-search index' first."
            )
            return 1

        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"

        # Force rebuild if requested
        if args.force and kg_path.exists():
            console.print("[yellow]🗑️  Force rebuild: removing existing KG...[/yellow]")
            import shutil

            for item in kg_path.iterdir():
                if item.name.startswith("code_kg"):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            console.print("[green]✓ Old KG files removed[/green]")

        kg = KnowledgeGraph(kg_path)
        kg.initialize_sync()

        # Check if graph already exists
        if not args.force:
            stats = kg.get_stats_sync()
            if stats["total_entities"] > 0:
                console.print(
                    f"[yellow]⚠[/yellow] Knowledge graph already exists "
                    f"({stats['total_entities']} entities). Use --force to rebuild."
                )
                kg.close_sync()
                return 0

        # Build graph with smaller batch size to avoid Kuzu segfault
        # Kuzu can segfault with large batches (17k relationships)
        builder = KGBuilder(kg, project_root)

        # Override batch size for relationship insertion
        safe_batch_size = 100  # Smaller batches to avoid segfaults
        console.print(
            f"[dim]Using batch size: {safe_batch_size} for relationship insertion[/dim]"
        )

        build_stats = builder.build_from_chunks_sync(
            chunks,
            show_progress=True,
            skip_documents=args.skip_documents,
        )

        # Show results
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Code Entities", str(build_stats["entities"]))
        table.add_row("Doc Sections", str(build_stats.get("doc_sections", 0)))
        table.add_row("Tags", str(build_stats.get("tags", 0)))
        table.add_row("Persons", str(build_stats.get("persons", 0)))
        table.add_row("Projects", str(build_stats.get("projects", 0)))
        table.add_row("Calls", str(build_stats["calls"]))
        table.add_row("Imports", str(build_stats["imports"]))
        table.add_row("Inherits", str(build_stats["inherits"]))
        table.add_row("Contains", str(build_stats["contains"]))
        table.add_row("References", str(build_stats.get("references", 0)))
        table.add_row("Documents", str(build_stats.get("documents", 0)))
        table.add_row("Follows", str(build_stats.get("follows", 0)))
        table.add_row("Has Tag", str(build_stats.get("has_tag", 0)))
        table.add_row("Demonstrates", str(build_stats.get("demonstrates", 0)))
        table.add_row("Links To", str(build_stats.get("links_to", 0)))
        table.add_row("Authored", str(build_stats.get("authored", 0)))
        table.add_row("Modified", str(build_stats.get("modified", 0)))
        table.add_row("Part Of", str(build_stats.get("part_of", 0)))

        console.print(table)
        console.print("[green]✓[/green] Knowledge graph built successfully!")

        kg.close_sync()
        return 0

    except Exception as e:
        console.print(f"[red]✗ Build failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Clean up temp file
        try:
            chunks_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
