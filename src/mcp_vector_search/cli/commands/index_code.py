"""Index-code command for building code-specific embeddings using CodeT5+."""

import asyncio
import sys
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from ...core.chunks_backend import ChunksBackend
from ...core.embeddings import create_embedding_function
from ...core.progress import ProgressTracker
from ...core.vectors_backend import VectorsBackend

console = Console(stderr=True)

# Typer app for index-code command
app = typer.Typer(
    help="Build code-specific embeddings using CodeT5+ model",
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    project_root: str = typer.Argument(
        ".", help="Project root directory (default: current directory)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild code index from scratch"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Build code-specific embeddings using CodeT5+ model.

    Creates a separate code_vectors index alongside the main index.
    Uses Salesforce/codet5p-110m-embedding for code-to-code similarity.

    Requirements:
      - Run 'mcp-vector-search index' first to parse and chunk files

    Example:
      mcp-vector-search index-code              # Build code index
      mcp-vector-search index-code --force      # Rebuild from scratch
      mcp-vector-search index-code --verbose    # Show detailed progress
    """
    # Skip if subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    try:
        asyncio.run(_index_code(project_root, force, verbose))
    except KeyboardInterrupt:
        console.print("\n[yellow]Code indexing cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            logger.exception("Code indexing failed")
        sys.exit(1)


async def _index_code(project_root_str: str, force: bool, verbose: bool) -> None:
    """Internal async function to build code-specific index.

    Args:
        project_root_str: Path to project root
        force: Force rebuild even if index exists
        verbose: Enable verbose logging
    """
    project_root = Path(project_root_str).resolve()

    # 1. Check chunks.lance exists
    index_path = project_root / ".mcp-vector-search" / "lance"
    chunks_path = index_path / "chunks.lance"

    if not chunks_path.exists():
        console.print(
            "[red]Error: No chunks index found. Run 'mcp-vector-search index' first.[/red]"
        )
        console.print(
            "[dim]The main index must be built before creating a code-specific index.[/dim]"
        )
        sys.exit(1)

    # 2. Check if code_vectors already exists (unless --force)
    code_vectors_path = index_path / "code_vectors.lance"
    if code_vectors_path.exists() and not force:
        console.print(
            "[yellow]Code vectors index already exists. Use --force to rebuild.[/yellow]"
        )
        console.print(f"[dim]Index location: {code_vectors_path}[/dim]")
        sys.exit(0)

    # 3. Load chunks from chunks.lance, filter to code-only
    console.print("\n[bold blue]Building Code-Specific Index[/bold blue]")
    console.print("━" * 50)
    console.print("  Model: [cyan]Salesforce/codet5p-110m-embedding[/cyan]")
    console.print("  Loading chunks from: [dim]chunks.lance[/dim]")

    # Get all chunks directly from chunks table
    chunks_backend = ChunksBackend(index_path)
    await chunks_backend.initialize()

    # Read all chunks from LanceDB table
    if chunks_backend._table is None:
        console.print("[red]Error: Chunks table not found.[/red]")
        sys.exit(1)

    # Convert chunks table to list of dicts
    chunks_df = chunks_backend._table.to_pandas()
    all_chunks = chunks_df.to_dict("records")

    # Filter to code chunks only
    code_chunk_types = {
        "function",
        "class",
        "method",
        "constructor",
        "impl",
        "struct",
        "enum",
        "trait",
        "interface",
        "module",
    }
    doc_languages = {"text", "markdown", "rst"}

    code_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.get("chunk_type") in code_chunk_types
        and chunk.get("language") not in doc_languages
    ]

    if not code_chunks:
        console.print("\n[yellow]No code chunks found. Nothing to index.[/yellow]")
        console.print(
            "[dim]Tip: Make sure you've indexed code files (not just docs).[/dim]"
        )
        sys.exit(0)

    console.print(
        f"  Code chunks: [green]{len(code_chunks):,}[/green] (of {len(all_chunks):,} total)"
    )
    console.print()

    # 4. Create CodeT5+ embedding function
    cache_path = project_root / ".mcp-vector-search" / "cache" / "code_model"

    console.print("[cyan]Phase 1/2: Loading CodeT5+ model...[/cyan]")
    start_time = time.time()

    embedding_fn, embedding_cache = create_embedding_function(
        model_name="Salesforce/codet5p-110m-embedding",
        cache_dir=cache_path,
    )

    model_load_time = time.time() - start_time
    actual_dim = embedding_fn.ndims() if hasattr(embedding_fn, "ndims") else 256
    console.print(
        f"  [green]✓[/green] Model loaded ({model_load_time:.1f}s, {actual_dim}d vectors)"
    )

    # 5. Create code_vectors backend
    code_vectors_backend = VectorsBackend(
        db_path=index_path,
        vector_dim=actual_dim,
        table_name="code_vectors",
    )
    await code_vectors_backend.initialize()

    # 6. Embed code chunks with progress bar
    console.print(
        f"\n[cyan]Phase 2/2: Embedding {len(code_chunks):,} code chunks...[/cyan]"
    )

    tracker = ProgressTracker(console, verbose=verbose)

    embed_start = time.time()
    batch_size = 32
    total_embedded = 0
    batches = []

    for i in range(0, len(code_chunks), batch_size):
        batch = code_chunks[i : i + batch_size]
        contents = [chunk["content"] for chunk in batch]

        # Generate embeddings
        vectors = embedding_fn(contents)

        # Build records for storage
        batch_records = []
        for j, chunk in enumerate(batch):
            record = {
                "chunk_id": chunk["chunk_id"],
                "vector": vectors[j],
                "file_path": chunk["file_path"],
                "content": chunk["content"],
                "language": chunk["language"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_type": chunk["chunk_type"],
                "name": chunk["name"],
                "hierarchy_path": chunk.get("hierarchy_path", ""),
            }
            batch_records.append(record)

        batches.extend(batch_records)
        total_embedded += len(batch)

        # Show progress
        tracker.progress_bar_with_eta(
            total_embedded,
            len(code_chunks),
            prefix="Embedding code chunks",
            start_time=embed_start,
        )

    # 7. Store all vectors
    if batches:
        await code_vectors_backend.add_vectors(
            batches,
            model_version="codet5p-110m-embedding",
        )

    embed_time = time.time() - embed_start
    total_time = time.time() - start_time

    # 8. Summary
    console.print("\n[green]✓ Code index built successfully[/green]")
    console.print(f"  Chunks embedded: [cyan]{total_embedded:,}[/cyan]")
    console.print(f"  Vector dimensions: [cyan]{actual_dim}[/cyan]")
    console.print(f"  Model load time: [dim]{model_load_time:.1f}s[/dim]")
    console.print(f"  Embedding time: [dim]{embed_time:.1f}s[/dim]")
    console.print(f"  Total time: [cyan]{total_time:.1f}s[/cyan]")
    console.print(
        f"  Speed: [dim]{total_embedded / max(embed_time, 0.1):.0f} chunks/sec[/dim]"
    )
    console.print(f"  Index: [dim]{code_vectors_path}[/dim]")

    # Close embedding cache if needed
    if embedding_cache and hasattr(embedding_cache, "close"):
        embedding_cache.close()

    # Close backends
    await code_vectors_backend.close()
    await chunks_backend.close()


if __name__ == "__main__":
    app()
