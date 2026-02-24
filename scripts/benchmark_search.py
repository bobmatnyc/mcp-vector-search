#!/usr/bin/env python3
"""Benchmark search query performance.

Tests both search paths:
1. LanceVectorDatabase (legacy path used by CLI search) - no nprobes/refine_factor
2. VectorsBackend (two-phase pipeline path) - with nprobes=20, refine_factor=5, IVF_PQ

Also verifies:
- IVF-PQ index existence and structure
- Contextual chunking output
- Chunk count and index status
"""

import asyncio
import statistics
import time
from pathlib import Path


async def benchmark_vectors_backend(
    db_path: Path,
    embedding_function,
    queries: list[str],
) -> dict:
    """Benchmark VectorsBackend (two-phase pipeline with IVF_PQ + nprobes/refine_factor)."""
    from mcp_vector_search.core.vectors_backend import VectorsBackend

    backend = VectorsBackend(db_path=db_path)
    await backend.initialize()

    if backend._table is None:
        return {"error": "VectorsBackend table not found"}

    row_count = backend._table.count_rows()

    # Warm up
    warmup_start = time.perf_counter()
    warmup_vec = embedding_function(["warmup query"])[0]
    await backend.search(warmup_vec, limit=1)
    warmup_time = time.perf_counter() - warmup_start

    # Run benchmarks
    all_times = []
    query_results = []

    for query in queries:
        times = []
        result_count = 0

        for _ in range(3):
            query_vec = embedding_function([query])[0]
            start = time.perf_counter()
            results = await backend.search(query_vec, limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            result_count = len(results)

        median_ms = statistics.median(times)
        all_times.append(median_ms)
        query_results.append((query, median_ms, result_count))

    await backend.close()

    return {
        "backend": "VectorsBackend (IVF_PQ + nprobes=20 + refine_factor=5)",
        "table": "vectors.lance",
        "row_count": row_count,
        "warmup_time": warmup_time,
        "query_results": query_results,
        "all_times": all_times,
    }


async def benchmark_lance_database(
    lance_path: Path,
    embedding_function,
    queries: list[str],
) -> dict:
    """Benchmark LanceVectorDatabase (legacy path used by CLI search - no ANN params)."""
    from mcp_vector_search.core.lancedb_backend import LanceVectorDatabase

    db = LanceVectorDatabase(
        persist_directory=lance_path,
        embedding_function=embedding_function,
        collection_name="vectors",
    )
    await db.initialize()

    if db._table is None:
        return {"error": "LanceVectorDatabase table not found"}

    row_count = db._table.count_rows()

    # Warm up
    warmup_start = time.perf_counter()
    await db.search("warmup query", limit=1, similarity_threshold=0.0)
    warmup_time = time.perf_counter() - warmup_start

    # Run benchmarks
    all_times = []
    query_results = []

    for query in queries:
        times = []
        result_count = 0

        for _ in range(3):
            start = time.perf_counter()
            results = await db.search(query, limit=10, similarity_threshold=0.0)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            result_count = len(results)

        median_ms = statistics.median(times)
        all_times.append(median_ms)
        query_results.append((query, median_ms, result_count))

    await db.close()

    return {
        "backend": "LanceVectorDatabase (legacy CLI path, no ANN params)",
        "table": "vectors.lance (via chunks collection_name='vectors')",
        "row_count": row_count,
        "warmup_time": warmup_time,
        "query_results": query_results,
        "all_times": all_times,
    }


def check_ivf_pq_index(lance_path: Path) -> dict:
    """Check for IVF-PQ index files in the LanceDB directory."""
    vectors_lance = lance_path / "vectors.lance"
    indices_dir = vectors_lance / "_indices"

    result = {
        "vectors_lance_exists": vectors_lance.exists(),
        "indices_dir_exists": indices_dir.exists(),
        "index_files": [],
        "index_sizes": {},
    }

    if indices_dir.exists():
        for idx_dir in indices_dir.iterdir():
            if idx_dir.is_dir():
                for f in idx_dir.iterdir():
                    if f.is_file():
                        size_kb = f.stat().st_size / 1024
                        result["index_files"].append(str(f.name))
                        result["index_sizes"][f.name] = f"{size_kb:.1f} KB"

    return result


def show_contextual_chunking_sample(lance_path: Path) -> None:
    """Show a sample of contextual text built from stored chunk data."""
    import lancedb

    from mcp_vector_search.core.context_builder import build_contextual_text

    try:
        db = lancedb.connect(str(lance_path))
        tables = db.list_tables()
        table_names = tables.tables if hasattr(tables, "tables") else tables

        if "vectors" in table_names:
            table = db.open_table("vectors")
            # Get a sample Python chunk with a function name
            sample_results = (
                table.search()
                .limit(100)
                .where("language = 'python' AND function_name != ''")
                .to_list()
            )

            if sample_results:
                # Pick first result with a function name
                sample = sample_results[0]
                # Build a mock chunk dict to show contextual text
                mock_chunk = {
                    "file_path": sample.get("file_path", ""),
                    "language": sample.get("language", ""),
                    "class_name": sample.get("class_name", ""),
                    "function_name": sample.get("function_name", ""),
                    "imports": [],
                    "docstring": "",
                    "content": sample.get("content", "")[:200],
                }
                ctx_text = build_contextual_text(mock_chunk)
                print("\nContextual Chunking Sample:")
                print("-" * 50)
                print(ctx_text[:500])
                print("-" * 50)
                return

        # Fall back to chunks table
        if "chunks" in table_names:
            table = db.open_table("chunks")
            sample_results = table.search().limit(5).to_list()
            if sample_results:
                sample = sample_results[0]
                from mcp_vector_search.core.models import CodeChunk
                # Just show what context builder would produce

                class MockChunk:
                    file_path = sample.get("file_path", "")
                    language = sample.get("language", "")
                    class_name = sample.get("class_name", "")
                    function_name = sample.get("function_name", "")
                    imports = []
                    docstring = sample.get("docstring", "")
                    content = sample.get("content", "")[:200]

                ctx_text = build_contextual_text(MockChunk())
                print("\nContextual Chunking Sample (from chunks table):")
                print("-" * 50)
                print(ctx_text[:500])
                print("-" * 50)

    except Exception as e:
        print(f"\nCould not load contextual chunking sample: {e}")


def print_results(result: dict) -> None:
    """Print benchmark results in a formatted table."""
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    query_results = result.get("query_results", [])
    all_times = result.get("all_times", [])

    print(f"  Table:    {result.get('table', 'unknown')}")
    print(f"  Rows:     {result.get('row_count', 0):,}")
    print(f"  Warmup:   {result.get('warmup_time', 0):.3f}s")
    print()
    print(f"  {'Query':<40} {'Time (ms)':>10} {'Results':>8}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 8}")

    for query, median_ms, result_count in query_results:
        print(f"  {query:<40} {median_ms:>8.1f}ms {result_count:>7}")

    if all_times:
        sorted_times = sorted(all_times)
        p95_idx = min(int(len(all_times) * 0.95), len(all_times) - 1)
        print()
        print(
            f"  Summary: median={statistics.median(all_times):.1f}ms "
            f"mean={statistics.mean(all_times):.1f}ms "
            f"p95={sorted_times[p95_idx]:.1f}ms "
            f"min={min(all_times):.1f}ms "
            f"max={max(all_times):.1f}ms"
        )


async def main():
    from mcp_vector_search.core.embeddings import create_embedding_function
    from mcp_vector_search.core.project import ProjectManager

    project_root = Path("/Users/masa/Projects/mcp-vector-search")
    pm = ProjectManager(project_root)
    config = pm.load_config()

    embedding_function, _ = create_embedding_function(model_name=config.embedding_model)
    lance_path = config.index_path / "lance"

    print("=" * 65)
    print("Vector Search Performance Benchmark")
    print("=" * 65)
    print(f"Project:    {project_root}")
    print(f"Model:      {config.embedding_model or 'default (all-MiniLM-L6-v2)'}")
    print(f"Lance path: {lance_path}")
    print()

    # Step 1: Check IVF-PQ index
    print("IVF-PQ Index Check")
    print("-" * 65)
    index_info = check_ivf_pq_index(lance_path)
    print(f"  vectors.lance exists:  {index_info['vectors_lance_exists']}")
    print(f"  _indices dir exists:   {index_info['indices_dir_exists']}")
    if index_info["index_files"]:
        print("  Index files found:")
        for fname, size in index_info["index_sizes"].items():
            print(f"    {fname}: {size}")
        print("  STATUS: IVF-PQ index is PRESENT")
    else:
        print("  STATUS: No IVF-PQ index found (brute-force search will be used)")
    print()

    # Test queries
    queries = [
        "search function",
        "embedding model configuration",
        "parse javascript imports",
        "create vector index",
        "file watcher reindex",
        "async database connection",
        "CLI command handler",
        "knowledge graph build",
        "cross encoder reranking",
        "tree sitter AST parsing",
    ]

    # Step 2: Benchmark VectorsBackend (optimized path)
    print("Benchmark 1: VectorsBackend (IVF_PQ + nprobes=20 + refine_factor=5)")
    print("-" * 65)
    try:
        result1 = await benchmark_vectors_backend(
            lance_path, embedding_function, queries
        )
        print_results(result1)
    except Exception as e:
        print(f"  VectorsBackend benchmark failed: {e}")
        result1 = {"error": str(e)}
    print()

    # Step 3: Benchmark LanceVectorDatabase (legacy CLI path)
    print("Benchmark 2: LanceVectorDatabase (legacy CLI path, cosine search)")
    print("-" * 65)
    try:
        result2 = await benchmark_lance_database(
            lance_path, embedding_function, queries
        )
        print_results(result2)
    except Exception as e:
        print(f"  LanceVectorDatabase benchmark failed: {e}")
        result2 = {"error": str(e)}
    print()

    # Step 4: Show contextual chunking
    show_contextual_chunking_sample(lance_path)

    # Step 5: Speed comparison summary
    if "all_times" in result1 and "all_times" in result2:
        t1 = statistics.median(result1["all_times"])
        t2 = statistics.median(result2["all_times"])
        print()
        print("=" * 65)
        print("Comparison Summary")
        print("=" * 65)
        print(f"  VectorsBackend (IVF_PQ):  {t1:.1f}ms median")
        print(f"  LanceVectorDatabase:       {t2:.1f}ms median")
        if t1 < t2:
            speedup = t2 / t1
            print(f"  VectorsBackend is {speedup:.2f}x faster")
        elif t2 < t1:
            speedup = t1 / t2
            print(f"  LanceVectorDatabase is {speedup:.2f}x faster")
        else:
            print("  Both backends have similar performance")

    print()
    print("=" * 65)
    print("Benchmark complete")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
