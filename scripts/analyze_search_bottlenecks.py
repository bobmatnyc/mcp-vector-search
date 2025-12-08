#!/usr/bin/env python3
"""Analyze search performance bottlenecks and suggest optimizations."""

import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.search import SemanticSearchEngine


class PerformanceAnalyzer:
    """Analyze search performance and identify bottlenecks."""

    def __init__(self):
        self.results = {}
        self.recommendations = []

    async def time_with_breakdown(self, name: str, operation, *args, **kwargs):
        """Time an operation and break down the components."""
        start_total = time.perf_counter()

        # Time the operation
        result = await operation(*args, **kwargs)

        end_total = time.perf_counter()
        total_time = (end_total - start_total) * 1000

        self.results[name] = {
            "total_time": total_time,
            "result_count": len(result) if hasattr(result, "__len__") else 0,
            "result": result,
        }

        return result, total_time

    def analyze_embedding_performance(self):
        """Analyze embedding function performance."""
        print("🧠 Analyzing embedding performance...")

        # Test embedding creation time
        times = []
        for _i in range(3):
            start = time.perf_counter()
            embedding_function, _ = create_embedding_function(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = statistics.mean(times)
        print(f"  Embedding function creation: {avg_time:.2f}ms (avg of 3 runs)")

        if avg_time > 2000:  # 2 seconds
            self.recommendations.append(
                "⚠️  Embedding function creation is slow. Consider caching or using a lighter model."
            )

        return embedding_function

    async def analyze_database_performance(self, database, project_dir):
        """Analyze database initialization and operations."""
        print("💾 Analyzing database performance...")

        # Test database initialization
        start = time.perf_counter()
        await database.initialize()
        init_time = (time.perf_counter() - start) * 1000
        print(f"  Database initialization: {init_time:.2f}ms")

        if init_time > 100:
            self.recommendations.append(
                "⚠️  Database initialization is slow. Check disk I/O and consider SSD storage."
            )

    async def analyze_indexing_performance(self, indexer, project_dir):
        """Analyze indexing performance in detail."""
        print("📚 Analyzing indexing performance...")

        # Create test files of different sizes
        test_files = {
            "small.py": "def hello(): return 'world'",
            "medium.py": "\n".join(
                [f"def function_{i}(): return {i}" for i in range(50)]
            ),
            "large.py": "\n".join(
                [f"def function_{i}(): return {i}" for i in range(200)]
            ),
        }

        for filename, content in test_files.items():
            (project_dir / filename).write_text(content)

        # Time full indexing
        start = time.perf_counter()
        indexed_count = await indexer.index_project()
        total_indexing_time = (time.perf_counter() - start) * 1000

        stats = await indexer.get_indexing_stats()

        print(f"  Total indexing time: {total_indexing_time:.2f}ms")
        print(f"  Files indexed: {indexed_count}")
        print(f"  Chunks created: {stats['total_chunks']}")
        print(
            f"  Indexing rate: {stats['total_chunks'] / (total_indexing_time / 1000):.1f} chunks/sec"
        )

        if total_indexing_time / indexed_count > 100:  # > 100ms per file
            self.recommendations.append(
                "⚠️  Indexing is slow per file. Consider optimizing chunk size or parser performance."
            )

        return stats

    async def analyze_search_components(self, search_engine):
        """Analyze individual search components."""
        print("🔍 Analyzing search component performance...")

        # Test different aspects of search
        test_queries = [
            "function",
            "class User",
            "def hello",
            "return value",
            "import module",
        ]

        search_times = []

        for query in test_queries:
            # Time the search with very detailed logging
            start = time.perf_counter()

            # Test query preprocessing
            preprocess_start = time.perf_counter()
            search_engine._preprocess_query(query)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000

            # Test the actual search
            search_start = time.perf_counter()
            results = await search_engine.search(
                query, limit=10, similarity_threshold=0.01
            )
            search_time = (time.perf_counter() - search_start) * 1000

            total_time = (time.perf_counter() - start) * 1000
            search_times.append(total_time)

            print(f"  Query '{query}':")
            print(f"    Preprocessing: {preprocess_time:.2f}ms")
            print(f"    Search execution: {search_time:.2f}ms")
            print(f"    Total: {total_time:.2f}ms")
            print(f"    Results: {len(results)}")

        avg_search_time = statistics.mean(search_times)

        if avg_search_time > 50:
            self.recommendations.append(
                "⚠️  Search is slow. Consider optimizing similarity threshold or result processing."
            )

        return search_times

    async def analyze_concurrent_performance(self, search_engine):
        """Analyze concurrent search performance."""
        print("🚀 Analyzing concurrent search performance...")

        async def single_search(query, delay=0):
            if delay:
                await asyncio.sleep(delay)
            start = time.perf_counter()
            result = await search_engine.search(
                query, limit=5, similarity_threshold=0.01
            )
            end = time.perf_counter()
            return len(result), (end - start) * 1000

        # Test different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        queries = ["function", "class", "import", "return", "def"]

        for concurrency in concurrency_levels:
            print(f"\n  Testing {concurrency} concurrent searches:")

            # Run concurrent searches
            start_wall = time.perf_counter()
            tasks = [
                single_search(queries[i % len(queries)]) for i in range(concurrency)
            ]
            results = await asyncio.gather(*tasks)
            wall_time = (time.perf_counter() - start_wall) * 1000

            individual_times = [r[1] for r in results]
            total_results = sum(r[0] for r in results)
            avg_individual = statistics.mean(individual_times)

            print(f"    Wall time: {wall_time:.2f}ms")
            print(f"    Avg individual: {avg_individual:.2f}ms")
            print(
                f"    Efficiency: {(avg_individual * concurrency / wall_time) * 100:.1f}%"
            )
            print(f"    Total results: {total_results}")

            # Check for performance degradation
            if concurrency > 1:
                efficiency = (avg_individual * concurrency / wall_time) * 100
                if efficiency < 80:
                    self.recommendations.append(
                        f"⚠️  Concurrent performance degrades at {concurrency} searches. "
                        f"Consider connection pooling or async optimization."
                    )

    async def analyze_memory_usage(self, search_engine):
        """Analyze memory usage patterns."""
        print("💾 Analyzing memory usage patterns...")

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run many searches to see memory growth
            for i in range(100):
                await search_engine.search(
                    f"test query {i}", limit=5, similarity_threshold=0.01
                )

                if i % 20 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"    After {i} searches: {current_memory:.1f}MB")

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory

            print(f"  Memory growth after 100 searches: {memory_growth:.1f}MB")

            if memory_growth > 50:  # 50MB growth
                self.recommendations.append(
                    "⚠️  Significant memory growth detected. Check for memory leaks or caching issues."
                )

        except ImportError:
            print("  psutil not available, skipping memory analysis")

    def print_recommendations(self):
        """Print performance recommendations."""
        print("\n💡 PERFORMANCE RECOMMENDATIONS:")
        print("=" * 50)

        if not self.recommendations:
            print("✅ No performance issues detected! Your search is well optimized.")
        else:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")

        print("\n🎯 GENERAL OPTIMIZATION TIPS:")
        print("- Use appropriate similarity thresholds (0.1-0.3 for most cases)")
        print("- Limit result counts to what you actually need")
        print("- Consider caching frequently used queries")
        print("- Monitor indexing performance as your codebase grows")
        print("- Use concurrent searches for batch operations")


async def main():
    """Run comprehensive performance analysis."""
    print("🔬 Search Performance Bottleneck Analysis")
    print("=" * 50)

    analyzer = PerformanceAnalyzer()

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Analyze embedding performance
        embedding_function = analyzer.analyze_embedding_performance()

        # Set up components
        database = ChromaVectorDatabase(
            persist_directory=project_dir / "chroma_db",
            embedding_function=embedding_function,
            collection_name="performance_analysis",
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_dir,
            file_extensions=[".py"],
        )

        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_dir,
            similarity_threshold=0.1,
        )

        async with database:
            # Analyze database performance
            await analyzer.analyze_database_performance(database, project_dir)

            # Analyze indexing performance
            await analyzer.analyze_indexing_performance(indexer, project_dir)

            # Analyze search components
            await analyzer.analyze_search_components(search_engine)

            # Analyze concurrent performance
            await analyzer.analyze_concurrent_performance(search_engine)

            # Analyze memory usage
            await analyzer.analyze_memory_usage(search_engine)

        # Print recommendations
        analyzer.print_recommendations()


if __name__ == "__main__":
    asyncio.run(main())
