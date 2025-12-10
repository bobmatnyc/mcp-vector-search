"""Performance tests for search operations."""

import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.search import SemanticSearchEngine


class SearchPerformanceTester:
    """Utility class for measuring search performance."""

    def __init__(
        self, database: ChromaVectorDatabase, search_engine: SemanticSearchEngine
    ):
        self.database = database
        self.search_engine = search_engine
        self.timing_results: dict[str, list[float]] = {}

    async def time_operation(
        self, operation_name: str, operation_func, *args, **kwargs
    ) -> Any:
        """Time an async operation and store the result."""
        async with self.database:
            start_time = time.perf_counter()
            result = await operation_func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            if operation_name not in self.timing_results:
                self.timing_results[operation_name] = []
            self.timing_results[operation_name].append(elapsed)

            return result, elapsed

    def get_stats(self, operation_name: str) -> dict[str, float]:
        """Get timing statistics for an operation."""
        if operation_name not in self.timing_results:
            return {}

        times = self.timing_results[operation_name]
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "total": sum(times),
        }

    def print_performance_report(self):
        """Print a detailed performance report."""
        print("\n" + "=" * 60)
        print("SEARCH PERFORMANCE REPORT")
        print("=" * 60)

        for operation_name in sorted(self.timing_results.keys()):
            stats = self.get_stats(operation_name)
            print(f"\n{operation_name.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:  {stats['mean'] * 1000:.2f}ms")
            print(f"  Median: {stats['median'] * 1000:.2f}ms")
            print(f"  Min:   {stats['min'] * 1000:.2f}ms")
            print(f"  Max:   {stats['max'] * 1000:.2f}ms")
            print(f"  StdDev: {stats['std_dev'] * 1000:.2f}ms")
            print(f"  Total: {stats['total'] * 1000:.2f}ms")


@pytest_asyncio.fixture
async def performance_tester(tmp_path: Path) -> SearchPerformanceTester:
    """Create a performance tester with indexed data."""
    from mcp_vector_search.core.embeddings import create_embedding_function

    embedding_function, _ = create_embedding_function(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    database = ChromaVectorDatabase(
        persist_directory=tmp_path / "chroma_db",
        embedding_function=embedding_function,
        collection_name="test_performance",
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=tmp_path,
        file_extensions=[".py"],
    )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=tmp_path,
        similarity_threshold=0.2,
    )

    # Create test files with varying content
    test_files = [
        ("main.py", "def main():\n    print('Hello world')\n    calculate_sum(1, 2)"),
        (
            "utils.py",
            "def calculate_sum(a, b):\n    return a + b\n\ndef process_data(data):\n    return [x * 2 for x in data]",
        ),
        (
            "models.py",
            "class User:\n    def __init__(self, name):\n        self.name = name\n\n    def get_profile(self):\n        return {'name': self.name}",
        ),
        (
            "database.py",
            "import sqlite3\n\ndef connect_db():\n    return sqlite3.connect('app.db')\n\ndef query_users():\n    conn = connect_db()\n    return conn.execute('SELECT * FROM users').fetchall()",
        ),
        (
            "api.py",
            "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/users')\ndef get_users():\n    return query_users()",
        ),
    ]

    for filename, content in test_files:
        file_path = tmp_path / filename
        file_path.write_text(content)

    # Index the project
    async with database:
        await indexer.index_project()

    return SearchPerformanceTester(database, search_engine)


@pytest.mark.asyncio
async def test_basic_search_timing(performance_tester: SearchPerformanceTester):
    """Test basic search operation timing."""
    queries = [
        "function",
        "calculate",
        "user",
        "database",
        "hello world",
        "process data",
        "flask app",
        "sqlite connection",
    ]

    print(f"\n🔍 Testing basic search timing with {len(queries)} queries...")

    for query in queries:
        result, elapsed = await performance_tester.time_operation(
            "basic_search", performance_tester.search_engine.search, query, limit=10
        )
        print(f"  '{query}': {elapsed * 1000:.2f}ms ({len(result)} results)")

    stats = performance_tester.get_stats("basic_search")
    print("\nBasic Search Stats:")
    print(f"  Average: {stats['mean'] * 1000:.2f}ms")
    print(f"  Median:  {stats['median'] * 1000:.2f}ms")

    # Performance assertions
    assert stats["mean"] < 0.5, (
        f"Average search time too slow: {stats['mean'] * 1000:.2f}ms"
    )
    assert stats["max"] < 1.0, f"Max search time too slow: {stats['max'] * 1000:.2f}ms"


@pytest.mark.asyncio
async def test_search_with_different_limits(
    performance_tester: SearchPerformanceTester,
):
    """Test how search performance varies with different result limits."""
    query = "function"
    limits = [1, 5, 10, 20, 50]

    print("\n📊 Testing search performance with different limits...")

    for limit in limits:
        result, elapsed = await performance_tester.time_operation(
            f"search_limit_{limit}",
            performance_tester.search_engine.search,
            query,
            limit=limit,
        )
        print(f"  Limit {limit:2d}: {elapsed * 1000:.2f}ms ({len(result)} results)")

    # Check if performance scales reasonably with limit
    stats_1 = performance_tester.get_stats("search_limit_1")
    stats_50 = performance_tester.get_stats("search_limit_50")

    # Performance shouldn't degrade too much with higher limits
    ratio = stats_50["mean"] / stats_1["mean"]
    print(f"\nPerformance ratio (limit 50 vs 1): {ratio:.2f}x")
    assert ratio < 3.0, (
        f"Performance degrades too much with higher limits: {ratio:.2f}x"
    )


@pytest.mark.asyncio
async def test_search_with_different_thresholds(
    performance_tester: SearchPerformanceTester,
):
    """Test how search performance varies with different similarity thresholds."""
    query = "calculate"
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n🎯 Testing search performance with different thresholds...")

    for threshold in thresholds:
        result, elapsed = await performance_tester.time_operation(
            f"search_threshold_{threshold}",
            performance_tester.search_engine.search,
            query,
            limit=10,
            similarity_threshold=threshold,
        )
        print(
            f"  Threshold {threshold:.1f}: {elapsed * 1000:.2f}ms ({len(result)} results)"
        )

    # Lower thresholds might be slightly slower due to more results to process
    stats_low = performance_tester.get_stats("search_threshold_0.1")
    stats_high = performance_tester.get_stats("search_threshold_0.9")

    print(f"\nLow threshold: {stats_low['mean'] * 1000:.2f}ms")
    print(f"High threshold: {stats_high['mean'] * 1000:.2f}ms")


@pytest.mark.asyncio
async def test_concurrent_search_performance(
    performance_tester: SearchPerformanceTester,
):
    """Test search performance under concurrent load."""
    queries = ["function", "class", "import", "return", "def"]
    concurrent_levels = [1, 2, 5, 10]

    print("\n🚀 Testing concurrent search performance...")

    for concurrency in concurrent_levels:
        print(f"\n  Testing with {concurrency} concurrent searches...")

        async def run_search(query: str) -> tuple:
            result, elapsed = await performance_tester.time_operation(
                f"concurrent_{concurrency}",
                performance_tester.search_engine.search,
                query,
                limit=5,
            )
            return len(result), elapsed

        # Run concurrent searches
        start_time = time.perf_counter()
        tasks = [run_search(queries[i % len(queries)]) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Calculate stats
        individual_times = [r[1] for r in results]
        avg_individual = statistics.mean(individual_times)
        total_results = sum(r[0] for r in results)

        print(f"    Total time: {total_time * 1000:.2f}ms")
        print(f"    Avg individual: {avg_individual * 1000:.2f}ms")
        print(f"    Total results: {total_results}")
        print(f"    Throughput: {concurrency / total_time:.1f} searches/sec")


@pytest.mark.asyncio
async def test_search_query_complexity_performance(
    performance_tester: SearchPerformanceTester,
):
    """Test how query complexity affects search performance."""
    queries = [
        # Simple queries
        ("simple_word", "function"),
        ("simple_phrase", "hello world"),
        # Medium complexity
        ("medium_phrase", "calculate sum of numbers"),
        ("medium_technical", "database connection query"),
        # Complex queries
        ("complex_sentence", "create a flask application with user authentication"),
        (
            "complex_technical",
            "implement sqlite database connection with error handling",
        ),
    ]

    print("\n🧠 Testing search performance with different query complexities...")

    for query_type, query in queries:
        result, elapsed = await performance_tester.time_operation(
            f"complexity_{query_type}",
            performance_tester.search_engine.search,
            query,
            limit=10,
        )
        print(
            f"  {query_type:15s}: {elapsed * 1000:.2f}ms ({len(result)} results) - '{query}'"
        )

    # Analyze if complexity significantly affects performance
    simple_stats = performance_tester.get_stats("complexity_simple_word")
    complex_stats = performance_tester.get_stats("complexity_complex_technical")

    if simple_stats and complex_stats:
        ratio = complex_stats["mean"] / simple_stats["mean"]
        print(f"\nComplexity impact: {ratio:.2f}x slower for complex queries")


if __name__ == "__main__":
    # Run a quick performance test
    import tempfile

    async def main():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create the performance tester fixture manually
            database = ChromaVectorDatabase(
                persist_directory=temp_path / "chroma_db",
                collection_name="test_performance",
            )

            indexer = SemanticIndexer(
                database=database,
                project_root=temp_path,
                file_extensions=[".py"],
            )

            search_engine = SemanticSearchEngine(
                database=database,
                project_root=temp_path,
                similarity_threshold=0.2,
            )

            async with database:
                # Create test files
                test_files = [
                    (
                        "main.py",
                        "def main():\n    print('Hello world')\n    calculate_sum(1, 2)",
                    ),
                    ("utils.py", "def calculate_sum(a, b):\n    return a + b"),
                ]

                for filename, content in test_files:
                    file_path = temp_path / filename
                    file_path.write_text(content)

                # Index and test
                await indexer.index_project()

                tester = SearchPerformanceTester(database, search_engine)

                # Run a few quick tests
                queries = ["function", "calculate", "hello"]
                for query in queries:
                    result, elapsed = await tester.time_operation(
                        "quick_test", search_engine.search, query
                    )
                    print(f"'{query}': {elapsed * 1000:.2f}ms ({len(result)} results)")

                tester.print_performance_report()

    asyncio.run(main())
