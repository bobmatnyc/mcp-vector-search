#!/usr/bin/env python3
"""Comprehensive search functionality testing and performance analysis."""

import asyncio
import statistics

# Add src to path for imports
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


@dataclass
class SearchTestResult:
    """Results from a search test."""

    query: str
    execution_time_ms: float
    result_count: int
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    results: list[SearchResult]
    success: bool
    error: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for search operations."""

    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_execution_time_ms: float
    median_execution_time_ms: float
    p95_execution_time_ms: float
    avg_results_per_query: float
    avg_similarity_score: float
    throughput_queries_per_second: float


class ComprehensiveSearchTester:
    """Comprehensive search functionality tester."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.search_engine = None
        self.test_results: list[SearchTestResult] = []

    async def setup_test_environment(self) -> None:
        """Set up the test environment with real data."""
        print("🔧 Setting up test environment...")

        # Initialize project
        project_manager = ProjectManager(self.project_root)

        # Check if already initialized
        if not project_manager.is_initialized():
            config = project_manager.initialize(
                file_extensions=[".py", ".js", ".ts", ".md"],
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                similarity_threshold=0.2,
                force=True,
            )
        else:
            config = project_manager.load_config()

        print(f"  ✓ Project initialized at {self.project_root}")

        # Create embedding function
        embedding_function, _ = create_embedding_function(config.embedding_model)
        print(f"  ✓ Embedding function created: {config.embedding_model}")

        # Create database
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        # Create indexer
        indexer = SemanticIndexer(
            database=database,
            project_root=self.project_root,
            file_extensions=config.file_extensions,
        )

        # Create search engine
        self.search_engine = SemanticSearchEngine(
            database=database,
            project_root=self.project_root,
            similarity_threshold=config.similarity_threshold,
        )

        # Initialize database and index project
        await database.initialize()
        print("  🔍 Indexing project...")
        indexed_count = await indexer.index_project()
        print(f"  ✓ Indexed {indexed_count} files")

        # Get stats
        stats = await database.get_stats()
        print(
            f"  📊 Database stats: {stats.total_chunks} chunks, {stats.total_files} files"
        )

        # Keep database open for search operations
        # Note: Database will be closed when the search engine is done

    async def run_basic_search_tests(self) -> None:
        """Run basic search functionality tests."""
        print("\n🔍 Running basic search tests...")

        # Define test queries with expected characteristics
        test_queries = [
            # Simple keyword searches
            ("function", "Should find function definitions"),
            ("class", "Should find class definitions"),
            ("import", "Should find import statements"),
            ("async", "Should find async functions"),
            # Semantic searches
            ("user authentication", "Should find auth-related code"),
            ("database connection", "Should find DB connection code"),
            ("error handling", "Should find error handling patterns"),
            ("configuration settings", "Should find config-related code"),
            # Code pattern searches
            ("try except", "Should find exception handling"),
            ("for loop", "Should find iteration patterns"),
            ("if condition", "Should find conditional logic"),
            ("return value", "Should find return statements"),
            # Complex semantic searches
            ("parse command line arguments", "Should find CLI parsing"),
            ("validate user input", "Should find validation logic"),
            ("serialize data to json", "Should find JSON serialization"),
            ("handle file operations", "Should find file I/O"),
        ]

        for query, description in test_queries:
            await self._test_single_query(query, description)

    async def run_performance_tests(self) -> None:
        """Run performance-focused tests."""
        print("\n⚡ Running performance tests...")

        # Test different query lengths
        await self._test_query_length_performance()

        # Test different result limits
        await self._test_result_limit_performance()

        # Test different similarity thresholds
        await self._test_similarity_threshold_performance()

        # Test concurrent searches
        await self._test_concurrent_search_performance()

    async def run_edge_case_tests(self) -> None:
        """Run edge case and error handling tests."""
        print("\n🧪 Running edge case tests...")

        edge_cases = [
            ("", "Empty query"),
            ("   ", "Whitespace only query"),
            ("a", "Single character query"),
            ("x" * 1000, "Very long query"),
            ("🚀🔍💻", "Unicode/emoji query"),
            ("SELECT * FROM users", "SQL injection attempt"),
            ("../../../etc/passwd", "Path traversal attempt"),
            ("nonexistentfunctionname12345", "Non-existent code pattern"),
        ]

        for query, description in edge_cases:
            await self._test_single_query(query, description, expect_results=False)

    async def run_filter_tests(self) -> None:
        """Test search filtering functionality."""
        print("\n🔧 Running filter tests...")

        filter_tests = [
            ({"language": "python"}, "Python files only"),
            ({"chunk_type": "function"}, "Functions only"),
            ({"chunk_type": "class"}, "Classes only"),
            ({"file_path": "*.py"}, "Python files by extension"),
        ]

        base_query = "function"

        for filters, description in filter_tests:
            await self._test_filtered_query(base_query, filters, description)

    async def run_similarity_analysis(self) -> None:
        """Analyze similarity score distributions."""
        print("\n📊 Running similarity analysis...")

        # Test queries with different expected similarity patterns
        similarity_queries = [
            ("exact function name", "High similarity expected"),
            ("similar concept different words", "Medium similarity expected"),
            ("completely unrelated topic", "Low similarity expected"),
        ]

        for query, description in similarity_queries:
            result = await self._test_single_query(
                query, description, analyze_similarity=True
            )
            if result.success and result.results:
                self._analyze_similarity_distribution(result)

    async def _test_single_query(
        self,
        query: str,
        description: str,
        expect_results: bool = True,
        analyze_similarity: bool = False,
    ) -> SearchTestResult:
        """Test a single search query."""
        start_time = time.perf_counter()

        try:
            results = await self.search_engine.search(
                query=query,
                limit=20,
                similarity_threshold=0.1,  # Low threshold to get more results
            )

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            # Calculate similarity statistics
            similarities = [r.similarity_score for r in results] if results else [0.0]
            avg_similarity = statistics.mean(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)

            result = SearchTestResult(
                query=query,
                execution_time_ms=execution_time,
                result_count=len(results),
                avg_similarity=avg_similarity,
                max_similarity=max_similarity,
                min_similarity=min_similarity,
                results=results,
                success=True,
            )

            # Log result
            status = "✓" if (results or not expect_results) else "⚠"
            print(f"  {status} '{query}' ({description})")
            print(
                f"    Time: {execution_time:.2f}ms, Results: {len(results)}, Avg Similarity: {avg_similarity:.3f}"
            )

            if analyze_similarity and results:
                print(
                    f"    Similarity range: {min_similarity:.3f} - {max_similarity:.3f}"
                )

        except Exception as e:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            result = SearchTestResult(
                query=query,
                execution_time_ms=execution_time,
                result_count=0,
                avg_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                results=[],
                success=False,
                error=str(e),
            )

            print(f"  ❌ '{query}' ({description}) - Error: {e}")

        self.test_results.append(result)
        return result

    async def _test_filtered_query(
        self, query: str, filters: dict[str, Any], description: str
    ) -> None:
        """Test a query with filters."""
        start_time = time.perf_counter()

        try:
            results = await self.search_engine.search(
                query=query,
                limit=10,
                filters=filters,
                similarity_threshold=0.1,
            )

            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000

            print(f"  ✓ Filter test: {description}")
            print(f"    Query: '{query}', Filters: {filters}")
            print(f"    Time: {execution_time:.2f}ms, Results: {len(results)}")

            # Verify filter was applied correctly
            if filters.get("language") and results:
                languages = {r.language for r in results}
                print(f"    Languages found: {languages}")

        except Exception as e:
            print(f"  ❌ Filter test failed: {description} - Error: {e}")

    async def _test_query_length_performance(self) -> None:
        """Test performance with different query lengths."""
        print("  📏 Testing query length performance...")

        base_query = "function that handles user authentication and validation"
        lengths = [1, 2, 5, 10, 20, 50]

        for length in lengths:
            words = base_query.split()[:length]
            if not words:
                continue

            query = " ".join(words)
            await self._test_single_query(
                query, f"Query length {length} words", expect_results=False
            )

    async def _test_result_limit_performance(self) -> None:
        """Test performance with different result limits."""
        print("  📊 Testing result limit performance...")

        query = "function"
        limits = [1, 5, 10, 20, 50, 100]

        for limit in limits:
            start_time = time.perf_counter()

            try:
                results = await self.search_engine.search(
                    query=query,
                    limit=limit,
                    similarity_threshold=0.1,
                )

                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000

                print(
                    f"    Limit {limit}: {execution_time:.2f}ms, {len(results)} results"
                )

            except Exception as e:
                print(f"    Limit {limit}: Error - {e}")

    async def _test_similarity_threshold_performance(self) -> None:
        """Test performance with different similarity thresholds."""
        print("  🎯 Testing similarity threshold performance...")

        query = "function"
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            start_time = time.perf_counter()

            try:
                results = await self.search_engine.search(
                    query=query,
                    limit=20,
                    similarity_threshold=threshold,
                )

                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000

                avg_sim = (
                    statistics.mean([r.similarity_score for r in results])
                    if results
                    else 0.0
                )
                print(
                    f"    Threshold {threshold}: {execution_time:.2f}ms, {len(results)} results, avg sim {avg_sim:.3f}"
                )

            except Exception as e:
                print(f"    Threshold {threshold}: Error - {e}")

    async def _test_concurrent_search_performance(self) -> None:
        """Test concurrent search performance."""
        print("  🚀 Testing concurrent search performance...")

        queries = [
            "function definition",
            "class implementation",
            "error handling",
            "database connection",
            "user authentication",
        ]

        # Test different concurrency levels
        for concurrency in [1, 2, 5, 10]:
            start_time = time.perf_counter()

            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                query = queries[i % len(queries)]
                task = self.search_engine.search(
                    query, limit=10, similarity_threshold=0.1
                )
                tasks.append(task)

            try:
                results = await asyncio.gather(*tasks)
                end_time = time.perf_counter()

                total_time = (end_time - start_time) * 1000
                total_results = sum(len(r) for r in results)

                print(
                    f"    Concurrency {concurrency}: {total_time:.2f}ms total, {total_results} results"
                )
                print(
                    f"      Throughput: {concurrency / (total_time / 1000):.1f} queries/sec"
                )

            except Exception as e:
                print(f"    Concurrency {concurrency}: Error - {e}")

    def _analyze_similarity_distribution(self, result: SearchTestResult) -> None:
        """Analyze similarity score distribution."""
        similarities = [r.similarity_score for r in result.results]

        if len(similarities) > 1:
            stdev = statistics.stdev(similarities)
            print(f"    Similarity distribution: std dev {stdev:.3f}")

            # Check for clustering
            high_sim = sum(1 for s in similarities if s > 0.8)
            med_sim = sum(1 for s in similarities if 0.5 <= s <= 0.8)
            low_sim = sum(1 for s in similarities if s < 0.5)

            print(
                f"    Score distribution: High({high_sim}) Med({med_sim}) Low({low_sim})"
            )

    def generate_performance_report(self) -> PerformanceMetrics:
        """Generate comprehensive performance report."""
        if not self.test_results:
            return None

        successful_results = [r for r in self.test_results if r.success]
        failed_results = [r for r in self.test_results if not r.success]

        if not successful_results:
            print("❌ No successful test results to analyze")
            return None

        execution_times = [r.execution_time_ms for r in successful_results]
        result_counts = [r.result_count for r in successful_results]
        similarity_scores = []

        for result in successful_results:
            if result.results:
                similarity_scores.extend([r.similarity_score for r in result.results])

        metrics = PerformanceMetrics(
            total_tests=len(self.test_results),
            successful_tests=len(successful_results),
            failed_tests=len(failed_results),
            avg_execution_time_ms=statistics.mean(execution_times),
            median_execution_time_ms=statistics.median(execution_times),
            p95_execution_time_ms=(
                statistics.quantiles(execution_times, n=20)[18]
                if len(execution_times) >= 20
                else max(execution_times)
            ),
            avg_results_per_query=statistics.mean(result_counts),
            avg_similarity_score=(
                statistics.mean(similarity_scores) if similarity_scores else 0.0
            ),
            throughput_queries_per_second=1000 / statistics.mean(execution_times),
        )

        return metrics

    def print_performance_report(self, metrics: PerformanceMetrics) -> None:
        """Print detailed performance report."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE SEARCH PERFORMANCE REPORT")
        print("=" * 60)

        print("\n🧪 Test Summary:")
        print(f"  Total tests: {metrics.total_tests}")
        print(
            f"  Successful: {metrics.successful_tests} ({metrics.successful_tests / metrics.total_tests * 100:.1f}%)"
        )
        print(
            f"  Failed: {metrics.failed_tests} ({metrics.failed_tests / metrics.total_tests * 100:.1f}%)"
        )

        print("\n⚡ Performance Metrics:")
        print(f"  Average execution time: {metrics.avg_execution_time_ms:.2f}ms")
        print(f"  Median execution time: {metrics.median_execution_time_ms:.2f}ms")
        print(f"  95th percentile: {metrics.p95_execution_time_ms:.2f}ms")
        print(
            f"  Throughput: {metrics.throughput_queries_per_second:.1f} queries/second"
        )

        print("\n🎯 Result Quality:")
        print(f"  Average results per query: {metrics.avg_results_per_query:.1f}")
        print(f"  Average similarity score: {metrics.avg_similarity_score:.3f}")

        # Performance assessment
        print("\n📈 Performance Assessment:")
        if metrics.avg_execution_time_ms < 50:
            print("  ✅ Excellent response time (< 50ms)")
        elif metrics.avg_execution_time_ms < 100:
            print("  ✅ Good response time (< 100ms)")
        elif metrics.avg_execution_time_ms < 200:
            print("  ⚠️  Acceptable response time (< 200ms)")
        else:
            print("  ❌ Slow response time (> 200ms)")

        if metrics.throughput_queries_per_second > 50:
            print("  ✅ Excellent throughput (> 50 q/s)")
        elif metrics.throughput_queries_per_second > 20:
            print("  ✅ Good throughput (> 20 q/s)")
        elif metrics.throughput_queries_per_second > 10:
            print("  ⚠️  Acceptable throughput (> 10 q/s)")
        else:
            print("  ❌ Low throughput (< 10 q/s)")

        if metrics.avg_similarity_score > 0.7:
            print("  ✅ High quality results (avg similarity > 0.7)")
        elif metrics.avg_similarity_score > 0.5:
            print("  ✅ Good quality results (avg similarity > 0.5)")
        elif metrics.avg_similarity_score > 0.3:
            print("  ⚠️  Acceptable quality results (avg similarity > 0.3)")
        else:
            print("  ❌ Low quality results (avg similarity < 0.3)")

        # Recommendations
        print("\n💡 Recommendations:")
        if metrics.avg_execution_time_ms > 100:
            print("  • Consider implementing connection pooling for better performance")
            print("  • Optimize embedding model or use caching")

        if metrics.avg_results_per_query < 3:
            print("  • Consider lowering similarity threshold for more results")
            print("  • Improve query preprocessing and expansion")

        if metrics.avg_similarity_score < 0.5:
            print("  • Review embedding model quality")
            print("  • Improve code chunking strategy")
            print("  • Enhance query preprocessing")

        print("\n" + "=" * 60)


async def main():
    """Main test execution."""
    print("🔍 MCP Vector Search - Comprehensive Search Testing")
    print("=" * 60)

    # Use current project as test subject
    project_root = Path.cwd()

    tester = ComprehensiveSearchTester(project_root)

    try:
        # Setup test environment
        await tester.setup_test_environment()

        # Run all test suites
        await tester.run_basic_search_tests()
        await tester.run_performance_tests()
        await tester.run_edge_case_tests()
        await tester.run_filter_tests()
        await tester.run_similarity_analysis()

        # Generate and print report
        metrics = tester.generate_performance_report()
        if metrics:
            tester.print_performance_report(metrics)

        print("\n🎉 Comprehensive search testing completed!")

    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
