#!/usr/bin/env python3
"""Lightweight search performance monitoring for ongoing quality assurance."""

import asyncio
import json
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
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


@dataclass
class QuickMetrics:
    """Quick performance metrics."""

    avg_response_time_ms: float
    throughput_qps: float
    avg_results_per_query: float
    avg_similarity_score: float
    success_rate: float
    timestamp: str


class SearchPerformanceMonitor:
    """Lightweight performance monitor for search functionality."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.search_engine = None

        # Standard test queries for consistent monitoring
        self.test_queries = [
            "function",
            "class",
            "async",
            "error handling",
            "database connection",
            "test",
            "configuration",
            "import",
        ]

    async def setup(self) -> None:
        """Quick setup for monitoring."""
        print("🔧 Setting up performance monitor...")

        # Initialize project
        project_manager = ProjectManager(self.project_root)

        if not project_manager.is_initialized():
            print("❌ Project not initialized. Run 'mcp-vector-search init' first.")
            return False

        config = project_manager.load_config()

        # Create components
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        self.search_engine = SemanticSearchEngine(
            database=database,
            project_root=self.project_root,
            similarity_threshold=config.similarity_threshold,
        )

        # Initialize database
        await database.initialize()
        print("✓ Monitor ready")
        return True

    async def run_quick_check(self) -> QuickMetrics:
        """Run a quick performance check."""
        print("⚡ Running quick performance check...")

        start_time = time.time()
        response_times = []
        result_counts = []
        similarity_scores = []
        successful_queries = 0

        for query in self.test_queries:
            try:
                query_start = time.perf_counter()

                results = await self.search_engine.search(
                    query=query,
                    limit=10,
                    similarity_threshold=0.1,
                )

                query_end = time.perf_counter()
                response_time = (query_end - query_start) * 1000

                response_times.append(response_time)
                result_counts.append(len(results))

                if results:
                    avg_sim = sum(r.similarity_score for r in results) / len(results)
                    similarity_scores.append(avg_sim)

                successful_queries += 1

            except Exception as e:
                print(f"  ❌ Query '{query}' failed: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        throughput = len(self.test_queries) / total_time if total_time > 0 else 0.0
        avg_results = statistics.mean(result_counts) if result_counts else 0.0
        avg_similarity = (
            statistics.mean(similarity_scores) if similarity_scores else 0.0
        )
        success_rate = successful_queries / len(self.test_queries)

        metrics = QuickMetrics(
            avg_response_time_ms=avg_response_time,
            throughput_qps=throughput,
            avg_results_per_query=avg_results,
            avg_similarity_score=avg_similarity,
            success_rate=success_rate,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        return metrics

    def print_metrics(self, metrics: QuickMetrics) -> None:
        """Print metrics in a readable format."""
        print("\n" + "=" * 50)
        print("📊 SEARCH PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Success Rate: {metrics.success_rate:.1%}")
        print(f"Avg Response Time: {metrics.avg_response_time_ms:.2f}ms")
        print(f"Throughput: {metrics.throughput_qps:.1f} queries/sec")
        print(f"Avg Results/Query: {metrics.avg_results_per_query:.1f}")
        print(f"Avg Similarity: {metrics.avg_similarity_score:.3f}")

        # Performance assessment
        print("\n📈 Assessment:")

        if metrics.success_rate < 0.9:
            print("  ❌ Low success rate - check for errors")
        else:
            print("  ✅ Good success rate")

        if metrics.avg_response_time_ms < 20:
            print("  ✅ Excellent response time")
        elif metrics.avg_response_time_ms < 50:
            print("  ✅ Good response time")
        else:
            print("  ⚠️  Slow response time")

        if metrics.throughput_qps > 50:
            print("  ✅ Excellent throughput")
        elif metrics.throughput_qps > 20:
            print("  ✅ Good throughput")
        else:
            print("  ⚠️  Low throughput")

        if metrics.avg_similarity_score > 0.5:
            print("  ✅ High quality results")
        elif metrics.avg_similarity_score > 0.3:
            print("  ✅ Good quality results")
        else:
            print("  ⚠️  Low quality results")

        print("=" * 50)

    def save_metrics(self, metrics: QuickMetrics, output_file: Path = None) -> None:
        """Save metrics to file for tracking over time."""
        if output_file is None:
            output_file = (
                self.project_root / ".mcp-vector-search" / "performance_metrics.jsonl"
            )

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        metrics_dict = {
            "timestamp": metrics.timestamp,
            "avg_response_time_ms": metrics.avg_response_time_ms,
            "throughput_qps": metrics.throughput_qps,
            "avg_results_per_query": metrics.avg_results_per_query,
            "avg_similarity_score": metrics.avg_similarity_score,
            "success_rate": metrics.success_rate,
        }

        # Append to JSONL file
        with open(output_file, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

        print(f"📁 Metrics saved to {output_file}")

    async def run_stress_test(self, duration_seconds: int = 30) -> dict[str, Any]:
        """Run a stress test for the specified duration."""
        print(f"🔥 Running stress test for {duration_seconds} seconds...")

        start_time = time.time()
        end_time = start_time + duration_seconds

        query_count = 0
        response_times = []
        errors = 0

        while time.time() < end_time:
            query = self.test_queries[query_count % len(self.test_queries)]

            try:
                query_start = time.perf_counter()

                await self.search_engine.search(
                    query=query,
                    limit=5,
                    similarity_threshold=0.2,
                )

                query_end = time.perf_counter()
                response_times.append((query_end - query_start) * 1000)

            except Exception:
                errors += 1

            query_count += 1

        actual_duration = time.time() - start_time

        # Calculate stress test metrics
        stress_metrics = {
            "duration_seconds": actual_duration,
            "total_queries": query_count,
            "errors": errors,
            "success_rate": (
                (query_count - errors) / query_count if query_count > 0 else 0
            ),
            "avg_response_time_ms": (
                statistics.mean(response_times) if response_times else 0
            ),
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "throughput_qps": query_count / actual_duration,
        }

        print("\n🔥 Stress Test Results:")
        print(f"  Duration: {stress_metrics['duration_seconds']:.1f}s")
        print(f"  Total Queries: {stress_metrics['total_queries']}")
        print(f"  Errors: {stress_metrics['errors']}")
        print(f"  Success Rate: {stress_metrics['success_rate']:.1%}")
        print(f"  Throughput: {stress_metrics['throughput_qps']:.1f} q/s")
        print(f"  Avg Response: {stress_metrics['avg_response_time_ms']:.2f}ms")
        print(f"  Max Response: {stress_metrics['max_response_time_ms']:.2f}ms")
        print(f"  Min Response: {stress_metrics['min_response_time_ms']:.2f}ms")

        return stress_metrics

    async def check_search_quality(self) -> dict[str, float]:
        """Quick search quality check."""
        print("🎯 Checking search quality...")

        quality_queries = [
            ("function definition", ["def", "function"]),
            ("class inheritance", ["class", "inherit"]),
            ("error handling", ["error", "exception", "try"]),
            ("async await", ["async", "await"]),
        ]

        quality_scores = []

        for query, expected_keywords in quality_queries:
            try:
                results = await self.search_engine.search(
                    query=query,
                    limit=10,
                    similarity_threshold=0.1,
                )

                if results:
                    # Check keyword coverage
                    all_content = " ".join(r.content.lower() for r in results)
                    keyword_matches = sum(
                        1 for kw in expected_keywords if kw in all_content
                    )
                    keyword_coverage = keyword_matches / len(expected_keywords)

                    # Check similarity scores
                    avg_similarity = sum(r.similarity_score for r in results) / len(
                        results
                    )

                    # Combined quality score
                    quality_score = keyword_coverage * 0.6 + avg_similarity * 0.4
                    quality_scores.append(quality_score)

                    print(
                        f"  '{query}': {quality_score:.3f} (coverage: {keyword_coverage:.3f}, similarity: {avg_similarity:.3f})"
                    )
                else:
                    print(f"  '{query}': 0.000 (no results)")
                    quality_scores.append(0.0)

            except Exception as e:
                print(f"  '{query}': Error - {e}")
                quality_scores.append(0.0)

        overall_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        print(f"\n🎯 Overall Quality Score: {overall_quality:.3f}")

        return {
            "overall_quality": overall_quality,
            "individual_scores": quality_scores,
        }


async def main():
    """Main monitoring execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Search Performance Monitor")
    parser.add_argument("--stress", type=int, help="Run stress test for N seconds")
    parser.add_argument("--quality", action="store_true", help="Run quality check")
    parser.add_argument("--save", action="store_true", help="Save metrics to file")
    args = parser.parse_args()

    print("🔍 MCP Vector Search - Performance Monitor")
    print("=" * 50)

    project_root = Path.cwd()
    monitor = SearchPerformanceMonitor(project_root)

    try:
        # Setup
        success = await monitor.setup()
        if not success:
            return

        # Run quick check
        metrics = await monitor.run_quick_check()
        monitor.print_metrics(metrics)

        if args.save:
            monitor.save_metrics(metrics)

        # Run additional tests if requested
        if args.stress:
            await monitor.run_stress_test(args.stress)

        if args.quality:
            await monitor.check_search_quality()

        print("\n✅ Monitoring completed!")

    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
