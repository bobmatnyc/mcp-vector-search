#!/usr/bin/env python3
"""Simple performance monitoring script for ongoing search performance tracking."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.search import SemanticSearchEngine


class PerformanceMonitor:
    """Simple performance monitor for search operations."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.results = []

        # Initialize components
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.database = ChromaVectorDatabase(
            persist_directory=database_path,
            embedding_function=embedding_function,
            collection_name="code_search",
        )

        self.search_engine = SemanticSearchEngine(
            database=self.database,
            project_root=database_path.parent,
            similarity_threshold=0.2,
        )

    async def run_performance_check(self, queries: list = None) -> dict:
        """Run a quick performance check."""
        if queries is None:
            queries = ["function", "class", "import", "async", "return"]

        results = {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(self.database_path),
            "queries": [],
            "summary": {},
        }

        search_times = []
        total_results = 0

        async with self.database:
            for query in queries:
                start_time = time.perf_counter()
                search_results = await self.search_engine.search(
                    query, limit=10, similarity_threshold=0.1
                )
                end_time = time.perf_counter()

                search_time = (end_time - start_time) * 1000  # Convert to ms
                search_times.append(search_time)
                total_results += len(search_results)

                results["queries"].append(
                    {
                        "query": query,
                        "time_ms": round(search_time, 2),
                        "result_count": len(search_results),
                    }
                )

        # Calculate summary statistics
        if search_times:
            results["summary"] = {
                "avg_time_ms": round(sum(search_times) / len(search_times), 2),
                "min_time_ms": round(min(search_times), 2),
                "max_time_ms": round(max(search_times), 2),
                "total_queries": len(queries),
                "total_results": total_results,
                "throughput_qps": round(len(queries) / (sum(search_times) / 1000), 1),
            }

        return results

    def save_results(self, results: dict, output_file: Path = None):
        """Save results to a JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"performance_check_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")

    def print_results(self, results: dict):
        """Print results in a human-readable format."""
        print("\n" + "=" * 50)
        print("SEARCH PERFORMANCE CHECK")
        print("=" * 50)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Database: {results['database_path']}")

        print("\nSUMMARY:")
        summary = results["summary"]
        print(f"  Average time: {summary['avg_time_ms']}ms")
        print(
            f"  Min/Max time: {summary['min_time_ms']}ms / {summary['max_time_ms']}ms"
        )
        print(f"  Total queries: {summary['total_queries']}")
        print(f"  Total results: {summary['total_results']}")
        print(f"  Throughput: {summary['throughput_qps']} queries/sec")

        print("\nDETAILED RESULTS:")
        for query_result in results["queries"]:
            print(
                f"  '{query_result['query']}': {query_result['time_ms']}ms ({query_result['result_count']} results)"
            )

        # Performance assessment
        avg_time = summary["avg_time_ms"]
        print("\nPERFORMANCE ASSESSMENT:")
        if avg_time < 10:
            print("  ✅ Excellent performance (< 10ms)")
        elif avg_time < 25:
            print("  ✅ Good performance (< 25ms)")
        elif avg_time < 50:
            print("  ⚠️  Acceptable performance (< 50ms)")
        else:
            print("  ❌ Poor performance (> 50ms) - investigation needed")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor search performance")
    parser.add_argument("database_path", help="Path to the database directory")
    parser.add_argument("--queries", nargs="+", help="Custom queries to test")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only save results, don't print"
    )
    parser.add_argument(
        "--continuous", "-c", type=int, help="Run continuously every N seconds"
    )

    args = parser.parse_args()

    database_path = Path(args.database_path)
    if not database_path.exists():
        print(f"Error: Database path does not exist: {database_path}")
        return 1

    monitor = PerformanceMonitor(database_path)

    if args.continuous:
        print(f"Running continuous monitoring every {args.continuous} seconds...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                results = await monitor.run_performance_check(args.queries)

                if not args.quiet:
                    monitor.print_results(results)

                if args.output:
                    # Append timestamp to output file for continuous mode
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = Path(f"{args.output}_{timestamp}.json")
                    monitor.save_results(results, output_file)

                await asyncio.sleep(args.continuous)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        # Single run
        results = await monitor.run_performance_check(args.queries)

        if not args.quiet:
            monitor.print_results(results)

        if args.output:
            monitor.save_results(results, Path(args.output))

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
