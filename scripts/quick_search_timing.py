#!/usr/bin/env python3
"""Quick search timing test to identify performance bottlenecks."""

import asyncio
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


async def time_operation(name: str, operation):
    """Time an async operation."""
    start = time.perf_counter()
    result = await operation
    end = time.perf_counter()
    duration = (end - start) * 1000  # Convert to ms
    print(f"  {name}: {duration:.2f}ms")
    return result, duration


async def main():
    """Run quick timing tests."""
    print("🚀 Quick Search Timing Test")
    print("=" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create test files
        test_files = {
            "main.py": """
def main():
    print("Hello world")
    result = calculate_sum(1, 2)
    return result

def process_data(data):
    return [x * 2 for x in data]
""",
            "utils.py": """
def calculate_sum(a, b):
    return a + b

def multiply(x, y):
    return x * y

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
""",
            "models.py": """
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.active = True

    def get_profile(self):
        return {
            'name': self.name,
            'email': self.email,
            'active': self.active
        }

class UserManager:
    def __init__(self):
        self.users = {}

    def create_user(self, name, email):
        user = User(name, email)
        self.users[user.name] = user
        return user
""",
        }

        print("📁 Creating test files...")
        for filename, content in test_files.items():
            (project_dir / filename).write_text(content)

        # Initialize components
        print("\n🔧 Initializing components...")

        start_time = time.perf_counter()
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_time = (time.perf_counter() - start_time) * 1000
        print(f"  Embedding function creation: {embedding_time:.2f}ms")

        database = ChromaVectorDatabase(
            persist_directory=project_dir / "chroma_db",
            embedding_function=embedding_function,
            collection_name="timing_test",
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
            # Test indexing performance
            print("\n📚 Testing indexing performance...")
            _, indexing_time = await time_operation(
                "Full project indexing", indexer.index_project()
            )

            stats = await indexer.get_indexing_stats()
            print(f"  Indexed {stats['total_chunks']} chunks")
            print(
                f"  Indexing rate: {stats['total_chunks'] / (indexing_time / 1000):.1f} chunks/sec"
            )

            # Test search performance with different queries
            print("\n🔍 Testing search performance...")

            test_queries = [
                ("simple", "function"),
                ("exact_match", "calculate_sum"),
                ("class_search", "User"),
                ("method_search", "get_profile"),
                ("compound", "user management"),
                ("long_query", "create user with email and profile"),
            ]

            search_times = []

            for query_type, query in test_queries:
                result, search_time = await time_operation(
                    f"Search '{query}' ({query_type})",
                    search_engine.search(query, limit=10, similarity_threshold=0.05),
                )
                search_times.append(search_time)
                print(f"    → {len(result)} results")

            # Performance analysis
            print("\n📊 Performance Analysis:")
            avg_search_time = sum(search_times) / len(search_times)
            print(f"  Average search time: {avg_search_time:.2f}ms")
            print(f"  Fastest search: {min(search_times):.2f}ms")
            print(f"  Slowest search: {max(search_times):.2f}ms")
            print(f"  Search throughput: {1000 / avg_search_time:.1f} searches/sec")

            # Test concurrent searches
            print("\n🚀 Testing concurrent search performance...")

            async def concurrent_search(query):
                start = time.perf_counter()
                result = await search_engine.search(
                    query, limit=5, similarity_threshold=0.05
                )
                end = time.perf_counter()
                return len(result), (end - start) * 1000

            # Run 5 concurrent searches
            concurrent_queries = ["function", "class", "user", "calculate", "profile"]

            start_concurrent = time.perf_counter()
            concurrent_results = await asyncio.gather(
                *[concurrent_search(query) for query in concurrent_queries]
            )
            end_concurrent = time.perf_counter()

            total_concurrent_time = (end_concurrent - start_concurrent) * 1000
            individual_times = [result[1] for result in concurrent_results]
            total_results = sum(result[0] for result in concurrent_results)

            print("  5 concurrent searches:")
            print(f"    Total wall time: {total_concurrent_time:.2f}ms")
            print(
                f"    Average individual time: {sum(individual_times) / len(individual_times):.2f}ms"
            )
            print(f"    Total results: {total_results}")
            print(
                f"    Concurrent throughput: {len(concurrent_queries) / (total_concurrent_time / 1000):.1f} searches/sec"
            )

            # Test different result limits
            print("\n📏 Testing performance with different result limits...")

            limits = [1, 5, 10, 20, 50]
            for limit in limits:
                result, limit_time = await time_operation(
                    f"Search with limit {limit}",
                    search_engine.search(
                        "function", limit=limit, similarity_threshold=0.05
                    ),
                )
                print(f"    → {len(result)} results")

            # Test different similarity thresholds
            print("\n🎯 Testing performance with different similarity thresholds...")

            thresholds = [0.01, 0.05, 0.1, 0.3, 0.5]
            for threshold in thresholds:
                result, threshold_time = await time_operation(
                    f"Search with threshold {threshold}",
                    search_engine.search(
                        "function", limit=10, similarity_threshold=threshold
                    ),
                )
                print(f"    → {len(result)} results")

            print("\n✅ Timing tests completed!")

            # Performance recommendations
            print("\n💡 Performance Insights:")
            if avg_search_time < 10:
                print("  ✅ Search performance is excellent (< 10ms)")
            elif avg_search_time < 50:
                print("  ✅ Search performance is good (< 50ms)")
            elif avg_search_time < 100:
                print("  ⚠️  Search performance is acceptable (< 100ms)")
            else:
                print("  ❌ Search performance needs improvement (> 100ms)")

            if indexing_time > 5000:  # 5 seconds
                print(
                    "  ⚠️  Indexing is slow - consider optimizing chunk size or batch processing"
                )
            else:
                print("  ✅ Indexing performance is good")

            print("\n📈 Key Metrics:")
            print(
                f"  - Indexing: {indexing_time:.0f}ms for {stats['total_chunks']} chunks"
            )
            print(f"  - Search: {avg_search_time:.1f}ms average")
            print(f"  - Throughput: {1000 / avg_search_time:.1f} searches/sec")


if __name__ == "__main__":
    asyncio.run(main())
