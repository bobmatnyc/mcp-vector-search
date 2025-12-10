#!/usr/bin/env python3
"""Test script to compare performance between pooled and non-pooled database connections."""

import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import (
    ChromaVectorDatabase,
    PooledChromaVectorDatabase,
)
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.search import SemanticSearchEngine


async def time_operation(name: str, operation):
    """Time an async operation."""
    start = time.perf_counter()
    result = await operation
    end = time.perf_counter()
    duration = (end - start) * 1000  # Convert to ms
    return result, duration


async def setup_test_data(project_dir: Path, database_class, embedding_function):
    """Set up test data for performance comparison."""
    # Create test files
    test_files = {
        "main.py": """
def main():
    print("Hello world")
    result = calculate_sum(1, 2)
    process_data([1, 2, 3, 4, 5])
    return result

def process_data(data):
    return [x * 2 for x in data]

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""",
        "utils.py": """
def calculate_sum(a, b):
    return a + b

def multiply(x, y):
    return x * y

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        return self.history.copy()
""",
        "models.py": """
from typing import List, Optional

class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self.active = True

    def get_profile(self):
        return {
            'name': self.name,
            'email': self.email,
            'active': self.active
        }

    def deactivate(self):
        self.active = False

class UserManager:
    def __init__(self):
        self.users = {}

    def create_user(self, name: str, email: str) -> User:
        user = User(name, email)
        self.users[user.name] = user
        return user

    def get_user(self, name: str) -> Optional[User]:
        return self.users.get(name)

    def list_active_users(self) -> List[User]:
        return [user for user in self.users.values() if user.active]
""",
        "api.py": """
from flask import Flask, jsonify, request
from models import UserManager

app = Flask(__name__)
user_manager = UserManager()

@app.route('/users', methods=['GET'])
def get_users():
    users = user_manager.list_active_users()
    return jsonify([user.get_profile() for user in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user = user_manager.create_user(data['name'], data['email'])
    return jsonify(user.get_profile()), 201

@app.route('/users/<name>', methods=['GET'])
def get_user(name):
    user = user_manager.get_user(name)
    if user:
        return jsonify(user.get_profile())
    return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
""",
    }

    # Ensure project directory exists
    project_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in test_files.items():
        (project_dir / filename).write_text(content)

    # Initialize database and indexer
    database = database_class(
        persist_directory=project_dir / "chroma_db",
        embedding_function=embedding_function,
        collection_name="performance_test",
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

    # Index the project
    async with database:
        await indexer.index_project()

    return database, search_engine


async def test_search_performance(
    database, search_engine, test_name: str, num_searches: int = 20
):
    """Test search performance with multiple queries."""
    queries = [
        "function",
        "class",
        "user",
        "calculate",
        "process",
        "import",
        "return",
        "def main",
        "flask app",
        "get profile",
    ]

    search_times = []
    total_results = 0

    print(f"\n🔍 Testing {test_name} ({num_searches} searches)...")

    # Warm up with a single search
    async with database:
        await search_engine.search("warmup", limit=5, similarity_threshold=0.05)

    # Time individual searches - keep connection alive for pooled database
    async with database:
        for i in range(num_searches):
            query = queries[i % len(queries)]

            result, search_time = await time_operation(
                f"Search {i + 1}",
                search_engine.search(query, limit=10, similarity_threshold=0.05),
            )

            search_times.append(search_time)
            total_results += len(result)

            if i < 5:  # Show first 5 searches
                print(
                    f"  Search {i + 1} ('{query}'): {search_time:.2f}ms ({len(result)} results)"
                )

    # Calculate statistics
    avg_time = statistics.mean(search_times)
    median_time = statistics.median(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    std_dev = statistics.stdev(search_times) if len(search_times) > 1 else 0

    print(f"\n📊 {test_name} Results:")
    print(f"  Total searches: {num_searches}")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Median time: {median_time:.2f}ms")
    print(f"  Min/Max time: {min_time:.2f}ms / {max_time:.2f}ms")
    print(f"  Std deviation: {std_dev:.2f}ms")
    print(f"  Total results: {total_results}")
    print(f"  Throughput: {1000 / avg_time:.1f} searches/sec")

    return {
        "test_name": test_name,
        "num_searches": num_searches,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_dev": std_dev,
        "total_results": total_results,
        "throughput": 1000 / avg_time,
    }


async def test_concurrent_performance(
    database, search_engine, test_name: str, concurrency: int = 5
):
    """Test concurrent search performance."""
    queries = ["function", "class", "user", "calculate", "process"]

    print(
        f"\n🚀 Testing {test_name} concurrent performance ({concurrency} concurrent searches)..."
    )

    async def single_search(query):
        start = time.perf_counter()
        result = await search_engine.search(query, limit=5, similarity_threshold=0.05)
        end = time.perf_counter()
        return len(result), (end - start) * 1000

    # Run concurrent searches with database connection
    async with database:
        start_wall = time.perf_counter()
        tasks = [single_search(queries[i % len(queries)]) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        wall_time = (time.perf_counter() - start_wall) * 1000

    individual_times = [r[1] for r in results]
    total_results = sum(r[0] for r in results)
    avg_individual = statistics.mean(individual_times)

    print(f"  Wall time: {wall_time:.2f}ms")
    print(f"  Avg individual time: {avg_individual:.2f}ms")
    print(f"  Total results: {total_results}")
    print(
        f"  Concurrent throughput: {concurrency / (wall_time / 1000):.1f} searches/sec"
    )
    print(f"  Efficiency: {(avg_individual * concurrency / wall_time) * 100:.1f}%")

    return {
        "test_name": f"{test_name}_concurrent",
        "concurrency": concurrency,
        "wall_time": wall_time,
        "avg_individual_time": avg_individual,
        "total_results": total_results,
        "concurrent_throughput": concurrency / (wall_time / 1000),
        "efficiency": (avg_individual * concurrency / wall_time) * 100,
    }


async def main():
    """Main comparison function."""
    print("🔬 Connection Pooling Performance Comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create embedding function
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Test regular database
        print("\n📁 Setting up regular database...")
        regular_db, regular_search = await setup_test_data(
            project_dir / "regular", ChromaVectorDatabase, embedding_function
        )

        # Test pooled database
        print("\n📁 Setting up pooled database...")
        pooled_db, pooled_search = await setup_test_data(
            project_dir / "pooled", PooledChromaVectorDatabase, embedding_function
        )

        # Performance tests
        results = []

        # Test regular database performance
        regular_results = await test_search_performance(
            regular_db, regular_search, "Regular Database", 20
        )
        results.append(regular_results)

        # Test pooled database performance
        pooled_results = await test_search_performance(
            pooled_db, pooled_search, "Pooled Database", 20
        )
        results.append(pooled_results)

        # Test concurrent performance
        regular_concurrent = await test_concurrent_performance(
            regular_db, regular_search, "Regular Database", 5
        )
        results.append(regular_concurrent)

        pooled_concurrent = await test_concurrent_performance(
            pooled_db, pooled_search, "Pooled Database", 5
        )
        results.append(pooled_concurrent)

        # Show pool statistics
        if hasattr(pooled_db, "get_pool_stats"):
            print("\n📊 Connection Pool Statistics:")
            stats = pooled_db.get_pool_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

        # Performance comparison
        print("\n🏆 PERFORMANCE COMPARISON")
        print("=" * 60)

        regular_avg = regular_results["avg_time"]
        pooled_avg = pooled_results["avg_time"]
        improvement = ((regular_avg - pooled_avg) / regular_avg) * 100

        print("Sequential Search Performance:")
        print(f"  Regular Database: {regular_avg:.2f}ms")
        print(f"  Pooled Database:  {pooled_avg:.2f}ms")
        print(f"  Improvement:      {improvement:.1f}%")

        regular_concurrent_throughput = regular_concurrent["concurrent_throughput"]
        pooled_concurrent_throughput = pooled_concurrent["concurrent_throughput"]
        concurrent_improvement = (
            (pooled_concurrent_throughput - regular_concurrent_throughput)
            / regular_concurrent_throughput
        ) * 100

        print("\nConcurrent Search Performance:")
        print(f"  Regular Database: {regular_concurrent_throughput:.1f} searches/sec")
        print(f"  Pooled Database:  {pooled_concurrent_throughput:.1f} searches/sec")
        print(f"  Improvement:      {concurrent_improvement:.1f}%")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if improvement > 10:
            print(
                "  ✅ Connection pooling provides significant performance improvement!"
            )
            print("  ✅ Recommended for production use")
        elif improvement > 0:
            print("  ✅ Connection pooling provides modest performance improvement")
            print("  ✅ Consider using for high-throughput scenarios")
        else:
            print("  ⚠️  Connection pooling shows minimal improvement")
            print("  ⚠️  May not be necessary for low-volume usage")

        if concurrent_improvement > 20:
            print("  ✅ Excellent concurrent performance improvement!")
            print("  ✅ Highly recommended for concurrent workloads")


if __name__ == "__main__":
    asyncio.run(main())
