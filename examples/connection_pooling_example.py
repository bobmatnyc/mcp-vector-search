#!/usr/bin/env python3
"""Example demonstrating connection pooling usage for improved performance."""

import asyncio

# Add src to path for imports
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import PooledChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.search import SemanticSearchEngine


async def demonstrate_connection_pooling():
    """Demonstrate the benefits of connection pooling."""

    print("🔗 Connection Pooling Example")
    print("=" * 50)

    # Create a temporary project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create some example code files
        example_files = {
            "main.py": """
def main():
    \"\"\"Main application entry point.\"\"\"
    print("Starting application...")
    user_service = UserService()
    users = user_service.get_all_users()
    print(f"Found {len(users)} users")

if __name__ == "__main__":
    main()
""",
            "user_service.py": """
from typing import List, Optional

class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email

    def __repr__(self):
        return f"User(id={self.id}, name='{self.name}', email='{self.email}')"

class UserService:
    def __init__(self):
        self.users = []

    def create_user(self, name: str, email: str) -> User:
        \"\"\"Create a new user.\"\"\"
        user_id = len(self.users) + 1
        user = User(user_id, name, email)
        self.users.append(user)
        return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        \"\"\"Get user by ID.\"\"\"
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_all_users(self) -> List[User]:
        \"\"\"Get all users.\"\"\"
        return self.users.copy()

    def delete_user(self, user_id: int) -> bool:
        \"\"\"Delete user by ID.\"\"\"
        for i, user in enumerate(self.users):
            if user.id == user_id:
                del self.users[i]
                return True
        return False
""",
            "database.py": """
import sqlite3
from typing import List, Dict, Any

class DatabaseConnection:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        \"\"\"Establish database connection.\"\"\"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def disconnect(self):
        \"\"\"Close database connection.\"\"\"
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        \"\"\"Execute a SELECT query.\"\"\"
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        \"\"\"Execute an INSERT/UPDATE/DELETE query.\"\"\"
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor.rowcount
""",
        }

        print("📁 Creating example project files...")
        for filename, content in example_files.items():
            (project_dir / filename).write_text(content)

        # Initialize embedding function
        print("🧠 Initializing embedding function...")
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize pooled database
        print("🔗 Initializing pooled database...")
        database = PooledChromaVectorDatabase(
            persist_directory=project_dir / "chroma_db",
            embedding_function=embedding_function,
            collection_name="example_search",
            max_connections=5,
            min_connections=2,
            max_idle_time=300.0,  # 5 minutes
        )

        # Initialize indexer and search engine
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
        print("📚 Indexing project...")
        async with database:
            indexed_count = await indexer.index_project()
            stats = await indexer.get_indexing_stats()
            print(f"  Indexed {indexed_count} files, {stats['total_chunks']} chunks")

        # Demonstrate search performance with connection pooling
        print("\n🔍 Demonstrating search with connection pooling...")

        search_queries = [
            "user management",
            "database connection",
            "create user",
            "main function",
            "get all users",
            "sqlite query",
            "class definition",
            "error handling",
        ]

        # Perform searches with timing
        search_times = []
        total_results = 0

        async with database:
            print("  Running searches with pooled connections...")

            for i, query in enumerate(search_queries, 1):
                start_time = time.perf_counter()
                results = await search_engine.search(
                    query, limit=5, similarity_threshold=0.05
                )
                end_time = time.perf_counter()

                search_time = (end_time - start_time) * 1000  # Convert to ms
                search_times.append(search_time)
                total_results += len(results)

                print(
                    f"    {i}. '{query}': {search_time:.2f}ms ({len(results)} results)"
                )

        # Show performance statistics
        avg_time = sum(search_times) / len(search_times)
        min_time = min(search_times)
        max_time = max(search_times)

        print("\n📊 Search Performance Summary:")
        print(f"  Total searches: {len(search_queries)}")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Min/Max time: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"  Total results: {total_results}")
        print(f"  Throughput: {1000 / avg_time:.1f} searches/sec")

        # Show connection pool statistics
        pool_stats = database.get_pool_stats()
        print("\n🔗 Connection Pool Statistics:")
        print(f"  Pool size: {pool_stats['pool_size']}")
        print(f"  Active connections: {pool_stats['active_connections']}")
        print(f"  Idle connections: {pool_stats['idle_connections']}")
        print(f"  Connections created: {pool_stats['connections_created']}")
        print(f"  Connections reused: {pool_stats['connections_reused']}")
        print(f"  Pool hits: {pool_stats['pool_hits']}")
        print(f"  Pool misses: {pool_stats['pool_misses']}")

        if pool_stats["pool_hits"] + pool_stats["pool_misses"] > 0:
            hit_rate = (
                pool_stats["pool_hits"]
                / (pool_stats["pool_hits"] + pool_stats["pool_misses"])
                * 100
            )
            print(f"  Pool hit rate: {hit_rate:.1f}%")

        # Demonstrate concurrent searches
        print("\n🚀 Demonstrating concurrent searches...")

        async def concurrent_search(query: str) -> tuple:
            start_time = time.perf_counter()
            results = await search_engine.search(
                query, limit=3, similarity_threshold=0.05
            )
            end_time = time.perf_counter()
            return query, len(results), (end_time - start_time) * 1000

        # Run 5 concurrent searches
        concurrent_queries = ["user", "database", "function", "class", "connection"]

        async with database:
            start_wall = time.perf_counter()
            tasks = [concurrent_search(query) for query in concurrent_queries]
            concurrent_results = await asyncio.gather(*tasks)
            wall_time = (time.perf_counter() - start_wall) * 1000

        print("  Concurrent searches completed:")
        for query, result_count, duration in concurrent_results:
            print(f"    '{query}': {duration:.2f}ms ({result_count} results)")

        individual_avg = sum(r[2] for r in concurrent_results) / len(concurrent_results)
        print(f"  Wall time: {wall_time:.2f}ms")
        print(f"  Average individual time: {individual_avg:.2f}ms")
        print(
            f"  Concurrent throughput: {len(concurrent_queries) / (wall_time / 1000):.1f} searches/sec"
        )

        # Final pool statistics
        final_stats = database.get_pool_stats()
        print("\n📈 Final Pool Statistics:")
        print(f"  Total connections created: {final_stats['connections_created']}")
        print(f"  Total connections reused: {final_stats['connections_reused']}")
        print(
            f"  Reuse ratio: {final_stats['connections_reused'] / max(1, final_stats['connections_created']):.1f}x"
        )

        # Health check
        is_healthy = await database.health_check()
        print(f"  Pool health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

        print("\n✅ Connection pooling demonstration completed!")
        print("\nKey Benefits Observed:")
        print("  • Reduced connection overhead")
        print("  • Improved search performance")
        print("  • Efficient resource utilization")
        print("  • Excellent concurrent performance")


async def main():
    """Main function."""
    try:
        await demonstrate_connection_pooling()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
