#!/usr/bin/env python3
"""Script to run comprehensive search timing tests and identify performance bottlenecks."""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.search import SemanticSearchEngine
from mcp_vector_search.utils.timing import PerformanceProfiler, SearchProfiler


class SearchTimingTestSuite:
    """Comprehensive search timing test suite."""

    def __init__(self, project_dir: Path, verbose: bool = False):
        self.project_dir = project_dir
        self.verbose = verbose
        self.profiler = PerformanceProfiler("search_timing_tests")
        self.search_profiler = SearchProfiler()

        # Initialize components
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.database = ChromaVectorDatabase(
            persist_directory=project_dir / "timing_test_db",
            embedding_function=embedding_function,
            collection_name="timing_tests",
        )

        self.indexer = SemanticIndexer(
            database=self.database,
            project_root=project_dir,
            file_extensions=[".py", ".js", ".ts", ".md"],
        )

        self.search_engine = SemanticSearchEngine(
            database=self.database,
            project_root=project_dir,
            similarity_threshold=0.1,  # Lower threshold to get more results
        )

    async def setup_test_data(self):
        """Create test data for timing tests."""
        print("🔧 Setting up test data...")

        # Create various test files with different characteristics
        test_files = [
            # Small files
            (
                "small_utils.py",
                "def add(a, b):\n    return a + b\n\ndef multiply(x, y):\n    return x * y",
            ),
            (
                "small_main.py",
                "from utils import add\n\nif __name__ == '__main__':\n    print(add(1, 2))",
            ),
            # Medium files
            ("medium_class.py", self._generate_medium_python_file()),
            ("medium_functions.py", self._generate_function_heavy_file()),
            # Large files
            ("large_module.py", self._generate_large_python_file()),
            # Different languages
            ("app.js", self._generate_javascript_file()),
            ("types.ts", self._generate_typescript_file()),
            ("README.md", self._generate_markdown_file()),
        ]

        for filename, content in test_files:
            file_path = self.project_dir / filename
            file_path.write_text(content)
            if self.verbose:
                print(f"  Created {filename} ({len(content)} chars)")

        # Index the project
        async with self.database:
            with self.profiler.time_operation("indexing"):
                indexed_count = await self.indexer.index_project()

            stats = await self.indexer.get_indexing_stats()
            print(f"✅ Indexed {indexed_count} files, {stats['total_chunks']} chunks")

    def _generate_medium_python_file(self) -> str:
        """Generate a medium-sized Python file."""
        return '''"""Medium-sized Python module for testing."""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class User:
    """User data class."""
    id: int
    name: str
    email: str
    active: bool = True

    def get_display_name(self) -> str:
        """Get user display name."""
        return f"{self.name} ({self.email})"

    def deactivate(self) -> None:
        """Deactivate user."""
        self.active = False


class UserManager:
    """Manages user operations."""

    def __init__(self):
        self.users: Dict[int, User] = {}
        self._next_id = 1

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(
            id=self._next_id,
            name=name,
            email=email
        )
        self.users[user.id] = user
        self._next_id += 1
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def list_active_users(self) -> List[User]:
        """List all active users."""
        return [user for user in self.users.values() if user.active]

    def search_users(self, query: str) -> List[User]:
        """Search users by name or email."""
        query_lower = query.lower()
        return [
            user for user in self.users.values()
            if query_lower in user.name.lower() or query_lower in user.email.lower()
        ]


def process_user_data(data: Dict[str, Any]) -> User:
    """Process raw user data into User object."""
    return User(
        id=data.get('id', 0),
        name=data.get('name', ''),
        email=data.get('email', ''),
        active=data.get('active', True)
    )


def validate_email(email: str) -> bool:
    """Simple email validation."""
    return '@' in email and '.' in email.split('@')[1]
'''

    def _generate_function_heavy_file(self) -> str:
        """Generate a file with many functions."""
        functions = []
        for i in range(20):
            functions.append(
                f'''
def function_{i}(param1, param2=None):
    """Function number {i}."""
    if param2 is None:
        param2 = param1 * 2

    result = param1 + param2
    return result * {i + 1}
'''
            )

        return "# Function-heavy module\n" + "\n".join(functions)

    def _generate_large_python_file(self) -> str:
        """Generate a large Python file."""
        base_content = self._generate_medium_python_file()

        # Add many more classes and functions
        additional_content = []
        for i in range(10):
            additional_content.append(
                f'''

class Service{i}:
    """Service class {i}."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the service."""
        try:
            self._setup_connections()
            self._load_configuration()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize Service{i}: {{e}}")
            return False

    def _setup_connections(self) -> None:
        """Setup service connections."""
        pass

    def _load_configuration(self) -> None:
        """Load service configuration."""
        pass

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a service request."""
        if not self.initialized:
            raise RuntimeError("Service not initialized")

        # Process the request
        result = {{
            "status": "success",
            "data": request_data,
            "service": "Service{i}",
            "timestamp": time.time()
        }}

        return result
'''
            )

        return base_content + "\n".join(additional_content)

    def _generate_javascript_file(self) -> str:
        """Generate a JavaScript file."""
        return """// JavaScript application file

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }

    multiply(a, b) {
        const result = a * b;
        this.history.push(`${a} * ${b} = ${result}`);
        return result;
    }

    getHistory() {
        return this.history.slice();
    }

    clearHistory() {
        this.history = [];
    }
}

function processData(data) {
    return data.map(item => ({
        ...item,
        processed: true,
        timestamp: Date.now()
    }));
}

async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch user data:', error);
        return null;
    }
}

export { Calculator, processData, fetchUserData };
"""

    def _generate_typescript_file(self) -> str:
        """Generate a TypeScript file."""
        return """// TypeScript types and interfaces

interface User {
    id: number;
    name: string;
    email: string;
    active: boolean;
    createdAt: Date;
}

interface ApiResponse<T> {
    success: boolean;
    data: T;
    error?: string;
}

type UserRole = 'admin' | 'user' | 'guest';

class UserService {
    private users: Map<number, User> = new Map();

    constructor(private apiUrl: string) {}

    async getUser(id: number): Promise<User | null> {
        const user = this.users.get(id);
        if (user) {
            return user;
        }

        try {
            const response = await fetch(`${this.apiUrl}/users/${id}`);
            const apiResponse: ApiResponse<User> = await response.json();

            if (apiResponse.success) {
                this.users.set(id, apiResponse.data);
                return apiResponse.data;
            }
        } catch (error) {
            console.error('Failed to fetch user:', error);
        }

        return null;
    }

    async createUser(userData: Omit<User, 'id' | 'createdAt'>): Promise<User | null> {
        try {
            const response = await fetch(`${this.apiUrl}/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userData)
            });

            const apiResponse: ApiResponse<User> = await response.json();

            if (apiResponse.success) {
                this.users.set(apiResponse.data.id, apiResponse.data);
                return apiResponse.data;
            }
        } catch (error) {
            console.error('Failed to create user:', error);
        }

        return null;
    }
}

export { User, ApiResponse, UserRole, UserService };
"""

    def _generate_markdown_file(self) -> str:
        """Generate a Markdown file."""
        return """# Project Documentation

This is a comprehensive documentation file for testing search performance.

## Overview

This project demonstrates various programming concepts and patterns across multiple languages.

### Features

- **User Management**: Complete user lifecycle management
- **Data Processing**: Efficient data transformation utilities
- **API Integration**: RESTful API client implementations
- **Type Safety**: Strong typing with TypeScript interfaces

## Code Examples

### Python Example

```python
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```

### JavaScript Example

```javascript
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
```

## Performance Considerations

When implementing search functionality, consider:

1. **Indexing Strategy**: Choose appropriate indexing for your data
2. **Query Optimization**: Optimize queries for common use cases
3. **Caching**: Implement caching for frequently accessed data
4. **Pagination**: Use pagination for large result sets

## Testing

Run the test suite with:

```bash
pytest tests/
```

For performance testing:

```bash
python scripts/run_search_timing_tests.py
```
"""

    async def run_basic_search_timing_tests(self):
        """Run basic search timing tests."""
        print("\n🔍 Running basic search timing tests...")

        test_queries = [
            "function",
            "class",
            "user",
            "data",
            "process",
            "calculate",
            "async",
            "interface",
            "service",
            "api",
        ]

        async with self.database:
            for query in test_queries:
                with self.profiler.time_operation(f"search_{query}", {"query": query}):
                    results = await self.search_engine.search(
                        query, limit=10, similarity_threshold=0.05
                    )

                if self.verbose:
                    print(f"  '{query}': {len(results)} results")

        # Print basic timing stats
        search_stats = {
            op: self.profiler.get_stats(op)
            for op in self.profiler.get_operation_breakdown().keys()
            if op.startswith("search_")
        }

        if search_stats:
            durations = [stats["mean"] * 1000 for stats in search_stats.values()]
            avg_duration = sum(durations) / len(durations)
            print(f"  Average search time: {avg_duration:.2f}ms")
            print(f"  Fastest: {min(durations):.2f}ms")
            print(f"  Slowest: {max(durations):.2f}ms")

    async def run_detailed_profiling(self):
        """Run detailed profiling of search operations."""
        print("\n🔬 Running detailed search profiling...")

        test_cases = [
            ("simple_word", "function"),
            ("compound_query", "user management"),
            ("technical_term", "async await"),
            ("code_pattern", "class constructor"),
            ("long_query", "implement user authentication with database integration"),
        ]

        async with self.database:
            for test_name, query in test_cases:
                print(f"\n  Profiling: {test_name} - '{query}'")

                result, timing_breakdown = await self.search_profiler.profile_search(
                    self.search_engine.search, query, limit=10
                )

                print(f"    Results: {len(result)}")
                for operation, duration_ms in timing_breakdown.items():
                    print(f"    {operation}: {duration_ms:.2f}ms")

    async def run_all_tests(self):
        """Run all timing tests."""
        print("🚀 Starting comprehensive search timing tests...")

        try:
            await self.setup_test_data()
            await self.run_basic_search_timing_tests()
            await self.run_detailed_profiling()

            print("\n📊 Final Performance Report:")
            self.profiler.print_report()

            print("\n🔍 Search-Specific Profiling:")
            self.search_profiler.print_report()

        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run search timing tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--temp-dir", help="Use specific temp directory")

    args = parser.parse_args()

    if args.temp_dir:
        test_dir = Path(args.temp_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

        test_suite = SearchTimingTestSuite(test_dir, verbose=args.verbose)
        await test_suite.run_all_tests()
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_suite = SearchTimingTestSuite(Path(temp_dir), verbose=args.verbose)
            await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
