#!/usr/bin/env python3
"""Demonstration of semi-automatic reindexing strategies."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.auto_indexer import AutoIndexer
from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.git_hooks import GitHookManager
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.scheduler import SchedulerManager
from mcp_vector_search.core.search import SemanticSearchEngine


async def demonstrate_semi_automatic_reindexing():
    """Demonstrate various semi-automatic reindexing strategies."""

    print("🔄 Semi-Automatic Reindexing Strategies Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create example files
        example_files = {
            "main.py": '''
def main():
    """Main application entry point."""
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
''',
            "utils.py": '''
def helper_function():
    """A helpful utility function."""
    return "helper"

class UtilityClass:
    def method(self):
        return "utility"
''',
        }

        print("📁 Creating example project...")
        for filename, content in example_files.items():
            (project_dir / filename).write_text(content)

        # Initialize components
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        database = ChromaVectorDatabase(
            persist_directory=project_dir / "chroma_db",
            embedding_function=embedding_function,
            collection_name="semi_auto_demo",
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_dir,
            file_extensions=[".py"],
        )

        # Initial indexing
        print("\n📚 Initial indexing...")
        async with database:
            indexed_count = await indexer.index_project()
            print(f"  Indexed {indexed_count} files")

        # Strategy 1: Search-Triggered Auto-Indexing
        print("\n🔍 Strategy 1: Search-Triggered Auto-Indexing")
        print("-" * 50)

        auto_indexer = AutoIndexer(
            indexer=indexer,
            database=database,
            auto_reindex_threshold=3,  # Auto-reindex up to 3 files
            staleness_threshold=1.0,  # 1 second for demo
        )

        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_dir,
            similarity_threshold=0.1,
            auto_indexer=auto_indexer,
            enable_auto_reindex=True,
        )

        # Modify a file
        await asyncio.sleep(0.1)  # Ensure different mtime
        (project_dir / "main.py").write_text(
            '''
def main():
    """Main application entry point."""
    print("Hello, Updated World!")
    return 0

def new_function():
    """A new function added."""
    return "new"

if __name__ == "__main__":
    main()
'''
        )

        print("  Modified main.py with new content")

        # Perform searches - should trigger auto-reindexing
        async with database:
            print("  Performing searches (should trigger auto-reindex)...")

            # First search should trigger reindex check
            results = await search_engine.search("main function", limit=3)
            print(f"    Search 1: {len(results)} results")

            # Search for new content
            results = await search_engine.search("new function", limit=3)
            print(f"    Search 2: {len(results)} results")

            if results:
                print(f"    ✅ Found new content: {results[0].content[:30]}...")

            # Get search stats
            if search_engine.search_triggered_indexer:
                stats = search_engine.search_triggered_indexer.get_search_stats()
                print(f"    Search stats: {stats['total_searches']} total searches")

        # Strategy 2: Manual Auto-Indexing Check
        print("\n🔧 Strategy 2: Manual Auto-Indexing Check")
        print("-" * 50)

        # Modify another file
        await asyncio.sleep(0.1)
        (project_dir / "utils.py").write_text(
            '''
def helper_function():
    """A helpful utility function."""
    return "helper"

def another_helper():
    """Another helper function."""
    return "another"

class UtilityClass:
    def method(self):
        return "utility"

    def new_method(self):
        return "new_method"
'''
        )

        print("  Modified utils.py with new content")

        async with database:
            # Manual check
            print("  Running manual auto-reindex check...")
            reindexed, file_count = await auto_indexer.check_and_reindex_if_needed(
                force_check=True, interactive=False
            )

            if reindexed:
                print(f"    ✅ Auto-reindexed {file_count} files")
            else:
                print(f"    ℹ️  {file_count} files were stale but not auto-reindexed")

            # Get staleness info
            staleness_info = auto_indexer.get_staleness_info()
            print("    Staleness info:")
            print(f"      Total files: {staleness_info['total_files']}")
            print(f"      Stale files: {staleness_info['stale_files']}")
            print(f"      Index age: {staleness_info['staleness_seconds']:.1f}s")

        # Strategy 3: Git Hooks (Demo Setup)
        print("\n🔗 Strategy 3: Git Hooks Integration")
        print("-" * 50)

        git_manager = GitHookManager(project_dir)

        if git_manager.is_git_repo():
            print("  Git repository detected")

            # Install hooks (demo - won't actually work without real git repo)
            print("  Installing Git hooks...")
            success = git_manager.install_hooks()
            print(f"    Hook installation: {'✅ Success' if success else '❌ Failed'}")

            # Get hook status
            status = git_manager.get_hook_status()
            print(f"    Hook status: {status}")

        else:
            print("  ℹ️  Not a Git repository - Git hooks would work in real Git repos")
            print("  Git hooks would trigger reindexing after:")
            print("    • git commit")
            print("    • git merge")
            print("    • git checkout")

        # Strategy 4: Scheduled Tasks (Demo Setup)
        print("\n⏰ Strategy 4: Scheduled Tasks")
        print("-" * 50)

        scheduler = SchedulerManager(project_dir)

        print(f"  System: {scheduler.system}")
        print("  Scheduled tasks would:")
        print("    • Run every N minutes (configurable)")
        print("    • Check for stale files automatically")
        print("    • Reindex small numbers of changed files")
        print("    • Work across system reboots")

        # Demo task status (won't actually install for demo)
        task_status = scheduler.get_scheduled_task_status()
        print(f"    Task status: {task_status}")

        # Strategy 5: Periodic Checker (In-Process)
        print("\n🔄 Strategy 5: Periodic In-Process Checker")
        print("-" * 50)

        from mcp_vector_search.core.auto_indexer import PeriodicIndexChecker

        periodic_checker = PeriodicIndexChecker(
            auto_indexer=auto_indexer,
            check_interval=5.0,  # 5 seconds for demo
        )

        print("  Periodic checker configured (5 second interval)")

        # Simulate periodic checks
        for i in range(3):
            print(f"    Check {i + 1}:")

            # Maybe check and reindex
            async with database:
                reindexed = await periodic_checker.maybe_check_and_reindex()
                time_until_next = periodic_checker.time_until_next_check()

                print(f"      Reindexed: {'✅ Yes' if reindexed else '❌ No'}")
                print(f"      Next check in: {time_until_next:.1f}s")

            if i < 2:  # Don't sleep on last iteration
                await asyncio.sleep(2)

        # Summary and Recommendations
        print("\n📋 Semi-Automatic Reindexing Summary")
        print("=" * 60)

        print("✅ Available Strategies:")
        print("  1. Search-Triggered: Built-in, zero setup, works during searches")
        print("  2. Manual Checks: On-demand via CLI commands")
        print("  3. Git Hooks: Triggers on Git operations (commit, merge, etc.)")
        print("  4. Scheduled Tasks: System-level cron/task scheduler")
        print("  5. Periodic Checker: In-process periodic checks")

        print("\n💡 Recommendations:")
        print("  • Start with search-triggered (always enabled)")
        print("  • Add Git hooks for development workflows")
        print("  • Use scheduled tasks for production/CI environments")
        print("  • Manual checks for troubleshooting")
        print("  • Periodic checker for long-running processes")

        print("\n🚀 Getting Started:")
        print("  mcp-vector-search auto-index setup --method all")
        print("  mcp-vector-search auto-index status")
        print("  mcp-vector-search auto-index check --auto-reindex")

        print("\n✨ Benefits:")
        print("  • No daemon processes required")
        print("  • Minimal resource usage")
        print("  • Multiple trigger mechanisms")
        print("  • Configurable thresholds")
        print("  • Graceful degradation")


async def main():
    """Main function."""
    try:
        await demonstrate_semi_automatic_reindexing()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
