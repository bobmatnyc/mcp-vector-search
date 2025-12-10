#!/usr/bin/env python3
"""Test gitignore functionality on EWTN project."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_vector_search.utils.gitignore import GitignoreParser


def test_gitignore():
    """Test gitignore parsing and matching."""
    project_root = Path("/Users/masa/Clients/EWTN/projects")

    print(f"Testing gitignore for: {project_root}")
    print(f".gitignore exists: {(project_root / '.gitignore').exists()}")
    print()

    # Create parser
    parser = GitignoreParser(project_root)

    print(f"Loaded {len(parser.patterns)} patterns:")
    for i, pattern in enumerate(parser.patterns, 1):
        print(f"  {i}. {pattern.pattern!r} (dir_only={pattern.is_directory_only})")
    print()

    # Test some paths
    test_paths = [
        ("node_modules", True),  # Should be ignored (directory)
        ("acidigital-rebuild/node_modules", True),  # Should be ignored (nested)
        ("src/index.js", False),  # Should NOT be ignored
        ("dist", True),  # Should be ignored
        ("build", True),  # Should be ignored
        (".vscode", True),  # Should be ignored
        ("src/components/Header.tsx", False),  # Should NOT be ignored
    ]

    print("Testing pattern matching:")
    for path_str, is_dir in test_paths:
        path = project_root / path_str
        # Use the optimized version with is_directory hint
        ignored = parser.is_ignored(path, is_directory=is_dir)
        status = "✓ IGNORED" if ignored else "✗ NOT IGNORED"
        expected = (
            "✓ IGNORED"
            if is_dir
            and any(p in path_str for p in ["node_modules", "dist", "build", ".vscode"])
            else "✗ NOT IGNORED"
        )
        match = "✓" if status == expected else "❌ MISMATCH"
        print(f"  {match} {path_str}: {status}")

    print()

    # Test performance with hint vs without
    import time

    test_path = project_root / "acidigital-rebuild" / "node_modules"

    # With hint (optimized)
    start = time.perf_counter()
    for _ in range(1000):
        parser.is_ignored(test_path, is_directory=True)
    elapsed_with_hint = time.perf_counter() - start

    # Without hint (slow - requires stat)
    start = time.perf_counter()
    for _ in range(1000):
        parser.is_ignored(test_path, is_directory=None)
    elapsed_without_hint = time.perf_counter() - start

    print("Performance test (1000 iterations):")
    print(f"  With is_directory hint:    {elapsed_with_hint:.4f}s")
    print(f"  Without hint (stat calls): {elapsed_without_hint:.4f}s")
    print(f"  Speedup: {elapsed_without_hint / elapsed_with_hint:.1f}x")


if __name__ == "__main__":
    test_gitignore()
