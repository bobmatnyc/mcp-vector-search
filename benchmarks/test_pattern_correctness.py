#!/usr/bin/env python3
"""Test that optimized pattern matching produces correct results."""

from pathlib import Path

from src.mcp_vector_search.core.file_discovery import FileDiscovery


def test_pattern_matching():
    """Verify that compiled patterns match the same paths as original fnmatch."""
    project_root = Path(".")
    file_extensions = {".py"}

    # Test patterns
    test_patterns = {
        "node_modules",
        ".*",  # dotfiles
        "*.pyc",
        "build",
        "__pycache__",
        "venv*",
        ".git",
    }

    fd = FileDiscovery(project_root, file_extensions, ignore_patterns=test_patterns)

    # Test cases: (path_part, should_match)
    test_cases = [
        ("node_modules", True),
        (".git", True),
        (".gitignore", True),  # matches .*
        ("build", True),
        ("__pycache__", True),
        ("venv", True),  # matches venv*
        ("venv-test", True),  # matches venv*
        ("main.pyc", True),  # matches *.pyc
        ("src", False),
        ("test.py", False),
        ("README.md", False),
    ]

    print("Testing pattern matching correctness...\n")

    all_passed = True
    for part, expected_match in test_cases:
        result = fd._matches_compiled_patterns(part)
        status = "✅" if result == expected_match else "❌"
        all_passed = all_passed and (result == expected_match)

        print(f"{status} '{part}': expected={expected_match}, got={result}")

    if all_passed:
        print("\n✅ All pattern matching tests PASSED!")
    else:
        print("\n❌ Some pattern matching tests FAILED!")

    return all_passed


def test_directory_filtering():
    """Test that directories are filtered correctly."""
    project_root = Path(".")
    file_extensions = {".py"}

    fd = FileDiscovery(project_root, file_extensions)

    # Test known directories that should be ignored
    ignored_dirs = [
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
    ]

    # Test known directories that should NOT be ignored
    allowed_dirs = [
        "src",
        "tests",
        "docs",
    ]

    print("\nTesting directory filtering...\n")

    all_passed = True

    print("Should be ignored:")
    for dir_name in ignored_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            should_ignore = fd.should_ignore_path(dir_path, is_directory=True)
            status = "✅" if should_ignore else "❌"
            all_passed = all_passed and should_ignore
            print(f"{status} {dir_name}: ignored={should_ignore}")

    print("\nShould NOT be ignored:")
    for dir_name in allowed_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            should_ignore = fd.should_ignore_path(dir_path, is_directory=True)
            status = "✅" if not should_ignore else "❌"
            all_passed = all_passed and not should_ignore
            print(f"{status} {dir_name}: ignored={should_ignore}")

    if all_passed:
        print("\n✅ All directory filtering tests PASSED!")
    else:
        print("\n❌ Some directory filtering tests FAILED!")

    return all_passed


if __name__ == "__main__":
    pattern_passed = test_pattern_matching()
    dir_passed = test_directory_filtering()

    if pattern_passed and dir_passed:
        print("\n" + "=" * 60)
        print("✅ ALL CORRECTNESS TESTS PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME CORRECTNESS TESTS FAILED!")
        print("=" * 60)
        exit(1)
