#!/usr/bin/env python3
"""Test file scanning on EWTN project."""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS
from mcp_vector_search.utils.gitignore import GitignoreParser


def test_file_scan():
    """Test file scanning with gitignore filtering."""
    project_root = Path("/Users/masa/Clients/EWTN/projects")

    print(f"Testing file scan for: {project_root}")
    print()

    # Create parser
    parser = GitignoreParser(project_root)
    print(f"Loaded {len(parser.patterns)} gitignore patterns")

    # File extensions to index
    file_extensions = set(DEFAULT_FILE_EXTENSIONS)
    print(f"File extensions: {len(file_extensions)} types")
    print()

    # Scan files with os.walk
    print("Scanning files with os.walk + gitignore filtering...")
    start = time.time()

    indexable_files = []
    dirs_checked = 0
    dirs_filtered = 0
    files_checked = 0

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        dirs_checked += len(dirs)

        # Filter directories IN-PLACE
        original_dir_count = len(dirs)
        dirs[:] = [
            d for d in dirs if not parser.is_ignored(root_path / d, is_directory=True)
        ]
        dirs_filtered += original_dir_count - len(dirs)

        # Check files
        for filename in files:
            files_checked += 1
            file_path = root_path / filename

            # Check extension
            if file_path.suffix.lower() not in file_extensions:
                continue

            # Check if ignored
            if parser.is_ignored(file_path, is_directory=False):
                continue

            indexable_files.append(file_path)

    elapsed = time.time() - start

    print(f"✓ Scan completed in {elapsed:.2f}s")
    print()
    print("Statistics:")
    print(f"  Directories checked: {dirs_checked:,}")
    print(
        f"  Directories filtered: {dirs_filtered:,} ({dirs_filtered / max(dirs_checked, 1) * 100:.1f}%)"
    )
    print(f"  Files checked: {files_checked:,}")
    print(f"  Indexable files found: {len(indexable_files):,}")
    print()

    if indexable_files:
        print("Sample indexable files (first 10):")
        for f in indexable_files[:10]:
            print(f"  - {f.relative_to(project_root)}")
        print()

    # Expected: ~1,259 files if node_modules is properly excluded
    expected_max = 5000
    if len(indexable_files) > expected_max:
        print(
            f"⚠️  WARNING: Found {len(indexable_files):,} files (expected <{expected_max:,})"
        )
        print("   This suggests gitignore filtering may not be working correctly")
    else:
        print(f"✓ File count looks reasonable ({len(indexable_files):,} files)")

    return len(indexable_files), elapsed


if __name__ == "__main__":
    count, elapsed = test_file_scan()
    print()
    print(f"Summary: Found {count:,} indexable files in {elapsed:.2f}s")
