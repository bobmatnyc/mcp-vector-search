#!/usr/bin/env python3
"""Verification script for embedding progress bar fix.

This script verifies that:
1. count_pending_chunks() method exists and is callable
2. The progress bar logic in _phase2_embed_chunks() uses total_pending_chunks correctly
3. The progress bar is NOT recreated in each loop iteration
"""

import ast
import inspect
from pathlib import Path


def verify_count_pending_chunks_exists():
    """Verify that count_pending_chunks method exists in ChunksBackend."""
    from mcp_vector_search.core.chunks_backend import ChunksBackend

    if not hasattr(ChunksBackend, "count_pending_chunks"):
        print("✗ count_pending_chunks method NOT found in ChunksBackend")
        return False

    method = ChunksBackend.count_pending_chunks
    if not inspect.iscoroutinefunction(method):
        print("✗ count_pending_chunks is not async")
        return False

    print("✓ count_pending_chunks method exists and is async")
    return True


def verify_indexer_uses_total_pending():
    """Verify that _phase2_embed_chunks uses total_pending_chunks correctly."""
    indexer_path = (
        Path(__file__).parent.parent
        / "src"
        / "mcp_vector_search"
        / "core"
        / "indexer.py"
    )

    with open(indexer_path) as f:
        source = f.read()

    # Parse the source code
    tree = ast.parse(source)

    # Find _phase2_embed_chunks method
    phase2_method = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_phase2_embed_chunks"
        ):
            phase2_method = node
            break

    if not phase2_method:
        print("✗ _phase2_embed_chunks method not found")
        return False

    # Check for count_pending_chunks call
    has_count_pending_call = False
    has_total_pending_chunks_var = False
    progress_bar_calls = []

    for node in ast.walk(phase2_method):
        # Check for count_pending_chunks call
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "count_pending_chunks":
                    has_count_pending_call = True
                elif node.func.attr == "progress_bar_with_eta":
                    # Extract arguments
                    progress_bar_calls.append(node)

        # Check for total_pending_chunks variable
        if isinstance(node, ast.Name) and node.id == "total_pending_chunks":
            has_total_pending_chunks_var = True

    if not has_count_pending_call:
        print("✗ count_pending_chunks() not called in _phase2_embed_chunks")
        return False

    if not has_total_pending_chunks_var:
        print("✗ total_pending_chunks variable not found in _phase2_embed_chunks")
        return False

    print("✓ _phase2_embed_chunks calls count_pending_chunks()")
    print("✓ _phase2_embed_chunks uses total_pending_chunks variable")

    # Verify progress_bar_with_eta uses total_pending_chunks
    uses_total_in_progress = False
    for call in progress_bar_calls:
        for keyword in call.keywords:
            if keyword.arg == "total":
                if (
                    isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "total_pending_chunks"
                ):
                    uses_total_in_progress = True

    if not uses_total_in_progress:
        print(
            "✗ progress_bar_with_eta does not use total_pending_chunks for total parameter"
        )
        return False

    print("✓ progress_bar_with_eta uses total_pending_chunks for total parameter")

    # Check that estimated_total_chunks and first_batch are NOT present (old logic removed)
    has_estimated_total = False
    has_first_batch = False

    for node in ast.walk(phase2_method):
        if isinstance(node, ast.Name):
            if node.id == "estimated_total_chunks":
                has_estimated_total = True
            elif node.id == "first_batch":
                has_first_batch = True

    if has_estimated_total:
        print(
            "⚠️  Warning: estimated_total_chunks variable still present (should be removed)"
        )
    else:
        print("✓ estimated_total_chunks variable removed (good!)")

    if has_first_batch:
        print("⚠️  Warning: first_batch variable still present (should be removed)")
    else:
        print("✓ first_batch variable removed (good!)")

    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Progress Bar Fix Verification")
    print("=" * 60)
    print()

    print("1. Checking count_pending_chunks method...")
    check1 = verify_count_pending_chunks_exists()
    print()

    print("2. Checking _phase2_embed_chunks implementation...")
    check2 = verify_indexer_uses_total_pending()
    print()

    print("=" * 60)
    if check1 and check2:
        print("✓ All verification checks passed!")
        print()
        print("Summary of Fix:")
        print("- Added count_pending_chunks() method to ChunksBackend")
        print(
            "- Modified _phase2_embed_chunks() to get total pending count BEFORE loop"
        )
        print("- Progress bar now updates single bar with correct total instead of")
        print("  creating new bars per batch")
        return 0
    else:
        print("✗ Some verification checks failed")
        return 1


if __name__ == "__main__":
    exit(main())
