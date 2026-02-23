#!/usr/bin/env python3
"""Verification script for reindex and KeyboardInterrupt bug fixes.

Bug 1: Reindex defaults to fresh=True instead of incremental
- FIXED: Changed default from True to False in reindex.py

Bug 2: KeyboardInterrupt shows "Migration interrupted" but continues
- FIXED: Added raise KeyboardInterrupt() in MigrationRunner._handle_interrupt
- FIXED: Added clean KeyboardInterrupt handling in reindex command
"""

import subprocess
import sys
from pathlib import Path


def verify_reindex_default():
    """Verify that reindex defaults to incremental (fresh=False)."""
    print("🔍 Verifying Bug 1: Reindex default behavior...")

    # Check the code
    reindex_file = (
        Path(__file__).parent.parent / "src/mcp_vector_search/cli/commands/reindex.py"
    )
    content = reindex_file.read_text()

    # Check that default is False
    if "fresh: bool = typer.Option(\n        False," in content:
        print("✅ Bug 1 FIXED: reindex defaults to incremental (fresh=False)")
        return True
    else:
        print("❌ Bug 1 NOT FIXED: reindex still defaults to fresh=True")
        return False


def verify_keyboard_interrupt_handling():
    """Verify that KeyboardInterrupt is properly propagated."""
    print("\n🔍 Verifying Bug 2: KeyboardInterrupt handling...")

    # Check MigrationRunner
    runner_file = (
        Path(__file__).parent.parent / "src/mcp_vector_search/migrations/runner.py"
    )
    content = runner_file.read_text()

    # Check that _handle_interrupt raises KeyboardInterrupt
    if "raise KeyboardInterrupt()" in content:
        print("✅ Bug 2 FIXED (Part 1): MigrationRunner raises KeyboardInterrupt")
        part1 = True
    else:
        print(
            "❌ Bug 2 NOT FIXED (Part 1): MigrationRunner doesn't raise KeyboardInterrupt"
        )
        part1 = False

    # Check reindex command
    reindex_file = (
        Path(__file__).parent.parent / "src/mcp_vector_search/cli/commands/reindex.py"
    )
    content = reindex_file.read_text()

    # Check that reindex handles KeyboardInterrupt
    if "except KeyboardInterrupt:" in content and "raise typer.Exit(130)" in content:
        print(
            "✅ Bug 2 FIXED (Part 2): reindex command handles KeyboardInterrupt cleanly"
        )
        part2 = True
    else:
        print(
            "❌ Bug 2 NOT FIXED (Part 2): reindex command doesn't handle KeyboardInterrupt"
        )
        part2 = False

    return part1 and part2


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Bug Fix Verification Script")
    print("=" * 60)
    print()

    results = []

    # Verify Bug 1
    results.append(verify_reindex_default())

    # Verify Bug 2
    results.append(verify_keyboard_interrupt_handling())

    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL BUGS FIXED!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME BUGS NOT FIXED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
