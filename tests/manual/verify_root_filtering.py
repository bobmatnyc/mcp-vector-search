#!/usr/bin/env python3
"""Verify that root node filtering excludes code chunks.

This script checks that the Phase 1 visualization only includes
structural nodes (directories, files, subprojects) and excludes
all code-level chunks (functions, classes, methods).
"""

import re
from pathlib import Path


def extract_root_filtering_logic(scripts_file: Path) -> str:
    """Extract the root node filtering code from scripts.py."""
    content = scripts_file.read_text()

    # Find the initializeVisualizationV2 function
    pattern = r"const rootNodesList = allNodes\.filter\(.*?\}\);"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError("Could not find rootNodesList filtering code")

    return match.group(0)


def verify_type_filtering(filter_code: str) -> dict[str, bool]:
    """Verify that type filtering is present in the code.

    Returns:
        Dictionary of verification results
    """
    results = {
        "has_type_check": False,
        "checks_directory": False,
        "checks_file": False,
        "checks_subproject": False,
        "filters_before_parent_check": False,
    }

    # Check for type filtering
    if "n.type ===" in filter_code:
        results["has_type_check"] = True

    # Check specific types
    if "'directory'" in filter_code or '"directory"' in filter_code:
        results["checks_directory"] = True

    if "'file'" in filter_code or '"file"' in filter_code:
        results["checks_file"] = True

    if "'subproject'" in filter_code or '"subproject"' in filter_code:
        results["checks_subproject"] = True

    # Check that type filtering comes before parent check
    type_check_pos = filter_code.find("n.type")
    parent_check_pos = filter_code.find("hasParent")

    if type_check_pos != -1 and parent_check_pos != -1:
        if type_check_pos < parent_check_pos:
            results["filters_before_parent_check"] = True

    return results


def verify_debug_logging(scripts_file: Path) -> bool:
    """Check if debug logging for node types exists."""
    content = scripts_file.read_text()

    # Look for the debug logging code
    has_type_counts = "nodeTypeCounts" in content
    has_non_structural = "nonStructural" in content
    has_warning = "console.warn" in content and "non-structural" in content.lower()

    return has_type_counts and has_non_structural and has_warning


def main():
    """Run verification checks."""
    # Find scripts.py
    project_root = Path(__file__).parent.parent.parent
    scripts_file = (
        project_root
        / "src/mcp_vector_search/cli/commands/visualize/templates/scripts.py"
    )

    if not scripts_file.exists():
        print(f"❌ ERROR: Could not find {scripts_file}")
        return False

    print("=" * 70)
    print("Root Node Filtering Verification")
    print("=" * 70)
    print()

    # Extract and verify filtering logic
    try:
        filter_code = extract_root_filtering_logic(scripts_file)
        print("✅ Found root node filtering code")
        print()

        # Verify type filtering
        results = verify_type_filtering(filter_code)

        print("Type Filtering Checks:")
        print("-" * 70)
        for check, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check.replace('_', ' ').title()}")
        print()

        # Verify debug logging
        has_debug = verify_debug_logging(scripts_file)
        print("Debug Logging:")
        print("-" * 70)
        print(f"{'✅' if has_debug else '❌'} Debug logging present")
        print()

        # Overall result
        all_checks = list(results.values()) + [has_debug]
        if all(all_checks):
            print("=" * 70)
            print("✅ ALL CHECKS PASSED - Root filtering is correctly implemented")
            print("=" * 70)
            print()
            print("Expected behavior:")
            print("  • Phase 1 shows ONLY directories, files, and subprojects")
            print("  • Functions, classes, and methods are hidden in Phase 1")
            print("  • Console logs show node type distribution")
            print("  • Console warns if non-structural nodes appear")
            print()
            return True
        else:
            print("=" * 70)
            print("❌ SOME CHECKS FAILED - Review the implementation")
            print("=" * 70)
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
