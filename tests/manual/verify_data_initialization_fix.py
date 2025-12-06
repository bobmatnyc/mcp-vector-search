#!/usr/bin/env python3
"""Verification test for the data initialization bug fix.

This test verifies that the JavaScript data loading logic correctly initializes
allNodes and allLinks for both small graphs (<500 nodes) and large graphs (>500 nodes).

Bug Fixed: For graphs with >500 nodes, the code auto-selects Dagre layout but
was skipping the visualizeGraph(data) call that initializes allNodes and allLinks,
resulting in an empty graph (0 nodes, 0 links).

Fix: Always initialize allNodes and allLinks before any layout selection.
"""

from mcp_vector_search.cli.commands.visualize.templates.scripts import (
    get_data_loading_logic,
)


def test_data_initialization_always_happens():
    """Verify that data initialization happens before layout selection."""
    js_code = get_data_loading_logic()

    # Check that the critical initialization code exists
    assert "allNodes = data.nodes;" in js_code, "allNodes initialization missing"
    assert "allLinks = data.links;" in js_code, "allLinks initialization missing"

    # Check that the initialization happens in the data loading success handler
    assert ".then(data => {" in js_code, "Data loading handler missing"

    # Verify the initialization comment exists
    assert "CRITICAL: Always initialize data arrays first" in js_code, (
        "Critical comment missing"
    )

    # Verify both code paths exist
    assert "visualizeGraph(data);" in js_code, "visualizeGraph call missing"
    assert "switchToCytoscapeLayout('dagre')" in js_code, (
        "switchToCytoscapeLayout call missing"
    )

    # Verify the >500 nodes check exists
    assert "data.nodes.length > 500" in js_code, "Large graph check missing"

    print("✓ All data initialization checks passed!")


def test_initialization_order():
    """Verify that data initialization happens BEFORE layout selection."""
    js_code = get_data_loading_logic()

    # Find the position of key code elements
    init_pos = js_code.find("allNodes = data.nodes;")
    dagre_pos = js_code.find("switchToCytoscapeLayout('dagre')")
    visualize_pos = js_code.find("visualizeGraph(data);")

    # Verify initialization happens before both layout options
    assert init_pos > 0, "Data initialization not found"
    assert dagre_pos > 0, "Dagre layout switch not found"
    assert visualize_pos > 0, "visualizeGraph not found"

    # Critical check: initialization must happen BEFORE both layout calls
    assert init_pos < dagre_pos, (
        "allNodes initialization happens AFTER switchToCytoscapeLayout (BUG!)"
    )
    assert init_pos < visualize_pos, (
        "allNodes initialization happens AFTER visualizeGraph (BUG!)"
    )

    print("✓ Data initialization order is correct!")
    print(f"  - allNodes init at position: {init_pos}")
    print(f"  - switchToCytoscapeLayout at: {dagre_pos}")
    print(f"  - visualizeGraph at: {visualize_pos}")


def test_no_duplicate_initialization():
    """Verify we're not initializing the same data multiple times unnecessarily."""
    js_code = get_data_loading_logic()

    # Count how many times we initialize allNodes in the data loading logic
    # (should be exactly once in the main data handler)
    count = js_code.count("allNodes = data.nodes;")

    # We expect exactly 1 initialization in get_data_loading_logic
    # (visualizeGraph has its own, but that's in a different function)
    assert count == 1, f"Expected 1 initialization, found {count}"

    print("✓ Data initialization happens exactly once in data loading logic")


if __name__ == "__main__":
    print("Testing data initialization bug fix...\n")
    test_data_initialization_always_happens()
    print()
    test_initialization_order()
    print()
    test_no_duplicate_initialization()
    print("\n✓ All verification tests passed!")
    print(
        "\nBug fix verified: Data arrays are now initialized before layout selection."
    )
