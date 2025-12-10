#!/usr/bin/env python3
"""Manual test script for compact folder layout visualization.

This script generates a test visualization to verify that:
1. Folders are positioned in a compact grid layout
2. All folders are visible in the initial viewport
3. Reset view maintains the compact layout
4. Spacing is appropriate (150px between folders)

Usage:
    uv run python tests/manual/test_compact_folder_layout.py
    # Then open the generated HTML file in a browser
"""

import json
from pathlib import Path


def create_test_graph_data():
    """Create test graph data with various folder counts for testing."""

    test_scenarios = [
        ("small", 4, "Test with 4 folders (2×2 grid)"),
        ("medium", 9, "Test with 9 folders (3×3 grid)"),
        ("large", 16, "Test with 16 folders (4×4 grid)"),
    ]

    for name, folder_count, description in test_scenarios:
        print(f"\n{description}")

        nodes = []
        links = []

        # Create folder nodes
        for i in range(folder_count):
            nodes.append(
                {
                    "id": f"folder_{i}",
                    "name": f"folder_{i}",
                    "type": "directory",
                    "file_path": f"/test/folder_{i}",
                    "depth": 0,
                    "color": "#79c0ff",
                }
            )

        # Add some files to a few folders to test expansion
        for i in range(min(3, folder_count)):
            file_id = f"file_in_folder_{i}"
            nodes.append(
                {
                    "id": file_id,
                    "name": f"test_file_{i}.py",
                    "type": "file",
                    "file_path": f"/test/folder_{i}/test_file_{i}.py",
                    "depth": 1,
                    "color": "#58a6ff",
                }
            )

            # Link file to folder
            links.append(
                {"source": f"folder_{i}", "target": file_id, "type": "dir_containment"}
            )

        # Create graph data
        graph_data = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_files": len([n for n in nodes if n["type"] == "file"]),
                "is_monorepo": False,
            },
        }

        # Save to file
        output_file = Path(__file__).parent / f"test_graph_{name}.json"
        with open(output_file, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"  ✓ Created {output_file}")
        print(f"  - {folder_count} folders")
        print(
            f"  - Expected grid: {folder_count // int(folder_count**0.5)}×{int(folder_count**0.5)}"
        )
        print("  - Spacing: 150px")


def create_test_instructions():
    """Create instructions for manual testing."""

    instructions = """
# Manual Testing Instructions for Compact Folder Layout

## Prerequisites
1. Run: `mcp-vector-search visualize export`
2. This will generate the visualization HTML file

## Test Scenarios

### Small Project (4 folders - 2×2 grid)
1. Copy `test_graph_small.json` to `chunk-graph.json` in visualization directory
2. Open visualization HTML in browser
3. Verify:
   - [ ] 4 folders visible in viewport
   - [ ] Arranged in ~2×2 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

### Medium Project (9 folders - 3×3 grid)
1. Copy `test_graph_medium.json` to `chunk-graph.json`
2. Open visualization HTML in browser
3. Verify:
   - [ ] 9 folders visible in viewport
   - [ ] Arranged in 3×3 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

### Large Project (16 folders - 4×4 grid)
1. Copy `test_graph_large.json` to `chunk-graph.json`
2. Open visualization HTML in browser
3. Verify:
   - [ ] 16 folders visible in viewport
   - [ ] Arranged in 4×4 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

## Interaction Tests

For each scenario, also test:

1. **Initial Load**
   - [ ] All folders visible immediately
   - [ ] No need to zoom/pan to see all folders
   - [ ] Smooth animation to final positions

2. **Expansion**
   - [ ] Click folder to expand
   - [ ] Files appear below folder
   - [ ] Grid layout maintained for other folders

3. **Collapse**
   - [ ] Click expanded folder to collapse
   - [ ] Files disappear
   - [ ] Grid layout restored

4. **Reset View**
   - [ ] Click "Reset View" button
   - [ ] All folders visible again
   - [ ] Same grid layout as initial load
   - [ ] Smooth transition

5. **Zoom/Pan**
   - [ ] Manually zoom in/out
   - [ ] Click "Reset View"
   - [ ] Zoom restored to show all folders

## Success Criteria

✅ All folders visible in initial viewport
✅ Grid pattern clear and organized
✅ Spacing consistent (approximately 150px)
✅ Reset view restores compact layout
✅ Smooth transitions and animations
✅ No performance issues

## Troubleshooting

**Problem**: Folders scattered randomly
- **Solution**: Check that `positionFoldersCompactly()` is being called
- **Check**: Browser console for errors

**Problem**: Not all folders visible
- **Solution**: Check `zoomToFit()` is being called with correct timing
- **Check**: Increase padding or adjust scale factor

**Problem**: Folders overlap
- **Solution**: Increase spacing constant (currently 150px)
- **Check**: Collision detection radius settings

**Problem**: Jerky animation
- **Solution**: Check timing of position fixes/releases
- **Check**: Browser performance in dev tools

## Reporting Results

After testing, report:
1. Which scenarios passed/failed
2. Browser used (Chrome, Firefox, Safari)
3. Viewport size when testing
4. Any visual issues or bugs observed
5. Performance observations

## Files Generated

- `test_graph_small.json` - 4 folders
- `test_graph_medium.json` - 9 folders
- `test_graph_large.json` - 16 folders
- `test_instructions.md` - This file
"""

    output_file = Path(__file__).parent / "test_instructions.md"
    with open(output_file, "w") as f:
        f.write(instructions.strip())

    print(f"\n✓ Created {output_file}")


def main():
    """Generate test data and instructions."""
    print("=" * 60)
    print("Compact Folder Layout - Test Data Generator")
    print("=" * 60)

    create_test_graph_data()
    create_test_instructions()

    print("\n" + "=" * 60)
    print("Test data generated successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: mcp-vector-search visualize export")
    print("2. Copy one of the test_graph_*.json files to chunk-graph.json")
    print("3. Open visualization HTML in browser")
    print("4. Follow instructions in test_instructions.md")
    print("\nTest files created in:", Path(__file__).parent)


if __name__ == "__main__":
    main()
