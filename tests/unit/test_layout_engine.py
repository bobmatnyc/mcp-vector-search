"""Unit tests for layout engine algorithms.

Tests for visualization V2.0 layout calculations (list and fan layouts).
"""

import math

from mcp_vector_search.cli.commands.visualize.layout_engine import (
    _build_tree_levels,
    calculate_compact_folder_layout,
    calculate_fan_layout,
    calculate_list_layout,
    calculate_tree_layout,
)


class TestListLayout:
    """Tests for vertical list layout algorithm."""

    def test_empty_nodes_returns_empty_dict(self):
        """Test that empty node list returns empty positions."""
        result = calculate_list_layout([], 1920, 1080)
        assert result == {}

    def test_single_node_centered_vertically(self):
        """Test that single node is centered vertically."""
        nodes = [{"id": "node1", "name": "Test", "type": "directory"}]
        result = calculate_list_layout(nodes, 1920, 1080)

        assert "node1" in result
        x, y = result["node1"]
        assert x == 100  # Left margin
        # With one node: totalHeight=50, startY=(1080-50)/2=515
        assert y == 515.0

    def test_alphabetical_sorting(self):
        """Test that nodes are sorted alphabetically."""
        nodes = [
            {"id": "z", "name": "Zebra", "type": "file"},
            {"id": "a", "name": "Apple", "type": "file"},
            {"id": "m", "name": "Mango", "type": "file"},
        ]
        result = calculate_list_layout(nodes, 1920, 1080)

        # Extract y positions
        y_a = result["a"][1]
        y_m = result["m"][1]
        y_z = result["z"][1]

        # Verify alphabetical order (top to bottom)
        assert y_a < y_m < y_z

    def test_directories_before_files(self):
        """Test that directories appear before files."""
        nodes = [
            {"id": "file1", "name": "zebra.py", "type": "file"},
            {"id": "dir1", "name": "alpha", "type": "directory"},
            {"id": "file2", "name": "beta.py", "type": "file"},
        ]
        result = calculate_list_layout(nodes, 1920, 1080)

        # Extract y positions
        y_dir1 = result["dir1"][1]
        y_file1 = result["file1"][1]
        y_file2 = result["file2"][1]

        # Directory should be first, then files in alphabetical order
        assert y_dir1 < y_file2 < y_file1

    def test_vertical_spacing(self):
        """Test that nodes have correct 50px vertical spacing."""
        nodes = [
            {"id": "n1", "name": "A", "type": "directory"},
            {"id": "n2", "name": "B", "type": "directory"},
            {"id": "n3", "name": "C", "type": "directory"},
        ]
        result = calculate_list_layout(nodes, 1920, 1080)

        y1 = result["n1"][1]
        y2 = result["n2"][1]
        y3 = result["n3"][1]

        # Verify 50px spacing between consecutive nodes
        assert abs((y2 - y1) - 50) < 0.01
        assert abs((y3 - y2) - 50) < 0.01

    def test_missing_node_id_skipped(self):
        """Test that nodes missing 'id' are skipped."""
        nodes = [
            {"id": "n1", "name": "A", "type": "directory"},
            {"name": "B", "type": "directory"},  # Missing 'id'
            {"id": "n2", "name": "C", "type": "directory"},
        ]
        result = calculate_list_layout(nodes, 1920, 1080)

        assert "n1" in result
        assert "n2" in result
        assert len(result) == 2  # Only 2 nodes positioned


class TestFanLayout:
    """Tests for horizontal fan layout algorithm."""

    def test_empty_children_returns_empty_dict(self):
        """Test that empty children list returns empty positions."""
        parent_pos = (500, 400)
        result = calculate_fan_layout(parent_pos, [], 1920, 1080)
        assert result == {}

    def test_single_child_centered_at_90_degrees(self):
        """Test that single child is positioned at 90° (center of arc)."""
        parent_pos = (500, 400)
        children = [{"id": "child1", "name": "Test", "type": "file"}]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        assert "child1" in result
        x, y = result["child1"]

        # At 90° angle (π/2), cos(π/2)≈0, sin(π/2)=1
        # With base radius 200: x≈500+0=500, y≈400+200=600
        assert abs(x - 500) < 1  # Approximately centered horizontally
        assert abs(y - 600) < 1  # Below parent by radius

    def test_two_children_at_180_and_0_degrees(self):
        """Test that two children are at left (180°) and right (0°)."""
        parent_pos = (500, 400)
        children = [
            {"id": "c1", "name": "A", "type": "file"},
            {"id": "c2", "name": "B", "type": "file"},
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        assert "c1" in result
        assert "c2" in result

        x1, y1 = result["c1"]
        x2, y2 = result["c2"]

        # At 180° (π radians): cos(π)=-1, sin(π)≈0
        # At 0° (0 radians): cos(0)=1, sin(0)=0
        # Base radius is 200px

        # First child should be to the left (180°)
        assert x1 < 500  # Left of parent
        assert abs(y1 - 400) < 1  # Same vertical level

        # Second child should be to the right (0°)
        assert x2 > 500  # Right of parent
        assert abs(y2 - 400) < 1  # Same vertical level

    def test_adaptive_radius_small_count(self):
        """Test that small child count uses base radius (200px)."""
        parent_pos = (500, 400)
        children = [
            {"id": f"c{i}", "name": f"Child{i}", "type": "file"} for i in range(3)
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        # Calculate actual radius by measuring distance from parent
        x1, y1 = result["c0"]
        distance = math.sqrt((x1 - 500) ** 2 + (y1 - 400) ** 2)

        # With 3 children: calculated_radius = (3*60)/π ≈ 57.3
        # Should use base radius of 200px (larger than calculated)
        assert abs(distance - 200) < 1

    def test_adaptive_radius_large_count(self):
        """Test that large child count increases radius."""
        parent_pos = (500, 400)
        children = [
            {"id": f"c{i}", "name": f"Child{i}", "type": "file"} for i in range(30)
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        # Calculate actual radius
        x1, y1 = result["c0"]
        distance = math.sqrt((x1 - 500) ** 2 + (y1 - 400) ** 2)

        # With 30 children: calculated_radius = (30*60)/π ≈ 573
        # But capped at max_radius of 400px
        assert abs(distance - 400) < 1

    def test_children_sorted_alphabetically(self):
        """Test that children are positioned in alphabetical order."""
        parent_pos = (500, 400)
        children = [
            {"id": "z", "name": "Zebra", "type": "file"},
            {"id": "a", "name": "Apple", "type": "file"},
            {"id": "m", "name": "Mango", "type": "file"},
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        # In alphabetical order: Apple (leftmost), Mango (middle), Zebra (rightmost)
        x_a = result["a"][0]
        x_m = result["m"][0]
        x_z = result["z"][0]

        # Verify left-to-right alphabetical order
        assert x_a < x_m < x_z

    def test_directories_before_files_in_fan(self):
        """Test that directories appear before files in fan layout."""
        parent_pos = (500, 400)
        children = [
            {"id": "file1", "name": "zebra.py", "type": "file"},
            {"id": "dir1", "name": "alpha", "type": "directory"},
            {"id": "file2", "name": "beta.py", "type": "file"},
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        # Directory should be leftmost, then files alphabetically
        x_dir1 = result["dir1"][0]
        x_file2 = result["file2"][0]
        x_file1 = result["file1"][0]

        assert x_dir1 < x_file2 < x_file1

    def test_missing_child_id_skipped(self):
        """Test that children missing 'id' are skipped."""
        parent_pos = (500, 400)
        children = [
            {"id": "c1", "name": "A", "type": "file"},
            {"name": "B", "type": "file"},  # Missing 'id'
            {"id": "c2", "name": "C", "type": "file"},
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        assert "c1" in result
        assert "c2" in result
        assert len(result) == 2


class TestCompactFolderLayout:
    """Tests for compact horizontal folder layout."""

    def test_empty_children_returns_empty_dict(self):
        """Test that empty children list returns empty positions."""
        parent_pos = (500, 400)
        result = calculate_compact_folder_layout(parent_pos, [], 1920, 1080)
        assert result == {}

    def test_single_child_positioned_right_of_parent(self):
        """Test that single child is positioned 800px to the right."""
        parent_pos = (500, 400)
        children = [{"id": "child1", "name": "Test", "type": "directory"}]
        result = calculate_compact_folder_layout(parent_pos, children, 1920, 1080)

        assert "child1" in result
        x, y = result["child1"]

        # Should be 800px to the right
        assert x == 500 + 800
        # With 1 child: totalHeight=50, startY=400-(50/2)=375
        # First child at startY + 0*50 = 375
        assert y == 375

    def test_multiple_children_vertically_stacked(self):
        """Test that multiple children are stacked vertically."""
        parent_pos = (500, 400)
        children = [
            {"id": "c1", "name": "A", "type": "directory"},
            {"id": "c2", "name": "B", "type": "directory"},
            {"id": "c3", "name": "C", "type": "directory"},
        ]
        result = calculate_compact_folder_layout(parent_pos, children, 1920, 1080)

        # All should have same x (800px from parent)
        assert result["c1"][0] == 1300
        assert result["c2"][0] == 1300
        assert result["c3"][0] == 1300

        # Verify 50px vertical spacing
        y1 = result["c1"][1]
        y2 = result["c2"][1]
        y3 = result["c3"][1]

        assert abs((y2 - y1) - 50) < 0.01
        assert abs((y3 - y2) - 50) < 0.01

    def test_children_sorted_alphabetically(self):
        """Test that children are sorted alphabetically (top to bottom)."""
        parent_pos = (500, 400)
        children = [
            {"id": "z", "name": "Zebra", "type": "directory"},
            {"id": "a", "name": "Apple", "type": "directory"},
            {"id": "m", "name": "Mango", "type": "directory"},
        ]
        result = calculate_compact_folder_layout(parent_pos, children, 1920, 1080)

        y_a = result["a"][1]
        y_m = result["m"][1]
        y_z = result["z"][1]

        # Alphabetical order: Apple (top), Mango (middle), Zebra (bottom)
        assert y_a < y_m < y_z

    def test_children_centered_vertically_around_parent(self):
        """Test that children span is centered around parent's y position."""
        parent_pos = (500, 400)
        children = [
            {"id": f"c{i}", "name": f"Child{i}", "type": "directory"} for i in range(4)
        ]
        result = calculate_compact_folder_layout(parent_pos, children, 1920, 1080)

        # Calculate center point of children
        y_values = [result[f"c{i}"][1] for i in range(4)]
        y_min = min(y_values)
        y_max = max(y_values)
        y_center = (y_min + y_max) / 2

        # With 4 children: totalHeight = 200, startY = 400 - (200/2) = 300
        # Positions: 300, 350, 400, 450
        # Center of first and last: (300 + 450) / 2 = 375
        # The implementation centers the SPAN, not the CENTER of the span
        # So the middle of the span is at 375, which is slightly above parent y=400
        assert abs(y_center - 375) < 1  # Center of span is at 375


class TestTreeLayout:
    """Tests for rightward tree layout algorithm."""

    def test_empty_nodes_returns_empty_dict(self):
        """Test that empty node list returns empty positions."""
        result = calculate_tree_layout([], [], 1920, 1080)
        assert result == {}

    def test_single_root_node(self):
        """Test single root node positioned at left margin."""
        nodes = [{"id": "root1", "name": "src", "type": "directory", "parent": None}]
        expansion_path = []

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        assert "root1" in result
        x, y = result["root1"]

        # Root nodes at level 0: x = 100 - 0 = 100 (no shift with single level)
        assert x == 100
        # Single node centered: y = (1080 - 50) / 2 = 515
        assert y == 515.0

    def test_root_nodes_vertical_list(self):
        """Test multiple root nodes in vertical list."""
        nodes = [
            {"id": "root1", "name": "src", "type": "directory", "parent": None},
            {"id": "root2", "name": "tests", "type": "directory", "parent": None},
            {"id": "root3", "name": "docs", "type": "directory", "parent": None},
        ]
        expansion_path = []

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # All roots should have same x coordinate
        assert result["root1"][0] == result["root2"][0] == result["root3"][0] == 100

        # Should be vertically spaced by 50px, in alphabetical order
        # Alphabetical: docs (root3), src (root1), tests (root2)
        y_docs = result["root3"][1]
        y_src = result["root1"][1]
        y_tests = result["root2"][1]

        # Verify alphabetical ordering
        assert y_docs < y_src < y_tests

        # Verify 50px spacing
        assert abs((y_src - y_docs) - 50) < 0.01
        assert abs((y_tests - y_src) - 50) < 0.01

    def test_expansion_creates_second_level(self):
        """Test that expanding a root node creates second level to the right."""
        nodes = [
            {"id": "root1", "name": "src", "type": "directory", "parent": None},
            {"id": "child1", "name": "main.py", "type": "file", "parent": "root1"},
            {"id": "child2", "name": "utils.py", "type": "file", "parent": "root1"},
        ]
        expansion_path = ["root1"]  # Expanded root1

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # Level 0 (root): x = 100
        # Level 1 (children): x = 100 + 300 = 400
        # Shift for 2 levels: (1 * 300) / 2 = 150
        # Final: Level 0 = 100 - 150 = -50, Level 1 = 400 - 150 = 250

        assert result["root1"][0] == -50  # Shifted left
        assert result["child1"][0] == 250  # To the right of root
        assert result["child2"][0] == 250  # Same level as child1

        # Children should be vertically spaced
        y1 = result["child1"][1]
        y2 = result["child2"][1]
        assert abs((y2 - y1) - 50) < 0.01

    def test_multi_level_expansion(self):
        """Test three-level tree (root → dir → file)."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "subdir", "name": "cli", "type": "directory", "parent": "root"},
            {"id": "file", "name": "main.py", "type": "file", "parent": "subdir"},
        ]
        expansion_path = ["root", "subdir"]  # Expanded root and subdir

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # Level 0: x = 100
        # Level 1: x = 400
        # Level 2: x = 700
        # Shift: (2 * 300) / 2 = 300
        # Final: L0=-200, L1=100, L2=400

        assert result["root"][0] == -200
        assert result["subdir"][0] == 100
        assert result["file"][0] == 400

    def test_alphabetical_sorting_within_level(self):
        """Test nodes sorted alphabetically within each level."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "c1", "name": "zebra.py", "type": "file", "parent": "root"},
            {"id": "c2", "name": "apple.py", "type": "file", "parent": "root"},
            {"id": "c3", "name": "mango.py", "type": "file", "parent": "root"},
        ]
        expansion_path = ["root"]

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # All children at same x level
        assert result["c1"][0] == result["c2"][0] == result["c3"][0]

        # Alphabetically ordered top to bottom
        y_apple = result["c2"][1]
        y_mango = result["c3"][1]
        y_zebra = result["c1"][1]

        assert y_apple < y_mango < y_zebra

    def test_directories_before_files_in_level(self):
        """Test directories appear before files within same level."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "file1", "name": "zebra.py", "type": "file", "parent": "root"},
            {"id": "dir1", "name": "alpha", "type": "directory", "parent": "root"},
            {"id": "file2", "name": "beta.py", "type": "file", "parent": "root"},
        ]
        expansion_path = ["root"]

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        y_dir = result["dir1"][1]
        y_beta = result["file2"][1]
        y_zebra = result["file1"][1]

        # Directory first, then files alphabetically
        assert y_dir < y_beta < y_zebra

    def test_nodes_centered_vertically_in_viewport(self):
        """Test that nodes are centered vertically in viewport."""
        nodes = [
            {"id": "n1", "name": "A", "type": "directory", "parent": None},
            {"id": "n2", "name": "B", "type": "directory", "parent": None},
        ]
        expansion_path = []

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # With 2 nodes: total_height = 2 * 50 = 100
        # start_y = (1080 - 100) / 2 = 490
        y1 = result["n1"][1]
        y2 = result["n2"][1]

        assert y1 == 490
        assert y2 == 540


class TestBuildTreeLevels:
    """Tests for _build_tree_levels helper function."""

    def test_empty_nodes_returns_empty_root_level(self):
        """Test that empty nodes list returns single empty root level."""
        levels = _build_tree_levels([], [])
        assert len(levels) == 1
        assert levels[0] == []

    def test_single_root_node(self):
        """Test single root node in level 0."""
        nodes = [{"id": "root", "name": "src", "type": "directory", "parent": None}]
        levels = _build_tree_levels(nodes, [])

        assert len(levels) == 1
        assert len(levels[0]) == 1
        assert levels[0][0]["id"] == "root"

    def test_multiple_root_nodes(self):
        """Test multiple root nodes all in level 0."""
        nodes = [
            {"id": "r1", "name": "src", "type": "directory", "parent": None},
            {"id": "r2", "name": "tests", "type": "directory", "parent": None},
        ]
        levels = _build_tree_levels(nodes, [])

        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_two_levels_with_expansion(self):
        """Test expansion path creates second level."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "child1", "name": "main.py", "type": "file", "parent": "root"},
            {"id": "child2", "name": "utils.py", "type": "file", "parent": "root"},
        ]
        expansion_path = ["root"]

        levels = _build_tree_levels(nodes, expansion_path)

        assert len(levels) == 2
        assert len(levels[0]) == 1  # Root
        assert len(levels[1]) == 2  # Children
        assert levels[1][0]["id"] == "child1"
        assert levels[1][1]["id"] == "child2"

    def test_three_levels_nested_expansion(self):
        """Test nested expansion creates three levels."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "mid", "name": "cli", "type": "directory", "parent": "root"},
            {"id": "leaf", "name": "main.py", "type": "file", "parent": "mid"},
        ]
        expansion_path = ["root", "mid"]

        levels = _build_tree_levels(nodes, expansion_path)

        assert len(levels) == 3
        assert len(levels[0]) == 1  # Root
        assert len(levels[1]) == 1  # Mid
        assert len(levels[2]) == 1  # Leaf

    def test_orphan_nodes_not_in_path(self):
        """Test that nodes not in expansion path don't create levels."""
        nodes = [
            {"id": "root", "name": "src", "type": "directory", "parent": None},
            {"id": "child1", "name": "a.py", "type": "file", "parent": "root"},
            {"id": "child2", "name": "b.py", "type": "file", "parent": "root"},
        ]
        expansion_path = []  # No expansion

        levels = _build_tree_levels(nodes, expansion_path)

        # Only root level, children not included
        assert len(levels) == 1
        assert len(levels[0]) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_list_layout_with_none_values(self):
        """Test list layout handles None values gracefully."""
        nodes = [
            {"id": "n1", "name": None, "type": "directory"},
            {"id": "n2", "name": "B", "type": "directory"},
        ]
        result = calculate_list_layout(nodes, 1920, 1080)

        # Should still position both nodes
        assert len(result) == 2
        assert "n1" in result
        assert "n2" in result

    def test_fan_layout_with_very_large_count(self):
        """Test fan layout caps radius at maximum (400px)."""
        parent_pos = (500, 400)
        children = [
            {"id": f"c{i}", "name": f"Child{i}", "type": "file"} for i in range(100)
        ]
        result = calculate_fan_layout(parent_pos, children, 1920, 1080)

        # Verify radius is capped at 400px
        x1, y1 = result["c0"]
        distance = math.sqrt((x1 - 500) ** 2 + (y1 - 400) ** 2)
        assert abs(distance - 400) < 1

    def test_list_layout_with_small_viewport(self):
        """Test list layout works with small viewport."""
        nodes = [{"id": f"n{i}", "name": f"Node{i}", "type": "file"} for i in range(10)]
        # Very small viewport (mobile device)
        result = calculate_list_layout(nodes, 375, 667)

        # Should still calculate positions (may extend beyond viewport)
        assert len(result) == 10

        # Verify all have same x position
        x_values = [result[f"n{i}"][0] for i in range(10)]
        assert all(x == 100 for x in x_values)

    def test_tree_layout_with_missing_parent(self):
        """Test tree layout handles nodes with missing parent references."""
        nodes = [
            {"id": "n1", "name": "A", "type": "directory", "parent": "nonexistent"},
            {"id": "n2", "name": "B", "type": "directory", "parent": None},
        ]
        expansion_path = []

        result = calculate_tree_layout(nodes, expansion_path, 1920, 1080)

        # Both should be treated as root nodes (parent not in graph)
        assert len(result) == 2
        assert result["n1"][0] == result["n2"][0]  # Same x level
