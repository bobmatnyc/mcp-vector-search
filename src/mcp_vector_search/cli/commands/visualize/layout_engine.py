"""Layout calculation algorithms for visualization V2.0.

This module implements layout algorithms for hierarchical list-based navigation:
    - List Layout: Vertical alphabetical positioning of root nodes
    - Fan Layout: Horizontal arc (180°) for expanded directory/file children

Design Principles:
    - Deterministic: Same input → same output (no randomness)
    - Adaptive: Radius and spacing adjust to child count
    - Sorted: Alphabetical order (directories first) for predictability
    - Performance: O(n) time complexity, <50ms for 100 nodes

Reference: docs/development/VISUALIZATION_ARCHITECTURE_V2.md
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger


def calculate_list_layout(
    nodes: list[dict[str, Any]], canvas_width: int, canvas_height: int
) -> dict[str, tuple[float, float]]:
    """Calculate vertical list positions for nodes.

    Positions nodes in a vertical list with fixed spacing, sorted
    alphabetically with directories appearing before files.

    Design Decision: Vertical list for alphabetical browsing

    Rationale: Users expect alphabetical file/folder listings (like Finder, Explorer).
    Vertical layout mirrors familiar file manager interfaces.

    Trade-offs:
        - Familiarity: Matches OS file managers vs. novel layouts
        - Vertical space: Requires scrolling for many files vs. compact layouts
        - Readability: One item per line is clearest vs. dense grids

    Args:
        nodes: List of node dictionaries with 'id', 'name', and 'type' keys
        canvas_width: SVG viewport width in pixels
        canvas_height: SVG viewport height in pixels

    Returns:
        Dictionary mapping node_id -> (x, y) position

    Time Complexity: O(n log n) where n = number of nodes (due to sorting)
    Space Complexity: O(n) for position dictionary

    Example:
        >>> nodes = [
        ...     {'id': 'dir1', 'name': 'src', 'type': 'directory'},
        ...     {'id': 'file1', 'name': 'main.py', 'type': 'file'}
        ... ]
        >>> positions = calculate_list_layout(nodes, 1920, 1080)
        >>> positions['dir1']  # (x, y) coordinates
        (100, 490.0)
        >>> positions['file1']
        (100, 540.0)
    """
    if not nodes:
        logger.debug("No nodes to layout")
        return {}

    # Sort alphabetically (directories first, then files)
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (
            0 if n.get("type") == "directory" else 1,  # Directories first
            (n.get("name") or "").lower(),  # Then alphabetical
        ),
    )

    # Layout parameters
    node_height = 50  # Vertical space per node (icon + label)
    x_position = 100  # Left margin
    total_height = len(sorted_nodes) * node_height

    # Center vertically in viewport
    start_y = (canvas_height - total_height) / 2

    # Calculate positions
    positions = {}
    for i, node in enumerate(sorted_nodes):
        node_id = node.get("id")
        if not node_id:
            logger.warning(f"Node missing 'id': {node}")
            continue

        y_position = start_y + (i * node_height)
        positions[node_id] = (x_position, y_position)

    logger.debug(
        f"List layout: {len(positions)} nodes, "
        f"height={total_height}px, "
        f"start_y={start_y:.1f}"
    )

    return positions


def calculate_fan_layout(
    parent_pos: tuple[float, float],
    children: list[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
    fan_type: str = "horizontal",
) -> dict[str, tuple[float, float]]:
    """Calculate horizontal fan positions for child nodes.

    Arranges children in a 180° arc (horizontal fan) emanating from parent node.
    Radius adapts to child count (200-400px range). Children sorted alphabetically.

    Design Decision: Horizontal fan for directory expansion

    Rationale: Horizontal layout provides intuitive left-to-right navigation,
    matching Western reading patterns. 180° arc provides clear parent-child
    visual connection while maintaining adequate spacing between nodes.

    Trade-offs:
        - Horizontal space: Limited by viewport width vs. vertical fans
        - Readability: Left-to-right natural vs. radial (360°) all directions
        - Density: 180° arc provides more space per node vs. full circle
        - Visual connection: Clear parent→child line vs. grid layout

    Alternatives Considered:
        1. Full circle (360°): Rejected - confusing orientation, no clear "top"
        2. Vertical fan: Rejected - conflicts with list view scrolling
        3. Grid layout: Rejected - loses parent-child visual connection
        4. Tree layout: Rejected - doesn't support sibling switching gracefully

    Extension Points: Arc angle parameterizable (currently 180°) for different
    density preferences. Radius calculation can use different formulas.

    Args:
        parent_pos: (x, y) coordinates of parent node
        children: List of child node dictionaries with 'id', 'name', 'type'
        canvas_width: SVG viewport width in pixels
        canvas_height: SVG viewport height in pixels
        fan_type: Type of fan ("horizontal" for 180° arc)

    Returns:
        Dictionary mapping child_id -> (x, y) position

    Time Complexity: O(n log n) where n = number of children (sorting)
    Space Complexity: O(n) for position dictionary

    Performance:
        - Expected: <10ms for 50 children
        - Tested: <50ms for 500 children

    Example:
        >>> parent = (500, 400)
        >>> children = [
        ...     {'id': 'c1', 'name': 'utils', 'type': 'directory'},
        ...     {'id': 'c2', 'name': 'tests', 'type': 'directory'},
        ...     {'id': 'c3', 'name': 'main.py', 'type': 'file'}
        ... ]
        >>> positions = calculate_fan_layout(parent, children, 1920, 1080)
        >>> # Children positioned in 180° arc from left to right
        >>> len(positions)
        3
    """
    if not children:
        logger.debug("No children to layout in fan")
        return {}

    parent_x, parent_y = parent_pos

    # Calculate adaptive radius based on child count
    # More children → larger radius to prevent overlap
    base_radius = 200  # Minimum radius
    max_radius = 400  # Maximum radius
    spacing_per_child = 60  # Horizontal space needed per child

    # Arc length = radius * angle (in radians)
    # For 180° arc: arc_length = radius * π
    # We want: arc_length >= num_children * spacing_per_child
    # Therefore: radius >= (num_children * spacing_per_child) / π
    calculated_radius = (len(children) * spacing_per_child) / math.pi
    radius = max(base_radius, min(calculated_radius, max_radius))

    # Horizontal fan: 180° arc from left to right
    # Start at π radians (180°, left side)
    # End at 0 radians (0°, right side)
    start_angle = math.pi  # Left (180°)
    end_angle = 0  # Right (0°)
    angle_range = start_angle - end_angle  # π radians

    # Sort children (directories first, then alphabetical)
    sorted_children = sorted(
        children,
        key=lambda n: (
            0 if n.get("type") == "directory" else 1,  # Directories first
            (n.get("name") or "").lower(),  # Then alphabetical
        ),
    )

    # Calculate positions
    positions = {}
    num_children = len(sorted_children)

    for i, child in enumerate(sorted_children):
        child_id = child.get("id")
        if not child_id:
            logger.warning(f"Child node missing 'id': {child}")
            continue

        # Calculate angle for this child
        if num_children == 1:
            # Single child: center of arc (90°)
            angle = math.pi / 2
        else:
            # Distribute evenly across arc
            # angle = start_angle - (progress * angle_range)
            # progress = i / (num_children - 1) goes from 0 to 1
            progress = i / (num_children - 1)
            angle = start_angle - (progress * angle_range)

        # Convert polar coordinates (radius, angle) to cartesian (x, y)
        x = parent_x + radius * math.cos(angle)
        y = parent_y + radius * math.sin(angle)

        positions[child_id] = (x, y)

    logger.debug(
        f"Fan layout: {len(positions)} children, "
        f"radius={radius:.1f}px, "
        f"arc={math.degrees(angle_range):.0f}°"
    )

    return positions


def calculate_compact_folder_layout(
    parent_pos: tuple[float, float],
    children: list[dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
) -> dict[str, tuple[float, float]]:
    """Calculate compact horizontal folder layout for directory children.

    Arranges children horizontally in a straight line to the right of parent,
    with fixed 800px spacing. Designed for directory-only views where vertical
    space is limited.

    This layout is optimized for horizontal directory navigation without
    the arc geometry of fan layout.

    Args:
        parent_pos: (x, y) coordinates of parent node
        children: List of child node dictionaries with 'id', 'name', 'type'
        canvas_width: SVG viewport width in pixels
        canvas_height: SVG viewport height in pixels

    Returns:
        Dictionary mapping child_id -> (x, y) position

    Time Complexity: O(n log n) where n = number of children (sorting)
    Space Complexity: O(n) for position dictionary
    """
    if not children:
        logger.debug("No children to layout in compact folder mode")
        return {}

    parent_x, parent_y = parent_pos

    # Compact folder layout parameters
    horizontal_offset = 800  # Fixed horizontal spacing from parent
    vertical_spacing = 50  # Vertical spacing between children

    # Sort children (directories first, then alphabetical)
    sorted_children = sorted(
        children,
        key=lambda n: (
            0 if n.get("type") == "directory" else 1,
            (n.get("name") or "").lower(),
        ),
    )

    # Calculate vertical centering
    total_height = len(sorted_children) * vertical_spacing
    start_y = parent_y - (total_height / 2)

    # Calculate positions
    positions = {}
    for i, child in enumerate(sorted_children):
        child_id = child.get("id")
        if not child_id:
            logger.warning(f"Child node missing 'id': {child}")
            continue

        x = parent_x + horizontal_offset
        y = start_y + (i * vertical_spacing)

        positions[child_id] = (x, y)

    logger.debug(
        f"Compact folder layout: {len(positions)} children, "
        f"offset={horizontal_offset}px, "
        f"spacing={vertical_spacing}px"
    )

    return positions


def calculate_tree_layout(
    nodes: list[dict[str, Any]],
    expansion_path: list[str],
    canvas_width: int,
    canvas_height: int,
    level_spacing: int = 300,
    node_spacing: int = 50,
) -> dict[str, tuple[float, float]]:
    """Calculate positions for rightward tree layout.

    Design Decision: Rightward Tree Layout

    Rationale: Traditional file explorer tree layout matches user expectations
    for hierarchical file system navigation. Nodes expand horizontally to the
    right (like Finder/Explorer) rather than radially, providing clear depth
    visualization and natural left-to-right reading flow.

    Trade-offs:
        - Familiarity: Matches OS file explorers vs. novel graph layouts
        - Depth visibility: Clear level separation vs. radial compactness
        - Horizontal space: Requires wider viewport vs. vertical layouts
        - Simplicity: No angle calculations vs. fan layouts

    Args:
        nodes: All nodes in the graph
        expansion_path: List of expanded node IDs (root → current)
        canvas_width: Canvas width
        canvas_height: Canvas height
        level_spacing: Horizontal distance between tree levels (default 300px)
        node_spacing: Vertical distance between sibling nodes (default 50px)

    Returns:
        Dict mapping node_id → (x, y) position

    Layout Logic:
        - Root nodes: x=100, y=vertical list (50px spacing)
        - Level 1 (children of expanded root): x=400, y=vertical under parent
        - Level 2: x=700, etc.
        - Center view by shifting all x positions left as tree grows

    Time Complexity: O(n log n) where n = number of nodes (sorting)
    Space Complexity: O(n) for position dictionary and level grouping

    Example:
        >>> nodes = [
        ...     {'id': 'root1', 'name': 'src', 'type': 'directory', 'parent': None},
        ...     {'id': 'child1', 'name': 'main.py', 'type': 'file', 'parent': 'root1'}
        ... ]
        >>> expansion_path = ['root1']
        >>> positions = calculate_tree_layout(nodes, expansion_path, 1920, 1080)
        >>> # root1 at x=100, child1 at x=400
    """
    if not nodes:
        logger.debug("No nodes to layout in tree")
        return {}

    positions = {}

    # Group nodes by depth level based on expansion path
    levels = _build_tree_levels(nodes, expansion_path)

    if not levels:
        logger.warning("No levels built from expansion path")
        return {}

    # Calculate positions level by level
    for level_index, level_nodes in enumerate(levels):
        x = 100 + (level_index * level_spacing)

        # Sort alphabetically (directories first)
        sorted_nodes = sorted(
            level_nodes,
            key=lambda n: (
                0 if n.get("type") == "directory" else 1,
                (n.get("name") or "").lower(),
            ),
        )

        # Position vertically
        total_height = len(sorted_nodes) * node_spacing
        start_y = (canvas_height - total_height) / 2

        for i, node in enumerate(sorted_nodes):
            node_id = node.get("id")
            if not node_id:
                logger.warning(f"Node missing 'id': {node}")
                continue

            y = start_y + (i * node_spacing)
            positions[node_id] = (x, y)

    # Center the view (shift left based on deepest level)
    max_level = len(levels) - 1
    shift_x = (max_level * level_spacing) // 2

    for node_id in positions:
        x, y = positions[node_id]
        positions[node_id] = (x - shift_x, y)

    logger.debug(
        f"Tree layout: {len(positions)} nodes across {len(levels)} levels, "
        f"shift_x={shift_x}px"
    )

    return positions


def _build_tree_levels(
    nodes: list[dict[str, Any]], expansion_path: list[str]
) -> list[list[dict[str, Any]]]:
    """Build tree levels based on expansion path.

    Returns:
        List of levels, where each level is a list of nodes
        Level 0: Root nodes (no parent)
        Level 1: Children of expanded root node
        Level 2: Children of expanded level-1 node
        etc.

    Example:
        >>> nodes = [
        ...     {'id': 'a', 'parent': None},
        ...     {'id': 'b', 'parent': 'a'},
        ...     {'id': 'c', 'parent': 'a'}
        ... ]
        >>> expansion_path = ['a']
        >>> levels = _build_tree_levels(nodes, expansion_path)
        >>> len(levels)
        2
        >>> len(levels[0])  # Root level
        1
        >>> len(levels[1])  # Children of 'a'
        2
    """
    levels: list[list[dict[str, Any]]] = []

    # Build node ID to node map for quick lookup
    node_map = {node.get("id"): node for node in nodes if node.get("id")}

    # Level 0: Root nodes (no parent or parent not in graph)
    root_nodes = [
        node
        for node in nodes
        if not node.get("parent") or node.get("parent") not in node_map
    ]
    levels.append(root_nodes)

    # Subsequent levels: Children of expanded nodes in expansion path
    for expanded_node_id in expansion_path:
        # Find children of this expanded node
        children = [node for node in nodes if node.get("parent") == expanded_node_id]

        if children:
            levels.append(children)

    logger.debug(
        f"Built {len(levels)} tree levels from expansion path: {expansion_path}"
    )

    return levels
