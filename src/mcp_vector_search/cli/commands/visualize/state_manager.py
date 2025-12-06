"""State management system for visualization V2.0.

This module implements the hierarchical list-based navigation state,
including expansion paths, node visibility, and layout modes.

Design Principles:
    - Sibling Exclusivity: Only one child expanded per depth level
    - List/Fan Modes: Root list view vs. horizontal fan expansion
    - AST-Only Edges: Show only function calls within expanded files
    - Explicit State: No implicit behavior, all state transitions documented

Reference: docs/development/VISUALIZATION_ARCHITECTURE_V2.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class ViewMode(str, Enum):
    """View mode for visualization layout.

    Design Decision: Tree-based view modes for file explorer metaphor

    Rationale: Replaced fan-based modes with tree-based modes to match
    traditional file system navigation (Finder, Explorer). Tree modes
    provide clearer hierarchical relationships and familiar UX patterns.

    Attributes:
        TREE_ROOT: Root list view (vertical alphabetical list, no edges)
        TREE_EXPANDED: Tree with expanded directories (rightward expansion)
        FILE_DETAIL: File with AST chunks (function call edges visible)

    Migration from V1:
        - LIST → TREE_ROOT (same behavior, clearer naming)
        - DIRECTORY_FAN → TREE_EXPANDED (fan arc → tree levels)
        - FILE_FAN → FILE_DETAIL (fan arc → rightward tree + edges)
    """

    TREE_ROOT = "tree_root"  # Vertical list of root nodes, no edges
    TREE_EXPANDED = "tree_expanded"  # Tree with expanded nodes, hierarchical edges
    FILE_DETAIL = "file_detail"  # File with AST chunks and call edges


@dataclass
class NodeState:
    """State of a single node in the visualization.

    Attributes:
        node_id: Unique identifier for the node
        expanded: Whether this node's children are visible
        visible: Whether this node is currently visible
        children_visible: Whether this node's children are visible
        position_override: Optional fixed position for layout
    """

    node_id: str
    expanded: bool = False
    visible: bool = True
    children_visible: bool = False
    position_override: tuple[float, float] | None = None


@dataclass
class VisualizationState:
    """Core state manager for visualization V2.0.

    Manages expansion paths, node visibility, and layout modes using
    explicit state transitions. Enforces sibling exclusivity and
    AST-only edge filtering.

    Design Decision: Sibling Exclusivity
        When expanding a node at depth D, any previously expanded sibling
        at depth D is automatically collapsed. This reduces visual clutter
        and maintains a single focused path through the hierarchy.

    Trade-offs:
        - Simplicity: Only one path visible at a time
        - Focus: Clear navigation context
        - Limitation: Cannot compare siblings side-by-side

    Attributes:
        view_mode: Current view mode (tree_root, tree_expanded, file_detail)
        expansion_path: Ordered list of expanded node IDs (root to current)
        node_states: Map of node ID to NodeState
        visible_edges: Set of visible AST call edges (source_id, target_id)
    """

    view_mode: ViewMode = ViewMode.TREE_ROOT
    expansion_path: list[str] = field(default_factory=list)
    node_states: dict[str, NodeState] = field(default_factory=dict)
    visible_edges: set[tuple[str, str]] = field(default_factory=set)

    def _get_or_create_state(self, node_id: str) -> NodeState:
        """Get or create node state.

        Args:
            node_id: Node identifier

        Returns:
            NodeState for the given node
        """
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeState(node_id=node_id)
        return self.node_states[node_id]

    def expand_node(
        self,
        node_id: str,
        node_type: str,
        children: list[str],
        parent_id: str | None = None,
    ) -> None:
        """Expand a node (directory or file).

        When expanding a node:
        1. Check if a sibling at same depth is already expanded
        2. If yes, collapse that sibling first (sibling exclusivity)
        3. Expand this node and show its children
        4. Update view mode based on node type

        Args:
            node_id: ID of node to expand
            node_type: Type of node ("directory" or "file")
            children: List of child node IDs
            parent_id: Optional parent node ID (for depth calculation)

        Raises:
            ValueError: If node_type is not "directory" or "file"
        """
        if node_type not in ("directory", "file"):
            raise ValueError(f"Cannot expand node type: {node_type}")

        logger.debug(
            f"Expanding {node_type} node: {node_id} with {len(children)} children"
        )

        # Get node state
        node_state = self._get_or_create_state(node_id)

        # Calculate depth (distance from root)
        if parent_id and parent_id in self.expansion_path:
            # If parent is in path, depth is parent_index + 1
            parent_index = self.expansion_path.index(parent_id)
            depth = parent_index + 1
        else:
            # No parent or parent not in path: this is a root-level sibling
            depth = 0

        # Sibling exclusivity: Check if another sibling is expanded at this depth
        if depth < len(self.expansion_path):
            # There's already an expanded node at this depth
            old_sibling = self.expansion_path[depth]
            if old_sibling != node_id:
                logger.debug(
                    f"Sibling exclusivity: Collapsing {old_sibling} "
                    f"before expanding {node_id}"
                )
                # Collapse old path from this depth onward
                nodes_to_collapse = self.expansion_path[depth:]
                self.expansion_path = self.expansion_path[:depth]
                for old_node in nodes_to_collapse:
                    self._collapse_node_internal(old_node)

        # Mark node as expanded
        node_state.expanded = True
        node_state.children_visible = True

        # Add to expansion path
        if node_id not in self.expansion_path:
            self.expansion_path.append(node_id)

        # Make children visible
        for child_id in children:
            child_state = self._get_or_create_state(child_id)
            child_state.visible = True

        # Update view mode based on node type
        if node_type == "directory":
            self.view_mode = ViewMode.TREE_EXPANDED
        elif node_type == "file":
            self.view_mode = ViewMode.FILE_DETAIL

        logger.info(
            f"Expanded {node_type} {node_id}, "
            f"path: {' > '.join(self.expansion_path)}, "
            f"mode: {self.view_mode.value}"
        )

    def _collapse_node_internal(self, node_id: str) -> None:
        """Internal collapse without path manipulation.

        Args:
            node_id: Node to collapse
        """
        node_state = self.node_states.get(node_id)
        if not node_state:
            return

        # Mark as collapsed
        node_state.expanded = False
        node_state.children_visible = False

        logger.debug(f"Collapsed node: {node_id}")

    def collapse_node(self, node_id: str, all_nodes: dict[str, dict]) -> None:
        """Collapse a node and hide all its descendants.

        Recursively hides all descendants of the collapsed node.
        If the expansion path becomes empty, revert to LIST view.

        Args:
            node_id: ID of node to collapse
            all_nodes: Dictionary of all nodes (id -> node_data) for traversal

        Performance:
            Time Complexity: O(d) where d = number of descendants
            Space Complexity: O(d) for recursion stack
        """
        logger.debug(f"Collapsing node: {node_id}")

        # Remove from expansion path
        if node_id in self.expansion_path:
            path_index = self.expansion_path.index(node_id)
            # Also remove all descendants in path
            self.expansion_path = self.expansion_path[:path_index]

        # Mark node as collapsed
        self._collapse_node_internal(node_id)

        # Hide all descendants recursively
        def hide_descendants(parent_id: str) -> None:
            """Recursively hide all descendants.

            Args:
                parent_id: Parent node ID
            """
            # Find children
            node_data = all_nodes.get(parent_id)
            if not node_data:
                return

            # Get children IDs (implementation depends on node structure)
            # This is a placeholder - actual implementation needs to find children
            children = []  # TODO: Extract from node data or links

            for child_id in children:
                child_state = self.node_states.get(child_id)
                if child_state:
                    child_state.visible = False
                    child_state.expanded = False
                    child_state.children_visible = False

                # Recurse
                hide_descendants(child_id)

        hide_descendants(node_id)

        # Update view mode if path is empty
        if len(self.expansion_path) == 0:
            self.view_mode = ViewMode.TREE_ROOT
            logger.info("Collapsed to root, switching to TREE_ROOT view")

    def switch_sibling(
        self,
        old_node_id: str,
        new_node_id: str,
        new_node_type: str,
        new_children: list[str],
        all_nodes: dict[str, dict],
    ) -> None:
        """Close old sibling path, open new sibling path.

        This is a convenience method that combines collapse and expand
        for switching between siblings at the same depth.

        Args:
            old_node_id: ID of currently expanded sibling
            new_node_id: ID of sibling to expand
            new_node_type: Type of new node ("directory" or "file")
            new_children: Children of new node
            all_nodes: Dictionary of all nodes for traversal
        """
        logger.debug(f"Switching sibling: {old_node_id} → {new_node_id}")

        # Collapse old sibling
        self.collapse_node(old_node_id, all_nodes)

        # Expand new sibling
        self.expand_node(new_node_id, new_node_type, new_children)

    def get_visible_nodes(self) -> list[str]:
        """Return list of currently visible node IDs.

        Returns:
            List of node IDs that should be rendered
        """
        visible = []
        for node_id, state in self.node_states.items():
            if state.visible:
                visible.append(node_id)
        return visible

    def get_visible_edges(
        self, all_edges: list[dict], expanded_file_id: str | None = None
    ) -> set[tuple[str, str]]:
        """Return set of visible edges (AST calls only).

        Edge Filtering Rules:
            - TREE_ROOT mode: No edges shown
            - TREE_EXPANDED mode: No edges shown
            - FILE_DETAIL mode: Only AST call edges within expanded file

        Args:
            all_edges: List of all edge dictionaries
            expanded_file_id: ID of currently expanded file (if in FILE_DETAIL mode)

        Returns:
            Set of (source_id, target_id) tuples for visible edges
        """
        if self.view_mode != ViewMode.FILE_DETAIL or not expanded_file_id:
            # No edges in TREE_ROOT or TREE_EXPANDED modes
            return set()

        visible = set()
        for edge in all_edges:
            # Only show "caller" type edges (function calls)
            if edge.get("type") != "caller":
                continue

            source_id = edge.get("source")
            target_id = edge.get("target")

            if not source_id or not target_id:
                continue

            # Both nodes must be visible
            source_state = self.node_states.get(source_id)
            target_state = self.node_states.get(target_id)

            if (
                source_state
                and target_state
                and source_state.visible
                and target_state.visible
            ):
                visible.add((source_id, target_id))

        return visible

    def to_dict(self) -> dict:
        """Serialize state for JavaScript.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            {
                "view_mode": "directory_fan",
                "expansion_path": ["dir1", "dir2"],
                "visible_nodes": ["node1", "node2", "node3"],
                "visible_edges": [["func1", "func2"], ["func2", "func3"]]
            }
        """
        return {
            "view_mode": self.view_mode.value,
            "expansion_path": self.expansion_path.copy(),
            "visible_nodes": self.get_visible_nodes(),
            "visible_edges": [list(edge) for edge in self.visible_edges],
            "node_states": {
                node_id: {
                    "expanded": state.expanded,
                    "visible": state.visible,
                    "children_visible": state.children_visible,
                    "position_override": state.position_override,
                }
                for node_id, state in self.node_states.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> VisualizationState:
        """Deserialize state from JavaScript.

        Args:
            data: Dictionary from JavaScript state (via JSON)

        Returns:
            Reconstructed VisualizationState instance
        """
        state = cls()

        # Handle view mode with backward compatibility
        view_mode_str = data.get("view_mode", "tree_root")

        # Map old view modes to new ones
        view_mode_migration = {
            "list": "tree_root",
            "directory_fan": "tree_expanded",
            "file_fan": "file_detail",
        }

        # Apply migration if needed
        if view_mode_str in view_mode_migration:
            view_mode_str = view_mode_migration[view_mode_str]

        state.view_mode = ViewMode(view_mode_str)
        state.expansion_path = data.get("expansion_path", [])

        # Reconstruct node states
        node_states_data = data.get("node_states", {})
        for node_id, node_data in node_states_data.items():
            state.node_states[node_id] = NodeState(
                node_id=node_id,
                expanded=node_data.get("expanded", False),
                visible=node_data.get("visible", True),
                children_visible=node_data.get("children_visible", False),
                position_override=node_data.get("position_override"),
            )

        # Reconstruct visible edges
        visible_edges_data = data.get("visible_edges", [])
        state.visible_edges = {tuple(edge) for edge in visible_edges_data}

        return state
