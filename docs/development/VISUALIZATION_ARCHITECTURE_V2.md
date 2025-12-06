# MCP Vector Search - Visualization Architecture V2.0

**Design Document for List-Based Hierarchical Navigation**

**Author**: Claude Engineer
**Date**: 2025-12-06
**Status**: Design Review
**Target**: mcp-vector-search visualization module

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Requirements Specification](#requirements-specification)
4. [Data Model Design](#data-model-design)
5. [Layout Engine Architecture](#layout-engine-architecture)
6. [State Management System](#state-management-system)
7. [Interaction Handlers](#interaction-handlers)
8. [Rendering Strategy](#rendering-strategy)
9. [Implementation Plan](#implementation-plan)
10. [Risk Analysis](#risk-analysis)
11. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Purpose
Replace the current force-directed graph visualization with a hierarchical list-based navigation system that allows users to drill down through directories, files, and AST chunks with horizontal fan layouts.

### Key Design Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Vertical list for root** | Alphabetical sorting is intuitive for file systems | Requires more vertical space vs. force layout |
| **Horizontal fan expansion** | Natural metaphor for "opening" a folder | Animation complexity vs. instant display |
| **Sibling exclusivity** | Reduces visual clutter, maintains focus | User must close sibling to see another vs. simultaneous exploration |
| **AST-only relationships** | Explicit code dependencies are actionable | Misses implicit relationships vs. semantic similarity |
| **D3.js for layout** | Already integrated, powerful transitions | Learning curve vs. simpler library |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interaction Layer                   │
│  (Click handlers, breadcrumbs, expand/collapse controls)    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    State Management Layer                    │
│  • Expansion path tracking (array of node IDs)              │
│  • Node visibility (Set<nodeId>)                            │
│  • Layout mode (list | horizontal_fan)                      │
│  • Active sibling tracking                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                      Layout Engine                          │
│  • List Layout: Vertical positioning with fixed spacing     │
│  • Fan Layout: Radial layout from parent node              │
│  • Transition Manager: Smooth animations between states     │
│  • Collision Detector: Overlap prevention                   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    Rendering Layer (D3.js)                   │
│  • SVG node rendering                                        │
│  • Edge drawing (AST calls only)                            │
│  • Animation engine (position transitions)                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Current State Analysis

### Existing Implementation

**File**: `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Current Features**:
- Force-directed layout using D3.js
- Hierarchical expand/collapse (all descendants shown)
- Multiple relationship types (containment, semantic, caller, imports)
- Cytoscape.js for alternate layouts (Dagre, Circle)
- Content pane with code viewer
- Breadcrumb navigation

**Current Data Model** (`visualizeGraph()` function):
```javascript
{
  allNodes: Array<Node>,           // All graph nodes
  allLinks: Array<Link>,           // All relationships
  visibleNodes: Set<nodeId>,       // Currently visible nodes
  collapsedNodes: Set<nodeId>,     // Nodes that have children but are collapsed
  highlightedNode: Node | null,    // Currently selected node
  rootNodes: Array<Node>,          // Top-level entry points
  edgeFilters: {                   // Relationship visibility toggles
    containment: true,
    calls: true,
    imports: false,
    semantic: false,
    cycles: true
  }
}
```

**Current Node Structure**:
```typescript
interface Node {
  id: string;
  name: string;
  type: 'directory' | 'file' | 'function' | 'class' | 'method' | 'subproject';
  file_path: string;
  start_line?: number;
  end_line?: number;
  complexity?: number;
  depth: number;
  parent_id?: string;
  parent_dir_id?: string;
  content?: string;
  docstring?: string;
  language?: string;
  callers?: Array<CallerInfo>;
  x?: number;  // D3 position
  y?: number;  // D3 position
}
```

**Current Link Structure**:
```typescript
interface Link {
  source: string | Node;
  target: string | Node;
  type: 'dir_containment' | 'dir_hierarchy' | 'file_containment' | 'caller' | 'semantic' | 'imports' | 'dependency';
  is_cycle?: boolean;
  similarity?: number;  // For semantic links
}
```

### Problems with Current Approach

1. **Visual Clutter**: Force-directed layout shows all descendants simultaneously
2. **No Hierarchical Control**: Cannot explore siblings independently
3. **Relationship Overload**: Semantic/import links add noise for code navigation
4. **Spatial Confusion**: Nodes move during force simulation
5. **No List View**: Cannot see alphabetical directory listing

---

## Requirements Specification

### Functional Requirements

#### FR-1: List View (Root Level)
- **Display**: Vertical list of all root-level directories and files
- **Sorting**: Alphabetical order (directories first, then files)
- **Layout**: Fixed vertical spacing, left-aligned
- **Interaction**: Click to expand

**Acceptance Criteria**:
- [ ] Root nodes displayed in vertical column
- [ ] Alphabetical sorting maintained
- [ ] Directories shown before files at same depth
- [ ] Click triggers expansion without moving other nodes
- [ ] Smooth scroll to view all nodes

#### FR-2: Directory Expansion (Horizontal Fan)
- **Trigger**: Click on directory node
- **Layout**: Horizontal radial fan centered on parent
- **Content**: Direct children only (files + subdirectories)
- **Exclusivity**: Opening sibling closes previously opened path
- **Collapse**: Click "-" button to close

**Acceptance Criteria**:
- [ ] Children appear in horizontal arc from parent
- [ ] Fan radius adapts to child count
- [ ] Sibling directories mutually exclusive
- [ ] Collapse button clearly visible
- [ ] Smooth open/close animation

#### FR-3: File Expansion (AST Chunks)
- **Trigger**: Click on file node
- **Layout**: Horizontal fan showing functions/classes
- **Content**: AST chunks from that file only
- **Relationships**: Show function call edges ONLY
- **Filter**: No semantic, import, or containment edges

**Acceptance Criteria**:
- [ ] AST chunks displayed in fan layout
- [ ] Only actual function calls shown as edges
- [ ] No implicit relationships rendered
- [ ] Chunk type icons clearly visible
- [ ] Click chunk to view code

#### FR-4: File Viewer (Unchanged)
- **Preserve**: Current content pane implementation
- **Navigation**: Breadcrumb, back/forward buttons
- **Content**: Code display with syntax highlighting

**Acceptance Criteria**:
- [ ] Existing content pane works with new layout
- [ ] Breadcrumbs reflect expansion path
- [ ] Code viewer unchanged

### Non-Functional Requirements

#### NFR-1: Performance
- **Rendering**: <100ms per expand/collapse action
- **Animation**: 60fps smooth transitions
- **Large Directories**: Handle 500+ children in fan

#### NFR-2: Usability
- **Discoverability**: Expandable nodes clearly indicated
- **Feedback**: Immediate visual response to clicks
- **Consistency**: Predictable navigation patterns

#### NFR-3: Accessibility
- **Keyboard**: Tab navigation, Enter to expand
- **Screen Readers**: ARIA labels on all interactive elements
- **Contrast**: WCAG AA compliant colors

---

## Data Model Design

### New State Properties

```typescript
interface VisualizationState {
  // Existing (preserved)
  allNodes: Array<Node>;
  allLinks: Array<Link>;
  highlightedNode: Node | null;

  // New/Modified
  expansionPath: Array<string>;        // Stack of expanded node IDs (root to current)
  visibleNodes: Set<string>;           // Nodes currently visible (incl. in-fan nodes)
  nodeStates: Map<string, NodeState>;  // Per-node metadata
  layoutMode: 'list' | 'fan';          // Current layout type
  activeSibling: string | null;        // Currently expanded sibling at each depth
}

interface NodeState {
  id: string;
  isExpanded: boolean;              // Has children shown?
  isInFan: boolean;                 // Currently in a horizontal fan?
  layoutOverride?: LayoutPosition;  // Fixed position for list/fan layout
  depth: number;                    // Distance from root
  parentId: string | null;          // Parent node ID
  childrenIds: Array<string>;       // Direct children
}

interface LayoutPosition {
  x: number;
  y: number;
  fixed: boolean;  // True = don't apply force simulation
}

interface ExpansionPathEntry {
  nodeId: string;
  nodeType: 'directory' | 'file';
  children: Array<string>;  // IDs of visible children
}
```

### Data Model Modifications

**No changes required to**:
- `Node` structure from `graph_builder.py`
- `Link` structure from `graph_builder.py`
- Backend data generation

**Changes required in JavaScript**:
```javascript
// New global state variables
let expansionPath = [];          // NEW: Track expansion stack
let nodeStates = new Map();      // NEW: Per-node state
let layoutMode = 'list';         // NEW: Current layout type
let activeSibling = null;        // NEW: Current expanded sibling

// Modified existing variables
let visibleNodes = new Set();    // MODIFIED: Now includes fan nodes
let collapsedNodes = new Set();  // MODIFIED: Now unused (replaced by nodeStates)
```

### State Transitions

```
┌──────────────────────────────────────────────────────────────┐
│  INITIAL STATE: List View                                    │
│  • expansionPath = []                                        │
│  • visibleNodes = {root_nodes}                               │
│  • layoutMode = 'list'                                       │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Click directory node D1
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  DIRECTORY EXPANDED: Horizontal Fan                          │
│  • expansionPath = [D1]                                      │
│  • visibleNodes = {root_nodes, D1.children}                  │
│  • layoutMode = 'fan'                                        │
│  • activeSibling = D1                                        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Click sibling directory D2
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  SIBLING SWITCH: Close D1, Open D2                           │
│  • expansionPath = [D2]                                      │
│  • visibleNodes = {root_nodes, D2.children}                  │
│  • D1.children removed from visibleNodes                     │
│  • activeSibling = D2                                        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Click file F1 in D2
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  FILE EXPANDED: AST Fan                                      │
│  • expansionPath = [D2, F1]                                  │
│  • visibleNodes = {root_nodes, D2.children, F1.ast_chunks}   │
│  • Show ONLY caller edges between F1.ast_chunks              │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Click collapse "-" on D2
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  COLLAPSED: Back to List View                                │
│  • expansionPath = []                                        │
│  • visibleNodes = {root_nodes}                               │
│  • layoutMode = 'list'                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Layout Engine Architecture

### Layout Algorithms

#### Algorithm 1: Vertical List Layout

**Purpose**: Position root-level nodes in alphabetical vertical column

**Input**:
- `nodes`: Array of root nodes
- `containerWidth`: SVG viewport width
- `containerHeight`: SVG viewport height

**Output**:
- `positions`: Map<nodeId, {x, y}>

**Algorithm**:
```python
def calculate_list_layout(nodes, containerWidth, containerHeight):
    """
    Position nodes in vertical list with fixed spacing.

    Time Complexity: O(n) where n = number of nodes
    Space Complexity: O(n) for position map
    """
    # Sort alphabetically (directories first)
    sorted_nodes = sorted(nodes, key=lambda n: (
        0 if n.type == 'directory' else 1,
        n.name.lower()
    ))

    # Calculate spacing
    node_height = 50  # Height occupied by each node (icon + label)
    total_height = len(sorted_nodes) * node_height
    start_y = (containerHeight - total_height) / 2  # Center vertically
    x_position = 100  # Left margin

    positions = {}
    for i, node in enumerate(sorted_nodes):
        positions[node.id] = {
            'x': x_position,
            'y': start_y + (i * node_height),
            'fixed': True  # Don't apply force simulation
        }

    return positions
```

**Pseudocode**:
```
FUNCTION calculateListLayout(nodes, width, height):
    sortedNodes = SORT nodes BY (type=='directory' ? 0 : 1, name.lowercase)

    nodeHeight = 50
    totalHeight = COUNT(sortedNodes) * nodeHeight
    startY = (height - totalHeight) / 2
    xPosition = 100

    positions = EMPTY_MAP

    FOR each node, index IN sortedNodes:
        positions[node.id] = {
            x: xPosition,
            y: startY + (index * nodeHeight),
            fixed: true
        }

    RETURN positions
END
```

#### Algorithm 2: Horizontal Fan Layout

**Purpose**: Arrange children in horizontal arc from parent node

**Input**:
- `parentNode`: Node being expanded
- `childNodes`: Array of direct children
- `containerWidth`: SVG viewport width
- `containerHeight`: SVG viewport height

**Output**:
- `positions`: Map<nodeId, {x, y}>

**Algorithm**:
```python
import math

def calculate_fan_layout(parentNode, childNodes, containerWidth, containerHeight):
    """
    Arrange children in horizontal fan pattern from parent.

    Time Complexity: O(n) where n = number of children
    Space Complexity: O(n) for position map

    Design Decision: Horizontal fan for intuitive left-to-right reading

    Rationale: Western reading patterns favor horizontal layout over vertical.
    Children spread left-to-right in arc centered on parent.

    Trade-offs:
    - Horizontal space: Limited by viewport width vs. vertical scroll
    - Readability: Left-to-right natural vs. radial all directions
    - Density: 180° arc vs. 360° circle (more space per node)

    Alternatives Considered:
    1. Full circle (360°): Rejected - confusing orientation
    2. Vertical fan: Rejected - conflicts with list view scrolling
    3. Grid layout: Rejected - loses parent-child visual connection

    Extension Points: Arc angle can be parameterized (currently 180°)
    for different density preferences.
    """
    parent_x = parentNode.x or containerWidth * 0.3
    parent_y = parentNode.y or containerHeight / 2

    # Calculate adaptive radius based on child count
    base_radius = 200
    spacing_per_child = 60  # Horizontal space needed per child
    calculated_radius = (len(childNodes) * spacing_per_child) / math.pi
    radius = max(base_radius, min(calculated_radius, 400))

    # Horizontal fan: 180° arc (left to right)
    start_angle = math.pi  # Start at left (180°)
    end_angle = 0          # End at right (0°)
    angle_range = start_angle - end_angle

    positions = {}

    # Sort children (directories first, then alphabetical)
    sorted_children = sorted(childNodes, key=lambda n: (
        0 if n.type == 'directory' else 1,
        n.name.lower()
    ))

    for i, child in enumerate(sorted_children):
        # Distribute evenly across arc
        if len(sorted_children) == 1:
            angle = math.pi / 2  # Center if only one child
        else:
            angle = start_angle - (i / (len(sorted_children) - 1)) * angle_range

        # Calculate position on arc
        x = parent_x + radius * math.cos(angle)
        y = parent_y + radius * math.sin(angle)

        positions[child.id] = {
            'x': x,
            'y': y,
            'fixed': True
        }

    return positions
```

**Pseudocode**:
```
FUNCTION calculateFanLayout(parent, children, width, height):
    parentX = parent.x OR (width * 0.3)
    parentY = parent.y OR (height / 2)

    # Adaptive radius
    baseRadius = 200
    spacingPerChild = 60
    calculatedRadius = (COUNT(children) * spacingPerChild) / PI
    radius = MAX(baseRadius, MIN(calculatedRadius, 400))

    # Horizontal arc: 180 degrees (π radians)
    startAngle = π   # Left side
    endAngle = 0     # Right side

    sortedChildren = SORT children BY (type=='directory' ? 0 : 1, name.lowercase)

    positions = EMPTY_MAP

    FOR each child, index IN sortedChildren:
        IF COUNT(sortedChildren) == 1:
            angle = π / 2  # Center
        ELSE:
            angle = startAngle - (index / (COUNT(sortedChildren) - 1)) * (startAngle - endAngle)

        x = parentX + radius * COS(angle)
        y = parentY + radius * SIN(angle)

        positions[child.id] = {x: x, y: y, fixed: true}

    RETURN positions
END
```

#### Algorithm 3: Transition Animation

**Purpose**: Smoothly animate nodes between layouts

**Input**:
- `nodes`: Nodes to animate
- `oldPositions`: Previous layout positions
- `newPositions`: Target layout positions
- `duration`: Animation time (ms)

**Output**:
- Animated SVG elements

**Algorithm**:
```javascript
function animateLayoutTransition(nodes, oldPositions, newPositions, duration = 750) {
    """
    Animate nodes from old positions to new positions.

    Time Complexity: O(n) where n = number of nodes
    Space Complexity: O(1) - uses D3 transitions

    Performance:
    - Expected FPS: 60fps (16.67ms per frame)
    - Animation uses requestAnimationFrame (browser-optimized)
    - D3 easing: cubic-in-out for smooth acceleration/deceleration
    """
    // Select all nodes that need animation
    const selection = d3.selectAll('.node')
        .filter(d => newPositions.has(d.id));

    // Apply D3 transition
    selection.transition()
        .duration(duration)
        .ease(d3.easeCubicInOut)
        .attr('transform', d => {
            const pos = newPositions.get(d.id);
            return `translate(${pos.x}, ${pos.y})`;
        });

    // Also animate connected edges
    const affectedLinks = allLinks.filter(link =>
        newPositions.has(link.source.id) || newPositions.has(link.target.id)
    );

    d3.selectAll('.link')
        .filter(l => affectedLinks.includes(l))
        .transition()
        .duration(duration)
        .ease(d3.easeCubicInOut)
        .attr('x1', d => newPositions.get(d.source.id)?.x || d.source.x)
        .attr('y1', d => newPositions.get(d.source.id)?.y || d.source.y)
        .attr('x2', d => newPositions.get(d.target.id)?.x || d.target.x)
        .attr('y2', d => newPositions.get(d.target.id)?.y || d.target.y);
}
```

**Error Handling**:
- **Missing positions**: Fall back to current position (no animation)
- **Invalid duration**: Clamp to [100ms, 2000ms]
- **Concurrent transitions**: Cancel previous, start new

### Collision Detection

**Problem**: Horizontal fan nodes may overlap with other UI elements

**Solution**: Bounding box collision detection with repositioning

```javascript
function detectAndResolveCollisions(positions, containerWidth, containerHeight) {
    """
    Detect overlapping nodes and adjust positions.

    Time Complexity: O(n²) naive, O(n log n) with spatial indexing
    Space Complexity: O(n) for collision grid

    Optimization: Use quad-tree for large node counts (>1000 nodes)
    """
    const nodeRadius = 30;  // Approximate node size
    const minDistance = nodeRadius * 2 + 10;  // Minimum safe distance

    // Build spatial grid for faster collision detection
    const gridSize = 100;
    const grid = new Map();

    for (const [nodeId, pos] of positions.entries()) {
        const gridX = Math.floor(pos.x / gridSize);
        const gridY = Math.floor(pos.y / gridSize);
        const key = `${gridX},${gridY}`;

        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push({nodeId, pos});
    }

    // Check collisions only within same or adjacent grid cells
    const adjustedPositions = new Map(positions);

    for (const [nodeId, pos] of positions.entries()) {
        const gridX = Math.floor(pos.x / gridSize);
        const gridY = Math.floor(pos.y / gridSize);

        // Check 3x3 grid around node
        for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
                const key = `${gridX + dx},${gridY + dy}`;
                const neighbors = grid.get(key) || [];

                for (const neighbor of neighbors) {
                    if (neighbor.nodeId === nodeId) continue;

                    const dx = pos.x - neighbor.pos.x;
                    const dy = pos.y - neighbor.pos.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < minDistance) {
                        // Push apart
                        const angle = Math.atan2(dy, dx);
                        const pushDistance = (minDistance - distance) / 2;

                        adjustedPositions.get(nodeId).x += pushDistance * Math.cos(angle);
                        adjustedPositions.get(nodeId).y += pushDistance * Math.sin(angle);
                    }
                }
            }
        }
    }

    return adjustedPositions;
}
```

---

## State Management System

### State Variables

```javascript
// Global state management
const VisualizationState = {
    // Core data (unchanged)
    allNodes: [],
    allLinks: [],

    // Expansion tracking (NEW)
    expansionPath: [],           // Array of expanded node IDs
    nodeStates: new Map(),       // nodeId -> NodeState

    // Visibility (MODIFIED)
    visibleNodes: new Set(),     // Includes list + fan nodes

    // Layout (NEW)
    layoutMode: 'list',          // 'list' | 'fan'
    layoutPositions: new Map(),  // nodeId -> {x, y, fixed}

    // Selection (unchanged)
    highlightedNode: null,

    // Sibling management (NEW)
    activeSiblings: new Map(),   // depth -> currently expanded nodeId

    // Relationship filtering (MODIFIED - simplified)
    edgeFilters: {
        ast_calls: true,         // Function calls within files
        containment: true,       // Directory/file hierarchy
        // REMOVED: semantic, imports (not shown in this mode)
    }
};

// Node state tracking
class NodeStateManager {
    constructor() {
        this.states = new Map();
    }

    getState(nodeId) {
        if (!this.states.has(nodeId)) {
            this.states.set(nodeId, {
                isExpanded: false,
                isInFan: false,
                layoutOverride: null,
                depth: 0,
                parentId: null,
                childrenIds: []
            });
        }
        return this.states.get(nodeId);
    }

    setExpanded(nodeId, expanded) {
        const state = this.getState(nodeId);
        state.isExpanded = expanded;
    }

    isExpanded(nodeId) {
        return this.getState(nodeId).isExpanded;
    }

    setInFan(nodeId, inFan) {
        const state = this.getState(nodeId);
        state.isInFan = inFan;
    }

    isInFan(nodeId) {
        return this.getState(nodeId).isInFan;
    }
}

const nodeStateManager = new NodeStateManager();
```

### State Update Functions

```javascript
/**
 * Expand a directory or file node.
 *
 * Design Decision: Sibling exclusivity at each depth level
 *
 * When expanding node N at depth D:
 * 1. Check if another sibling is expanded at depth D
 * 2. If yes, collapse that sibling first
 * 3. Then expand N and show its children in fan layout
 *
 * This ensures only one expansion path is visible at a time,
 * reducing visual complexity.
 */
function expandNode(nodeId) {
    const node = allNodes.find(n => n.id === nodeId);
    if (!node) return;

    const depth = node.depth || 0;

    // Check for active sibling at this depth
    const activeSibling = VisualizationState.activeSiblings.get(depth);
    if (activeSibling && activeSibling !== nodeId) {
        // Close the sibling first
        collapseNode(activeSibling);
    }

    // Mark this node as expanded
    nodeStateManager.setExpanded(nodeId, true);
    VisualizationState.activeSiblings.set(depth, nodeId);

    // Add to expansion path
    VisualizationState.expansionPath.push(nodeId);

    // Find direct children
    const children = allLinks
        .filter(link =>
            (link.source.id || link.source) === nodeId &&
            link.type === 'dir_containment' || link.type === 'file_containment'
        )
        .map(link => {
            const childId = link.target.id || link.target;
            return allNodes.find(n => n.id === childId);
        })
        .filter(n => n);

    // Add children to visible set
    children.forEach(child => {
        VisualizationState.visibleNodes.add(child.id);
        nodeStateManager.setInFan(child.id, true);
    });

    // Calculate fan layout
    const fanPositions = calculateFanLayout(
        node,
        children,
        width,
        height
    );

    // Store layout positions
    fanPositions.forEach((pos, childId) => {
        VisualizationState.layoutPositions.set(childId, pos);
    });

    // Update layout mode
    VisualizationState.layoutMode = 'fan';

    // Trigger re-render with animation
    renderGraphWithTransition();
}

/**
 * Collapse a node and hide all its descendants.
 */
function collapseNode(nodeId) {
    const node = allNodes.find(n => n.id === nodeId);
    if (!node) return;

    // Mark as collapsed
    nodeStateManager.setExpanded(nodeId, false);

    // Remove from expansion path
    const pathIndex = VisualizationState.expansionPath.indexOf(nodeId);
    if (pathIndex !== -1) {
        VisualizationState.expansionPath.splice(pathIndex);
    }

    // Remove from active siblings
    const depth = node.depth || 0;
    if (VisualizationState.activeSiblings.get(depth) === nodeId) {
        VisualizationState.activeSiblings.delete(depth);
    }

    // Hide all descendants recursively
    function hideDescendants(parentId) {
        const children = allLinks
            .filter(link => (link.source.id || link.source) === parentId)
            .map(link => link.target.id || link.target);

        children.forEach(childId => {
            VisualizationState.visibleNodes.delete(childId);
            nodeStateManager.setInFan(childId, false);
            nodeStateManager.setExpanded(childId, false);
            VisualizationState.layoutPositions.delete(childId);

            // Recurse to grandchildren
            hideDescendants(childId);
        });
    }

    hideDescendants(nodeId);

    // If no nodes expanded, return to list mode
    if (VisualizationState.expansionPath.length === 0) {
        VisualizationState.layoutMode = 'list';
    }

    // Trigger re-render with animation
    renderGraphWithTransition();
}

/**
 * Reset to initial list view.
 */
function resetToListView() {
    // Collapse all nodes
    VisualizationState.expansionPath.forEach(nodeId => {
        nodeStateManager.setExpanded(nodeId, false);
    });

    // Clear state
    VisualizationState.expansionPath = [];
    VisualizationState.activeSiblings.clear();
    VisualizationState.visibleNodes = new Set(rootNodes.map(n => n.id));
    VisualizationState.layoutMode = 'list';
    VisualizationState.layoutPositions.clear();

    // Calculate list layout for root nodes
    const listPositions = calculateListLayout(rootNodes, width, height);
    listPositions.forEach((pos, nodeId) => {
        VisualizationState.layoutPositions.set(nodeId, pos);
    });

    // Clear selection
    VisualizationState.highlightedNode = null;

    // Close content pane
    closeContentPane();

    // Re-render
    renderGraphWithTransition();
}
```

---

## Interaction Handlers

### Click Handler

```javascript
/**
 * Handle node click events.
 *
 * Behavior:
 * - Directory: Expand/collapse with horizontal fan
 * - File: Expand/collapse AST chunks in horizontal fan
 * - AST Chunk: Show in content pane, no expansion
 */
function handleNodeClick(event, nodeData) {
    event.stopPropagation();

    const node = allNodes.find(n => n.id === nodeData.id);
    if (!node) return;

    // Always show content pane
    showContentPane(node);

    // Handle expansion based on node type
    if (node.type === 'directory' || node.type === 'file') {
        const isExpanded = nodeStateManager.isExpanded(node.id);

        if (isExpanded) {
            collapseNode(node.id);
        } else {
            expandNode(node.id);
        }
    }
    // AST chunks (function, class, method) don't expand
}
```

### Breadcrumb Navigation

```javascript
/**
 * Generate breadcrumb navigation from expansion path.
 *
 * Shows: Root > Dir1 > Dir2 > File
 * Each segment is clickable to navigate back.
 */
function generateBreadcrumbs() {
    const breadcrumbs = ['<span class="breadcrumb-root" onclick="resetToListView()">🏠 Root</span>'];

    VisualizationState.expansionPath.forEach((nodeId, index) => {
        const node = allNodes.find(n => n.id === nodeId);
        if (!node) return;

        const isLast = (index === VisualizationState.expansionPath.length - 1);

        breadcrumbs.push(' / ');

        if (isLast) {
            // Current node: not clickable, highlighted
            breadcrumbs.push(`<span class="breadcrumb-current">${escapeHtml(node.name)}</span>`);
        } else {
            // Parent nodes: clickable
            breadcrumbs.push(
                `<span class="breadcrumb-link" onclick="navigateToNodeInPath('${node.id}')">` +
                `${escapeHtml(node.name)}</span>`
            );
        }
    });

    return breadcrumbs.join('');
}

/**
 * Navigate back to a node in the expansion path.
 * Collapses all descendants of that node.
 */
function navigateToNodeInPath(nodeId) {
    const pathIndex = VisualizationState.expansionPath.indexOf(nodeId);
    if (pathIndex === -1) return;

    // Collapse all nodes after this one in the path
    const nodesToCollapse = VisualizationState.expansionPath.slice(pathIndex + 1);
    nodesToCollapse.forEach(id => collapseNode(id));

    // Show the node in content pane
    const node = allNodes.find(n => n.id === nodeId);
    if (node) {
        showContentPane(node);
    }
}
```

### Keyboard Shortcuts

```javascript
/**
 * Keyboard navigation handlers.
 */
document.addEventListener('keydown', (event) => {
    // Escape: Close content pane and collapse current node
    if (event.key === 'Escape') {
        event.preventDefault();
        if (VisualizationState.expansionPath.length > 0) {
            const currentNode = VisualizationState.expansionPath[
                VisualizationState.expansionPath.length - 1
            ];
            collapseNode(currentNode);
        }
        closeContentPane();
    }

    // Backspace: Navigate up one level
    if (event.key === 'Backspace' && !isTyping()) {
        event.preventDefault();
        if (VisualizationState.expansionPath.length > 0) {
            const currentNode = VisualizationState.expansionPath.pop();
            collapseNode(currentNode);
        }
    }

    // Home: Reset to list view
    if (event.key === 'Home') {
        event.preventDefault();
        resetToListView();
    }
});

function isTyping() {
    const activeElement = document.activeElement;
    return activeElement && (
        activeElement.tagName === 'INPUT' ||
        activeElement.tagName === 'TEXTAREA' ||
        activeElement.isContentEditable
    );
}
```

---

## Rendering Strategy

### Rendering Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  1. STATE UPDATE                                             │
│     • User clicks node                                       │
│     • expandNode() or collapseNode() called                 │
│     • State variables updated                                │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  2. LAYOUT CALCULATION                                       │
│     • Determine layout mode (list or fan)                   │
│     • Calculate positions for all visible nodes              │
│     • Detect and resolve collisions                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  3. EDGE FILTERING                                           │
│     • Filter edges based on visible nodes                   │
│     • Apply edge type filters (AST calls only)              │
│     • Remove containment edges in fan view                  │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  4. D3 RENDERING                                             │
│     • Update node selection (enter/update/exit)             │
│     • Update link selection                                  │
│     • Apply transitions (750ms animation)                   │
│     • Update expand/collapse indicators                     │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  5. POST-RENDER                                              │
│     • Update breadcrumbs                                     │
│     • Update stats display                                   │
│     • Zoom to fit if needed                                 │
└──────────────────────────────────────────────────────────────┘
```

### Main Render Function

```javascript
/**
 * Main rendering function with transition animation.
 *
 * Performance:
 * - Time Complexity: O(n + m) where n=nodes, m=links
 * - Expected render time: <100ms for 1000 nodes
 */
function renderGraphWithTransition(duration = 750) {
    // 1. Get visible nodes
    const visibleNodesList = Array.from(VisualizationState.visibleNodes)
        .map(id => allNodes.find(n => n.id === id))
        .filter(n => n);

    // 2. Calculate layout positions
    const positions = new Map();

    if (VisualizationState.layoutMode === 'list') {
        // List layout for root nodes
        const listPos = calculateListLayout(visibleNodesList, width, height);
        listPos.forEach((pos, nodeId) => positions.set(nodeId, pos));
    } else if (VisualizationState.layoutMode === 'fan') {
        // Use cached fan positions
        VisualizationState.layoutPositions.forEach((pos, nodeId) => {
            positions.set(nodeId, pos);
        });
    }

    // 3. Filter edges
    const visibleLinks = getFilteredLinksForCurrentView();

    // 4. D3 rendering

    // --- NODES ---
    const nodeSelection = g.selectAll('.node')
        .data(visibleNodesList, d => d.id);

    // ENTER: New nodes
    const nodeEnter = nodeSelection.enter()
        .append('g')
        .attr('class', d => `node ${d.type}`)
        .attr('transform', d => {
            // Start at parent position or center
            const pos = positions.get(d.id);
            return `translate(${pos?.x || width/2}, ${pos?.y || height/2})`;
        })
        .on('click', handleNodeClick)
        .on('mouseover', showTooltip)
        .on('mouseout', hideTooltip);

    // Add node visuals (icons, labels, etc.)
    addNodeVisuals(nodeEnter);

    // UPDATE: Existing nodes with transition
    nodeSelection.transition()
        .duration(duration)
        .attr('transform', d => {
            const pos = positions.get(d.id);
            return `translate(${pos?.x || d.x}, ${pos?.y || d.y})`;
        });

    // Update expand/collapse indicators
    nodeSelection.selectAll('.expand-indicator')
        .text(d => {
            if (!hasChildren(d)) return '';
            return nodeStateManager.isExpanded(d.id) ? '−' : '+';
        });

    // EXIT: Remove nodes
    nodeSelection.exit()
        .transition()
        .duration(duration)
        .style('opacity', 0)
        .remove();

    // --- LINKS ---
    const linkSelection = g.selectAll('.link')
        .data(visibleLinks, d => `${d.source.id}-${d.target.id}`);

    // ENTER: New links
    linkSelection.enter()
        .append('line')
        .attr('class', d => `link ${d.type}`)
        .attr('x1', d => positions.get(d.source.id)?.x || d.source.x)
        .attr('y1', d => positions.get(d.source.id)?.y || d.source.y)
        .attr('x2', d => positions.get(d.target.id)?.x || d.target.x)
        .attr('y2', d => positions.get(d.target.id)?.y || d.target.y)
        .style('opacity', 0)
        .transition()
        .duration(duration)
        .style('opacity', 1);

    // UPDATE: Existing links
    linkSelection.transition()
        .duration(duration)
        .attr('x1', d => positions.get(d.source.id)?.x || d.source.x)
        .attr('y1', d => positions.get(d.source.id)?.y || d.source.y)
        .attr('x2', d => positions.get(d.target.id)?.x || d.target.x)
        .attr('y2', d => positions.get(d.target.id)?.y || d.target.y);

    // EXIT: Remove links
    linkSelection.exit()
        .transition()
        .duration(duration)
        .style('opacity', 0)
        .remove();

    // 5. Post-render updates
    updateBreadcrumbs();
    updateStats();
}

/**
 * Filter links for current view mode.
 *
 * In fan view: Show ONLY AST call relationships within expanded file
 * In list view: Show ONLY containment hierarchy (no edges)
 */
function getFilteredLinksForCurrentView() {
    if (VisualizationState.layoutMode === 'list') {
        // List view: No edges (just nodes)
        return [];
    }

    if (VisualizationState.layoutMode === 'fan') {
        // Fan view: Check what's expanded
        const expandedFile = VisualizationState.expansionPath.find(nodeId => {
            const node = allNodes.find(n => n.id === nodeId);
            return node && node.type === 'file';
        });

        if (expandedFile) {
            // Show only AST calls within this file
            return allLinks.filter(link => {
                // Must be caller relationship
                if (link.type !== 'caller') return false;

                // Both source and target must be AST chunks of the expanded file
                const source = allNodes.find(n => n.id === (link.source.id || link.source));
                const target = allNodes.find(n => n.id === (link.target.id || link.target));

                return source && target &&
                       source.file_path === target.file_path &&
                       source.file_path === allNodes.find(n => n.id === expandedFile)?.file_path;
            });
        } else {
            // Directory fan: No edges (just containment implied by layout)
            return [];
        }
    }

    return [];
}
```

### Visual Styling

```css
/* List view nodes */
.node.directory {
    cursor: pointer;
}

.node.file {
    cursor: pointer;
}

/* Fan view nodes */
.node.in-fan {
    /* Highlighted visual for nodes in fan layout */
    filter: drop-shadow(0 0 4px rgba(88, 166, 255, 0.6));
}

/* Expand/collapse indicator */
.expand-indicator {
    font-size: 18px;
    font-weight: bold;
    fill: #ffffff;
    pointer-events: none;
    user-select: none;
}

/* AST call edges (in fan view) */
.link.caller {
    stroke: #58a6ff;
    stroke-width: 2px;
    marker-end: url(#arrowhead);
}

/* Breadcrumb navigation */
.breadcrumb-nav {
    padding: 12px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    font-size: 14px;
    color: #c9d1d9;
}

.breadcrumb-link {
    color: #58a6ff;
    cursor: pointer;
    text-decoration: none;
}

.breadcrumb-link:hover {
    text-decoration: underline;
}

.breadcrumb-current {
    color: #ffffff;
    font-weight: 600;
}

.breadcrumb-separator {
    color: #8b949e;
    margin: 0 8px;
}
```

---

## Implementation Plan

### Phase 1: Data Model & State Management (Week 1)

**Goal**: Implement new state management without changing rendering

**Tasks**:
1. ✅ Add `expansionPath`, `nodeStates`, `layoutMode` to global state
2. ✅ Implement `NodeStateManager` class
3. ✅ Add `activeSiblings` tracking
4. ✅ Write unit tests for state transitions
5. ✅ Update `visualizeGraph()` to initialize new state variables

**Files to Modify**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Add new state variables in `get_d3_initialization()`
  - Add `NodeStateManager` class in new function `get_state_management()`

**Files to Create**:
- `/tests/manual/test_state_management.html` (test harness)

**Success Criteria**:
- State transitions work without rendering
- Expansion path correctly tracks hierarchy
- Sibling exclusivity logic validated

**Estimated LOC**: +150 lines (state management functions)

---

### Phase 2: Layout Algorithms (Week 2)

**Goal**: Implement list and fan layout calculations

**Tasks**:
1. ✅ Implement `calculateListLayout()` function
2. ✅ Implement `calculateFanLayout()` function
3. ✅ Implement `detectAndResolveCollisions()` (optional, if needed)
4. ✅ Write layout algorithm tests with known inputs/outputs
5. ✅ Performance testing with 500+ nodes

**Files to Modify**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Add `get_layout_algorithms()` function with both algorithms

**Files to Create**:
- `/tests/manual/test_layouts.html` (visual layout tester)
- `/docs/development/layout_algorithm_tests.md` (test cases)

**Success Criteria**:
- List layout produces evenly spaced vertical column
- Fan layout produces horizontal arc from parent
- Layouts complete in <50ms for 100 nodes
- No overlapping nodes in either layout

**Estimated LOC**: +200 lines (layout algorithms)

---

### Phase 3: Interaction Handlers (Week 3)

**Goal**: Wire up click handlers and expand/collapse logic

**Tasks**:
1. ✅ Modify `handleNodeClick()` to use new expansion logic
2. ✅ Implement `expandNode()` function
3. ✅ Implement `collapseNode()` function
4. ✅ Implement `resetToListView()` function
5. ✅ Update breadcrumb generation for expansion path
6. ✅ Add keyboard shortcuts (Escape, Backspace, Home)

**Files to Modify**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Replace `get_interaction_handlers()` with new implementation
  - Update `get_breadcrumb_functions()` for expansion path

**Success Criteria**:
- Clicking directory expands horizontally
- Clicking sibling closes previous sibling
- Breadcrumbs reflect expansion path
- Keyboard shortcuts work

**Estimated LOC**: +150 lines, -50 lines (replace old handlers) = **+100 net**

---

### Phase 4: Rendering Integration (Week 4)

**Goal**: Integrate layouts with D3 rendering and animations

**Tasks**:
1. ✅ Implement `renderGraphWithTransition()` function
2. ✅ Implement `animateLayoutTransition()` function
3. ✅ Update `getFilteredLinksForCurrentView()` for AST-only edges
4. ✅ Add visual indicators for expand/collapse state
5. ✅ Test animations for smoothness (60fps)

**Files to Modify**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Replace `renderGraph()` with `renderGraphWithTransition()`
  - Update `get_graph_visualization_functions()`

**Success Criteria**:
- Smooth transitions between layouts (750ms)
- 60fps animation performance
- Nodes appear/disappear with fade effect
- Edges update smoothly

**Estimated LOC**: +250 lines, -100 lines (replace renderGraph) = **+150 net**

---

### Phase 5: Edge Filtering & AST Calls (Week 5)

**Goal**: Show only function call relationships within files

**Tasks**:
1. ✅ Filter edges to show ONLY `type: 'caller'` in file fan view
2. ✅ Ensure both source and target are in same file
3. ✅ Hide all semantic, import, and containment edges in fan view
4. ✅ Update edge styling for AST calls (blue, with arrows)
5. ✅ Test with files containing circular function calls

**Files to Modify**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Update `getFilteredLinksForCurrentView()` function

**Backend Verification**:
- Ensure `graph_builder.py` correctly generates `caller` edges with AST extraction

**Success Criteria**:
- Only function calls shown in file fan
- No semantic or import edges visible
- Circular calls clearly indicated
- Performance: <50ms edge filtering for 1000 edges

**Estimated LOC**: +50 lines (edge filtering logic)

---

### Phase 6: Testing & Polish (Week 6)

**Goal**: Comprehensive testing and UX improvements

**Tasks**:
1. ✅ User acceptance testing (UAT) with real codebases
2. ✅ Performance testing with large monorepos (10K+ files)
3. ✅ Accessibility testing (keyboard nav, screen readers)
4. ✅ Cross-browser testing (Chrome, Firefox, Safari)
5. ✅ Bug fixes and refinements
6. ✅ Documentation updates

**Files to Create**:
- `/docs/guides/VISUALIZATION_USAGE.md` (user guide)
- `/tests/manual/visualization_uat_checklist.md`

**Success Criteria**:
- No critical bugs in UAT
- Performance: <100ms expand/collapse for 500 nodes
- Keyboard navigation fully functional
- Works in all major browsers

**Estimated LOC**: +100 lines (bug fixes, polish)

---

### Summary: Total Implementation

| Phase | Duration | LOC Impact | Risk |
|-------|----------|------------|------|
| 1. State Management | 1 week | +150 | Low |
| 2. Layout Algorithms | 1 week | +200 | Medium |
| 3. Interaction Handlers | 1 week | +100 | Low |
| 4. Rendering Integration | 1 week | +150 | High |
| 5. Edge Filtering | 1 week | +50 | Low |
| 6. Testing & Polish | 1 week | +100 | Medium |
| **TOTAL** | **6 weeks** | **+750 lines** | **Medium** |

**Net LOC Impact**: +750 new, -150 removed = **+600 net lines**

**Files Modified**: 1 primary file (`scripts.py`)
**Files Created**: 5 test/documentation files

---

## Risk Analysis

### High-Risk Areas

#### Risk 1: Animation Performance

**Description**: Smooth 60fps animations with 500+ nodes may be challenging

**Probability**: Medium (40%)
**Impact**: High (poor UX if choppy)

**Mitigation**:
1. Use CSS transforms instead of D3 transitions where possible (GPU-accelerated)
2. Batch DOM updates with `requestAnimationFrame`
3. Implement progressive rendering (render in chunks)
4. Add performance mode that disables animations for large graphs

**Contingency**:
- Fall back to instant layout changes (no animation) if >1000 nodes
- Add user setting to disable animations

#### Risk 2: Complex State Management

**Description**: Managing expansion path, sibling exclusivity, and visibility may introduce bugs

**Probability**: High (60%)
**Impact**: Medium (confusing UI behavior)

**Mitigation**:
1. Comprehensive unit tests for state transitions
2. State validation function to detect inconsistencies
3. Add debug mode that logs state changes
4. Use immutable data structures where possible

**Contingency**:
- Add "Reset View" button to recover from bad states
- Implement state persistence to localStorage for debugging

#### Risk 3: Edge Filtering Complexity

**Description**: Correctly filtering AST call edges within files may miss or include wrong edges

**Probability**: Medium (30%)
**Impact**: Medium (incorrect relationships shown)

**Mitigation**:
1. Use explicit `type: 'caller'` check (not substring matching)
2. Verify source and target file paths match
3. Add visual debugging mode to highlight edge types
4. Backend tests to ensure caller edges are correctly generated

**Contingency**:
- Add edge type toggle to manually show/hide edge types
- Provide user feedback if no edges found (may indicate backend issue)

### Medium-Risk Areas

#### Risk 4: Browser Compatibility

**Description**: D3 transitions or SVG rendering may differ across browsers

**Probability**: Low (20%)
**Impact**: Medium (broken in some browsers)

**Mitigation**:
1. Test in Chrome, Firefox, Safari during development
2. Use D3 v7+ (good browser support)
3. Polyfill for older browsers if needed

**Contingency**:
- Detect browser and show warning if unsupported
- Provide fallback static layout (no animations)

#### Risk 5: Large File AST Chunks

**Description**: Files with 100+ functions may create unreadable fan layout

**Probability**: Medium (30%)
**Impact**: Low (can zoom, but awkward)

**Mitigation**:
1. Adaptive fan radius based on child count
2. Add "Grouped View" that clusters functions by class
3. Implement search/filter within fan

**Contingency**:
- Add "List View" toggle for file AST chunks
- Limit fan to top 50 chunks, show "View All" button

---

## Testing Strategy

### Unit Tests

**Test File**: `/tests/manual/test_visualization_v2.html`

**Test Cases**:

```javascript
describe('State Management', () => {
    test('expandNode adds to expansionPath', () => {
        const state = new VisualizationState();
        expandNode('dir1');
        expect(state.expansionPath).toEqual(['dir1']);
    });

    test('expandNode sibling closes previous sibling', () => {
        expandNode('dir1');
        expandNode('dir2'); // dir1 and dir2 are siblings
        expect(state.expansionPath).toEqual(['dir2']);
        expect(state.visibleNodes.has('dir1-child1')).toBe(false);
    });

    test('collapseNode removes descendants', () => {
        expandNode('dir1');
        expandNode('file1'); // child of dir1
        collapseNode('dir1');
        expect(state.visibleNodes.has('file1')).toBe(false);
    });
});

describe('Layout Algorithms', () => {
    test('calculateListLayout positions nodes vertically', () => {
        const nodes = [
            {id: 'a', name: 'Alpha', type: 'directory'},
            {id: 'b', name: 'Beta', type: 'directory'}
        ];
        const positions = calculateListLayout(nodes, 1000, 800);

        expect(positions.get('a').x).toBe(100); // Fixed x
        expect(positions.get('b').x).toBe(100);
        expect(positions.get('b').y).toBeGreaterThan(positions.get('a').y); // Beta below Alpha
    });

    test('calculateFanLayout creates horizontal arc', () => {
        const parent = {id: 'p', x: 500, y: 400};
        const children = [{id: 'c1'}, {id: 'c2'}, {id: 'c3'}];
        const positions = calculateFanLayout(parent, children, 1000, 800);

        // All children should be on an arc around parent
        positions.forEach((pos, id) => {
            const dx = pos.x - 500;
            const dy = pos.y - 400;
            const distance = Math.sqrt(dx*dx + dy*dy);
            expect(distance).toBeCloseTo(200, 1); // Within 200px radius
        });

        // Leftmost child should have x < parent.x
        expect(positions.get('c1').x).toBeLessThan(500);
        // Rightmost child should have x > parent.x
        expect(positions.get('c3').x).toBeGreaterThan(500);
    });
});

describe('Edge Filtering', () => {
    test('getFilteredLinksForCurrentView shows only AST calls in file fan', () => {
        // Setup: Expand file with AST chunks
        expandNode('file1');

        // Mock links
        const allLinks = [
            {type: 'caller', source: {id: 'func1'}, target: {id: 'func2'}},
            {type: 'semantic', source: {id: 'func1'}, target: {id: 'func3'}},
            {type: 'imports', source: {id: 'func1'}, target: {id: 'external'}}
        ];

        const filtered = getFilteredLinksForCurrentView();

        expect(filtered.length).toBe(1);
        expect(filtered[0].type).toBe('caller');
    });
});
```

### Integration Tests

**Test Scenarios**:

1. **Directory Navigation**:
   - Start at root list view
   - Click directory → verify fan layout
   - Click sibling directory → verify first closes
   - Click collapse → verify return to list

2. **File Exploration**:
   - Expand directory
   - Click file → verify AST chunks shown
   - Verify only function call edges shown
   - Click AST chunk → verify content pane

3. **Breadcrumb Navigation**:
   - Expand: Root > Dir1 > Dir2 > File1
   - Verify breadcrumb shows full path
   - Click Dir1 in breadcrumb → verify collapse to Dir1 level

4. **Keyboard Navigation**:
   - Press Home → verify reset to list
   - Expand nodes, press Escape → verify collapse
   - Press Backspace → verify navigate up one level

### Performance Benchmarks

| Scenario | Node Count | Target Time | Acceptance Threshold |
|----------|------------|-------------|----------------------|
| Initial list render | 100 | <50ms | <100ms |
| Expand directory | 50 children | <100ms | <200ms |
| Expand file (AST) | 100 chunks | <150ms | <300ms |
| Collapse node | 200 descendants | <100ms | <200ms |
| Edge filtering | 1000 edges | <50ms | <100ms |
| Animation (60fps) | 500 nodes | 16.67ms/frame | <33ms/frame |

**Performance Testing Tool**:
```javascript
function benchmarkExpandNode(nodeId, iterations = 100) {
    const times = [];

    for (let i = 0; i < iterations; i++) {
        resetToListView(); // Reset state

        const start = performance.now();
        expandNode(nodeId);
        const end = performance.now();

        times.push(end - start);
    }

    const avg = times.reduce((a, b) => a + b) / times.length;
    const p95 = times.sort()[Math.floor(iterations * 0.95)];

    console.log(`Expand ${nodeId}: avg=${avg.toFixed(2)}ms, p95=${p95.toFixed(2)}ms`);
}
```

### User Acceptance Testing (UAT)

**Test Projects**:
1. Small project (<100 files): mcp-vector-search itself
2. Medium project (100-1000 files): Django project
3. Large monorepo (1000+ files): TypeScript monorepo

**UAT Checklist**:
- [ ] List view shows root directories alphabetically
- [ ] Clicking directory expands horizontally
- [ ] Clicking sibling directory closes previous
- [ ] File expansion shows AST chunks
- [ ] Only function call edges shown in file view
- [ ] Breadcrumbs update correctly
- [ ] Content pane works with new layout
- [ ] Keyboard shortcuts functional
- [ ] Animations smooth (60fps)
- [ ] No visual bugs (overlapping, missing nodes)
- [ ] Performance acceptable on large projects

---

## Appendix A: ASCII Diagrams

### List View Layout

```
Root Level (Vertical List)

┌─────────────────┐  ←─ y=100
│  📁 src         │ +
└─────────────────┘

┌─────────────────┐  ←─ y=150
│  📁 tests       │ +
└─────────────────┘

┌─────────────────┐  ←─ y=200
│  📁 docs        │ +
└─────────────────┘

┌─────────────────┐  ←─ y=250
│  📄 README.md   │
└─────────────────┘

x=100 (all nodes)
spacing=50px
```

### Horizontal Fan Layout

```
Expanded Directory "src"

                ┌─────────────┐
           ┌────│  📁 core    │
           │    └─────────────┘
           │
           │    ┌─────────────┐
           ├────│  📁 cli     │
           │    └─────────────┘
           │
┌──────────┴─┐  ┌─────────────┐
│  📁 src    ├──│  📁 tests   │
│     +      │  └─────────────┘
└──────────┬─┘
           │    ┌─────────────┐
           ├────│ 📄 __init__.py │
           │    └─────────────┘
           │
           │    ┌─────────────┐
           └────│ 📄 main.py  │
                └─────────────┘

radius=200px
arc=180° (horizontal)
children sorted alphabetically
```

### AST Fan Layout (File Expanded)

```
File "main.py" Expanded (AST Chunks)

      ┌──────────┐
  ┌───│ main()   │ ← Entry point
  │   └──────────┘
  │        │ calls
  │        ▼
  │   ┌──────────┐
  ├───│ setup()  │
  │   └──────────┘
  │        │ calls
  │        ▼
┌─┴──────────┐   ┌──────────┐
│ 📄 main.py ├───│ Config   │ ← Class
│     +      │   └──────────┘
└─┬──────────┘        │ calls
  │                   ▼
  │              ┌──────────┐
  └──────────────│ load()   │ ← Method
                 └──────────┘

Only 'caller' edges shown (function calls)
No semantic or import edges
```

---

## Appendix B: File Modification Checklist

### Primary File: `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Functions to Add**:
- [ ] `get_state_management()` → NodeStateManager class + state variables
- [ ] `get_layout_algorithms()` → calculateListLayout, calculateFanLayout
- [ ] `get_expansion_logic()` → expandNode, collapseNode, resetToListView
- [ ] `get_transition_animations()` → renderGraphWithTransition, animateLayoutTransition

**Functions to Modify**:
- [ ] `get_d3_initialization()` → Add new state variables
- [ ] `get_interaction_handlers()` → Update handleNodeClick
- [ ] `get_breadcrumb_functions()` → Use expansionPath instead of file_path
- [ ] `get_graph_visualization_functions()` → Call new render function
- [ ] `get_data_loading_logic()` → Initialize new state on load

**Functions to Remove**:
- [ ] `collapsedNodes` usage (replaced by nodeStates)
- [ ] Force simulation logic for list/fan layouts (keep for other modes)

### Secondary Files

**No changes required to**:
- `/src/mcp_vector_search/cli/commands/visualize/graph_builder.py` (backend data generation)
- `/src/mcp_vector_search/cli/commands/visualize/templates/base.py` (HTML structure)
- `/src/mcp_vector_search/cli/commands/visualize/templates/styles.py` (CSS, minor additions only)

**New Test Files**:
- [ ] `/tests/manual/test_state_management.html`
- [ ] `/tests/manual/test_layouts.html`
- [ ] `/tests/manual/visualization_uat_checklist.md`

**New Documentation**:
- [ ] `/docs/guides/VISUALIZATION_USAGE.md`
- [ ] `/docs/development/LAYOUT_ALGORITHMS.md`

---

## Appendix C: Migration Path (Preserving Existing Functionality)

### Backward Compatibility

**Preserve these features**:
- Force-directed layout (as alternate mode)
- Cytoscape layouts (Dagre, Circle)
- Edge type filters (containment, semantic, imports, cycles)
- Content pane with code viewer
- Search functionality
- File viewer

**Strategy**:
1. Add new layout modes alongside existing force layout
2. Add "Layout Mode" selector: List/Fan (new) vs. Force/Dagre/Circle (existing)
3. Keep all existing functions, add new ones
4. User can toggle between old and new visualizations

**UI Changes**:
```html
<!-- Add to layout controls -->
<div class="layout-mode-selector">
    <label>
        <input type="radio" name="layoutStyle" value="hierarchical" checked>
        Hierarchical (List/Fan)
    </label>
    <label>
        <input type="radio" name="layoutStyle" value="force">
        Force-Directed
    </label>
</div>
```

---

**END OF DESIGN DOCUMENT**

**Next Steps**:
1. Review this design with stakeholders
2. Approve/modify proposed architecture
3. Begin Phase 1 implementation (State Management)
4. Schedule weekly progress reviews

**Questions? Contact**: Claude Engineer (this document's author)
