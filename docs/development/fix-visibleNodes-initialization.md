# Fix: visibleNodes Initialization Bug for Large Graphs

**Date**: 2025-12-06
**Issue**: Critical bug preventing graph display for graphs with >500 nodes
**Status**: ✅ Fixed

## Problem Description

For graphs with more than 500 nodes, the visualization was completely blank because `visibleNodes` was never initialized. This resulted in Cytoscape receiving zero nodes to render.

### Root Cause

The code had a conditional that skipped `visualizeGraph(data)` for large graphs:

```javascript
// BUGGY CODE (BEFORE FIX)
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ Called with empty visibleNodes
} else {
    visualizeGraph(data);  // ← Only called for small graphs!
}
```

**Problem**: `visualizeGraph(data)` is responsible for:
- Initializing `visibleNodes` with root-level nodes
- Initializing `filteredLinks` with filtered edges
- Setting up the D3 force-directed layout

When this was skipped for large graphs, `visibleNodes.size === 0`, so `switchToCytoscapeLayout('dagre')` had no nodes to render.

## Solution

**Always call `visualizeGraph(data)` first**, then switch layouts for large graphs:

```javascript
// FIXED CODE (AFTER FIX)
// ALWAYS initialize through visualizeGraph first
// This sets up visibleNodes, filteredLinks, and root nodes
visualizeGraph(data);

// Then switch to Dagre for large graphs (if needed)
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ✅ Now has visibleNodes populated
}
```

## Behavior After Fix

### Large Graphs (>500 nodes)
1. `visualizeGraph(data)` runs → Initializes `visibleNodes` with root nodes
2. D3 force-directed layout starts (briefly)
3. `switchToCytoscapeLayout('dagre')` called → Switches to hierarchical Dagre layout
4. Result: **~50 root nodes visible** in hierarchical layout

### Small Graphs (<500 nodes)
1. `visualizeGraph(data)` runs → Initializes `visibleNodes` with root nodes
2. D3 force-directed layout starts and continues
3. No layout switch occurs
4. Result: **All root nodes visible** in force-directed layout

## File Changed

- **File**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
- **Lines**: 2145-2159 (in generated HTML JavaScript)
- **Function**: `get_data_loading_logic()`

## Testing

Tested with 1449-node graph:
- ✅ Root nodes (~50) now visible on initial load
- ✅ Dagre hierarchical layout applied
- ✅ Nodes clickable to expand children
- ✅ No regression for small graphs

## Impact

- **Lines Changed**: 12 lines (reordered logic)
- **Net LOC**: 0 (pure refactoring)
- **Breaking Changes**: None
- **Performance**: No impact (same operations, different order)

## Acceptance Criteria

- ✅ `visualizeGraph(data)` called unconditionally before layout selection
- ✅ `visibleNodes` is initialized for ALL graph sizes
- ✅ Large graphs (>500 nodes) display with Dagre layout
- ✅ Small graphs (<500 nodes) display with force-directed layout
- ✅ No regression in functionality
