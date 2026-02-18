# Bug Fix: Progressive Loading Node Expansion

## Issue
When clicking on nodes in the main visualizer (tree view), sub-nodes weren't loading. The progressive loading was fetching data from the backend but the UI wasn't displaying the expanded children.

## Root Cause
The problem was in the `buildTreeStructure()` function in `scripts.py`. When a node was clicked:

1. ✅ `handleNodeClick()` correctly called `expandNode(nodeId)`
2. ✅ `expandNode()` correctly fetched new nodes and links from `/api/graph-expand/{nodeId}`
3. ✅ New nodes and links were added to global `allNodes` and `allLinks` arrays
4. ✅ `buildTreeStructure()` was called to rebuild the tree
5. ❌ **BUG**: `collapseAll()` collapsed ALL nodes including the one that was just expanded

The issue was that after fetching new children, the tree rebuilding process would collapse everything again, requiring the user to click **twice** - once to fetch data and once again to actually see it.

## Solution
Modified `buildTreeStructure()` to preserve the expansion state of nodes that were previously expanded:

### Changes Made

1. **Track Expanded Nodes**: The `expandedNodes` Set tracks which nodes should remain expanded

2. **Preserve Expansion State in Tree Rebuild**: Replaced `collapseAll()` with `collapseAllExceptExpanded()`:
   - Collapses all nodes by default (initial state)
   - Keeps nodes in `expandedNodes` set expanded (preserves user interaction)
   - Recursively applies to entire tree

3. **Sync Manual Toggle with Expansion State**: When users manually toggle nodes:
   - Collapsing a node removes it from `expandedNodes` → stays collapsed on rebuild
   - Expanding a node adds it to `expandedNodes` → stays expanded on rebuild
   - Applies to directories, files, and code chunks

### Code Changes

**File**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

#### Change 1: New collapse function
```javascript
// OLD: Collapsed everything unconditionally
function collapseAll(node) {
    if (node.children && node.children.length > 0) {
        node.children.forEach(child => collapseAll(child));
        node._children = node.children;
        node.children = null;
    }
}

// NEW: Preserves expanded nodes
function collapseAllExceptExpanded(node) {
    if (node.children && node.children.length > 0) {
        node.children.forEach(child => collapseAllExceptExpanded(child));

        // Only collapse if NOT in expandedNodes set
        if (!expandedNodes.has(node.id)) {
            node._children = node.children;
            node.children = null;
        }
    }
}
```

#### Change 2: Use new function in buildTreeStructure
```javascript
// OLD:
if (treeData.children) {
    treeData.children.forEach(child => collapseAll(child));
}

// NEW:
if (treeData.children) {
    treeData.children.forEach(child => collapseAllExceptExpanded(child));
}
```

#### Change 3: Track manual expand/collapse in handleNodeClick
```javascript
// When user manually toggles a directory, file, or chunk:

// Collapsing:
nodeData._children = nodeData.children;
nodeData.children = null;
expandedNodes.delete(nodeData.id);  // NEW: Remove from tracking

// Expanding:
nodeData.children = nodeData._children;
nodeData._children = null;
expandedNodes.add(nodeData.id);  // NEW: Add to tracking
```

## Testing
To test the fix:

1. Start the visualizer: `mcp-vector-search visualize`
2. Open browser to http://localhost:8501
3. Click on any directory node (marked as expandable)
4. **Expected**: Children should load and display immediately
5. **Previous behavior**: Required clicking twice

## Impact
- ✅ Progressive loading now works correctly on first click
- ✅ No breaking changes to existing functionality
- ✅ Maintains collapsed state for non-expanded nodes
- ✅ Preserves user navigation state across tree rebuilds

## Files Modified
- `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

## Related Code
- Backend endpoint: `/api/graph-expand/{node_id}` in `server.py` (already working)
- Event handler: `handleNodeClick()` in `scripts.py` (already working)
- Expansion logic: `expandNode()` in `scripts.py` (already working)
- Tree building: `buildTreeStructure()` in `scripts.py` (**fixed**)
