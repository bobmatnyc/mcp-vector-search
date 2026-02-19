# Progressive Loading Fix - Root Cause Analysis

## Problem Summary
Progressive loading (expanding nodes to fetch children on-demand) was not working in the main visualizer. Users could click on directory nodes marked as `expandable: true`, but no children would appear.

## Root Cause

The bug was in the interaction between `buildTreeStructure()` and `handleNodeClick()`:

### The Broken Logic Flow

1. **Initial Load** (`/api/graph-initial`)
   - API returns nodes with `expandable: true, expanded: false`
   - These nodes have NO children yet (haven't been fetched)

2. **Tree Building** (`buildTreeStructure()`)
   ```javascript
   // Line 389-394
   treeNodes.forEach(node => {
       nodeMap.set(node.id, {
           ...node,
           children: []  // ← PROBLEM: Initializes ALL nodes with empty array
       });
   });
   ```
   - **Issue**: Every node gets `children: []`, even unexpanded ones

3. **Click Detection** (`handleNodeClick()`) - **BEFORE FIX**
   ```javascript
   // Lines 1769-1772 (old code)
   const hasLoadedChildren = (nodeData.children && nodeData.children.length > 0) ||
                              (nodeData._children && nodeData._children.length > 0);

   if (nodeData.expandable && !nodeData.expanded && !hasLoadedChildren) {
       expandNode(nodeData.id);  // Never reached!
   }
   ```
   - **Issue**: `hasLoadedChildren` checks `children.length > 0`
   - **Problem**: `children` is `[]` (empty array), so `length` is `0`
   - **Result**: `hasLoadedChildren = false` ✓
   - **Expectation**: Should trigger `expandNode()` ✓
   - **But then**: User clicks AGAIN...
   - **Second check**: Still `hasLoadedChildren = false` because array is still `[]`!
   - **Infinite loop**: Node appears expandable forever, never actually expands

4. **Why It Appeared to Work Initially**
   - The check `!hasLoadedChildren` would pass on first click
   - `expandNode()` WOULD be called and fetch children
   - **But** after `buildTreeStructure()` rebuilt the tree, children were collapsed
   - **Because** the newly added children weren't in `expandedNodes` set
   - **Wait, they should be!** Line 348 adds to set: `expandedNodes.add(nodeId)`
   - **But** `collapseAllExceptExpanded()` only preserves the PARENT, not the children
   - **So** children nodes get collapsed into `_children`
   - **Result**: Node appears collapsed still, user clicks again
   - **Second click**: `hasLoadedChildren` is STILL false (because `children` is `null` after collapse)
   - **Boom**: Fetches AGAIN from server (duplicate request)

## The Actual Issue

The check was using the WRONG signal to detect if a node had been expanded:

- **Wrong**: Check if `children.length > 0` (breaks because of empty array initialization)
- **Right**: Check the `expanded` flag that `expandNode()` sets after successful fetch

## The Fix

### Change 1: Directory Expansion Check

**Before** (lines 1766-1777):
```javascript
const hasLoadedChildren = (nodeData.children && nodeData.children.length > 0) ||
                           (nodeData._children && nodeData._children.length > 0);

if (nodeData.expandable && !nodeData.expanded && !hasLoadedChildren) {
    expandNode(nodeData.id);
    return;
}
```

**After**:
```javascript
// Node needs loading if: expandable=true AND not yet expanded
// The `expanded` flag is set by expandNode() after successful fetch (line 351)
// We CANNOT rely on children.length because buildTreeStructure initializes all nodes with children: []
if (nodeData.expandable && !nodeData.expanded) {
    expandNode(nodeData.id);
    return;
}
```

### Change 2: File Expansion Check

**Before** (line 1800):
```javascript
if (nodeData.expandable && !nodeData.expanded && !nodeData.children && !nodeData._children) {
    expandNode(nodeData.id);
}
```

**After**:
```javascript
// Same as directory: use expanded flag, not children array length
if (nodeData.expandable && !nodeData.expanded) {
    expandNode(nodeData.id);
}
```

### Change 3: Enhanced Logging

Added debug logging to trace the expansion flow:

1. **In `expandNode()`** (after line 351):
   ```javascript
   console.log(`Marked node ${nodeId} as expanded`);
   console.log(`expandedNodes set now contains: ${Array.from(expandedNodes).join(', ')}`);
   console.log(`Rebuilding tree structure after expanding ${nodeId}...`);
   ```

2. **In `collapseAllExceptExpanded()`** (after line 654):
   ```javascript
   console.log(`Collapsed node ${node.id} (not in expandedNodes)`);
   // OR
   console.log(`Preserving node ${node.id} (in expandedNodes with ${node.children.length} children)`);
   ```

## Why the Fix Works

1. **Reliable Signal**: The `expanded` flag is set ONCE by `expandNode()` after successful fetch
2. **Survives Rebuild**: The flag persists on the node object through tree rebuilds
3. **Simple Check**: Just test `!nodeData.expanded` instead of array length checks
4. **No False Positives**: Once expanded, stays expanded (no duplicate fetches)

## Expected Behavior After Fix

1. **Initial State**: Node has `expandable: true, expanded: false`
2. **First Click**: Check detects `!expanded` → calls `expandNode()`
3. **Fetch**: `expandNode()` fetches children from `/api/graph-expand/{id}`
4. **Mark**: Sets `node.expanded = true` and adds to `expandedNodes` set
5. **Rebuild**: `buildTreeStructure()` creates tree with new children
6. **Preserve**: `collapseAllExceptExpanded()` keeps node expanded (checks `expandedNodes` set)
7. **Render**: Children are visible in tree
8. **Second Click**: Check detects `expanded = true` → toggles collapse/expand (no refetch)

## Testing

Created test file: `test_progressive_loading.html`

Tests cover:
- Initial click on expandable node (should fetch)
- Second click after expansion (should NOT fetch)
- Node with actual children (should NOT fetch)
- File nodes (same behavior as directories)
- Full workflow simulation

## Files Modified

- `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`:
  - Lines 1766-1777: Directory expansion check
  - Lines 1800-1805: File expansion check
  - Lines 351-362: Enhanced logging in `expandNode()`
  - Lines 654-660: Enhanced logging in `collapseAllExceptExpanded()`

## Verification

To verify the fix works:

1. Start visualization server: `mcpvs visualize`
2. Open browser console (F12)
3. Click on a directory node with `expandable: true`
4. Check console for logs:
   - "Progressive loading: fetching children from server"
   - "Received X new nodes and Y new links"
   - "Marked node {id} as expanded"
   - "expandedNodes set now contains: ..."
   - "Preserving node {id} (in expandedNodes with N children)"
5. Verify children appear in tree
6. Click same node again - should toggle collapse/expand (no refetch)
7. Check console - should NOT see "fetching children from server" again

## Related Issues

This fix resolves the core progressive loading bug. Previous attempts failed because they tried to fix symptoms rather than the root cause:

- **Fix 1**: `collapseAllExceptExpanded()` - Already existed and worked correctly
- **Fix 2**: Check `hasLoadedChildren` with array length - Failed because of empty array initialization

The real issue was the detection logic relying on array length instead of the explicit `expanded` flag.
