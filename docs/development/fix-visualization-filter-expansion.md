# Fix: Visualization Filter Expansion Issue

**Date**: 2025-12-15
**Status**: Fixed
**Files Modified**:
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
- `/tests/manual/test_visualization_changes.py`

## Problem

When clicking Code/Docs filter buttons, the tree would expand into a massive radial view showing thousands of nodes, making the visualization unusable.

### Root Cause

The `applyFileFilter()` function was attempting to preserve the expand/collapse state across filter changes using `saveExpandedState()` and `restoreExpandedState()` helper functions. This state preservation logic was complex and not working correctly, causing all nodes to expand unintentionally.

## Solution

**Simplified Approach**: Don't try to preserve state. Instead, show a clean collapsed tree with only root + first-level children visible when filter changes.

### Changes Made

1. **Removed complex state preservation logic**:
   - Deleted `saveExpandedState()` function (lines 2456-2471)
   - Deleted `restoreExpandedState()` function (lines 2474-2491)
   - Removed calls to these functions in `applyFileFilter()`

2. **Implemented simple collapsed tree view**:
   ```javascript
   // After buildTreeStructure(), we have a collapsed tree (everything in _children).
   // Show only root expanded with first-level children visible, everything else collapsed.
   if (treeData) {
       // Ensure root is expanded
       if (treeData._children && !treeData.children) {
           treeData.children = treeData._children;
           treeData._children = null;
       }

       // Collapse all first-level children (so we see root + first level, but nothing deeper)
       if (treeData.children) {
           treeData.children.forEach(child => {
               if (child.children) {
                   child._children = child.children;
                   child.children = null;
               }
           });
       }

       console.log('Tree reset to collapsed state: root expanded, first-level children collapsed');
   }
   ```

### Key Insight

`buildTreeStructure()` already creates a collapsed tree structure by default (all nodes have `_children` instead of `children`). The fix leverages this by:
1. Expanding only the root node
2. Keeping all first-level children collapsed
3. Rendering this clean, manageable view

## Testing

Updated manual test to verify:
1. Tree stays collapsed after filtering (not expanding to thousands of nodes)
2. Node count remains reasonable (<1000 visible nodes)
3. Filter functionality still works correctly

Test: `/tests/manual/test_visualization_changes.py`

## Code Reduction

**Net LOC Impact**: -52 lines removed

- Removed: ~52 lines (state preservation functions + restoration logic)
- Added: ~20 lines (simple collapsed tree logic)
- **Net**: -32 lines

This aligns with the code minimization principle: simpler solution with less code.

## User Impact

**Before**: Clicking Code/Docs filter caused massive expansion (thousands of nodes)
**After**: Clicking filter shows clean collapsed tree (root + first level only)

Users can now:
- Use filters without performance issues
- See a manageable view of filtered content
- Manually expand directories as needed
- Get consistent behavior across filter changes

## Verification Steps

1. Start visualization: `mcp-vector-search visualize`
2. Click "Code" filter → Should show collapsed tree (not massive radial view)
3. Click "Docs" filter → Should show collapsed tree
4. Click "All" filter → Should restore full tree in collapsed state
5. Verify no JavaScript errors in console

## Related Issues

- Previous fix attempt: Expanded state preservation (d609b6a, 08ef29f)
- This fix: Simplified to collapsed-only approach

## Lessons Learned

1. **Simpler is better**: Complex state preservation was unnecessary and error-prone
2. **Leverage existing behavior**: `buildTreeStructure()` already does what we need
3. **Default to collapsed**: Safer and more performant than trying to expand intelligently
4. **Debug first**: Root cause was over-engineered solution, not missing feature
