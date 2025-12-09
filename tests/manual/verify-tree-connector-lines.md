# Manual Test: Tree Connector Lines Fix

**Date**: 2025-12-09
**Issue**: QA found zero connector lines in tree_root view
**Fix**: Changed `getFilteredLinksForCurrentViewV2()` to return containment links

## Test Setup

1. Start the visualization server:
   ```bash
   mcp-vector-search visualize
   ```

2. Open browser to `http://localhost:8000`

## Test Cases

### Test 1: Initial View Shows Connector Lines

**Expected**:
- Root nodes displayed in vertical list
- Curved connector lines visible between parent-child nodes
- Console shows: `[EdgeFilter] TREE_ROOT mode: X containment edges between root nodes` (where X > 0)

**Steps**:
1. Open browser DevTools (F12)
2. Go to Console tab
3. Refresh page
4. Look for the log message above
5. Visually verify connector lines are present in the tree

**Pass Criteria**:
- ✅ X > 0 (at least one connector line)
- ✅ Connector lines visible in SVG
- ✅ Lines connect parent to child nodes

### Test 2: Click Expands Folder with Tree Layout

**Expected**:
- Clicking a folder shows its children
- Tree layout applied with D3 algorithm
- Connector lines updated to show expanded hierarchy

**Steps**:
1. Click on any root folder (e.g., "src")
2. Check console for: `[Click] Node clicked: directory ...`
3. Check console for: `[StateManager] View mode: tree_expanded`
4. Verify children appear with connector lines

**Pass Criteria**:
- ✅ View mode changes from "tree_root" to "tree_expanded"
- ✅ Children nodes appear
- ✅ Connector lines connect parent to children
- ✅ Console shows: `[EdgeFilter] TREE_EXPANDED mode: X containment edges`

### Test 3: Connector Line Types

**Expected**:
- Only containment relationship lines shown
- No call or import edges in tree view

**Steps**:
1. In browser console, run:
   ```javascript
   getFilteredLinksForCurrentViewV2().map(l => l.type)
   ```
2. Verify all types are one of:
   - `dir_containment`
   - `file_containment`
   - `dir_hierarchy`

**Pass Criteria**:
- ✅ No "calls" edges
- ✅ No "imports" edges
- ✅ Only containment edges present

### Test 4: Visual Quality

**Expected**:
- Curved connector lines (not straight)
- Lines connect node centers properly
- No overlapping or crossing lines (in simple trees)

**Steps**:
1. Inspect SVG elements in DevTools
2. Find `<path>` elements for links
3. Verify they have curved paths (using D3 linkHorizontal or linkVertical)

**Pass Criteria**:
- ✅ Lines are curved (Bezier curves)
- ✅ Lines connect to node centers or edges
- ✅ Visual quality is acceptable

## Debugging Tips

If connector lines are still missing:

1. **Check data loading**:
   ```javascript
   console.log(allLinks.filter(l =>
       l.type === 'dir_containment' ||
       l.type === 'file_containment' ||
       l.type === 'dir_hierarchy'
   ).length)
   ```
   Should return > 0 if containment links exist in data.

2. **Check visibility**:
   ```javascript
   console.log(stateManager.getVisibleNodes())
   ```
   Should return array of visible node IDs.

3. **Check view mode**:
   ```javascript
   console.log(stateManager.viewMode)
   ```
   Should be "tree_root" initially.

4. **Check filtered links**:
   ```javascript
   console.log(getFilteredLinksForCurrentViewV2())
   ```
   Should return array with length > 0.

## Known Issues

- **Large codebases**: If project has thousands of files, initial tree may be too large
- **Flat structure**: If all files are in root with no subdirectories, tree will be flat list
- **No containment links**: If graph data doesn't include containment relationships, no lines will appear

## Success Criteria Summary

- ✅ Connector lines visible in initial tree_root view
- ✅ Lines connect parent-child directory relationships
- ✅ Click handlers expand folders correctly
- ✅ Tree layout applies D3 algorithm after expansion
- ✅ No JavaScript errors in console
- ✅ Performance acceptable (< 100ms render time)

## Rollback

If fix causes issues, revert the change in `scripts.py`:

```javascript
// Rollback to original (BROKEN) code:
if (stateManager.viewMode === 'tree_root') {
    return [];  // No links in tree_root mode
}
```

**Note**: This is NOT recommended as it breaks tree visualization.
