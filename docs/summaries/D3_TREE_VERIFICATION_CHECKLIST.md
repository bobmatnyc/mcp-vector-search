# D3.js Tree Layout Verification Checklist

**Purpose**: Verify the new D3.js tree layout implementation works correctly.

## Quick Start

```bash
# Start visualization server
mcp-vector-search visualize serve --port 8080

# Open in browser
# http://localhost:8080
```

## Visual Verification Checklist

### ✅ Phase 1: Initial View (Root List)
- [ ] See vertical list of root folders/files
- [ ] Items sorted alphabetically
- [ ] Folders show `+` expand indicator
- [ ] No edges/lines visible (correct for root view)

### ✅ Phase 2: Tree Expansion
- [ ] **Click any folder with `+` indicator**
- [ ] Tree should expand **horizontally to the right**
- [ ] Parent node stays on left
- [ ] Children appear to the right with vertical spacing
- [ ] **Curved connector lines** visible (not straight lines!)
- [ ] Lines connect parent to each child with Bézier curves

### ✅ Tree Structure Verification

**Expected Layout**:
```
Root Folder (+)  ────➤  Child 1 (+)  ────➤  Grandchild 1
                │
                ├────➤  Child 2
                │
                └────➤  Child 3 (+)  ────➤  Grandchild 2
                                      │
                                      └────➤  Grandchild 3
```

**Check**:
- [ ] Parent nodes are always to the **left** of children
- [ ] Children are **vertically stacked** (not horizontal spread)
- [ ] Deeper levels extend further **right**
- [ ] Spacing is consistent across all levels
- [ ] No overlapping nodes or labels

### ✅ Curved Lines (Critical!)

**What to Look For**:
- [ ] Lines are **curved** (smooth Bézier curves)
- [ ] NOT straight diagonal lines
- [ ] Curves flow naturally from parent to child
- [ ] Line color: Gray (`#4a5568`)
- [ ] Line opacity: 60% (semi-transparent)

**Visual Example**:
```
Parent ───╮
          ╰─────➤ Child
```
Not:
```
Parent
      \
       \_____ Child
```

### ✅ Interaction Tests

**Multi-Level Expansion**:
1. [ ] Click root folder → expands to show children
2. [ ] Click child folder → expands to show grandchildren
3. [ ] Tree grows **horizontally** (not vertically)
4. [ ] All nodes remain visible (may need to zoom out)
5. [ ] Breadcrumb shows navigation path

**Collapse/Expand**:
1. [ ] Click expanded folder (with `−` indicator)
2. [ ] Children should disappear
3. [ ] `−` changes back to `+`
4. [ ] Tree reflows to remove empty space

**Large Folder Test**:
1. [ ] Find folder with 10+ children
2. [ ] Click to expand
3. [ ] Children should spread vertically
4. [ ] Proper spacing (no overlap)
5. [ ] All curves visible and smooth

### ✅ Browser Console Checks

Open browser DevTools (F12) and check console for:

**Expected Messages**:
```
[D3 Layout] Built tree with X nodes, height=YYYpx, width=ZZZpx
[Render] D3 tree layout with X nodes, depth N
[EdgeFilter] TREE_EXPANDED mode: N containment edges
```

**No Errors**:
- [ ] No JavaScript errors in console
- [ ] No "undefined" position warnings
- [ ] D3 tree layout messages appear on expansion

### ✅ Performance Tests

**Large Tree (100+ nodes)**:
- [ ] Expansion completes within 1 second
- [ ] No browser lag or freezing
- [ ] Smooth transitions/animations
- [ ] Zoom/pan still responsive

**Memory Usage**:
- [ ] No memory leaks (check DevTools Memory tab)
- [ ] Heap size stays reasonable after multiple expansions
- [ ] Garbage collection works properly

## Troubleshooting

### Problem: Straight Lines Instead of Curves

**Cause**: d3.linkHorizontal() not working

**Check**:
1. Browser console for D3 errors
2. Verify D3.js library loaded (check Network tab)
3. Look for path elements: `<path class="link">` not `<line>`

**Fix**: Refresh page, clear browser cache

### Problem: Vertical List Instead of Tree

**Cause**: Tree layout not being applied

**Check Console For**:
```
[Render] PHASE 1 (tree_root): ...
```
Should change to:
```
[Render] TREE_EXPANDED: D3 tree layout ...
```

**Fix**: Ensure clicking folder triggers tree_expanded mode

### Problem: Nodes Overlapping

**Cause**: Spacing calculation issue

**Check**:
- Node count in console messages
- Tree height/width values
- Viewport size (try fullscreen)

**Fix**: Zoom out, or check adaptive spacing logic

### Problem: Missing Edges

**Cause**: Edge filtering

**Check Console For**:
```
[EdgeFilter] TREE_EXPANDED mode: 0 containment edges
```

**Fix**: Verify containment links exist in chunk-graph.json

## Expected Console Output

```
[Render] Rendering graph, mode: tree_root, phase: Phase 1 (overview)
[Render] Visible nodes: 15
[Render] PHASE 1 (tree_root): Vertical list with 15 root nodes
[Render] Calculated positions for 15 nodes
[Render] Visible links: 0

[Click on folder]

[Render] Rendering graph, mode: tree_expanded, phase: Phase 2 (radial)
[Render] Visible nodes: 45
[D3 Layout] Built tree with 45 nodes, height=600px, width=1720px
[Render] TREE_EXPANDED: D3 tree layout with 45 nodes, depth 2
[Render] Calculated positions for 45 nodes
[EdgeFilter] TREE_EXPANDED mode: 44 containment edges
[Render] Visible links: 44
```

## Success Criteria

**All Must Be True**:
- ✅ Tree expands horizontally (left to right)
- ✅ Curved connector lines (Bézier curves)
- ✅ Proper parent-child hierarchy
- ✅ No overlapping nodes
- ✅ Consistent spacing
- ✅ Smooth transitions
- ✅ No console errors
- ✅ Expand/collapse works correctly

## Visual Comparison

### Old Behavior (Incremental Positioning)
- Vertical stacking only
- Straight diagonal lines
- Inconsistent spacing

### New Behavior (D3 Tree Layout)
- Hierarchical tree structure
- Curved connector lines
- Professional appearance
- Matches https://d3js.org/d3-hierarchy/tree

## Next Steps

If all checks pass:
1. ✅ Mark verification complete
2. 📝 Document any edge cases found
3. 🎉 Tree layout implementation successful!

If issues found:
1. 📋 Note specific failing test
2. 🐛 Check browser console for errors
3. 💬 Report issue with console output

---

**Test Date**: _______________
**Tester**: _______________
**Browser**: _______________ (Chrome/Safari/Firefox)
**Result**: ⬜ Pass / ⬜ Fail
**Notes**: _____________________________________
