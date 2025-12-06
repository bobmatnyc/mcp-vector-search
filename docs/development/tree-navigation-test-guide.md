# Tree Navigation Test Guide

**Component**: Hybrid Visualization Frontend
**Test Type**: Manual Integration Testing
**Date**: December 6, 2025

## Test Setup

```bash
# 1. Index your project (if not already done)
mcp-vector-search index

# 2. Generate visualization
mcp-vector-search visualize export

# 3. Open in browser
# The HTML file is created in .mcp-vector-search/visualization/
open .mcp-vector-search/visualization/visualization.html
```

## Test Cases

### TC1: Root View (TREE_ROOT Mode)

**Expected Behavior**:
- ✅ Vertical list of root directories/files
- ✅ Alphabetical sorting (directories first)
- ✅ NO edges shown
- ✅ Expand indicators (+) visible on directories/files

**Console Output**:
```javascript
[StateManager] Initialized with mode: tree_root
[Render] TREE_ROOT: Vertical list with N root nodes
```

**How to Test**:
1. Open visualization
2. Check: Are nodes in a vertical list?
3. Check: Are directories shown before files?
4. Check: Are there NO edges/lines connecting nodes?

### TC2: Directory Expansion (TREE_EXPANDED Mode)

**Expected Behavior**:
- ✅ Click directory → expands rightward (800px offset)
- ✅ Children shown as vertical list to the right
- ✅ NO edges during navigation
- ✅ Parent directory shows collapse indicator (−)

**Console Output**:
```javascript
[StateManager] Expanding directory node: dir_abc123 with 5 children
[StateManager] Expansion path: dir_abc123
[StateManager] View mode: tree_expanded
[Layout] Tree: 5 children, offset=800px, spacing=50px
[Render] TREE_EXPANDED: Tree layout with N nodes, depth 1
```

**How to Test**:
1. Click a directory in root view
2. Check: Do children appear 800px to the right?
3. Check: Are children in a vertical list?
4. Check: Are there still NO edges?
5. Check: Does breadcrumb show "🏠 Root / DirectoryName"?

### TC3: Sibling Exclusivity

**Expected Behavior**:
- ✅ Click sibling directory → previous sibling collapses
- ✅ Only ONE expansion path visible at each depth
- ✅ Smooth transition between siblings

**Console Output**:
```javascript
[StateManager] Sibling exclusivity: collapsing dir_old123
[StateManager] Expanding directory node: dir_new456 with 3 children
[StateManager] Expansion path: dir_new456
```

**How to Test**:
1. Expand directory A
2. Click directory B (sibling of A)
3. Check: Did directory A's children disappear?
4. Check: Did directory B's children appear?
5. Check: Is only ONE directory expanded at this depth?

### TC4: Nested Directory Expansion

**Expected Behavior**:
- ✅ Each level expands 800px further right
- ✅ Breadcrumb shows full path
- ✅ NO edges shown at any depth

**Console Output**:
```javascript
[StateManager] Expansion path: dir_a > dir_b > dir_c
[StateManager] View mode: tree_expanded
[Render] TREE_EXPANDED: Tree layout with N nodes, depth 3
```

**How to Test**:
1. Expand directory A
2. Expand child directory B (inside A)
3. Expand child directory C (inside B)
4. Check: Are nodes at positions x=100, x=900, x=1700, x=2500?
5. Check: Does breadcrumb show "🏠 Root / A / B / C"?
6. Check: Can you scroll right to see deep levels?

### TC5: File View with AST Chunks (FILE_DETAIL Mode)

**Expected Behavior**:
- ✅ Click file → expands AST chunks rightward
- ✅ Chunks shown in vertical tree (by line number)
- ✅ Function call edges VISIBLE within file
- ✅ Edge type: "caller" only

**Console Output**:
```javascript
[StateManager] Expanding file node: file_xyz789 with 10 children
[StateManager] View mode: file_detail
[Layout] Tree: 10 children, offset=800px, spacing=50px
[EdgeFilter] FILE_DETAIL mode: 5 call edges in file example.py
[Render] FILE_DETAIL: Tree layout with N nodes, depth 2
```

**How to Test**:
1. Navigate to a directory containing Python files
2. Click a .py file
3. Check: Do AST chunks (functions/classes) appear to the right?
4. Check: Are they in vertical order (by line number)?
5. Check: Are there edges connecting functions that call each other?
6. Check: Are edges ONLY shown for this file (not external calls)?
7. Check: Do edges have arrows showing call direction?

### TC6: Collapse Operations

**Expected Behavior**:
- ✅ Click expanded node → collapses with animation
- ✅ Children disappear smoothly
- ✅ View mode reverts if returning to root

**Console Output**:
```javascript
[StateManager] Collapsing node: dir_abc123
[StateManager] Collapsed to root, switching to TREE_ROOT view
```

**How to Test**:
1. Expand several levels deep
2. Click an expanded directory
3. Check: Do all descendants disappear?
4. Check: Is the transition smooth (750ms)?
5. Click root breadcrumb (🏠 Root)
6. Check: Does view return to root list?
7. Check: Does view mode show "tree_root"?

### TC7: Breadcrumb Navigation

**Expected Behavior**:
- ✅ Breadcrumb shows full path
- ✅ Clicking breadcrumb navigates to that level
- ✅ Last item is not clickable (current node)

**How to Test**:
1. Expand path: Root > DirA > DirB > File.py
2. Check: Breadcrumb shows "🏠 Root / DirA / DirB / File.py"
3. Check: "File.py" is NOT clickable (highlighted)
4. Click "DirA" in breadcrumb
5. Check: Does view collapse to DirA level?
6. Check: Does expansion path update correctly?

### TC8: Content Pane Integration

**Expected Behavior**:
- ✅ Clicking any node shows metadata in sidebar
- ✅ File nodes show file info
- ✅ AST chunks show code preview

**How to Test**:
1. Click a directory
2. Check: Does sidebar show directory info?
3. Click a file
4. Check: Does sidebar show file path, language?
5. Click an AST chunk (function)
6. Check: Does sidebar show code snippet?

## Edge Cases

### EC1: Empty Directory
**Test**: Click directory with no children
**Expected**: Directory expands but shows "No children" or simply no nodes to the right

### EC2: Single Child
**Test**: Expand directory with only one child
**Expected**: Child appears centered vertically relative to parent

### EC3: Many Children (>20)
**Test**: Expand directory with 50+ children
**Expected**: Vertical scrolling works, spacing adapts

### EC4: Deep Nesting (>5 levels)
**Test**: Expand 10 levels deep
**Expected**: Horizontal scrolling works, layout remains consistent

### EC5: Rapid Clicking
**Test**: Click multiple directories quickly
**Expected**: State updates correctly, no visual glitches

## Performance Metrics

### Expected Performance
- **Initial Render**: <2 seconds for 1000 nodes
- **Expand Animation**: 750ms smooth transition
- **Collapse Animation**: 750ms smooth transition
- **Layout Calculation**: <100ms for 100 nodes

### How to Measure
```javascript
// Open browser console and run:
console.time('expand');
// Click a node
console.timeEnd('expand');
// Should log: expand: ~750ms
```

## Browser Console Commands

### Check Current State
```javascript
console.log('View Mode:', stateManager.viewMode);
console.log('Expansion Path:', stateManager.expansionPath);
console.log('Visible Nodes:', stateManager.getVisibleNodes().length);
```

### Toggle Debug Mode
```javascript
// Enable detailed logging
localStorage.setItem('debug_visualization', 'true');
location.reload();
```

### Inspect Node
```javascript
// Find node by name
const node = allNodes.find(n => n.name === 'example.py');
console.log(node);
```

## Common Issues & Solutions

### Issue: No edges shown in file view
**Cause**: File has no function calls OR chunks not in same file
**Solution**: Check that file contains functions that call each other

### Issue: Layout looks cramped
**Cause**: Many nodes at same level
**Solution**: Expected behavior - vertical scrolling should work

### Issue: Sibling doesn't collapse
**Cause**: JavaScript error or state desync
**Solution**: Check browser console for errors, reload page

### Issue: Breadcrumb missing
**Cause**: Breadcrumb element not rendered
**Solution**: Check DOM for `.breadcrumb-nav` element

## Success Criteria

✅ **All test cases pass**
✅ **No JavaScript errors in console**
✅ **Smooth animations (750ms)**
✅ **Correct edge filtering (none in nav, calls in file view)**
✅ **Sibling exclusivity works**
✅ **Breadcrumbs update correctly**

## Reporting Bugs

If you find issues, report with:
1. Test case number (e.g., TC4)
2. Expected vs. actual behavior
3. Browser console errors
4. Screenshot if visual issue
5. Steps to reproduce

---

**Test Completion**: [ ] All TCs passed
**Performance**: [ ] Meets metrics
**Browser Tested**: [ ] Chrome [ ] Firefox [ ] Safari
**Date Tested**: ___________
