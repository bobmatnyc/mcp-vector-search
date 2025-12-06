# Investigation Report: "I still see no nodes"

**Date**: December 6, 2025, 12:03 PST
**Investigator**: Web QA Agent
**Issue**: Empty visualization despite successful data loading
**Status**: ✅ ROOT CAUSE IDENTIFIED

---

## Executive Summary

**The visualization is empty because `visibleNodes` is never initialized for graphs with > 500 nodes.**

- ✅ Data loads correctly (1449 nodes, 360825 links)
- ✅ `allNodes` and `allLinks` are properly set
- ❌ `visibleNodes` remains empty (size = 0)
- ❌ Zero nodes passed to Cytoscape renderer
- **Impact**: 100% of medium-to-large projects see blank screen

**Fix**: 3-line code change (see Solution section below)

---

## Investigation Process

### 1. Data Verification ✅

```bash
$ python3 -c "import json; data = json.load(open('.mcp-vector-search/visualization/chunk-graph.json')); print(f'Nodes: {len(data[\"nodes\"])}'); print(f'Links: {len(data[\"links\"])}')"
Nodes: 1449
Links: 360825
```

**Result**: Data file is valid and loaded successfully.

### 2. HTML Structure Analysis ✅

**File**: `.mcp-vector-search/visualization/index.html`

**Data initialization** (lines 2759-2760):
```javascript
allNodes = data.nodes;  // ✅ Sets to 1449 nodes
allLinks = data.links;  // ✅ Sets to 360825 links
```

**Result**: Data arrays are properly assigned.

### 3. Code Flow Analysis ❌

**The critical branching logic** (lines 2762-2769):

```javascript
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ BUG HERE
} else {
    visualizeGraph(data);  // ← Only called when nodes ≤ 500
}
```

**For your graph**:
- `data.nodes.length = 1449`
- `1449 > 500` → **TRUE**
- Code calls `switchToCytoscapeLayout('dagre')`
- Code **SKIPS** `visualizeGraph(data)`

### 4. The Missing Initialization ❌

**`visibleNodes` is only initialized in `visualizeGraph()`** (lines 1417-1418):

```javascript
function visualizeGraph(data) {
    // ... (setup code)

    // Line 1417-1418: CRITICAL INITIALIZATION
    visibleNodes = new Set(rootNodes.map(n => n.id));
    collapsedNodes = new Set(rootNodes.map(n => n.id));

    // ... (rendering code)
}
```

**But for graphs > 500 nodes, this function is NEVER CALLED!**

### 5. The Filtering Failure ❌

**Inside `switchToCytoscapeLayout()`** (line 1164):

```javascript
const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
```

**State at this point**:
- `allNodes.length = 1449` ✅
- `visibleNodes.size = 0` ❌ (never initialized)
- `visibleNodes.has(n.id)` returns `false` for ALL nodes
- `visibleNodesList.length = 0` ❌

**Result**: Zero nodes passed to Cytoscape!

### 6. The Empty Render ❌

**Cytoscape initialization** (lines 1207-1252):

```javascript
cy = cytoscape({
    container: cyContainer,
    elements: cyElements,  // ← EMPTY ARRAY []
    style: [...],
    layout: {...}
});
```

**Result**: Cytoscape renders an empty graph. User sees blank screen.

---

## Code Flow Diagram

```
User Opens Page
       ↓
DOMContentLoaded Event
       ↓
fetch("chunk-graph.json")
       ↓
Data Loads Successfully
       ├─ allNodes = 1449 nodes ✅
       └─ allLinks = 360825 links ✅
       ↓
Is nodes.length > 500?
       ├─ NO (≤ 500)  → visualizeGraph(data) ✅
       │                ├─ visibleNodes = new Set(...) ✅
       │                └─ renderGraph() ✅
       │                     → Nodes rendered ✅
       │
       └─ YES (> 500) → switchToCytoscapeLayout('dagre') ❌
                        ├─ visibleNodes.size = 0 ❌ (NEVER INITIALIZED!)
                        ├─ visibleNodesList = [] ❌
                        ├─ cyElements = [] ❌
                        └─ Cytoscape renders empty ❌
                             → Blank screen 🚨
```

---

## State Comparison Table

| Variable | Expected (< 500 nodes) | Actual (> 500 nodes) | Status |
|----------|----------------------|---------------------|--------|
| `allNodes.length` | 1449 | 1449 | ✅ Correct |
| `allLinks.length` | 360825 | 360825 | ✅ Correct |
| `visibleNodes.size` | ~50 (root nodes) | **0** | ❌ WRONG |
| `collapsedNodes.size` | ~50 (root nodes) | **0** | ❌ WRONG |
| `visibleNodesList.length` | ~50 | **0** | ❌ WRONG |
| `cyElements.length` | ~100 (nodes + edges) | **0** | ❌ WRONG |
| `cy.nodes().length` | ~50 | **0** | ❌ WRONG |

---

## Root Cause Summary

**The bug is a logic error in the large-graph optimization path:**

1. ✅ Code correctly identifies large graphs (> 500 nodes)
2. ✅ Code correctly chooses Dagre layout for performance
3. ❌ Code INCORRECTLY skips the initialization logic in `visualizeGraph()`
4. ❌ `switchToCytoscapeLayout()` assumes `visibleNodes` is already initialized
5. ❌ With empty `visibleNodes`, all nodes are filtered out
6. ❌ Cytoscape renders empty graph

**This is a CRITICAL path bug affecting all medium-to-large codebases.**

---

## Solution

### Option 3: Always Call visualizeGraph() First (RECOMMENDED)

**File**: `.mcp-vector-search/visualization/index.html`
**Lines**: 2762-2769

**BEFORE (broken)**:
```javascript
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ visibleNodes not initialized
} else {
    visualizeGraph(data);
}
```

**AFTER (fixed)**:
```javascript
const layoutSelector = document.getElementById('layoutSelector');

// ALWAYS initialize through visualizeGraph
visualizeGraph(data);  // ✅ Initializes visibleNodes, collapsedNodes, rootNodes

// Then switch to Dagre for large graphs
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ✅ NOW visibleNodes is initialized
}
```

### Why This Works

1. ✅ **Consistent initialization**: All graphs go through the same setup path
2. ✅ **No code duplication**: Root node logic stays in `visualizeGraph()`
3. ✅ **Minimal performance impact**: `visualizeGraph()` is fast; Dagre overwrites layout immediately
4. ✅ **Maintains all features**: Node expansion, collapse, filtering all work
5. ✅ **Simple and safe**: 3-line change, no complex refactoring

### Alternative Solutions (Not Recommended)

**Option 1**: Duplicate initialization logic before `switchToCytoscapeLayout()`
- ❌ Code duplication
- ❌ Higher maintenance burden
- ❌ Risk of logic divergence

**Option 2**: Make `switchToCytoscapeLayout()` handle uninitialized state
- ❌ Adds complexity to layout function
- ❌ Breaks separation of concerns
- ❌ Harder to test and debug

---

## Testing Plan

### Test Cases

- [ ] **Test 1**: Small graph (< 500 nodes)
  - Should use D3 force layout
  - Should show root nodes
  - No regression expected

- [ ] **Test 2**: Medium graph (500-1000 nodes)
  - Should auto-switch to Dagre
  - Should show root nodes ← **CURRENTLY BROKEN**
  - Fix should resolve this

- [ ] **Test 3**: Large graph (> 1000 nodes, e.g., 1449)
  - Should auto-switch to Dagre
  - Should show root nodes ← **CURRENTLY BROKEN**
  - Fix should resolve this ← **USER'S CASE**

- [ ] **Test 4**: Layout switching
  - Switch from Force → Dagre
  - Switch from Dagre → Circle
  - Switch from Circle → Force
  - All should maintain visibleNodes state

- [ ] **Test 5**: Node interactions
  - Expand collapsed nodes
  - Collapse expanded nodes
  - Click to view content
  - Search and highlight

### Browser Console Checks

After applying fix, verify in browser console:

```javascript
console.log('allNodes:', allNodes?.length);          // Should be 1449
console.log('allLinks:', allLinks?.length);          // Should be 360825
console.log('visibleNodes:', visibleNodes?.size);    // Should be ~50 (root nodes)
console.log('cy nodes:', cy?.nodes()?.length);       // Should be ~50 (root nodes)
```

### Visual Verification

- [ ] Page loads without errors
- [ ] Loading spinner shows progress
- [ ] Graph displays root nodes (folders/files)
- [ ] Legend shows node type colors
- [ ] Controls are visible and functional
- [ ] Dagre layout is active for large graphs
- [ ] Nodes can be clicked and expanded
- [ ] No blank screen ✅

---

## Deliverables

### Files Created

1. **Investigation Report**: `docs/summaries/INVESTIGATION_REPORT_empty_nodes.md` (this file)
2. **Bug Report**: `docs/summaries/CRITICAL_BUG_REPORT_visibleNodes_initialization.md`
3. **Test Page**: `tests/manual/test_visualization.html` (demonstrates the bug)
4. **Bug Visualization**: `tests/manual/bug_visualization.html` (visual explanation)

### Recommended Actions

1. **Apply the fix** (3-line change to index.html)
2. **Test with user's 1449-node graph**
3. **Verify browser console** (check visibleNodes.size)
4. **Test layout switching**
5. **Deploy and notify user**

---

## Impact Assessment

| Metric | Value |
|--------|-------|
| **Severity** | CRITICAL |
| **Users Affected** | 100% of graphs > 500 nodes |
| **Data Loss** | None (data loads correctly) |
| **Workaround** | None (users see blank screen) |
| **Fix Complexity** | LOW (3-line change) |
| **Testing Effort** | MEDIUM (5 test cases) |
| **Regression Risk** | LOW (maintains existing behavior for small graphs) |

---

## Timeline

- **12:00 PST**: User report received ("I still see no nodes")
- **12:03 PST**: Investigation started
- **12:05 PST**: Root cause identified (visibleNodes not initialized)
- **12:10 PST**: Solution designed (Option 3)
- **12:15 PST**: Test pages created
- **12:20 PST**: Documentation completed
- **NEXT**: Apply fix and verify

---

## Conclusion

**The visualization is empty due to a code path bug, NOT a data loading issue.**

The fix is straightforward: always call `visualizeGraph(data)` before switching to Dagre layout for large graphs. This ensures `visibleNodes` is properly initialized regardless of graph size.

**Estimated fix time**: 5 minutes
**Estimated test time**: 15 minutes
**Total resolution time**: 20 minutes

**Recommendation**: Apply fix immediately and test with user's 1449-node graph.

---

**Visual Explanation**: Open `tests/manual/bug_visualization.html` in browser for interactive diagram

**Test Verification**: Open `tests/manual/test_visualization.html` in browser to see the bug demonstrated

**Next Step**: Apply the 3-line fix to `index.html` and retest
