# CRITICAL BUG REPORT: Empty Visualization - visibleNodes Not Initialized

**Date**: 2025-12-06
**Severity**: CRITICAL
**Status**: ROOT CAUSE IDENTIFIED
**Reporter**: Web QA Agent

## User Report
"I still see no nodes" - after applying legend CSS and data initialization fixes

## Investigation Summary

### Data Verification ✅
- `chunk-graph.json` loads successfully: **1449 nodes, 360825 links**
- `allNodes` and `allLinks` ARE properly initialized (lines 2759-2760)
- Data is NOT the problem

### Root Cause Identified ❌

**The bug is in the code flow for large graphs (> 500 nodes):**

#### File: `index.html`
#### Lines: 2762-2769

```javascript
// Auto-select Dagre for large graphs
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ BUG: visibleNodes not initialized!
} else {
    visualizeGraph(data);  // ✅ This initializes visibleNodes
}
```

### The Problem Flow

**When nodes > 500 (our case: 1449 nodes):**

1. **Line 2759-2760**: `allNodes` and `allLinks` are initialized ✅
2. **Line 2766**: Code calls `switchToCytoscapeLayout('dagre')` directly ❌
3. **SKIP**: `visualizeGraph()` is NEVER called
4. **Line 1417-1418**: `visibleNodes = new Set(...)` is NEVER executed
5. **Line 1164**: `switchToCytoscapeLayout()` tries to filter with empty `visibleNodes`

```javascript
// Line 1164 in switchToCytoscapeLayout()
const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
// visibleNodes.size = 0 → visibleNodesList.length = 0
```

6. **Line 1175-1185**: Zero nodes added to Cytoscape elements
7. **Result**: Empty visualization! 🚨

### The Missing Initialization

`visibleNodes` is only initialized in `visualizeGraph()` (lines 1417-1418):

```javascript
function visualizeGraph(data) {
    // ... (lines omitted)

    // Line 1417-1418: Initialize visibleNodes
    visibleNodes = new Set(rootNodes.map(n => n.id));
    collapsedNodes = new Set(rootNodes.map(n => n.id));
    // ...
}
```

But for large graphs, this function is NEVER CALLED!

## Impact

**100% of users with > 500 nodes see empty visualization**

- Default behavior for medium-to-large codebases
- Affects: monorepos, enterprise projects, comprehensive indexing
- No error message, just blank screen

## Verification Test

Created: `tests/manual/test_visualization.html`

Opens at: http://localhost:8082/../../../tests/manual/test_visualization.html

This test demonstrates:
1. `visibleNodes.size = 0` before fix
2. `visibleNodesList.length = 0` → no nodes rendered
3. After proper initialization: `visibleNodes.size = [root count]` → nodes render

## The Fix Required

**Option 1: Initialize visibleNodes before switchToCytoscapeLayout() (RECOMMENDED)**

```javascript
// Line 2762-2769 (FIXED)
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';

    // CRITICAL FIX: Initialize visibleNodes first!
    // (Extract this logic from visualizeGraph)
    let rootNodes = [];
    if (data.metadata && data.metadata.is_monorepo) {
        rootNodes = allNodes.filter(n => n.type === 'subproject');
    } else {
        const dirNodes = allNodes.filter(n => n.type === 'directory');
        const fileNodes = allNodes.filter(n => n.type === 'file');
        const minDirDepth = dirNodes.length > 0
            ? Math.min(...dirNodes.map(n => n.depth))
            : Infinity;
        const minFileDepth = fileNodes.length > 0
            ? Math.min(...fileNodes.map(n => n.depth))
            : Infinity;
        rootNodes = [
            ...dirNodes.filter(n => n.depth === minDirDepth),
            ...fileNodes.filter(n => n.depth === minFileDepth)
        ];
        if (rootNodes.length === 0) {
            rootNodes = fileNodes;
        }
    }

    visibleNodes = new Set(rootNodes.map(n => n.id));
    collapsedNodes = new Set(rootNodes.map(n => n.id));
    highlightedNode = null;

    switchToCytoscapeLayout('dagre');
} else {
    visualizeGraph(data);
}
```

**Option 2: Make switchToCytoscapeLayout handle uninitialized visibleNodes**

```javascript
// Line 1164 in switchToCytoscapeLayout() (ALTERNATIVE FIX)
// If visibleNodes is empty, show all root nodes
let visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));

if (visibleNodesList.length === 0) {
    // visibleNodes not initialized - show root nodes
    let rootNodes = [];
    // ... (same root-finding logic)
    visibleNodes = new Set(rootNodes.map(n => n.id));
    visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
}
```

**Option 3: Always call visualizeGraph first (SIMPLEST)**

```javascript
// Line 2762-2769 (SIMPLEST FIX)
const layoutSelector = document.getElementById('layoutSelector');

// Always initialize through visualizeGraph
visualizeGraph(data);

// Then switch to Dagre for large graphs
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');
}
```

## Recommended Solution

**Use Option 3 (simplest)** - Always call `visualizeGraph(data)` first to ensure proper initialization, then switch layouts if needed.

This ensures:
- ✅ `visibleNodes` is always initialized
- ✅ `collapsedNodes` is always initialized
- ✅ `rootNodes` is always calculated
- ✅ No code duplication
- ✅ Consistent initialization path

## Testing Plan

1. Test with < 500 nodes (should work already)
2. Test with > 500 nodes (currently broken, should fix)
3. Test with monorepo metadata
4. Test with regular project metadata
5. Verify Dagre layout renders all root nodes
6. Verify force layout still works after switching

## Timeline

- **Discovery**: 2025-12-06 12:03 PST
- **Root Cause**: 2025-12-06 12:05 PST
- **Fix Required**: ASAP
- **User Impact**: HIGH (all large projects broken)

---

**Next Steps**: Apply Option 3 fix to `index.html` and verify with user's 1449-node graph.
