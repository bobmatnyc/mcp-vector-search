# Code Graph Visualization - UAT Test Report

**Date**: December 6, 2025
**Tester**: Web QA Agent
**URL**: http://localhost:8082
**Test Type**: User Acceptance Testing (UAT) + Technical Investigation

---

## Executive Summary

**STATUS**: ❌ **CRITICAL BUG IDENTIFIED - NO NODES DISPLAYED**

The code graph visualization at http://localhost:8082 fails to display any graph nodes for projects with > 500 nodes. The user report "I don't see any nodes" is **confirmed and validated**.

### Root Cause
**Data initialization bug in JavaScript code**: For graphs with > 500 nodes, the application auto-selects Cytoscape/Dagre layout but fails to populate the global `allNodes` and `allLinks` arrays, resulting in an empty graph visualization.

---

## Test Environment

- **Browser**: Chromium (Playwright-controlled)
- **Server**: Python SimpleHTTP/0.6 on port 8082
- **Data File**: chunk-graph.json (54MB, 1,449 nodes, 360,826 links)
- **Server Mode**: `--code-only` mode
- **Working Directory**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization`

---

## Business Requirements Validation

### Requirement: Display Interactive Code Graph
- **Expected**: Users should see a visual graph representation of code relationships
- **Actual**: ❌ **NO nodes displayed** - only blank dark area with controls
- **Business Impact**: **CRITICAL** - Core feature is completely non-functional for production-scale codebases

### Requirement: Support Large Codebases
- **Expected**: Handle graphs with 1,000+ nodes efficiently
- **Actual**: ❌ **FAILS for > 500 nodes** - automatic optimization triggers bug
- **Business Impact**: **CRITICAL** - Feature unusable for real-world projects

### Requirement: Provide Layout Options
- **Expected**: Users can select different graph layouts (Dagre, Force-Directed, etc.)
- **Actual**: ⚠️ **PARTIAL** - UI controls visible but graph is empty
- **Business Impact**: **HIGH** - Controls are present but non-functional due to missing data

---

## Technical Test Results

### Phase 1: UI Controls Check ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| Layout Selector | ✅ Visible | Dropdown shows "Hierarchical (Dagre)" |
| Edge Filter Checkboxes | ✅ Visible | 5 checkboxes present (Containment, Function Calls, etc.) |
| Legend | ✅ Visible | Shows color codes for Functions, Classes, Methods, File types |
| Reset View Button | ✅ Visible | Button present in top-right |

**Screenshot Evidence**: `/Users/masa/Projects/mcp-vector-search/docs/research/screenshots/visualization_initial.png`

### Phase 2: Graph Rendering Check ❌

| Element | Expected | Actual | Status |
|---------|----------|--------|--------|
| SVG #graph | Present | ✅ Present (1 child element) | ✅ OK |
| SVG circles (nodes) | 1,449 | 0 | ❌ FAIL |
| SVG rectangles (nodes) | Multiple | 0 | ❌ FAIL |
| SVG lines (edges) | 360,826 | 0 | ❌ FAIL |
| Cytoscape container | Created dynamically | ✅ Created | ✅ OK |
| Cytoscape instance | Initialized with data | ❌ Empty (0 nodes) | ❌ FAIL |

### Phase 3: Data Loading Check ⚠️

| Check | Result | Evidence |
|-------|--------|----------|
| chunk-graph.json request | ✅ SUCCESS | HTTP 200 OK, 54MB downloaded |
| JSON parsing | ✅ SUCCESS | "Parsing JSON data..." message shown |
| Data assignment | ❌ **FAILED** | Global variables remain empty |
| Loading indicator | ✅ Shows success | "✓ Graph loaded successfully" |

**Critical Finding**: Loading indicator incorrectly shows success, but data is not assigned to rendering variables.

### Phase 4: JavaScript State Analysis ❌

**Measured at 0s, 2s, 4s, 6s, 8s, 10s, 12s, 14s, 16s, 18s, 20s, 22s, 24s, 26s, 28s:**

```javascript
allNodes: 0        // Expected: 1449
allLinks: 0        // Expected: 360826
cy exists: true    // Cytoscape instance created
currentLayout: "dagre"  // Correct layout selected
visibleNodes: 0    // No nodes to display
```

**Conclusion**: Data never populates global variables despite successful download and parsing.

### Phase 5: Browser Console Errors

Only 1 non-critical error detected:
```
[ERROR] Failed to load resource: the server responded with a status of 404 (File not found)
Location: http://localhost:8082/favicon.ico
```

**Assessment**: This is cosmetic (missing favicon) and not related to the graph rendering issue.

---

## Root Cause Analysis

### Code Flow Analysis

**File**: `visualization HTML template (served at http://localhost:8082)`

#### Normal Flow (Small Graphs < 500 nodes):
```javascript
fetch("chunk-graph.json")
  .then(response => response.json())
  .then(data => {
    visualizeGraph(data);  // ✅ Sets allNodes and allLinks
  });
```

#### Broken Flow (Large Graphs > 500 nodes):
```javascript
fetch("chunk-graph.json")
  .then(response => response.json())
  .then(data => {
    if (data.nodes.length > 500) {
      switchToCytoscapeLayout('dagre');  // ❌ Doesn't pass 'data'!
    } else {
      visualizeGraph(data);  // This is skipped for large graphs
    }
  });
```

### The Bug (Lines 2757-2761 in HTML template):

```javascript
// Auto-select Dagre for large graphs
const layoutSelector = document.getElementById('layoutSelector');
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ BUG: 'data' not passed or assigned
} else {
    visualizeGraph(data);  // ✅ This sets allNodes and allLinks
}
```

### What Should Happen:

The `visualizeGraph()` function (line 1381) initializes global state:

```javascript
function visualizeGraph(data) {
    g.selectAll("*").remove();

    allNodes = data.nodes;  // ← Critical initialization
    allLinks = data.links;  // ← Critical initialization

    // ... rest of visualization logic
}
```

### What Actually Happens:

The `switchToCytoscapeLayout()` function (line 1141) assumes `allNodes` and `allLinks` are already populated:

```javascript
function switchToCytoscapeLayout(layoutName) {
    // ...

    // Get visible nodes and filtered links
    const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));  // ❌ allNodes is []
    const filteredLinks = getFilteredLinks();  // ❌ Uses empty allLinks
    const visibleLinks = filteredLinks.filter(/* ... */);

    // Convert to Cytoscape format
    const cyElements = [];

    visibleNodesList.forEach(node => {  // ❌ Empty array - no nodes added
        cyElements.push({ /* ... */ });
    });

    // Initialize Cytoscape with empty elements
    cy = cytoscape({
        container: cyContainer,
        elements: cyElements,  // ❌ Empty array []
        // ...
    });
}
```

---

## Bug Impact Assessment

### Severity: **CRITICAL (P0)**

**Affected Users**: Anyone with codebases resulting in > 500 nodes
- **This project**: 1,449 nodes ❌ **BROKEN**
- **Small projects**: < 500 nodes ✅ **WORKS** (uses D3.js force-directed layout)
- **Production codebases**: Typically > 500 nodes ❌ **BROKEN**

### Business Impact:

1. **Feature Unusability**: Core visualization feature is completely non-functional for real-world use cases
2. **User Confusion**: Success message shown despite empty graph
3. **Lost Value**: Performance optimization (Dagre layout) triggers the bug
4. **Scalability Failure**: Feature works for small demos but fails at scale

### User Experience Impact:

- User sees loading progress bar complete successfully ✅
- User sees "✓ Graph loaded successfully" ✅
- User sees all controls and UI elements ✅
- User sees **NO GRAPH NODES** ❌
- User cannot explore code relationships ❌
- User cannot verify visualization accuracy ❌

---

## Recommended Fix

### Priority 1: Immediate Fix

**Location**: Data loading callback (around line 2757-2761)

**Current Code**:
```javascript
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');  // ❌ Missing data initialization
} else {
    visualizeGraph(data);
}
```

**Fixed Code (Option 1 - Initialize data first)**:
```javascript
// ALWAYS initialize global data first
allNodes = data.nodes;
allLinks = data.links;

if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre');
} else {
    visualizeGraph(data);
}
```

**Fixed Code (Option 2 - Pass data to function)**:
```javascript
if (layoutSelector && data.nodes && data.nodes.length > 500) {
    layoutSelector.value = 'dagre';
    switchToCytoscapeLayout('dagre', data);  // Pass data parameter
} else {
    visualizeGraph(data);
}

// Then update switchToCytoscapeLayout signature:
function switchToCytoscapeLayout(layoutName, data = null) {
    // Initialize allNodes and allLinks if data provided
    if (data) {
        allNodes = data.nodes;
        allLinks = data.links;
    }

    // ... rest of function
}
```

**Fixed Code (Option 3 - Extract initialization)**:
```javascript
// Extract data initialization into separate function
function initializeGraphData(data) {
    allNodes = data.nodes;
    allLinks = data.links;

    // Initialize other state
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
}

// Then in loading callback:
.then(data => {
    clearTimeout(timeout);
    loadingEl.innerHTML = '<label style="color: #238636;">✓ Graph loaded successfully</label>';
    setTimeout(() => loadingEl.style.display = 'none', 2000);

    // Show controls
    const layoutControls = document.getElementById('layout-controls');
    const edgeFilters = document.getElementById('edge-filters');
    if (layoutControls) layoutControls.style.display = 'block';
    if (edgeFilters) edgeFilters.style.display = 'block';

    // Initialize data FIRST (always)
    initializeGraphData(data);

    // Then choose layout
    const layoutSelector = document.getElementById('layoutSelector');
    if (layoutSelector && data.nodes && data.nodes.length > 500) {
        layoutSelector.value = 'dagre';
        switchToCytoscapeLayout('dagre');
    } else {
        renderGraph();  // D3.js rendering
        positionNodesCompactly(allNodes.filter(n => visibleNodes.has(n.id)));
        setTimeout(() => zoomToFit(750), 300);
    }
})
```

**Recommended**: **Option 3** - Most maintainable, avoids code duplication, clearer separation of concerns.

### Priority 2: Validation & Testing

1. **Add JavaScript Unit Tests**: Test data initialization for both small and large graphs
2. **Add Console Logging**: Temporary debug logs to verify data loading
3. **Add Error Handling**: Catch and report data initialization failures
4. **Update Success Message**: Only show after data is confirmed loaded

### Priority 3: User Experience Improvements

1. **Loading States**: Show "Initializing graph..." during data processing
2. **Error States**: Display clear error if no nodes after 5 seconds
3. **Debug Mode**: Add `?debug=1` URL parameter to show data loading stats
4. **Performance Metrics**: Log time taken for each loading stage

---

## Test Evidence Artifacts

### Screenshots
- ✅ `/Users/masa/Projects/mcp-vector-search/docs/research/screenshots/visualization_initial.png` - Shows empty graph with controls visible
- ✅ `/Users/masa/Projects/mcp-vector-search/docs/research/screenshots/visualization_final.png` - Confirms no change after page load
- ✅ `/Users/masa/Projects/mcp-vector-search/docs/research/screenshots/debug_simple.png` - Debug view showing empty state

### Test Scripts
- ✅ `/Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_visualization_playwright.py` - Comprehensive Playwright test
- ✅ `/Users/masa/Projects/mcp-vector-search/tests/manual/debug_visualization_simple.py` - Simple debug test
- ✅ `/Users/masa/Projects/mcp-vector-search/tests/manual/debug_loading_timing.py` - Timing analysis test

### Test Data
- ✅ `/Users/masa/Projects/mcp-vector-search/docs/research/visualization_test_report.json` - Detailed console and network logs

---

## Regression Testing Plan

Once fixed, verify:

1. **Small Graphs (< 500 nodes)**: Still work with D3.js force-directed layout
2. **Large Graphs (> 500 nodes)**: Display correctly with Cytoscape Dagre layout
3. **Edge Cases**:
   - Exactly 500 nodes (boundary condition)
   - 501 nodes (just over threshold)
   - Empty graph (0 nodes)
   - Single node graph
4. **Performance**: Measure time to first node visible
5. **Browser Compatibility**: Test in Chrome, Firefox, Safari
6. **Network Conditions**: Test with slow connections (data should stream)

---

## Acceptance Criteria for Fix

✅ **Fix is complete when**:

1. Navigate to http://localhost:8082
2. Wait for "✓ Graph loaded successfully" message
3. **VERIFY**: Graph nodes are visible in the viewport
4. **VERIFY**: `allNodes.length === 1449` in browser console
5. **VERIFY**: `allLinks.length === 360826` in browser console
6. **VERIFY**: Graph is interactive (zoom, pan, click nodes)
7. **VERIFY**: Layout selector works correctly
8. **VERIFY**: Edge filters modify visible links

---

## Conclusion

### User Report Validation: ✅ **CONFIRMED**

The user's report "I don't see any nodes" is **100% accurate and reproducible**.

### Technical Validation: ✅ **ROOT CAUSE IDENTIFIED**

Data initialization bug in JavaScript prevents nodes from displaying for graphs with > 500 nodes.

### Business Impact: 🚨 **CRITICAL**

This bug makes the visualization feature completely unusable for production-scale codebases, which is the primary target use case.

### Fix Difficulty: ✅ **LOW**

Simple code change with clear solution path. Estimated fix time: 30 minutes coding + 1 hour testing.

### Recommendation: **IMMEDIATE FIX REQUIRED**

This is a P0 bug that should be fixed before any other feature work. The bug affects core functionality and has a straightforward solution.

---

**Report Generated**: December 6, 2025
**QA Agent**: Web QA Agent (UAT + Technical Testing)
**Next Steps**: Implement Priority 1 fix and run regression tests
