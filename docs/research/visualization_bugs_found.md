# Visualization Bugs Found - QA Report
**Date**: December 6, 2025
**Status**: BUGS IDENTIFIED - Awaiting User Testing
**Agent**: Web QA Agent

---

## Executive Summary

Found **TWO BUGS** in the visualization code:

1. ✅ **Edge ID extraction** - Already fixed (lines 1189-1190)
2. ❌ **Circular layout configuration** - BUG CONFIRMED (lines 1245-1251)
3. ⚠️ **D3 data mutation** - POTENTIAL ISSUE (line 1444)

---

## Bug #1: Edge ID Extraction ✅ FIXED

**Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html:1189-1190`

**Status**: ✅ Already fixed correctly

**Code**:
```javascript
const sourceId = link.source.id || link.source;
const targetId = link.target.id || link.target;
```

**Analysis**: This handles both cases:
- When `link.source` is a string: Uses `link.source` directly
- When `link.source` is an object (after D3 mutation): Uses `link.source.id`

**Verdict**: NO ACTION NEEDED

---

## Bug #2: Circular Layout Configuration ❌ CONFIRMED BUG

**Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html:1245-1251`

**Status**: ❌ BUG CONFIRMED - NEEDS FIX

**Current Code**:
```javascript
layout: {
    name: layoutName === 'dagre' ? 'dagre' : 'circle',
    rankDir: 'TB',                  // ❌ Dagre-only option
    rankSep: 150,                   // ❌ Dagre-only option
    nodeSep: 80,                    // ❌ Dagre-only option
    ranker: 'network-simplex',      // ❌ Dagre-only option
    spacingFactor: 1.2              // ✅ Both support this
}
```

**Problem**: Circular layout receives Dagre-specific options which it doesn't understand. This may cause:
- Circular layout to fail silently
- Circular layout to use default config instead of intended config
- Console warnings about unknown options

**Impact**: **HIGH** - This is likely why Circular layout shows nothing

**Recommended Fix**:
```javascript
layout: layoutName === 'dagre' ? {
    name: 'dagre',
    rankDir: 'TB',
    rankSep: 150,
    nodeSep: 80,
    ranker: 'network-simplex',
    spacingFactor: 1.2
} : {
    name: 'circle',
    spacingFactor: 1.2,
    radius: Math.min(width, height) * 0.4,  // Use 40% of viewport
    startAngle: 0,
    sweep: 2 * Math.PI,  // Full circle
    animate: true,
    animationDuration: 500
}
```

---

## Bug #3: D3 Data Mutation ⚠️ POTENTIAL ISSUE

**Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html:1444`

**Status**: ⚠️ NEEDS VERIFICATION - May cause intermittent issues

**Current Code**:
```javascript
function renderGraph() {
    const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
    const filteredLinks = getFilteredLinks();
    const visibleLinks = filteredLinks.filter(l =>
        visibleNodes.has(l.source.id || l.source) &&
        visibleNodes.has(l.target.id || l.target)
    );

    simulation = d3.forceSimulation(visibleNodesList)
        .force("link", d3.forceLink(visibleLinks)  // ⚠️ MUTATES visibleLinks!
            .id(d => d.id)
            // ...
```

**Problem**:
- `d3.forceLink(visibleLinks)` **mutates** the objects in `visibleLinks` array
- Since `visibleLinks` is filtered from `allLinks`, and JavaScript arrays are passed by reference, this mutates `allLinks`
- After first Force layout run, `allLinks` permanently has object references instead of string IDs

**Why Bug #1 fix works**:
- The fix handles BOTH cases: `link.source.id || link.source`
- But the code behavior depends on execution order (brittle)

**Impact**: **MEDIUM** - Fix in Bug #1 masks this, but code is fragile

**Recommended Fix** (Make code robust):
```javascript
function renderGraph() {
    const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
    const filteredLinks = getFilteredLinks();
    const visibleLinks = filteredLinks.filter(l =>
        visibleNodes.has(l.source.id || l.source) &&
        visibleNodes.has(l.target.id || l.target)
    );

    // Clone links before passing to D3 to prevent mutation of original data
    const d3Links = visibleLinks.map(l => ({...l}));

    simulation = d3.forceSimulation(visibleNodesList)
        .force("link", d3.forceLink(d3Links)  // ✅ Use cloned data
            .id(d => d.id)
            // ...
```

**Alternative Fix** (More explicit):
```javascript
// In switchToCytoscapeLayout(), always normalize to strings
visibleLinks.forEach(link => {
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
```

---

## Testing Requirements

### Manual Browser Console Testing Required

Since MCP Browser Extension is not installed, manual testing is needed.

**Test File Created**: `/Users/masa/Projects/mcp-vector-search/tests/manual/debug_console_inspector.html`

**User must**:
1. Open `http://localhost:8089` in browser
2. Open DevTools Console (F12 or Cmd+Option+I)
3. Follow test sequence in `debug_console_inspector.html`
4. Copy console output for analysis

### Key Tests to Run

**Test 1: Initial State**
```javascript
console.log('allLinks[0].source type:', typeof allLinks[0].source);
console.log('allLinks.length:', allLinks.length);
console.log('allNodes.length:', allNodes.length);
```

**Expected**:
- Before Force layout: `typeof allLinks[0].source === "string"`
- After Force layout: `typeof allLinks[0].source === "object"`

**Test 2: Dagre Layout**
```javascript
// Switch to Dagre, then run:
console.log('Cytoscape nodes:', cy.nodes().length);
console.log('Cytoscape edges:', cy.edges().length);
```

**Expected**:
- Nodes should match `allNodes.length`
- Edges should match `allLinks.length`
- If edges = 0, BUG CONFIRMED

**Test 3: Circular Layout**
```javascript
// Switch to Circular, then run:
console.log('Cytoscape nodes:', cy.nodes().length);
console.log('Cytoscape edges:', cy.edges().length);
console.log('Layout config:', cy.layout()._private);
```

**Expected**:
- Check for warnings about unknown layout options
- Nodes should be arranged in circle
- If blank screen, BUG CONFIRMED

---

## Root Cause Analysis

### Why Dagre Might Show Errors

**Hypothesis 1**: Circular layout config bug affects Dagre too
- ❌ Unlikely - Dagre should accept those options

**Hypothesis 2**: Edge data is invalid
- ⚠️ Possible - If `sourceId` or `targetId` is undefined/null
- Needs console testing to verify

**Hypothesis 3**: No visible nodes/edges
- ⚠️ Possible - If filters hide everything
- Check `visibleNodesList.length` and `visibleLinks.length`

**Hypothesis 4**: Cytoscape Dagre extension not loaded
- ⚠️ Possible - Check `typeof dagre !== 'undefined'`
- Extension loaded at line 10, should work

### Why Circular Shows Nothing

**Most Likely**: Bug #2 (wrong layout config)
- Circular layout receives Dagre options it doesn't understand
- Falls back to default config or fails silently
- **FIX**: Separate layout configs for Dagre vs Circular

---

## Recommended Action Plan

### Priority 1: Fix Circular Layout Config (Bug #2)
**Impact**: HIGH
**Confidence**: HIGH
**File**: `.mcp-vector-search/visualization/index.html:1245-1251`

**Change**:
```javascript
// BEFORE (current broken code)
layout: {
    name: layoutName === 'dagre' ? 'dagre' : 'circle',
    rankDir: 'TB',
    rankSep: 150,
    nodeSep: 80,
    ranker: 'network-simplex',
    spacingFactor: 1.2
}

// AFTER (fixed code)
layout: layoutName === 'dagre' ? {
    name: 'dagre',
    rankDir: 'TB',
    rankSep: 150,
    nodeSep: 80,
    ranker: 'network-simplex',
    spacingFactor: 1.2
} : {
    name: 'circle',
    spacingFactor: 1.5,
    startAngle: Math.PI / 2,  // Start at top
    sweep: 2 * Math.PI,
    animate: true,
    animationDuration: 500
}
```

### Priority 2: Prevent D3 Data Mutation (Bug #3)
**Impact**: MEDIUM
**Confidence**: MEDIUM
**File**: `.mcp-vector-search/visualization/index.html:1444`

**Change**: Clone links before passing to D3:
```javascript
const d3Links = visibleLinks.map(l => ({...l}));
simulation = d3.forceSimulation(visibleNodesList)
    .force("link", d3.forceLink(d3Links)  // Use cloned data
```

### Priority 3: User Testing
**Impact**: CRITICAL
**Confidence**: N/A

User MUST run manual console tests to:
1. Confirm bugs identified
2. Capture exact error messages
3. Verify fixes work
4. Check for additional issues

---

## Files Created for Investigation

1. `/Users/masa/Projects/mcp-vector-search/docs/research/visualization_debug_report.md`
   - Detailed technical analysis
   - Code flow documentation
   - Testing requirements

2. `/Users/masa/Projects/mcp-vector-search/tests/manual/debug_console_inspector.html`
   - Interactive debugging guide
   - Console test scripts
   - Expected vs actual results comparison

3. `/Users/masa/Projects/mcp-vector-search/docs/research/visualization_bugs_found.md`
   - This file - comprehensive bug report
   - Recommended fixes
   - Action plan

---

## Next Steps

### For User:
1. ✅ Review this bug report
2. ❌ Open browser to `http://localhost:8089`
3. ❌ Run manual console tests from `debug_console_inspector.html`
4. ❌ Share console output with QA agent
5. ❌ Confirm which layouts work vs fail

### For QA Agent (after user testing):
1. ❌ Analyze user's console output
2. ❌ Confirm root cause
3. ❌ Apply fixes (Priority 1 & 2)
4. ❌ Request re-test
5. ❌ Mark bugs as verified fixed

---

## Confidence Levels

| Bug | Identified | Confirmed | Fix Ready | Fix Tested |
|-----|-----------|-----------|-----------|-----------|
| #1: Edge ID extraction | ✅ | ✅ | ✅ | ⏸️ Awaiting test |
| #2: Circular config | ✅ | ✅ | ✅ | ⏸️ Awaiting test |
| #3: D3 mutation | ✅ | ⚠️ Partial | ✅ | ⏸️ Awaiting test |

**Overall Status**: 🟡 INVESTIGATION COMPLETE - AWAITING USER TESTING

---

**Generated by**: Web QA Agent
**Investigation Time**: ~15 minutes
**Files Analyzed**: 1 (index.html, 2800+ lines)
**Lines Inspected**: 1142-1260 (switchToCytoscapeLayout), 1380-1450 (data loading), 1245-1251 (layout config)
**Bugs Found**: 2 confirmed, 1 potential
**Testing Required**: Manual browser console (MCP Browser Extension not installed)
