# Visualization Layout Configuration Analysis

**Date:** 2025-12-08
**Researcher:** Claude (Research Agent)
**Project:** mcp-vector-search
**Component:** Visualization System

---

## Executive Summary

Analysis of the visualization layout configuration reveals:

1. **Cytoscape Layout**: Currently configured for **vertical (TB)** layout at line 1956
2. **Initial Node State**: Nodes start **collapsed** by design (lines 372-374)
3. **Layout Mode**: Default is D3 force-directed, not Cytoscape dagre
4. **Issue**: User perception may be due to D3 force layout, not dagre configuration

---

## Findings

### 1. Cytoscape Dagre Layout Configuration

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Location:** Lines 1954-1961

```javascript
layout: {
    name: layoutName === 'dagre' ? 'dagre' : 'circle',
    rankDir: 'TB',  // ← VERTICAL LAYOUT (Top-to-Bottom)
    rankSep: 150,
    nodeSep: 80,
    ranker: 'network-simplex',
    spacingFactor: 1.2
}
```

**Finding:** `rankDir: 'TB'` is CORRECT for vertical layout.
- `TB` = Top-to-Bottom (vertical) ✅
- `LR` = Left-to-Right (horizontal)

**Conclusion:** If user sees horizontal layout, they are NOT using dagre layout mode.

---

### 2. Initial Node Expansion State

**File:** Same file
**Location:** Lines 372-374 (initialization), Lines 732-733 (reset)

```javascript
// Start with only root nodes visible, all collapsed
visibleNodes = new Set(rootNodes.map(n => n.id));
collapsedNodes = new Set(rootNodes.map(n => n.id));
```

**Finding:** All root nodes start in **collapsed** state by design.

**Behavior:**
- `visibleNodes`: Contains all root-level nodes (visible)
- `collapsedNodes`: Contains all root-level nodes (marked as collapsed)
- Children are not visible until parent is clicked/expanded

**Location in Code:**
- Initial state: Line 372-374
- Reset function: Line 732-733

---

### 3. Default Layout Mode

**File:** Same file
**Location:** Line 39

```javascript
let currentLayout = 'force';  // Track current layout type
```

**Finding:** Default visualization uses **D3 force-directed layout**, NOT dagre.

**Layout Options:**
1. `'force'` - D3 force-directed (default)
2. `'dagre'` - Cytoscape hierarchical (vertical TB)
3. `'circle'` - Cytoscape circular

**User Issue Analysis:**
- If user sees "horizontal" layout, they're likely viewing D3 force layout
- D3 force layout has no inherent direction (nodes spread naturally)
- To get vertical layout, user must switch to dagre layout explicitly

---

### 4. D3 Force Layout Configuration

**Location:** Lines 398-429

```javascript
simulation = d3.forceSimulation(visibleNodesList)
    .force("link", d3.forceLink(visibleLinks)
        .id(d => d.id)
        .distance(d => { /* distance logic */ })
    )
    .force("charge", d3.forceManyBody().strength(-200))
    .force("collision", d3.forceCollide().radius(30))
    .force("center", d3.forceCenter(width / 2, height / 2).strength(0.1))
```

**Finding:** D3 force layout has NO vertical/horizontal preference.
- Uses physics simulation
- Nodes spread based on forces (charge, link distance, collision)
- Natural spreading appears "horizontal-ish" due to screen aspect ratio

**No `forceX` or `forceY` constraints found** - layout is purely force-based.

---

## Issue Diagnosis

### User Reports
1. ❌ "Visualization showing horizontally when should be vertical"
2. ❌ "Nodes are expanded when they should start collapsed"

### Reality Check

**Issue #1: Horizontal Layout**
- **Actual Cause:** User viewing D3 force layout (default), not dagre
- **Dagre Configuration:** CORRECT (`rankDir: 'TB'` at line 1956)
- **Solution:** User needs to switch to dagre layout explicitly

**Issue #2: Nodes Expanded**
- **Actual State:** Nodes START COLLAPSED (lines 372-374)
- **Root Cause Analysis:**
  - If nodes appear "expanded", they may be:
    - a) Switched to expanded state via interaction
    - b) Layout makes nodes look spread out (D3 force physics)
    - c) `collapsedNodes` Set not being used correctly in UI rendering

---

## Code References

### Key Line Numbers

| Item | Line(s) | Description |
|------|---------|-------------|
| **rankDir** | 1956 | Cytoscape dagre `rankDir: 'TB'` (CORRECT) |
| **visibleNodes init** | 373 | `visibleNodes = new Set(rootNodes.map(...))` |
| **collapsedNodes init** | 374 | `collapsedNodes = new Set(rootNodes.map(...))` |
| **Default layout** | 39 | `currentLayout = 'force'` |
| **D3 simulation** | 398-429 | Force-directed layout configuration |
| **Cytoscape layout** | 1954-1961 | Dagre layout options |
| **Switch to dagre** | 1851-1962 | `switchToCytoscapeLayout(layoutName)` |
| **Switch to force** | 1975-1991 | `switchToForceLayout()` |
| **Reset view** | 730-734 | Reset to collapsed root nodes |

---

## Recommendations

### For User (Immediate Actions)

1. **Check Current Layout Mode:**
   - Look for layout selector in UI
   - Verify if "force", "dagre", or "circle" is active
   - If "force" is active, switch to "dagre" for vertical layout

2. **Verify Node State:**
   - Inspect `collapsedNodes` Set in browser console
   - Check if nodes are truly expanded or just visually spread out
   - Test reset function to restore collapsed state

3. **Expected Behavior:**
   - Force layout: Nodes spread naturally (no vertical preference)
   - Dagre layout: Vertical hierarchy (TB direction) ✅
   - Initial state: All nodes collapsed ✅

### For Developers (Code Changes)

**If user wants vertical force layout:**
```javascript
// Add forceY constraint to D3 simulation
.force("y", d3.forceY(d => {
    // Calculate vertical position based on hierarchy level
    return calculateDepth(d) * 100;
}).strength(0.3))
```

**If user wants dagre as default:**
```javascript
// Change line 39 from:
let currentLayout = 'force';
// To:
let currentLayout = 'dagre';

// And initialize with dagre layout instead of force
```

---

## Memory Usage

- **Files Analyzed:** 3 (layout_engine.py, scripts.py, base structure)
- **Total Lines Read:** ~500 lines (strategic sampling)
- **Search Queries:** 10 targeted grep searches
- **File Size:** scripts.py exceeds 35K tokens (used grep instead of full read)

---

## Next Steps

1. **User Action Required:**
   - Report which layout mode is currently active
   - Provide screenshot showing "horizontal" layout
   - Confirm if issue persists after switching to dagre layout

2. **Verification Tests:**
   ```bash
   # Open visualization
   mcp-vector-search visualize

   # In browser console:
   console.log(currentLayout);  // Should show 'force', 'dagre', or 'circle'
   console.log(collapsedNodes);  // Should show Set of collapsed node IDs
   console.log(visibleNodes);    // Should show Set of visible node IDs
   ```

3. **If Issue Persists:**
   - Check browser console for JavaScript errors
   - Verify data initialization (see previous bug reports)
   - Test with minimal dataset (2-3 files)

---

## Related Documentation

- `layout_engine.py`: Server-side layout calculations
- Previous bug reports in `docs/research/visualization_*`
- Architecture: `docs/development/VISUALIZATION_ARCHITECTURE_V2.md`

---

## Conclusion

**Current Configuration is CORRECT:**
- ✅ Dagre layout: `rankDir: 'TB'` (vertical)
- ✅ Initial state: Nodes collapsed
- ✅ Layout options: Multiple modes available

**User Issue Likely Due To:**
- Using wrong layout mode (force instead of dagre)
- Misinterpreting force layout spread as "horizontal"
- Possible UI state inconsistency (needs verification)

**Action Required:**
User must confirm which layout mode they are viewing and provide evidence (screenshot/console logs) to diagnose actual issue.
