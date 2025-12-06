# Code Graph Visualization - Layout Controls Hidden Issue

**Date**: December 5, 2025
**Issue**: Layout controls and edge filters appear to be covered/hidden in the visualization
**Status**: ✅ ROOT CAUSE IDENTIFIED

---

## Investigation Summary

### Observed Behavior
- User reports layout controls are "covered over"
- Only the legend is visible on the left side panel
- Layout selector dropdown and edge filter checkboxes are not visible

### Technical Analysis

#### 1. Data Loading: ✅ WORKING
- Graph data (`chunk-graph.json`) loads successfully
- 1449 nodes, 0 edges
- JavaScript successfully sets `#layout-controls` and `#edge-filters` to `display: block`

#### 2. Element Visibility: ✅ ELEMENTS ARE VISIBLE
Safari inspection confirms:
```json
{
  "layoutControls": {
    "display": "block",
    "visibility": "visible",
    "opacity": "1",
    "width": "250px",
    "height": "46px"
  },
  "edgeFilters": {
    "display": "block",
    "visibility": "visible",
    "opacity": "1",
    "width": "250px",
    "height": "139px"
  }
}
```

#### 3. Root Cause: ❌ CSS POSITIONING CONFLICT

**The Issue:**
The `.legend` div has conflicting CSS positioning that breaks the layout.

**HTML Structure:**
```html
<div id="controls">          <!-- position: absolute; top: 20px; left: 20px; -->
    <h1>🔍 Code Graph</h1>
    <div id="layout-controls">...</div>
    <div id="edge-filters">...</div>
    <h3>Legend</h3>
    <div class="legend">...< /div>   <!-- position: absolute; top: 70px; right: 20px; -->
</div>
```

**CSS Conflict:**
```css
#controls {
    position: absolute;
    top: 20px;
    left: 20px;
    /* ... */
}

.legend {
    position: absolute;   /* ❌ PROBLEM: Absolute inside absolute */
    top: 70px;
    right: 20px;
    /* ... */
}
```

**What's Happening:**
1. The `#controls` div is absolutely positioned at top-left of viewport
2. The `.legend` div INSIDE controls is ALSO absolutely positioned
3. The `.legend` absolute positioning (top: 70px, right: 20px) positions it relative to `#controls`
4. This causes the legend to overlay/cover the `#layout-controls` and `#edge-filters` divs
5. The legend appears at the top because it's positioned absolutely, pushing other content down or covering it

---

## Recommendations

### Option 1: Remove Absolute Positioning from .legend (Recommended)
**Change the `.legend` CSS to use static/relative positioning:**

```css
/* BEFORE */
.legend {
    position: absolute;
    top: 70px;
    right: 20px;
    background: #161b22;
    /* ... */
}

/* AFTER */
.legend {
    position: static;  /* or position: relative; */
    background: #161b22;
    margin-top: 12px;  /* Add spacing from controls above */
    /* Remove top and right properties */
    /* ... */
}
```

### Option 2: Move Legend Outside Controls (Alternative)
**Restructure HTML to have legend as a separate panel:**

```html
<div id="controls">
    <h1>🔍 Code Graph</h1>
    <div id="layout-controls">...</div>
    <div id="edge-filters">...</div>
</div>

<div id="legend-panel">
    <h3>Legend</h3>
    <div class="legend">...</div>
</div>
```

Then keep the absolute positioning for the legend panel.

### Option 3: Use z-index to Layer Properly
**If absolute positioning must be kept, fix the stacking:**

```css
#layout-controls,
#edge-filters {
    position: relative;
    z-index: 10;  /* Ensure they appear above legend */
}

.legend {
    position: relative;  /* Change to relative */
    z-index: 1;
}
```

---

## Testing Performed

### Tools Used
- ✅ Safari + AppleScript inspection
- ✅ HTTP server verification (localhost:8082)
- ✅ JSON data loading validation
- ✅ DOM computed styles analysis
- ✅ Screenshot capture for visual confirmation

### Console Monitoring
- ✅ No JavaScript errors detected
- ✅ Graph loaded successfully message displayed
- ✅ Control visibility JavaScript executed successfully

### Verification Commands
```bash
# Verify graph data loads
curl -s http://localhost:8082/chunk-graph.json | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'{len(data.get(\"nodes\", []))} nodes')"

# Check control element states
osascript -e 'tell application "Safari" to do JavaScript "..." in current tab of window 1'
```

---

## Implementation Priority

**Severity**: Medium - Controls are functional but not visible, impacting user experience
**Recommended Fix**: Option 1 (Remove absolute positioning from .legend)
**Estimated Effort**: 5 minutes
**Testing Required**: Visual regression test across browsers

---

## Files Involved

- `src/mcp_vector_search/visualize/export.py` - Template generation
- HTML template (embedded in Python file)
- CSS rules for `#controls` and `.legend`

---

## Next Steps

1. ✅ Identify the exact line in `export.py` where the legend CSS is defined
2. ⏳ Modify the CSS to use relative positioning
3. ⏳ Test the fix in Safari
4. ⏳ Verify across different graph sizes (small vs large)
5. ⏳ Commit the fix with proper documentation

---

**Investigation completed by**: Web QA Agent
**Diagnostic artifacts**:
- `/tests/manual/inspect_visualization_controls.js`
- `/tests/manual/test_controls_safari.sh`
- `/.mcp-vector-search/visualization/controls-screenshot.png`
