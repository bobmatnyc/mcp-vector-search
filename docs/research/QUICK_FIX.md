# Quick Fix for Visualization Layout Errors

## TL;DR
**Circular layout bug found!** Wrong layout config causes Circular (and possibly Dagre) to fail.

---

## The Bug

**File**: `.mcp-vector-search/visualization/index.html`
**Lines**: 1245-1251

**Current (Broken) Code**:
```javascript
layout: {
    name: layoutName === 'dagre' ? 'dagre' : 'circle',
    rankDir: 'TB',              // ❌ Dagre-only!
    rankSep: 150,               // ❌ Dagre-only!
    nodeSep: 80,                // ❌ Dagre-only!
    ranker: 'network-simplex',  // ❌ Dagre-only!
    spacingFactor: 1.2
}
```

**Problem**: Circular layout gets Dagre options it doesn't understand → Fails/shows nothing

---

## The Fix

Replace lines 1245-1251 with:

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
    spacingFactor: 1.5,
    startAngle: Math.PI / 2,
    sweep: 2 * Math.PI,
    animate: true,
    animationDuration: 500
}
```

---

## Testing Required

### Before Fix:
```bash
# Open http://localhost:8089 in browser
# Open DevTools Console (F12)
# Switch to "Circular" layout
# Run in console:
console.log('Nodes:', cy.nodes().length, 'Edges:', cy.edges().length);
# Expected: Edges = 0 (bug confirmed)
```

### After Fix:
```bash
# Refresh page
# Switch to "Circular" layout
# Run in console:
console.log('Nodes:', cy.nodes().length, 'Edges:', cy.edges().length);
# Expected: Both > 0, circular arrangement visible
```

---

## Files for Details

- **Full Analysis**: `docs/research/visualization_bugs_found.md`
- **Debug Guide**: `tests/manual/debug_console_inspector.html`
- **Technical Deep Dive**: `docs/research/visualization_debug_report.md`

---

**Status**: ⏸️ Awaiting user confirmation before applying fix
