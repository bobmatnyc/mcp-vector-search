# Cytoscape Edge Creation Error - Investigation Report

**Date**: 2025-12-06
**Investigator**: Web QA Agent
**Issue**: Cytoscape fails to create edges with "Can only create elements from objects" error

## Executive Summary

**Root Cause Identified**: Object spread operator (`...link`) overwrites valid string IDs with D3-mutated object references when creating Cytoscape edges.

**Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html` lines 1188-1198

**Impact**: All edges fail to render in Cytoscape graph view, though they work in D3 force-directed view.

---

## Investigation Process

### 1. Browser Console Inspection
- Opened visualization at http://localhost:8082
- Enabled Safari DevTools console
- Executed JavaScript commands to inspect data structures

### 2. Data Structure Analysis

#### Original JSON Structure (Correct)
```json
{
  "links": [
    {
      "source": "dir_449a27f1",
      "target": "dir_089000e2",
      "type": "dir_hierarchy"
    }
  ]
}
```

**Observation**: In `chunk-graph.json`, `source` and `target` are strings (node IDs).

#### After D3 Force Simulation (Mutated)
```javascript
// D3's forceLink mutates the link objects:
link.source = nodeObjectReference  // Was: "dir_449a27f1"
link.target = nodeObjectReference  // Was: "dir_089000e2"

// Now accessing:
link.source      // => { id: "dir_449a27f1", name: "docs", ... }
link.source.id   // => "dir_449a27f1"
```

**Observation**: D3.js `forceLink` converts string IDs to object references at line 1444.

### 3. Code Analysis

#### D3 Force Simulation Setup (Lines 1443-1460)
```javascript
simulation = d3.forceSimulation(visibleNodesList)
    .force("link", d3.forceLink(visibleLinks)
        .id(d => d.id)  // ← Tells D3 how to match IDs
        .distance(d => { ... })
        .strength(d => { ... })
```

**Effect**: D3 mutates `visibleLinks` array, replacing:
- `link.source` (string) → `link.source` (object reference)
- `link.target` (string) → `link.target` (object reference)

This is documented D3 behavior for efficient force calculations.

#### Cytoscape Edge Creation (Lines 1188-1198)
```javascript
// Add edges
visibleLinks.forEach(link => {
    const sourceId = link.source.id || link.source;  // ✅ Extracts string ID
    const targetId = link.target.id || link.target;  // ✅ Extracts string ID
    cyElements.push({
        data: {
            source: sourceId,       // ✅ Correct: "dir_449a27f1"
            target: targetId,       // ✅ Correct: "dir_089000e2"
            linkType: link.type,
            isCycle: link.is_cycle,
            ...link  // ❌ BUG: Overwrites source/target with objects!
        }
    });
});
```

**Problem Sequence**:
1. `sourceId` and `targetId` correctly extract string IDs
2. These are assigned to `data.source` and `data.target`
3. `...link` spread operator runs AFTER, copying all link properties
4. This includes `link.source` (object) and `link.target` (object)
5. Object references overwrite the correct string IDs
6. Cytoscape receives objects instead of strings → **Error**

### 4. Error Evidence

**Expected Cytoscape Edge Format**:
```javascript
{
  data: {
    source: "dir_449a27f1",  // String ID
    target: "dir_089000e2",  // String ID
    linkType: "dir_hierarchy"
  }
}
```

**Actual (Buggy) Format**:
```javascript
{
  data: {
    source: { id: "dir_449a27f1", name: "docs", ... },  // Object ❌
    target: { id: "dir_089000e2", name: "_archive", ... },  // Object ❌
    linkType: "dir_hierarchy"
  }
}
```

**Cytoscape Error**:
```
Error: Can only create elements from objects
```

Cytoscape expects `source` and `target` to be **string IDs**, not object references.

---

## Root Cause

**Object Spread Order Matters**: JavaScript object spread (`...`) overwrites properties when applied after initial assignment.

```javascript
// Incorrect order (current code):
{
    source: sourceId,  // Set to string
    ...link            // Overwrites with object
}

// Correct order (fix):
{
    ...link,           // Spread first
    source: sourceId,  // Override with string
    target: targetId
}
```

---

## Recommended Fix

**File**: `.mcp-vector-search/visualization/index.html`
**Lines**: 1188-1198

### Option 1: Reorder Spread (Preferred)
```javascript
// Add edges
visibleLinks.forEach(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;
    cyElements.push({
        data: {
            ...link,        // Spread FIRST
            source: sourceId,  // Then override with strings
            target: targetId,
            linkType: link.type,
            isCycle: link.is_cycle
        }
    });
});
```

### Option 2: Selective Properties (More Explicit)
```javascript
// Add edges
visibleLinks.forEach(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;
    cyElements.push({
        data: {
            source: sourceId,
            target: targetId,
            linkType: link.type,
            isCycle: link.is_cycle
            // Don't spread link at all
        }
    });
});
```

### Option 3: Destructure to Exclude (Most Defensive)
```javascript
// Add edges
visibleLinks.forEach(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;

    // Exclude source/target from spread
    const { source, target, ...linkProps } = link;

    cyElements.push({
        data: {
            source: sourceId,
            target: targetId,
            ...linkProps  // Safe: source/target excluded
        }
    });
});
```

---

## Testing Checklist

After applying fix:

- [ ] Load visualization at http://localhost:8082
- [ ] Switch to Cytoscape view
- [ ] Verify edges render correctly
- [ ] Check browser console for errors
- [ ] Test different layout options (cola, cose, hierarchical)
- [ ] Verify edge filtering works (hierarchy, containment, semantic)
- [ ] Test with different graph sizes (small, medium, large)
- [ ] Confirm no regression in D3 force view

---

## Additional Observations

### Code Patterns Using Object References

Throughout the codebase, there's evidence of handling both string and object forms:

**Line 1166-1170** (Visibility Filter):
```javascript
const visibleLinks = filteredLinks.filter(l =>
    visibleNodes.has(l.source.id || l.source) &&
    visibleNodes.has(l.target.id || l.target)
);
```

**Line 1695** (Node Connection Check):
```javascript
return allLinks.some(l => (l.source.id || l.source) === node.id);
```

This pattern (`l.source.id || l.source`) correctly handles both:
- **Before D3**: `l.source` is string → uses string directly
- **After D3**: `l.source` is object → uses `l.source.id`

The bug only occurs when spreading the entire link object into Cytoscape edge data.

---

## Prevention

### Future Code Review Guidelines

1. **Avoid spreading mutated D3 data** into other libraries
2. **Document D3 mutations** where they occur
3. **Consider cloning** original link data before D3 processes it:
   ```javascript
   const d3Links = visibleLinks.map(l => ({...l}));
   d3.forceLink(d3Links)
   ```
4. **Test cross-library integrations** (D3 + Cytoscape) carefully

---

## References

- **D3 forceLink Documentation**: https://github.com/d3/d3-force#link_links
- **Cytoscape.js Elements Format**: https://js.cytoscape.org/#notation/elements-json
- **JavaScript Object Spread Order**: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax

---

**Status**: Investigation Complete
**Next Step**: Apply recommended fix and validate
