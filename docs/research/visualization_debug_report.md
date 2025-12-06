# Visualization Debug Report
**Date**: December 6, 2025
**Issue**: Dagre and Circular layouts showing errors/blank screens

## Investigation Findings

### Current Code Status
✅ **Fix WAS applied** - Lines 1189-1190 correctly extract source/target IDs:
```javascript
// Add edges
visibleLinks.forEach(link => {
    const sourceId = link.source.id || link.source;
    const targetId = link.target.id || link.target;
```

### Root Cause Identified: D3 Data Mutation

**The Problem:**
1. `allLinks` is assigned direct reference to loaded JSON data (line 1386)
2. D3's `forceLink()` **mutates** link objects in place (line 1444)
3. After Force-Directed layout runs ONCE, `allLinks` is permanently mutated:
   - Before: `{source: "node1", target: "node2", ...}`
   - After: `{source: {id: "node1", ...}, target: {id: "node2", ...}, ...}`
4. The fix on lines 1189-1190 handles this with `link.source.id || link.source`
5. **BUT** this means the code path depends on execution order!

**Scenario 1: Force-Directed First (Current Behavior)**
- User loads page → Force layout active → D3 mutates `allLinks` → Switch to Dagre → Works (because `link.source.id` exists)

**Scenario 2: Dagre/Circular First (User's Issue)**
- User loads page → Immediately switch to Dagre → `allLinks` NOT mutated yet → `link.source` is string → `link.source.id` is undefined → Falls back to `link.source` (string) → ✅ Should work!

**Scenario 3: Dagre → Force → Dagre (Potential Issue)**
- Dagre works → Switch to Force → D3 mutates data → Switch back to Dagre → Should still work with `link.source.id`

## Hypothesis: The Error is Elsewhere

Based on code analysis, the edge creation logic SHOULD work. Possible issues:

### 1. **Layout Configuration Error**
Lines 1231-1239 show layout config:
```javascript
layout: {
    name: layoutName === 'dagre' ? 'dagre' : 'circle',
    rankDir: 'TB',
    rankSep: 150,
    nodeSep: 80,
    ranker: 'network-simplex',
    spacingFactor: 1.2
}
```

**Circular layout issue**: Config includes `rankDir`, `rankSep`, `nodeSep`, `ranker` - these are **Dagre-specific** options! Circular layout doesn't use these.

### 2. **Cytoscape-Dagre Extension Loading**
Lines 8-10 load libraries:
```html
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
```

Need to verify: Is `cytoscape-dagre` properly registered?

### 3. **Empty/Invalid Data**
Check if `visibleLinks` or `visibleNodesList` is empty when switching layouts.

## Required Manual Testing

Since browser console monitoring is not available, user needs to:

### Test 1: Check Data Before Cytoscape Initialization
Open browser DevTools console and run:
```javascript
// Before clicking Dagre, run in console:
console.log('allLinks[0]:', allLinks[0]);
console.log('allLinks[0].source type:', typeof allLinks[0].source);
console.log('allNodes.length:', allNodes.length);
console.log('allLinks.length:', allLinks.length);
```

### Test 2: Check Cytoscape Elements
After switching to Dagre (when error occurs):
```javascript
// Run in console after error:
console.log('cy elements:', cy.elements().length);
console.log('cy nodes:', cy.nodes().length);
console.log('cy edges:', cy.edges().length);
console.log('First edge data:', cy.edges()[0]?.data());
```

### Test 3: Check for JavaScript Errors
Look for:
- ❌ "Cannot read property 'id' of undefined"
- ❌ "dagre is not a function"
- ❌ "Layout 'dagre' not found"
- ❌ Any edge-related errors with ID values

### Test 4: Test Layout Switching Sequence
1. Load page (Force-Directed active)
2. **Immediately** switch to Dagre - does it work?
3. Switch to Circular - does it work?
4. Switch back to Force-Directed
5. Switch to Dagre again - does it work?

## Expected Browser Console Output

**When Force-Directed loads first:**
```
allLinks[0].source: {id: "src/main.py", type: "file", ...}  // Object
typeof allLinks[0].source: "object"
```

**When switching to Dagre immediately after load:**
```
allLinks[0].source: "src/main.py"  // String
typeof allLinks[0].source: "string"
```

## Recommended Fix

To make the code robust regardless of execution order:

```javascript
// Add edges with deep clone to prevent mutation issues
visibleLinks.forEach(link => {
    // Handle both mutated (object) and unmutated (string) formats
    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target.id : link.target;

    cyElements.push({
        data: {
            source: sourceId,
            target: targetId,
            linkType: link.type,
            isCycle: link.is_cycle,
            ...link
        }
    });
});
```

Or better: **Prevent D3 from mutating original data**:

```javascript
// In renderGraph(), create a copy for D3 to mutate
function renderGraph() {
    const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
    const filteredLinks = getFilteredLinks();
    const visibleLinks = filteredLinks.filter(l =>
        visibleNodes.has(l.source.id || l.source) &&
        visibleNodes.has(l.target.id || l.target)
    );

    // CLONE links before passing to D3 to prevent mutation
    const d3Links = visibleLinks.map(l => ({...l}));

    simulation = d3.forceSimulation(visibleNodesList)
        .force("link", d3.forceLink(d3Links)  // Use cloned links
            .id(d => d.id)
            // ... rest of config
```

## Next Steps

1. ✅ User manually tests in browser console (follow Test 1-4 above)
2. ❌ Capture exact error messages and console output
3. ❌ Verify which layout works vs fails
4. ❌ Check if error is consistent or intermittent
5. ❌ Apply appropriate fix based on findings

---

## Files Analyzed
- `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html`
  - Lines 1142-1260: `switchToCytoscapeLayout()` function
  - Lines 1189-1190: Edge ID extraction (fix applied)
  - Lines 1231-1239: Layout configuration
  - Lines 1443-1444: D3 force simulation (mutation point)
  - Lines 1386: `allLinks` assignment
  - Lines 677-680: Layout selector HTML
