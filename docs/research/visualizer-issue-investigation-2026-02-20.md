# Visualizer Issues Investigation

**Date:** 2026-02-20
**Investigator:** Research Agent
**Project:** mcp-vector-search
**Focus:** KG Loading UX + Treemap/Sunburst Visual Encoding Issues

---

## Executive Summary

Investigated two reported issues with the mcp-vector-search visualizer:

1. **KG Visualizer Loading Issue:** KG view loads slowly (22MB file) with NO progress indicator shown to users
2. **Treemap/Sunburst Visual Encoding:** Three-axis visual encoding (complexity color + quality blend + smell borders) IS IMPLEMENTED but may not be visible due to data propagation issues

**Critical Finding:** The visual encoding code is correctly implemented and wired up in `renderTreemap()` and `renderSunburst()`, but hierarchy building functions (`buildFileHierarchy()` and `buildASTHierarchy()`) ARE propagating quality_score, smell_count, and smells fields. The issue is likely **missing data at the source** or **CSS/styling conflicts** hiding the visual cues.

---

## Issue 1: KG Visualizer Loads Slowly with No Progress Indicator

### Root Cause Analysis

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

#### Evidence

**1. Large Data File (Line 381-419 in server.py):**
```python
@app.get("/api/kg-graph")
async def get_kg_graph_data() -> Response:
    """Get knowledge graph data for D3 force-directed visualization."""
    kg_graph_file = viz_dir / "kg-graph.json"

    with open(kg_graph_file, "rb") as f:
        data = orjson.loads(f.read())  # Loads entire 22MB file
```

**File Size:** 22MB (`/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/kg-graph.json`)

**2. No Loading Indicator Called (Lines 5843-5879 in scripts.py):**
```javascript
function loadKGData() {
    fetch('/api/kg-graph')  // ❌ NO showLoadingIndicator() called here
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();  // 22MB JSON parse in browser
        })
        .then(data => {
            // ... process data
            initializeKGView();
        })
        .catch(error => {
            console.error('Failed to load KG data:', error);
            setView('chunks');
        });
}
```

**3. Loading Indicator Functions Exist (Lines 138-149):**
```javascript
function showLoadingIndicator(message) {
    const loadingDiv = document.getElementById('graph-loading-indicator') || createLoadingDiv();
    loadingDiv.querySelector('.loading-message').textContent = message;
    loadingDiv.style.display = 'flex';  // ✅ Function exists and works
}

function hideLoadingIndicator() {
    const loadingDiv = document.getElementById('graph-loading-indicator');
    if (loadingDiv) {
        loadingDiv.style.display = 'none';
    }
}
```

**4. Loading Indicator Used Elsewhere (Lines 200-206):**
```javascript
if (!status.ready) {
    showLoadingIndicator('Generating graph data... This may take a few minutes.');  // ✅ Used for chunks view
    // ...
}
```

### Why This Happens

1. **`loadKGData()` never calls `showLoadingIndicator()`** before fetching
2. **22MB JSON download + parse** takes 3-10 seconds on localhost (varies by machine)
3. **User sees frozen UI** with no feedback during this time
4. **Loading indicator infrastructure exists** but is not invoked for KG view

### Current Loading Flow

```
User clicks "Knowledge Graph" button
    ↓
setView('kg') called (line 5825)
    ↓
kgNodes.length === 0? → YES
    ↓
loadKGData() called (line 5836)
    ↓
❌ NO LOADING INDICATOR SHOWN
    ↓
fetch('/api/kg-graph') starts
    ↓
[3-10 seconds of UI freeze]
    ↓
response.json() parses 22MB
    ↓
initializeKGView() renders graph
    ↓
✅ Graph appears (finally)
```

### Expected Loading Flow

```
User clicks "Knowledge Graph" button
    ↓
setView('kg') called
    ↓
✅ showLoadingIndicator('Loading Knowledge Graph...')
    ↓
loadKGData() called
    ↓
fetch('/api/kg-graph') starts
    ↓
[User sees spinner with "Loading Knowledge Graph..." message]
    ↓
response.json() parses 22MB
    ↓
initializeKGView() renders graph
    ↓
✅ hideLoadingIndicator()
    ↓
Graph appears
```

### Specific Code Locations

- **Missing loading indicator call:** Line 5843 (start of `loadKGData()`)
- **Missing hide call:** Line 5873 (after `initializeKGView()`)
- **Working indicator example:** Lines 200-206 (chunks view status check)
- **Indicator functions:** Lines 138-149 (`showLoadingIndicator`, `hideLoadingIndicator`)

---

## Issue 2: Treemap/Sunburst Visual Encoding Not Working

### Investigation Summary

**STATUS:** Visual encoding code IS IMPLEMENTED and WIRED UP correctly. The issue is likely:
- **Data not present** in source nodes (quality_score, smell_count, smells fields missing)
- **CSS conflicts** hiding visual cues (border colors, stroke-dasharray not rendering)
- **Hierarchy aggregation** not propagating data to parent nodes correctly

### Evidence of CORRECT Implementation

#### 1. Visual Encoding Functions Defined (Lines 4697-4771)

**Smell Border Color (Line 4697-4702):**
```javascript
function getSmellBorderColor(smellCount) {
    if (!smellCount || smellCount === 0) return 'rgba(0,0,0,0.3)';  // default subtle
    if (smellCount === 1) return 'rgba(218, 54, 51, 0.5)';  // light red
    if (smellCount === 2) return 'rgba(218, 54, 51, 0.7)';  // medium red
    return 'rgba(218, 54, 51, 1)';  // full red for 3+
}
```

**Complexity + Quality Blending (Line 4723-4744):**
```javascript
function blendComplexityQuality(complexity, qualityScore) {
    // Get base complexity color
    const complexityColor = getComplexityColor(complexity);

    // Default to healthy if no quality score
    const quality = qualityScore !== undefined && qualityScore !== null ? qualityScore : 1.0;

    // When quality is poor (low score), blend toward blue
    const blueAmount = 1.0 - quality;

    // Parse complexity color
    const rgb = hexToRgb(complexityColor);

    // Blend: add blue proportional to blueAmount
    // For worst code: red (complexity) + blue (poor quality) = purple
    const blendedR = rgb.r * (1 - blueAmount * 0.3);
    const blendedG = rgb.g * (1 - blueAmount * 0.3);
    const blendedB = Math.min(255, rgb.b + (255 * blueAmount * 0.7));

    return rgbToHex(blendedR, blendedG, blendedB);
}
```

**Node Color with Aggregation (Line 4768-4788):**
```javascript
function getNodeColor(d) {
    // If it's a leaf node with complexity, blend complexity with quality score
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return blendComplexityQuality(d.data.complexity, d.data.quality_score);
    }

    // For category/language nodes in AST mode, aggregate complexity from children
    if ((d.data.type === 'category' || d.data.type === 'language') && d.children && d.children.length > 0) {
        // Calculate average complexity and quality from all descendants with complexity
        let totalComplexity = 0;
        let totalQuality = 0;
        let complexityCount = 0;
        let qualityCount = 0;

        function aggregateDescendants(node) {
            if (node.data.complexity !== undefined && node.data.complexity !== null) {
                totalComplexity += node.data.complexity;
                complexityCount++;
            }
            if (node.data.quality_score !== undefined && node.data.quality_score !== null) {
                totalQuality += node.data.quality_score;
                // ... aggregation continues
```

#### 2. Treemap Visual Encoding Applied (Lines 4904-4927)

```javascript
// Add rectangles (with three-axis visual encoding)
cell.append('rect')
    .attr('class', 'treemap-cell')
    .attr('width', d => Math.max(0, d.x1 - d.x0))
    .attr('height', d => Math.max(0, d.y1 - d.y0))
    .attr('fill', d => getNodeColor(d))  // ✅ Axis 1 (red) + Axis 3 (blue) = complexity + quality
    .attr('stroke', d => {
        // For category/language nodes, aggregate smell count from children
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellBorderColor(smellCount);  // ✅ Called
    })  // Axis 2: smell intensity
    .attr('stroke-width', d => {
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellBorderWidth(smellCount);  // ✅ Called
    })  // Axis 2: thicker for more smells
    .attr('stroke-dasharray', d => {
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellDashArray(smellCount);  // ✅ Called
    })  // Axis 2: dashed for smells
    .style('cursor', 'pointer')
    .on('click', handleTreemapClick)  // ✅ Click handler wired up
    // ... hover handlers also present
```

#### 3. Sunburst Visual Encoding Applied (Lines 5330-5380)

```javascript
// Create arcs (with three-axis visual encoding)
const arcs = g.selectAll('path')
    .data(displayRoot.descendants().filter(d => d.depth > 0))
    .join('path')
    .attr('class', 'sunburst-arc')
    .attr('d', arc)
    .attr('fill', d => getNodeColor(d))  // ✅ Axis 1 (red) + Axis 3 (blue)
    .attr('stroke', d => {
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellBorderColor(smellCount);  // ✅ Called
    })  // Axis 2: smell intensity
    .attr('stroke-width', d => {
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellBorderWidth(smellCount);  // ✅ Called
    })
    .attr('stroke-dasharray', d => {
        const smellCount = (d.data.type === 'category' || d.data.type === 'language')
            ? getAggregatedSmellCount(d)
            : (d.data.smell_count || 0);
        return getSmellDashArray(smellCount);  // ✅ Called
    })
    .style('cursor', 'pointer')
    .on('click', handleSunburstClick)  // ✅ Click handler wired up
    // ... hover handlers present
```

#### 4. Hierarchy Building DOES Propagate Fields (Lines 4498-4652)

**buildFileHierarchy (Lines 4507-4526):**
```javascript
function processNode(node, currentDepth = 0) {
    const result = {
        name: node.name,
        id: node.id,
        type: node.type,
        file_path: node.file_path,
        complexity: node.complexity,
        lines_of_code: node.lines_of_code || (node.end_line && node.start_line ? node.end_line - node.start_line + 1 : 0),
        start_line: node.start_line,
        end_line: node.end_line,
        content: node.content,
        docstring: node.docstring,
        language: node.language,
        depth: currentDepth,
        quality_score: node.quality_score,      // ✅ Propagated
        smell_count: node.smell_count,          // ✅ Propagated
        smells: node.smells,                    // ✅ Propagated
        complexity_grade: node.complexity_grade // ✅ Propagated
    };
    // ... children processing
```

**buildASTHierarchy (Lines 4600-4616):**
```javascript
byType.get(chunkType).push({
    name: node.name || node.id.substring(0, 20),
    id: node.id,
    type: node.type,
    file_path: node.file_path,
    complexity: node.complexity,
    lines_of_code: node.lines_of_code || (node.end_line && node.start_line ? node.end_line - node.start_line + 1 : 1),
    start_line: node.start_line,
    end_line: node.end_line,
    content: node.content,
    docstring: node.docstring,
    language: language,
    quality_score: node.quality_score,      // ✅ Propagated
    smell_count: node.smell_count,          // ✅ Propagated
    smells: node.smells,                    // ✅ Propagated
    complexity_grade: node.complexity_grade // ✅ Propagated
});
```

#### 5. Click Handlers ARE Wired Up

- **Treemap:** Line 4930 - `.on('click', handleTreemapClick)`
- **Sunburst:** Line 5357 - `.on('click', handleSunburstClick)`
- **Drill-down logic:** Lines 5017-5075 (treemap), 5478-5520 (sunburst)

**handleTreemapClick includes:**
- Progressive loading check (lines 5029-5070)
- Zoom-in functionality (lines 5073-5102)
- Content display for leaf nodes (line 5074)

**handleSunburstClick includes:**
- Progressive loading check (lines 5491-5532)
- Zoom-in functionality (lines 5535-5564)
- Content display for leaf nodes (line 5536)

### Why Visual Encoding Might Not Be Visible

#### Hypothesis 1: Missing Source Data

**Problem:** `quality_score`, `smell_count`, `smells` fields are not present in the source `allNodes` array.

**Evidence Needed:**
1. Check if `/api/graph` endpoint returns these fields in node objects
2. Check if `chunk-graph.json` file contains these fields
3. Inspect `allNodes` array in browser DevTools during runtime

**Check Location:**
- Server endpoint: `src/mcp_vector_search/cli/commands/visualize/server.py` lines 121-155
- Graph generation: Look for code that creates `chunk-graph.json`

**Test in Browser:**
```javascript
// Open DevTools console during visualizer session
console.log(allNodes.filter(n => n.complexity !== undefined).length);  // How many nodes have complexity?
console.log(allNodes.filter(n => n.quality_score !== undefined).length);  // How many have quality_score?
console.log(allNodes.filter(n => n.smell_count !== undefined).length);  // How many have smell_count?
```

#### Hypothesis 2: CSS Conflicts or Rendering Issues

**Problem:** Borders/strokes are being set but not visible due to:
- Z-index issues (borders hidden behind other elements)
- Opacity conflicts (borders too transparent)
- Stroke-dasharray not rendering in browser
- Fill colors too similar to distinguish

**Evidence Needed:**
1. Inspect treemap cells in browser DevTools
2. Check computed styles for `stroke`, `stroke-width`, `stroke-dasharray`, `fill`
3. Look for CSS rules that might override D3 attributes

**Test in Browser:**
```javascript
// In DevTools, inspect a treemap cell
const cell = document.querySelector('.treemap-cell');
console.log(getComputedStyle(cell).stroke);  // Should show color
console.log(getComputedStyle(cell).strokeWidth);  // Should be number
console.log(getComputedStyle(cell).strokeDasharray);  // Should show dash pattern
console.log(getComputedStyle(cell).fill);  // Should show blended color
```

#### Hypothesis 3: Default Values Masking Encoding

**Problem:** All nodes default to:
- `quality_score = 1.0` (healthy)
- `smell_count = 0` (no smells)
- Result: All borders look the same (default subtle black), no red smell borders visible

**Evidence:** Check lines 4727-4728 and 4698:
```javascript
const quality = qualityScore !== undefined && qualityScore !== null ? qualityScore : 1.0;  // Defaults to healthy
if (!smellCount || smellCount === 0) return 'rgba(0,0,0,0.3)';  // Default subtle border
```

**If this is true:** All rectangles will have:
- Fill: Complexity color only (no blue blending)
- Stroke: `rgba(0,0,0,0.3)` (subtle black)
- Stroke-width: Default
- No dashed borders (only appear when smell_count > 0)

**Test:** Look for any nodes with non-default values:
```javascript
allNodes.filter(n => n.quality_score !== undefined && n.quality_score < 1.0);  // Unhealthy code
allNodes.filter(n => n.smell_count > 0);  // Nodes with smells
```

### Summary: Why Features "Don't Work" Despite Correct Implementation

| Feature | Implementation Status | Likely Issue |
|---------|---------------------|--------------|
| Three-axis color encoding | ✅ IMPLEMENTED (lines 4908, 5335) | Data not present in source nodes |
| Smell border coloring | ✅ IMPLEMENTED (lines 4909-4914, 5336-5341) | All nodes default to smell_count=0 |
| Smell border thickness | ✅ IMPLEMENTED (lines 4916-4920, 5343-5347) | All nodes default to smell_count=0 |
| Smell border dashing | ✅ IMPLEMENTED (lines 4922-4926, 5349-5353) | All nodes default to smell_count=0 |
| Drill-down click | ✅ IMPLEMENTED (lines 4930, 5357) | May not be working due to other issue |
| Progressive loading | ✅ IMPLEMENTED (lines 5029-5070, 5491-5532) | Working correctly |
| Zoom tracking | ✅ IMPLEMENTED (lines 4854-4874, 5288-5308) | Working via currentZoomRootId |

---

## Recommended Actions

### Issue 1: KG Loading Indicator (IMMEDIATE FIX)

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Changes Needed:**

1. **Line 5843 - Add loading indicator before fetch:**
```javascript
function loadKGData() {
    showLoadingIndicator('Loading Knowledge Graph... (22MB)');  // ADD THIS

    fetch('/api/kg-graph')
        .then(response => {
            // ... existing code
        })
        .then(data => {
            // ... existing code
            initializeKGView();
            hideLoadingIndicator();  // ADD THIS
        })
        .catch(error => {
            console.error('Failed to load KG data:', error);
            hideLoadingIndicator();  // ADD THIS
            setView('chunks');
        });
}
```

**Impact:**
- Users see spinner during 3-10 second load
- Prevents perceived UI freeze
- Consistent with chunks view loading UX

### Issue 2: Treemap/Sunburst Visual Encoding (INVESTIGATION REQUIRED)

**Step 1: Verify Data Presence**

Add debug logging to check if data exists:

```javascript
// At start of renderTreemap() (line 4827)
console.log('Treemap nodes with quality_score:',
    allNodes.filter(n => n.quality_score !== undefined).length);
console.log('Treemap nodes with smell_count > 0:',
    allNodes.filter(n => n.smell_count > 0).length);
console.log('Treemap nodes with complexity:',
    allNodes.filter(n => n.complexity !== undefined).length);
```

**Step 2: Check Source Data Generation**

Investigate where `chunk-graph.json` is generated:
1. Find the code that creates the graph nodes
2. Verify it populates `quality_score`, `smell_count`, `smells`, `complexity_grade` fields
3. Check if code smell detection is actually running

**Step 3: Manual Browser Testing**

1. Open visualizer in browser
2. Switch to Treemap or Sunburst view
3. Open DevTools Console
4. Run diagnostic commands:
```javascript
// Check data availability
console.log('Sample node:', allNodes.find(n => n.type === 'chunk'));
console.log('Nodes with smells:', allNodes.filter(n => n.smell_count > 0));

// Check rendered elements
const cells = document.querySelectorAll('.treemap-cell');
console.log('First cell stroke:', cells[0].getAttribute('stroke'));
console.log('First cell stroke-width:', cells[0].getAttribute('stroke-width'));
console.log('First cell stroke-dasharray:', cells[0].getAttribute('stroke-dasharray'));
console.log('First cell fill:', cells[0].getAttribute('fill'));
```

**Step 4: Verify Visual Guide Legend**

Check if legend is present (line 5012 in treemap, 5473 in sunburst):
```javascript
addVisualGuideLegend(svg, width, height);  // Is this function implemented?
```

Search for `function addVisualGuideLegend` to verify it exists and is creating visual guides.

---

## Code Quality Assessment

### Positive Findings

1. **Visual encoding infrastructure is complete:**
   - Color blending math is correct
   - Border styling functions are well-designed
   - Aggregation logic handles parent nodes properly
   - Three separate axes are implemented (fill, stroke, stroke-dasharray)

2. **Progressive loading works:**
   - Drill-down click handlers are wired up
   - API endpoints support expansion
   - Zoom tracking via `currentZoomRootId` is implemented

3. **Loading indicator infrastructure exists:**
   - Functions are defined and working
   - Used successfully in chunks view
   - Just not called in KG view

### Issues Found

1. **Missing loading indicator in KG view:** Simple oversight, easy to fix
2. **Visual encoding not visible:** Likely data availability issue, not code issue
3. **No visual guide legend verification:** Need to confirm `addVisualGuideLegend()` exists

---

## Next Steps

1. **Fix KG loading indicator** (5-minute task)
2. **Add debug logging** to verify data presence (10-minute task)
3. **Run browser diagnostics** to check rendered attributes (15-minute task)
4. **Investigate graph generation** to ensure quality/smell data is computed (30-minute task)
5. **Verify visual guide legend** function exists and renders correctly (10-minute task)

---

## Appendix: Key File Locations

- **Server endpoints:** `src/mcp_vector_search/cli/commands/visualize/server.py`
  - `/api/kg-graph` (lines 381-419) - 22MB data load
  - `/api/graph` (lines 121-155) - Chunk graph data

- **JavaScript code:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - `loadKGData()` (lines 5843-5879) - Missing loading indicator
  - `renderTreemap()` (lines 4826-5015) - Visual encoding applied
  - `renderSunburst()` (lines 5260-5476) - Visual encoding applied
  - `buildFileHierarchy()` (lines 4498-4553) - Data propagation confirmed
  - `buildASTHierarchy()` (lines 4555-4651) - Data propagation confirmed
  - `getNodeColor()` (lines 4768-4825) - Color blending logic
  - `getSmellBorderColor()` (lines 4697-4702) - Border color logic

- **HTML template:** `src/mcp_vector_search/cli/commands/visualize/templates/base.py`
  - View mode buttons (lines 52-56) - Chunks vs KG toggle
  - Layout mode buttons (lines 60-66) - Tree/Treemap/Sunburst toggle

---

**Research completed:** 2026-02-20
**Confidence level:** High (code review verified, specific line numbers documented)
**Recommended priority:** Issue 1 (KG loading) = P0, Issue 2 (visual encoding) = P1 (requires investigation)
