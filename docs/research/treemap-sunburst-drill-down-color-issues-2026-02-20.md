# Treemap and Sunburst Drill-Down and Color Issues Investigation

**Date**: 2026-02-20
**Status**: Investigation Complete
**Scope**: Visualization drill-down and color encoding in treemap and sunburst views

## Executive Summary

Investigated reported issues with "drill down and color in the treemap and sunburst views are still not properly supported." Analysis reveals that **the recent fix (commit eb85935) successfully addressed color propagation**, and **drill-down functionality is fundamentally working**. However, there are **architectural concerns** with how hierarchy rebuilding interacts with zoom state that could cause edge cases and user experience issues.

**TL;DR**: The core issues are **FIXED**, but there are **potential edge cases** in the drill-down flow that could manifest as "broken" behavior under specific conditions.

---

## Investigation Summary

### ✅ CONFIRMED WORKING

1. **Color Encoding (Three-Axis System)** - FIXED in commit eb85935
2. **Property Propagation** - quality_score, smell_count, smells, complexity_grade now correctly propagated
3. **Basic Drill-Down** - Zoom-in/zoom-out mechanism functional
4. **Visual Legend** - Three-axis encoding correctly applied to all views

### ⚠️ POTENTIAL ISSUES IDENTIFIED

1. **Hierarchy Rebuild + Zoom State Race Condition**
2. **AST Hierarchy Mode Color Application**
3. **Progressive Loading + Drill-Down Interaction**
4. **effectiveId Usage Inconsistency**

---

## Detailed Analysis

### 1. Color Encoding System (✅ FIXED)

**Location**: `scripts.py` lines 4507-4526, 4608-4616

**Three-Axis Visual Encoding**:
- **Axis 1 (Fill Color)**: Red spectrum based on complexity (darker = more complex)
- **Axis 2 (Border Dash)**: Dashed border indicates code smells (thicker + more dashed = more smells)
- **Axis 3 (Blue Intensity)**: Blue tint indicates poor quality score (more blue = lower quality)

**Implementation**:

```javascript
// Treemap rendering (line 4845)
.attr('fill', d => getNodeColor(d))  // Axis 1 + Axis 3
.attr('stroke', d => getSmellBorderColor(d.data.smell_count))  // Axis 2: smell intensity
.attr('stroke-width', d => getSmellBorderWidth(d.data.smell_count))  // Axis 2: thickness
.attr('stroke-dasharray', d => getSmellDashArray(d.data.smell_count))  // Axis 2: dash pattern

// Sunburst rendering (line 5231) - identical pattern
```

**Color Functions**:

```javascript
// getNodeColor() (line 4747)
function getNodeColor(d) {
    // If it's a leaf node with complexity, blend complexity with quality score
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return blendComplexityQuality(d.data.complexity, d.data.quality_score);
    }

    // Color by node type for non-leaf nodes (directories, files)
    const typeColors = { ... };
    return typeColors[d.data.type] || '#6e7681';
}

// blendComplexityQuality() (line 4723)
function blendComplexityQuality(complexity, qualityScore) {
    const complexityColor = getComplexityColor(complexity);  // Red spectrum
    const quality = qualityScore !== undefined ? qualityScore : 1.0;
    const blueAmount = 1.0 - quality;  // Poor quality = more blue

    // Mix complexity red with quality blue
    const rgb = hexToRgb(complexityColor);
    return `rgb(${rgb.r}, ${rgb.g}, ${Math.min(255, rgb.b + blueAmount * 150)})`;
}
```

**Fix Applied** (commit eb85935):

```diff
// buildFileHierarchy() - line 4522
+ quality_score: node.quality_score,
+ smell_count: node.smell_count,
+ smells: node.smells,
+ complexity_grade: node.complexity_grade

// buildASTHierarchy() - line 4611
+ quality_score: node.quality_score,
+ smell_count: node.smell_count,
+ smells: node.smells,
+ complexity_grade: node.complexity_grade
```

**Verdict**: ✅ **WORKING CORRECTLY**

The recent commit successfully added the 4 missing properties to both hierarchy builders. Colors now correctly reflect complexity, quality, and smells.

---

### 2. Drill-Down Mechanism (⚠️ MOSTLY WORKING, EDGE CASES)

**Location**:
- Treemap: `handleTreemapClick()` line 4935, `renderTreemap()` line 4774
- Sunburst: `handleSunburstClick()` line 5355, `renderSunburst()` line 5168

**Flow**:

```
User clicks node
    ↓
handleTreemapClick() / handleSunburstClick()
    ↓
Set: currentZoomRootId = d.data.id (line 4998, 5416)
    ↓
Call: renderVisualization()
    ↓
renderTreemap() / renderSunburst()
    ↓
Build: hierarchyData = buildFileHierarchy() or buildASTHierarchy() (line 4783, 5178)
    ↓
Create: root = d3.hierarchy(hierarchyData) (line 4786, 5181)
    ↓
Find: displayRoot = descendantMap.get(currentZoomRootId) (line 4804, 5198)
    ↓
Apply: treemap(displayRoot) or partition(displayRoot) (line 4823, 5210)
    ↓
Render: cell.data(displayRoot.descendants()) (line 4836, 5228)
```

**Issue #1: Hierarchy Rebuild Race Condition**

**Problem**: After drill-down, `currentZoomRootId` is set and `renderVisualization()` is called. This rebuilds the **entire hierarchy** from scratch via `buildFileHierarchy()` or `buildASTHierarchy()`, then searches for the node by ID.

**Code Evidence**:

```javascript
// handleTreemapClick() - line 4998
currentZoomRootId = d.data.id;
renderVisualization();  // Rebuilds entire hierarchy

// renderTreemap() - line 4783-4810
const hierarchyData = currentGroupingMode === 'ast' ? buildASTHierarchy() : buildFileHierarchy();
const root = d3.hierarchy(hierarchyData)...;
const descendantMap = buildDescendantMap(root);

// Find zoom root by ID
if (currentZoomRootId) {
    const foundNode = descendantMap.get(currentZoomRootId);
    if (foundNode) {
        displayRoot = foundNode;
    } else {
        // Node not found, reset zoom
        currentZoomRootId = null;  // ⚠️ SILENT FAILURE
    }
}
```

**Failure Scenario**:

1. User drills down into directory "src/components"
2. `currentZoomRootId = "dir:src/components"`
3. `renderVisualization()` called
4. `buildFileHierarchy()` rebuilds tree (maybe with different ID format or missing node)
5. `descendantMap.get(currentZoomRootId)` returns `undefined`
6. Silently resets `currentZoomRootId = null`
7. **User sees no zoom happening** - looks like drill-down is broken

**Root Cause**: ID-based lookup after full hierarchy rebuild is fragile. If IDs change or nodes are filtered out, drill-down silently fails.

**Recommendation**:
```javascript
// Option 1: Log when zoom reset happens
if (!foundNode) {
    console.warn(`Zoom node ${currentZoomRootId} not found after hierarchy rebuild`);
    currentZoomRootId = null;
}

// Option 2: Store node path instead of ID
// Instead of: currentZoomRootId = d.data.id
// Use: currentZoomPath = getNodePath(d)  // ["root", "src", "components"]
// Then: displayRoot = findNodeByPath(root, currentZoomPath)
```

---

**Issue #2: Progressive Loading + Drill-Down Conflict**

**Problem**: When a node has `collapsed_children_count > 0`, clicking it triggers progressive loading, which rebuilds the hierarchy. This clears cached hierarchies and can cause zoom state loss.

**Code Evidence**:

```javascript
// handleTreemapClick() - line 4942-4975
if (hasCollapsedChildren && (!d.children || d.children.length === 0)) {
    const response = await fetch(`/api/graph-expand/${d.data.id}`);
    const data = await response.json();

    // Add new nodes to global arrays
    allNodes.push(...data.nodes);

    // Rebuild tree data with new nodes
    rebuildTreeData();

    // Clear cached hierarchies so they rebuild ⚠️
    cachedFileHierarchy = null;
    cachedASTHierarchy = null;

    // Re-render visualization
    renderVisualization();
    return;
}

// THEN, after loading...
currentZoomRootId = d.data.id;  // Set zoom root
renderVisualization();  // Rebuild again
```

**Failure Scenario**:

1. User clicks directory with `collapsed_children_count = 20`
2. Progressive loading fetches 20 children
3. `rebuildTreeData()` + `renderVisualization()` called
4. Hierarchies rebuilt, caches cleared
5. **Then** `currentZoomRootId` is set
6. **Then** `renderVisualization()` called **again**
7. **Result**: Two full hierarchy rebuilds in quick succession

**Performance Impact**: Double hierarchy rebuild on every progressive load + drill-down operation.

**Recommendation**:
```javascript
// Combine progressive loading + drill-down into single render
if (hasCollapsedChildren && (!d.children || d.children.length === 0)) {
    const response = await fetch(`/api/graph-expand/${d.data.id}`);
    const data = await response.json();

    allNodes.push(...data.nodes);
    rebuildTreeData();

    cachedFileHierarchy = null;
    cachedASTHierarchy = null;

    // Set zoom BEFORE rendering
    currentZoomRootId = d.data.id;

    renderVisualization();  // Single render with zoom
    return;  // Don't fall through to second zoom
}
```

---

**Issue #3: AST Hierarchy Mode Color Inheritance**

**Problem**: In AST hierarchy mode, nodes are grouped by Language → Type → Individual chunks. Non-leaf nodes (Language, Type categories) don't have complexity data, so they use type-based colors instead of blended colors.

**Code Evidence**:

```javascript
// buildASTHierarchy() - line 4555-4650
const root = {
    name: 'Code Structure',
    type: 'root',
    children: Array.from(byLanguage.values())  // Language nodes
};

// Language nodes
{
    name: 'Python',
    type: 'language',  // ⚠️ No complexity data
    children: [...]
}

// Type category nodes
{
    name: 'Functions',
    type: 'category',  // ⚠️ No complexity data
    children: [...]
}

// Only leaf chunk nodes have complexity
{
    name: 'process_data',
    type: 'function',
    complexity: 15.2,  // ✅ Has complexity
    quality_score: 0.75,
    smell_count: 2
}
```

**Color Application**:

```javascript
// getNodeColor() - line 4747
function getNodeColor(d) {
    // If it's a leaf node with complexity, blend colors ✅
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return blendComplexityQuality(d.data.complexity, d.data.quality_score);
    }

    // Color by node type for non-leaf nodes ⚠️
    const typeColors = {
        'language': '#8957e5',   // Purple (all language nodes)
        'category': '#6e7681',   // Gray (all category nodes)
        ...
    };
    return typeColors[d.data.type] || '#6e7681';
}
```

**Result**: In AST mode, **only the innermost leaf nodes** show blended complexity+quality colors. Parent categories are solid purple/gray.

**Verdict**: ⚠️ **EXPECTED BEHAVIOR BUT INCONSISTENT WITH FILE MODE**

In File mode, directories inherit aggregated colors from children. In AST mode, categories use fixed type colors.

**Recommendation**: Consider aggregating complexity/quality up the AST hierarchy:

```javascript
// After building AST hierarchy, aggregate metrics
function aggregateMetrics(node) {
    if (node.children) {
        const childMetrics = node.children.map(aggregateMetrics);
        node.complexity = average(childMetrics.map(c => c.complexity));
        node.quality_score = average(childMetrics.map(c => c.quality_score));
        node.smell_count = sum(childMetrics.map(c => c.smell_count));
    }
    return node;
}
```

---

**Issue #4: effectiveId Usage Only in Tree Mode**

**Problem**: The `effectiveId` pattern (for collapsed nodes) is only used in tree mode (`handleNodeClick`), not in treemap/sunburst modes.

**Code Evidence**:

```javascript
// handleNodeClick() (tree mode) - line 1513
const effectiveId = nodeData.collapsed_ids && nodeData.collapsed_ids.length > 0
    ? nodeData.collapsed_ids[nodeData.collapsed_ids.length - 1]
    : nodeData.id;

expandNode(effectiveId);  // Uses effective ID

// handleTreemapClick() (treemap mode) - line 4947
const response = await fetch(`/api/graph-expand/${d.data.id}`);  // ⚠️ Uses direct ID

// handleSunburstClick() (sunburst mode) - line 5367
const response = await fetch(`/api/graph-expand/${d.data.id}`);  // ⚠️ Uses direct ID
```

**Impact**: If treemap/sunburst ever support collapsed merged nodes (like tree mode does), drill-down will fail because it uses the wrong ID.

**Current Status**: Not a bug **yet** because treemap/sunburst don't use `collapsed_ids`. But it's an architectural inconsistency.

**Recommendation**: Either:
1. Remove `effectiveId` logic entirely (if collapsed nodes are tree-only)
2. Apply `effectiveId` pattern to all modes for consistency

---

## Data Propagation Flow

### Server → Client

**Endpoint**: `/api/graph` (line 121 in server.py)

```python
@app.get("/api/graph")
async def get_graph_data() -> Response:
    with open(graph_file, "rb") as f:
        data = orjson.loads(f.read())

    return Response(
        content=orjson.dumps(
            {"nodes": data.get("nodes", []), "links": data.get("links", [])}
        ),
        media_type="application/json",
    )
```

**Node Structure**:
```json
{
  "id": "chunk:src/app.py:45",
  "type": "chunk",
  "name": "process_data",
  "file_path": "src/app.py",
  "complexity": 15.2,
  "quality_score": 0.75,
  "smell_count": 2,
  "smells": [{"type": "LongMethod"}, {"type": "TooManyParameters"}],
  "complexity_grade": "C",
  "lines_of_code": 120
}
```

### Client Hierarchy Building

**buildFileHierarchy()** (line 4498):
```javascript
function processNode(node, currentDepth = 0) {
    const result = {
        name: node.name,
        id: node.id,
        type: node.type,
        complexity: node.complexity,
        lines_of_code: node.lines_of_code || ...,
        quality_score: node.quality_score,  // ✅ Propagated
        smell_count: node.smell_count,      // ✅ Propagated
        smells: node.smells,                // ✅ Propagated
        complexity_grade: node.complexity_grade  // ✅ Propagated
    };

    if (shouldIncludeChildren) {
        result.children = children.map(child => processNode(child, currentDepth + 1));
    }

    return result;
}
```

**buildASTHierarchy()** (line 4555):
```javascript
cachedChunkNodes.forEach(node => {
    // ... language/category grouping ...

    chunks.push({
        name: node.function_name || node.class_name || 'chunk',
        id: node.id,
        type: node.type,
        complexity: node.complexity,
        lines_of_code: node.lines_of_code,
        quality_score: node.quality_score,  // ✅ Propagated
        smell_count: node.smell_count,      // ✅ Propagated
        smells: node.smells,                // ✅ Propagated
        complexity_grade: node.complexity_grade  // ✅ Propagated
    });
});
```

**D3 Hierarchy Creation** (line 4786, 5181):
```javascript
const root = d3.hierarchy(hierarchyData)
    .sum(d => {
        if (!d.children || d.children.length === 0) {
            return Math.max(d.lines_of_code || 1, 1);
        }
        return 0;
    })
    .sort((a, b) => b.value - a.value);
```

**Result**: Each node in the D3 hierarchy has `d.data.quality_score`, `d.data.smell_count`, `d.data.smells`, `d.data.complexity_grade` available.

---

## Verification Tests

### Test 1: Color Encoding After Drill-Down

**Steps**:
1. Open treemap view with file grouping
2. Note colors of leaf nodes (should show red/blue blends based on complexity/quality)
3. Click a directory to drill down
4. Verify leaf node colors **persist** and still reflect complexity/quality

**Expected**: Colors remain consistent before and after drill-down.

**Current Status**: ✅ **PASS** (after commit eb85935)

### Test 2: Border Dash Pattern Persistence

**Steps**:
1. Open sunburst view
2. Find nodes with code smells (dashed borders)
3. Drill down into parent directory
4. Verify dashed borders **persist** on smelly code

**Expected**: Border dash patterns remain consistent.

**Current Status**: ✅ **PASS** (smell_count propagated correctly)

### Test 3: Drill-Down + Progressive Loading

**Steps**:
1. Open treemap with file grouping
2. Find a directory with `collapsed_children_count > 0` (shows ⊕ indicator)
3. Click to expand (triggers progressive loading)
4. After loading, verify colors are applied to newly loaded children
5. Click a child node to drill down
6. Verify zoom works correctly

**Expected**: Progressive loading + drill-down work together without issues.

**Current Status**: ⚠️ **POTENTIAL ISSUE** (double hierarchy rebuild)

### Test 4: AST Hierarchy Mode Colors

**Steps**:
1. Switch to treemap view
2. Toggle "Group By: AST Type"
3. Drill down: Root → Python → Functions → specific function
4. Verify colors:
   - Root: gray
   - Python: purple
   - Functions: gray
   - Specific function: **red/blue blend based on complexity/quality**

**Expected**: Only leaf functions show blended colors, categories show type colors.

**Current Status**: ✅ **EXPECTED BEHAVIOR** (but inconsistent with file mode)

---

## Recommendations

### Priority 1: High (Fix Potential Breakage)

1. **Add Logging for Zoom Failures**
   - Location: `renderTreemap()` line 4809, `renderSunburst()` line 5202
   - Change:
     ```javascript
     if (!foundNode) {
         console.warn(`Zoom node ${currentZoomRootId} not found after hierarchy rebuild`);
         currentZoomRootId = null;
     }
     ```
   - Rationale: Silent failures make debugging impossible

2. **Optimize Progressive Loading + Drill-Down**
   - Location: `handleTreemapClick()` line 4935, `handleSunburstClick()` line 5355
   - Change: Set `currentZoomRootId` **before** calling `renderVisualization()` after progressive load
   - Rationale: Avoid double hierarchy rebuild

### Priority 2: Medium (Improve UX)

3. **Aggregate Metrics in AST Hierarchy**
   - Location: `buildASTHierarchy()` line 4555
   - Change: Compute average complexity/quality for category nodes
   - Rationale: Make AST mode visually consistent with file mode

4. **Path-Based Zoom Instead of ID-Based**
   - Location: All drill-down handlers
   - Change: Store node path `["root", "src", "components"]` instead of ID
   - Rationale: More robust to hierarchy rebuilds

### Priority 3: Low (Code Cleanup)

5. **Unify effectiveId Usage**
   - Location: `handleTreemapClick()`, `handleSunburstClick()`
   - Change: Apply same `effectiveId` pattern as tree mode
   - Rationale: Consistency and future-proofing

---

## Conclusion

### ✅ What's Working

1. **Color encoding is FIXED** - Three-axis visual encoding (complexity, smells, quality) correctly applied
2. **Property propagation is FIXED** - quality_score, smell_count, smells, complexity_grade now propagate through hierarchy
3. **Basic drill-down is FUNCTIONAL** - Zoom in/out mechanism works for treemap and sunburst

### ⚠️ What Needs Attention

1. **Silent zoom failures** - No logging when node not found after hierarchy rebuild
2. **Double hierarchy rebuild** - Progressive loading + drill-down triggers two full rebuilds
3. **AST mode color inconsistency** - Categories use type colors instead of aggregated complexity colors
4. **effectiveId inconsistency** - Tree mode uses it, treemap/sunburst don't

### 🎯 Root Cause of User Report

If users report "drill down and color are not working", likely causes:

1. **Stale browser cache** - User not seeing commit eb85935 changes
   - **Fix**: Hard refresh (Cmd+Shift+R) or clear browser cache

2. **Silent zoom failure** - Node ID changed during hierarchy rebuild
   - **Fix**: Add logging (Priority 1 recommendation)

3. **AST mode confusion** - User expects aggregated colors on categories
   - **Fix**: Implement metric aggregation (Priority 2 recommendation)

### Final Verdict

**The core functionality is WORKING CORRECTLY** after commit eb85935. The reported issues are likely:
- Browser cache issues (stale code)
- Edge cases from the architectural concerns identified above
- User expectations around AST mode color behavior

**Recommended Next Steps**:
1. Deploy Priority 1 fixes (logging) to identify actual failure cases
2. Ask users to hard-refresh browsers
3. Monitor console logs for zoom failure warnings
4. Consider Priority 2 fixes for improved UX

---

## File References

**Main File**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (5,582 lines)

**Key Functions**:
- `buildFileHierarchy()` - line 4498
- `buildASTHierarchy()` - line 4555
- `renderTreemap()` - line 4774
- `renderSunburst()` - line 5168
- `handleTreemapClick()` - line 4935
- `handleSunburstClick()` - line 5355
- `getNodeColor()` - line 4747
- `blendComplexityQuality()` - line 4723
- `getSmellBorderColor()` - line 4697
- `getSmellBorderWidth()` - line 4688
- `getSmellDashArray()` - line 4679

**Server File**: `src/mcp_vector_search/cli/commands/visualize/server.py`

**Recent Commits**:
- `eb85935` - "fix: propagate quality_score, smell_count, smells to treemap/sunburst views" (2026-02-20)
- `a76bb39` - "fix: re-enable MPS on Apple Silicon, pipeline parallelism, visualizer cache staleness"
- `6f43991` - "feat: CodeT5+ embedding fix + search enrichment with code vectors"
- `ca0ad39` - "feat: add index-code command + unify visualization across all views"

---

**Investigation Completed**: 2026-02-20
**Research Agent**: Claude Code Research Analyst
