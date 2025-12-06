# D3.js Automatic Spacing Configuration Research

**Research Date**: December 5, 2025
**Context**: Current implementation uses hardcoded spacing values (800px) in compact folder layout
**Goal**: Replace hardcoded values with automatic spacing that adapts to graph size
**Location**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

---

## Executive Summary

The current D3.js force-directed graph implementation uses hardcoded spacing values (800px) in the `positionNodesCompactly()` function. Research shows that D3.js best practices recommend calculating spacing dynamically based on graph density (ratio of node count to viewport area) using adaptive formulas. Industry-standard approaches combine viewport dimensions, node count, and force simulation parameters to achieve automatic spacing without manual pixel specification.

**Key Recommendation**: Implement density-based spacing using the formula:
```javascript
const k = Math.sqrt(nodes.length / (width * height))
spacing = Math.sqrt(width * height / nodeCount) * scaleFactor
```

---

## 1. Current Implementation Analysis

### Location and Structure
- **File**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
- **Function**: `positionNodesCompactly()` (lines 133-173)
- **Viewport**: Derived from `window.innerWidth` and `window.innerHeight` (lines 15-16)

### Current Hardcoded Values
```javascript
// Line 141: Hardcoded spacing for folders
const spacing = 800; // Extreme spacing: prevent any overlap whatsoever

// Line 157: Hardcoded cluster radius for outliers
const clusterRadius = 800; // Very wide spiral: maximum room
```

### How Current Implementation Works

**Folder Positioning (Grid Layout):**
```javascript
if (folders.length > 0) {
    const cols = Math.ceil(Math.sqrt(folders.length));
    const spacing = 800; // HARDCODED
    const startX = width / 2 - (cols * spacing) / 2;
    const startY = height / 2 - (Math.ceil(folders.length / cols) * spacing) / 2;

    folders.forEach((folder, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        folder.x = startX + col * spacing;
        folder.y = startY + row * spacing;
        folder.fx = folder.x; // Fix position initially
        folder.fy = folder.y;
    });
}
```

**Outlier Positioning (Spiral Layout):**
```javascript
if (outliers.length > 0) {
    const clusterRadius = 800; // HARDCODED
    outliers.forEach((node, i) => {
        const angle = (i / outliers.length) * 2 * Math.PI;
        const radius = clusterRadius * Math.sqrt(i / outliers.length);
        node.x = width / 2 + radius * Math.cos(angle);
        node.y = height / 2 + radius * Math.sin(angle);
    });
}
```

### Problems with Current Approach

1. **Fixed spacing doesn't adapt to viewport size**
   - 800px may be too large for small screens (mobile, laptop)
   - May be too small for ultra-wide monitors or large displays

2. **Doesn't account for node count**
   - Same 800px spacing for 5 folders vs. 50 folders
   - Can cause unnecessary whitespace or overcrowding

3. **No relationship to force simulation parameters**
   - Current charge: `-30` for directories, `-60` for others (lines 258-264)
   - Current link distance: `40-100` (lines 238-245)
   - Spacing should coordinate with these forces

4. **Collision radius mismatch**
   - Collision radius: `24-30` pixels (lines 276-283)
   - Spacing: 800 pixels
   - Massive gap between collision bounds and initial spacing

---

## 2. D3.js Best Practices for Automatic Spacing

### 2.1 Density-Based Spacing Formula

**Industry Standard Approach** (from Stack Overflow and D3 community):

```javascript
// Calculate graph density
const k = Math.sqrt(nodes.length / (width * height));

// Set force parameters based on density
simulation
    .force("charge", d3.forceManyBody().strength(-10 / k))
    .force("gravity", d3.forceCenter(width / 2, height / 2).strength(100 * k));
```

**Why this works:**
- `nodes.length / (width * height)` = graph density (nodes per pixel²)
- Square root scaling accounts for inverse-square law of charge force
- Gravity scales linearly with distance from center, increases with density

### 2.2 Adaptive Spacing Based on Viewport

**Formula for initial spacing:**
```javascript
// Calculate available area per node
const areaPerNode = (width * height) / nodes.length;
const spacing = Math.sqrt(areaPerNode) * scaleFactor;
```

Where:
- `scaleFactor` = 0.5 to 1.5 depending on desired compactness
- 0.5 = tighter packing (50% of available space)
- 1.0 = balanced spacing
- 1.5 = looser packing (50% extra space)

**Alternative formula** (fixed spacing with density adjustment):
```javascript
const baseSpacing = 150; // Base spacing in pixels
const densityFactor = Math.min(1.0, Math.sqrt((width * height) / (nodes.length * 10000)));
const spacing = baseSpacing * densityFactor;
```

### 2.3 Force Simulation Parameter Coordination

**Link distance** should scale with spacing:
```javascript
const spacing = calculateSpacing(nodes.length, width, height);
const linkDistance = spacing * 0.3; // Links should be 30% of spacing
```

**Charge strength** should scale inversely with density:
```javascript
const k = Math.sqrt(nodes.length / (width * height));
const chargeStrength = -30 / k; // Weaker repulsion for dense graphs
```

**Collision radius** should be independent:
```javascript
const collisionRadius = 30; // Based on visual node size, not spacing
```

---

## 3. Common Approaches in Graph Visualization Libraries

### 3.1 Observable D3 Examples

**Approach 1: Viewport-based scaling**
```javascript
const width = window.innerWidth;
const height = window.innerHeight;
const nodeRadius = 5;

simulation
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("charge", d3.forceManyBody().strength(-30))
    .force("link", d3.forceLink().distance(50))
    .force("collide", d3.forceCollide(nodeRadius + 1));
```

**Approach 2: Density-aware initialization**
```javascript
const density = nodes.length / (width * height);
const scale = Math.sqrt(1 / density);

nodes.forEach((node, i) => {
    node.x = (i % cols) * scale * 100;
    node.y = Math.floor(i / cols) * scale * 100;
});
```

### 3.2 NebulaGraph Studio Optimization

NebulaGraph uses **adaptive force parameters** based on graph complexity:

```javascript
// Base parameters
const baseCharge = -30;
const baseDistance = 50;

// Scale based on node count
const nodeScale = Math.min(1, 100 / nodes.length);
const charge = baseCharge * nodeScale;
const distance = baseDistance * nodeScale;

simulation
    .force("charge", d3.forceManyBody().strength(charge))
    .force("link", d3.forceLink().distance(distance));
```

### 3.3 Size-Based Categorization

**Small graphs (< 50 nodes):**
- Can use larger spacing (200-300px)
- Stronger charge forces (-100 to -200)
- Allow more whitespace

**Medium graphs (50-500 nodes):**
- Moderate spacing (100-200px)
- Balanced charge forces (-50 to -100)
- Density-based adjustments

**Large graphs (> 500 nodes):**
- Tight spacing (50-100px)
- Weaker charge forces (-20 to -50)
- Aggressive collision prevention

---

## 4. Recommended Automatic Spacing Approach

### 4.1 Adaptive Spacing Formula

**Recommended implementation:**

```javascript
function calculateAdaptiveSpacing(nodeCount, width, height, spacingMode = 'balanced') {
    // Calculate available area per node
    const areaPerNode = (width * height) / nodeCount;
    const baseSpacing = Math.sqrt(areaPerNode);

    // Apply mode-specific scaling
    const modeScales = {
        'tight': 0.4,
        'balanced': 0.6,
        'loose': 0.8
    };
    const scaleFactor = modeScales[spacingMode] || 0.6;

    // Apply min/max bounds
    const spacing = Math.max(80, Math.min(300, baseSpacing * scaleFactor));

    return spacing;
}

// Usage in positionNodesCompactly()
function positionNodesCompactly(nodes) {
    const folders = nodes.filter(n => n.type === 'directory');
    const outliers = nodes.filter(n => n.type !== 'directory');

    // Calculate adaptive spacing
    const folderSpacing = calculateAdaptiveSpacing(
        folders.length || 1,
        width,
        height,
        'balanced'
    );

    const clusterRadius = calculateAdaptiveSpacing(
        outliers.length || 1,
        width * 0.5, // Use half viewport for cluster
        height * 0.5,
        'tight'
    ) * 2; // Double for spiral radius

    // ... rest of positioning logic using folderSpacing and clusterRadius
}
```

### 4.2 Coordinated Force Parameters

**Update force simulation to match spacing:**

```javascript
function calculateForceParameters(nodeCount, width, height, spacing) {
    // Density-based scaling
    const k = Math.sqrt(nodeCount / (width * height));

    return {
        linkDistance: spacing * 0.25, // 25% of spacing
        chargeStrength: -10 / k,       // Weaker for dense graphs
        collideRadius: 30,             // Fixed based on visual size
        centerStrength: 0.05 + (0.1 * k), // Stronger for dense graphs
        radialStrength: 0.05 + (0.15 * k)  // Stronger for dense graphs
    };
}

// Apply in renderGraph()
const spacing = calculateAdaptiveSpacing(visibleNodesList.length, width, height);
const forceParams = calculateForceParameters(visibleNodesList.length, width, height, spacing);

simulation = d3.forceSimulation(visibleNodesList)
    .force("link", d3.forceLink(visibleLinks)
        .id(d => d.id)
        .distance(forceParams.linkDistance)
        // ... rest of link config
    )
    .force("charge", d3.forceManyBody()
        .strength(d => {
            if (d.type === 'directory') {
                return forceParams.chargeStrength * 3; // Directories repel more
            }
            return forceParams.chargeStrength * 6; // Files repel more
        })
    )
    .force("collision", d3.forceCollide()
        .radius(forceParams.collideRadius)
        .strength(1.0)
    )
    .force("center", d3.forceCenter(width / 2, height / 2)
        .strength(forceParams.centerStrength)
    )
    .force("radial", d3.forceRadial(100, width / 2, height / 2)
        .strength(d => {
            if (d.type === 'directory') return 0;
            return forceParams.radialStrength;
        })
    );
```

### 4.3 Size-Based Adjustments

**Add graph size categorization:**

```javascript
function getGraphSizeCategory(nodeCount) {
    if (nodeCount < 50) return 'small';
    if (nodeCount < 500) return 'medium';
    return 'large';
}

function getSpacingForCategory(category, width, height) {
    const baseFactors = {
        'small': { scale: 0.8, minSpacing: 150, maxSpacing: 400 },
        'medium': { scale: 0.6, minSpacing: 100, maxSpacing: 250 },
        'large': { scale: 0.4, minSpacing: 60, maxSpacing: 150 }
    };

    const config = baseFactors[category];
    const area = width * height;
    // ... calculate spacing using config
}
```

---

## 5. Code Changes Needed

### 5.1 Add Helper Functions

**Location**: `scripts.py` after `get_graph_visualization_functions()`

```python
def get_spacing_calculation_functions() -> str:
    """Get automatic spacing calculation functions.

    Returns:
        JavaScript string for spacing calculations
    """
    return """
        // Calculate adaptive spacing based on node count and viewport
        function calculateAdaptiveSpacing(nodeCount, viewportWidth, viewportHeight, mode = 'balanced') {
            if (nodeCount === 0) return 100; // Fallback for empty graphs

            // Calculate available area per node
            const areaPerNode = (viewportWidth * viewportHeight) / nodeCount;
            const baseSpacing = Math.sqrt(areaPerNode);

            // Mode-specific scaling
            const modeScales = {
                'tight': 0.4,
                'balanced': 0.6,
                'loose': 0.8
            };
            const scaleFactor = modeScales[mode] || 0.6;

            // Calculate spacing with bounds
            const spacing = baseSpacing * scaleFactor;

            // Apply min/max constraints based on graph size
            const nodeCountCategory = getNodeCountCategory(nodeCount);
            const bounds = getSpacingBounds(nodeCountCategory);

            return Math.max(bounds.min, Math.min(bounds.max, spacing));
        }

        function getNodeCountCategory(nodeCount) {
            if (nodeCount < 50) return 'small';
            if (nodeCount < 500) return 'medium';
            return 'large';
        }

        function getSpacingBounds(category) {
            const bounds = {
                'small': { min: 150, max: 400 },
                'medium': { min: 100, max: 250 },
                'large': { min: 60, max: 150 }
            };
            return bounds[category] || bounds['medium'];
        }

        // Calculate force parameters based on graph density
        function calculateForceParameters(nodeCount, viewportWidth, viewportHeight, spacing) {
            // Graph density (nodes per pixel²)
            const density = nodeCount / (viewportWidth * viewportHeight);
            const k = Math.sqrt(density);

            return {
                linkDistance: spacing * 0.25,
                chargeDirectory: -10 / k * 3,    // Directories repel 3x
                chargeOther: -10 / k * 6,        // Other nodes repel 6x
                collideRadius: 30,               // Fixed collision size
                centerStrength: 0.05 + (0.1 * k),
                radialStrength: 0.05 + (0.15 * k)
            };
        }
    """
```

### 5.2 Update `positionNodesCompactly()`

**Replace lines 133-173:**

```python
def get_positioning_functions() -> str:
    """Get node positioning functions with automatic spacing.

    Returns:
        JavaScript string for node positioning
    """
    return """
        // Position ALL nodes in an adaptive initial layout
        function positionNodesCompactly(nodes) {
            const folders = nodes.filter(n => n.type === 'directory');
            const outliers = nodes.filter(n => n.type !== 'directory');

            // Calculate adaptive spacing for folders (grid layout)
            if (folders.length > 0) {
                const folderSpacing = calculateAdaptiveSpacing(
                    folders.length,
                    width,
                    height,
                    'balanced'
                );

                const cols = Math.ceil(Math.sqrt(folders.length));
                const startX = width / 2 - (cols * folderSpacing) / 2;
                const startY = height / 2 - (Math.ceil(folders.length / cols) * folderSpacing) / 2;

                folders.forEach((folder, i) => {
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    folder.x = startX + col * folderSpacing;
                    folder.y = startY + row * folderSpacing;
                    folder.fx = folder.x; // Fix position initially
                    folder.fy = folder.y;
                });
            }

            // Calculate adaptive radius for outliers (spiral layout)
            if (outliers.length > 0) {
                // Use half viewport for cluster area, then double result for spiral radius
                const clusterSpacing = calculateAdaptiveSpacing(
                    outliers.length,
                    width * 0.6,
                    height * 0.6,
                    'tight'
                );
                const clusterRadius = clusterSpacing * 2;

                outliers.forEach((node, i) => {
                    const angle = (i / outliers.length) * 2 * Math.PI;
                    const radius = clusterRadius * Math.sqrt(i / outliers.length);
                    node.x = width / 2 + radius * Math.cos(angle);
                    node.y = height / 2 + radius * Math.sin(angle);
                });
            }

            // Release fixed folder positions after settling
            setTimeout(() => {
                folders.forEach(folder => {
                    folder.fx = null;
                    folder.fy = null;
                });
            }, 1000);
        }
    """
```

### 5.3 Update Force Simulation in `renderGraph()`

**Modify lines 235-287 to use calculated parameters:**

```python
# In get_graph_visualization_functions(), update renderGraph():
"""
function renderGraph() {
    const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
    const visibleLinks = allLinks.filter(l =>
        visibleNodes.has(l.source.id || l.source) &&
        visibleNodes.has(l.target.id || l.target)
    );

    // Calculate adaptive spacing and force parameters
    const adaptiveSpacing = calculateAdaptiveSpacing(
        visibleNodesList.length,
        width,
        height,
        'balanced'
    );
    const forceParams = calculateForceParameters(
        visibleNodesList.length,
        width,
        height,
        adaptiveSpacing
    );

    simulation = d3.forceSimulation(visibleNodesList)
        .force("link", d3.forceLink(visibleLinks)
            .id(d => d.id)
            .distance(d => {
                // Use calculated link distance as base
                if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') {
                    return forceParams.linkDistance * 0.5; // Tighter for hierarchy
                }
                if (d.is_cycle) return forceParams.linkDistance * 1.0;
                if (d.type === 'semantic') return forceParams.linkDistance * 1.5;
                return forceParams.linkDistance;
            })
            .strength(d => {
                // Keep existing strength logic
                if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') {
                    return 0.8;
                }
                if (d.is_cycle) return 0.4;
                if (d.type === 'semantic') return 0.3;
                return 0.7;
            })
        )
        .force("charge", d3.forceManyBody()
            .strength(d => {
                // Use calculated charge based on node type
                if (d.type === 'directory') {
                    return forceParams.chargeDirectory;
                }
                return forceParams.chargeOther;
            })
        )
        .force("center", d3.forceCenter(width / 2, height / 2)
            .strength(forceParams.centerStrength)
        )
        .force("radial", d3.forceRadial(100, width / 2, height / 2)
            .strength(d => {
                if (d.type === 'directory') return 0;
                return forceParams.radialStrength;
            })
        )
        .force("collision", d3.forceCollide()
            .radius(forceParams.collideRadius)
            .strength(1.0)
        )
        .velocityDecay(0.6)
        .alphaDecay(0.02);

    // ... rest of renderGraph() unchanged
}
"""
```

### 5.4 Update `get_all_scripts()`

**Modify function to include new helper (line 1462):**

```python
def get_all_scripts() -> str:
    """Get all JavaScript code combined.

    Returns:
        Complete JavaScript string for the visualization
    """
    return "".join(
        [
            get_d3_initialization(),
            get_file_type_functions(),
            get_spacing_calculation_functions(),  # NEW
            get_graph_visualization_functions(),
            get_zoom_and_navigation_functions(),
            get_interaction_handlers(),
            get_tooltip_logic(),
            get_drag_and_stats_functions(),
            get_content_pane_functions(),
            get_data_loading_logic(),
        ]
    )
```

---

## 6. Potential Issues and Edge Cases

### 6.1 Viewport Size Edge Cases

**Issue**: Very small viewports (mobile devices)
- `width = 375px, height = 667px` (iPhone)
- With 50 nodes: `areaPerNode = 5,002px²`, `baseSpacing = 70.7px`
- With 0.6 scale: `spacing = 42.4px` → clamped to `min = 60px` ✓

**Issue**: Very large viewports (ultra-wide monitors)
- `width = 3840px, height = 2160px` (4K)
- With 10 nodes: `areaPerNode = 829,440px²`, `baseSpacing = 910px`
- With 0.6 scale: `spacing = 546px` → clamped to `max = 400px` ✓

**Mitigation**: Min/max bounds prevent extreme values

### 6.2 Node Count Edge Cases

**Issue**: Single node
- `nodeCount = 1`: `areaPerNode = width * height`
- Spacing would be huge → clamped to max bounds ✓

**Issue**: Zero nodes
- `nodeCount = 0`: Division by zero
- **Fix**: Add guard clause returning default spacing (100px)

**Issue**: Thousands of nodes
- `nodeCount = 5000` on 1920x1080 viewport
- `areaPerNode = 415px²`, `baseSpacing = 20.4px`
- With 0.6 scale: `spacing = 12.2px` → clamped to `min = 60px`
- **Problem**: May still be too dense for very large graphs
- **Solution**: Consider additional categorization for "very large" (>1000 nodes)

### 6.3 Mixed Node Types

**Issue**: Unequal distribution
- 2 folders + 100 outliers
- Folder spacing calculated for 2 nodes → very large spacing
- Outlier spacing calculated for 100 nodes → much tighter
- **Result**: Asymmetric layout, but this is intentional ✓

### 6.4 Force Simulation Stability

**Issue**: Calculated charge strength may be too weak
- Very dense graphs (1000+ nodes) → `k = large` → `charge = -10/k = very small`
- Nodes may not repel enough, causing overlap
- **Mitigation**: Collision force with `strength = 1.0` prevents overlap
- **Monitoring**: Test with various node counts to validate

**Issue**: Link distance too short
- `linkDistance = spacing * 0.25`
- For minimum spacing (60px): `linkDistance = 15px`
- May be too short for clear separation
- **Solution**: Use minimum link distance of 30px:
  ```javascript
  linkDistance: Math.max(30, spacing * 0.25)
  ```

### 6.5 Performance Considerations

**Issue**: Recalculating on every render
- `calculateAdaptiveSpacing()` called in `positionNodesCompactly()` and `renderGraph()`
- Minimal computation, but could be cached
- **Optimization**: Cache spacing value per node count
  ```javascript
  const spacingCache = new Map();
  function getCachedSpacing(nodeCount, width, height, mode) {
      const key = `${nodeCount}-${width}-${height}-${mode}`;
      if (!spacingCache.has(key)) {
          spacingCache.set(key, calculateAdaptiveSpacing(...));
      }
      return spacingCache.get(key);
  }
  ```

### 6.6 Responsive Behavior

**Issue**: Viewport resize during interaction
- User resizes window after graph is positioned
- Spacing should update, but current implementation doesn't handle resize
- **Solution**: Add window resize listener:
  ```javascript
  window.addEventListener('resize', debounce(() => {
      width = window.innerWidth;
      height = window.innerHeight;
      resetView(); // Re-layout with new spacing
  }, 250));
  ```

---

## 7. Testing Strategy

### 7.1 Manual Testing Scenarios

**Scenario 1: Small graph (< 50 nodes)**
- Expected: Larger spacing (150-400px range)
- Validate: Nodes spread out comfortably, no overlap

**Scenario 2: Medium graph (50-500 nodes)**
- Expected: Moderate spacing (100-250px range)
- Validate: Balanced layout, no excessive whitespace

**Scenario 3: Large graph (> 500 nodes)**
- Expected: Tight spacing (60-150px range)
- Validate: Compact layout, collision detection prevents overlap

**Scenario 4: Viewport variations**
- Test on: Mobile (375×667), Laptop (1366×768), Desktop (1920×1080), 4K (3840×2160)
- Expected: Spacing scales proportionally with viewport size

**Scenario 5: Mixed node types**
- Graph with: 5 directories + 50 files
- Expected: Directories in loose grid, files in tighter cluster

### 7.2 Automated Testing (Optional)

**Unit tests for spacing calculations:**
```javascript
describe('calculateAdaptiveSpacing', () => {
    it('should handle zero nodes gracefully', () => {
        expect(calculateAdaptiveSpacing(0, 1920, 1080)).toBe(100);
    });

    it('should respect minimum bounds for large graphs', () => {
        const spacing = calculateAdaptiveSpacing(1000, 1920, 1080, 'balanced');
        expect(spacing).toBeGreaterThanOrEqual(60);
    });

    it('should respect maximum bounds for small graphs', () => {
        const spacing = calculateAdaptiveSpacing(5, 1920, 1080, 'balanced');
        expect(spacing).toBeLessThanOrEqual(400);
    });

    it('should scale with viewport size', () => {
        const small = calculateAdaptiveSpacing(50, 800, 600, 'balanced');
        const large = calculateAdaptiveSpacing(50, 1920, 1080, 'balanced');
        expect(large).toBeGreaterThan(small);
    });
});
```

### 7.3 Visual Regression Testing

**Capture screenshots for:**
- Before/after comparison with hardcoded spacing
- Different node counts: 10, 50, 100, 500, 1000
- Different viewports: mobile, tablet, desktop, 4K
- Different graph types: folder-heavy, file-heavy, mixed

**Metrics to compare:**
- Average inter-node distance
- Overlap count (should be 0)
- Viewport utilization (% of viewport containing nodes)
- Zoom level required to fit all nodes

---

## 8. Implementation Checklist

- [ ] Add `get_spacing_calculation_functions()` helper
- [ ] Update `positionNodesCompactly()` to use adaptive spacing
- [ ] Update `renderGraph()` force parameters to use calculated values
- [ ] Add zero-node guard clause to prevent division by zero
- [ ] Add minimum link distance (30px) to prevent too-short links
- [ ] Update `get_all_scripts()` to include spacing calculations
- [ ] Add window resize handler for responsive spacing (optional)
- [ ] Add spacing cache for performance (optional)
- [ ] Test with graphs: 10, 50, 100, 500, 1000+ nodes
- [ ] Test on viewports: mobile, laptop, desktop, 4K
- [ ] Visual regression test: compare before/after layouts
- [ ] Document new automatic spacing behavior in user guide

---

## 9. References

### Web Resources
1. **D3.js Official Documentation**
   - Force Simulation: https://d3js.org/d3-force/simulation
   - Force Link: https://d3js.org/d3-force/link
   - Force Many-Body: https://d3js.org/d3-force/many-body

2. **Stack Overflow Discussions**
   - Optimizing charge, linkDistance, and gravity: https://stackoverflow.com/questions/15076157
   - Responsive force layout: https://stackoverflow.com/questions/11942500
   - Charge based on size: https://stackoverflow.com/questions/9901565

3. **Community Articles**
   - D3 Force Layout Guide: https://www.d3indepth.com/force-layout/
   - Getting started with D3 force simulations: https://medium.com/@bryony_17728
   - NebulaGraph D3-Force Optimization: https://www.nebula-graph.io/posts/d3-force-layout-optimization

### Code Examples
- D3 Force GitHub: https://github.com/d3/d3-force
- 3D Force Graph: https://github.com/vasturiano/3d-force-graph
- Force-directed graph examples: Observable HQ D3 gallery

---

## 10. Conclusion

Replacing hardcoded 800px spacing with automatic density-based calculations will significantly improve the visualization's adaptability across different graph sizes and viewport dimensions. The recommended implementation uses industry-standard formulas from the D3 community, coordinating spacing with force simulation parameters for optimal layout quality.

**Key Benefits:**
1. **Adaptive to viewport size**: Works on mobile, laptop, desktop, and ultra-wide displays
2. **Scales with node count**: Automatically tightens for large graphs, loosens for small graphs
3. **Coordinated forces**: Link distance, charge strength, and spacing work together
4. **Bounded safety**: Min/max constraints prevent extreme values
5. **Customizable modes**: 'tight', 'balanced', 'loose' for different use cases

**Implementation Effort:**
- **Estimated time**: 2-3 hours (code changes + testing)
- **Risk level**: Low (fallback to defaults if calculations fail)
- **Breaking changes**: None (purely internal calculation change)

**Next Steps:**
1. Implement code changes from Section 5
2. Test with sample graphs (Section 7.1)
3. Validate edge cases (Section 6)
4. Document behavior for users
