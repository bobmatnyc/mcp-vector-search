# Code Graph Visualization Best Practices for Non-Linear Relationships

**Research Date:** December 5, 2025
**Problem:** Force-directed layout creates cluttered "hairball" visualization with 1,447 nodes and 360,823 edges
**Goal:** Find layout algorithms and interaction patterns to reveal hierarchical, modular, and circular dependencies clearly

---

## Executive Summary

After analyzing current state-of-the-art research, academic papers, and production tools, the "hairball problem" in large code graph visualization is best solved through a **combination approach**:

1. **Hierarchical layout algorithms** (Sugiyama/Dagre) for structural clarity
2. **Community detection + clustering** to group related modules
3. **Progressive disclosure** (collapsed by default, expand on demand)
4. **Edge bundling** to reduce visual clutter from 360K edges
5. **Focus+context techniques** (fisheye zoom, semantic zoom) for navigation

**Key Insight:** The hairball is a **design problem, not a computational one**. Don't try to render all 360K edges—redesign around user workflows and show task-relevant subgraphs.

---

## Current Implementation Analysis

### Existing Visualization Stack
- **Technology:** D3.js v7 force-directed layout
- **Data:** 1,447 nodes, 360,823 edges
- **Node Types:** directories, files, functions, classes, methods
- **Edge Types:** containment, calls, imports, semantic similarity, cycles
- **Current Features:**
  - Adaptive spacing calculation
  - Collapse/expand hierarchies
  - Progressive disclosure (start with root nodes only)
  - Content pane for code viewing
  - Cycle detection (circular dependencies highlighted)

### Problems with Force-Directed Layout
1. **Visual clutter:** 360K edges create overlapping mass ("hairball")
2. **Poor hierarchy representation:** No clear visual hierarchy despite logical parent-child relationships
3. **No modular grouping:** Related code not visually clustered
4. **Difficult navigation:** Hard to find specific files/functions in the tangle
5. **Performance:** Force simulation struggles with 1,447 nodes at scale

**Source:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (lines 388-439)

---

## Top 5 Layout Algorithm Approaches

### 1. Hierarchical Layered Layout (Sugiyama/Dagre) ⭐ RECOMMENDED

**What it is:** Organizes nodes into horizontal layers based on graph structure, with edges flowing top-to-bottom.

**Best for:**
- Directed graphs with clear hierarchy (our case: directories → files → functions)
- Revealing call flows and dependency chains
- Small to medium DAGs (1,000-5,000 nodes)

**Pros:**
- ✅ Clear visual hierarchy matches code structure
- ✅ Predictable, deterministic layout (no randomness)
- ✅ Easy to follow call chains vertically
- ✅ Works with DAGs and near-DAGs (some cycles)

**Cons:**
- ❌ Can be wide with many parallel branches
- ❌ Struggles with dense interconnections (needs edge bundling)
- ❌ May need manual layer assignment for non-tree structures

**Libraries:**
- **cytoscape-dagre** (most mature, 67 extensions)
- **d3-dag** (experimental, light maintenance mode)
- **ELK.js** (Eclipse Layout Kernel, handles complex DAGs better)

**Implementation estimate:** 2-3 days (swap force layout for dagre layout)

**Example configuration:**
```javascript
cy.layout({
  name: 'dagre',
  rankDir: 'TB',  // top-to-bottom
  rankSep: 150,   // vertical spacing between layers
  nodeSep: 80,    // horizontal spacing between nodes
  edgeSep: 10,    // edge separation
  ranker: 'network-simplex'  // fast layer assignment
});
```

**Sources:**
- [cytoscape-dagre GitHub](https://github.com/cytoscape/cytoscape.js-dagre)
- [d3-dag GitHub](https://github.com/erikbrinkman/d3-dag)

---

### 2. Community Detection + ForceAtlas2

**What it is:** Use graph algorithms (Louvain, modularity) to detect clusters, then layout using community-aware force simulation.

**Best for:**
- Revealing modular code architecture
- Finding tightly coupled components
- Graphs with natural clustering (libraries, packages, modules)

**Pros:**
- ✅ Automatically groups related code visually
- ✅ Highlights architectural boundaries
- ✅ Can color-code modules for clarity
- ✅ Works with any graph structure (trees, DAGs, cyclic)

**Cons:**
- ❌ Requires preprocessing (community detection algorithm)
- ❌ Still uses force-directed (can have local hairballs)
- ❌ Communities may not match intended architecture

**Algorithm workflow:**
1. Run Louvain community detection on graph
2. Assign community IDs to nodes
3. Use ForceAtlas2 with:
   - Stronger attraction within communities
   - Repulsion between communities
   - "Prevent Overlap" option enabled

**Implementation estimate:** 3-4 days (add community detection, modify force parameters)

**Example (conceptual):**
```javascript
// Step 1: Detect communities (use Louvain algorithm)
const communities = detectCommunities(graph);

// Step 2: Layout with community-aware forces
d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links)
    .strength(d => {
      // Stronger links within same community
      return communities[d.source] === communities[d.target] ? 0.9 : 0.2;
    })
  )
  .force("charge", d3.forceManyBody()
    .strength(d => communities[d.id] === focusCommunity ? -500 : -100)
  );
```

**Sources:**
- [Gephi Community Detection Manual](https://jveerbeek.gitlab.io/gephi/docs/community.html)
- [Louvain Algorithm Implementation](https://medium.com/analytics-vidhya/implement-louvain-community-detection-algorithm-using-python-and-gephi-with-visualization-871250fb2f25)

---

### 3. Radial/Circular Layout with Hierarchical Levels

**What it is:** Arrange nodes in concentric circles based on depth/hierarchy level, with root at center.

**Best for:**
- Tree-like structures (our directory → file → function hierarchy)
- Showing depth levels clearly
- Compact visualization of hierarchies

**Pros:**
- ✅ Natural fit for hierarchical code structures
- ✅ Depth immediately visible (distance from center)
- ✅ Compact layout (better space utilization than tree)
- ✅ Easy to add fisheye zoom on center node

**Cons:**
- ❌ Difficult to show non-hierarchical relationships (cross-package imports)
- ❌ Outer rings become crowded with many leaf nodes
- ❌ Edge crossings can still create clutter

**Implementation estimate:** 2-3 days (implement radial layout logic)

**D3.js implementation pattern:**
```javascript
// Radial layout based on node depth
nodes.forEach(node => {
  const radius = node.depth * 200;  // 200px per level
  const angle = (node.indexInLevel / node.totalAtLevel) * 2 * Math.PI;
  node.x = width/2 + radius * Math.cos(angle);
  node.y = height/2 + radius * Math.sin(angle);
});
```

**Example:** React's component tree visualizations use radial layouts effectively.

---

### 4. Hybrid: Hierarchical + Force-Directed

**What it is:** Use hierarchical layout for main structure, force-directed for positioning within each level.

**Best for:**
- Graphs with clear hierarchy but complex within-level relationships
- Balancing structure with flexibility
- Our use case: directories hierarchical, functions force-directed

**Pros:**
- ✅ Combines predictability (hierarchy) with flexibility (force)
- ✅ Can constraint vertical positions, let horizontal positions float
- ✅ Works with existing D3 force simulation (add constraints)

**Cons:**
- ❌ More complex to implement (two layout passes)
- ❌ Force simulation can still create local clutter
- ❌ Needs careful tuning of force strengths

**Implementation estimate:** 4-5 days (add hierarchical constraints to force simulation)

**WebCola approach:**
```javascript
// Use constraints to enforce layering
cola.layout()
  .nodes(nodes)
  .links(links)
  .constraints([
    // Vertical alignment constraints per layer
    { type: 'alignment', axis: 'y', offsets: layerConstraints }
  ])
  .start();
```

**Sources:**
- [WebCola with D3.js tutorial](https://mitratech.com/resource-hub/resources-alyne/using-webcola-and-d3-js-to-create-hierarchical-layout/)

---

### 5. Matrix View (Adjacency Matrix)

**What it is:** Represent graph as 2D matrix where cell (i,j) shows relationship between nodes i and j.

**Best for:**
- Analyzing dense connections (our 360K edges!)
- Finding patterns (clusters, bottlenecks)
- Complementing node-link diagrams

**Pros:**
- ✅ No edge clutter (relationships shown as cells)
- ✅ Excellent for dense graphs (our case!)
- ✅ Easy to spot patterns (diagonal clusters = modules)
- ✅ Can sort/filter rows/columns dynamically

**Cons:**
- ❌ Not intuitive for non-technical users
- ❌ Loses spatial/hierarchical intuition
- ❌ Hard to see paths (A→B→C requires multiple lookups)
- ❌ Scales poorly beyond ~500 nodes (1,447 = 2M cells)

**Implementation estimate:** 3-4 days (build matrix view as alternative)

**Recommendation:** Add as **secondary view**, not primary. Let users toggle between node-link and matrix.

**Example (conceptual):**
```javascript
// Matrix heatmap showing dependency strength
const matrix = d3.select("#matrix")
  .selectAll("rect")
  .data(linkMatrix)
  .join("rect")
  .attr("x", d => xScale(d.source))
  .attr("y", d => yScale(d.target))
  .attr("fill", d => colorScale(d.strength));
```

**Sources:**
- [Safely Restructure Codebase with Dependency Graphs](https://understandlegacycode.com/blog/safely-restructure-codebase-with-dependency-graphs/)

---

## Recommended Approach for MCP Vector Search

### Phase 1: Quick Win (1-2 weeks)
**Use Sugiyama/Dagre hierarchical layout as primary view**

**Why:** Matches our existing directory → file → function hierarchy perfectly, proven to work for DAGs up to 5,000 nodes.

**Implementation:**
1. Switch from `d3.forceSimulation()` to `cytoscape-dagre` layout
2. Keep progressive disclosure (collapse/expand by level)
3. Add edge bundling for cross-package imports
4. Maintain existing content pane and code viewer

**Libraries to add:**
- `cytoscape` (core library, 500KB minified)
- `cytoscape-dagre` (layout extension, 50KB)

**Code changes:**
- Replace `renderGraph()` force simulation in `scripts.py` (lines 388-439)
- Add hierarchical layout logic based on `node.depth`
- Adjust edge routing to reduce crossings

**Expected result:** Clear vertical hierarchy, reduced visual clutter, better scanability.

---

### Phase 2: Enhanced Navigation (2-3 weeks)
**Add focus+context interaction patterns**

**Features to implement:**

1. **Semantic Zoom Levels:**
   - Zoom level 1: Show only directories
   - Zoom level 2: Show directories + files
   - Zoom level 3: Show files + top-level functions
   - Zoom level 4: Show all functions/classes

2. **Fisheye Lens:**
   - Magnify nodes near mouse cursor
   - Compress distant nodes
   - Maintain spatial relationships

3. **Edge Filtering:**
   - Toggle edge types (calls, imports, semantic, cycles)
   - Show only edges connected to selected node
   - Highlight paths between two nodes

4. **Minimap Navigation:**
   - Bird's eye view of entire graph
   - Click to jump to region
   - Show current viewport as rectangle

**Implementation estimate:** 2-3 weeks

**Example (semantic zoom):**
```javascript
function updateVisibilityByZoom(zoomLevel) {
  nodes.forEach(node => {
    if (zoomLevel < 0.3) {
      node.visible = node.type === 'directory';
    } else if (zoomLevel < 0.6) {
      node.visible = ['directory', 'file'].includes(node.type);
    } else {
      node.visible = true;  // show all
    }
  });
}
```

**Sources:**
- [iSphere: Focus+Context Sphere Visualization](https://www.researchgate.net/publication/316653025_iSphere_FocusContext_Sphere_Visualization_for_Interactive_Large_Graph_Exploration)
- [Fisheye Tree Views and Lenses](https://www.semanticscholar.org/paper/Fisheye-Tree-Views-and-Lenses-for-Graph-Tominski-Abello/db6b4b9f27f9e3113c0bfa423fb848c934e2d651)

---

### Phase 3: Advanced Features (3-4 weeks)
**Add community detection and edge bundling**

1. **Louvain Community Detection:**
   - Detect modules/packages algorithmically
   - Color-code nodes by community
   - Add "Show community X only" filter

2. **Hierarchical Edge Bundling:**
   - Bundle edges that share common paths
   - Reduce 360K edges to ~50K visual bundles
   - Use D3's `d3.curveBundle` for smooth routing

3. **Matrix View Toggle:**
   - Add alternative matrix view for dense regions
   - Show matrix for selected module/directory
   - Enable drag-select in matrix to filter graph

**Implementation estimate:** 3-4 weeks

**Hierarchical edge bundling example:**
```javascript
const bundle = d3.curveBundle.beta(0.85);  // bundling strength

const line = d3.line()
  .curve(bundle)
  .x(d => d.x)
  .y(d => d.y);

// Create bundled paths via common ancestors
links.forEach(link => {
  const path = getPathViaAncestor(link.source, link.target);
  link.pathData = line(path);
});
```

**Sources:**
- [Edge Bundling in Information Visualization](https://www.researchgate.net/publication/260653028_Edge_Bundling_in_Information_Visualization)
- [Holten's Hierarchical Edge Bundles (IEEE 2006)](http://www.aviz.fr/wiki/uploads/Teaching2014/bundles_infovis.pdf)

---

## Implementation Recommendations

### Immediate Actions (This Sprint)

1. **Prototype Dagre layout** (2 days)
   - Install `cytoscape` and `cytoscape-dagre`
   - Create alternative layout mode: `?layout=dagre`
   - Compare side-by-side with force-directed

2. **Add edge type filtering** (1 day)
   - UI toggles for: calls, imports, semantic, cycles
   - Update `renderGraph()` to filter `visibleLinks`
   - Default: show only containment + calls (reduce from 360K to ~10K edges)

3. **Implement "show connected only" mode** (1 day)
   - When node selected, show only edges connected to it
   - Dim unconnected nodes
   - Reveal local neighborhood structure

### Near-Term Enhancements (Next 2 Sprints)

4. **Semantic zoom levels** (3 days)
   - Tie node visibility to zoom scale
   - Smooth transitions between zoom levels
   - Persist zoom state in URL params

5. **Community detection** (5 days)
   - Integrate Louvain algorithm (use `graphology-communities`)
   - Color nodes by detected community
   - Add legend showing communities

6. **Edge bundling** (5 days)
   - Implement hierarchical edge bundling
   - Use for cross-module edges only
   - Add "bundle strength" slider

### Long-Term Vision (Future Roadmap)

7. **Multi-view dashboard**
   - Node-link (hierarchical)
   - Matrix view
   - Sunburst/treemap (for directory sizes)
   - Synchronized selection across views

8. **Query-driven exploration**
   - "Show all callers of function X"
   - "Find circular dependencies in module Y"
   - "Show semantic duplicates (similarity > 0.8)"

9. **Performance optimization**
   - WebGL rendering for 10K+ nodes
   - Virtual scrolling for large matrices
   - Web Worker for layout calculations

---

## Performance Considerations

### Current Bottlenecks
- **360,823 edges:** Rendering all edges kills browser performance
- **D3 force simulation:** O(n²) complexity for charge force
- **DOM manipulation:** 1,447 SVG nodes + 360K lines = slow

### Solutions

**1. Edge Reduction Strategy**
- Default: Show only structural edges (containment = ~5K edges)
- On-demand: Load call/import edges for selected nodes
- Never render all 360K edges simultaneously

**2. Canvas Rendering**
- Use `<canvas>` instead of SVG for edges (10x faster)
- Keep SVG for nodes (need interactivity)
- Redraw canvas on zoom/pan

**3. Level-of-Detail (LOD)**
- Zoom out: Show aggregated "meta-nodes" (directories)
- Zoom in: Show individual functions
- Dynamic detail based on viewport

**4. Virtual Viewport**
- Only render nodes visible in viewport
- Cull off-screen nodes from DOM
- Use spatial index (quadtree) for fast lookup

**Example (edge culling):**
```javascript
// Only show edges for visible nodes
const visibleEdges = allLinks.filter(link => {
  const sourceVisible = isInViewport(link.source);
  const targetVisible = isInViewport(link.target);
  return sourceVisible && targetVisible;
});

// Render canvas instead of SVG
const ctx = canvas.getContext('2d');
visibleEdges.forEach(edge => {
  ctx.moveTo(edge.source.x, edge.source.y);
  ctx.lineTo(edge.target.x, edge.target.y);
  ctx.stroke();
});
```

---

## Comparison Table: Layout Algorithms

| Algorithm | Best For | Pros | Cons | Implementation | Performance |
|-----------|----------|------|------|----------------|-------------|
| **Sugiyama/Dagre** | Hierarchical DAGs | Clear structure, predictable | Wide graphs, some manual tuning | ⭐⭐⭐⭐ (mature libraries) | O(n log n) |
| **Community + ForceAtlas2** | Modular codebases | Auto-grouping, flexible | Requires preprocessing | ⭐⭐⭐ (need extra algorithm) | O(n²) |
| **Radial/Circular** | Tree-like structures | Compact, clear depth | Crowded outer rings | ⭐⭐⭐⭐ (simple math) | O(n) |
| **Hybrid (Hier+Force)** | Mixed hierarchy | Balanced structure/flexibility | Complex to implement | ⭐⭐ (needs constraints) | O(n²) |
| **Matrix View** | Dense connections | No edge clutter, pattern detection | Not intuitive, poor scalability | ⭐⭐⭐ (simple heatmap) | O(n²) memory |
| **Force-Directed** (current) | General graphs | Flexible, well-understood | Hairball with large graphs | ⭐⭐⭐⭐⭐ (already have) | O(n²) |

**Legend:**
- ⭐⭐⭐⭐⭐ Trivial (< 1 day)
- ⭐⭐⭐⭐ Easy (1-2 days)
- ⭐⭐⭐ Moderate (3-5 days)
- ⭐⭐ Difficult (1-2 weeks)
- ⭐ Very Difficult (3+ weeks)

---

## Key Research Papers & Resources

### Academic Papers
1. **"Edge Bundling in Information Visualization"** (Holten, IEEE 2006)
   *The foundational paper on hierarchical edge bundling techniques.*

2. **"Edge Routing with Ordered Bundles"** (Pupyrev et al., 2012)
   *Improved edge bundling that preserves visual order.*

3. **"iSphere: Focus+Context Sphere Visualization"** (Du & Cao, 2017)
   *Novel focus+context technique using Riemann sphere mapping.*

4. **"Grooming the Hairball"** (ResearchGate, 2015)
   *Comprehensive survey of techniques to fix hairball visualizations.*

### Tools & Libraries
1. **Cytoscape.js** - [https://js.cytoscape.org/](https://js.cytoscape.org/)
   *Most mature graph visualization library, 67 extensions, excellent dagre support.*

2. **D3-DAG** - [https://github.com/erikbrinkman/d3-dag](https://github.com/erikbrinkman/d3-dag)
   *D3-compatible DAG layouts (Sugiyama, Zherebko, Grid).*

3. **ELK.js** - [https://github.com/kieler/elkjs](https://github.com/kieler/elkjs)
   *Eclipse Layout Kernel in JavaScript, handles complex DAGs.*

4. **Graphology** - [https://graphology.github.io/](https://graphology.github.io/)
   *Graph manipulation library with Louvain community detection.*

5. **Gephi** - [https://gephi.org/](https://gephi.org/)
   *Desktop tool for graph analysis (not web, but useful for prototyping layouts).*

### Production Examples
1. **Sourcegraph** - Code intelligence platform
   *Uses hierarchical layouts with progressive disclosure.*

2. **CodeSee** - Codebase visualization tool
   *Combines directory trees with dependency graphs.*

3. **Structure101** - Architecture analysis tool
   *Uses matrix view + hierarchical diagrams.*

4. **Neo4j Bloom** - Graph visualization
   *Excellent example of focus+context with graph databases.*

---

## Example Implementations to Study

### 1. Cytoscape.js Dagre Demo
**URL:** [https://codesandbox.io/examples/package/cytoscape-dagre](https://codesandbox.io/examples/package/cytoscape-dagre)

**What to learn:**
- Dagre layout configuration options
- Edge routing quality
- Performance with 1,000+ nodes

**Adapting to our codebase:**
- Map directory/file hierarchy to Dagre ranks
- Use `rankDir: 'TB'` for top-to-bottom flow
- Set `nodeSep` based on our adaptive spacing function

---

### 2. D3 Hierarchical Edge Bundling
**URL:** [https://observablehq.com/@d3/hierarchical-edge-bundling](https://observablehq.com/@d3/hierarchical-edge-bundling)

**What to learn:**
- How to bundle edges via common ancestors
- Curve parameters (`beta` value)
- Color coding by edge type

**Adapting to our codebase:**
- Bundle cross-package imports
- Keep containment edges unbundled (clear hierarchy)
- Use `similarity_score` to color semantic edges

---

### 3. Gephi ForceAtlas2 with Communities
**URL:** [https://medium.com/analytics-vidhya/implement-louvain-community-detection-algorithm-using-python-and-gephi-with-visualization-871250fb2f25](https://medium.com/analytics-vidhya/implement-louvain-community-detection-algorithm-using-python-and-gephi-with-visualization-871250fb2f25)

**What to learn:**
- Louvain algorithm workflow
- Community-aware force parameters
- Color scheme for communities

**Adapting to our codebase:**
- Run Louvain on function call graph
- Color nodes by detected module
- Add "Show community X" filter in UI

---

## Next Steps

### Recommended Order of Implementation

**Week 1-2: Dagre Prototype**
1. Install cytoscape.js and cytoscape-dagre
2. Create alternative layout mode
3. Test with current 1,447-node graph
4. Gather user feedback

**Week 3-4: Edge Filtering & Performance**
5. Add edge type toggles
6. Implement "connected only" mode
7. Switch edge rendering to canvas
8. Optimize for 10K+ edges

**Week 5-6: Enhanced Navigation**
9. Add semantic zoom levels
10. Implement minimap
11. Add breadcrumb trail (already have!)
12. Keyboard shortcuts for navigation

**Week 7-8: Advanced Features**
13. Integrate Louvain community detection
14. Implement hierarchical edge bundling
15. Add matrix view toggle
16. Performance testing with real codebases

### Success Metrics

**Before (Current State):**
- 😞 Users report "can't see anything" with large graphs
- 😞 360K edges render as solid mass
- 😞 Difficult to find specific functions
- 😞 No clear code architecture visible

**After (Target State):**
- ✅ Clear visual hierarchy (directories → files → functions)
- ✅ Reduced visible edges to <10K (filtering + bundling)
- ✅ Easy navigation to any function (search + zoom)
- ✅ Architectural patterns visible (modules, coupling)
- ✅ Circular dependencies highlighted (already have!)

### Open Questions for User Research

1. **Primary use case:** Are users exploring unfamiliar codebases OR analyzing known code for refactoring?
2. **Edge importance:** Which edge types matter most (calls, imports, semantic)?
3. **Zoom behavior:** Should zoom reveal more detail OR just magnify existing view?
4. **Color coding:** Prefer by file type, complexity, or community?

---

## Conclusion

The "hairball" problem in our code graph visualization is solvable through a **phased approach**:

1. **Quick win:** Switch to Dagre hierarchical layout (1-2 weeks)
2. **Navigation:** Add focus+context interactions (2-3 weeks)
3. **Advanced:** Community detection + edge bundling (3-4 weeks)

**Recommended first step:** Prototype Dagre layout alongside existing force-directed view and A/B test with real users.

**Key principle:** *"Hairballs in your knowledge graph are a good thing. Just don't let them anywhere near your UI."* Focus on task-specific views rather than attempting to render the entire graph.

---

## Files Analyzed

- `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (1,827 lines)
- `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` (578 lines)

**Current force simulation configuration:** Lines 388-439 in `scripts.py`
**Adaptive spacing calculation:** Lines 144-186 in `scripts.py`
**Progressive disclosure logic:** Lines 328-379, 763-832 in `scripts.py`

---

## References

1. Cambridge Intelligence. "Graph visualization: fixing data hairballs." [https://cambridge-intelligence.com/how-to-fix-hairballs/](https://cambridge-intelligence.com/how-to-fix-hairballs/)

2. Cytoscape.js. "The Dagre layout for DAGs and trees." [https://github.com/cytoscape/cytoscape.js-dagre](https://github.com/cytoscape/cytoscape.js-dagre)

3. D3-DAG. "Layout algorithms for visualizing directed acyclic graphs." [https://github.com/erikbrinkman/d3-dag](https://github.com/erikbrinkman/d3-dag)

4. Du, F., & Cao, N. "iSphere: Focus+Context Sphere Visualization for Interactive Large Graph Exploration." CHI 2017. [ResearchGate](https://www.researchgate.net/publication/316653025)

5. Holten, D. "Hierarchical Edge Bundles: Visualization of Adjacency Relations in Hierarchical Data." IEEE InfoVis 2006.

6. Pupyrev, S., et al. "Edge Routing with Ordered Bundles." Graph Drawing 2012. [Springer Link](https://link.springer.com/chapter/10.1007/978-3-642-25878-7_14)

7. Gephi Documentation. "Community Detection." [https://jveerbeek.gitlab.io/gephi/docs/community.html](https://jveerbeek.gitlab.io/gephi/docs/community.html)

8. Understand Legacy Code. "Safely restructure your codebase with Dependency Graphs." [https://understandlegacycode.com/blog/safely-restructure-codebase-with-dependency-graphs/](https://understandlegacycode.com/blog/safely-restructure-codebase-with-dependency-graphs/)
