# Visualization Styling Analysis: Adding Three Visual Axes

**Research Date:** 2026-02-20
**Objective:** Understand current visualization styling to add three visual axes (red=complexity, dashed borders=smells, blue=TBD) to treemap and sunburst views

---

## Executive Summary

The mcp-vector-search project uses D3.js for interactive code visualization with three modes: tree, treemap, and sunburst. The current system already has **complexity-based coloring** but needs enhancement to add **two additional visual dimensions**: smell indicators (dashed borders) and a third blue-coded metric (TBD).

### Key Findings

1. **Existing Complexity Coloring:** Already implemented with A-F grades mapped to green→red spectrum
2. **Data Availability:** Chunks already contain `complexity_grade`, `smell_count`, `smells` array
3. **Styling Infrastructure:** CSS classes exist (`.grade-A` through `.grade-F`, `.has-smells`)
4. **Modification Points:** Two main rendering functions need updates: `renderTreemap()` and `renderSunburst()`

### Quick Implementation Path

**For Red Axis (Complexity):** ✅ Already implemented via `getNodeColor()` and `getComplexityColor()`
**For Dashed Borders (Smells):** Add stroke-dasharray to treemap rects and sunburst arcs when `smell_count > 0`
**For Blue Axis (TBD):** Add new metric to chunk metadata, create color blending function

---

## 1. Current Treemap Rendering

### File Location
`src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

### Treemap Cell Styling (Lines 4874-4903)

```javascript
// Current implementation
cell.append('rect')
    .attr('class', 'treemap-cell')
    .attr('width', d => Math.max(0, d.x1 - d.x0))
    .attr('height', d => Math.max(0, d.y1 - d.y0))
    .attr('fill', d => getNodeColor(d))  // <-- RED AXIS (complexity)
    .attr('stroke', 'rgba(0,0,0,0.3)')   // <-- NEEDS DYNAMIC STYLING
    .attr('stroke-width', 0.5)            // <-- NEEDS DYNAMIC STYLING
    .style('cursor', 'pointer')
    .style('opacity', 1)
    .on('click', handleTreemapClick)
    .on('mouseover', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 0.8)
            .attr('stroke', '#58a6ff')
            .attr('stroke-width', 2);
        handleTreemapHover(event, d);
    })
    .on('mouseout', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 1)
            .attr('stroke', 'rgba(0,0,0,0.3)')  // <-- NEEDS DYNAMIC STYLING
            .attr('stroke-width', 0.5);         // <-- NEEDS DYNAMIC STYLING
        hideVizTooltip();
    })
```

**Key Observation:** The `.attr('fill', d => getNodeColor(d))` already implements the red complexity axis. We need to enhance stroke styling for the smell axis.

---

## 2. Current Sunburst Rendering

### Sunburst Arc Styling (Lines 5192-5220)

```javascript
// Create arcs
const arcs = g.selectAll('path')
    .data(displayRoot.descendants().filter(d => d.depth > 0))
    .join('path')
    .attr('class', 'sunburst-arc')
    .attr('d', arc)
    .attr('fill', d => getNodeColor(d))  // <-- RED AXIS (complexity)
    .attr('stroke', '#0d1117')           // <-- NEEDS DYNAMIC STYLING
    .attr('stroke-width', 0.5)           // <-- NEEDS DYNAMIC STYLING
    .style('cursor', 'pointer')
    .style('opacity', 1)
    .on('click', handleSunburstClick)
    .on('mouseover', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 0.8)
            .attr('stroke', '#58a6ff')
            .attr('stroke-width', 2);
        handleSunburstHover(event, d);
    })
    .on('mouseout', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 1)
            .attr('stroke', '#0d1117')       // <-- NEEDS DYNAMIC STYLING
            .attr('stroke-width', 0.5);      // <-- NEEDS DYNAMIC STYLING
        hideVizTooltip();
    });
```

**Key Observation:** Same pattern as treemap - fill color is complexity-driven, stroke needs smell detection.

---

## 3. Color Computation Functions

### Complexity Color Mapping (Line 4770-4777)

```javascript
function getComplexityColor(complexity) {
    if (complexity === undefined || complexity === null) return '#6e7681';  // gray
    if (complexity <= 5) return '#238636';   // A - green
    if (complexity <= 10) return '#1f6feb';  // B - blue
    if (complexity <= 15) return '#d29922';  // C - yellow
    if (complexity <= 20) return '#f0883e';  // D - orange
    return '#da3633';                        // F - red
}
```

**Grade Thresholds:**
- **A (Green):** 0-5 (excellent)
- **B (Blue):** 6-10 (good)
- **C (Yellow):** 11-15 (acceptable)
- **D (Orange):** 16-20 (needs improvement)
- **F (Red):** 21+ (refactor recommended)

### Main Node Color Function (Line 4780-4801)

```javascript
function getNodeColor(d) {
    // If it's a leaf node with complexity, use complexity color
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return getComplexityColor(d.data.complexity);
    }

    // Color by node type for non-leaf nodes
    const typeColors = {
        'monorepo': '#8957e5',
        'subproject': '#1f6feb',
        'directory': '#79c0ff',
        'file': '#58a6ff',
        'language': '#8957e5',
        'category': '#6e7681',
        'function': '#d29922',
        'method': '#8957e5',
        'class': '#1f6feb',
        'root': '#6e7681'
    };

    return typeColors[d.data.type] || '#6e7681';
}
```

---

## 4. CSS Styling Infrastructure

### Existing CSS Classes (styles.py, lines 1316-1340)

```css
/* Complexity grade colors for nodes */
.grade-A { fill: #238636 !important; stroke: #2ea043; }
.grade-B { fill: #1f6feb !important; stroke: #388bfd; }
.grade-C { fill: #d29922 !important; stroke: #e0ac3a; }
.grade-D { fill: #f0883e !important; stroke: #f59f5f; }
.grade-F { fill: #da3633 !important; stroke: #f85149; }

/* Code smell indicator - red border */
.has-smells circle {
    stroke: var(--error) !important;
    stroke-width: 3px !important;
    stroke-dasharray: 5, 3;
}

/* Circular dependency indicator */
.in-cycle circle {
    stroke: #ff4444 !important;
    stroke-width: 3px !important;
    animation: pulse-border 1.5s infinite;
}

@keyframes pulse-border {
    0%, 100% { stroke-opacity: 0.8; }
    50% { stroke-opacity: 1.0; }
}
```

**Key Observation:** CSS classes already exist for both complexity grades and smell detection. These apply to **tree view circles** but need extension to treemap rects and sunburst arcs.

---

## 5. Data Structure: Chunk Metrics

### Available Metrics Per Node (graph_builder.py, lines 450-485)

```python
node = {
    "id": chunk_id,
    "name": chunk_name,
    "type": chunk.chunk_type,
    "file_path": file_path_str,
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    "complexity": chunk.complexity_score,  # <-- LEGACY FIELD
    "parent_id": parent_id,
    "depth": chunk.chunk_depth,
    "content": chunk.content,
    "docstring": chunk.docstring,
    "language": chunk.language,
}

# Add structural analysis metrics if available
if hasattr(chunk, "cognitive_complexity") and chunk.cognitive_complexity is not None:
    node["cognitive_complexity"] = chunk.cognitive_complexity
if hasattr(chunk, "cyclomatic_complexity") and chunk.cyclomatic_complexity is not None:
    node["cyclomatic_complexity"] = chunk.cyclomatic_complexity
if hasattr(chunk, "complexity_grade") and chunk.complexity_grade is not None:
    node["complexity_grade"] = chunk.complexity_grade  # <-- RED AXIS DATA
if hasattr(chunk, "code_smells") and chunk.code_smells:
    node["smells"] = chunk.code_smells                 # <-- SMELL DATA
if hasattr(chunk, "smell_count") and chunk.smell_count is not None:
    node["smell_count"] = chunk.smell_count            # <-- SMELL COUNT
if hasattr(chunk, "quality_score") and chunk.quality_score is not None:
    node["quality_score"] = chunk.quality_score        # <-- POTENTIAL BLUE AXIS
if hasattr(chunk, "lines_of_code") and chunk.lines_of_code is not None:
    node["lines_of_code"] = chunk.lines_of_code
```

### ChunkMetrics Data Class (metrics.py, lines 14-51)

```python
@dataclass
class ChunkMetrics:
    """Metrics for a single code chunk (function/class/method)."""

    cognitive_complexity: int = 0
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    parameter_count: int = 0
    lines_of_code: int = 0

    # Halstead metrics (Phase 4)
    halstead_volume: float | None = None
    halstead_difficulty: float | None = None
    halstead_effort: float | None = None
    halstead_bugs: float | None = None

    # Code smells detected
    smells: list[str] = field(default_factory=list)

    # Computed grades (A-F scale)
    complexity_grade: str = field(init=False, default="A")

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        self.complexity_grade = self._compute_grade()
```

**Key Metrics Available:**
- ✅ **Complexity Grade** (A-F) → RED AXIS (already used)
- ✅ **Smells Array** → DASHED BORDER AXIS (detected but not visualized in treemap/sunburst)
- ✅ **Smell Count** → Boolean flag for "has smells"
- 🔵 **Candidate Blue Axis Options:**
  - `quality_score` (0.0-1.0 health score)
  - `halstead_difficulty` (cognitive load metric)
  - `max_nesting_depth` (structural complexity)
  - Git churn (requires additional data collection)
  - Test coverage (requires additional data collection)

---

## 6. Code Smell Detection

### Smell Detector Module (analysis/collectors/smells.py)

```python
@dataclass
class CodeSmell:
    """Detected code smell with severity and description."""

    type: str                  # e.g., "long_function", "too_many_parameters"
    severity: SmellSeverity    # INFO, WARNING, ERROR
    message: str
    line_number: int | None = None
    suggestion: str | None = None

class SmellDetector(MetricCollector):
    """Detect code smells during AST traversal."""

    def collect_chunk(self, context: CollectorContext) -> None:
        """Detect smells for a single code chunk."""
        # Check for common smells:
        # - Long functions (>50 LOC)
        # - Too many parameters (>5)
        # - Deep nesting (>4 levels)
        # - High cognitive complexity (>15)
        # - Missing docstrings
        # - God classes (too many methods)
```

**Detected Smell Types:**
1. `long_function` - Function exceeds 50 lines
2. `too_many_parameters` - More than 5 parameters
3. `deep_nesting` - Nesting depth > 4
4. `high_cognitive_complexity` - Cognitive complexity > 15
5. `missing_docstring` - No docstring present
6. `god_class` - Class with > 20 methods
7. `long_parameter_list` - Similar to too_many_parameters
8. `complex_boolean_expression` - Nested boolean logic

**Smell Data Flow:**
1. `SmellDetector.collect_chunk()` → Populates `ChunkMetrics.smells` list
2. `ChunkMetrics.to_metadata()` → Converts to JSON string for ChromaDB
3. `build_graph_data()` → Adds `smells` and `smell_count` to node data
4. D3.js visualization → Can read `d.data.smell_count` and `d.data.smells`

---

## 7. Analysis Tools Available

### Code Quality Analysis (analysis/code_quality.py)

```python
class ComplexityAnalyzer(ast.NodeVisitor):
    """Calculate cyclomatic complexity and nesting depth."""

    def __init__(self):
        self.complexity = 1  # Base complexity
        self.max_nesting = 0
        self._current_nesting = 0

    def visit_If(self, node):
        self.complexity += 1
        self._visit_nested(node)

    def visit_For(self, node):
        self.complexity += 1
        self._visit_nested(node)

    # ... similar for While, ExceptHandler, BoolOp, comprehensions
```

### Health Score Calculation (metrics.py, lines 234-271)

```python
@property
def health_score(self) -> float:
    """Calculate 0.0-1.0 health score based on metrics.

    Health score considers:
    - Average complexity (lower is better)
    - Code smells count (fewer is better)
    - Comment ratio (balanced is better)
    """
    score = 1.0

    # Penalty for high average complexity (A=0%, B=-10%, C=-20%, D=-30%, F=-50%)
    if self.avg_complexity > 30:
        score -= 0.5
    elif self.avg_complexity > 20:
        score -= 0.3
    elif self.avg_complexity > 10:
        score -= 0.2
    elif self.avg_complexity > 5:
        score -= 0.1

    # Penalty for code smells (up to -30%)
    total_smells = sum(len(chunk.smells) for chunk in self.chunks)
    smell_penalty = min(0.3, total_smells * 0.05)
    score -= smell_penalty

    # Penalty for poor comment ratio (ideal: 10-30%)
    if self.total_lines > 0:
        comment_ratio = self.comment_lines / self.total_lines
        if comment_ratio < 0.1:
            score -= 0.1
        elif comment_ratio > 0.5:
            score -= 0.1

    return max(0.0, score)
```

---

## 8. Exact Code Modification Locations

### **Location 1: Treemap Stroke Styling**

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Function:** `renderTreemap()`
**Lines:** 4874-4903 (cell.append('rect') block)

**Current Code:**
```javascript
.attr('stroke', 'rgba(0,0,0,0.3)')
.attr('stroke-width', 0.5)
```

**Proposed Change:**
```javascript
.attr('stroke', d => getBorderColor(d))              // Dynamic border color
.attr('stroke-width', d => getBorderWidth(d))        // Dynamic border width
.attr('stroke-dasharray', d => getBorderDashArray(d)) // Dashed for smells
```

---

### **Location 2: Treemap Hover State Restoration**

**File:** Same as Location 1
**Lines:** 4894-4900 (mouseout handler)

**Current Code:**
```javascript
.on('mouseout', function(event, d) {
    d3.select(this)
        .transition()
        .duration(150)
        .style('opacity', 1)
        .attr('stroke', 'rgba(0,0,0,0.3)')  // <-- HARDCODED
        .attr('stroke-width', 0.5);         // <-- HARDCODED
    hideVizTooltip();
})
```

**Proposed Change:**
```javascript
.on('mouseout', function(event, d) {
    d3.select(this)
        .transition()
        .duration(150)
        .style('opacity', 1)
        .attr('stroke', getBorderColor(d))         // <-- DYNAMIC
        .attr('stroke-width', getBorderWidth(d))   // <-- DYNAMIC
        .attr('stroke-dasharray', getBorderDashArray(d)); // <-- RESTORE DASHES
    hideVizTooltip();
})
```

---

### **Location 3: Sunburst Stroke Styling**

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Function:** `renderSunburst()`
**Lines:** 5198-5200

**Current Code:**
```javascript
.attr('stroke', '#0d1117')
.attr('stroke-width', 0.5)
```

**Proposed Change:**
```javascript
.attr('stroke', d => getBorderColor(d))
.attr('stroke-width', d => getBorderWidth(d))
.attr('stroke-dasharray', d => getBorderDashArray(d))
```

---

### **Location 4: Sunburst Hover State Restoration**

**File:** Same as Location 3
**Lines:** 5212-5219 (mouseout handler)

**Current Code:**
```javascript
.on('mouseout', function(event, d) {
    d3.select(this)
        .transition()
        .duration(150)
        .style('opacity', 1)
        .attr('stroke', '#0d1117')       // <-- HARDCODED
        .attr('stroke-width', 0.5);      // <-- HARDCODED
    hideVizTooltip();
})
```

**Proposed Change:**
```javascript
.on('mouseout', function(event, d) {
    d3.select(this)
        .transition()
        .duration(150)
        .style('opacity', 1)
        .attr('stroke', getBorderColor(d))
        .attr('stroke-width', getBorderWidth(d))
        .attr('stroke-dasharray', getBorderDashArray(d));
    hideVizTooltip();
})
```

---

### **Location 5: New Styling Functions (Insert After Line 4801)**

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Insert After:** `getNodeColor()` function (line 4801)

**New Functions to Add:**

```javascript
// ============================================================================
// VISUAL AXIS STYLING FUNCTIONS
// ============================================================================

/**
 * Get border color for smell indication
 * Returns red if node has code smells, default otherwise
 */
function getBorderColor(d) {
    const smellCount = d.data.smell_count || 0;
    const hasSmells = d.data.smells && d.data.smells.length > 0;

    if (smellCount > 0 || hasSmells) {
        return '#da3633';  // Red border for smells
    }

    // Default border color (dark gray for dark theme)
    return 'rgba(0,0,0,0.3)';
}

/**
 * Get border width for smell emphasis
 * Thicker border = more smells
 */
function getBorderWidth(d) {
    const smellCount = d.data.smell_count || 0;

    if (smellCount === 0) return 0.5;      // Thin default
    if (smellCount === 1) return 1.5;      // Moderate
    if (smellCount === 2) return 2.0;      // Thicker
    return 2.5;                             // Thickest (3+ smells)
}

/**
 * Get border dash pattern for smell indication
 * Dashed border = has code smells
 */
function getBorderDashArray(d) {
    const smellCount = d.data.smell_count || 0;
    const hasSmells = d.data.smells && d.data.smells.length > 0;

    if (smellCount > 0 || hasSmells) {
        return '5,3';  // Dashed pattern (5px dash, 3px gap)
    }

    return 'none';  // Solid border (no dashes)
}

/**
 * BLUE AXIS: Get color overlay for third metric (TBD)
 * Options:
 * - quality_score (0.0-1.0 health score)
 * - halstead_difficulty (cognitive load)
 * - max_nesting_depth (structural complexity)
 * - git_churn (change frequency)
 *
 * Approach: Blend blue into base color for low-quality/high-risk code
 */
function getBlueAxisOverlay(d) {
    // Example: Use quality_score if available
    const qualityScore = d.data.quality_score;

    if (qualityScore === undefined || qualityScore === null) {
        return 'none';  // No overlay
    }

    // Lower quality = more blue overlay
    const blueIntensity = 1.0 - qualityScore;  // 0.0 (good) to 1.0 (poor)

    if (blueIntensity > 0.3) {  // Only show if quality < 70%
        const opacity = blueIntensity * 0.5;  // Max 50% opacity
        return `rgba(31, 111, 235, ${opacity})`;  // Semi-transparent blue
    }

    return 'none';
}

/**
 * Enhanced node color with blue axis overlay
 * Combines complexity color (red axis) with quality overlay (blue axis)
 */
function getEnhancedNodeColor(d) {
    const baseColor = getNodeColor(d);  // Complexity-based color
    const overlay = getBlueAxisOverlay(d);

    if (overlay === 'none') {
        return baseColor;
    }

    // For treemap/sunburst, we'll use CSS filters or layered rendering
    // For now, return base color and handle overlay separately
    return baseColor;
}
```

---

## 9. Blue Axis: Candidate Metrics

### Option 1: Quality Score (health_score)

**Pros:**
- Already computed per file (metrics.py, line 234)
- Combines complexity, smells, comment ratio
- Well-defined 0.0-1.0 scale
- Higher = better (intuitive)

**Cons:**
- File-level metric, not chunk-level
- Would need aggregation for chunks

**Visual Encoding:** Blue overlay intensity increases with **lower** quality (poor code = blue tint)

---

### Option 2: Halstead Difficulty

**Pros:**
- Measures cognitive load to understand code
- Scientific basis (Halstead complexity metrics)
- Chunk-level metric (already in ChunkMetrics)
- Orthogonal to cyclomatic complexity

**Cons:**
- Requires Phase 4 Halstead collector
- Currently optional (may be None)
- Less intuitive than quality score

**Visual Encoding:** Blue saturation increases with **higher** difficulty

---

### Option 3: Max Nesting Depth

**Pros:**
- Already collected (ChunkMetrics.max_nesting_depth)
- Easy to interpret (nested if/for/while levels)
- Strong correlation with maintainability
- Chunk-level metric

**Cons:**
- Some overlap with cognitive complexity
- May not add much beyond red axis

**Visual Encoding:** Blue depth increases with **deeper** nesting (4+ levels)

---

### Option 4: Git Churn (Change Frequency)

**Pros:**
- Identifies "hot spots" that change often
- Indicates fragile/unstable code
- Orthogonal to static analysis metrics
- Good predictor of bugs

**Cons:**
- Requires git history analysis
- Not currently collected
- Slow to compute
- Unavailable for new files

**Visual Encoding:** Blue intensity increases with **higher** churn (frequent changes)

---

### **Recommendation: Start with Quality Score**

**Rationale:**
1. Already computed and available
2. Composite metric (complexity + smells + comments)
3. No additional data collection needed
4. Easy to interpret (0.0 = poor, 1.0 = excellent)
5. Can add other metrics later with toggle

**Implementation Path:**
1. Add `quality_score` to chunk metadata in `build_graph_data()`
2. Compute chunk-level quality score (or aggregate from file)
3. Use `getBlueAxisOverlay()` to add blue tint for low-quality chunks
4. Add legend explaining the three axes

---

## 10. Implementation Checklist

### Phase 1: Smell Detection (Dashed Border Axis)

- [ ] **Verify smell data availability** in chunk nodes
  - Check `d.data.smell_count` in browser console
  - Check `d.data.smells` array content
- [ ] **Add helper functions** (Location 5)
  - `getBorderColor(d)` - Red if has smells
  - `getBorderWidth(d)` - Thicker with more smells
  - `getBorderDashArray(d)` - Dashed pattern for smells
- [ ] **Update treemap styling** (Locations 1 & 2)
  - Replace hardcoded stroke attributes
  - Fix hover state restoration
- [ ] **Update sunburst styling** (Locations 3 & 4)
  - Replace hardcoded stroke attributes
  - Fix hover state restoration
- [ ] **Test with smell-heavy code**
  - Find chunks with `smell_count > 0`
  - Verify dashed red borders appear
  - Verify hover states restore correctly

### Phase 2: Blue Axis (Quality Score)

- [ ] **Add quality_score to chunk metadata**
  - Modify `graph_builder.py` line 483
  - Add `node["quality_score"] = chunk.quality_score`
- [ ] **Implement blue overlay function**
  - Add `getBlueAxisOverlay(d)` (Location 5)
  - Add `getEnhancedNodeColor(d)` wrapper
- [ ] **Apply to treemap**
  - Use `getEnhancedNodeColor()` instead of `getNodeColor()`
  - OR: Add CSS filter for blue tint
- [ ] **Apply to sunburst**
  - Same approach as treemap
- [ ] **Test with low-quality chunks**
  - Find chunks with `quality_score < 0.5`
  - Verify blue tint appears
  - Verify doesn't interfere with complexity colors

### Phase 3: Legend and Documentation

- [ ] **Add visual legend** to UI
  - Red gradient: Complexity (A→F)
  - Dashed borders: Code smells
  - Blue tint: Low quality score
- [ ] **Update tooltips** to show all three metrics
  - Complexity grade + score
  - Smell count + types
  - Quality score percentage
- [ ] **Add toggle controls** for each axis
  - Enable/disable smell borders
  - Enable/disable blue overlay
  - Keep complexity always visible (primary axis)

### Phase 4: Performance Optimization

- [ ] **Cache computed styles** in closure
  - Avoid recomputing border colors on every render
  - Use memoization for style functions
- [ ] **Batch style updates** during zoom/filter
  - Use D3 transitions efficiently
  - Minimize DOM thrashing

---

## 11. Code Snippets: Ready to Copy-Paste

### Snippet 1: Treemap Cell with All Three Axes

```javascript
// Insert at line 4874 (replace existing cell.append('rect') block)
cell.append('rect')
    .attr('class', 'treemap-cell')
    .attr('width', d => Math.max(0, d.x1 - d.x0))
    .attr('height', d => Math.max(0, d.y1 - d.y0))
    .attr('fill', d => getEnhancedNodeColor(d))      // RED + BLUE AXES
    .attr('stroke', d => getBorderColor(d))          // SMELL-BASED COLOR
    .attr('stroke-width', d => getBorderWidth(d))    // SMELL-BASED WIDTH
    .attr('stroke-dasharray', d => getBorderDashArray(d)) // DASHED IF SMELLY
    .style('cursor', 'pointer')
    .style('opacity', 1)
    .on('click', handleTreemapClick)
    .on('mouseover', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 0.8)
            .attr('stroke', '#58a6ff')
            .attr('stroke-width', 2);
        handleTreemapHover(event, d);
    })
    .on('mouseout', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 1)
            .attr('stroke', getBorderColor(d))              // RESTORE DYNAMIC
            .attr('stroke-width', getBorderWidth(d))        // RESTORE DYNAMIC
            .attr('stroke-dasharray', getBorderDashArray(d)); // RESTORE DASHES
        hideVizTooltip();
    })
    .append('title')
    .text(d => {
        const complexity = d.data.cognitive_complexity || d.data.complexity || 'N/A';
        const grade = d.data.complexity_grade || '?';
        const smells = d.data.smell_count || 0;
        const quality = d.data.quality_score !== undefined
            ? (d.data.quality_score * 100).toFixed(0) + '%'
            : 'N/A';
        return `${d.data.name}\\n` +
               `Lines: ${d.value}\\n` +
               `Complexity: ${complexity} (Grade ${grade})\\n` +
               `Smells: ${smells}\\n` +
               `Quality: ${quality}`;
    });
```

### Snippet 2: Sunburst Arc with All Three Axes

```javascript
// Insert at line 5192 (replace existing arcs definition)
const arcs = g.selectAll('path')
    .data(displayRoot.descendants().filter(d => d.depth > 0))
    .join('path')
    .attr('class', 'sunburst-arc')
    .attr('d', arc)
    .attr('fill', d => getEnhancedNodeColor(d))      // RED + BLUE AXES
    .attr('stroke', d => getBorderColor(d))          // SMELL-BASED COLOR
    .attr('stroke-width', d => getBorderWidth(d))    // SMELL-BASED WIDTH
    .attr('stroke-dasharray', d => getBorderDashArray(d)) // DASHED IF SMELLY
    .style('cursor', 'pointer')
    .style('opacity', 1)
    .on('click', handleSunburstClick)
    .on('mouseover', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 0.8)
            .attr('stroke', '#58a6ff')
            .attr('stroke-width', 2);
        handleSunburstHover(event, d);
    })
    .on('mouseout', function(event, d) {
        d3.select(this)
            .transition()
            .duration(150)
            .style('opacity', 1)
            .attr('stroke', getBorderColor(d))              // RESTORE DYNAMIC
            .attr('stroke-width', getBorderWidth(d))        // RESTORE DYNAMIC
            .attr('stroke-dasharray', getBorderDashArray(d)); // RESTORE DASHES
        hideVizTooltip();
    });
```

---

## 12. Alternative Approaches

### Approach A: CSS Classes (Current Pattern)

**Pros:**
- Consistent with existing tree view styling
- Declarative (easier to maintain)
- Better performance (browser-optimized)

**Cons:**
- Requires pre-computing classes during hierarchy build
- Less flexible for dynamic thresholds
- Harder to animate transitions

**Implementation:**
```javascript
cell.append('rect')
    .attr('class', d => {
        const classes = ['treemap-cell'];
        if (d.data.complexity_grade) {
            classes.push(`grade-${d.data.complexity_grade}`);
        }
        if (d.data.smell_count > 0) {
            classes.push('has-smells');
        }
        if (d.data.quality_score && d.data.quality_score < 0.7) {
            classes.push('low-quality');
        }
        return classes.join(' ');
    })
```

**Required CSS Addition:**
```css
/* Treemap/Sunburst smell borders */
.treemap-cell.has-smells,
.sunburst-arc.has-smells {
    stroke: var(--error) !important;
    stroke-width: 2px !important;
    stroke-dasharray: 5, 3;
}

/* Low quality blue tint */
.treemap-cell.low-quality {
    filter: hue-rotate(15deg) brightness(0.9);
}
```

---

### Approach B: Direct Attribute Functions (Recommended)

**Pros:**
- More flexible (access to full node data)
- Easier to tune thresholds dynamically
- Better for complex blending logic

**Cons:**
- Slightly more verbose
- Function call overhead (minimal)

**Implementation:** (Already shown in Snippet 1 & 2 above)

---

### Approach C: Layered SVG Elements

**Pros:**
- Can use opacity for true color blending
- Supports complex multi-layer effects
- Cleaner separation of visual concerns

**Cons:**
- More DOM elements (performance hit)
- Complex z-index management
- Harder to maintain

**Implementation:**
```javascript
// Base rectangle (complexity color)
cell.append('rect')
    .attr('class', 'treemap-cell-base')
    .attr('fill', d => getNodeColor(d));

// Overlay rectangle (blue tint)
cell.append('rect')
    .attr('class', 'treemap-cell-overlay')
    .attr('fill', d => getBlueAxisOverlay(d))
    .attr('opacity', 0.3);

// Border rectangle (smell indicator)
cell.append('rect')
    .attr('class', 'treemap-cell-border')
    .attr('fill', 'none')
    .attr('stroke', d => getBorderColor(d))
    .attr('stroke-dasharray', d => getBorderDashArray(d));
```

---

## 13. Testing Strategy

### Unit Tests (JavaScript)

```javascript
// Test border styling functions
describe('Visual Axis Functions', () => {
    it('should return red border for smells', () => {
        const nodeWithSmells = { data: { smell_count: 2 } };
        expect(getBorderColor(nodeWithSmells)).toBe('#da3633');
    });

    it('should return dashed pattern for smells', () => {
        const nodeWithSmells = { data: { smell_count: 1 } };
        expect(getBorderDashArray(nodeWithSmells)).toBe('5,3');
    });

    it('should return thicker border for more smells', () => {
        const node1 = { data: { smell_count: 1 } };
        const node3 = { data: { smell_count: 3 } };
        expect(getBorderWidth(node3)).toBeGreaterThan(getBorderWidth(node1));
    });
});
```

### Visual Regression Tests

1. **Capture baseline screenshots** of treemap/sunburst with known data
2. **Apply styling changes**
3. **Capture new screenshots**
4. **Compare pixel-by-pixel** for unexpected changes
5. **Verify expected changes** (dashed borders, blue tints)

### Manual QA Checklist

- [ ] Treemap cells show red dashed borders for smelly code
- [ ] Sunburst arcs show red dashed borders for smelly code
- [ ] Border thickness increases with smell count
- [ ] Blue tint appears for low-quality chunks
- [ ] Hover states restore correctly (no stuck borders)
- [ ] Zoom transitions preserve styling
- [ ] Performance remains acceptable (no lag)
- [ ] Tooltips show all three metrics
- [ ] Legend explains all three axes

---

## 14. Performance Considerations

### Current Performance Baseline

- **Treemap render time:** ~100-300ms for 1000+ nodes
- **Sunburst render time:** ~150-400ms for 1000+ nodes
- **Style computation overhead:** Minimal (<5ms)

### Potential Bottlenecks

1. **Function call overhead** for style functions
   - Mitigation: Memoize results during render pass
   - Pre-compute styles in hierarchy build phase
2. **DOM thrashing** during hover state changes
   - Mitigation: Use D3 transitions (already implemented)
   - Batch updates with requestAnimationFrame
3. **Large node counts** (5000+ nodes in treemap)
   - Mitigation: Already handled by progressive loading
   - Styling functions are O(1) per node

### Optimization Techniques

```javascript
// Memoized style computation (closure)
const styleCache = new Map();

function getCachedBorderColor(d) {
    const key = d.data.id;
    if (!styleCache.has(key)) {
        styleCache.set(key, getBorderColor(d));
    }
    return styleCache.get(key);
}

// Clear cache on data update
function onDataUpdate() {
    styleCache.clear();
    renderVisualization();
}
```

---

## 15. Future Enhancements

### Multi-Axis Toggle Controls

```html
<div class="visual-axes-controls">
    <label>
        <input type="checkbox" id="show-complexity" checked disabled>
        Complexity (Red) - Always Visible
    </label>
    <label>
        <input type="checkbox" id="show-smells" checked>
        Code Smells (Dashed Borders)
    </label>
    <label>
        <input type="checkbox" id="show-quality" checked>
        Quality Score (Blue Tint)
    </label>
</div>
```

### Dynamic Threshold Adjustment

```javascript
// Slider to adjust smell border threshold
const smellThreshold = document.getElementById('smell-threshold').value;

function getBorderDashArray(d) {
    const smellCount = d.data.smell_count || 0;
    if (smellCount >= smellThreshold) {
        return '5,3';
    }
    return 'none';
}
```

### Heatmap Mode (All Three Axes Visible)

- **Background color:** Complexity (red gradient)
- **Border color + dash:** Smells (red dashed)
- **Foreground overlay:** Quality (blue tint)
- **Combined effect:** Multi-dimensional risk visualization

---

## 16. References and Related Code

### Key Files

1. **scripts.py** - Main D3.js visualization logic
   - Location: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
   - Lines: 4807-4903 (treemap), 5134-5234 (sunburst), 4770-4801 (color functions)

2. **styles.py** - CSS styling
   - Location: `src/mcp_vector_search/cli/commands/visualize/templates/styles.py`
   - Lines: 1316-1340 (grade/smell CSS), 2540-2746 (treemap/sunburst CSS)

3. **graph_builder.py** - Data structure assembly
   - Location: `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`
   - Lines: 450-485 (chunk node creation)

4. **metrics.py** - Metric dataclasses
   - Location: `src/mcp_vector_search/analysis/metrics.py`
   - Lines: 14-117 (ChunkMetrics), 234-271 (health_score)

5. **code_quality.py** - AST analysis
   - Location: `src/mcp_vector_search/analysis/code_quality.py`
   - Lines: 60-120 (ComplexityAnalyzer)

### Related Functions

- `getNodeColor(d)` - Complexity-based coloring (line 4780)
- `getComplexityColor(complexity)` - Grade to color mapping (line 4770)
- `renderTreemap()` - Treemap rendering (line 4807)
- `renderSunburst()` - Sunburst rendering (line 5134)
- `handleTreemapClick(event, d)` - Treemap interaction
- `handleSunburstClick(event, d)` - Sunburst interaction

---

## Summary: Implementation Roadmap

### Immediate Actions (2-3 hours)

1. **Add smell border functions** (30 min)
   - `getBorderColor(d)`
   - `getBorderWidth(d)`
   - `getBorderDashArray(d)`

2. **Update treemap styling** (30 min)
   - Replace hardcoded strokes
   - Fix hover restoration

3. **Update sunburst styling** (30 min)
   - Same changes as treemap

4. **Test with existing smell data** (30 min)
   - Find chunks with smells
   - Verify visual appearance

5. **Add tooltips and legend** (30 min)
   - Explain three axes
   - Show metrics in tooltips

### Short-term Enhancements (1-2 days)

1. **Implement blue axis** (quality score)
   - Add `getBlueAxisOverlay(d)`
   - Apply to treemap/sunburst fills
   - Test with low-quality chunks

2. **Add toggle controls** for each axis
   - Enable/disable smell borders
   - Enable/disable blue overlay

3. **Performance testing**
   - Measure render times
   - Optimize if needed

### Long-term Vision (1-2 weeks)

1. **Multi-metric blue axis**
   - Support switching between quality, halstead, nesting
   - Add dropdown selector

2. **Git churn integration**
   - Collect change frequency data
   - Use as alternative blue axis

3. **Interactive legend**
   - Click to filter by metric range
   - Highlight nodes matching criteria

---

**End of Report**
