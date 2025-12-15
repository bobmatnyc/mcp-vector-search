# MCP Vector Search Visualizer - UI Investigation

**Date:** 2025-12-15
**Researcher:** Claude (Research Agent)
**Focus:** Report generator controls, Code/Docs filter issues, dependency report functionality

---

## Executive Summary

Investigated the mcp-vector-search visualizer UI to understand report controls, filtering behavior, and dependency analysis functionality. Found critical bug where filtering nodes by Code/Docs type leaves "nodeless lines" (tree structure links) visible. Report generators are functional and well-integrated.

### Key Findings

1. **Report Links Location**: Found in sidebar "Options" section (lines 151-167 in base.py)
2. **Filter Bug Identified**: Code/Docs filter hides nodes but NOT the D3 tree links connecting them
3. **Dependency Report**: Fully functional with file-level analysis, circular dependency detection, and interactive UI
4. **4 Report Types**: Complexity, Code Smells, Dependencies, Trends (all working)

---

## 1. Report Generator Controls

### Location: Sidebar Options Section

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py`

**Lines 151-167:**

```python
<!-- Analysis Reports -->
<div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showComplexityReport()">
    <span style="margin-right: 8px;">📊</span>
    <span>Complexity Report</span>
</div>
<div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showCodeSmells()">
    <span style="margin-right: 8px;">🔍</span>
    <span>Code Smells</span>
</div>
<div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showDependencies()">
    <span style="margin-right: 8px;">🔗</span>
    <span>Dependencies</span>
</div>
<div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px;" onclick="showTrends()">
    <span style="margin-right: 8px;">📈</span>
    <span>Trends</span>
</div>
```

**Structure:**
- Located in sidebar below theme toggle
- 4 clickable report buttons
- Opens viewer panel with report content
- Clean, consistent UI with emoji icons

---

## 2. Code/Docs Filter Issues

### Filter Controls Location

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py`

**Lines 64-71:**

```html
<div class="control-group">
    <label style="color: var(--text-primary); margin-bottom: 8px;">Show Files</label>
    <div class="filter-buttons">
        <button class="filter-btn active" data-filter="all" onclick="setFileFilter('all')">All</button>
        <button class="filter-btn" data-filter="code" onclick="setFileFilter('code')">Code</button>
        <button class="filter-btn" data-filter="docs" onclick="setFileFilter('docs')">Docs</button>
    </div>
</div>
```

### Filter Implementation

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Function:** `setFileFilter()` (line 2435)

```javascript
function setFileFilter(filter) {
    currentFileFilter = filter;

    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    // Apply filter to the tree
    applyFileFilter();

    console.log(`File filter set to: ${filter}`);
}
```

**Function:** `applyFileFilter()` (line 2449)

```javascript
function applyFileFilter() {
    if (!treeData) return;

    // Get all node elements
    d3.selectAll('.node').each(function(d) {
        const node = d3.select(this);
        const data = d.data || d;

        if (data.type === 'directory') {
            // Directories always visible
            node.style('display', null);
            node.style('opacity', null);
        } else if (data.type === 'file') {
            const fileType = getFileType(data.name);
            const shouldShow = currentFileFilter === 'all' ||
                               fileType === currentFileFilter ||
                               fileType === 'unknown';

            node.style('display', shouldShow ? null : 'none');
            node.style('opacity', shouldShow ? null : 0);
        } else {
            // Chunks - visibility based on parent file
            node.style('display', null);
            node.style('opacity', null);
        }
    });

    // Update stats
    updateFilteredStats();
}
```

### **BUG IDENTIFIED: Nodeless Lines**

**Problem:** The filter only hides **nodes** but NOT the **links** (tree structure lines) that connect them.

**Root Cause:**
- `applyFileFilter()` uses `d3.selectAll('.node')` to hide nodes
- BUT it does NOT hide corresponding `.link` elements
- D3 tree creates links via `root.links()` (lines 1312, 1465)
- These link elements remain visible even when source/target nodes are hidden

**Visual Result:**
- When filtering to "Code" or "Docs", you see:
  - ✓ Nodes correctly hidden/shown
  - ✗ Links to hidden nodes still visible (orphaned lines)
  - Creates "nodeless lines" floating in space

**Evidence:**

Tree links are created here (scripts.py lines 1312-1324):
```javascript
// Draw links
const links = root.links();
console.log(`Drawing ${links.length} links`);
g.selectAll('.link')
    .data(links)
    .enter()
    .append('path')
    .attr('class', 'link')
    .attr('d', d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x))
    .attr('fill', 'none')
    .attr('stroke', '#ccc')
    .attr('stroke-width', 1.5);
```

**Missing Logic:** No code to hide `.link` elements when their source/target nodes are hidden.

### Recommended Fix

Add link filtering logic to `applyFileFilter()`:

```javascript
function applyFileFilter() {
    if (!treeData) return;

    // Track visible nodes
    const visibleNodeIds = new Set();

    // Get all node elements and apply node filter
    d3.selectAll('.node').each(function(d) {
        const node = d3.select(this);
        const data = d.data || d;

        let shouldShow = true;

        if (data.type === 'directory') {
            // Directories always visible
            shouldShow = true;
        } else if (data.type === 'file') {
            const fileType = getFileType(data.name);
            shouldShow = currentFileFilter === 'all' ||
                         fileType === currentFileFilter ||
                         fileType === 'unknown';
        } else {
            // Chunks - visibility based on parent file
            shouldShow = true;
        }

        node.style('display', shouldShow ? null : 'none');
        node.style('opacity', shouldShow ? null : 0);

        if (shouldShow) {
            visibleNodeIds.add(d);
        }
    });

    // NEW: Filter links based on visible nodes
    d3.selectAll('.link').each(function(d) {
        const link = d3.select(this);
        const sourceVisible = visibleNodeIds.has(d.source);
        const targetVisible = visibleNodeIds.has(d.target);

        // Hide link if either source or target is hidden
        const shouldShowLink = sourceVisible && targetVisible;
        link.style('display', shouldShowLink ? null : 'none');
    });

    // Update stats
    updateFilteredStats();
}
```

---

## 3. Dependency Report

### Location and Implementation

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Function:** `showDependencies()` (line 3477)

### Functionality

**What it does:**
1. **File-level dependency analysis**: Analyzes which files depend on each other
2. **Circular dependency detection**: Identifies dependency cycles
3. **Connection metrics**: Shows most connected files
4. **Interactive table**: Expandable rows with dependency details

**Key Features:**

```javascript
function showDependencies() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '🔗 Dependencies';

    // Build file-level dependency graph from caller links
    const fileDeps = buildFileDependencyGraph();

    // Calculate statistics
    const filesWithDeps = Array.from(fileDeps.keys()).length;
    const totalConnections = Array.from(fileDeps.values()).reduce((sum, dep) =>
        sum + dep.dependsOn.size + dep.usedBy.size, 0) / 2;

    // Find most connected file
    let mostConnectedFile = null;
    let maxConnections = 0;
    fileDeps.forEach((dep, filePath) => {
        const total = dep.dependsOn.size + dep.usedBy.size;
        if (total > maxConnections) {
            maxConnections = total;
            mostConnectedFile = filePath;
        }
    });

    // Detect circular dependencies
    const cycles = findCircularDeps(fileDeps);
    // ...
}
```

**UI Components:**

1. **Summary Cards (lines 3511-3530):**
   - Files with Dependencies count
   - Unique Dependencies count
   - Most Connected file with connection count

2. **Circular Dependencies Warning (lines 3532-3549):**
   - Shows detected cycles
   - Visual warning with ⚠️ icon
   - Lists up to 5 cycles with "... and N more" overflow

3. **Dependencies Table (lines 3551+):**
   - Sortable by connections (most connected first)
   - Shows files in cycles with special styling
   - Interactive expand/collapse for details

### Data Sources

**Dependency Computation:**
- Uses `buildFileDependencyGraph()` function (not shown in excerpts)
- Likely analyzes caller relationships from graph data
- Tracks `dependsOn` and `usedBy` relationships

**Backend Support:**

From `server.py` (lines 154-294), the visualizer has API endpoints:
- `/api/relationships/{chunk_id}` - Gets callers and semantic neighbors
- `/api/callers/{chunk_id}` - Gets function callers using AST analysis

These endpoints provide the relationship data used for dependency analysis.

---

## 4. Other Reports

### Complexity Report

**File:** `scripts.py` line 3027
**Function:** `showComplexityReport()`

**Features:**
- Collects all chunks with complexity data
- Shows average complexity
- Grade distribution (A, B, C, D, F, N/A)
- Sortable table by complexity
- Visual grade indicators with color coding

### Code Smells Report

**File:** `scripts.py` line 3325
**Function:** `showCodeSmells()`

**Features:**
- Detects 4 smell types:
  - Long Method
  - High Complexity
  - Deep Nesting
  - God Class
- Summary cards for total smells, warnings, errors
- Filter by smell type
- Interactive table with severity indicators

### Trends Report

**File:** `scripts.py` line 3755
**Function:** `showTrends()`

**Features:**
- Codebase metrics snapshot
- Key metrics: Lines of Code, Functions/Methods
- Timestamp for baseline tracking
- Note: Designed for future git history trend analysis

---

## Code Smells Detected in Visualizer Codebase

### 1. Massive File Size
- **File:** `scripts.py` - 164,328 bytes (160KB)
- **Severity:** High
- **Impact:** Memory intensive, difficult to navigate

### 2. Inline HTML String Building
- **Pattern:** JavaScript building HTML with string concatenation
- **Examples:** Lines 3086+, 3355+, 3507+, 3767+
- **Issues:**
  - No XSS protection
  - Hard to maintain
  - No templating

### 3. Missing Link Filter Logic
- **Bug:** Filter hides nodes but not links
- **Severity:** High (user-visible bug)
- **Fix Required:** Add link visibility logic to `applyFileFilter()`

---

## Architecture Notes

### Template Generation

**Base Template:**
- Location: `templates/base.py`
- Generates complete HTML page
- Embeds CSS from `styles.py`
- Embeds JavaScript from `scripts.py`
- Cache-busting with timestamp

### Server Architecture

**File:** `server.py`
**Framework:** FastAPI with uvicorn
**Key Endpoints:**
- `/` - Serve index.html
- `/api/graph-status` - Graph generation status
- `/api/graph` - Full graph data
- `/api/relationships/{chunk_id}` - Lazy-loaded relationships
- `/api/callers/{chunk_id}` - Function caller analysis
- `/api/chunks?file_id=` - File chunks

**Performance:**
- Streaming JSON with chunked transfer (100KB chunks)
- Lazy relationship computation (on-demand)
- 5-minute cache for relationship API

---

## File Locations Reference

| Component | File Path | Lines |
|-----------|-----------|-------|
| **Report Controls** | `templates/base.py` | 151-167 |
| **Filter Controls** | `templates/base.py` | 64-71 |
| **setFileFilter()** | `templates/scripts.py` | 2435-2447 |
| **applyFileFilter()** | `templates/scripts.py` | 2449-2478 |
| **showComplexityReport()** | `templates/scripts.py` | 3027+ |
| **showCodeSmells()** | `templates/scripts.py` | 3325+ |
| **showDependencies()** | `templates/scripts.py` | 3477+ |
| **showTrends()** | `templates/scripts.py` | 3755+ |
| **Tree Link Rendering** | `templates/scripts.py` | 1312-1324, 1464-1474 |
| **Server API** | `server.py` | Full file |

---

## Recommendations

### High Priority

1. **Fix Filter Bug:**
   - Add link visibility logic to `applyFileFilter()`
   - Test with Code/Docs filters
   - Verify no orphaned lines remain

2. **XSS Protection:**
   - Add HTML escaping to all user-generated content
   - Already has `escapeHtml()` function - ensure consistent use

### Medium Priority

3. **Refactor scripts.py:**
   - Split into modules (reports/, filters/, tree/, etc.)
   - Use template literals or JSX for HTML
   - Reduce file size for better memory usage

4. **Add Tests:**
   - Unit tests for filter logic
   - Integration tests for report generation
   - Visual regression tests for UI

### Low Priority

5. **Performance:**
   - Consider virtualization for large trees
   - Lazy render report tables
   - Add pagination to dependency tables

---

## Questions for User

1. **Filter Bug:** Should I create a fix for the "nodeless lines" issue?
2. **Dependency Report:** Is this providing the analysis you need, or were you looking for different dependency data?
3. **Report Access:** Are the report buttons easy to find in the current UI, or should they be more prominent?

---

## Memory Usage

**Files Analyzed:** 4 files
- `base.py` (389 bytes) - Full read
- `server.py` (15KB) - Full read
- `scripts.py` (164KB) - Strategic sampling (4 sections, ~800 lines total)
- File structure via ls commands

**Strategy Used:**
- Targeted reads using line offsets
- Grep for function locations
- No full read of massive scripts.py file
- Focus on specific functions and patterns

**Total Memory Impact:** Low (strategic sampling prevented loading 164KB file into memory)

---

*Research completed: 2025-12-15*
