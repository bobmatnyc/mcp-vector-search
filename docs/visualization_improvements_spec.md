# Visualization Improvements Specification

## Changes Requested

1. **Add metadata display**: Function lines, file lines, and cyclomatic complexity
2. **Add loading spinner**: Replace emoji with CSS animation spinner
3. **Add link/connector tooltips**: Show connection type and significance on mouseover
4. **Dead code detection**: Identify and highlight unused code chunks

## Implementation Details

### 1. Add CSS Spinner (Add to `<style>` section around line 600-800)

```css
/* Loading spinner animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #30363d;
    border-top-color: #58a6ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
```

### 2. Update Loading HTML (Line 2278)

**Current:**
```javascript
loadingEl.innerHTML = '<label style="color: #58a6ff;">⏳ Loading graph data...</label><br>' +
```

**Replace with:**
```javascript
loadingEl.innerHTML = '<label style="color: #58a6ff;"><span class="spinner"></span>Loading graph data...</label><br>' +
```

### 3. Add Metadata to Footer (Lines 1902-1911)

**Current footer building code:**
```javascript
// Build footer with annotations
let footerHtml = '';
if (node.language) {
    footerHtml += `<span class="footer-item"><span class="footer-label">Language:</span> ${node.language}</span>`;
}
footerHtml += `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;
if (node.start_line) {
    footerHtml += `<span class="footer-item"><span class="footer-label">Location:</span> Lines ${node.start_line}-${node.end_line}</span>`;
}
footer.innerHTML = footerHtml;
```

**Replace with (adds metadata):**
```javascript
// Build footer with annotations
let footerHtml = '';
if (node.language) {
    footerHtml += `<span class="footer-item"><span class="footer-label">Language:</span> ${node.language}</span>`;
}
footerHtml += `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;

// Add line information and complexity
if (node.start_line !== undefined && node.end_line !== undefined) {
    const totalLines = node.end_line - node.start_line + 1;

    if (node.type === 'function' || node.type === 'class' || node.type === 'method') {
        // For functions/classes: show function lines
        footerHtml += `<span class="footer-item"><span class="footer-label">Lines:</span> ${node.start_line}-${node.end_line} (${totalLines} lines)</span>`;
    } else if (node.type === 'file') {
        // For files: show file lines
        footerHtml += `<span class="footer-item"><span class="footer-label">File Lines:</span> ${totalLines}</span>`;
    } else {
        // For other types: show location
        footerHtml += `<span class="footer-item"><span class="footer-label">Location:</span> Lines ${node.start_line}-${node.end_line}</span>`;
    }

    // Add cyclomatic complexity if available and > 0
    if (node.complexity && node.complexity > 0) {
        footerHtml += `<span class="footer-item"><span class="footer-label">Complexity:</span> ${node.complexity}</span>`;
    }
}

footer.innerHTML = footerHtml;
```

### 4. Enhance Link Tooltips (Lines 1785-1816)

**Current function** only handles `is_cycle` and `semantic` link types. Need to expand for all types.

**Link Types Found in Data:**
- `caller` (176,751) - Function calls, most common
- `semantic` (3,963) - Semantic similarity
- `imports` (579) - Import dependencies
- `file_containment` (119) - File contains chunk
- `dir_containment` (88) - Directory contains file
- `dir_hierarchy` (23) - Directory structure
- `method`, `module` - Other relationships

**Replace showLinkTooltip function with:**

```javascript
function showLinkTooltip(event, d) {
    const sourceName = allNodes.find(n => n.id === (d.source.id || d.source))?.name || 'Unknown';
    const targetName = allNodes.find(n => n.id === (d.target.id || d.target))?.name || 'Unknown';

    // Special tooltip for cycle links
    if (d.is_cycle) {
        tooltip
            .style("display", "block")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .html(`
                <div style="color: #ff4444;"><strong>⚠️ Circular Dependency Detected</strong></div>
                <div style="margin-top: 8px;">Path: ${sourceName} → ${targetName}</div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">
                    This indicates a circular call relationship that may lead to infinite recursion or tight coupling.
                </div>
            `);
        return;
    }

    // Tooltip content based on link type
    let typeLabel = '';
    let typeDescription = '';
    let extraInfo = '';

    switch(d.type) {
        case 'caller':
            typeLabel = '📞 Function Call';
            typeDescription = `${sourceName} calls ${targetName}`;
            extraInfo = 'This is a direct function call relationship, the most common type of code dependency.';
            break;
        case 'semantic':
            typeLabel = '🔗 Semantic Similarity';
            typeDescription = `${(d.similarity * 100).toFixed(1)}% similar`;
            extraInfo = `These code chunks have similar meaning or purpose based on their content.`;
            break;
        case 'imports':
            typeLabel = '📦 Import Dependency';
            typeDescription = `${sourceName} imports ${targetName}`;
            extraInfo = 'This is an explicit import/dependency declaration.';
            break;
        case 'file_containment':
            typeLabel = '📄 File Contains';
            typeDescription = `${sourceName} contains ${targetName}`;
            extraInfo = 'This file contains the code chunk or function.';
            break;
        case 'dir_containment':
            typeLabel = '📁 Directory Contains';
            typeDescription = `${sourceName} contains ${targetName}`;
            extraInfo = 'This directory contains the file or subdirectory.';
            break;
        case 'dir_hierarchy':
            typeLabel = '🗂️ Directory Hierarchy';
            typeDescription = `${sourceName} → ${targetName}`;
            extraInfo = 'Parent-child directory structure relationship.';
            break;
        case 'method':
            typeLabel = '⚙️ Method Relationship';
            typeDescription = `${sourceName} ↔ ${targetName}`;
            extraInfo = 'Class method relationship.';
            break;
        case 'module':
            typeLabel = '📚 Module Relationship';
            typeDescription = `${sourceName} ↔ ${targetName}`;
            extraInfo = 'Module-level relationship.';
            break;
        case 'dependency':
            typeLabel = '🔀 Dependency';
            typeDescription = `${sourceName} depends on ${targetName}`;
            extraInfo = 'General code dependency relationship.';
            break;
        default:
            typeLabel = `🔗 ${d.type || 'Unknown'}`;
            typeDescription = `${sourceName} → ${targetName}`;
            extraInfo = 'Code relationship.';
    }

    tooltip
        .style("display", "block")
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY + 10) + "px")
        .html(`
            <div><strong>${typeLabel}</strong></div>
            <div style="margin-top: 4px;">${typeDescription}</div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">
                ${extraInfo}
            </div>
        `);
}
```

### 5. Dead Code Detection Strategy

**Approach**: Identify nodes with no incoming `caller` or `imports` edges (potentially unused).

**Implementation Options:**

**Option A - Visual Indicator (Simplest):**
Add a visual marker (opacity, border, icon) to nodes with no incoming edges during rendering.

```javascript
// In node rendering (around line 1514-1522)
node.filter(d => !isDocNode(d) && !isFileOrDir(d))
    .append("circle")
    .attr("r", d => {
        if (d.type === 'subproject') return 24;
        return d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
    })
    .attr("stroke", d => {
        // Check if node has incoming caller/imports edges
        const hasIncoming = allLinks.some(l =>
            (l.target.id || l.target) === d.id &&
            (l.type === 'caller' || l.type === 'imports')
        );
        if (!hasIncoming && d.type === 'function') {
            return "#ff6b6b"; // Red border for potentially dead code
        }
        return hasChildren(d) ? "#ffffff" : "none";
    })
    .attr("stroke-width", d => {
        const hasIncoming = allLinks.some(l =>
            (l.target.id || l.target) === d.id &&
            (l.type === 'caller' || l.type === 'imports')
        );
        if (!hasIncoming && d.type === 'function') {
            return 3; // Thicker red border
        }
        return hasChildren(d) ? 2 : 0;
    })
    .style("fill", d => d.color || null);
```

**Option B - Dead Code Panel (More Complex):**
Add a new sidebar panel listing all potentially dead code chunks.

**Option C - Filter Toggle (User Control):**
Add a button to toggle visibility of potentially dead code.

**Recommended**: Start with Option A (visual indicator) as it's non-intrusive and immediately visible.

**Notes:**
- Entry points (main functions, test files, CLI commands) might appear "dead" but are actually entry points
- Consider excluding files matching patterns like `main.py`, `__main__.py`, `cli.py`, `test_*.py`
- Add tooltip explanation when hovering over red-bordered nodes

## Files to Modify

- **File**: `src/mcp_vector_search/cli/commands/visualize.py`
- **Lines to change**:
  - Add CSS spinner (around line 600-800 in `<style>` section)
  - Update loading HTML (line 2278)
  - Update footer building (lines 1902-1911)
  - Replace showLinkTooltip function (lines 1785-1816)
  - Add dead code detection to node rendering (lines 1514-1522)

## Testing

After implementing:
1. Run `uv run mcp-vector-search visualize export`
2. Start server: `mcp-vector-search visualize serve --port 8080`
3. Open browser to http://localhost:8080

### Test Cases:

**1. Loading Spinner**
- Verify animated spinner appears while loading (instead of ⏳ emoji)
- Spinner should be blue rotating circle with progress bar below
- Shows "Loading graph data..." text

**2. Metadata Display**
- Click on a function/class node
- Footer should show: Lines (start-end with total), Complexity
- Click on a file node
- Footer should show: File Lines (total)

**3. Link Tooltips**
- Hover over different connector types:
  - Function call edges (most common) - should show "📞 Function Call"
  - Import dependencies - should show "📦 Import Dependency"
  - Semantic similarity edges - should show "🔗 Semantic Similarity" with percentage
  - File/directory containment - should show appropriate icon and description
- Verify tooltip follows mouse cursor
- Verify tooltip disappears on mouseout

**4. Dead Code Detection**
- Look for nodes with thick red borders (3px)
- These indicate functions with no incoming caller/imports edges
- Hover to verify these are legitimate dead code candidates
- Entry points (main.py, test files, CLI) should be manually verified

## Expected Output

**For a function node footer:**
```
Language: python | File: /path/to/file.py | Lines: 19-269 (250 lines) | Complexity: 10
```

**For a file node footer:**
```
Language: python | File: /path/to/file.py | File Lines: 500
```

**Loading spinner:**
- Animated rotating circle (blue border with gray background)
- Replaces ⏳ emoji
- Shows before "Loading graph data..."

**Link tooltip examples:**

*Function call:*
```
📞 Function Call
demonstrate_connection_pooling calls get_chunk_by_id
This is a direct function call relationship, the most common type of code dependency.
```

*Semantic similarity:*
```
🔗 Semantic Similarity
87.3% similar
These code chunks have similar meaning or purpose based on their content.
```

*Import dependency:*
```
📦 Import Dependency
database imports PostgresAdapter
This is an explicit import/dependency declaration.
```

**Dead code visual:**
- Red bordered nodes (3px width, #ff6b6b color)
- Only applies to function/class/method nodes
- Excluded: entry points, test files, main modules
