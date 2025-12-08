# Visualization Features - Complete Implementation Plan

## Summary of Changes

This document provides the exact changes needed to implement all 7 requested features in one pass.

## Changes to `src/mcp_vector_search/cli/commands/visualize.py`

### 1. Add CSS Spinner (Around line 800, after existing styles)

**Location**: Insert before the closing `</style>` tag (around line 1200)

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

### 2. Store Root Nodes (Line 1340, add global variable)

**Current** (around line 1340):
```javascript
let allNodes = [];
let allLinks = [];
let visibleNodes = new Set();
let collapsedNodes = new Set();
let highlightedNode = null;
```

**Replace with**:
```javascript
let allNodes = [];
let allLinks = [];
let visibleNodes = new Set();
let collapsedNodes = new Set();
let highlightedNode = null;
let rootNodes = [];  // NEW: Store root nodes for reset function
```

### 3. Update visualizeGraph to Store Root Nodes (Lines 1440-1442)

**Current**:
```javascript
// Start with only root nodes visible, all collapsed
visibleNodes = new Set(rootNodes.map(n => n.id));
collapsedNodes = new Set(rootNodes.map(n => n.id));
```

**Replace with**:
```javascript
// Start with only root nodes visible, all collapsed
rootNodes = rootNodes;  // Store for reset function
visibleNodes = new Set(rootNodes.map(n => n.id));
collapsedNodes = new Set(rootNodes.map(n => n.id));
```

Actually, the variable assignment is redundant. Just need to ensure it's declared globally (done in step 2).

### 4. Add Reset View Function (After zoomToFit function, around line 1686)

**Insert after** the `centerNode()` function:

```javascript
function resetView() {
    // Reset to root level nodes only
    visibleNodes = new Set(rootNodes.map(n => n.id));
    collapsedNodes = new Set(rootNodes.map(n => n.id));
    highlightedNode = null;

    // Re-render graph
    renderGraph();

    // Zoom to fit after rendering
    setTimeout(() => {
        zoomToFit(750);
    }, 200);
}
```

### 5. Update Reset Button Handler (Line 2381-2383)

**Current**:
```javascript
// Reset view button event handler
document.getElementById('reset-view-btn').addEventListener('click', () => {
    zoomToFit(750);
});
```

**Replace with**:
```javascript
// Reset view button event handler
document.getElementById('reset-view-btn').addEventListener('click', () => {
    resetView();
});
```

### 6. Update Loading HTML with Spinner (Line 2278)

**Current**:
```javascript
loadingEl.innerHTML = '<label style="color: #58a6ff;">⏳ Loading graph data...</label><br>' +
```

**Replace with**:
```javascript
loadingEl.innerHTML = '<label style="color: #58a6ff;"><span class="spinner"></span>Loading graph data...</label><br>' +
```

### 7. Enhance Footer with Metadata (Lines 1902-1911)

**Current**:
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

**Replace with**:
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

### 8. Expand Link Tooltips (Lines 1785-1816 - Replace entire function)

**Replace** the entire `showLinkTooltip` function with:

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

### 9. Add Dead Code Detection (Lines 1514-1522 - Modify circle rendering)

**Current**:
```javascript
// Add circles for regular code nodes (not files/dirs/docs)
node.filter(d => !isDocNode(d) && !isFileOrDir(d))
    .append("circle")
    .attr("r", d => {
        if (d.type === 'subproject') return 24;
        return d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
    })
    .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
    .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
    .style("fill", d => d.color || null);  // Use custom color if available
```

**Replace with**:
```javascript
// Add circles for regular code nodes (not files/dirs/docs)
node.filter(d => !isDocNode(d) && !isFileOrDir(d))
    .append("circle")
    .attr("r", d => {
        if (d.type === 'subproject') return 24;
        return d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
    })
    .attr("stroke", d => {
        // Check if node has incoming caller/imports edges (dead code detection)
        const hasIncoming = allLinks.some(l =>
            (l.target.id || l.target) === d.id &&
            (l.type === 'caller' || l.type === 'imports')
        );
        if (!hasIncoming && (d.type === 'function' || d.type === 'class' || d.type === 'method')) {
            // Check if it's not an entry point (main, test, cli files)
            const isEntryPoint = d.file_path && (
                d.file_path.includes('main.py') ||
                d.file_path.includes('__main__.py') ||
                d.file_path.includes('cli.py') ||
                d.file_path.includes('test_')
            );
            if (!isEntryPoint) {
                return "#ff6b6b"; // Red border for potentially dead code
            }
        }
        return hasChildren(d) ? "#ffffff" : "none";
    })
    .attr("stroke-width", d => {
        const hasIncoming = allLinks.some(l =>
            (l.target.id || l.target) === d.id &&
            (l.type === 'caller' || l.type === 'imports')
        );
        if (!hasIncoming && (d.type === 'function' || d.type === 'class' || d.type === 'method')) {
            const isEntryPoint = d.file_path && (
                d.file_path.includes('main.py') ||
                d.file_path.includes('__main__.py') ||
                d.file_path.includes('cli.py') ||
                d.file_path.includes('test_')
            );
            if (!isEntryPoint) {
                return 3; // Thicker red border
            }
        }
        return hasChildren(d) ? 2 : 0;
    })
    .style("fill", d => d.color || null);  // Use custom color if available
```

## Implementation Order

1. **Add CSS spinner** (easy)
2. **Add global rootNodes variable** (easy)
3. **Add resetView function** (easy)
4. **Update reset button handler** (easy)
5. **Update loading HTML** (easy)
6. **Enhance footer metadata** (moderate)
7. **Expand link tooltips** (moderate)
8. **Add dead code detection** (complex - requires careful logic)

## Testing Checklist

After implementation:
- [ ] CSS spinner appears during loading
- [ ] Reset View button returns to root level nodes
- [ ] Footer shows function lines and complexity
- [ ] Footer shows file lines for file nodes
- [ ] Hover over function call edges shows "📞 Function Call"
- [ ] Hover over import edges shows "📦 Import Dependency"
- [ ] Hover over semantic edges shows "🔗 Semantic Similarity" with %
- [ ] Red-bordered nodes indicate potentially dead code
- [ ] Entry points (main.py, test files) not marked as dead

## Notes

- All changes are in a single file: `src/mcp_vector_search/cli/commands/visualize.py`
- File is 2,390 lines total
- Changes are surgical and non-overlapping
- Each feature is independent and can be verified separately
