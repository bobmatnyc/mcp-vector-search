# Fix: Parent-Child Linking Issue in File Filtering

**Date**: December 15, 2024
**Component**: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Function**: `applyFileFilter()`

## Problem

When filtering nodes by file type (e.g., "Code" or "Docs"), child nodes were becoming orphaned and appearing directly under the virtual root instead of under their parent directories.

### Root Cause

The `applyFileFilter()` function was:
1. Filtering `allNodes` to include only matching file types
2. Filtering `finalFilteredNodes` to include only necessary directories
3. Calling `buildTreeStructure()` which uses `allLinks` to establish parent-child relationships

**Issue**: `buildTreeStructure()` uses `allLinks` to build the tree hierarchy. When a parent node is filtered out, its children lose their parent link and become root nodes connected directly to the virtual root.

### Example

Before filtering:
```
Project Root
├─ /docs (directory)
│  └─ guide.md (docs)
└─ /src (directory)
   └─ main.py (code)
```

After applying "Code" filter:
- `finalFilteredNodes` includes: `/src`, `main.py`
- Does NOT include: `/docs`, `guide.md`
- `allLinks` still contains link: `/docs` → `guide.md`
- Problem: `/docs` is filtered out, so `guide.md` becomes orphaned

Result (WRONG):
```
Project Root
├─ /src
│  └─ main.py ✓
└─ guide.md ✗ (orphaned!)
```

## Solution

### Path-Based Tree Building for Filtered Views

Instead of using `allLinks` for filtered views, build the tree hierarchy from file paths:

1. **For filtered views** (`currentFileFilter !== 'all'`):
   - Use new `buildTreeFromPaths()` function
   - Constructs parent-child relationships from `file_path` or `id` strings
   - Walks up directory hierarchy to find nearest ancestor in filtered set
   - Prevents orphaned nodes by using path structure, not links

2. **For "all" filter** (`currentFileFilter === 'all'`):
   - Continue using `buildTreeStructure()` with `allLinks`
   - Preserves original link-based behavior for unfiltered view

### Implementation

```javascript
// For filtered views, use path-based tree building
if (currentFileFilter !== 'all') {
    console.log('Using path-based tree building for filtered view');
    const rootNodes = buildTreeFromPaths(finalFilteredNodes);

    // Create treeData from roots
    // ... (create virtual root if needed)

    // Apply single-child chain collapsing
    // Collapse all nodes

} else {
    console.log('Using link-based tree building for "all" filter');
    // Temporarily replace allNodes with filtered nodes
    const originalNodes = allNodes;
    allNodes = finalFilteredNodes;

    // Rebuild tree structure with filtered nodes
    buildTreeStructure();

    // Restore original allNodes
    allNodes = originalNodes;
}
```

### Path-Based Tree Building Algorithm

```javascript
function buildTreeFromPaths(nodes) {
    const nodeMap = new Map();
    const rootNodes = [];

    // First pass: create all nodes with empty children
    nodes.forEach(node => {
        nodeMap.set(node.id, {...node, children: []});
    });

    // Second pass: establish parent-child from paths
    nodes.forEach(node => {
        const path = node.file_path || node.id;
        if (!path) return;

        // Get parent path
        const parts = path.split('/').filter(p => p);
        if (parts.length <= 1) {
            // Root level node
            rootNodes.push(nodeMap.get(node.id));
            return;
        }

        // Find parent by path
        parts.pop(); // Remove current item
        let parentPath = '/' + parts.join('/');

        // Try to find parent in filtered nodes
        let parent = nodeMap.get(parentPath);

        // Walk up to find nearest ancestor if direct parent not found
        while (!parent && parts.length > 0) {
            parts.pop();
            parentPath = parts.length > 0 ? '/' + parts.join('/') : null;
            parent = parentPath ? nodeMap.get(parentPath) : null;
        }

        if (parent) {
            parent.children.push(nodeMap.get(node.id));
        } else {
            // No ancestor found - this is a root
            rootNodes.push(nodeMap.get(node.id));
        }
    });

    return rootNodes;
}
```

## Benefits

1. **Correct Hierarchy**: Child nodes appear under their parent directories, even when some intermediate parents are filtered out
2. **No Orphaned Nodes**: All nodes find their nearest ancestor in the filtered set
3. **Backward Compatible**: Original link-based behavior preserved for "all" filter
4. **Consistent Behavior**: Filtered views now match user expectations

## Testing

### Manual Test

1. Generate visualization: `mcp-vector-search visualize`
2. Click "Code" filter button
3. Verify:
   - Code files appear under parent directories
   - No orphaned files at root level
   - Proper nesting preserved
4. Click "Docs" filter button
5. Verify same behavior for documentation files
6. Click "All" filter button
7. Verify complete tree structure shown

### Expected Console Output

**For filtered views**:
```
Using path-based tree building for filtered view
=== BUILDING TREE FROM PATHS ===
Created N nodes in map
Built tree with M root nodes
Root node types: [...]
=== END BUILDING TREE FROM PATHS ===
```

**For "all" filter**:
```
Using link-based tree building for "all" filter
=== BUILDING TREE RELATIONSHIPS ===
[... normal buildTreeStructure debug output ...]
```

## Related Issues

This fix addresses the orphaned nodes issue when filtering by file type in the visualization tree view.

## Files Modified

- `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Added `buildTreeFromPaths()` helper function
  - Modified `applyFileFilter()` to use path-based building for filtered views
  - Preserved link-based building for "all" filter

## Files Created

- `tests/manual/test_filter_tree_hierarchy.py` - Manual test documentation
- `docs/development/fix-filter-tree-hierarchy.md` - This document
