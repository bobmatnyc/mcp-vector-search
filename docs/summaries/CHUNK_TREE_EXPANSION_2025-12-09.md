# D3 Tree Visualization - Chunk Expansion Feature

**Date**: 2025-12-09
**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

## Summary

Updated D3 tree visualization to expand files in-tree to show their code chunks as child nodes, replacing the previous side-panel approach.

## Changes Made

### 1. Tree Structure Building (`buildTreeStructure()`)

**Before**: Filtered to only directories and files
```javascript
const treeNodes = allNodes.filter(node => {
    const type = node.type;
    return type === 'directory' || type === 'file';
});
```

**After**: Include chunks in the tree structure
```javascript
const treeNodes = allNodes.filter(node => {
    const type = node.type;
    return type === 'directory' || type === 'file' || type === 'chunk';
});
```

**Key Addition**: Second pass to attach chunks to parent files
```javascript
// Second pass: Attach chunks to their parent files
allNodes.forEach(node => {
    if (node.type === 'chunk' && node.file_id) {
        const parentFile = nodeMap.get(node.file_id);
        const chunkNode = nodeMap.get(node.id);

        if (parentFile && chunkNode) {
            parentFile.children.push(chunkNode);
            parentMap.set(node.id, node.file_id);
        }
    }
});
```

### 2. Node Styling

**Color Scheme**:
- 🟠 **Orange** (`#f39c12`): Collapsed directory (has `_children`)
- 🔵 **Blue** (`#3498db`): Expanded directory (has `children`)
- ⚪ **Gray** (`#95a5a6`): Collapsed file (has `_children`)
- ⚫ **Light Gray** (`#ecf0f1`): Expanded file or file with no chunks
- 🟣 **Purple** (`#9b59b6`): Chunk nodes

**Node Sizes**:
- Directories and files: 6px radius circles
- Chunks: 4px radius circles (smaller)

**Text Styling**:
- Regular nodes: 12px black text
- Chunks: 10px purple text (`#7d3c98`)

### 3. Click Handler Update (`handleNodeClick()`)

**Before**: Files showed chunks in side panel
```javascript
else if (nodeData.type === 'file') {
    loadChunksForFile(nodeData.id);
}
```

**After**: Files expand/collapse like directories
```javascript
if (nodeData.type === 'directory' || nodeData.type === 'file') {
    // Toggle: swap children <-> _children
    if (nodeData.children) {
        nodeData._children = nodeData.children;
        nodeData.children = null;
    } else if (nodeData._children) {
        nodeData.children = nodeData._children;
        nodeData._children = null;
    }
    renderVisualization();
} else if (nodeData.type === 'chunk') {
    displayChunkContent(nodeData);
}
```

### 4. New Function: `displayChunkContent()`

Replaced `displayChunks()` and `loadChunksForFile()` with simpler chunk display:

```javascript
function displayChunkContent(chunkData) {
    // Shows individual chunk with type, line numbers, and content
    // No async fetch needed - data already in tree
}
```

### 5. Label Updates

Both linear and circular layouts now show chunk types as labels:

```javascript
.text(d => {
    if (d.data.type === 'chunk') {
        return d.data.chunk_type || d.data.name || 'chunk';
    }
    return d.data.name;
})
```

## Behavioral Changes

### User Experience

**Before**:
1. Click directory → expands to show files
2. Click file → shows chunks in side panel
3. Side panel lists all chunks with scrolling

**After**:
1. Click directory → expands to show files
2. Click file → expands to show chunk nodes as children
3. Click chunk → shows individual chunk content in side panel

### Data Flow

**Before**:
- Chunks excluded from tree structure
- Fetched via `/api/chunks?file_id=...` on file click
- Displayed in side panel list

**After**:
- Chunks included in initial tree structure
- No additional API calls needed
- Chunks are regular tree nodes (collapsed by default)
- Side panel only used for individual chunk content

## Benefits

1. **Consistent Interaction Model**: Files behave like directories (expand/collapse)
2. **Visual Hierarchy**: Clear parent-child relationship between files and chunks
3. **Performance**: No additional API calls when expanding files
4. **Simplicity**: Removed async chunk loading function
5. **Scalability**: Tree naturally handles large numbers of chunks with collapse/expand

## Line Count Impact

**Net LOC**: Approximately neutral (~10 lines added for chunk attachment, ~30 lines removed from old chunk loading)

**Updated docstring** reflects new behavior and line count (~450 lines total)

## Testing Recommendations

1. **Basic Expansion**: Verify files expand to show chunk children
2. **Chunk Colors**: Confirm purple nodes for chunks
3. **Chunk Labels**: Check chunk type displays correctly
4. **Chunk Click**: Verify side panel shows individual chunk content
5. **Circular Layout**: Test chunk expansion in both linear and circular modes
6. **Empty Files**: Test files with no chunks (should not expand)
7. **Large Files**: Test performance with files having many chunks

## Related Files

- **Server**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py`
  - Existing `/api/graph` endpoint provides chunk data
  - `/api/chunks` endpoint now unused but kept for backward compatibility

## Migration Notes

- Existing chunk-graph.json files work without changes
- Chunks must have `file_id` property pointing to parent file
- Chunk nodes must have `type: 'chunk'` to be properly styled
- Optional `chunk_type` property used for labels

---

**Status**: Implementation complete, ready for testing
