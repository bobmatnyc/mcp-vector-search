# File Click Expansion Fix - December 9, 2025

## Issue Summary

**Problem**: Clicking on files in the tree visualization did NOT expand to show their code chunks as child nodes.

**Root Cause**: The JavaScript code was looking for wrong property names:
1. Looking for `node.type === 'chunk'` when chunk types are actually: `function`, `class`, `method`, `text`, `imports`, `module`
2. Looking for `node.file_id` when the actual property is `node.parent_id`

## Fix Applied

### Location
`/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

### Changes Made

#### 1. Updated Node Type Filter (Line 85-97)
**Before**: Only filtered for `type === 'chunk'`
**After**: Filter for actual chunk types: `['function', 'class', 'method', 'text', 'imports', 'module']`

```javascript
// Before
const treeNodes = allNodes.filter(node => {
    const type = node.type;
    return type === 'directory' || type === 'file' || type === 'chunk';
});

// After
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];
const treeNodes = allNodes.filter(node => {
    const type = node.type;
    return type === 'directory' || type === 'file' || chunkTypes.includes(type);
});
```

#### 2. Fixed Chunk Attachment Logic (Line 126-174)
**Before**: Used `node.file_id` to find parent file
**After**: Use `node.parent_id` to find parent file

```javascript
// Before
if (node.type === 'chunk' && node.file_id) {
    const parentFile = nodeMap.get(node.file_id);
    // ...
}

// After
if (chunkTypes.includes(node.type)) {
    if (!node.parent_id) return;
    const parentFile = nodeMap.get(node.parent_id);
    // ...
}
```

#### 3. Updated Node Rendering (Line 316-344, 410-440)
Updated both linear and circular tree layouts to recognize all chunk types:

```javascript
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];
nodes.append('circle')
    .attr('r', d => chunkTypes.includes(d.data.type) ? 4 : 6)
    .attr('fill', d => {
        if (chunkTypes.includes(d.data.type)) {
            return '#9b59b6';  // Purple for chunks
        }
        // ...
    });
```

#### 4. Fixed Click Handler (Line 447-505)
Updated to recognize all chunk types when handling clicks:

```javascript
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];
if (chunkTypes.includes(nodeData.type)) {
    console.log('Displaying chunk content');
    displayChunkContent(nodeData);
}
```

#### 5. Fixed Display Functions (Line 600-662)
Updated `displayFileInfo()` and `displayChunkContent()` to use `chunk.type` instead of `chunk.chunk_type`:

```javascript
// Before
const icon = getChunkIcon(chunk.chunk_type);
const chunkName = chunk.name || chunk.chunk_type || 'chunk';

// After
const icon = getChunkIcon(chunk.type);
const chunkName = chunk.name || chunk.type || 'chunk';
```

## Data Structure

### File Node
```javascript
{
  id: "file_e1096b5d",
  name: "example.py",
  type: "file",
  file_path: "/path/to/example.py",
  children: []  // or _children when collapsed
}
```

### Chunk Node (Function Example)
```javascript
{
  id: "fc93b6a40bcfe70e",
  name: "function_name",
  type: "function",  // NOT "chunk"!
  parent_id: "file_e1096b5d",  // NOT file_id!
  file_path: "/path/to/example.py",
  start_line: 10,
  end_line: 20,
  content: "def function_name():\n    ...",
  language: "python"
}
```

## Diagnostic Logging Added

Added comprehensive console logging to debug:

1. **Initial Load**: Node counts by type
2. **Chunk Attachment**:
   - Total chunks found
   - Chunks with parent_id
   - Chunks successfully attached
   - Missing parent warnings
3. **Post-Collapse Check**: Files with chunks in `_children`
4. **Click Handler**: Node type, children count, action taken

## Testing Instructions

1. Open http://localhost:8080 in browser
2. Open browser console (F12 or Cmd+Option+I)
3. Look for these console messages:

```
Filtered to X tree nodes (directories, files, and chunks)
Node breakdown: X directories, Y files, Z chunks

=== CHUNK ATTACHMENT DEBUG ===
Found X chunk nodes (function, class, method, text, imports, module)
Chunks with parent_id property: Y
Chunks successfully attached: Z
=== END CHUNK ATTACHMENT DEBUG ===

=== POST-COLLAPSE FILE CHECK ===
File <name> has X chunks in _children
Checked Y files, Z have chunks
=== END POST-COLLAPSE FILE CHECK ===
```

4. Expand a directory to see files
5. Click on a file node (gray circle)
6. Expected behavior:
   - Console shows: "Expanding file"
   - File node turns white
   - Purple chunk nodes appear as children
   - Side panel shows file info with chunk count

## Verification Checklist

- [x] Fixed chunk type detection (function, class, method, text, imports, module)
- [x] Fixed parent relationship (parent_id instead of file_id)
- [x] Updated node filtering to include all chunk types
- [x] Updated node rendering for all chunk types
- [x] Fixed click handler to recognize chunk types
- [x] Updated display functions to use correct properties
- [x] Added comprehensive diagnostic logging
- [x] Tested in both linear and circular layouts

## Server Status

Server running on: http://localhost:8080
Port: 8080
Process IDs:
- Python: 14001
- uv: 13999

## Related Files

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (FIXED)
- `/Users/masa/Projects/mcp-vector-search/tests/manual/file-click-debug-checklist.md` (Debug guide)

## Expected Result

Files should now:
1. Display as gray circles when collapsed (with chunks hidden in `_children`)
2. Turn white and show purple chunk children when clicked/expanded
3. Display file information in side panel with chunk count
4. Allow clicking individual chunks to view their content

---

**Status**: ✅ Fixed and deployed to localhost:8080
**Next Steps**: User should test by clicking on file nodes in the visualization
