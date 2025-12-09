# Tree Hierarchy Fix - December 9, 2025

## Problem

The visualization was showing **6758 root nodes** instead of a proper hierarchical tree structure. Everything was at the root level with no parent-child relationships being built.

## Root Cause

The JavaScript code was trying to build the hierarchy using the `parent_id` property on chunk nodes, but:

1. **6746 out of 7932 chunks had empty `parent_id` values** (`""`)
2. The parent-child relationships for chunks were actually stored in **file_containment links**, not in node properties
3. The code was only using `parent_id` property, ignoring the file_containment links

## Data Structure Analysis

From API `/api/graph`:

### Nodes
- Total: 8336 nodes
- Directories: 50
- Files: 354
- Chunks: 7932 (function, class, method, text, imports, module)

### Links
- Total: 14758 links
- `dir_hierarchy`: 43 (connect directories to parent directories)
- `dir_containment`: 349 (connect directories to child directories/files)
- `file_containment`: 6746 (connect files to chunks) ← **KEY INSIGHT**
- `semantic`: 6315 (similarity relationships)
- `caller`: 119 (function call relationships)
- `null`: 1186 (unknown type)

### Key Observation
The number of `file_containment` links (6746) **exactly matches** the number of chunks with empty `parent_id` (6746). This proves that the parent-child relationships are stored in links, not node properties.

## Solution

Changed the chunk attachment logic from using `parent_id` properties to using `file_containment` links:

### Before (Incorrect)
```javascript
// Tried to use parent_id property (which was mostly empty)
allNodes.forEach(node => {
    if (chunkTypes.includes(node.type)) {
        if (!node.parent_id) return;  // 6746 chunks skipped here!

        const parentFile = nodeMap.get(node.parent_id);
        if (parentFile) {
            parentFile.children.push(nodeMap.get(node.id));
            parentMap.set(node.id, node.parent_id);
        }
    }
});
```

### After (Correct)
```javascript
// Use file_containment links to build hierarchy
allLinks.forEach(link => {
    if (link.type === 'file_containment') {
        // source = FILE, target = CHUNK
        const parentFile = nodeMap.get(link.source);
        const chunkNode = nodeMap.get(link.target);

        if (parentFile && chunkNode) {
            parentFile.children.push(chunkNode);
            parentMap.set(link.target, link.source);
        }
    }
});
```

## Expected Result

After the fix:
- Directories contain subdirectories and files
- Files contain their code chunks
- Chunks are properly nested under their parent files
- Tree structure should have ~50 root directories instead of 6758 root nodes

## Files Modified

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Lines 190-235: Rewrote chunk attachment logic to use file_containment links

## Testing

1. Restart visualization server: `uv run mcp-vector-search visualize serve --port 8080`
2. Open browser to: http://127.0.0.1:8080/
3. Check browser console for:
   - `Found X root nodes` (should be ~50, not 6758)
   - `Processed 6746 file_containment links`
   - `Successfully matched: 6746`

## Lessons Learned

1. **Don't assume data structure without verification**: The `parent_id` property seemed like the obvious place for parent references, but it was actually empty for most nodes.

2. **Use links for relationships**: In graph data structures, relationships are often stored in edges/links rather than node properties.

3. **Debug with real data**: The bug was only visible when examining actual API responses, not from reading code.

4. **Link type naming matters**: The link type `file_containment` clearly indicates file → chunk relationships.

## Related Issues

This fix addresses the core problem reported in the user's message about 6758 root nodes. The collapse logic was working correctly; the issue was that there were no proper parent-child relationships to collapse in the first place.

## Performance Impact

- No performance impact
- Same number of operations, just using different data source (links instead of properties)
- More robust as it works regardless of whether `parent_id` is set

## Future Improvements

Consider adding validation to warn if:
- Chunks have `parent_id` but no corresponding file_containment link
- file_containment links exist but `parent_id` is empty
- Inconsistent parent_id vs link relationships

This would help catch data integrity issues earlier in the pipeline.
