# Single-Child Chain Collapsing Analysis

**Research Date**: 2025-12-10
**Researcher**: Claude (Research Agent)
**Scope**: Visualization tree collapsing logic for directory chains and file-chunk relationships

---

## Executive Summary

The visualization code already implements sophisticated single-child chain collapsing for directories (e.g., `src/mcp_vector_search`). For files with single chunks, it currently **promotes chunk children to file level** rather than collapsing the names. To achieve `constants.py/L1` display, we need to modify the file-chunk collapsing logic to **combine names** (like directories do) instead of promoting children.

**Key Finding**: The collapsing logic is in `collapseSingleChildChains()` function at **lines 297-345** of `scripts.py`.

---

## 1. Location of Collapsing Logic

### File
`/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

### Function
`collapseSingleChildChains(node)` (lines 297-345)

### Invocation
Called at lines 347-352:
```javascript
// Apply single-child chain collapsing to all root children
console.log('=== COLLAPSING SINGLE-CHILD CHAINS ===');
if (treeData.children) {
    treeData.children.forEach(child => collapseSingleChildChains(child));
}
console.log('=== END COLLAPSING SINGLE-CHILD CHAINS ===');
```

---

## 2. How Directory Collapsing Works (Case 1)

### Code (Lines 303-320)
```javascript
// Case 1: Directory with single directory child - combine names
if (node.type === 'directory' && node.children.length === 1) {
    const onlyChild = node.children[0];
    if (onlyChild.type === 'directory') {
        // Merge: combine names with "/"
        console.log(`Collapsing dir chain: ${node.name} + ${onlyChild.name}`);
        node.name = `${node.name}/${onlyChild.name}`;  // ← KEY: Name combination
        // Take the child's children as our own
        node.children = onlyChild.children || [];
        node._children = onlyChild._children || null;
        // Preserve the deepest node's id for any link references
        node.collapsed_ids = node.collapsed_ids || [node.id];
        node.collapsed_ids.push(onlyChild.id);

        // Recursively check again in case there's another single child
        collapseSingleChildChains(node);
    }
}
```

### Behavior
- **Detection**: `type === 'directory'` AND `children.length === 1` AND `onlyChild.type === 'directory'`
- **Action**: Combine names with `/` separator
- **Result**: `src` + `mcp_vector_search` → `src/mcp_vector_search`
- **Metadata**: Stores `collapsed_ids` array to track merged nodes
- **Recursion**: Continues collapsing if new node also has single directory child

---

## 3. How File-Chunk Collapsing Works (Case 2)

### Code (Lines 322-344)
```javascript
// Case 2: File with single chunk child - promote chunk's children to file
// This handles files where there's just one L1 (e.g., imports or a single class)
if (node.type === 'file' && node.children && node.children.length === 1) {
    const onlyChild = node.children[0];
    if (chunkTypes.includes(onlyChild.type)) {
        // If the chunk has children, promote them to the file level
        const chunkChildren = onlyChild.children || onlyChild._children || [];
        if (chunkChildren.length > 0) {
            console.log(`Promoting ${chunkChildren.length} children from ${onlyChild.type} to file ${node.name}`);
            // Replace the single chunk with its children  ← KEY: Child promotion, NOT name combination
            node.children = chunkChildren;
            // Store info about the collapsed chunk
            node.collapsed_chunk = {
                type: onlyChild.type,
                name: onlyChild.name,
                id: onlyChild.id
            };
        } else {
            // Chunk has no children - just keep as is (will show chunk content on click)
            console.log(`File ${node.name} has single chunk ${onlyChild.type} with no children - keeping as is`);
        }
    }
}
```

### Behavior
- **Detection**: `type === 'file'` AND `children.length === 1` AND `onlyChild.type` in `chunkTypes`
- **Action (if chunk has children)**: Promote chunk's children to file level
  - File gets chunk's children directly
  - Chunk metadata stored in `node.collapsed_chunk`
  - **File name stays unchanged** (no combination)
- **Action (if chunk has NO children)**: Keep as-is
  - File still has single chunk child
  - Displays chunk content when clicked

### Current Problem
For `constants.py` with single `L1` chunk (no children):
- **Current behavior**: File node shows as `constants.py`, chunk shows as `L1` below it
- **Desired behavior**: File+chunk should collapse into `constants.py/L1` (like directory chains)

---

## 4. Chunk Type Definitions

### Supported Types (Line 54)
```javascript
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];
```

### Usage
- Used to identify code chunk nodes vs. structural nodes (directory, file)
- Prevents chunks from becoming root nodes (lines 246-250)
- Determines node styling, sizing, and behavior throughout visualization

---

## 5. Proposed Solution: Extend Collapsing to File+Chunk Names

### Modification Strategy
Update `collapseSingleChildChains()` Case 2 to **combine names** when chunk has no children:

```javascript
// Case 2: File with single chunk child
if (node.type === 'file' && node.children && node.children.length === 1) {
    const onlyChild = node.children[0];
    if (chunkTypes.includes(onlyChild.type)) {
        const chunkChildren = onlyChild.children || onlyChild._children || [];

        if (chunkChildren.length > 0) {
            // Existing behavior: promote chunk's children to file level
            console.log(`Promoting ${chunkChildren.length} children from ${onlyChild.type} to file ${node.name}`);
            node.children = chunkChildren;
            node.collapsed_chunk = {
                type: onlyChild.type,
                name: onlyChild.name,
                id: onlyChild.id
            };
        } else {
            // NEW BEHAVIOR: Collapse file+chunk into combined name
            console.log(`Collapsing file+chunk: ${node.name} + ${onlyChild.name}`);
            node.name = `${node.name}/${onlyChild.name}`;  // ← NEW: Name combination
            node.children = [];  // ← NEW: Remove chunk child (merged into name)
            node.collapsed_ids = node.collapsed_ids || [node.id];
            node.collapsed_ids.push(onlyChild.id);

            // Store chunk data for display (when clicked)
            node.collapsed_chunk = {
                type: onlyChild.type,
                name: onlyChild.name,
                id: onlyChild.id,
                content: onlyChild.content,  // Preserve chunk content
                start_line: onlyChild.start_line,
                end_line: onlyChild.end_line,
                complexity: onlyChild.complexity
            };
        }
    }
}
```

### Expected Behavior After Change

| File | Current Display | New Display |
|------|----------------|-------------|
| `constants.py` with single `L1` chunk (no children) | `constants.py`<br>└─ `L1` | `constants.py/L1` (collapsed) |
| `__init__.py` with single `imports` chunk with 3 function children | `__init__.py`<br>├─ `func1`<br>├─ `func2`<br>└─ `func3` | Same (chunk children promoted) |
| `utils.py` with 2 chunks | `utils.py`<br>├─ `L1`<br>└─ `L2` | Same (no collapsing) |

---

## 6. Additional Considerations

### Click Behavior
When user clicks `constants.py/L1`:
- **Current logic** (lines 1272-1307): Single-chunk files already display chunk content directly
- **After change**: Need to ensure collapsed file+chunk still displays chunk content
- **Solution**: Check for `node.collapsed_chunk` existence to determine content display

### Node Type
- **Question**: Should collapsed `file+chunk` keep `type: 'file'` or change to chunk type?
- **Recommendation**: Keep `type: 'file'` for consistency with directory collapsing
  - Directory `src/mcp` keeps `type: 'directory'`
  - File `constants.py/L1` should keep `type: 'file'`
  - Use `node.collapsed_chunk.type` to determine actual chunk type for styling

### Metadata Preservation
Store all chunk metadata in `node.collapsed_chunk`:
- `type`: Chunk type (function, class, text, etc.)
- `name`: Original chunk name (L1, L2, etc.)
- `id`: Chunk ID for link references
- `content`: Code content for viewer panel
- `start_line`, `end_line`: Line numbers
- `complexity`: Complexity metrics

### Recursion
Unlike directories, file+chunk collapsing doesn't need recursion:
- Files are leaf nodes in the structural hierarchy
- Chunks are always children of files (never nested file→chunk→file)
- No need to call `collapseSingleChildChains()` recursively after file+chunk merge

---

## 7. Testing Strategy

### Test Cases
1. **Single chunk, no children**: `constants.py` with `L1` → Should collapse to `constants.py/L1`
2. **Single chunk, has children**: `__init__.py` with `imports` containing 3 functions → Should promote children (existing behavior)
3. **Multiple chunks**: `utils.py` with `L1` and `L2` → Should not collapse (existing behavior)
4. **Empty file**: File with no chunks → Should display as-is (existing behavior)

### Validation
- **Visual inspection**: Check tree display shows `file.py/ChunkName` for single-chunk files
- **Click behavior**: Verify clicking collapsed node displays chunk content
- **Metadata**: Confirm `collapsed_chunk` metadata is preserved
- **Links**: Test that call graph links still work with collapsed nodes

---

## 8. Related Code Sections

### Node Click Handler (Lines 1272-1307)
Handles displaying content when node is clicked:
```javascript
if (childrenArray && childrenArray.length === 1) {
    const onlyChild = childrenArray[0];
    if (chunkTypes.includes(onlyChild.type)) {
        console.log(`Single-chunk file: ${nodeData.name}, showing content directly`);
        // Expand the file visually (for tree consistency)
        // ... (code to display chunk content)
    }
}
```
**Impact**: May need update to handle `node.collapsed_chunk` case

### Chunk Icon Helper (Referenced in lines 1467, 1504)
Gets icon for chunk type:
```javascript
getChunkIcon(chunkData.type)
```
**Impact**: Should work with `node.collapsed_chunk.type`

---

## 9. Implementation Checklist

- [ ] Modify `collapseSingleChildChains()` Case 2 to combine file+chunk names when chunk has no children
- [ ] Store complete chunk metadata in `node.collapsed_chunk`
- [ ] Update node click handler to detect `node.collapsed_chunk` for content display
- [ ] Test visual display of collapsed `file/chunk` nodes
- [ ] Test click behavior opens chunk content viewer
- [ ] Verify call graph links work with collapsed nodes
- [ ] Update console logging to reflect new collapsing behavior
- [ ] Test edge cases (empty files, multiple chunks, chunk with children)

---

## 10. Memory Usage Analysis

**Files Analyzed**: 1 file (`scripts.py`)
**Method**: Targeted grep searches (6 patterns) + strategic sampling (60 lines read)
**Memory Efficiency**: ✅ High (avoided loading 27KB file into memory)
**Search Tools Used**:
- Grep with context (-C flag) for pattern discovery
- Strategic offset reading for code inspection
- No full file loading required

---

## Conclusion

The single-child chain collapsing logic is well-structured and follows a clear pattern for directory chains. Extending this to file+chunk name combination requires:

1. **Minimal code change**: ~10 lines in `collapseSingleChildChains()` Case 2
2. **Consistent pattern**: Mimic directory collapsing logic (name combination + metadata storage)
3. **Backward compatible**: Only affects files with single chunk (no children)
4. **Low risk**: Existing promotion logic for chunks with children remains unchanged

**Next Steps**: Implement the proposed modification in `collapseSingleChildChains()` and test with representative codebase examples.

---

**Files Referenced**:
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (lines 54, 297-352, 1272-1307)
