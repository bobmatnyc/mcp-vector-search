# File Click Expansion - Debug Checklist

**Issue**: Files are not expanding to show chunks when clicked in the tree visualization.

**Server**: Running on http://localhost:8080

## Browser Console Checks

When you load the page, you should see these debug messages:

### 1. Initial Data Load
```
Loaded X nodes and Y links
```

### 2. Tree Structure Building
```
Filtered to X tree nodes (directories, files, and chunks)
Node breakdown: X directories, Y files, Z chunks
```

### 3. Chunk Attachment (CRITICAL)
Look for this section:
```
=== CHUNK ATTACHMENT DEBUG ===
Total nodes to check: X
Found Y chunk nodes
Chunks with file_id property: Z
Chunks successfully attached: N
Chunks missing parent file: M
Chunks missing in nodeMap: K
=== END CHUNK ATTACHMENT DEBUG ===
```

**Expected**: `Chunks successfully attached` should be > 0
**Problem indicators**:
- "Chunks with file_id property: 0" - chunks don't have file_id
- "Chunks missing parent file" > 0 - file IDs don't match
- "Chunks missing in nodeMap" > 0 - chunks not in tree

### 4. Post-Collapse File Check
```
=== POST-COLLAPSE FILE CHECK ===
File <name> has X chunks in _children
Checked Y files, Z have chunks
=== END POST-COLLAPSE FILE CHECK ===
```

**Expected**: Some files should have chunks in `_children`
**Problem indicator**: "0 have chunks" means chunks weren't attached

### 5. File Click Behavior
When you click a file, look for:
```
=== NODE CLICK DEBUG ===
Clicked node: <filename> (type: file, id: <id>)
Has children: 0
Has _children: X
Expanding file
=== END NODE CLICK DEBUG ===
```

**Expected**: `Has _children: <number>` should be > 0 for files with chunks
**Problem indicator**: "Has _children: 0" means no chunks attached

## Testing Steps

1. Open http://localhost:8080 in browser
2. Open browser console (F12 or Cmd+Option+I)
3. Expand a directory to see files
4. Click a file node
5. Check console for the debug messages above
6. Report findings below

## Findings

### Initial Load
- Total nodes: ___
- Directories: ___
- Files: ___
- Chunks: ___

### Chunk Attachment
- Chunks with file_id: ___
- Chunks attached: ___
- Chunks missing parent: ___
- Chunks missing in map: ___

### Post-Collapse Check
- Files checked: ___
- Files with chunks: ___

### File Click
- File clicked: ___
- Has _children: ___
- Behavior: [Expanded | No change | Error]

## Expected Data Structure

A file node should look like this:
```javascript
{
  id: "file-123",
  name: "example.py",
  type: "file",
  path: "/path/to/example.py",
  _children: [  // Initially collapsed
    {
      id: "chunk-456",
      name: "function_name",
      type: "chunk",
      chunk_type: "function",
      file_id: "file-123",  // MUST match parent file id
      content: "...",
      start_line: 10,
      end_line: 20
    },
    // ... more chunks
  ]
}
```

## Common Issues

1. **Chunks don't have file_id property**
   - Check graph data generation
   - Verify indexer adds file_id to chunks

2. **file_id doesn't match parent file's id**
   - Check ID generation consistency
   - Verify chunk creation uses correct file reference

3. **Chunks not included in treeNodes filter**
   - Check line 85-88 in scripts.py
   - Ensure chunks are included in filter

4. **Chunks attached but not collapsed**
   - Check collapseAll function (line 195-207)
   - Verify it processes file nodes

## Next Steps

Based on console output, determine:
- [ ] Are chunks being loaded from the API?
- [ ] Do chunks have file_id properties?
- [ ] Are chunks being attached to files?
- [ ] Are files collapsed correctly (chunks in _children)?
- [ ] Does clicking files trigger expansion logic?
