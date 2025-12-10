# Tree Hierarchy Fix Verification Checklist

## Quick Testing Guide

### 1. Restart Visualization Server
```bash
# Stop current server (Ctrl+C if running)
# Then restart:
mcp-vector-search visualize
```

### 2. Open Browser
Navigate to: `http://localhost:8000`

### 3. Browser Console Checks

Open Developer Tools (F12) and check console output:

**✅ Expected Console Output:**
```
=== LINK STRUCTURE DEBUG ===
Total links: 7138
Link types found: ["dir_hierarchy", "dir_containment", "file_containment", ...]
Link type counts: {dir_hierarchy: 43, dir_containment: 349, file_containment: 6746}
...

=== BUILDING TREE RELATIONSHIPS ===
Relationship processing summary:
  dir_hierarchy: 43/43 matched
  dir_containment: 349/349 matched
  file_containment: 6746/6746 matched
  Total parent-child links: 7138
=== END TREE RELATIONSHIPS ===

=== ROOT NODE ANALYSIS ===
Found 1 root nodes (nodes without parents)
Root node type breakdown: {directory: 1}
INFO: 0 file nodes are roots (this is normal for files not in subdirectories)
=== END ROOT NODE ANALYSIS ===
```

**❌ Bad Console Output (Before Fix):**
```
Found 6758 root nodes
Root node type breakdown: {function: 3241, class: 1829, file: 349, ...}
WARNING: 6746 chunk nodes are roots - they should be children of files!
```

### 4. Visual Verification

**✅ Expected Visual Structure:**
```
Project Root
├── src/
│   ├── mcp_vector_search/
│   │   ├── core/
│   │   │   ├── indexer.py [gray - collapsed]
│   │   │   ├── search.py [gray - collapsed]
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── tests/
│   └── ...
└── docs/
    └── ...
```

**❌ Bad Visual Structure (Before Fix):**
```
Project Root
├── function: index_codebase (chunk - should be in file!)
├── function: search_similar (chunk - should be in file!)
├── class: Indexer (chunk - should be in file!)
├── ... (6758 separate root nodes)
```

### 5. Interaction Tests

**Test 1: Expand Directory**
1. Click on `src/` directory (orange circle)
2. **Expected:** Directory expands (turns blue), shows subdirectories and files
3. **Fail:** Nothing happens or visualization crashes

**Test 2: Expand File**
1. Click on a file (gray circle) like `indexer.py`
2. **Expected:** File expands (turns white), shows code chunks (purple circles)
3. **Fail:** No chunks appear or file has no children

**Test 3: View Chunk**
1. Expand a file to show chunks
2. Click on a chunk (purple circle)
3. **Expected:** Side panel opens showing chunk code and metadata
4. **Fail:** Panel doesn't open or shows "No content available"

### 6. Metrics Validation

**Expected Metrics (from console):**
- Total tree nodes: ~7,138 (directories + files + chunks)
- Total parent-child links: ~7,138 (dir_hierarchy + dir_containment + file_containment)
- Root nodes: 1-10 (likely just 1 if all files are in directories)
- Chunk roots: 0 (no WARNING messages)
- File roots: 0-5 (top-level files like README.md, if any)

**Success Criteria:**
- ✅ Root nodes < 20 (ideally 1-5)
- ✅ No chunk nodes as roots
- ✅ All link types fully matched (matched === processed)
- ✅ Tree displays hierarchical structure
- ✅ Expand/collapse works for directories and files
- ✅ Chunk content displays in side panel

## Troubleshooting

### Issue: Still seeing 6758 root nodes
**Cause:** Browser cache serving old JavaScript
**Solution:**
1. Hard refresh browser: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
2. Or clear cache in Developer Tools → Application → Storage → Clear site data
3. Or restart server with new timestamp (automatically regenerates HTML)

### Issue: Console shows "link skipped" messages
**Cause:** Nodes referenced in links but not in the graph data
**Investigation:**
1. Check which link types are being skipped
2. Verify node IDs match between nodes and links
3. Check if nodes are being filtered out by type filter

### Issue: Files have no chunks
**Cause:** file_containment links not matching
**Investigation:**
1. Check console for "file_containment: X/Y matched" where X < Y
2. Look for "link skipped" messages for file_containment
3. Verify chunk types are in chunkTypes array: `['function', 'class', 'method', 'text', 'imports', 'module']`

### Issue: Visualization crashes on click
**Cause:** Null reference in click handler
**Investigation:**
1. Check browser console for JavaScript errors
2. Verify node data has expected structure (name, type, id)
3. Check if `children` and `_children` arrays are properly initialized

## Regression Testing

After fix, verify these still work:
- ✅ Linear layout mode (default)
- ✅ Circular layout mode (toggle switch)
- ✅ Zoom and pan functionality
- ✅ Side panel open/close
- ✅ Directory info display
- ✅ File info display
- ✅ Chunk content display

## Performance Baseline

**Expected Performance (for ~7000 nodes):**
- Initial load: < 2 seconds
- Tree rendering: < 500ms
- Layout toggle: < 500ms
- Node expand/collapse: < 200ms
- Side panel open: < 100ms

**Red Flags:**
- Load time > 5 seconds
- Rendering hangs browser
- Memory leaks (check Task Manager after multiple interactions)
- Frame rate drops below 30fps during interactions
