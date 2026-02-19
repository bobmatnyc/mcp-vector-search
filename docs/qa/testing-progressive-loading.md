# Testing Progressive Loading Implementation

## Quick Start

### 1. Run Automated Tests
```bash
# Test the progressive loading logic on existing graph data
python test_progressive_loading.py
```

Expected output:
```
TEST: Initial Graph View
Total nodes in graph: 5432
Total links in graph: 8765

Initial view:
  - Nodes: 145
  - Links: 144
  - Collapsed children: 5287

Performance improvement:
  - Node reduction: 97.3%
  - Link reduction: 98.4%

✅ Initial view size is optimal (145 nodes)
```

### 2. Manual UI Testing
```bash
# Start the visualizer
mcp-vector-search visualize --port 8501

# Browser will open automatically
```

## Test Checklist

### ✅ Initial Load Performance
- [ ] Page loads in < 1 second
- [ ] Only top-level directories visible
- [ ] "+" indicators appear on directories
- [ ] No console errors in browser DevTools
- [ ] Network tab shows only 1 request to `/api/graph-initial`

### ✅ Directory Expansion
- [ ] Click directory with "+" icon
- [ ] Loading indicator appears briefly
- [ ] Children nodes appear in tree
- [ ] "+" icon disappears after expansion
- [ ] Network tab shows request to `/api/graph-expand/{node_id}`
- [ ] Expansion completes in < 500ms

### ✅ File Expansion
- [ ] Expand directory to reveal files
- [ ] Click file node
- [ ] Code chunks appear below file
- [ ] Code viewer shows file content
- [ ] Network tab shows request to `/api/graph-expand/{file_id}`

### ✅ Collapse Behavior
- [ ] Click expanded directory to collapse
- [ ] Children disappear from tree
- [ ] "+" icon reappears (if has children)
- [ ] No network request (local operation)

### ✅ Re-expansion (Cached)
- [ ] Click collapsed directory again
- [ ] Children reappear instantly
- [ ] No network request (cached)
- [ ] No loading indicator

### ✅ Deep Navigation
- [ ] Expand directory → subdirectory → file → chunk
- [ ] Each level loads incrementally
- [ ] Tree structure remains consistent
- [ ] Code viewer updates correctly

### ✅ Search Integration
- [ ] Use search box to find node
- [ ] Click search result
- [ ] Tree expands path to node
- [ ] Node is highlighted
- [ ] All parent nodes expanded automatically

## Performance Benchmarks

### Target Metrics
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Initial load | < 1s | Network tab: time to `/api/graph-initial` response |
| Node expansion | < 500ms | Network tab: time to `/api/graph-expand` response |
| Memory usage | < 50MB | DevTools Memory: Heap snapshot after initial load |
| Total nodes (initial) | 100-200 | Console log: "Loaded X nodes" |
| Network requests | 1 + N expansions | Network tab: count of `/api/graph-expand` calls |

### Measuring in Browser
1. Open browser DevTools (F12)
2. Go to Network tab
3. Refresh page
4. Check timing:
   - `/api/graph-initial`: Should be < 1s
   - `/api/graph-expand/{id}`: Should be < 500ms

5. Go to Memory tab
6. Take heap snapshot
7. Check memory usage: Should be < 50MB initially

### Command Line Testing
```bash
# Test with different codebase sizes
cd /path/to/small-project    # < 100 files
mcp-vector-search index
mcp-vector-search visualize

cd /path/to/medium-project   # 100-1000 files
mcp-vector-search index
mcp-vector-search visualize

cd /path/to/large-project    # 1000+ files
mcp-vector-search index
mcp-vector-search visualize
```

## Edge Cases to Test

### 1. Empty Directory
- **Setup**: Create empty directory in codebase
- **Expected**: Directory appears but has no "+" indicator
- **Test**: Click directory → no expansion occurs

### 2. Single-File Directory
- **Setup**: Directory with only one file
- **Expected**: Directory has "+" indicator
- **Test**: Click directory → file appears → click file → chunks load

### 3. Deeply Nested Structure
- **Setup**: Nested directories 10+ levels deep
- **Expected**: Each level expands incrementally
- **Test**: Expand all levels → no performance degradation

### 4. Large File (100+ chunks)
- **Setup**: File with many functions/classes
- **Expected**: All chunks load on file expansion
- **Test**: Click file → chunks appear → scroll works smoothly

### 5. Rapid Clicking
- **Setup**: Click multiple directories quickly
- **Expected**: No duplicate API calls, no race conditions
- **Test**: Click 5 directories rapidly → check Network tab for duplicates

### 6. Browser Back/Forward
- **Setup**: Expand several nodes, navigate away, return
- **Expected**: State is reset, fresh initial view
- **Test**: Browser back → page reloads → initial view only

### 7. Concurrent Expansions
- **Setup**: Expand two directories simultaneously
- **Expected**: Both load correctly, no conflicts
- **Test**: Click dir1, immediately click dir2 → both expand

## Debugging Tips

### Issue: Initial load is slow (> 1s)
**Check:**
- Graph file size: `ls -lh .mcp-vector-search/visualizations/chunk-graph.json`
- Number of top-level directories: Look for `depth: 0` and `depth: 1` nodes
- Network latency: Test on localhost

**Fix:**
- Reduce depth threshold in `/api/graph-initial` (change `depth <= 1` to `depth <= 0`)
- Add caching headers to API responses

### Issue: Expansion doesn't work (no children)
**Check:**
- Console errors in browser DevTools
- Network tab: Does `/api/graph-expand` return 200?
- Response JSON: Does it contain `nodes` and `links` arrays?

**Fix:**
- Verify node ID is correct (check `node.id` in console)
- Check link types in graph data (should be `dir_containment`, `file_containment`)

### Issue: "+" indicator doesn't disappear
**Check:**
- Is `node.expanded = true` being set in `expandNode()`?
- Is `renderVisualization()` being called after expansion?

**Fix:**
- Add debug log in `expandNode()`: `console.log('Marking as expanded:', nodeId)`
- Check if `buildTreeStructure()` preserves `expanded` flag

### Issue: Duplicate API calls
**Check:**
- Is node already in `expandedNodes` Set?
- Are multiple click handlers registered?

**Fix:**
- Add guard: `if (expandedNodes.has(nodeId)) return;`
- Use `event.stopPropagation()` to prevent bubbling

### Issue: Memory leak on repeated expansions
**Check:**
- Are old nodes being removed when collapsing?
- Is D3 selection cleanup happening?

**Fix:**
- Use `.remove()` on D3 selections before re-render
- Clear event listeners on removed nodes

## Console Commands for Debugging

### Check initial state
```javascript
// In browser console
console.log('Total nodes:', allNodes.length);
console.log('Expanded nodes:', expandedNodes);
console.log('Node types:', allNodes.reduce((acc, n) => {
    acc[n.type] = (acc[n.type] || 0) + 1;
    return acc;
}, {}));
```

### Manually expand a node
```javascript
// Find node by name
const node = allNodes.find(n => n.name === 'src');
console.log('Node ID:', node.id);

// Expand it
await expandNode(node.id);
```

### Check for duplicate nodes
```javascript
const ids = allNodes.map(n => n.id);
const duplicates = ids.filter((id, i) => ids.indexOf(id) !== i);
console.log('Duplicate IDs:', duplicates);
```

### Monitor API calls
```javascript
// Override fetch to log calls
const originalFetch = window.fetch;
window.fetch = function(...args) {
    console.log('Fetch:', args[0]);
    return originalFetch.apply(this, args);
};
```

## Performance Profiling

### Using Chrome DevTools

1. **Record Performance**:
   - Open DevTools → Performance tab
   - Click record (red circle)
   - Expand a few nodes
   - Stop recording
   - Analyze flamegraph for slow functions

2. **Memory Profiling**:
   - Open DevTools → Memory tab
   - Take heap snapshot before expansion
   - Expand several nodes
   - Take another heap snapshot
   - Compare allocations

3. **Network Throttling**:
   - Open DevTools → Network tab
   - Select "Slow 3G" from throttling dropdown
   - Test if expansion still works smoothly
   - Adjust timeout values if needed

## Regression Testing

### Before Deploying
1. Test with small project (< 100 files)
2. Test with medium project (100-1000 files)
3. Test with large project (1000+ files)
4. Test on Safari, Chrome, Firefox
5. Test on mobile (responsive view)
6. Test with keyboard navigation
7. Test with screen reader (accessibility)

### Automated Testing (Future)
```bash
# Unit tests for API endpoints
pytest tests/unit/cli/commands/visualize/test_progressive_loading.py

# Integration tests for expansion logic
pytest tests/integration/visualize/test_expansion.py

# E2E tests with Selenium
pytest tests/e2e/visualize/test_progressive_ui.py
```

## Known Limitations

1. **Aggregation nodes**: Not yet implemented for directories with 100+ children
2. **Prefetching**: No hover prefetching (could add in Phase 2)
3. **Virtual scrolling**: Large lists (1000+ items) may lag
4. **State persistence**: Expansion state not saved on page refresh
5. **Deep linking**: Cannot link directly to expanded state

## Success Criteria

✅ **Must Have**:
- Initial load < 1s for any codebase size
- Expansion < 500ms per node
- No duplicate API calls
- No memory leaks
- Consistent tree structure

✅ **Should Have**:
- Expand indicators visible
- Loading feedback clear
- Smooth animations
- Keyboard accessible
- Mobile responsive

✅ **Nice to Have**:
- Hover prefetching
- State persistence
- Breadcrumb navigation
- Bulk expand/collapse
- Deep linking support

---

**Test Coverage Target**: 90%+
**Performance Target**: 5-10x improvement over full load
**User Experience**: Instant initial load, responsive interactions
