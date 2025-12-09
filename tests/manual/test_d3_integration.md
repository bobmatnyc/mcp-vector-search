# D3.js Integration Manual Test Guide

## Quick Test Procedure

### 1. Generate Visualization
```bash
cd /Users/masa/Projects/mcp-vector-search
mcp-vector-search visualize
```

### 2. Open in Browser
The visualization should open automatically. If not:
```bash
open .mcp-vector-search/visualization/index.html
```

### 3. Open Browser Console
- **Chrome/Edge**: Press `F12` or `Cmd+Option+I` (Mac)
- **Firefox**: Press `F12` or `Cmd+Option+K` (Mac)
- **Safari**: Enable Developer menu first, then `Cmd+Option+C`

### 4. Verify D3.js Loaded
In the console, type:
```javascript
typeof d3
```
Expected output: `"object"`

If you see `"undefined"`, D3.js failed to load. Check:
- Network tab for failed requests to `d3js.org`
- Console for CORS errors
- Internet connectivity

### 5. Test Basic Expansion

**Step 1**: Click any root directory node (e.g., `src/`)

**Expected Behavior**:
- Node expands with smooth 750ms animation
- Children arrange in radial circle around parent
- Console shows: `[D3 Layout] Positioned N children using D3 tree layout, radius=XXX.Xpx, arc=XXX.Xpx/child`

**Step 2**: Click one of the child directories

**Expected Behavior**:
- Its children fan out radially around it (second ring)
- Parent remains at same position
- New children smoothly animate in

**Step 3**: Continue expanding 3-4 levels deep

**Expected Behavior**:
- Concentric rings form (radial tree pattern)
- No node overlap
- All nodes visible on screen

### 6. Verify On-Demand Loading

**Test**: Expand a directory with subdirectories

**Expected**:
- ✅ Only immediate children appear
- ✅ Grandchildren NOT shown until their parent is clicked
- ❌ Should NOT see entire subtree expanded at once

**Console Verification**:
```
[Expand] directory - showing 8 immediate children only (on-demand expansion)
```

### 7. Check Layout Quality

**Spacing**:
- [ ] Children evenly distributed around parent
- [ ] No overlapping nodes
- [ ] No overlapping labels

**Radius Adaptation**:
- [ ] Few children (1-5): Small radius (~200px)
- [ ] Many children (20+): Larger radius (~300-400px)

**Visual Hierarchy**:
- [ ] Clear parent-child relationships
- [ ] Connecting edges visible
- [ ] Directory icons vs file icons distinguishable

### 8. Performance Check

**Test Large Expansion**:
1. Find a directory with 20+ files
2. Click to expand
3. Measure expansion time (should be <100ms)

**Console Check**:
- Look for any JavaScript errors
- Check for warnings about missing nodes
- Verify all layout calculations complete

### 9. Test Fallback (Optional)

**Simulate D3.js Failure**:
1. Open DevTools → Network tab
2. Right-click on `d3.v7.min.js` request
3. Select "Block request URL" (Chrome) or similar
4. Reload page

**Expected Fallback Behavior**:
- Console warning: `[Layout] D3.js not available, using fallback radial layout`
- Visualization still works
- Layout uses simple radial fallback

## Success Criteria Checklist

### Critical (Must Pass)
- [ ] D3.js loads without errors (`typeof d3 === "object"`)
- [ ] Clicking nodes expands children in radial pattern
- [ ] On-demand expansion (only immediate children shown)
- [ ] No JavaScript errors in console
- [ ] Smooth 750ms transitions

### Important (Should Pass)
- [ ] Console shows `[D3 Layout]` debug messages
- [ ] Children evenly spaced in circle
- [ ] No node overlap or collision
- [ ] All nodes fit on screen (1920x1080)
- [ ] Connecting edges render correctly

### Nice to Have (Good to Check)
- [ ] Adaptive radius (200-400px range observed)
- [ ] Directories appear before files in circle
- [ ] Labels readable (no text overlap)
- [ ] Multiple levels form concentric rings
- [ ] Fallback works when D3.js blocked

## Common Issues & Solutions

### Issue: `d3 is not defined`
**Cause**: D3.js script failed to load from CDN
**Solutions**:
1. Check internet connection
2. Try different browser
3. Check browser console for CORS errors
4. Fallback should activate automatically

### Issue: Nodes Overlap
**Cause**: Too many children for current radius
**Solutions**:
1. Check if radius calculation is correct (max 400px)
2. Verify separation function is being called
3. May need to increase `maxRadius` in code

### Issue: No Animation
**Cause**: Transition not applying
**Solutions**:
1. Check `renderGraphV2()` is called with duration
2. Verify D3/SVG transitions are working
3. Look for CSS that might disable transitions

### Issue: Children Don't Expand
**Cause**: Event handlers not attached
**Solutions**:
1. Check `expandNodeV2()` is being called
2. Verify `getImmediateChildren()` returns nodes
3. Check state manager is updating correctly

## Screenshot Comparison

**Before D3.js Integration** (Custom Radial):
- Fixed spacing algorithm
- Simple circular distribution
- Manual radius calculation

**After D3.js Integration** (D3 Tree):
- Adaptive separation based on tree depth
- Collision avoidance
- Industry-standard spacing heuristics

## Expected Console Output

### Successful Load
```
[Init V2] Initializing graph data...
[Init V2] View mode: tree_root
[Init V2] Visible nodes: 12
[Render] Rendering graph, mode: tree_root, phase: Phase 1 (overview)
```

### First Expansion
```
[Expand] directory - showing 8 immediate children only (on-demand expansion)
[D3 Layout] Positioned 8 children using D3 tree layout, radius=250.0px, arc=196.3px/child
[Render] Rendering graph, mode: tree_expanded, phase: Phase 2 (radial)
[Render] tree_expanded: Radial layout, 9 nodes positioned
```

### Subsequent Expansions
```
[Expand] directory - showing 5 immediate children only (on-demand expansion)
[D3 Layout] Positioned 5 children using D3 tree layout, radius=200.0px, arc=251.3px/child
[Render] tree_expanded: Radial layout, 14 nodes positioned
```

## Reporting Issues

If you encounter problems, collect:

1. **Browser Info**: Name, version, OS
2. **Console Output**: Copy all error messages
3. **Network Tab**: Any failed requests
4. **Screenshot**: Visual of the issue
5. **Steps to Reproduce**: Exact click sequence

Save to: `/tests/manual/d3_integration_results.md`

## Performance Benchmarks

Record expansion times for comparison:

| Child Count | D3 Layout Time | Render Time | Total |
|-------------|----------------|-------------|-------|
| 5 children  | ~1-2ms         | ~10-20ms    | <30ms |
| 10 children | ~2-3ms         | ~15-25ms    | <40ms |
| 20 children | ~3-5ms         | ~20-30ms    | <50ms |
| 50 children | ~5-8ms         | ~30-50ms    | <80ms |

Use browser DevTools Performance tab to measure.
