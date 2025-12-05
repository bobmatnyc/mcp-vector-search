# Manual Testing Instructions for Compact Folder Layout

## Prerequisites
1. Run: `mcp-vector-search visualize export`
2. This will generate the visualization HTML file

## Test Scenarios

### Small Project (4 folders - 2×2 grid)
1. Copy `test_graph_small.json` to `chunk-graph.json` in visualization directory
2. Open visualization HTML in browser
3. Verify:
   - [ ] 4 folders visible in viewport
   - [ ] Arranged in ~2×2 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

### Medium Project (9 folders - 3×3 grid)
1. Copy `test_graph_medium.json` to `chunk-graph.json`
2. Open visualization HTML in browser
3. Verify:
   - [ ] 9 folders visible in viewport
   - [ ] Arranged in 3×3 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

### Large Project (16 folders - 4×4 grid)
1. Copy `test_graph_large.json` to `chunk-graph.json`
2. Open visualization HTML in browser
3. Verify:
   - [ ] 16 folders visible in viewport
   - [ ] Arranged in 4×4 grid pattern
   - [ ] Spacing approximately 150px
   - [ ] All folders centered in viewport
   - [ ] Click "Reset View" - same layout restored

## Interaction Tests

For each scenario, also test:

1. **Initial Load**
   - [ ] All folders visible immediately
   - [ ] No need to zoom/pan to see all folders
   - [ ] Smooth animation to final positions

2. **Expansion**
   - [ ] Click folder to expand
   - [ ] Files appear below folder
   - [ ] Grid layout maintained for other folders

3. **Collapse**
   - [ ] Click expanded folder to collapse
   - [ ] Files disappear
   - [ ] Grid layout restored

4. **Reset View**
   - [ ] Click "Reset View" button
   - [ ] All folders visible again
   - [ ] Same grid layout as initial load
   - [ ] Smooth transition

5. **Zoom/Pan**
   - [ ] Manually zoom in/out
   - [ ] Click "Reset View"
   - [ ] Zoom restored to show all folders

## Success Criteria

✅ All folders visible in initial viewport
✅ Grid pattern clear and organized
✅ Spacing consistent (approximately 150px)
✅ Reset view restores compact layout
✅ Smooth transitions and animations
✅ No performance issues

## Troubleshooting

**Problem**: Folders scattered randomly
- **Solution**: Check that `positionFoldersCompactly()` is being called
- **Check**: Browser console for errors

**Problem**: Not all folders visible
- **Solution**: Check `zoomToFit()` is being called with correct timing
- **Check**: Increase padding or adjust scale factor

**Problem**: Folders overlap
- **Solution**: Increase spacing constant (currently 150px)
- **Check**: Collision detection radius settings

**Problem**: Jerky animation
- **Solution**: Check timing of position fixes/releases
- **Check**: Browser performance in dev tools

## Reporting Results

After testing, report:
1. Which scenarios passed/failed
2. Browser used (Chrome, Firefox, Safari)
3. Viewport size when testing
4. Any visual issues or bugs observed
5. Performance observations

## Files Generated

- `test_graph_small.json` - 4 folders
- `test_graph_medium.json` - 9 folders
- `test_graph_large.json` - 16 folders
- `test_instructions.md` - This file
