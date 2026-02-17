# KG Integration Verification Checklist

## Pre-Deployment Verification

Use this checklist to verify the integration is working correctly before deploying.

## ✅ File Changes

- [x] `/src/mcp_vector_search/cli/commands/visualize/templates/base.py`
  - [x] Added view switcher (Chunks/Knowledge Graph buttons)
  - [x] Added `#chunk-controls` div with layout mode buttons
  - [x] Added `#kg-controls` div with relationship filters
  - [x] Added `<svg id="kg-graph">` container

- [x] `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - [x] Added `setView(view)` function
  - [x] Added `loadKGData()` function
  - [x] Added `renderKG()` function
  - [x] Added `filterKGLinks()` function
  - [x] Added `showKGNodeInfo(node)` function
  - [x] Added D3.js drag handlers (dragStarted, dragged, dragEnded)
  - [x] Added color schemes (kgNodeColors, kgLinkColors)

- [x] `/src/mcp_vector_search/cli/commands/visualize/templates/styles.py`
  - [x] Added `get_kg_styles()` function
  - [x] Integrated into `get_all_styles()`

- [x] `/src/mcp_vector_search/cli/commands/visualize/cli.py`
  - [x] Added `_export_kg_data(viz_dir)` function
  - [x] Added auto-export of KG data in `serve()` command

- [x] `/src/mcp_vector_search/cli/commands/visualize/server.py`
  - [x] Verified `/api/kg-graph` endpoint exists (already implemented)

## 🧪 Functional Tests

### Test 1: View Switching
```bash
# Start visualizer
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Page loads with "Chunks" view active
2. [ ] Click "Knowledge Graph" button
3. [ ] Chunks view hides, KG view shows
4. [ ] "Chunk Controls" hide, "KG Controls" show
5. [ ] Click "Chunks" button
6. [ ] KG view hides, Chunks view shows
7. [ ] "KG Controls" hide, "Chunk Controls" show

**Expected:** Seamless switching with no errors in console.

### Test 2: KG Data Loading (With KG Built)
```bash
# Ensure KG is built
mcp-vector-search kg build

# Start visualizer
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Click "Knowledge Graph" button
2. [ ] Console shows: "Loading KG data..."
3. [ ] Console shows: "Loaded X KG nodes, Y KG links"
4. [ ] Console shows: "Rendering KG..."
5. [ ] Force-directed graph appears
6. [ ] Nodes are colored by type
7. [ ] Links are colored by relationship type

**Expected:** Graph renders without errors.

### Test 3: KG Data Loading (Without KG Built)
```bash
# Remove KG data
rm -rf .mcp-vector-search/knowledge_graph/

# Start visualizer
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Click "Knowledge Graph" button
2. [ ] Alert shows: "Knowledge graph not found. Run: mcp-vector-search kg build"
3. [ ] View switches back to "Chunks"

**Expected:** Graceful fallback to chunks view.

### Test 4: Node Interaction
```bash
mcp-vector-search kg build
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Switch to "Knowledge Graph" view
2. [ ] Hover over node → stroke color changes to blue (#60a5fa)
3. [ ] Click node → side panel opens with node details
4. [ ] Details show: Name, Type, File, Connections
5. [ ] Drag node → node follows mouse
6. [ ] Release node → physics simulation continues

**Expected:** All interactions work smoothly.

### Test 5: Relationship Filtering
**In Browser:**
1. [ ] Switch to "Knowledge Graph" view
2. [ ] Uncheck "Calls" → orange links disappear
3. [ ] Uncheck "Imports" → blue links disappear
4. [ ] Uncheck "Inherits" → red links disappear
5. [ ] Uncheck "Contains" → gray links disappear
6. [ ] Re-check all → all links reappear

**Expected:** Links hide/show instantly with no re-render.

### Test 6: Zoom and Pan
**In Browser:**
1. [ ] Switch to "Knowledge Graph" view
2. [ ] Scroll up → graph zooms in
3. [ ] Scroll down → graph zooms out
4. [ ] Drag background → graph pans
5. [ ] Zoom and pan persist when switching views

**Expected:** Smooth zoom/pan behavior.

### Test 7: Link Hover
**In Browser:**
1. [ ] Switch to "Knowledge Graph" view
2. [ ] Hover over link → opacity increases, width increases
3. [ ] Move away → opacity/width returns to normal

**Expected:** Visual feedback on hover.

### Test 8: Performance
**In Browser:**
1. [ ] Switch to "Knowledge Graph" view
2. [ ] Open DevTools → Performance tab
3. [ ] Record performance while dragging nodes
4. [ ] Check frame rate: Should maintain ~60fps
5. [ ] Check memory: Should stay <50MB

**Expected:** Smooth performance with no frame drops.

## 🔍 Code Quality Checks

### Python Code
```bash
# Type checking (if mypy is configured)
mypy src/mcp_vector_search/cli/commands/visualize/

# Linting
flake8 src/mcp_vector_search/cli/commands/visualize/

# Format check
black --check src/mcp_vector_search/cli/commands/visualize/
```

**Expected:** No errors or warnings.

### JavaScript Code (Manual Review)
1. [ ] No unused variables
2. [ ] Consistent naming conventions
3. [ ] Error handling for all async operations
4. [ ] Comments for complex logic
5. [ ] No console.log statements in production code

### CSS Code (Manual Review)
1. [ ] Uses CSS variables for theming
2. [ ] Consistent spacing and indentation
3. [ ] No duplicate selectors
4. [ ] Mobile-responsive (if applicable)

## 📊 Data Validation

### Chunk Graph Data
```bash
# Check chunk graph JSON
cat .mcp-vector-search/visualization/chunk-graph.json | jq '.nodes | length'
cat .mcp-vector-search/visualization/chunk-graph.json | jq '.links | length'
```

**Expected:** Non-zero counts.

### KG Graph Data
```bash
# Check KG graph JSON
cat .mcp-vector-search/visualization/kg-graph.json | jq '.nodes | length'
cat .mcp-vector-search/visualization/kg-graph.json | jq '.links | length'
```

**Expected:** Non-zero counts (if KG built).

### API Endpoints
```bash
# Test chunk graph endpoint
curl -s http://localhost:8501/api/graph | jq '.nodes | length'

# Test KG graph endpoint
curl -s http://localhost:8501/api/kg-graph | jq '.nodes | length'
```

**Expected:** Non-zero counts returned.

## 🌐 Browser Compatibility

Test in multiple browsers:

### Chrome/Edge
1. [ ] View switching works
2. [ ] KG renders correctly
3. [ ] Interactions work (drag, zoom, pan)
4. [ ] No console errors

### Firefox
1. [ ] View switching works
2. [ ] KG renders correctly
3. [ ] Interactions work (drag, zoom, pan)
4. [ ] No console errors

### Safari
1. [ ] View switching works
2. [ ] KG renders correctly
3. [ ] Interactions work (drag, zoom, pan)
4. [ ] No console errors

## 📱 Responsive Design (Optional)

Test on mobile/tablet:
1. [ ] Controls are accessible
2. [ ] View switcher works
3. [ ] Graph is zoomable/pannable
4. [ ] No horizontal scrolling

## 🛠️ Edge Cases

### Test 1: Empty KG
```bash
# Create empty KG data
echo '{"nodes": [], "links": []}' > .mcp-vector-search/visualization/kg-graph.json
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Switch to "Knowledge Graph"
2. [ ] Empty graph appears (no nodes/links)
3. [ ] No errors in console

**Expected:** Graceful handling of empty data.

### Test 2: Malformed KG Data
```bash
# Create invalid JSON
echo '{"nodes": [}' > .mcp-vector-search/visualization/kg-graph.json
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Switch to "Knowledge Graph"
2. [ ] Error alert appears
3. [ ] View switches back to "Chunks"

**Expected:** Error handling prevents crash.

### Test 3: Missing KG File
```bash
# Remove KG file
rm .mcp-vector-search/visualization/kg-graph.json
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Switch to "Knowledge Graph"
2. [ ] 404 error handled gracefully
3. [ ] Alert shows appropriate message

**Expected:** Fallback to chunks view.

### Test 4: Large Graph (>1000 nodes)
```bash
# Build full KG on large codebase
mcp-vector-search kg build
mcp-vector-search visualize
```

**In Browser:**
1. [ ] Switch to "Knowledge Graph"
2. [ ] Loading time <5 seconds
3. [ ] Rendering time <2 seconds
4. [ ] Interactions remain smooth

**Expected:** Acceptable performance at scale.

## 📝 Documentation Checks

1. [ ] INTEGRATION_SUMMARY.md is accurate
2. [ ] KG_INTEGRATION_ARCHITECTURE.md diagrams are correct
3. [ ] KG_USAGE_EXAMPLES.md examples work
4. [ ] VERIFICATION_CHECKLIST.md (this file) is complete
5. [ ] Code comments explain complex logic
6. [ ] API endpoints are documented

## 🚀 Pre-Deployment Checklist

Before merging/deploying:

1. [ ] All functional tests pass
2. [ ] All code quality checks pass
3. [ ] All data validation tests pass
4. [ ] Browser compatibility verified
5. [ ] Edge cases handled gracefully
6. [ ] Documentation is complete and accurate
7. [ ] No breaking changes to existing functionality
8. [ ] Performance is acceptable
9. [ ] Error handling is robust
10. [ ] User experience is smooth

## 🐛 Known Issues (If Any)

Document any known issues here:

- [ ] Issue 1: [Description]
  - **Impact:** [High/Medium/Low]
  - **Workaround:** [If available]
  - **Fix:** [Planned/In Progress/Won't Fix]

- [ ] Issue 2: [Description]
  - **Impact:** [High/Medium/Low]
  - **Workaround:** [If available]
  - **Fix:** [Planned/In Progress/Won't Fix]

## ✅ Sign-Off

Once all checks pass:

- [ ] Developer: [Name] - [Date]
- [ ] Reviewer: [Name] - [Date]
- [ ] QA: [Name] - [Date]

## 📞 Support

If issues are found during verification:

1. Check console for errors
2. Review server logs
3. Verify KG was built: `mcp-vector-search kg stats`
4. Test with minimal data: `mcp-vector-search kg build --limit 10`
5. Report issues with:
   - Browser version
   - OS version
   - Console errors
   - Steps to reproduce

## Next Steps After Verification

1. Create PR with changes
2. Request code review
3. Run CI/CD pipeline
4. Deploy to staging
5. User acceptance testing
6. Deploy to production
7. Monitor for issues
8. Update changelog
