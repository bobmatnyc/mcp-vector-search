# Chunk Tree Expansion - Verification Checklist

**Date**: 2025-12-09
**Feature**: File expansion to show chunks as tree children

## Pre-Testing Setup

- [ ] Run `mcp-vector-search index` to ensure chunk-graph.json is up to date
- [ ] Verify chunk-graph.json contains chunks with `file_id` properties
- [ ] Start visualization server: `mcp-vector-search visualize`

## Visual Verification

### Node Colors
- [ ] Directories show orange when collapsed (⚫ with `_children`)
- [ ] Directories show blue when expanded (⚫ with `children`)
- [ ] Files show gray when collapsed (⚫ with `_children`)
- [ ] Files show light gray/white when expanded or empty
- [ ] Chunks show purple circles (smaller than files/dirs)

### Node Sizes
- [ ] Directory circles are 6px radius
- [ ] File circles are 6px radius
- [ ] Chunk circles are 4px radius (visibly smaller)

### Text Labels
- [ ] Directory/file names show in 12px black text
- [ ] Chunk labels show chunk type (function, class, method, etc.) in 10px purple text
- [ ] Labels are readable in both linear and circular layouts

## Interaction Testing

### Directory Behavior (unchanged)
- [ ] Click collapsed directory → expands to show children
- [ ] Click expanded directory → collapses to hide children
- [ ] Directory toggle works in linear layout
- [ ] Directory toggle works in circular layout

### File Behavior (NEW)
- [ ] Click collapsed file → expands to show chunk children
- [ ] Click expanded file → collapses to hide chunks
- [ ] Files without chunks don't expand (no visible action)
- [ ] File toggle works in linear layout
- [ ] File toggle works in circular layout

### Chunk Behavior (NEW)
- [ ] Click chunk → opens side panel with chunk details
- [ ] Side panel shows chunk type
- [ ] Side panel shows line numbers (start - end)
- [ ] Side panel shows chunk content (code)
- [ ] Close button on side panel works
- [ ] Multiple chunk clicks update side panel content

### Tree Hierarchy
- [ ] Files appear as children of directories
- [ ] Chunks appear as children of files (when expanded)
- [ ] Tree depth reflects directory > file > chunk hierarchy
- [ ] Collapsing a directory hides all descendants (files + chunks)
- [ ] Collapsing a file hides only its chunks

## Layout Toggle Testing

### Linear Layout
- [ ] Switch to linear layout button works
- [ ] All node types render correctly
- [ ] Colors match specification
- [ ] Expansion/collapse works
- [ ] Tree orientation is left-to-right

### Circular Layout
- [ ] Switch to circular layout button works
- [ ] All node types render correctly
- [ ] Colors match specification
- [ ] Expansion/collapse works
- [ ] Tree orientation is radial from center

### Layout Switching
- [ ] Switch between layouts preserves expansion state
- [ ] No JavaScript errors in console during switch
- [ ] Zoom level resets appropriately

## Performance Testing

### Small Codebase (<100 files)
- [ ] Initial load completes in <2 seconds
- [ ] Tree renders without lag
- [ ] Expand/collapse is instant
- [ ] No console errors

### Medium Codebase (100-500 files)
- [ ] Initial load completes in <5 seconds
- [ ] Tree renders smoothly
- [ ] Expand/collapse is responsive (<500ms)
- [ ] Browser memory usage acceptable

### Large Codebase (>500 files)
- [ ] Initial load completes in <10 seconds
- [ ] Tree renders (may have slight delay)
- [ ] Expand/collapse is functional
- [ ] Browser doesn't freeze or crash

## Edge Cases

### Files with No Chunks
- [ ] Empty files don't show expand indicator
- [ ] Clicking empty file has no effect
- [ ] No console errors

### Files with Many Chunks (>20)
- [ ] All chunks render when file expanded
- [ ] Layout adjusts to accommodate children
- [ ] Scrolling/panning works to view all chunks
- [ ] No significant performance degradation

### Deeply Nested Directories
- [ ] Deep nesting (>5 levels) renders correctly
- [ ] Chunks at deep levels are accessible
- [ ] Layout doesn't break with deep hierarchies

### Special Characters in Names
- [ ] Files with spaces render correctly
- [ ] Chunks with special characters in type names display properly
- [ ] Unicode characters in labels work

## API/Data Verification

### Chunk Data Structure
- [ ] Verify chunks have `id` property
- [ ] Verify chunks have `type: 'chunk'`
- [ ] Verify chunks have `file_id` pointing to parent
- [ ] Verify chunks have `chunk_type` (function, class, etc.)
- [ ] Verify chunks have `content` property
- [ ] Verify chunks have `start_line` and `end_line`

### Network Requests
- [ ] Only one request to `/api/graph` on page load
- [ ] No requests to `/api/chunks` (deprecated for tree view)
- [ ] No excessive API calls during interaction

## Console Testing

### Expected Console Output
```
Loaded X nodes and Y links
Filtered to Z tree nodes (directories, files, and chunks)
Found N root nodes
Tree structure built with all directories and files collapsed
```

### Error Checking
- [ ] No JavaScript errors in console
- [ ] No React/Vue framework warnings (shouldn't be any)
- [ ] No 404 errors for missing resources
- [ ] No CORS errors

## Regression Testing

### Existing Features Still Work
- [ ] Zoom (mouse wheel) works
- [ ] Pan (click and drag background) works
- [ ] Layout toggle button works
- [ ] Side panel close button works
- [ ] Browser back/forward buttons work (if applicable)

## Browser Compatibility

- [ ] Chrome/Chromium (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

## Cleanup Verification

- [ ] No test files left in project root
- [ ] Summary document saved to `/docs/summaries/`
- [ ] No debug console.log statements in production code
- [ ] No commented-out code blocks

## Success Criteria

All checkboxes above should be checked before considering feature complete.

**Priority Issues** (must fix):
- Chunks not appearing when file expanded
- JavaScript errors in console
- Click handler not working
- Wrong colors for node types

**Nice to Have** (can defer):
- Performance optimization for very large trees
- Custom chunk label formatting
- Configurable chunk colors
- Animation for expand/collapse

---

**Test Date**: _______________
**Tester**: _______________
**Status**: ⬜ Pass | ⬜ Fail | ⬜ Needs Fixes
