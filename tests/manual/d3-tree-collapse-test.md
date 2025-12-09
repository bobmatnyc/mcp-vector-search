# D3 Tree Visualization - Manual Test Plan

**Test Date**: _____________________
**Tester**: _____________________
**Build**: December 9, 2025 fix

## Test Environment Setup

```bash
# Start visualization server
cd /Users/masa/Projects/mcp-vector-search
mcp-vector-search visualize
```

Expected: Browser opens to `http://localhost:8000`

---

## Test 1: Initial Tree State

**Objective**: Verify tree starts collapsed with only root level visible

### Steps:
1. Load visualization page
2. Observe initial tree structure

### Expected Results:
- [ ] **Only root-level nodes visible** (e.g., docs, src, tests, .github, etc.)
- [ ] **All directories have orange circles** (indicating collapsed state)
- [ ] **Files have gray circles**
- [ ] **No code chunks visible in tree** (no function/class names as nodes)
- [ ] **Tree is clean and readable** (not overwhelming)

### Actual Results:
```
Notes:
_________________________________________________________________
_________________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 2: Directory Expansion (Single Level)

**Objective**: Verify clicking a directory shows only its immediate children

### Steps:
1. Click on "src" directory (or any root directory)
2. Observe what appears

### Expected Results:
- [ ] **Directory circle turns blue** (expanded state)
- [ ] **Only immediate children visible** (e.g., src/mcp_vector_search/)
- [ ] **Grandchildren NOT visible** (not showing full subtree)
- [ ] **Other root directories still collapsed**
- [ ] **No code chunks appear** (still only directories/files)

### Actual Results:
```
Clicked directory: _____________
Children shown: _____________________________________________________
_________________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 3: Directory Collapse

**Objective**: Verify clicking an expanded directory collapses it

### Steps:
1. Expand a directory (from Test 2)
2. Click the same directory again
3. Observe behavior

### Expected Results:
- [ ] **Directory circle turns orange** (collapsed state)
- [ ] **Children disappear** (hidden)
- [ ] **Tree returns to previous state**
- [ ] **No visual artifacts** (clean re-render)

### Actual Results:
```
Notes:
_________________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 4: Nested Directory Expansion

**Objective**: Verify progressive disclosure through multiple levels

### Steps:
1. Expand "src" directory
2. Expand "mcp_vector_search" subdirectory
3. Expand "cli" subdirectory
4. Observe tree structure at each step

### Expected Results:
- [ ] **Each level shows only immediate children**
- [ ] **No automatic expansion of descendants**
- [ ] **Breadcrumb-like visual hierarchy** (indentation)
- [ ] **Still no code chunks in tree**

### Path expanded:
```
src/
  └─ mcp_vector_search/
       └─ cli/
            └─ [immediate children here]
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 5: File Click Shows Chunks

**Objective**: Verify code chunks appear ONLY in side panel, not in tree

### Steps:
1. Navigate to a Python file (e.g., expand path to any .py file)
2. Click on a file node
3. Observe side panel and tree

### Expected Results:
- [ ] **Side panel opens** (slides in from right)
- [ ] **Side panel shows code chunks** (functions, classes)
- [ ] **Chunks formatted as code blocks** (syntax highlighting)
- [ ] **Tree remains unchanged** (chunks NOT added to tree)
- [ ] **File node remains gray** (doesn't change color)

### File tested:
```
Path: _________________________________________________________________
Chunks shown: _________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 6: Layout Toggle Preserves State

**Objective**: Verify expansion state persists when switching layouts

### Steps:
1. Expand several directories (e.g., src, tests)
2. Click "Switch to Circular" button
3. Observe tree in circular layout
4. Click "Switch to Linear" button
5. Verify state

### Expected Results:
- [ ] **Circular layout shows same expanded nodes**
- [ ] **Colors preserved** (blue for expanded, orange for collapsed)
- [ ] **Switching back to linear maintains state**
- [ ] **No unexpected re-expansion** (state truly preserved)

### Notes:
```
Expanded before switch: _______________________________________________
State after circular: _________________________________________________
State after linear: ___________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 7: Large Directory Performance

**Objective**: Verify performance with many subdirectories

### Steps:
1. Find a directory with many children (e.g., node_modules if present, or src/)
2. Expand it
3. Measure responsiveness

### Expected Results:
- [ ] **Expansion is immediate** (< 100ms perceived delay)
- [ ] **No lag or freezing**
- [ ] **Smooth re-render**
- [ ] **Reasonable node count** (not thousands)

### Performance notes:
```
Directory tested: _____________________________________________________
Approximate child count: ______________________________________________
Perceived delay: ______________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 8: No Chunk Nodes in Tree (Critical)

**Objective**: Verify NO code chunks appear as tree nodes

### Steps:
1. Fully expand several file paths
2. Look for any function/class names in tree
3. Check browser console for "Filtered to X tree nodes" message

### Expected Results:
- [ ] **Zero function names in tree** (e.g., no "build_graph_data")
- [ ] **Zero class names in tree** (e.g., no "ChromaVectorDatabase")
- [ ] **Console shows filter message** ("Filtered to X tree nodes")
- [ ] **Node count reasonable** (< 500 for typical project)

### Console output:
```
Message: ______________________________________________________________
Node count before filter: _____________________________________________
Node count after filter: ______________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 9: Color Coding Accuracy

**Objective**: Verify visual indicators match actual state

### Test Matrix:

| Node Type | State | Expected Color | Actual Color | Pass/Fail |
|-----------|-------|----------------|--------------|-----------|
| Directory | Collapsed | 🟠 Orange (#f39c12) | | ⬜ |
| Directory | Expanded | 🔵 Blue (#3498db) | | ⬜ |
| File | Always | ⚪ Gray (#95a5a6) | | ⬜ |

### Notes:
```
_________________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Test 10: Edge Cases

**Objective**: Test unusual scenarios

### Scenario A: Empty Directory
- [ ] Expand a directory with no children
- [ ] Expected: No error, no visual change (maybe subtle indicator?)

### Scenario B: File with No Chunks
- [ ] Click a file that has no indexable chunks (e.g., .md file)
- [ ] Expected: Side panel shows "No code chunks found for this file."

### Scenario C: Rapid Clicking
- [ ] Rapidly click same directory multiple times
- [ ] Expected: Clean toggle behavior, no visual glitches

### Notes:
```
_________________________________________________________________
_________________________________________________________________
```

**Status**: ⬜ Pass / ⬜ Fail / ⬜ Partial

---

## Overall Test Summary

**Total Tests**: 10
**Passed**: _____
**Failed**: _____
**Partial**: _____

### Critical Issues Found:
```
1. _________________________________________________________________
2. _________________________________________________________________
3. _________________________________________________________________
```

### Minor Issues Found:
```
1. _________________________________________________________________
2. _________________________________________________________________
3. _________________________________________________________________
```

### Recommendations:
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

## Sign-off

**Tester Signature**: _________________________________
**Date**: _____________________________________________
**Approved for Release**: ⬜ Yes / ⬜ No / ⬜ Conditional

**Conditions for Approval** (if applicable):
```
_________________________________________________________________
_________________________________________________________________
```
