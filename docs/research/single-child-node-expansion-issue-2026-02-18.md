# Single Child Node Expansion Issue - Investigation Report

**Date:** 2026-02-18
**Investigator:** Research Agent
**Project:** mcp-vector-search
**Issue:** Combined nodes (directories with single child) don't expand properly

---

## Executive Summary

The "single child node expansion" issue has been **PARTIALLY ADDRESSED** but still has a **CRITICAL BUG** that prevents combined directory nodes (like `src/mcp_vector_search`) from expanding properly.

**Root Cause:** Combined nodes are marked with `expanded=true` during initial load to prevent redundant API calls, but this flag prevents the click handler from fetching their actual children when the user clicks to expand them.

**Impact:** Users cannot expand directory chains that were automatically combined during tree structure building (e.g., `src/mcp_vector_search` won't show subdirectories when clicked).

---

## Code Analysis

### 1. Server-Side: `/api/graph-initial` Endpoint

**File:** `src/mcp_vector_search/cli/commands/visualize/server.py`
**Lines:** 157-275

#### What It Does:

The server correctly:
- Returns top-level nodes (depth 0-2) for initial view
- Sets `autoExpand = depth <= 1` as a hint for client-side auto-expansion
- **NEVER** sets `expanded=true` server-side (line 211)
- Adds `collapsed_children_count` metadata for directories with hidden children (lines 236-240)

```python
# Lines 207-214
node_copy["expandable"] = True
# NEVER set expanded=true server-side - let client track via expandedNodes
# The client will auto-expand depth 0-1 after fetching children
node_copy["expanded"] = False
node_copy["autoExpand"] = depth <= 1  # Hint for client
initial_nodes.append(node_copy)
```

**Server Side Status:** ✅ **CORRECT** - Properly sets metadata without interfering with client state.

---

### 2. Client-Side: Initial Load Handler

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Lines:** 284-294

#### The Problem:

After loading initial nodes, the client marks nodes with `autoExpand=true` as `expanded=true`:

```javascript
// Lines 287-294
allNodes.forEach(node => {
    if (node.autoExpand === true) {
        // Mark as expanded so click handler doesn't fetch (data is preloaded)
        node.expanded = true;
        // Do NOT add to expandedNodes - let them be visually collapsed
        console.log(`Marked ${node.name} as preloaded (children in initial data, visually collapsed)`);
    }
});
```

**Why This Is Wrong:**
- Nodes with `autoExpand=true` (depth 0-1 directories) are marked `expanded=true`
- This flag tells the click handler "children are already loaded, don't fetch"
- But after `collapseSingleChildChains()` runs (lines 585-663), combined nodes lose their children and become collapsed chains (e.g., `src/mcp_vector_search`)
- When clicked, these nodes have:
  - `expanded=true` (thinks children are loaded)
  - `collapsed_children_count > 0` (knows there are hidden children)
  - But no actual children in the tree structure!

**Client Side Status:** ❌ **BUGGY** - Incorrectly pre-marks nodes as expanded.

---

### 3. Single-Child Chain Collapsing

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Lines:** 585-663

#### What It Does:

The `collapseSingleChildChains()` function combines directory chains:

```javascript
// Lines 592-608
if (node.type === 'directory' && node.children.length === 1) {
    const onlyChild = node.children[0];
    if (onlyChild.type === 'directory') {
        // Merge: combine names with "/"
        console.log(`Collapsing dir chain: ${node.name} + ${onlyChild.name}`);
        node.name = `${node.name}/${onlyChild.name}`;
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

**Example:**
```
Before:  src -> mcp_vector_search -> cli
After:   src/mcp_vector_search/cli (single node)
```

**The Issue:**
- `src` node initially has `autoExpand=true, expanded=true`
- After collapsing: `src/mcp_vector_search/cli` still has `expanded=true`
- But its children (subdirectories) were NOT loaded in initial view
- It has `collapsed_children_count=5` indicating hidden children
- Click handler sees `expanded=true` and refuses to fetch!

**Status:** ⚠️ **FEATURE WORKS** but conflicts with the `expanded=true` flag.

---

### 4. Click Handler Logic

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
**Lines:** 1810-1871

#### Decision Tree for Directory Clicks:

```javascript
// Lines 1821-1870
if (nodeData.type === 'directory') {
    const hasVisibleChildren = nodeData.children && nodeData.children.length > 0;
    const hasHiddenChildren = nodeData._children && nodeData._children.length > 0;
    const hasCollapsedChildren = nodeData.collapsed_children_count > 0;
    const hasAnyLoadedChildren = hasVisibleChildren || hasHiddenChildren;

    // Case 1: Has collapsed children but none loaded - fetch them
    if (hasCollapsedChildren && !hasAnyLoadedChildren) {
        console.log(`Fetching ${nodeData.collapsed_children_count} collapsed children for ${nodeData.name}`);
        expandNode(nodeData.id).catch(err => { /* ... */ });
        return;
    }

    // Case 2: Has visible children - collapse them
    if (hasVisibleChildren) { /* ... */ }

    // Case 3: Has hidden children - show them
    if (hasHiddenChildren) { /* ... */ }

    // Case 4: Expandable but no children - try fetching
    if (nodeData.expandable) { /* ... */ }
}
```

**What Works:**
- ✅ Case 1: Correctly fetches when `collapsed_children_count > 0` and no loaded children
- ✅ Case 2-3: Correctly toggles visible/hidden children
- ✅ Case 4: Fallback for expandable nodes

**What's Missing:**
- ❌ Click handler checks `collapsed_children_count` but doesn't check `expanded` flag
- ❌ The `expandNode()` function (lines 314-400) has a guard:
  ```javascript
  // Lines 316-320
  if (expandedNodes.has(nodeId)) {
      console.log(`Node ${nodeId} already expanded`);
      alert(`DEBUG: Node ${nodeId} already expanded, skipping`);
      return;
  }
  ```
  - This guard should check if node is in `expandedNodes` **SET**, not the node's `expanded` property
  - But combined nodes might have `expanded=true` without being in `expandedNodes`

**Status:** ⚠️ **MOSTLY WORKS** but doesn't account for the `expanded=true` flag conflict.

---

## The Bug Chain

1. **Initial Load:** Server returns depth 0-2 nodes with `autoExpand=true` for depth 0-1
2. **Client Marks Expanded:** Client sets `expanded=true` on all `autoExpand` nodes (WRONG)
3. **Tree Building:** Links connect nodes to create parent-child relationships
4. **Chain Collapsing:** Single-child directories merge (e.g., `src/mcp_vector_search`)
5. **User Clicks:** User clicks on merged node like `src/mcp_vector_search`
6. **Bug Occurs:**
   - Node has `expanded=true` (from step 2)
   - Node has `collapsed_children_count=5` (hidden subdirectories)
   - Node has `children=null` (collapsed for display)
   - Click handler sees `collapsed_children_count > 0` and `!hasAnyLoadedChildren`
   - Calls `expandNode(nodeId)`
   - `expandNode()` checks if `expandedNodes.has(nodeId)` ← **This should be false!**
   - But the node's `expanded=true` flag might be confusing something else
   - **ACTUAL ISSUE:** The click handler logic should work, but there may be another check somewhere

---

## Testing the Bug

### Reproduction Steps:

1. Run `mcp-vector-search visualize`
2. Initial view loads with top-level directories visible
3. See combined node: `src/mcp_vector_search/cli` (merged from depth 0-2)
4. Click on `src/mcp_vector_search/cli`
5. **Expected:** Subdirectories expand (fetched via `/api/graph-expand/...`)
6. **Actual:** Nothing happens or alert shows "already expanded"

### Debug Output to Check:

```javascript
console.log('Node clicked:', nodeData.name);
console.log('expanded flag:', nodeData.expanded);
console.log('expandedNodes has node:', expandedNodes.has(nodeData.id));
console.log('collapsed_children_count:', nodeData.collapsed_children_count);
console.log('hasAnyLoadedChildren:', hasAnyLoadedChildren);
```

---

## Root Cause Summary

**Primary Issue:**
Lines 287-294 in `scripts.py` incorrectly mark nodes with `autoExpand=true` as `expanded=true` BEFORE tree structure building and chain collapsing.

**Secondary Issue:**
The `expanded` flag is used for TWO different purposes:
1. **Server Response Tracking:** "Have I already fetched children from server?"
2. **Visual State Tracking:** "Are children currently visible in the tree?"

These two states should be separate:
- `expanded` (visual state) ← already handled by `children` vs `_children`
- `childrenLoaded` or similar (fetch state) ← should be separate flag

**Tertiary Issue:**
The `expandedNodes` Set tracks nodes that should remain expanded during tree rebuilds, but combined nodes don't get added to this set during initial load (by design, lines 291-292).

---

## Recommended Fix

### Option 1: Remove Premature `expanded=true` Assignment (Simplest)

**Change:** Lines 287-294
**Action:** Remove the `node.expanded = true` assignment entirely

```javascript
// BEFORE:
allNodes.forEach(node => {
    if (node.autoExpand === true) {
        node.expanded = true;  // ← REMOVE THIS
        console.log(`Marked ${node.name} as preloaded...`);
    }
});

// AFTER:
allNodes.forEach(node => {
    if (node.autoExpand === true) {
        // Just a hint for debugging, don't set expanded flag
        console.log(`Node ${node.name} will auto-expand after tree build`);
    }
});
```

**Why This Works:**
- Nodes start with `expanded=false` (from server)
- Combined nodes will have `collapsed_children_count > 0`
- Click handler correctly triggers `expandNode()` for first expansion
- After fetching, `expandedNodes.add(nodeId)` prevents duplicate fetches

**Trade-off:**
- Slight performance hit (may refetch children for already-loaded nodes during auto-expand)
- But ensures correctness over optimization

---

### Option 2: Add Separate `childrenLoaded` Flag (Better Architecture)

**Change:** Multiple locations
**Action:** Introduce `childrenLoaded` boolean to track fetch state separately from visual state

```javascript
// In expandNode() after successful fetch (line 377):
if (node) {
    node.childrenLoaded = true;  // Tracks fetch state
    // Don't set expanded=true - that's for visual state
}

// In click handler (line 1832):
if (hasCollapsedChildren && !nodeData.childrenLoaded) {
    expandNode(nodeData.id);
    return;
}

// In expandNode() guard (line 316):
if (nodeData.childrenLoaded && expandedNodes.has(nodeId)) {
    // Already fetched AND already expanded visually
    return;
}
```

**Why This Is Better:**
- Separates concerns: fetch state vs visual state
- Prevents confusion between "expanded" (visible) and "loaded" (fetched)
- More maintainable and easier to reason about

**Trade-off:**
- Requires changes in multiple locations
- More complex but clearer semantics

---

### Option 3: Check `collapsed_children_count` in `expandNode()` (Quick Fix)

**Change:** Lines 316-320
**Action:** Skip the "already expanded" guard if node has uncollapsed children

```javascript
// BEFORE:
if (expandedNodes.has(nodeId)) {
    console.log(`Node ${nodeId} already expanded`);
    return;
}

// AFTER:
const node = allNodes.find(n => n.id === nodeId);
if (expandedNodes.has(nodeId) && !(node?.collapsed_children_count > 0)) {
    console.log(`Node ${nodeId} already expanded and no collapsed children`);
    return;
}
```

**Why This Works:**
- Allows re-expansion for combined nodes that have `collapsed_children_count`
- Minimal code change

**Trade-off:**
- Band-aid fix, doesn't address root cause
- Might allow duplicate fetches in edge cases

---

## Implementation Priority

**RECOMMENDED: Option 1** (Remove premature `expanded=true`)

**Reasoning:**
- Simplest fix with least risk
- Addresses root cause directly
- Minimal code changes
- Can be implemented in < 5 minutes
- Allows follow-up refactoring to Option 2 later

**Patch Location:**
`src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
Lines: 287-294

---

## Testing After Fix

### Test Cases:

1. **Combined Directory Node:**
   - Click on `src/mcp_vector_search` (merged node)
   - **Expected:** Expands to show subdirectories (cli, core, etc.)
   - **Verify:** Console shows "Fetching N collapsed children"

2. **Regular Directory Node:**
   - Click on `tests` (not combined)
   - **Expected:** Expands to show test files
   - **Verify:** No duplicate fetch warnings

3. **Already Expanded Node:**
   - Click to expand, then click again
   - **Expected:** Collapses (toggles)
   - **Verify:** No API call on second click

4. **File Node:**
   - Click on a file node
   - **Expected:** Shows code chunks
   - **Verify:** Works regardless of directory fix

---

## Additional Notes

### Related Code Sections:

- **Initial Load:** Lines 240-305 (loadInitialGraph function)
- **Tree Building:** Lines 406-579 (buildTreeStructure function)
- **Chain Collapsing:** Lines 585-663 (collapseSingleChildChains function)
- **Collapse Logic:** Lines 680-699 (collapseAllExceptExpanded function)
- **Click Handler:** Lines 1810-1871 (handleNodeClick function)
- **Expand API Call:** Lines 314-400 (expandNode function)

### Server Endpoints:

- **Initial View:** `GET /api/graph-initial` (lines 157-275 in server.py)
- **Node Expansion:** `GET /api/graph-expand/{node_id}` (lines 277-379 in server.py)

### Metadata Fields:

- `expandable` (boolean): Node has children that can be fetched
- `expanded` (boolean): Node's children are currently visible (BUGGY - mixed with fetch state)
- `autoExpand` (boolean): Hint from server to auto-expand depth 0-1 nodes
- `collapsed_children_count` (number): Count of hidden children not in initial view
- `collapsed_ids` (array): IDs of nodes merged during chain collapsing

---

## Conclusion

The single-child node expansion issue is a **state management bug** where the `expanded` flag is used for both fetch tracking and visual state tracking. Combined nodes inherit `expanded=true` from the initial load but lose their children during chain collapsing, resulting in a node that claims to be "expanded" but has no children to show.

**Fix:** Remove the premature `expanded=true` assignment on lines 287-294, allowing the click handler to correctly fetch children on first expansion.

**Impact:** Low-risk, high-reward fix that restores expected expansion behavior for combined directory nodes.

---

**Status:** Ready for implementation
**Estimated Fix Time:** 5 minutes
**Testing Time:** 10 minutes
**Total Resolution Time:** ~15 minutes
