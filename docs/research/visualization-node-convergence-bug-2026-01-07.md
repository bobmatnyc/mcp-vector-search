# Visualization Bug: Nodes Converging to Root

**Investigation Date**: 2026-01-07
**Bug Severity**: High
**Impact**: Visual hierarchy completely broken in D3.js visualization

## Problem Summary

All nodes in the D3.js visualization are funneling/converging toward the "Project Root" node in a strange pattern, rather than maintaining proper hierarchical positioning with directories containing their children. The visual appearance is that containment links are not working correctly.

## Root Cause Analysis

### Primary Issue: Missing Link Types in Frontend Filter

**Location**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html` (lines 2806-2824)

The frontend code has a filter that processes hierarchical relationships, but it's **missing two critical link types** that the backend creates:

```javascript
// Process all relationship links
allLinks.forEach(link => {
    const linkType = link.type;

    // Determine relationship category
    let category = null;
    if (linkType === 'dir_hierarchy') {
        category = 'dir_hierarchy';
    } else if (linkType === 'dir_containment') {
        category = 'dir_containment';
    } else if (linkType === 'file_containment') {
        category = 'file_containment';
    } else if (linkType === 'chunk_hierarchy') {
        category = 'chunk_hierarchy';
    } else {
        // Skip semantic, caller, undefined, and other non-hierarchical links
        // This includes links without a 'type' field (e.g., subproject links)
        return;  // ← BUG: Missing link types are dropped here!
    }
```

### Missing Link Types

The backend (`graph_builder.py`) creates these link types that the frontend **does not recognize**:

1. **`subproject_containment`** (line 520 in `graph_builder.py`)
   - Created for monorepo projects
   - Links subproject root nodes to top-level chunks
   - Example: `subproject_auth` → `chunk_12345`

2. **`dependency`** (line 156 in `graph_builder.py`)
   - Created by `parse_project_dependencies()`
   - Links subprojects to each other based on package.json dependencies
   - Example: `subproject_frontend` → `subproject_backend`

### Consequence

When these link types are encountered:
1. They hit the `else` clause at line 2820
2. The code executes `return;` which **skips the link entirely**
3. Parent-child relationships are **never established** in the tree structure
4. Nodes with only these link types become **orphaned**
5. D3's force simulation has no containment constraints for these nodes
6. All orphaned nodes converge toward the root due to gravity forces

## Evidence from Code

### Backend Link Creation

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

**Subproject Containment Links** (lines 514-522):
```python
# Link to subproject root if in monorepo
if chunk.subproject_name and not chunk.parent_chunk_id:
    links.append(
        {
            "source": f"subproject_{chunk.subproject_name}",
            "target": chunk_id,
            "type": "subproject_containment",  # ← Created here
        }
    )
```

**Dependency Links** (lines 150-158):
```python
# Found inter-project dependency
dependency_links.append(
    {
        "source": f"subproject_{sp_name}",
        "target": f"subproject_{other_sp_name}",
        "type": "dependency",  # ← Created here
    }
)
```

### Frontend Link Processing

**File**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html`

**Link Type Filter** (lines 2806-2824):
- Only recognizes: `dir_hierarchy`, `dir_containment`, `file_containment`, `chunk_hierarchy`
- Missing: `subproject_containment`, `dependency`
- Result: Links with unrecognized types are **silently dropped**

## Impact Assessment

### Affected Scenarios

1. **Monorepo Projects** (High Impact)
   - All subproject nodes lose containment links
   - Chunks in subprojects become orphaned
   - Inter-project dependencies not visualized

2. **Single-Repo Projects** (Medium Impact)
   - May still exhibit convergence if other undefined link types exist
   - Less severe but still broken hierarchy

### User Experience

- Visualization appears completely broken
- Cannot understand project structure
- Tree hierarchy collapses into a ball of nodes
- No visual distinction between containment levels

## Recommended Fix

### Option 1: Add Missing Link Types to Frontend Filter (Preferred)

**File**: `/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/index.html`
**Location**: Lines 2806-2824

```javascript
// Process all relationship links
allLinks.forEach(link => {
    const linkType = link.type;

    // Determine relationship category
    let category = null;
    if (linkType === 'dir_hierarchy') {
        category = 'dir_hierarchy';
    } else if (linkType === 'dir_containment') {
        category = 'dir_containment';
    } else if (linkType === 'file_containment') {
        category = 'file_containment';
    } else if (linkType === 'chunk_hierarchy') {
        category = 'chunk_hierarchy';
    } else if (linkType === 'subproject_containment') {
        category = 'subproject_containment';  // ← ADD THIS
    } else if (linkType === 'dependency') {
        category = 'dependency';  // ← ADD THIS
    } else {
        // Skip semantic, caller, and other non-hierarchical links
        return;
    }

    // ... rest of processing
});
```

**Then add counters for the new categories** (around line 2800):
```javascript
const relationshipsProcessed = {
    dir_hierarchy: 0,
    dir_containment: 0,
    file_containment: 0,
    chunk_hierarchy: 0,
    subproject_containment: 0,  // ← ADD THIS
    dependency: 0                // ← ADD THIS
};
```

### Option 2: Defensive Programming - Log Unknown Types

Instead of silently dropping unknown link types, log them for debugging:

```javascript
} else {
    // Log unknown link types for debugging
    if (!loggedUnknownTypes.has(linkType)) {
        console.warn(`Unknown link type encountered: ${linkType}`);
        loggedUnknownTypes.add(linkType);
    }
    // Skip semantic, caller, and other non-hierarchical links
    return;
}
```

### Option 3: Generic Fallback Handling

Create a generic "unknown containment" category:

```javascript
} else if (linkType && linkType.includes('containment')) {
    category = linkType;  // Generic containment handling
} else {
    // Skip semantic, caller, and truly non-hierarchical links
    return;
}
```

## Testing Plan

1. **Test with Monorepo**
   - Use a project with multiple subprojects
   - Verify subproject nodes are positioned correctly
   - Check that dependency arrows are visible

2. **Test with Single Repo**
   - Verify existing hierarchy still works
   - Check no regression in file/directory structure

3. **Visual Verification**
   - Nodes should maintain hierarchical positions
   - No funnel/convergence pattern
   - Containment boundaries clearly visible

## Related Issues

- Comment in code mentions: "undefined link types are being misclassified, causing directory containment relationships to be lost"
- This confirms the issue was previously identified but not fully resolved
- The fix likely only addressed `dir_containment` but missed `subproject_containment` and `dependency`

## Priority

**HIGH** - This completely breaks visualization for monorepo projects and degrades UX for all projects.

## Estimated Fix Time

- **Code Change**: 5 minutes (add two link type cases)
- **Testing**: 15 minutes (verify with monorepo and single repo)
- **Total**: ~20 minutes

## Next Steps

1. Implement Option 1 (add missing link types to frontend filter)
2. Add defensive logging (Option 2) as additional safety measure
3. Test with both monorepo and single-repo projects
4. Consider adding unit tests for link type processing
5. Document all supported link types in code comments
