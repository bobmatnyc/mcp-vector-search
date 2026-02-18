# Monorepo Visualization Root Node Analysis

**Date:** 2026-02-18
**Researcher:** Claude Code Research Agent
**Objective:** Investigate how monorepos are currently handled in visualization system and identify changes needed to ensure monorepo directory is the root node

---

## Executive Summary

The mcp-vector-search system has **comprehensive monorepo detection infrastructure** but the **visualization layer creates virtual root nodes** instead of using the monorepo directory as the natural root. The tree, treemap, and sunburst visualizations all build hierarchies from `treeData`, which is constructed in `buildTreeStructure()` with the following logic:

**Current Behavior:**
- Single root node → Use that node as root
- Multiple root nodes → Create virtual "Project Root" node as parent
- **Subproject nodes exist but are treated as siblings**, not as children of the monorepo root

**Desired Behavior:**
- Monorepo detected → Use monorepo directory as root node
- All subprojects → Children of monorepo root
- Single project → Use project directory as root

---

## Key Findings

### 1. Monorepo Detection (✅ Fully Implemented)

**Location:** `src/mcp_vector_search/utils/monorepo.py`

The `MonorepoDetector` class provides robust detection:

```python
class Subproject(NamedTuple):
    name: str              # "ewtn-plus-foundation"
    path: Path             # Absolute path to subproject
    relative_path: str     # Relative to monorepo root
```

**Detection Methods (priority order):**
1. NPM Workspaces (`package.json` with `workspaces`)
2. Lerna Packages (`lerna.json`)
3. PNPM Workspaces (`pnpm-workspace.yaml`)
4. Nx Workspace (`nx.json`, scans `apps/`, `libs/`, `packages/`)
5. Fallback: Multiple `package.json` files

**Integration Points:**
- `core/indexer.py` - Creates detector during indexing
- `core/chunk_processor.py` - Assigns subproject metadata to chunks
- `cli/commands/visualize/graph_builder.py` - Creates subproject nodes for visualization

### 2. Subproject Nodes in Visualization (✅ Implemented)

**Location:** `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

The `build_graph_data()` function creates `subproject` type nodes:

```python
# Lines 269-286
if subprojects:
    console.print(f"[cyan]Detected monorepo with {len(subprojects)} subprojects[/cyan]")
    for sp_name, sp_data in subprojects.items():
        node = {
            "id": f"subproject_{sp_name}",
            "name": sp_name,
            "type": "subproject",
            "file_path": sp_data["path"] or "",
            "start_line": 0,
            "end_line": 0,
            "complexity": 0,
            "color": sp_data["color"],
            "depth": 0,  # ⚠️ ISSUE: Subprojects marked as depth 0
        }
        nodes.append(node)
```

**Link Types Created:**
- `subproject_containment` - Subproject → Directories (line 473-485)
- `dir_containment` - Directory → Subdirectories/Files (line 381-389)
- `file_containment` - File → Chunks (line 503-511)

### 3. Root Node Determination (⚠️ Issue Here)

**Location:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

The `buildTreeStructure()` function determines root nodes:

```javascript
// Lines 533-580
const rootNodes = treeNodes
    .filter(node => !parentMap.has(node.id))  // Find nodes with no parent
    .filter(node => !chunkTypes.includes(node.type))  // Exclude chunks
    .map(node => nodeMap.get(node.id))
    .filter(node => node !== undefined);

// Create virtual root if multiple roots
if (rootNodes.length === 0) {
    treeData = {name: 'Empty', id: 'root', type: 'directory', children: []};
} else if (rootNodes.length === 1) {
    treeData = rootNodes[0];  // ✅ Single root: use it directly
} else {
    // ⚠️ ISSUE: Multiple roots create virtual "Project Root"
    treeData = {
        name: 'Project Root',
        id: 'virtual-root',
        type: 'directory',
        children: rootNodes  // Subprojects become siblings under virtual root
    };
}
```

**Problem:** When monorepo is detected, subproject nodes are created with `depth: 0`, making them root nodes. The visualization then creates a **virtual "Project Root"** node as their parent, instead of using the monorepo directory itself.

### 4. Hierarchy Building for Treemap/Sunburst (Uses Same treeData)

**Location:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

Both treemap and sunburst visualizations use `buildFileHierarchy()` which processes `treeData`:

```javascript
// Line 4870-4919
function buildFileHierarchy(maxDepth = 3, includeAllForNodeId = null) {
    function processNode(node, currentDepth = 0) {
        const result = {
            name: node.name,
            id: node.id,
            type: node.type,
            // ... other properties
            depth: currentDepth
        };
        // Recursively process children
        if (shouldIncludeChildren && children.length > 0) {
            result.children = children.map(child => processNode(child, currentDepth + 1));
        }
        return result;
    }

    return processNode(treeData, 0);  // ⚠️ Uses treeData from buildTreeStructure
}
```

**AST Hierarchy (alternative grouping):**

```javascript
// Line 4923-5016
function buildASTHierarchy() {
    // Groups chunks by: Language → Type (function/class/method)
    // Creates hierarchy: Root → Languages → Types → Chunks
    // Does NOT use monorepo structure
}
```

### 5. Progressive Loading (/api/graph-initial)

**Location:** `src/mcp_vector_search/cli/commands/visualize/server.py`

The `/api/graph-initial` endpoint includes subproject nodes:

```python
# Lines 189-193
# 1. Include subproject nodes (monorepo roots)
for node in all_nodes:
    if node.get("type") == "subproject":
        initial_nodes.append(node)
        initial_node_ids.add(node["id"])
```

**Current Flow:**
1. Server sends subproject nodes with `depth: 0`
2. Client builds tree structure with `buildTreeStructure()`
3. Multiple depth-0 nodes trigger virtual root creation
4. Monorepo directory is never represented as a node

---

## Root Cause Analysis

### Why Virtual Root is Created

**Sequence of Events:**

1. **Graph Building** (`graph_builder.py` line 284):
   ```python
   "depth": 0,  # Subprojects marked as root-level
   ```

2. **Link Creation** (`graph_builder.py` lines 473-485):
   ```python
   # Link directories to subprojects (flat structure)
   if subprojects:
       for dir_path_str, dir_node in dir_nodes.items():
           for sp_name, sp_data in subprojects.items():
               if dir_path_str.startswith(sp_data.get("path", "")):
                   links.append({
                       "source": f"subproject_{sp_name}",  # Subproject is parent
                       "target": dir_node["id"],           # Directory is child
                       "type": "dir_containment",
                   })
   ```

3. **Root Detection** (`scripts.py` lines 536-538):
   ```javascript
   const rootNodes = treeNodes
       .filter(node => !parentMap.has(node.id))  // No parent = root
   ```

   **Result:** Multiple subproject nodes have no parent → all are roots

4. **Virtual Root Creation** (`scripts.py` lines 574-579):
   ```javascript
   treeData = {
       name: 'Project Root',
       id: 'virtual-root',
       type: 'directory',
       children: rootNodes  // All subprojects as siblings
   };
   ```

### Missing Component: Monorepo Root Node

**Problem:** The system creates `subproject` nodes but never creates a **monorepo root node** to serve as their parent.

**Expected Structure:**

```
Monorepo Root (e.g., "my-monorepo")
├── Subproject: packages/api
│   ├── src/
│   └── tests/
├── Subproject: packages/web
│   ├── src/
│   └── tests/
└── Subproject: packages/shared
    ├── utils/
    └── types/
```

**Current Structure:**

```
Virtual "Project Root"  ← Artificial node
├── Subproject: packages/api
├── Subproject: packages/web
└── Subproject: packages/shared
```

---

## Recommended Implementation

### Solution Overview

**Create a monorepo root directory node** when subprojects are detected, and link all subproject nodes to it.

### Changes Required

#### 1. Graph Builder (`graph_builder.py`)

**Location:** `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

**Modify `build_graph_data()` function:**

```python
# After line 268 (before creating subproject nodes)
async def build_graph_data(chunks, database, project_manager, code_only=False):
    # ... existing code ...

    # Add monorepo root node BEFORE subproject nodes
    if subprojects:
        console.print(f"[cyan]Detected monorepo with {len(subprojects)} subprojects[/cyan]")

        # 🆕 CREATE MONOREPO ROOT NODE
        monorepo_root_node = {
            "id": "monorepo_root",
            "name": project_manager.project_root.name,  # Use actual directory name
            "type": "monorepo",  # New type to distinguish from regular directories
            "file_path": str(project_manager.project_root),
            "start_line": 0,
            "end_line": 0,
            "complexity": 0,
            "color": "#6e7681",  # Neutral gray
            "depth": 0,
            "is_monorepo_root": True,  # Flag for special handling
        }
        nodes.append(monorepo_root_node)

        # Create subproject nodes as children of monorepo root
        for sp_name, sp_data in subprojects.items():
            node = {
                "id": f"subproject_{sp_name}",
                "name": sp_name,
                "type": "subproject",
                "file_path": sp_data["path"] or "",
                "start_line": 0,
                "end_line": 0,
                "complexity": 0,
                "color": sp_data["color"],
                "depth": 1,  # 🔧 CHANGED: Now depth 1 (child of monorepo root)
            }
            nodes.append(node)
```

**Add links from monorepo root to subprojects:**

```python
# After line 286 (after creating subproject nodes)
    # 🆕 LINK MONOREPO ROOT TO SUBPROJECTS
    if subprojects:
        for sp_name in subprojects.keys():
            links.append({
                "source": "monorepo_root",
                "target": f"subproject_{sp_name}",
                "type": "monorepo_containment",  # New link type
            })
```

#### 2. Progressive Loading Endpoint (`server.py`)

**Location:** `src/mcp_vector_search/cli/commands/visualize/server.py`

**Modify `/api/graph-initial` endpoint:**

```python
# After line 188 (in initial nodes filtering)
@app.get("/api/graph-initial")
async def get_graph_initial() -> Response:
    # ... existing code ...

    initial_nodes = []
    initial_node_ids = set()

    # 🆕 1. Include monorepo root node FIRST (if exists)
    for node in all_nodes:
        if node.get("type") == "monorepo":
            initial_nodes.append(node)
            initial_node_ids.add(node["id"])

    # 2. Include subproject nodes (monorepo roots)
    for node in all_nodes:
        if node.get("type") == "subproject":
            initial_nodes.append(node)
            initial_node_ids.add(node["id"])

    # ... rest of existing code ...
```

#### 3. Tree Structure Building (`scripts.py`)

**Location:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**No changes needed!** The existing logic will automatically use the monorepo root as the single root node:

```javascript
// Lines 567-580 (existing logic will work correctly)
if (rootNodes.length === 0) {
    treeData = {name: 'Empty', id: 'root', type: 'directory', children: []};
} else if (rootNodes.length === 1) {
    treeData = rootNodes[0];  // ✅ Will use monorepo root node
} else {
    // Will only trigger if monorepo root is missing
    treeData = {
        name: 'Project Root',
        id: 'virtual-root',
        type: 'directory',
        children: rootNodes
    };
}
```

**Optional Enhancement:** Update root node type filtering to include monorepo:

```javascript
// Line 536-538 (optional improvement)
const rootNodes = treeNodes
    .filter(node => !parentMap.has(node.id))
    .filter(node => !chunkTypes.includes(node.type))
    // 🆕 Prioritize monorepo root over other roots
    .sort((a, b) => {
        if (a.type === 'monorepo') return -1;  // Monorepo root first
        if (b.type === 'monorepo') return 1;
        return 0;
    })
    .map(node => nodeMap.get(node.id))
    .filter(node => node !== undefined);
```

#### 4. Styling and Rendering

**Location:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Add monorepo type handling to node color/size functions:**

```javascript
// Around line 5051 (getNodeColor function)
function getNodeColor(d) {
    // ... existing code ...

    const typeColors = {
        'monorepo': '#8957e5',      // 🆕 Purple for monorepo root
        'subproject': '#1f6feb',    // 🆕 Blue for subprojects
        'directory': '#79c0ff',
        'file': '#58a6ff',
        'language': '#8957e5',
        // ... rest of existing colors
    };

    return typeColors[d.data.type] || '#6e7681';
}
```

**Update tree visualization rendering:**

```javascript
// In renderTree function (around line 850)
function renderTree() {
    // ... existing code ...

    nodes.append('circle')
        .attr('class', 'node-circle')
        .attr('r', d => {
            if (d.data.type === 'monorepo') return 12;      // 🆕 Larger for monorepo
            if (d.data.type === 'subproject') return 10;    // 🆕 Medium for subprojects
            // ... existing sizing logic
        })
        .attr('fill', d => {
            if (d.data.type === 'monorepo') return '#8957e5';     // 🆕 Purple
            if (d.data.type === 'subproject') return '#1f6feb';   // 🆕 Blue
            // ... existing coloring logic
        });
}
```

---

## Implementation Checklist

### Phase 1: Core Changes (Required)

- [ ] **Modify `graph_builder.py`:**
  - [ ] Create monorepo root node when subprojects detected
  - [ ] Set subproject nodes to `depth: 1`
  - [ ] Add `monorepo_containment` links from root to subprojects

- [ ] **Update `server.py`:**
  - [ ] Include monorepo root node in `/api/graph-initial`
  - [ ] Ensure monorepo root is sent before subproject nodes

- [ ] **Test tree visualization:**
  - [ ] Verify monorepo root appears as single root
  - [ ] Verify subprojects appear as children
  - [ ] Verify existing projects (non-monorepo) still work

### Phase 2: Visual Enhancements (Recommended)

- [ ] **Update `scripts.py` colors:**
  - [ ] Add `monorepo` type color (purple `#8957e5`)
  - [ ] Add `subproject` type color (blue `#1f6feb`)

- [ ] **Update node sizing:**
  - [ ] Monorepo root: radius 12px
  - [ ] Subprojects: radius 10px

- [ ] **Add breadcrumb handling:**
  - [ ] Show monorepo name in breadcrumb when navigating
  - [ ] "Home" button returns to monorepo root (not virtual root)

### Phase 3: Edge Cases (Good to Have)

- [ ] **Handle hybrid structures:**
  - [ ] Monorepo with some files at root level
  - [ ] Nested monorepos (monorepo containing monorepos)

- [ ] **Add metadata:**
  - [ ] Count of subprojects in monorepo root node
  - [ ] Tooltip showing monorepo configuration (workspace.json, lerna.json, etc.)

- [ ] **Filtering/Navigation:**
  - [ ] Filter by subproject in tree view
  - [ ] Highlight subproject boundaries in treemap/sunburst

---

## Testing Strategy

### Test Cases

#### 1. Monorepo with Multiple Subprojects

**Setup:**
```
my-monorepo/
├── package.json (with workspaces: ["packages/*"])
├── packages/
│   ├── api/
│   │   ├── src/
│   │   └── package.json
│   ├── web/
│   │   ├── src/
│   │   └── package.json
│   └── shared/
│       ├── utils/
│       └── package.json
```

**Expected Visualization:**
```
my-monorepo (depth 0, type: monorepo)
├── api (depth 1, type: subproject)
│   └── src (depth 2, type: directory)
├── web (depth 1, type: subproject)
│   └── src (depth 2, type: directory)
└── shared (depth 1, type: subproject)
    └── utils (depth 2, type: directory)
```

**Verify:**
- ✅ Single root node named "my-monorepo"
- ✅ Three subproject nodes as direct children
- ✅ No virtual "Project Root" node
- ✅ Breadcrumb shows monorepo name at root
- ✅ Tree, treemap, and sunburst all use same hierarchy

#### 2. Single Project (Non-Monorepo)

**Setup:**
```
simple-project/
├── src/
├── tests/
└── package.json (no workspaces)
```

**Expected Visualization:**
```
simple-project (depth 0, type: directory)
├── src (depth 1, type: directory)
└── tests (depth 1, type: directory)
```

**Verify:**
- ✅ Single root node named "simple-project"
- ✅ No subproject nodes
- ✅ No monorepo node
- ✅ Behavior unchanged from current implementation

#### 3. Lerna Monorepo

**Setup:**
```
lerna-monorepo/
├── lerna.json (with packages: ["packages/*"])
├── packages/
│   ├── app-a/
│   └── app-b/
```

**Expected Visualization:**
```
lerna-monorepo (depth 0, type: monorepo)
├── app-a (depth 1, type: subproject)
└── app-b (depth 1, type: subproject)
```

**Verify:**
- ✅ Detects lerna monorepo configuration
- ✅ Creates monorepo root node
- ✅ Subprojects linked correctly

#### 4. Nx Workspace

**Setup:**
```
nx-workspace/
├── nx.json
├── apps/
│   ├── frontend/
│   └── backend/
├── libs/
│   └── shared/
```

**Expected Visualization:**
```
nx-workspace (depth 0, type: monorepo)
├── frontend (depth 1, type: subproject)
├── backend (depth 1, type: subproject)
└── shared (depth 1, type: subproject)
```

**Verify:**
- ✅ Detects Nx workspace configuration
- ✅ Includes apps and libs as subprojects

---

## File Locations Summary

### Files to Modify

| File | Purpose | Changes |
|------|---------|---------|
| `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` | Graph data construction | Add monorepo root node creation, update subproject depth, add containment links |
| `src/mcp_vector_search/cli/commands/visualize/server.py` | API endpoints | Include monorepo root in initial graph load |
| `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` | D3 visualization rendering | Add monorepo type colors/sizing (optional) |

### Files to Test

| File | Test Type | Verification |
|------|-----------|--------------|
| `tests/unit/test_monorepo.py` | Unit tests | Monorepo detection still works |
| `tests/manual/test_root_breadcrumb_reset.py` | Integration | Breadcrumb navigation with monorepo root |
| New: `tests/manual/test_monorepo_visualization.py` | Integration | End-to-end monorepo visualization test |

---

## Alternative Approaches Considered

### Option A: Use Project Directory as Root (Current Recommendation)

**Pros:**
- Natural hierarchy matches file system
- No artificial nodes
- Minimal code changes

**Cons:**
- Requires creating new node type (`monorepo`)
- Need to update link structure

### Option B: Keep Virtual Root but Rename It

**Pros:**
- No structural changes
- Quick fix

**Cons:**
- Still artificial node
- Doesn't match file system
- User confusion (what is "Project Root"?)

### Option C: Make Subprojects Top-Level Tabs

**Pros:**
- Clear subproject isolation
- Better for large monorepos

**Cons:**
- Major UI change
- Loses cross-project navigation
- Complex implementation

**Verdict:** Option A provides the best user experience with reasonable implementation complexity.

---

## Related Documentation

- **Monorepo Detection:** `docs/development/monorepo-detection-fix.md`
- **Visualization Architecture:** `docs/development/VISUALIZATION_ARCHITECTURE_V2.md`
- **Multi-Root Analysis:** `docs/research/multi-root-visualization-architecture-2025-02-16.md`
- **Directory Index:** `docs/guides/indexing.md`

---

## Conclusion

### Current State

- ✅ Monorepo detection fully implemented
- ✅ Subproject tracking in chunks and graph
- ✅ Subproject nodes created in visualization
- ⚠️ Virtual "Project Root" used instead of monorepo directory
- ⚠️ All visualizations (tree, treemap, sunburst) affected

### Recommended Changes

**3 files to modify:**
1. `graph_builder.py` - Add monorepo root node and links
2. `server.py` - Include monorepo root in initial load
3. `scripts.py` - (Optional) Add monorepo type styling

**Estimated effort:** 2-4 hours implementation + 1-2 hours testing

**Impact:**
- Tree visualization: Monorepo directory as root node
- Treemap visualization: Monorepo directory as top-level partition
- Sunburst visualization: Monorepo directory at center
- Breadcrumb navigation: Shows monorepo name at root level
- No breaking changes to existing single-project visualizations

### Next Steps

1. Implement monorepo root node creation in `graph_builder.py`
2. Update `/api/graph-initial` endpoint in `server.py`
3. Test with sample monorepos (npm workspaces, lerna, pnpm, nx)
4. (Optional) Add visual enhancements for monorepo/subproject types
5. Update documentation with monorepo visualization screenshots

---

## Appendix: Code Snippets

### Current Subproject Node Creation

```python
# graph_builder.py lines 269-286
if subprojects:
    for sp_name, sp_data in subprojects.items():
        node = {
            "id": f"subproject_{sp_name}",
            "name": sp_name,
            "type": "subproject",
            "file_path": sp_data["path"] or "",
            "depth": 0,  # ⚠️ Makes it a root node
            "color": sp_data["color"],
        }
        nodes.append(node)
```

### Current Root Node Detection

```javascript
// scripts.py lines 533-580
const rootNodes = treeNodes
    .filter(node => !parentMap.has(node.id))  // No parent = root
    .filter(node => !chunkTypes.includes(node.type));

if (rootNodes.length === 1) {
    treeData = rootNodes[0];
} else {
    treeData = {
        name: 'Project Root',  // ⚠️ Virtual root
        id: 'virtual-root',
        type: 'directory',
        children: rootNodes
    };
}
```

### Proposed Monorepo Root Creation

```python
# NEW: Add to graph_builder.py after line 268
if subprojects:
    # Create monorepo root node
    monorepo_root_node = {
        "id": "monorepo_root",
        "name": project_manager.project_root.name,
        "type": "monorepo",
        "file_path": str(project_manager.project_root),
        "depth": 0,
        "color": "#8957e5",
        "is_monorepo_root": True,
    }
    nodes.append(monorepo_root_node)

    # Create subproject nodes as children (depth 1)
    for sp_name, sp_data in subprojects.items():
        node = {
            "id": f"subproject_{sp_name}",
            "name": sp_name,
            "type": "subproject",
            "depth": 1,  # 🔧 Changed from 0
            "color": sp_data["color"],
        }
        nodes.append(node)

    # Link monorepo root to subprojects
    for sp_name in subprojects.keys():
        links.append({
            "source": "monorepo_root",
            "target": f"subproject_{sp_name}",
            "type": "monorepo_containment",
        })
```
