# Visualization Architecture Analysis

**Date**: 2025-12-06
**Purpose**: Comprehensive analysis of current visualization implementation to inform new interactive features
**Status**: Complete

---

## Executive Summary

The current visualization implementation is a sophisticated D3.js-based code graph system with hierarchical expansion, semantic relationships, and cycle detection. It uses a **force-directed layout** as the primary layout engine with optional Cytoscape layouts (Dagre hierarchical, circular). The system handles large codebases through progressive disclosure (collapsed nodes), streaming JSON loading, and adaptive spacing algorithms.

**Key Strengths**:
- Robust data model with full AST chunk metadata
- Multiple relationship types (containment, calls, semantic, cycles)
- Monorepo support with subproject detection
- Rich interaction model (expand/collapse, navigation stack, breadcrumbs)
- Performance optimizations (streaming JSON, adaptive spacing, edge filtering)

**Key Gaps for New Requirements**:
- No node grouping/ungrouping UI
- No saved view states (bookmarks)
- No filter persistence across sessions
- Limited layout customization (spacing is auto-calculated)
- No undo/redo for view changes

---

## 1. Current Architecture

### 1.1 Data Structures

#### **Node Types** (from `graph_builder.py` and `models.py`)

```python
# Core data model: CodeChunk (src/mcp_vector_search/core/models.py)
@dataclass
class CodeChunk:
    # Content and location
    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str

    # Classification
    chunk_type: str  # "code", "function", "class", "method", "comment", "docstring"
    function_name: str | None
    class_name: str | None
    docstring: str | None

    # Complexity and metrics
    complexity_score: float = 0.0

    # Hierarchical relationships
    chunk_id: str  # SHA256 hash of file_path:type:name:lines:content_hash
    parent_chunk_id: str | None
    child_chunk_ids: list[str]
    chunk_depth: int = 0

    # Enhanced metadata
    decorators: list[str]
    parameters: list[dict]
    return_type: str | None
    type_annotations: dict[str, str]

    # Monorepo support
    subproject_name: str | None
    subproject_path: str | None
```

**Visualization Node Structure** (JavaScript):
```javascript
// Node object structure in graph data
{
    id: string,                  // chunk_id or generated ID
    name: string,                // function_name, class_name, or "L{start_line}"
    type: string,                // "function", "class", "method", "file", "directory", "subproject"
    file_path: string,           // Absolute or relative path
    start_line: number,
    end_line: number,
    complexity: number,          // complexity_score from AST analysis

    // Hierarchical info
    parent_id: string | null,    // parent_chunk_id
    depth: number,               // chunk_depth

    // Display properties
    color: string,               // Auto-assigned based on type or subproject
    content: string,             // Full code content
    docstring: string,
    language: string,

    // Directory-specific (if type === "directory")
    dir_path: string,
    file_count: number,
    subdirectory_count: number,
    total_chunks: number,
    languages: object,
    is_package: boolean,
    last_modified: number,

    // File-specific (if type === "file")
    parent_dir_id: string,
    parent_dir_path: string,

    // Monorepo (if subproject_name exists)
    subproject: string,

    // Caller tracking (populated by external caller analysis)
    callers: [
        {
            file: string,
            chunk_id: string,
            name: string,
            type: string
        }
    ]
}
```

**Node Type Categories**:
1. **Container Nodes**: `directory`, `file`, `subproject`
2. **Code Nodes**: `function`, `class`, `method`
3. **Documentation Nodes**: `docstring`, `comment`

#### **Edge Types** (Link/Relationship Structure)

```javascript
// Link object structure
{
    source: string | object,  // node ID or node object (D3 mutates this)
    target: string | object,  // node ID or node object
    type: string,             // Relationship type (see below)

    // Optional properties based on type
    similarity: number,       // For type === "semantic" (0.0-1.0)
    is_cycle: boolean,        // True if part of circular dependency
}
```

**Relationship Types**:
1. **Containment Relationships** (hierarchical structure):
   - `dir_hierarchy`: Parent directory → Child directory
   - `dir_containment`: Directory → File OR Subproject → Directory
   - `file_containment`: File → Code chunk (top-level only)

2. **Code Relationships**:
   - `caller`: External function call (different files only)
   - `imports`: Import dependency
   - `dependency`: General dependency (used in monorepo inter-project deps)
   - `method`: Method relationship (parent-child in AST)
   - `module`: Module-level relationship

3. **Semantic Relationships**:
   - `semantic`: Similar code chunks (top 5 per chunk, threshold ≥ 0.2)

4. **Problem Indicators**:
   - `is_cycle: true`: Circular dependency detected via DFS

**Edge Filtering State** (in JavaScript):
```javascript
let edgeFilters = {
    containment: true,   // dir_hierarchy, dir_containment, file_containment
    calls: true,         // caller edges
    imports: false,      // import edges (hidden by default)
    semantic: false,     // semantic similarity edges (hidden by default)
    cycles: true         // circular dependency edges
};
```

#### **Metadata Structure**

```javascript
// Graph metadata
{
    total_chunks: number,
    total_files: number,
    languages: {
        "python": count,
        "javascript": count,
        // ...
    },
    is_monorepo: boolean,
    subprojects: string[]  // ["ewtn-plus-foundation", "other-project"]
}
```

### 1.2 Layout Algorithms

#### **Force-Directed Layout (Default)**
- **Engine**: D3.js force simulation
- **Forces**:
  ```javascript
  .force("link", d3.forceLink()
      .distance(d => {
          if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') return 40;
          if (d.is_cycle) return 80;
          if (d.type === 'semantic') return 100;
          return 60;
      })
      .strength(d => {
          if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') return 0.8;
          if (d.is_cycle) return 0.4;
          if (d.type === 'semantic') return 0.3;
          return 0.7;
      })
  )
  .force("charge", d3.forceManyBody()
      .strength(d => d.type === 'directory' ? -30 : -60)
  )
  .force("center", d3.forceCenter(width/2, height/2).strength(0.1))
  .force("radial", d3.forceRadial(100, width/2, height/2)
      .strength(d => d.type === 'directory' ? 0 : 0.1)
  )
  .force("collision", d3.forceCollide()
      .radius(d => d.type === 'directory' ? 30 : (d.type === 'file' ? 26 : 24))
      .strength(1.0)
  )
  ```

- **Initial Layout**: `positionNodesCompactly()`
  - Folders in grid layout with fixed positions (released after 1s)
  - Other nodes in spiral layout around center
  - Adaptive spacing based on graph density

#### **Adaptive Spacing Algorithm**
```javascript
function calculateAdaptiveSpacing(nodeCount, width, height, mode = 'balanced') {
    const areaPerNode = (width * height) / nodeCount;
    const baseSpacing = Math.sqrt(areaPerNode);

    const modeScales = {
        'tight': 0.4,
        'balanced': 0.6,
        'loose': 0.8
    };

    const calculatedSpacing = baseSpacing * modeScales[mode];

    // Bounds based on graph size
    if (nodeCount < 50) {
        return clamp(calculatedSpacing, 150, 400);
    } else if (nodeCount < 500) {
        return clamp(calculatedSpacing, 100, 250);
    } else {
        return clamp(calculatedSpacing, 60, 150);
    }
}
```

#### **Alternative Layouts (Cytoscape)**
- **Dagre (Hierarchical)**: Layered graph layout, top-down orientation
  - `rankDir: 'TB'`, `rankSep: 150`, `nodeSep: 80`
  - Used for large graphs (>500 nodes) by default
- **Circle**: Nodes arranged in circular pattern
  - Used for specific visual exploration scenarios

### 1.3 Directory/File Relationship Representation

**Hierarchy Construction** (from `graph_builder.py:509-547`):

1. **Directory Nodes**:
   - Loaded from `DirectoryIndex` (`.mcp-vector-search/directory_index.json`)
   - Contains metadata: `file_count`, `subdirectory_count`, `total_chunks`, `languages`, `is_package`
   - Linked via `dir_hierarchy` edges to parent directories

2. **File Nodes**:
   - Created from chunks' `file_path` attributes
   - Linked to parent directory via `dir_containment` edge
   - Parent directory ID lookup: matches relative path to directory index

3. **Chunk Nodes**:
   - Top-level chunks (no `parent_chunk_id`) linked to file via `file_containment`
   - Nested chunks linked to parent chunk via default edge (no explicit type)

4. **Monorepo Structure**:
   - Subproject root nodes (`type: "subproject"`) created for each detected subproject
   - Directories linked to subprojects via `dir_containment` if path starts with subproject path
   - Color-coded by subproject (8-color palette)

**Example Hierarchy**:
```
[subproject: my-app]
  └─[dir_containment]→ [directory: src/]
      ├─[dir_hierarchy]→ [directory: src/utils/]
      │   └─[dir_containment]→ [file: src/utils/helpers.py]
      │       └─[file_containment]→ [function: format_date]
      └─[dir_containment]→ [file: src/main.py]
          └─[file_containment]→ [function: main]
              └─[default edge]→ [method: setup]
```

### 1.4 AST Chunks and Relationships

**Chunk Extraction** (from indexer, not in visualization code):
- AST parsing via Tree-sitter (Python, JavaScript, TypeScript, Rust, Go)
- Chunk types extracted:
  - `function`: Function definitions
  - `class`: Class definitions
  - `method`: Methods within classes (parent_chunk_id = class chunk_id)
  - `docstring`: Function/class docstrings
  - `comment`: Standalone comments
  - `code`: Generic code blocks

**Relationship Discovery** (from `graph_builder.py:366-451`):

1. **External Caller Relationships**:
   ```python
   # Extract actual function calls using AST (avoids false positives)
   def extract_function_calls(code: str) -> set[str]:
       tree = ast.parse(code)
       calls = set()
       for node in ast.walk(tree):
           if isinstance(node, ast.Call):
               if isinstance(node.func, ast.Name):
                   calls.add(node.func.id)  # Direct call: foo()
               elif isinstance(node.func, ast.Attribute):
                   calls.add(node.func.attr)  # Method call: obj.foo()
       return calls
   ```
   - Only tracks **cross-file calls** (external dependencies)
   - Stored in `caller_map[chunk_id] = [caller_info]`
   - Attached to node as `node.callers` array

2. **Semantic Relationships** (from `graph_builder.py:291-364`):
   - Pre-computed for code chunks (`function`, `method`, `class`)
   - Top 5 similar chunks per chunk (using vector search, threshold ≥ 0.2)
   - Similarity score stored in edge: `{ type: "semantic", similarity: 0.7 }`

3. **Cycle Detection** (from `graph_builder.py:94-161`):
   - DFS with three-color marking (white=unvisited, gray=exploring, black=explored)
   - Detects **true cycles** in caller graph (A → B → C → A)
   - Marks edges with `is_cycle: true` for visualization
   - Used for circular dependency warnings

**Data Available for Each Chunk**:
- Full source code (`content`)
- Docstring (parsed from AST)
- Complexity score (cyclomatic complexity)
- Decorators (Python: `@property`, `@staticmethod`)
- Parameters (with type annotations)
- Return type
- Line range and language
- Parent-child relationships (hierarchical AST structure)

---

## 2. Current Interaction Model

### 2.1 Click Handlers

**Node Click** (from `scripts.py:776-808`):
```javascript
function handleNodeClick(event, d) {
    event.stopPropagation();

    // Always show content pane
    showContentPane(d);

    // If node has children, toggle expansion
    if (hasChildren(d)) {
        const wasCollapsed = collapsedNodes.has(d.id);
        if (wasCollapsed) {
            expandNode(d);  // Show direct children
        } else {
            collapseNode(d);  // Hide all descendants recursively
        }
        renderGraph();

        // Zoom behavior
        if (!wasCollapsed) {
            zoomToFit(750);  // Zoom to show all visible nodes
        } else {
            centerNode(d);  // Center on expanded node
        }
    } else {
        centerNode(d);
    }
}
```

**Behaviors**:
1. **Single click**: Show detail pane + toggle expansion (if has children)
2. **Expand**: Direct children become visible, start collapsed
3. **Collapse**: All descendants hidden recursively
4. **Auto-zoom**: Fit viewport after collapse, center node after expand

### 2.2 Expand/Collapse Logic

**State Tracking**:
```javascript
let visibleNodes = new Set();      // IDs of nodes currently rendered
let collapsedNodes = new Set();    // IDs of nodes with hidden children
let rootNodes = [];                // Top-level nodes for reset
```

**Expand Node**:
```javascript
function expandNode(node) {
    collapsedNodes.delete(node.id);

    // Find direct children via links
    const children = allLinks
        .filter(l => (l.source.id || l.source) === node.id)
        .map(l => allNodes.find(n => n.id === (l.target.id || l.target)));

    children.forEach(child => {
        visibleNodes.add(child.id);
        collapsedNodes.add(child.id);  // Children start collapsed
    });
}
```

**Collapse Node**:
```javascript
function collapseNode(node) {
    collapsedNodes.add(node.id);

    // Recursively hide all descendants
    function hideDescendants(parentId) {
        const children = allLinks
            .filter(l => (l.source.id || l.source) === parentId)
            .map(l => l.target.id || l.target);

        children.forEach(childId => {
            visibleNodes.delete(childId);
            collapsedNodes.delete(childId);
            hideDescendants(childId);  // Recursive
        });
    }

    hideDescendants(node.id);
}
```

### 2.3 State Management

**Global State** (in JavaScript):
```javascript
let allNodes = [];                 // Full graph data
let allLinks = [];                 // All edges
let visibleNodes = new Set();      // Visible node IDs
let collapsedNodes = new Set();    // Collapsed node IDs
let highlightedNode = null;        // Currently selected node
let rootNodes = [];                // Root-level nodes
let currentLayout = 'force';       // 'force', 'dagre', 'circle'
let edgeFilters = { ... };         // Edge type visibility
let cy = null;                     // Cytoscape instance (if using Dagre/Circle)
```

**Navigation Stack** (from `scripts.py:1697-1760`):
```javascript
const viewStack = {
    stack: [],             // Array of chunk IDs
    currentIndex: -1,      // Current position in stack

    push(chunkId) { ... },
    canGoBack() { ... },
    canGoForward() { ... },
    back() { ... },
    forward() { ... },
    updateButtons() { ... }
};
```
- Supports browser-style back/forward navigation
- Keyboard shortcuts: `Alt+Left` (back), `Alt+Right` (forward)
- Stack cleared on reset view

**Session Persistence**: None (all state lost on page reload)

### 2.4 Detail Pane Content

**Content Types** (from `scripts.py:1273-1283`):
1. **Directory**: List of children (subdirs, files, chunks) with click navigation
2. **File**: Code chunks section + full file content (stitched from chunks)
3. **Code Chunk**: Docstring + syntax-highlighted code with linkable references
4. **Import (depth=1)**: Import statement display

**Features**:
- **Breadcrumb navigation**: `🏠 Root / src / utils / helpers.py`
- **Code chunks list**: Clickable list of functions/classes in file
- **Linkable code**: Function/class names in code become clickable links
- **Caller tracking**: "Called By" section shows external callers with navigation
- **Syntax highlighting**: Python keywords/primitives bolded in red

**Footer Metadata**:
- Language, file path (clickable), line range, complexity score
- Docstring sections (Args, Returns, Examples)
- Caller information with navigation links

---

## 3. Data Flow

### 3.1 Codebase → Visualization Pipeline

```
1. Indexing (mcp-vector-search index)
   ├─ Tree-sitter AST parsing
   ├─ Chunk extraction (functions, classes, methods)
   ├─ Embedding generation (Sentence Transformers)
   ├─ Complexity analysis (cyclomatic complexity)
   └─ Store in ChromaDB

2. Directory Indexing (mcp-vector-search visualize)
   ├─ Scan project structure
   ├─ Build DirectoryIndex with metadata
   └─ Save to .mcp-vector-search/directory_index.json

3. Graph Building (build_graph_data in graph_builder.py)
   ├─ Load chunks from ChromaDB
   ├─ Load DirectoryIndex
   ├─ Create nodes:
   │  ├─ Subproject nodes (if monorepo)
   │  ├─ Directory nodes (from DirectoryIndex)
   │  ├─ File nodes (from chunks)
   │  └─ Chunk nodes (from ChromaDB)
   ├─ Create links:
   │  ├─ Hierarchical (dir → dir, dir → file, file → chunk)
   │  ├─ Semantic (vector similarity search)
   │  ├─ Caller (AST-based call extraction)
   │  └─ Cycles (DFS cycle detection)
   └─ Return graph_data = { nodes, links, metadata }

4. Visualization Export (mcp-vector-search visualize export)
   ├─ Generate HTML from template
   ├─ Write chunk-graph.json (graph data)
   └─ Open in browser

5. Client-Side Rendering (D3.js)
   ├─ Streaming JSON load with progress
   ├─ Initialize visibleNodes (root nodes only)
   ├─ Render graph with force simulation
   ├─ Apply adaptive spacing
   └─ Attach interaction handlers
```

### 3.2 Visualization Data Transform

**From ChromaDB Chunks → Graph Nodes**:
```python
# graph_builder.py:479-507
for chunk in chunks:
    node = {
        "id": chunk.chunk_id or chunk.id,
        "name": chunk.function_name or chunk.class_name or f"L{chunk.start_line}",
        "type": chunk.chunk_type,
        "file_path": str(chunk.file_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "complexity": chunk.complexity_score,
        "parent_id": chunk.parent_chunk_id,
        "depth": chunk.chunk_depth,
        "content": chunk.content,
        "docstring": chunk.docstring,
        "language": chunk.language,
    }

    # Add caller info if available
    if chunk_id in caller_map:
        node["callers"] = caller_map[chunk_id]

    # Add subproject info for monorepos
    if chunk.subproject_name:
        node["subproject"] = chunk.subproject_name
        node["color"] = subprojects[chunk.subproject_name]["color"]

    nodes.append(node)
```

**Color Assignment**:
- **Code chunks**: Color by type (function=yellow, class=blue, method=purple)
- **Monorepo chunks**: Color by subproject (8-color palette)
- **Complexity shading**: Darker = higher complexity (HSL lightness reduction)
- **Dead code**: Red border (no incoming caller/import edges, not entry point)

### 3.3 Relationship Data Sources

| Relationship Type | Data Source | Computation |
|------------------|-------------|-------------|
| `dir_hierarchy` | DirectoryIndex | Pre-computed during indexing |
| `dir_containment` | DirectoryIndex + chunks | Path matching |
| `file_containment` | Chunks (parent_chunk_id) | Top-level chunks only |
| `caller` | AST analysis | extract_function_calls() on all chunks |
| `semantic` | Vector search | Top 5 similar chunks (threshold ≥ 0.2) |
| `imports` | AST parsing (chunk.imports) | Not currently used in visualization |
| `dependency` | package.json parsing | Monorepo inter-project deps |
| Cycles | DFS analysis | Three-color marking on caller graph |

**Performance Optimizations**:
- Semantic relationships: Limited to top 5 per chunk, only for code chunks
- Caller relationships: Only cross-file calls (excludes internal)
- Directory hierarchy: Pre-indexed, not computed on-the-fly
- JSON loading: Streaming with progress bar, auto-switches to Dagre for >500 nodes

---

## 4. Gaps for New Requirements

### 4.1 Missing Features

**From User's New Requirements Analysis**:

1. **Node Grouping/Ungrouping**:
   - **Current**: Only expand/collapse (hierarchical)
   - **Missing**: Manual grouping of arbitrary nodes (non-hierarchical)
   - **Impact**: Can't create custom logical groups (e.g., "Auth Module", "API Endpoints")

2. **Saved View States (Bookmarks)**:
   - **Current**: No persistence of view state
   - **Missing**: Bookmark specific graph states (expanded nodes, zoom level, filters)
   - **Impact**: Can't return to previously explored views

3. **Filter Persistence**:
   - **Current**: Edge filters reset on page reload
   - **Missing**: Save/restore filter configurations
   - **Impact**: Must re-configure filters every session

4. **Layout Customization**:
   - **Current**: Spacing auto-calculated, no manual override
   - **Missing**: User-adjustable spacing, force parameters
   - **Impact**: Can't fine-tune layout for specific use cases

5. **Undo/Redo**:
   - **Current**: No history of view changes
   - **Missing**: Undo/redo for expand/collapse, grouping, filters
   - **Impact**: Can't revert accidental changes

6. **Group-Level Operations**:
   - **Current**: Operations only on individual nodes
   - **Missing**: Collapse/expand groups, apply filters to groups
   - **Impact**: Inefficient for large-scale view management

### 4.2 Architectural Limitations

1. **State Serialization**:
   - Current state is in-memory JavaScript objects
   - No serialization/deserialization infrastructure
   - Would need to implement state schema and localStorage API

2. **Group Data Model**:
   - No concept of "group" in current node/link structure
   - Would need to add:
     ```javascript
     let groups = new Map();  // groupId → { name, nodeIds, color, isCollapsed }
     ```
   - Group links vs. node links ambiguity

3. **History Management**:
   - No command pattern or memento pattern
   - Would need to track:
     ```javascript
     let history = {
         actions: [],        // [{ type, data, timestamp }]
         currentIndex: -1,
         undo() { ... },
         redo() { ... }
     };
     ```

4. **Layout Persistence**:
   - Node positions not saved (force simulation resets on load)
   - Would need to serialize `{ nodeId: { x, y, fx, fy } }`

### 4.3 UI/UX Gaps

1. **Group Creation UI**:
   - No multi-select mechanism
   - No context menu for "Create Group"
   - No group naming/editing interface

2. **Bookmark Management UI**:
   - No bookmark sidebar
   - No bookmark creation/deletion controls
   - No bookmark metadata (name, description, timestamp)

3. **Filter Presets**:
   - No saved filter combinations
   - No "Show All", "Hide All" quick actions
   - No filter presets (e.g., "Only Functions", "No Semantic Links")

4. **Layout Controls**:
   - No manual spacing slider
   - No force parameter tweaking UI
   - No "Lock Layout" to prevent force simulation updates

---

## 5. Recommended Refactoring Approach

### 5.1 High-Level Strategy

**Phase 1: State Management Foundation**
1. Extract state into centralized state manager
2. Implement state serialization/deserialization
3. Add localStorage persistence layer
4. Create state validation schema

**Phase 2: Group Data Model**
1. Define group structure and API
2. Extend graph data with group metadata
3. Implement group rendering logic (meta-nodes)
4. Add group interaction handlers

**Phase 3: History Management**
1. Implement command pattern for all mutations
2. Add undo/redo stack
3. Integrate with state manager
4. Add keyboard shortcuts (Ctrl+Z, Ctrl+Y)

**Phase 4: UI Extensions**
1. Multi-select interaction mode
2. Group creation/editing UI
3. Bookmark management sidebar
4. Filter preset system
5. Layout control panel

### 5.2 State Manager Design

```javascript
// Proposed state manager structure
class GraphState {
    constructor() {
        this.data = {
            // Graph data
            nodes: [],
            links: [],
            metadata: {},

            // View state
            visibleNodes: new Set(),
            collapsedNodes: new Set(),
            highlightedNode: null,

            // Groups
            groups: new Map(),  // groupId → GroupConfig

            // Filters
            edgeFilters: { ... },
            nodeFilters: { ... },

            // Layout
            currentLayout: 'force',
            layoutConfig: { ... },
            nodePositions: {},  // nodeId → { x, y }

            // Bookmarks
            bookmarks: [],  // [{ id, name, state, timestamp }]

            // Navigation
            viewStack: { stack: [], currentIndex: -1 }
        };

        this.listeners = new Set();
        this.history = new CommandHistory();
    }

    // State accessors
    getVisibleNodes() { ... }
    getFilteredLinks() { ... }

    // State mutations (via commands)
    expandNode(nodeId) {
        this.history.execute(new ExpandNodeCommand(nodeId));
    }

    createGroup(name, nodeIds) {
        this.history.execute(new CreateGroupCommand(name, nodeIds));
    }

    // Persistence
    serialize() { ... }
    deserialize(data) { ... }
    saveToBrowser() { localStorage.setItem('graph-state', this.serialize()); }
    loadFromBrowser() { ... }

    // Bookmarks
    createBookmark(name) { ... }
    restoreBookmark(id) { ... }

    // History
    undo() { this.history.undo(); }
    redo() { this.history.redo(); }

    // Observer pattern
    subscribe(listener) { this.listeners.add(listener); }
    notify(change) { this.listeners.forEach(l => l(change)); }
}
```

### 5.3 Group Model Design

```javascript
class GroupConfig {
    constructor(id, name, nodeIds, options = {}) {
        this.id = id;
        this.name = name;
        this.nodeIds = new Set(nodeIds);  // Set of node IDs in this group
        this.color = options.color || generateColor();
        this.isCollapsed = options.isCollapsed || false;
        this.position = options.position || null;  // { x, y }
        this.metadata = options.metadata || {};
    }

    // Serialization
    toJSON() { ... }
    static fromJSON(data) { ... }
}

// Rendering strategy: Meta-nodes
function renderGroups(groups) {
    // Option A: Groups as visual containers (bounding boxes)
    // - Draw convex hull around grouped nodes
    // - Label at top of hull

    // Option B: Groups as collapsed meta-nodes
    // - Replace grouped nodes with single "group node"
    // - Expand on click to show members
    // - Aggregate edges (group-to-node, group-to-group)

    // Recommendation: Hybrid approach
    // - Expanded groups: Show bounding box with label
    // - Collapsed groups: Show meta-node with member count
}
```

### 5.4 Command Pattern for History

```javascript
class Command {
    execute(state) { throw new Error("Must implement execute"); }
    undo(state) { throw new Error("Must implement undo"); }
}

class ExpandNodeCommand extends Command {
    constructor(nodeId) {
        super();
        this.nodeId = nodeId;
        this.previousState = null;
    }

    execute(state) {
        this.previousState = {
            visibleNodes: new Set(state.visibleNodes),
            collapsedNodes: new Set(state.collapsedNodes)
        };

        state.expandNode(this.nodeId);
    }

    undo(state) {
        state.visibleNodes = this.previousState.visibleNodes;
        state.collapsedNodes = this.previousState.collapsedNodes;
        state.notify({ type: 'view-changed' });
    }
}

class CreateGroupCommand extends Command {
    constructor(name, nodeIds) {
        super();
        this.groupId = generateId();
        this.name = name;
        this.nodeIds = nodeIds;
    }

    execute(state) {
        const group = new GroupConfig(this.groupId, this.name, this.nodeIds);
        state.groups.set(this.groupId, group);
        state.notify({ type: 'group-created', groupId: this.groupId });
    }

    undo(state) {
        state.groups.delete(this.groupId);
        state.notify({ type: 'group-deleted', groupId: this.groupId });
    }
}

class CommandHistory {
    constructor() {
        this.commands = [];
        this.currentIndex = -1;
    }

    execute(command) {
        // Remove forward history
        this.commands = this.commands.slice(0, this.currentIndex + 1);

        command.execute(graphState);
        this.commands.push(command);
        this.currentIndex++;
    }

    undo() {
        if (this.currentIndex >= 0) {
            this.commands[this.currentIndex].undo(graphState);
            this.currentIndex--;
        }
    }

    redo() {
        if (this.currentIndex < this.commands.length - 1) {
            this.currentIndex++;
            this.commands[this.currentIndex].execute(graphState);
        }
    }
}
```

### 5.5 Migration Path

**Step 1: Refactor Existing Code (Non-Breaking)**
1. Create `GraphState` class
2. Migrate global state variables into `GraphState.data`
3. Replace direct state mutations with `GraphState` methods
4. Add state change notifications
5. **No UI changes yet, just internal refactoring**

**Step 2: Add Persistence (Feature Parity)**
1. Implement `serialize()`/`deserialize()`
2. Add auto-save on state changes (debounced)
3. Load saved state on page load
4. **User sees: State persists across reloads**

**Step 3: Add History (New Feature)**
1. Implement command pattern for expand/collapse
2. Add undo/redo buttons
3. Add keyboard shortcuts
4. **User sees: Undo/redo for all actions**

**Step 4: Add Groups (New Feature)**
1. Implement `GroupConfig` model
2. Add multi-select interaction mode
3. Add "Create Group" context menu
4. Implement group rendering (bounding boxes + meta-nodes)
5. **User sees: Can create custom groups**

**Step 5: Add Bookmarks (New Feature)**
1. Implement bookmark creation from current state
2. Add bookmark sidebar UI
3. Add bookmark restoration
4. **User sees: Can save/restore view states**

**Step 6: Polish & Optimize**
1. Add filter presets
2. Add layout control panel
3. Performance testing with large graphs
4. Documentation and user guide

---

## 6. Implementation Recommendations

### 6.1 Priority Ranking

| Feature | Priority | Complexity | Dependencies | Impact |
|---------|----------|------------|--------------|--------|
| State Serialization | **P0** | Medium | None | Foundation for all features |
| Undo/Redo | **P0** | Medium | State Serialization | High user value |
| Node Grouping | **P1** | High | State Serialization | Core new feature |
| Bookmarks | **P1** | Low | State Serialization | High productivity boost |
| Filter Persistence | **P1** | Low | State Serialization | Quality-of-life |
| Layout Customization | **P2** | Medium | None | Advanced use cases |
| Multi-select UI | **P1** | Medium | None | Required for grouping |
| Group Meta-nodes | **P1** | High | Node Grouping | Visual clarity |

**Recommended Order**: P0 → P1 (in parallel) → P2

### 6.2 Testing Strategy

**Unit Tests**:
- `GraphState` class methods
- Command execute/undo logic
- Serialization/deserialization
- Group membership calculations

**Integration Tests**:
- State persistence across page reloads
- Undo/redo with multiple operations
- Group creation → collapse → bookmark → restore
- Edge filtering with groups

**Manual Tests**:
- Large graph performance (>1000 nodes)
- Browser compatibility (Chrome, Firefox, Safari)
- Mobile responsiveness (not currently supported)
- Accessibility (keyboard navigation)

### 6.3 Performance Considerations

**Current Bottlenecks**:
- Force simulation: O(n²) per tick
- Large JSON parsing: Blocks main thread
- DOM rendering: 500+ nodes causes lag

**Optimization Opportunities**:
- **WebWorker for force simulation**: Offload physics calculations
- **Virtual scrolling for large lists**: Only render visible elements
- **Incremental rendering**: Batch DOM updates
- **Indexing for group lookups**: O(1) node → group mapping

**Recommended Limits**:
- Max visible nodes: 500 (auto-switch to Dagre)
- Max group size: 100 nodes
- Max undo history: 50 actions
- Max bookmarks: 20 saved states

---

## 7. Example Usage Scenarios

### Scenario 1: Create Authentication Module Group
```javascript
// User flow:
1. Enable multi-select mode (checkbox or Shift+click)
2. Click nodes: login_handler, auth_middleware, jwt_utils
3. Right-click → "Create Group" → Name: "Auth Module"
4. System creates group with color-coded bounding box
5. User clicks group header → collapses to meta-node
6. User creates bookmark "Auth Module Review"

// State changes:
graphState.createGroup("Auth Module", [
    "chunk_id_login_handler",
    "chunk_id_auth_middleware",
    "chunk_id_jwt_utils"
]);

graphState.createBookmark("Auth Module Review");
```

### Scenario 2: Explore Circular Dependencies
```javascript
// User flow:
1. Toggle "Circular Dependencies" filter ON
2. See red dashed edges highlighting cycles
3. Click cycle edge → tooltip shows cycle path
4. Click node in cycle → detail pane shows callers
5. Expand both nodes in cycle
6. Bookmark view as "Cycle Investigation"

// State:
edgeFilters.cycles = true;
graphState.expandNode("node_A");
graphState.expandNode("node_B");
graphState.createBookmark("Cycle Investigation");
```

### Scenario 3: Compare Two Implementations
```javascript
// User flow:
1. Create group "Old Auth" (legacy_auth.py nodes)
2. Create group "New Auth" (auth_v2.py nodes)
3. Collapse both groups
4. Toggle semantic similarity ON
5. See similarity edges between groups
6. Bookmark as "Auth Migration"

// State:
const oldAuth = graphState.createGroup("Old Auth", [...]);
const newAuth = graphState.createGroup("New Auth", [...]);
graphState.collapseGroup(oldAuth.id);
graphState.collapseGroup(newAuth.id);
edgeFilters.semantic = true;
```

---

## 8. Open Questions

1. **Group Overlap**: Can a node belong to multiple groups?
   - Recommendation: No (simplifies UI), but allow group nesting

2. **Group Edge Aggregation**: How to visualize edges when groups are collapsed?
   - Option A: Show single edge with count badge (e.g., "3 connections")
   - Option B: Show all edges (may be cluttered)
   - Recommendation: Option A with tooltip showing details

3. **Bookmark Scope**: What to save in a bookmark?
   - Full state (expanded nodes, filters, layout, zoom)?
   - Just expanded nodes?
   - Recommendation: Full state minus node positions (to allow relayout)

4. **Filter Presets**: Should filter presets be project-specific or global?
   - Recommendation: Global presets + project-specific overrides

5. **Performance Limit**: What happens when >1000 nodes visible?
   - Recommendation: Warning message + force Dagre layout

---

## Appendix A: File Structure Reference

```
src/mcp_vector_search/cli/commands/visualize/
├── graph_builder.py          # Graph data construction
│   ├── build_graph_data()    # Main entry point
│   ├── detect_cycles()       # Cycle detection algorithm
│   ├── parse_project_dependencies()  # Monorepo deps
│   └── extract_function_calls()      # AST call extraction
│
├── templates/
│   ├── base.py              # HTML template generation
│   ├── scripts.py           # JavaScript code
│   │   ├── D3 initialization
│   │   ├── Graph visualization
│   │   ├── Interaction handlers
│   │   ├── Content pane logic
│   │   ├── Navigation stack
│   │   └── Layout switching
│   └── styles.py            # CSS styles
│
└── command.py               # CLI command handler
```

---

## Appendix B: Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INDEXING PHASE                                           │
├─────────────────────────────────────────────────────────────┤
│ Source Code → Tree-sitter → AST → Chunks → ChromaDB        │
│                                   ↓                         │
│                            Directory Index (JSON)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. GRAPH BUILDING PHASE                                     │
├─────────────────────────────────────────────────────────────┤
│ ChromaDB + DirectoryIndex                                   │
│    ↓                                                        │
│ build_graph_data()                                          │
│    ├─ Create nodes (subprojects, dirs, files, chunks)      │
│    ├─ Create links (hierarchy, calls, semantic, cycles)    │
│    └─ Compute metadata                                     │
│    ↓                                                        │
│ chunk-graph.json (exported)                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. VISUALIZATION PHASE                                      │
├─────────────────────────────────────────────────────────────┤
│ Browser loads HTML + chunk-graph.json                       │
│    ↓                                                        │
│ D3.js visualization                                         │
│    ├─ Initialize state (visibleNodes, collapsedNodes)      │
│    ├─ Render graph (force-directed layout)                 │
│    ├─ Attach interaction handlers                          │
│    └─ Show detail pane on click                            │
│    ↓                                                        │
│ User interactions (expand, collapse, navigate)             │
└─────────────────────────────────────────────────────────────┘
```

---

**End of Analysis**

This document provides a comprehensive foundation for implementing the new grouping, bookmarking, and state management features. The recommended refactoring approach preserves existing functionality while enabling incremental feature additions.
