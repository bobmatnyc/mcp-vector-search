# KG Integration Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Browser                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Single HTML Page (index.html)            │  │
│  │  ┌──────────┐  ┌─────────────────────┐              │  │
│  │  │ Chunks   │  │  Knowledge Graph    │  ← View Tabs │  │
│  │  └──────────┘  └─────────────────────┘              │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  View-Specific Controls                       │  │  │
│  │  │  • Chunks: Tree/Treemap/Sunburst              │  │  │
│  │  │  • KG: Filters (Calls/Imports/Inherits/etc)   │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  <svg id="graph">  (Chunks View)              │  │  │
│  │  │  • Hierarchical tree of code chunks           │  │  │
│  │  │  • Expandable directories and files           │  │  │
│  │  │  • Tree/Treemap/Sunburst layouts              │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  <svg id="kg-graph">  (KG View)               │  │  │
│  │  │  • Force-directed graph of entities           │  │  │
│  │  │  • Nodes: Files, Modules, Classes, Functions  │  │  │
│  │  │  • Links: Calls, Imports, Inherits, Contains  │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  Side Panel (Viewer)                          │  │  │
│  │  │  • Node details                               │  │  │
│  │  │  • Code preview                               │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↕ HTTP (localhost:8501)
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Server (server.py)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GET /                                                │  │
│  │  → Returns index.html                                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GET /api/graph                                       │  │
│  │  → Returns chunk graph data (nodes + links)          │  │
│  │  → Source: chunk-graph.json                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GET /api/kg-graph                                    │  │
│  │  → Returns KG data (entities + relationships)        │  │
│  │  → Source: kg-graph.json                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GET /api/relationships/{chunk_id}                   │  │
│  │  → Returns callers and semantic neighbors           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│           File System (.mcp-vector-search/)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  visualization/                                       │  │
│  │  ├── index.html   (Generated from templates)         │  │
│  │  ├── chunk-graph.json  (Chunk relationship data)     │  │
│  │  └── kg-graph.json     (Knowledge graph data)        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  knowledge_graph/                                     │  │
│  │  └── kuzu_db/  (KuzuDB with entities/relationships)  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  chroma.sqlite3  (Vector database with chunks)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Chunks View Flow
```
1. User opens visualizer → mcp-vector-search visualize
2. Server exports chunk-graph.json (if not exists)
3. Browser loads index.html
4. JavaScript fetches /api/graph
5. D3.js renders hierarchical tree
6. User clicks "Chunks" tab → shows chunk visualization
```

### KG View Flow
```
1. User clicks "Knowledge Graph" tab
2. JavaScript calls setView('kg')
3. If KG data not loaded:
   a. Fetch /api/kg-graph
   b. Parse JSON response
   c. Initialize force simulation
4. D3.js renders force-directed graph
5. User can:
   - Drag nodes
   - Zoom/pan
   - Filter relationships
   - Click nodes for details
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Command                             │
│  mcp-vector-search visualize                                │
│  ├── Check if index.html exists → export_to_html()          │
│  ├── Check if chunk-graph.json exists → _export_chunks()    │
│  ├── Check if kg-graph.json exists → _export_kg_data()      │
│  └── Start FastAPI server on port 8501                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   Template Generation                       │
│  base.py: generate_html_template()                          │
│  ├── styles.py: get_all_styles()                            │
│  │   └── get_kg_styles()  ← NEW                             │
│  └── scripts.py: get_all_scripts()                          │
│      ├── Chunks visualization logic                         │
│      └── KG visualization logic  ← NEW                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Browser Rendering                        │
│  index.html loaded                                          │
│  ├── CSS applied (dark/light theme)                         │
│  ├── D3.js v7 loaded                                        │
│  ├── View tabs initialized                                  │
│  ├── Chunks view rendered (default)                         │
│  └── KG view ready (lazy-loaded on demand)                  │
└─────────────────────────────────────────────────────────────┘
```

## State Management

```javascript
// Global State Variables

// View Management
currentView = 'chunks' | 'kg'  // Active view

// Chunks View State
allNodes = []           // All graph nodes
allLinks = []           // All graph links
treeData = null         // Hierarchical tree data
currentLayout = 'linear' | 'circular'
currentVizMode = 'tree' | 'treemap' | 'sunburst'
currentGroupingMode = 'file' | 'ast'

// KG View State
kgNodes = []            // KG entity nodes
kgLinks = []            // KG relationship links
kgSimulation = null     // D3 force simulation
kgLinkElements = null   // SVG link elements
kgNodeElements = null   // SVG node elements
kgLabelElements = null  // SVG label elements

// Shared State
isViewerOpen = false    // Side panel visibility
showCallLines = true    // Call relationship visibility
currentFileFilter = 'all' | 'code' | 'docs'
```

## Event Flow

```
User Action           →  Event Handler      →  State Update  →  UI Update
──────────────────────────────────────────────────────────────────────────
Click "KG" tab        →  setView('kg')      →  currentView   →  Show #kg-graph
                                                               →  Hide #graph
                                                               →  Show KG controls
                                                               →  Hide chunk controls

Click KG node         →  showKGNodeInfo()   →  isViewerOpen  →  Open side panel
                                                               →  Display node details

Drag KG node          →  dragged()          →  node.fx/fy    →  Update position
                                                               →  Simulation restart

Toggle filter         →  filterKGLinks()    →  link.display  →  Show/hide links

Zoom/Pan              →  D3 zoom handler    →  transform     →  Scale/translate SVG
```

## Performance Considerations

### Chunks View
- Lazy rendering: Only render visible nodes
- Cached hierarchies: Pre-compute file/AST structures
- Incremental updates: Only update changed nodes

### KG View
- Lazy loading: Fetch KG data only when needed
- Force simulation: Optimized with collision detection
- Link filtering: CSS display toggle (no re-render)
- Node dragging: Pin node position during drag

### Memory Usage
- **Chunks view**: ~2-5MB for 1000 files
- **KG view**: ~5-10MB for 1000 entities
- **Total**: ~10-15MB for typical project

## Error Handling

```
Scenario                    →  Handling                    →  Fallback
─────────────────────────────────────────────────────────────────────────
KG not built                →  Show alert                  →  Stay on chunks view
                            →  Create empty kg-graph.json

KG data load fails          →  Catch error                 →  Switch to chunks view
                            →  Log to console

Network error               →  Retry with timeout          →  Show error message
                            →  Graceful degradation

Invalid KG data             →  Validate JSON               →  Show empty graph
                            →  Log validation errors

Server not responding       →  Connection timeout          →  Show connection error
```

## Testing Strategy

### Unit Tests
- `setView()`: Verify view switching logic
- `loadKGData()`: Mock API response
- `renderKG()`: Test D3 rendering
- `filterKGLinks()`: Verify filter logic

### Integration Tests
- View switching preserves state
- KG data loads correctly
- Side panel displays node info
- Filters apply correctly

### E2E Tests
- Click "Knowledge Graph" tab
- Verify force-directed graph appears
- Drag nodes and verify position updates
- Toggle filters and verify links hide/show
- Click nodes and verify side panel opens

## Deployment

```bash
# Development
mcp-vector-search visualize --port 8501

# Production (with KG built)
mcp-vector-search kg build      # Build KG first
mcp-vector-search visualize     # Start visualizer
```

## Monitoring

Key metrics to track:
- **View switching time**: Should be <100ms
- **KG data load time**: Should be <1s for 1000 nodes
- **Render time**: Should be <500ms
- **Frame rate**: Should maintain 60fps during interactions
- **Memory usage**: Should stay <50MB
