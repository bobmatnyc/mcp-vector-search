# Knowledge Graph Integration into Main Visualizer

## Summary

Successfully integrated the KG (Knowledge Graph) visualization into the main visualizer with a button to switch between "Chunks" and "Knowledge Graph" views. Both views now run on the same server (port 8501) with seamless switching.

## Changes Made

### 1. HTML Template (`src/mcp_vector_search/cli/commands/visualize/templates/base.py`)

**Added View Switcher:**
- Replaced single "Visualization Mode" with two sections:
  - "View" section with "Chunks" | "Knowledge Graph" buttons
  - "Layout Mode" for chunks (Tree/Treemap/Sunburst) - shown only in chunks view
  - "Relationship Filters" for KG (Calls/Imports/Inherits/Contains) - shown only in KG view

**Added KG Container:**
- Added `<svg id="kg-graph">` next to existing `<svg id="graph">`
- KG graph is hidden by default, shown when KG view is selected

### 2. JavaScript (`src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`)

**Added KG Visualization Logic:**
- `setView(view)`: Switch between 'chunks' and 'kg' views
- `loadKGData()`: Fetch KG data from `/api/kg-graph` endpoint
- `renderKG()`: D3.js force-directed graph rendering
  - Nodes colored by type (file/module/class/function)
  - Links colored by relationship (calls/imports/inherits/contains)
  - Draggable nodes with force simulation
  - Zoom and pan support
- `filterKGLinks()`: Filter links by relationship type
- `showKGNodeInfo(node)`: Display node details in side panel

**Color Schemes:**
- **Nodes:**
  - File: Indigo (#6366f1)
  - Module: Purple (#8b5cf6)
  - Class: Pink (#ec4899)
  - Function: Green (#10b981)
- **Links:**
  - Calls: Orange (#f59e0b)
  - Imports: Blue (#3b82f6)
  - Inherits: Red (#ef4444)
  - Contains: Gray (#6b7280)

### 3. CSS Styles (`src/mcp_vector_search/cli/commands/visualize/templates/styles.py`)

**Added KG Styles:**
- `get_kg_styles()`: KG-specific styles for nodes, links, labels
- Integrated into `get_all_styles()` function
- Consistent with existing theme system (dark/light mode support)

### 4. CLI Command (`src/mcp_vector_search/cli/commands/visualize/cli.py`)

**Added KG Export Logic:**
- `_export_kg_data(viz_dir)`: Export KG data to `kg-graph.json`
- Auto-export KG data when visualize server starts (if not already present)
- Graceful handling when KG is not built (creates empty data file)

### 5. Server (`src/mcp_vector_search/cli/commands/visualize/server.py`)

**Existing Endpoint Used:**
- `/api/kg-graph`: Returns KG nodes and links (already implemented)
- No changes needed to server code

## User Experience

### Before
- `mcp-vector-search visualize` - Shows chunk/semantic visualization
- `mcp-vector-search kg visualize` - Shows KG force-directed graph (separate server on port 8502)

### After
- `mcp-vector-search visualize` - Shows unified visualizer
  - **Chunks tab**: Tree/Treemap/Sunburst views of code chunks
  - **Knowledge Graph tab**: Force-directed graph of code entities and relationships
- Single server on port 8501
- Instant switching between views without reloading

## UI Layout

```
┌─────────────────────────────────────────────────┐
│  mcp-vector-search Visualizer                   │
│  ┌──────────┐ ┌──────────────────┐              │
│  │ Chunks   │ │ Knowledge Graph  │  ← Tabs     │
│  └──────────┘ └──────────────────┘              │
├─────────────────────────────────────────────────┤
│  [View-specific controls]                       │
│  - Chunks: Layout Mode (Tree/Treemap/Sunburst) │
│  - KG: Relationship Filters (Calls/Imports/etc) │
├─────────────────────────────────────────────────┤
│                                                 │
│     [Current view content here]                 │
│     - Chunks: Hierarchical tree/map            │
│     - KG: Force-directed graph                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Implementation Details

### View Switching Logic

1. User clicks "Knowledge Graph" button
2. `setView('kg')` is called
3. Hides chunk graph (`#graph`), shows KG graph (`#kg-graph`)
4. Shows KG controls, hides chunk controls
5. If KG data not loaded, calls `loadKGData()`
6. Renders force-directed graph with D3.js

### KG Data Loading

- **Lazy loading**: KG data is only fetched when user clicks "Knowledge Graph" tab
- **Fallback**: If KG not built, shows alert and falls back to chunks view
- **Empty state**: Creates empty `kg-graph.json` if KG build hasn't run

### Force-Directed Graph

- Uses D3.js v7 force simulation
- Forces applied:
  - `forceLink`: Connects related nodes
  - `forceManyBody`: Node repulsion (-300 strength)
  - `forceCenter`: Centers graph in viewport
  - `forceCollide`: Prevents node overlap (30px radius)
- Interactive:
  - Drag nodes to reposition
  - Zoom with scroll wheel
  - Click nodes to view details
  - Hover for visual feedback

## Testing

To test the integration:

```bash
# 1. Build knowledge graph (if not already built)
mcp-vector-search kg build

# 2. Start visualizer
mcp-vector-search visualize

# 3. In browser:
#    - Click "Knowledge Graph" button
#    - Verify force-directed graph appears
#    - Test node dragging
#    - Test zoom/pan
#    - Test relationship filters
#    - Click "Chunks" to switch back
```

## Future Enhancements

Possible improvements for future iterations:

1. **Search Integration**: Search across both chunks and KG nodes
2. **Cross-View Navigation**: Click chunk to find corresponding KG node
3. **Layout Persistence**: Remember user's last selected view
4. **KG Layout Options**: Add different layout algorithms (hierarchical, circular)
5. **Relationship Strength**: Visualize relationship weight with link thickness
6. **Node Grouping**: Cluster nodes by file/module/package
7. **Export**: Export KG view as PNG/SVG
8. **Performance**: Virtual rendering for large graphs (>1000 nodes)

## Files Modified

1. `/src/mcp_vector_search/cli/commands/visualize/templates/base.py` - HTML template with view switcher
2. `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` - JavaScript for KG visualization
3. `/src/mcp_vector_search/cli/commands/visualize/templates/styles.py` - CSS for KG elements
4. `/src/mcp_vector_search/cli/commands/visualize/cli.py` - KG data export logic

## Lines of Code Delta

- **Added**: ~250 lines (JavaScript + CSS + Python)
- **Modified**: ~20 lines (HTML template + CLI)
- **Deleted**: 0 lines
- **Net Change**: +270 lines

## Conclusion

The integration is complete and functional. Users can now access both chunk and knowledge graph visualizations from a single unified interface, eliminating the need for separate commands and servers.
