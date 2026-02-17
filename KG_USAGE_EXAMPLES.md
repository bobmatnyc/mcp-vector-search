# Knowledge Graph Integration - Usage Examples

## Quick Start

### 1. Build Knowledge Graph (One-Time Setup)
```bash
# Navigate to your project
cd /path/to/your/project

# Initialize MCP Vector Search (if not done)
mcp-vector-search init

# Index your codebase (if not done)
mcp-vector-search index

# Build knowledge graph
mcp-vector-search kg build
```

Output:
```
┌─────────────────────────────────────────────┐
│  Building Knowledge Graph                   │
│  Project: /path/to/your/project             │
└─────────────────────────────────────────────┘

✓ Retrieved 1234 chunks
Processing entities and relationships...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

┌─────────────────────────────────────────────┐
│  Knowledge Graph Statistics                 │
├─────────────────────────────────────────────┤
│  Metric            │  Count                 │
├─────────────────────────────────────────────┤
│  Code Entities     │  456                   │
│  Calls             │  1234                  │
│  Imports           │  234                   │
│  Inherits          │  89                    │
│  Contains          │  567                   │
└─────────────────────────────────────────────┘

✓ Knowledge graph built successfully!
```

### 2. Start Unified Visualizer
```bash
# Start visualizer (auto-exports KG data if needed)
mcp-vector-search visualize
```

Output:
```
✓ Visualization server running

URL: http://localhost:8501
Directory: /path/to/project/.mcp-vector-search/visualization

Press Ctrl+C to stop
```

Browser opens automatically to `http://localhost:8501`

## Using the Visualizer

### Chunks View (Default)
When the visualizer opens, you'll see the **Chunks** view by default:

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 Code Tree                                    v2.2.21    │
├─────────────────────────────────────────────────────────────┤
│  View                                                       │
│  ┌──────────┐ ┌─────────────────┐                          │
│  │ Chunks ✓ │ │ Knowledge Graph │                          │
│  └──────────┘ └─────────────────┘                          │
│                                                             │
│  Layout Mode                                                │
│  ┌──────┐ ┌─────────┐ ┌──────────┐                         │
│  │ Tree │ │ Treemap │ │ Sunburst │                         │
│  └──────┘ └─────────┘ └──────────┘                         │
│                                                             │
│  Tree Layout                                                │
│  Linear ⚪──────────○ Circular                              │
│                                                             │
│  Show Files                                                 │
│  ┌─────┐ ┌──────┐ ┌──────┐                                 │
│  │ All │ │ Code │ │ Docs │                                 │
│  └─────┘ └──────┘ └──────┘                                 │
└─────────────────────────────────────────────────────────────┘

[Hierarchical tree visualization of code chunks]
```

**Features:**
- Expandable directory tree
- Click files to view code chunks
- Click chunks to see code in side panel
- Toggle between Tree/Treemap/Sunburst layouts

### Knowledge Graph View

Click **"Knowledge Graph"** button to switch views:

```
┌─────────────────────────────────────────────────────────────┐
│  🔍 Code Tree                                    v2.2.21    │
├─────────────────────────────────────────────────────────────┤
│  View                                                       │
│  ┌──────────┐ ┌─────────────────┐                          │
│  │  Chunks  │ │ Knowledge Graph ✓│                         │
│  └──────────┘ └─────────────────┘                          │
│                                                             │
│  Relationship Filters                                       │
│  ☑ Calls                                                    │
│  ☑ Imports                                                  │
│  ☑ Inherits                                                 │
│  ☑ Contains                                                 │
└─────────────────────────────────────────────────────────────┘

[Force-directed graph of code entities and relationships]
```

**Features:**
- Force-directed layout with physics simulation
- Drag nodes to reposition
- Zoom with mouse wheel
- Pan by dragging background
- Filter relationships with checkboxes
- Click nodes to view details

## Interactive Examples

### Example 1: Exploring Class Inheritance

**Scenario:** Find all classes that inherit from `BaseModel`

1. Click **"Knowledge Graph"** tab
2. Find the `BaseModel` node (pink circle)
3. Click the node to see details:
   ```
   ┌─────────────────────────────────┐
   │  Node Details                   │
   ├─────────────────────────────────┤
   │  Name: BaseModel                │
   │  Type: class                    │
   │  File: models/base.py           │
   │  Connections: 12                │
   └─────────────────────────────────┘
   ```
4. Red links show inheritance relationships
5. Follow red links to child classes

### Example 2: Tracing Function Calls

**Scenario:** See what functions call `process_data()`

1. Switch to **"Knowledge Graph"** view
2. Find `process_data` function (green circle)
3. Look for **orange (calls)** links pointing TO the node
4. Toggle off other relationships to isolate calls:
   - ☐ Imports
   - ☐ Inherits
   - ☐ Contains
   - ☑ Calls

### Example 3: Understanding Module Dependencies

**Scenario:** Visualize import relationships between modules

1. Go to **"Knowledge Graph"** view
2. Uncheck all filters except **Imports**:
   - ☐ Calls
   - ☑ Imports
   - ☐ Inherits
   - ☐ Contains
3. Blue links show import relationships
4. Identify modules with many imports (large number of blue links)
5. Find circular dependencies (blue links forming loops)

### Example 4: Comparing Views

**Scenario:** See both hierarchical structure and entity relationships

**Chunks View:**
```
src/
├── models/
│   ├── user.py
│   │   ├── User (class)
│   │   ├── validate_email() (function)
│   │   └── hash_password() (function)
│   └── base.py
│       └── BaseModel (class)
└── utils/
    └── crypto.py
        └── hash_password() (function)
```

**Knowledge Graph View:**
```
     BaseModel
        ↓ inherits (red)
     User ──→ hash_password() (calls, orange)
              ↓
          crypto.hash_password()
```

**Insight:** Both `user.py` and `crypto.py` have `hash_password()` functions, but User class calls the crypto version.

## Advanced Usage

### Filtering for Code Quality Analysis

**Find highly connected functions (potential complexity hotspots):**

1. Switch to Knowledge Graph view
2. Enable only **Calls** relationships
3. Look for nodes with many orange links
4. These are functions that call many others (high fan-out) or are called by many (high fan-in)

### Identifying Architectural Layers

**Visualize module dependencies:**

1. Knowledge Graph view
2. Enable **Imports** and **Contains**
3. Modules that import many others: Upper layers
4. Modules imported by many: Lower layers/utilities

### Finding Code Duplication

**Look for similar function patterns:**

1. Chunks view → Find function by name
2. Knowledge Graph view → Check for multiple nodes with similar names
3. Example: `process_data()` in multiple modules might indicate duplication

## Keyboard Shortcuts

### Both Views
- `Ctrl/Cmd + Scroll`: Zoom in/out
- `Drag Background`: Pan view
- `Click Node`: Show details in side panel
- `Esc`: Close side panel

### Knowledge Graph Specific
- `Drag Node`: Reposition (physics simulation continues)
- `Shift + Click Node`: Pin node in place
- `Double Click Node`: Center view on node

## Performance Tips

### For Large Codebases (>1000 files)

**Chunks View:**
- Use **Code** filter to hide documentation
- Start with collapsed directories
- Expand only the modules you need

**Knowledge Graph View:**
- Filter relationships to reduce visual clutter
- Focus on one relationship type at a time
- Use search to find specific entities

### Optimizing KG Build

```bash
# Faster build: skip expensive DOCUMENTS relationships
mcp-vector-search kg build --skip-documents

# Incremental build: only process new chunks
mcp-vector-search kg build --incremental

# Test with limited chunks
mcp-vector-search kg build --limit 100
```

## Troubleshooting

### "Knowledge graph not found" error

**Problem:** Clicked "Knowledge Graph" tab, got error message.

**Solution:**
```bash
# Build the knowledge graph first
mcp-vector-search kg build
```

### KG view is empty

**Problem:** Knowledge Graph view shows no nodes/links.

**Diagnosis:**
```bash
# Check KG statistics
mcp-vector-search kg stats
```

**If entities = 0:**
```bash
# Rebuild knowledge graph
mcp-vector-search kg build --force
```

### Slow rendering in KG view

**Problem:** Force-directed graph is laggy.

**Solutions:**
1. Filter relationships to reduce links:
   - Uncheck **Contains** (often has many links)
   - Focus on **Calls** or **Imports** only

2. Rebuild with filters:
   ```bash
   mcp-vector-search kg build --skip-documents
   ```

3. Check graph size:
   ```bash
   mcp-vector-search kg stats
   ```

   If >1000 entities, consider filtering by file:
   ```bash
   mcp-vector-search visualize export --file "src/core/*"
   ```

### Browser freezes on large graphs

**Problem:** >2000 entities causes browser to freeze.

**Solution:** Use code-only filter to reduce nodes:
```bash
mcp-vector-search visualize serve --code-only
```

## Integration with Other Commands

### Query KG from CLI

```bash
# Find related entities
mcp-vector-search kg query "SemanticSearchEngine"

# Show call graph
mcp-vector-search kg calls "search_code"

# Show inheritance tree
mcp-vector-search kg inherits "BaseModel"
```

### Export Data for Analysis

```bash
# Export chunk graph as JSON
mcp-vector-search visualize export -o chunk-graph.json

# Access raw KG data
cat .mcp-vector-search/visualization/kg-graph.json
```

### Refresh Data

```bash
# After code changes, refresh both indices
mcp-vector-search index              # Update chunk index
mcp-vector-search kg build --incremental  # Update KG with new chunks
mcp-vector-search visualize          # Restart visualizer
```

## Common Workflows

### Workflow 1: Onboarding New Developer

**Goal:** Understand codebase structure and dependencies

```bash
# Setup
mcp-vector-search init
mcp-vector-search index
mcp-vector-search kg build

# Visualize
mcp-vector-search visualize
```

1. Start with **Chunks** view to understand directory structure
2. Switch to **Knowledge Graph** to see:
   - Import relationships (module dependencies)
   - Class hierarchy (inheritance)
   - Call patterns (function interactions)

### Workflow 2: Refactoring Analysis

**Goal:** Assess impact of changing a function

```bash
# Find function in KG view
mcp-vector-search visualize
```

1. Click **"Knowledge Graph"**
2. Search for function name
3. Enable **Calls** filter only
4. Orange links pointing TO node = callers (will be affected)
5. Orange links pointing FROM node = callees (dependencies)

### Workflow 3: Circular Dependency Detection

**Goal:** Find and break circular imports

```bash
mcp-vector-search visualize
```

1. Switch to **Knowledge Graph**
2. Enable **Imports** only
3. Look for blue link cycles (loops)
4. Identify modules in cycle
5. Refactor to break dependency

## API Endpoints (for Custom Integrations)

The visualizer exposes REST APIs you can use:

```bash
# Get chunk graph data
curl http://localhost:8501/api/graph

# Get KG data
curl http://localhost:8501/api/kg-graph

# Get relationships for a specific chunk
curl http://localhost:8501/api/relationships/{chunk_id}

# Get callers for a function
curl http://localhost:8501/api/callers/{chunk_id}
```

Example response:
```json
{
  "nodes": [
    {
      "id": "entity_123",
      "name": "User",
      "type": "class",
      "file_path": "models/user.py",
      "group": 3
    }
  ],
  "links": [
    {
      "source": "entity_123",
      "target": "entity_456",
      "type": "calls",
      "weight": 1
    }
  ]
}
```

## Best Practices

1. **Build KG regularly**: Update after significant code changes
2. **Use filters**: Reduce visual clutter in large graphs
3. **Combine views**: Use both Chunks and KG for comprehensive understanding
4. **Export data**: Use CLI commands for detailed analysis
5. **Document findings**: Capture insights in architecture docs

## Next Steps

- Explore the [Integration Summary](INTEGRATION_SUMMARY.md) for technical details
- Review [Architecture Documentation](KG_INTEGRATION_ARCHITECTURE.md) for system design
- Check out [MCP Vector Search docs](https://github.com/yourusername/mcp-vector-search) for more features
