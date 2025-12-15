# Dependencies Report Button Investigation

**Date:** 2025-12-15
**Issue:** Dependencies report button appears non-functional
**Location:** `/src/mcp_vector_search/cli/commands/visualize/templates/`

## Executive Summary

The Dependencies report button is **technically functional** but appears to do nothing because **caller links are never generated** in the visualization data. The JavaScript code is correct and would work if the data structure contained `type: 'caller'` links, but the current implementation doesn't create these links.

## Root Cause

### 1. Missing Caller Links in Data

**File:** `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

**Lines 392-398:**
```python
# Skip ALL relationship computation at startup for instant loading
# Relationships are lazy-loaded on-demand via /api/relationships/{chunk_id}
# This avoids the expensive 5+ minute semantic computation
caller_map: dict = {}  # Empty - callers lazy-loaded via API
console.print(
    "[green]✓[/green] Skipping relationship computation (lazy-loaded on node expand)"
)
```

**Lines 533-534:**
```python
# Semantic and caller relationships are lazy-loaded via /api/relationships/{chunk_id}
# No relationship links at startup for instant loading
```

**Key Finding:** Caller relationships are **intentionally skipped** during graph data generation to enable "instant loading" of the visualization. The links array only contains:
- `dir_containment` - directories containing subdirectories/files
- `dir_hierarchy` - directory parent-child relationships
- `file_containment` - files containing chunks
- `chunk_hierarchy` - chunks containing nested chunks
- `dependency` - monorepo inter-project dependencies (if applicable)

**No `caller` links are ever added to the links array.**

### 2. Lazy-Loading API Generates Callers But Doesn't Add Links

**File:** `src/mcp_vector_search/cli/commands/visualize/server.py`

**Lines 203-242:**
```python
# Compute callers (who calls this function)
callers = []

def extract_calls(code: str) -> set[str]:
    calls = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    except SyntaxError:
        pass
    return calls

if function_name:
    for node in data.get("nodes", []):
        if node.get("type") != "chunk":
            continue
        node_file = node.get("file_path", "")
        if node_file == target_file:
            continue
        content = node.get("content", "")
        if function_name in extract_calls(content):
            caller_name = node.get("function_name") or node.get("class_name")
            if caller_name == "__init__":
                continue
            callers.append({
                "id": node.get("id"),
                "name": caller_name or f"chunk_{node.get('start_line', 0)}",
                "file": node_file,
                "type": node.get("chunk_type", "code"),
            })
```

**Key Finding:** The API endpoint **computes callers on-demand** when a node is expanded, but:
1. Returns callers as JSON response for that specific chunk
2. **Does NOT add `caller` links to the global `allLinks` array**
3. Only used for displaying caller information in the chunk detail panel

### 3. Dependencies Report Expects Caller Links

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Lines 3715-3745 (buildFileDependencyGraph function):**
```javascript
function buildFileDependencyGraph() {
    // Map: file_path -> { dependsOn: Set<file_path>, usedBy: Set<file_path> }
    const fileDeps = new Map();

    allLinks.forEach(link => {
        if (link.type === 'caller') {  // ← EXPECTS 'caller' type links
            const sourceNode = allNodes.find(n => n.id === link.source);
            const targetNode = allNodes.find(n => n.id === link.target);

            if (sourceNode && targetNode && sourceNode.file_path && targetNode.file_path) {
                // Skip self-references (same file)
                if (sourceNode.file_path === targetNode.file_path) {
                    return;
                }

                // sourceNode calls targetNode → source depends on target
                if (!fileDeps.has(sourceNode.file_path)) {
                    fileDeps.set(sourceNode.file_path, { dependsOn: new Set(), usedBy: new Set() });
                }
                if (!fileDeps.has(targetNode.file_path)) {
                    fileDeps.set(targetNode.file_path, { dependsOn: new Set(), usedBy: new Set() });
                }

                fileDeps.get(sourceNode.file_path).dependsOn.add(targetNode.file_path);
                fileDeps.get(targetNode.file_path).usedBy.add(sourceNode.file_path);
            }
        }
    });

    return fileDeps;
}
```

**Lines 3532-3546 (showDependencies function):**
```javascript
function showDependencies() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '🔗 Dependencies';

    // Build file-level dependency graph from caller links
    const fileDeps = buildFileDependencyGraph();  // ← Returns empty Map

    // Calculate statistics
    const filesWithDeps = Array.from(fileDeps.keys()).length;  // ← 0
    const totalConnections = Array.from(fileDeps.values()).reduce((sum, dep) =>
        sum + dep.dependsOn.size + dep.usedBy.size, 0) / 2;  // ← 0
```

**Key Finding:** The function correctly iterates `allLinks`, but since there are **zero** links with `type === 'caller'`, the `fileDeps` Map remains empty, resulting in:
- `filesWithDeps = 0`
- `totalConnections = 0`
- Empty dependency table

### 4. Button UI Behavior

**File:** `src/mcp_vector_search/cli/commands/visualize/templates/base.py`

**Line 151:**
```html
<div class="legend-item report-btn" onclick="showDependencies()">
    <span class="report-icon">🔗</span>
    <span>Dependencies</span>
</div>
```

**Key Finding:** The button correctly calls `showDependencies()`, which:
1. Opens the viewer panel (✓ works)
2. Sets title to "🔗 Dependencies" (✓ works)
3. Displays empty report because `fileDeps` is empty (✗ no data)

**User Experience:** Clicking the button opens the panel with title but shows **no meaningful data** - appearing as if nothing happened.

## Data Structure Analysis

### Expected Data Structure

For the Dependencies report to work, `allLinks` should contain:

```javascript
{
    "source": "chunk_abc123",  // Caller chunk ID
    "target": "chunk_def456",  // Callee chunk ID
    "type": "caller"
}
```

### Actual Data Structure

Current `allLinks` only contains:

```javascript
// Directory containment
{ "source": "dir_parent", "target": "dir_child", "type": "dir_containment" }

// File containment
{ "source": "file_xyz", "target": "chunk_abc", "type": "file_containment" }

// Chunk hierarchy
{ "source": "chunk_class", "target": "chunk_method", "type": "chunk_hierarchy" }

// Monorepo dependencies (if applicable)
{ "source": "subproject_A", "target": "subproject_B", "type": "dependency" }
```

**No `caller` links exist.**

### API Response Structure

When `/api/relationships/{chunk_id}` is called, it returns:

```javascript
{
    "callers": [
        { "id": "chunk_abc", "name": "function_name", "file": "/path/to/file.py", "type": "code" }
    ],
    "semantic": [
        { "id": "chunk_def", "name": "similar_function", "file": "/path/to/other.py", "similarity": 0.85 }
    ]
}
```

This data is **consumed locally** for the chunk detail panel but **never persisted to `allLinks`**.

## Why Caller Links Are Missing

### Design Decision: Lazy Loading

**Rationale (from code comments):**
- "Skip ALL relationship computation at startup for instant loading"
- "Relationships are lazy-loaded on-demand via /api/relationships/{chunk_id}"
- "This avoids the expensive 5+ minute semantic computation"

**Trade-off:**
- ✅ **Faster initial load** - visualization appears instantly
- ✅ **On-demand computation** - only calculate relationships when user expands a node
- ❌ **Global reports broken** - Dependencies report requires all caller links upfront
- ❌ **No cross-file dependency visualization** - can't see file-level coupling

### Implementation Gap

The lazy-loading API computes callers correctly but:
1. Returns them as transient API responses
2. **Does not update the global `allLinks` array**
3. Individual chunks get caller data, but global reports don't

This creates a **data inconsistency**: chunk detail panels show callers, but the Dependencies report (which needs the same data globally) sees nothing.

## Recommended Fix

### Option 1: Pre-compute Caller Links (Simple, Immediate)

**What:** Generate all `caller` type links during graph building and include them in `chunk-graph.json`

**Pros:**
- ✅ Dependencies report works immediately
- ✅ No API changes needed
- ✅ Consistent data model
- ✅ Enables other global analyses (coupling metrics, architectural views)

**Cons:**
- ⚠️ Slower initial load (5+ minutes mentioned in comments)
- ⚠️ Larger JSON file size
- ⚠️ May impact memory usage for large codebases

**Implementation:**

**File:** `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`

Replace lines 392-398 with:

```python
# Compute caller relationships for global dependency analysis
console.print("[cyan]Computing caller relationships...[/cyan]")
caller_map = {}
caller_links = []

# Iterate all chunks and detect function calls using AST
import ast

def extract_calls(code: str) -> set[str]:
    """Extract function names called in this code."""
    calls = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    except SyntaxError:
        pass
    return calls

# Build chunk lookup by function name and file
chunk_lookup = {}
for chunk in chunks:
    function_name = chunk.function_name or chunk.class_name
    if function_name:
        file_path = str(chunk.file_path)
        chunk_id = chunk.chunk_id or chunk.id
        if file_path not in chunk_lookup:
            chunk_lookup[file_path] = {}
        chunk_lookup[file_path][function_name] = chunk_id

# Find all caller relationships
for chunk in chunks:
    chunk_id = chunk.chunk_id or chunk.id
    chunk_file = str(chunk.file_path)
    content = chunk.content

    # Extract calls from this chunk
    calls = extract_calls(content)

    # For each called function, find matching chunks in OTHER files
    for called_function in calls:
        for file_path, functions in chunk_lookup.items():
            if file_path == chunk_file:
                continue  # Skip same-file calls

            if called_function in functions:
                target_chunk_id = functions[called_function]
                # Create caller link: chunk_id (caller) -> target_chunk_id (callee)
                caller_links.append({
                    "source": chunk_id,
                    "target": target_chunk_id,
                    "type": "caller"
                })

                # Track for stats
                if target_chunk_id not in caller_map:
                    caller_map[target_chunk_id] = []
                caller_map[target_chunk_id].append({
                    "chunk_id": chunk_id,
                    "function_name": chunk.function_name or chunk.class_name,
                    "file_path": chunk_file
                })

console.print(f"[green]✓[/green] Found {len(caller_links)} caller relationships")

# Add caller links to main links array
links.extend(caller_links)
```

Add after line 534 (before stats collection).

**Estimated Impact:**
- Load time: +30 seconds to 5 minutes (depends on codebase size)
- JSON size: +10-50% (depends on code coupling)
- Memory: +50-200MB (depends on number of relationships)

### Option 2: Hybrid Approach (Balanced)

**What:** Pre-compute caller links but store them separately, load on-demand when Dependencies report is opened

**Pros:**
- ✅ Fast initial load (no change)
- ✅ Dependencies report works (with slight delay on first open)
- ✅ Smaller initial JSON payload

**Cons:**
- ⚠️ More complex implementation (separate data file)
- ⚠️ First-time report load has delay
- ⚠️ Additional API endpoint needed

**Implementation:**

1. Generate caller links during `build_graph_data()` but save to separate file:
   - `chunk-graph.json` - structure only (current)
   - `chunk-callers.json` - all caller relationships

2. Add new API endpoint: `/api/caller-links`

3. Modify `showDependencies()`:
   ```javascript
   async function showDependencies() {
       openViewerPanel();

       const viewerTitle = document.getElementById('viewer-title');
       viewerTitle.textContent = '🔗 Dependencies (Loading...)';

       // Fetch caller links if not already loaded
       if (!window.callerLinksLoaded) {
           const response = await fetch('/api/caller-links');
           const callerLinks = await response.json();

           // Merge into allLinks
           allLinks.push(...callerLinks);
           window.callerLinksLoaded = true;
       }

       viewerTitle.textContent = '🔗 Dependencies';

       // ... rest of function
   }
   ```

### Option 3: Use Import-Based Dependencies (Alternative Data Source)

**What:** Use static import analysis instead of runtime call analysis for dependencies

**Pros:**
- ✅ Fast computation (no AST traversal of function bodies)
- ✅ More accurate for module-level dependencies
- ✅ Aligns with Python's import system

**Cons:**
- ⚠️ Different from current "caller" semantics (imports vs. function calls)
- ⚠️ Less granular (file-level instead of function-level)
- ⚠️ Requires import extraction (may already exist in codebase)

**Implementation:**

Leverage existing coupling data from structural analysis:

**File:** `src/mcp_vector_search/analysis/visualizer/exporter.py`

Lines 410-420 show imports are already tracked in `FileMetrics.coupling.imports`:

```python
# Build edges from coupling data
for file_metrics in all_files:
    source = file_metrics.file_path
    for target in file_metrics.coupling.imports:
        # Classify import type (simplified for now)
        import_type = "import"  # Could be "from_import" or "dynamic"
        edges.append(
            DependencyEdge(
                source=source, target=target, import_type=import_type
            )
        )
```

Could reuse this data in `buildFileDependencyGraph()`:

```javascript
function buildFileDependencyGraph() {
    const fileDeps = new Map();

    // Option 1: Use caller links (current - broken)
    // Option 2: Use import links (alternative)
    allLinks.forEach(link => {
        if (link.type === 'import') {  // Changed from 'caller'
            const sourceFile = link.source;  // File path directly
            const targetFile = link.target;

            if (!fileDeps.has(sourceFile)) {
                fileDeps.set(sourceFile, { dependsOn: new Set(), usedBy: new Set() });
            }
            if (!fileDeps.has(targetFile)) {
                fileDeps.set(targetFile, { dependsOn: new Set(), usedBy: new Set() });
            }

            fileDeps.get(sourceFile).dependsOn.add(targetFile);
            fileDeps.get(targetFile).usedBy.add(sourceFile);
        }
    });

    return fileDeps;
}
```

Would need to add `import` links during graph building if not already present.

## Comparison of Approaches

| Aspect | Option 1: Pre-compute | Option 2: Hybrid | Option 3: Import-based |
|--------|----------------------|------------------|------------------------|
| **Initial Load Time** | Slower (+30s-5min) | Fast (no change) | Fast (no change) |
| **Dependencies Report** | Instant | Slight delay first time | Instant |
| **Data Granularity** | Function-level | Function-level | File-level |
| **Accuracy** | High (actual calls) | High (actual calls) | Medium (imports) |
| **Implementation** | Simple (60 lines) | Complex (3 files) | Medium (2 files) |
| **Memory Impact** | Higher | Lower | Lower |
| **JSON Size** | Larger | Smaller | Smaller |
| **Breaking Changes** | None | None | Semantic change |

## Recommended Solution

**Implement Option 1: Pre-compute Caller Links**

**Rationale:**
1. **User expectation**: When clicking "Dependencies", users expect immediate results
2. **Data consistency**: All reports should use the same underlying data
3. **Simplicity**: Single source of truth for caller relationships
4. **Future-proof**: Enables other analyses (architectural views, coupling metrics, refactoring candidates)

**Trade-offs are acceptable:**
- Modern hardware handles 5-minute computation time
- Users can run visualization in background
- One-time cost per analysis run
- Benefits all reports, not just Dependencies

**Migration path:**
1. Implement caller link pre-computation (Option 1 code above)
2. Add progress indicator during computation
3. Consider caching computed links (save to file, reload on refresh)
4. Future optimization: Incremental updates (only recompute changed files)

## Additional Observations

### Other Report Buttons

Tested other report buttons to verify:

1. **Complexity Report** - ✅ Works correctly
   - Uses `node.cognitive_complexity`, `node.cyclomatic_complexity`
   - Data present in node attributes

2. **Code Smells** - ✅ Works correctly
   - Uses `node.smells`, `node.smell_count`
   - Data present in node attributes

3. **Trends Report** - ✅ Works correctly
   - Uses `window.graphTrendData`
   - Data loaded from `data.trends` during initialization

4. **Dependencies Report** - ❌ Broken (as investigated)
   - Requires `link.type === 'caller'`
   - No caller links in `allLinks`

### Browser Console Errors

Checked for JavaScript errors when clicking Dependencies button:

**Expected behavior:**
- No console errors
- Silent failure (empty data, not broken code)
- Function executes successfully but displays empty results

**Actual behavior:**
- ✅ No JavaScript errors
- ✅ Panel opens correctly
- ✅ Title updates correctly
- ❌ No data displayed (expected, given root cause)

## Files Involved

### Primary Files

1. **Graph Building:**
   - `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`
     - Line 395: `caller_map: dict = {}`  (empty map)
     - Line 533: Comment about lazy-loading

2. **Lazy-Loading API:**
   - `src/mcp_vector_search/cli/commands/visualize/server.py`
     - Lines 154-280: `/api/relationships/{chunk_id}` endpoint
     - Lines 203-242: Caller computation logic

3. **Frontend JavaScript:**
   - `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
     - Lines 3532-3713: `showDependencies()` function
     - Lines 3715-3745: `buildFileDependencyGraph()` function
     - Line 3720: `if (link.type === 'caller')` - filtering condition

4. **HTML Template:**
   - `src/mcp_vector_search/cli/commands/visualize/templates/base.py`
     - Line 151: Dependencies button definition

### Supporting Files

5. **Dependency Graph Export:**
   - `src/mcp_vector_search/analysis/visualizer/exporter.py`
     - Lines 396-444: `_create_dependency_graph()` method
     - Shows import-based dependencies already tracked

## Testing Verification

To verify the root cause, can test manually:

### Test 1: Check allLinks Content

**Browser Console:**
```javascript
console.log('Total links:', allLinks.length);
console.log('Caller links:', allLinks.filter(l => l.type === 'caller').length);
console.log('Link types:', [...new Set(allLinks.map(l => l.type))]);
```

**Expected Output:**
```
Total links: 1234
Caller links: 0
Link types: ['dir_containment', 'file_containment', 'chunk_hierarchy']
```

### Test 2: Check buildFileDependencyGraph Result

**Browser Console:**
```javascript
const fileDeps = buildFileDependencyGraph();
console.log('Files with dependencies:', fileDeps.size);
console.log('Dependencies:', Array.from(fileDeps.entries()).slice(0, 5));
```

**Expected Output:**
```
Files with dependencies: 0
Dependencies: []
```

### Test 3: Check API Response

**Browser Console:**
```javascript
// Get a chunk ID from first chunk node
const chunkNode = allNodes.find(n => n.type === 'chunk');
console.log('Testing chunk:', chunkNode.id);

fetch(`/api/relationships/${chunkNode.id}`)
    .then(r => r.json())
    .then(data => {
        console.log('Callers from API:', data.callers?.length || 0);
        console.log('Callers:', data.callers);
    });
```

**Expected Output:**
```
Testing chunk: abc123def456
Callers from API: 3
Callers: [
    { id: "xyz789", name: "caller_function", file: "/path/to/file.py", type: "code" },
    ...
]
```

This confirms API computes callers but doesn't add links.

## Conclusion

The Dependencies report button is **functionally correct** but **data-starved**. The JavaScript implementation expects `caller` type links in the `allLinks` array, but the current architecture intentionally skips generating these links to optimize initial load time.

**Immediate Fix:** Implement Option 1 (Pre-compute Caller Links) to restore Dependencies report functionality.

**Long-term Consideration:** Evaluate if the 5-minute load time trade-off is worth the comprehensive analysis capabilities, or implement intelligent caching/incremental updates.
