"""Simple D3.js tree visualization for code graph.

Clean, minimal implementation focusing on core functionality:
- Hierarchical tree layout (linear and circular)
- Expandable/collapsible directories and files
- File expansion shows code chunks as child nodes
- Chunk selection to view content in side panel

Design Decision: Complete rewrite from scratch
Rationale: Previous implementation was 4085 lines (5x over 800-line limit)
with excessive complexity. This minimal version provides core functionality
in <450 lines while maintaining clarity and maintainability.

Node Types and Colors:
- Orange (collapsed directory) / Blue (expanded directory)
- Gray (collapsed file) / White (expanded file)
- Purple (chunk nodes) - smaller circles with purple text

Trade-offs:
- Simplicity vs Features: Removed advanced features (force-directed, filters)
- Performance vs Clarity: Straightforward DOM updates over optimized rendering
- Flexibility vs Simplicity: Fixed layouts instead of customizable options

Extension Points: Add features incrementally based on user feedback rather
than preemptive feature bloat.
"""


def get_all_scripts() -> str:
    """Generate all JavaScript for the visualization.

    Returns:
        Complete JavaScript code as a single string
    """
    return """
// ============================================================================
// GLOBAL STATE
// ============================================================================

let allNodes = [];
let allLinks = [];
let currentLayout = 'linear';  // 'linear' or 'circular'
let treeData = null;
let isViewerOpen = false;

// Visualization mode: 'tree', 'treemap', 'sunburst'
let currentVizMode = 'tree';
// Grouping mode: 'file' (directory structure) or 'ast' (by code type)
let currentGroupingMode = 'file';
// For treemap/sunburst zoom state - store ID to find node after hierarchy rebuild
let currentZoomRootId = null;
let vizHierarchy = null;

// Performance optimization: Cached hierarchies and pre-filtered data
let cachedFileHierarchy = null;
let cachedASTHierarchy = null;
let cachedChunkNodes = null;  // Pre-filtered chunk nodes
let nodeIdMap = null;  // Map for O(1) node lookups by ID
let hierarchyCacheVersion = 0;  // Increment to invalidate caches

// Initialize performance caches - called once after data load
function initializeCaches() {
    console.time('initializeCaches');

    // Pre-filter chunk nodes once (avoids repeated O(n) filtering)
    cachedChunkNodes = allNodes.filter(node => chunkTypes.includes(node.type));
    console.log(`Cached ${cachedChunkNodes.length} chunk nodes`);

    // Build node ID map for O(1) lookups
    nodeIdMap = new Map();
    allNodes.forEach(node => {
        nodeIdMap.set(node.id, node);
    });
    console.log(`Built nodeIdMap with ${nodeIdMap.size} entries`);

    // Clear hierarchy caches (will be built on demand)
    cachedFileHierarchy = null;
    cachedASTHierarchy = null;
    hierarchyCacheVersion++;

    console.timeEnd('initializeCaches');
}

// Invalidate caches when data changes
function invalidateHierarchyCaches() {
    cachedFileHierarchy = null;
    cachedASTHierarchy = null;
    hierarchyCacheVersion++;
}

// Navigation history for back/forward
let navigationHistory = [];
let navigationIndex = -1;

// Call lines visibility
let showCallLines = true;

// File filter: 'all', 'code', 'docs'
let currentFileFilter = 'all';

// Chunk types for code nodes (function, class, method, text, imports, module)
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];

// Size scaling configuration
const sizeConfig = {
    minRadius: 12,      // Minimum node radius (50% larger for readability)
    maxRadius: 24,      // Maximum node radius
    chunkMinRadius: 5,  // Minimum for small chunks (more visible size contrast)
    chunkMaxRadius: 28  // Maximum for large chunks (more visible size contrast)
};

// Dynamic dimensions that update when viewer opens/closes
function getViewportDimensions() {
    const container = document.getElementById('main-container');
    return {
        width: container.clientWidth,
        height: container.clientHeight
    };
}

const margin = {top: 40, right: 120, bottom: 20, left: 120};

// ============================================================================
// DATA LOADING
// ============================================================================

let graphStatusCheckInterval = null;

async function checkGraphStatus() {
    try {
        const response = await fetch('/api/graph-status');
        const status = await response.json();
        return status;
    } catch (error) {
        console.error('Failed to check graph status:', error);
        return { ready: false, size: 0 };
    }
}

function showLoadingIndicator(message) {
    const loadingDiv = document.getElementById('graph-loading-indicator') || createLoadingDiv();
    loadingDiv.querySelector('.loading-message').textContent = message;
    loadingDiv.style.display = 'flex';
}

function hideLoadingIndicator() {
    const loadingDiv = document.getElementById('graph-loading-indicator');
    if (loadingDiv) {
        loadingDiv.style.display = 'none';
    }
}

function createLoadingDiv() {
    const div = document.createElement('div');
    div.id = 'graph-loading-indicator';
    div.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255, 255, 255, 0.95);
        padding: 30px 50px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
        z-index: 10000;
    `;

    const spinner = document.createElement('div');
    spinner.style.cssText = `
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    `;

    const message = document.createElement('div');
    message.className = 'loading-message';
    message.style.cssText = 'color: #333; font-size: 16px; font-family: Arial, sans-serif;';
    message.textContent = 'Loading graph data...';

    div.appendChild(spinner);
    div.appendChild(message);
    document.body.appendChild(div);

    // Add spinner animation
    const style = document.createElement('style');
    style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
    document.head.appendChild(style);

    return div;
}

async function loadGraphData() {
    try {
        // Check if graph is ready
        const status = await checkGraphStatus();

        if (!status.ready) {
            console.log('Graph data not ready yet, will poll every 5 seconds...');
            showLoadingIndicator('Generating graph data... This may take a few minutes.');

            // Start polling for graph readiness
            graphStatusCheckInterval = setInterval(async () => {
                const checkStatus = await checkGraphStatus();
                if (checkStatus.ready) {
                    clearInterval(graphStatusCheckInterval);
                    console.log('Graph data is now ready, loading...');
                    showLoadingIndicator('Graph data ready! Loading visualization...');
                    await loadGraphDataActual();
                    hideLoadingIndicator();
                }
            }, 5000); // Poll every 5 seconds

            return;
        }

        // Graph is already ready, load it
        await loadGraphDataActual();
    } catch (error) {
        console.error('Failed to load graph data:', error);
        hideLoadingIndicator();
        document.body.innerHTML =
            '<div style="color: red; padding: 20px; font-family: Arial;">Error loading visualization data. Check console for details.</div>';
    }
}

async function loadGraphDataActual() {
    try {
        const response = await fetch('/api/graph');
        const data = await response.json();

        // Check if we got an error response
        if (data.error) {
            console.warn('Graph data not available yet:', data.error);
            showLoadingIndicator('Waiting for graph data...');
            return;
        }

        allNodes = data.nodes || [];
        allLinks = data.links || [];

        // Store trend data globally for visualization
        window.graphTrendData = data.trends || null;
        if (window.graphTrendData) {
            console.log(`Loaded trend data: ${window.graphTrendData.entries_count} entries`);
        }

        console.log(`Loaded ${allNodes.length} nodes and ${allLinks.length} links`);

        // Performance optimization: Initialize caches
        initializeCaches();

        // DEBUG: Log first few nodes to see actual structure
        console.log('=== SAMPLE NODE STRUCTURE ===');
        if (allNodes.length > 0) {
            console.log('First node:', JSON.stringify(allNodes[0], null, 2));
            if (allNodes.length > 1) {
                console.log('Second node:', JSON.stringify(allNodes[1], null, 2));
            }
        }

        // Count node types
        const typeCounts = {};
        allNodes.forEach(node => {
            const type = node.type || 'undefined';
            typeCounts[type] = (typeCounts[type] || 0) + 1;
        });
        console.log('Node type counts:', typeCounts);
        console.log('=== END SAMPLE NODE STRUCTURE ===');

        buildTreeStructure();
        renderVisualization();
    } catch (error) {
        console.error('Failed to load graph data:', error);
        document.body.innerHTML =
            '<div style="color: red; padding: 20px; font-family: Arial;">Error loading visualization data. Check console for details.</div>';
    }
}

// ============================================================================
// TREE STRUCTURE BUILDING
// ============================================================================

function buildTreeStructure() {
    // Include directories, files, AND chunks (function, class, method, text, imports, module)
    const treeNodes = allNodes.filter(node => {
        const type = node.type;
        return type === 'directory' || type === 'file' || chunkTypes.includes(type);
    });

    console.log(`Filtered to ${treeNodes.length} tree nodes (directories, files, and chunks)`);

    // Count node types for debugging
    const dirCount = treeNodes.filter(n => n.type === 'directory').length;
    const fileCount = treeNodes.filter(n => n.type === 'file').length;
    const chunkCount = treeNodes.filter(n => chunkTypes.includes(n.type)).length;
    console.log(`Node breakdown: ${dirCount} directories, ${fileCount} files, ${chunkCount} chunks`);

    // Create lookup maps
    const nodeMap = new Map();
    treeNodes.forEach(node => {
        nodeMap.set(node.id, {
            ...node,
            children: []
        });
    });

    // Build parent-child relationships
    const parentMap = new Map();

    // DEBUG: Analyze link structure
    console.log('=== LINK STRUCTURE DEBUG ===');
    console.log(`Total links: ${allLinks.length}`);

    // Get unique link types (handle undefined)
    const linkTypes = [...new Set(allLinks.map(l => l.type || 'undefined'))];
    console.log('Link types found:', linkTypes);

    // Count links by type
    const linkTypeCounts = {};
    allLinks.forEach(link => {
        const type = link.type || 'undefined';
        linkTypeCounts[type] = (linkTypeCounts[type] || 0) + 1;
    });
    console.log('Link type counts:', linkTypeCounts);

    // Sample first few links
    console.log('Sample links (first 5):');
    allLinks.slice(0, 5).forEach((link, i) => {
        console.log(`  Link ${i}:`, JSON.stringify(link, null, 2));
    });

    // Check if links have properties we expect
    if (allLinks.length > 0) {
        const firstLink = allLinks[0];
        console.log('Link properties:', Object.keys(firstLink));
    }
    console.log('=== END LINK STRUCTURE DEBUG ===');

    // Build parent-child relationships from links
    // Process all containment and hierarchy links to establish the tree structure
    console.log('=== BUILDING TREE RELATIONSHIPS ===');

    let relationshipsProcessed = {
        dir_hierarchy: 0,
        dir_containment: 0,
        file_containment: 0,
        chunk_hierarchy: 0  // chunk_hierarchy links = class -> method
    };

    let relationshipsMatched = {
        dir_hierarchy: 0,
        dir_containment: 0,
        file_containment: 0,
        chunk_hierarchy: 0
    };

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
            // chunk_hierarchy links are chunk-to-chunk (e.g., class -> method)
            category = 'chunk_hierarchy';
        } else {
            // Skip semantic, caller, undefined, and other non-hierarchical links
            // This includes links without a 'type' field (e.g., subproject links)
            return;
        }

        relationshipsProcessed[category]++;

        // Get parent and child nodes from the map
        const parentNode = nodeMap.get(link.source);
        const childNode = nodeMap.get(link.target);

        // Both nodes must exist in our tree node set
        if (!parentNode || !childNode) {
            if (relationshipsProcessed[category] <= 3) {  // Log first few misses
                console.log(`${category} link skipped - parent: ${link.source} (exists: ${!!parentNode}), child: ${link.target} (exists: ${!!childNode})`);
            }
            return;
        }

        // Establish parent-child relationship
        // Add child to parent's children array
        parentNode.children.push(childNode);

        // Record the parent in parentMap (used to identify root nodes)
        parentMap.set(link.target, link.source);

        relationshipsMatched[category]++;
    });

    console.log('Relationship processing summary:');
    console.log(`  dir_hierarchy: ${relationshipsMatched.dir_hierarchy}/${relationshipsProcessed.dir_hierarchy} matched`);
    console.log(`  dir_containment: ${relationshipsMatched.dir_containment}/${relationshipsProcessed.dir_containment} matched`);
    console.log(`  file_containment: ${relationshipsMatched.file_containment}/${relationshipsProcessed.file_containment} matched`);
    console.log(`  chunk_hierarchy (class→method): ${relationshipsMatched.chunk_hierarchy}/${relationshipsProcessed.chunk_hierarchy} matched`);
    console.log(`  Total parent-child links: ${parentMap.size}`);
    console.log('=== END TREE RELATIONSHIPS ===');

    // Find root nodes (nodes with no parents)
    // IMPORTANT: Exclude chunk types from roots - they should only appear as children of files
    // Orphaned chunks (without file_containment links) are excluded from the tree
    const rootNodes = treeNodes
        .filter(node => !parentMap.has(node.id))
        .filter(node => !chunkTypes.includes(node.type))  // Exclude orphaned chunks
        .map(node => nodeMap.get(node.id))
        .filter(node => node !== undefined);

    console.log('=== ROOT NODE ANALYSIS ===');
    console.log(`Found ${rootNodes.length} root nodes (directories and files only)`);

    // DEBUG: Count root node types
    const rootTypeCounts = {};
    rootNodes.forEach(node => {
        const type = node.type || 'undefined';
        rootTypeCounts[type] = (rootTypeCounts[type] || 0) + 1;
    });
    console.log('Root node type breakdown:', rootTypeCounts);

    // If we have chunk nodes as roots, something went wrong
    const chunkRoots = rootNodes.filter(n => chunkTypes.includes(n.type)).length;
    if (chunkRoots > 0) {
        console.warn(`WARNING: ${chunkRoots} chunk nodes are roots - they should be children of files!`);
    }

    // If we have file nodes as roots (except for top-level files), might be missing dir_containment
    const fileRoots = rootNodes.filter(n => n.type === 'file').length;
    if (fileRoots > 0) {
        console.log(`INFO: ${fileRoots} file nodes are roots (this is normal for files not in subdirectories)`);
    }

    console.log('=== END ROOT NODE ANALYSIS ===');

    // Create virtual root if multiple roots
    if (rootNodes.length === 0) {
        console.error('No root nodes found!');
        treeData = {name: 'Empty', id: 'root', type: 'directory', children: []};
    } else if (rootNodes.length === 1) {
        treeData = rootNodes[0];
    } else {
        treeData = {
            name: 'Project Root',
            id: 'virtual-root',
            type: 'directory',
            children: rootNodes
        };
    }

    // Collapse single-child chains to make the tree more compact
    // - Directory with single directory child: src -> mcp_vector_search becomes "src/mcp_vector_search"
    // - File with single chunk child: promote the chunk's children to the file level
    function collapseSingleChildChains(node) {
        if (!node || !node.children) return;

        // First, recursively process all children
        node.children.forEach(child => collapseSingleChildChains(child));

        // Case 1: Directory with single directory child - combine names
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

        // Case 2: File with single chunk child - promote chunk's children to file
        // This handles files where there's just one L1 (e.g., imports or a single class)
        if (node.type === 'file' && node.children && node.children.length === 1) {
            const onlyChild = node.children[0];
            if (chunkTypes.includes(onlyChild.type)) {
                // If the chunk has children, promote them to the file level
                const chunkChildren = onlyChild.children || onlyChild._children || [];
                if (chunkChildren.length > 0) {
                    console.log(`Promoting ${chunkChildren.length} children from ${onlyChild.type} to file ${node.name}`);
                    // Replace the single chunk with its children
                    node.children = chunkChildren;
                    // Store info about the collapsed chunk (include ALL relevant properties)
                    node.collapsed_chunk = {
                        type: onlyChild.type,
                        name: onlyChild.name,
                        id: onlyChild.id,
                        content: onlyChild.content,
                        docstring: onlyChild.docstring,
                        start_line: onlyChild.start_line,
                        end_line: onlyChild.end_line,
                        file_path: onlyChild.file_path,
                        language: onlyChild.language,
                        complexity: onlyChild.complexity
                    };
                } else {
                    // Collapse file+chunk into combined name (like directory chains)
                    console.log(`Collapsing file+chunk: ${node.name}/${onlyChild.name}`);
                    node.name = `${node.name}/${onlyChild.name}`;
                    node.children = null;  // Remove chunk child - now a leaf node
                    node._children = null;
                    node.collapsed_ids = node.collapsed_ids || [node.id];
                    node.collapsed_ids.push(onlyChild.id);

                    // Store chunk data for display when clicked
                    node.collapsed_chunk = {
                        type: onlyChild.type,
                        name: onlyChild.name,
                        id: onlyChild.id,
                        content: onlyChild.content,
                        start_line: onlyChild.start_line,
                        end_line: onlyChild.end_line,
                        complexity: onlyChild.complexity
                    };
                }
            }
        }
    }

    // Apply single-child chain collapsing to all root children
    console.log('=== COLLAPSING SINGLE-CHILD CHAINS ===');
    if (treeData.children) {
        treeData.children.forEach(child => collapseSingleChildChains(child));
    }
    console.log('=== END COLLAPSING SINGLE-CHILD CHAINS ===');

    // Collapse all directories and files by default
    function collapseAll(node) {
        if (node.children && node.children.length > 0) {
            // First, recursively process all descendants
            node.children.forEach(child => collapseAll(child));

            // Then collapse this node (move children to _children)
            node._children = node.children;
            node.children = null;
        }
    }

    // Collapse ALL nodes except the root itself
    // This ensures only the root node is visible initially, all children are collapsed
    if (treeData.children) {
        treeData.children.forEach(child => collapseAll(child));
    }

    console.log('Tree structure built with all directories and files collapsed');

    // Calculate line counts for all nodes (for proportional node rendering)
    allLineCounts = [];  // Reset for fresh calculation
    calculateNodeSizes(treeData);
    calculatePercentiles();  // Calculate 20th/80th percentile thresholds
    console.log('Node sizes calculated with percentile-based sizing');

    // DEBUG: Check a few file nodes to see if they have chunks in _children
    console.log('=== POST-COLLAPSE FILE CHECK ===');
    let filesChecked = 0;
    let filesWithChunks = 0;

    function checkFilesRecursive(node) {
        if (node.type === 'file') {
            filesChecked++;
            const chunkCount = (node._children || []).length;
            if (chunkCount > 0) {
                filesWithChunks++;
                console.log(`File ${node.name} has ${chunkCount} chunks in _children`);
            }
        }

        // Check both visible and hidden children
        const childrenToCheck = node.children || node._children || [];
        childrenToCheck.forEach(child => checkFilesRecursive(child));
    }

    checkFilesRecursive(treeData);
    console.log(`Checked ${filesChecked} files, ${filesWithChunks} have chunks`);
    console.log('=== END POST-COLLAPSE FILE CHECK ===');
}

// ============================================================================
// NODE SIZE CALCULATION
// ============================================================================

// Global variables for size scaling - now tracking line counts
let globalMinLines = Infinity;
let globalMaxLines = 0;
let allLineCounts = [];  // Collect all line counts for percentile calculation

function calculateNodeSizes(node) {
    if (!node) return 0;

    // For chunks: use actual line count (primary metric)
    // Falls back to content-based estimate only if line numbers unavailable
    if (chunkTypes.includes(node.type)) {
        // Primary: use actual line span from start_line/end_line
        // This ensures visual correlation with displayed line ranges
        const contentLength = node.content ? node.content.length : 0;
        const lineCount = (node.start_line && node.end_line)
            ? node.end_line - node.start_line + 1
            : Math.max(1, Math.floor(contentLength / 40));  // Fallback: estimate ~40 chars per line

        // Use actual line count for sizing (NOT content length)
        // This prevents inversion where sparse 101-line files appear smaller than dense 3-line files
        node._lineCount = lineCount;
        allLineCounts.push(lineCount);

        if (lineCount > 0) {
            globalMinLines = Math.min(globalMinLines, lineCount);
            globalMaxLines = Math.max(globalMaxLines, lineCount);
        }
        return lineCount;
    }

    // For files and directories: sum of children line counts
    const children = node.children || node._children || [];
    let totalLines = 0;

    children.forEach(child => {
        totalLines += calculateNodeSizes(child);
    });

    // Handle collapsed file+chunk: use the collapsed chunk's line count
    if (node.type === 'file' && node.collapsed_chunk) {
        const cc = node.collapsed_chunk;
        if (cc.start_line && cc.end_line) {
            totalLines = cc.end_line - cc.start_line + 1;
        } else if (cc.content) {
            totalLines = Math.max(1, Math.floor(cc.content.length / 40));
        }
    }

    node._lineCount = totalLines || 1;  // Minimum 1 for empty dirs/files

    // DON'T add files/directories to allLineCounts - they skew percentiles
    // Only chunks should affect percentile calculation since only chunks use percentile sizing
    // (Files and directories use separate minRadius/maxRadius, not chunkMinRadius/chunkMaxRadius)

    if (node._lineCount > 0) {
        globalMinLines = Math.min(globalMinLines, node._lineCount);
        globalMaxLines = Math.max(globalMaxLines, node._lineCount);
    }

    return node._lineCount;
}

// Calculate percentile thresholds after all nodes are processed
let percentile20 = 0;
let percentile80 = 0;

function calculatePercentiles() {
    if (allLineCounts.length === 0) return;

    const sorted = [...allLineCounts].sort((a, b) => a - b);
    const p20Index = Math.floor(sorted.length * 0.20);
    const p80Index = Math.floor(sorted.length * 0.80);

    percentile20 = sorted[p20Index] || 1;
    percentile80 = sorted[p80Index] || sorted[sorted.length - 1] || 1;

    console.log(`Line count percentiles (chunks only): 20th=${percentile20}, 80th=${percentile80}, range=${percentile80 - percentile20}`);
    console.log(`Total chunks: ${allLineCounts.length}, min=${globalMinLines}, max=${globalMaxLines}`);
}

// Count external calls for a node
function getExternalCallCounts(nodeData) {
    if (!nodeData.id) return { inbound: 0, outbound: 0, inboundNodes: [], outboundNodes: [] };

    const nodeFilePath = nodeData.file_path;
    const inboundNodes = [];  // Array of {id, name, file_path}
    const outboundNodes = []; // Array of {id, name, file_path}

    // Use a Set to deduplicate by source/target node
    const inboundSeen = new Set();
    const outboundSeen = new Set();

    allLinks.forEach(link => {
        if (link.type === 'caller') {
            if (link.target === nodeData.id) {
                // Something calls this node
                const callerNode = allNodes.find(n => n.id === link.source);
                if (callerNode && callerNode.file_path !== nodeFilePath && !inboundSeen.has(callerNode.id)) {
                    inboundSeen.add(callerNode.id);
                    inboundNodes.push({ id: callerNode.id, name: callerNode.name, file_path: callerNode.file_path });
                }
            }
            if (link.source === nodeData.id) {
                // This node calls something
                const calleeNode = allNodes.find(n => n.id === link.target);
                if (calleeNode && calleeNode.file_path !== nodeFilePath && !outboundSeen.has(calleeNode.id)) {
                    outboundSeen.add(calleeNode.id);
                    outboundNodes.push({ id: calleeNode.id, name: calleeNode.name, file_path: calleeNode.file_path });
                }
            }
        }
    });

    return {
        inbound: inboundNodes.length,
        outbound: outboundNodes.length,
        inboundNodes,
        outboundNodes
    };
}

// Store external call data for line drawing
let externalCallData = [];

function collectExternalCallData() {
    externalCallData = [];

    allNodes.forEach(nodeData => {
        if (!chunkTypes.includes(nodeData.type)) return;

        const counts = getExternalCallCounts(nodeData);
        if (counts.inbound > 0 || counts.outbound > 0) {
            externalCallData.push({
                nodeId: nodeData.id,
                inboundNodes: counts.inboundNodes,
                outboundNodes: counts.outboundNodes
            });
        }
    });
}

function drawExternalCallLines(svg, root) {
    // Remove existing external call lines
    svg.selectAll('.external-call-line').remove();

    // Build a map of node positions from the tree (visible nodes only)
    const nodePositions = new Map();
    root.descendants().forEach(d => {
        nodePositions.set(d.data.id, { x: d.x, y: d.y, node: d });
    });

    // Build a map from node ID to tree node (for parent traversal)
    const treeNodeMap = new Map();
    root.descendants().forEach(d => {
        treeNodeMap.set(d.data.id, d);
    });

    // Helper: Find position for a node, falling back to visible ancestors
    // If the target node is not visible (collapsed), find its closest visible ancestor
    function getPositionWithFallback(nodeId) {
        // First check if node is directly visible
        if (nodePositions.has(nodeId)) {
            return nodePositions.get(nodeId);
        }

        // Node not visible - try to find via tree traversal
        // Strategy 1: Find by file_path matching in visible nodes
        const targetNode = allNodes.find(n => n.id === nodeId);
        if (!targetNode) {
            return null;
        }

        // Strategy 2: Look for the file that contains this chunk
        if (targetNode.file_path) {
            // Look for visible file nodes that match this path
            for (const [id, pos] of nodePositions) {
                const visibleNode = allNodes.find(n => n.id === id);
                if (visibleNode) {
                    // Check if this is the file containing our chunk
                    if (visibleNode.type === 'file' &&
                        (visibleNode.path === targetNode.file_path ||
                         visibleNode.file_path === targetNode.file_path ||
                         visibleNode.name === targetNode.file_path.split('/').pop())) {
                        return pos;
                    }
                }
            }

            // Strategy 3: Look for directory containing the file
            const pathParts = targetNode.file_path.split('/');
            // Go from most specific (file's directory) to least specific (root)
            for (let i = pathParts.length - 1; i >= 0; i--) {
                const dirName = pathParts[i];
                if (!dirName) continue;

                // Find a visible directory with this name
                for (const [id, pos] of nodePositions) {
                    const visibleNode = allNodes.find(n => n.id === id);
                    if (visibleNode && visibleNode.type === 'directory' && visibleNode.name === dirName) {
                        return pos;
                    }
                }
            }
        }

        return null;
    }

    // Create a group for external call lines (behind nodes)
    let lineGroup = svg.select('.external-lines-group');
    if (lineGroup.empty()) {
        lineGroup = svg.insert('g', ':first-child')
            .attr('class', 'external-lines-group');
    }

    // Respect the toggle state
    lineGroup.style('display', showCallLines ? 'block' : 'none');

    console.log(`[CallLines] Drawing lines for ${externalCallData.length} nodes with external calls`);

    let linesDrawn = 0;
    externalCallData.forEach(data => {
        const sourcePos = getPositionWithFallback(data.nodeId);
        if (!sourcePos) {
            console.log(`[CallLines] No source position for ${data.nodeId}`);
            return;
        }

        // Draw lines to inbound nodes (callers) - dashed blue (fainter)
        data.inboundNodes.forEach(caller => {
            const targetPos = getPositionWithFallback(caller.id);
            if (targetPos) {
                lineGroup.append('path')
                    .attr('class', 'external-call-line inbound-line')
                    .attr('d', `M${targetPos.y},${targetPos.x} C${(targetPos.y + sourcePos.y)/2},${targetPos.x} ${(targetPos.y + sourcePos.y)/2},${sourcePos.x} ${sourcePos.y},${sourcePos.x}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#58a6ff')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.35)
                    .attr('pointer-events', 'none');
                linesDrawn++;
            }
        });

        // Draw lines to outbound nodes (callees) - dashed orange (fainter)
        data.outboundNodes.forEach(callee => {
            const targetPos = getPositionWithFallback(callee.id);
            if (targetPos) {
                lineGroup.append('path')
                    .attr('class', 'external-call-line outbound-line')
                    .attr('d', `M${sourcePos.y},${sourcePos.x} C${(sourcePos.y + targetPos.y)/2},${sourcePos.x} ${(sourcePos.y + targetPos.y)/2},${targetPos.x} ${targetPos.y},${targetPos.x}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#f0883e')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.35)
                    .attr('pointer-events', 'none');
                linesDrawn++;
            }
        });
    });

    console.log(`[CallLines] Drew ${linesDrawn} call lines`);
}

// Draw external call lines for CIRCULAR layout
// Converts polar coordinates (angle, radius) to Cartesian (x, y)
function drawExternalCallLinesCircular(svg, root) {
    // Remove existing external call lines
    svg.selectAll('.external-call-line').remove();

    // Helper: Convert polar to Cartesian coordinates
    // In D3 radial layout: d.x = angle (radians), d.y = radius
    function polarToCartesian(angle, radius) {
        return {
            x: radius * Math.cos(angle - Math.PI / 2),
            y: radius * Math.sin(angle - Math.PI / 2)
        };
    }

    // Build a map of node positions from the tree (visible nodes only)
    const nodePositions = new Map();
    root.descendants().forEach(d => {
        const cartesian = polarToCartesian(d.x, d.y);
        nodePositions.set(d.data.id, { x: cartesian.x, y: cartesian.y, angle: d.x, radius: d.y, node: d });
    });

    // Helper: Find position for a node, falling back to visible ancestors
    function getPositionWithFallback(nodeId) {
        // First check if node is directly visible
        if (nodePositions.has(nodeId)) {
            return nodePositions.get(nodeId);
        }

        // Node not visible - try to find via tree traversal
        const targetNode = allNodes.find(n => n.id === nodeId);
        if (!targetNode) {
            return null;
        }

        // Look for the file that contains this chunk
        if (targetNode.file_path) {
            // Look for visible file nodes that match this path
            for (const [id, pos] of nodePositions) {
                const visibleNode = allNodes.find(n => n.id === id);
                if (visibleNode) {
                    if (visibleNode.type === 'file' &&
                        (visibleNode.path === targetNode.file_path ||
                         visibleNode.file_path === targetNode.file_path ||
                         visibleNode.name === targetNode.file_path.split('/').pop())) {
                        return pos;
                    }
                }
            }

            // Look for directory containing the file
            const pathParts = targetNode.file_path.split('/');
            for (let i = pathParts.length - 1; i >= 0; i--) {
                const dirName = pathParts[i];
                if (!dirName) continue;

                for (const [id, pos] of nodePositions) {
                    const visibleNode = allNodes.find(n => n.id === id);
                    if (visibleNode && visibleNode.type === 'directory' && visibleNode.name === dirName) {
                        return pos;
                    }
                }
            }
        }

        return null;
    }

    // Create a group for external call lines (behind nodes)
    let lineGroup = svg.select('.external-lines-group');
    if (lineGroup.empty()) {
        lineGroup = svg.insert('g', ':first-child')
            .attr('class', 'external-lines-group');
    }

    // Respect the toggle state
    lineGroup.style('display', showCallLines ? 'block' : 'none');

    console.log(`[CallLines Circular] Drawing lines for ${externalCallData.length} nodes with external calls`);

    let linesDrawn = 0;
    externalCallData.forEach(data => {
        const sourcePos = getPositionWithFallback(data.nodeId);
        if (!sourcePos) {
            return;
        }

        // Draw lines to inbound nodes (callers) - dashed blue (fainter)
        data.inboundNodes.forEach(caller => {
            const targetPos = getPositionWithFallback(caller.id);
            if (targetPos) {
                // Use quadratic bezier curves that go through the center for circular layout
                const midX = (sourcePos.x + targetPos.x) / 2 * 0.3;
                const midY = (sourcePos.y + targetPos.y) / 2 * 0.3;

                lineGroup.append('path')
                    .attr('class', 'external-call-line inbound-line')
                    .attr('d', `M${targetPos.x},${targetPos.y} Q${midX},${midY} ${sourcePos.x},${sourcePos.y}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#58a6ff')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.35)
                    .attr('pointer-events', 'none');
                linesDrawn++;
            }
        });

        // Draw lines to outbound nodes (callees) - dashed orange (fainter)
        data.outboundNodes.forEach(callee => {
            const targetPos = getPositionWithFallback(callee.id);
            if (targetPos) {
                // Use quadratic bezier curves that go through the center for circular layout
                const midX = (sourcePos.x + targetPos.x) / 2 * 0.3;
                const midY = (sourcePos.y + targetPos.y) / 2 * 0.3;

                lineGroup.append('path')
                    .attr('class', 'external-call-line outbound-line')
                    .attr('d', `M${sourcePos.x},${sourcePos.y} Q${midX},${midY} ${targetPos.x},${targetPos.y}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#f0883e')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.35)
                    .attr('pointer-events', 'none');
                linesDrawn++;
            }
        });
    });

    console.log(`[CallLines Circular] Drew ${linesDrawn} call lines`);
}

// Get color based on complexity (darker = more complex)
// Uses HSL color model for smooth gradients
function getComplexityColor(d, baseHue) {
    const nodeData = d.data;
    const complexity = nodeData.complexity;

    // If no complexity data, return a default based on type
    if (complexity === undefined || complexity === null) {
        // Default colors for non-complex nodes
        if (nodeData.type === 'directory') {
            return nodeData._children ? '#f39c12' : '#3498db';  // Orange/Blue
        } else if (nodeData.type === 'file') {
            return nodeData._children ? '#95a5a6' : '#ecf0f1';  // Gray/White
        } else if (chunkTypes.includes(nodeData.type)) {
            return '#9b59b6';  // Default purple
        }
        return '#95a5a6';
    }

    // Complexity ranges: 0-5 (low), 5-10 (medium), 10-20 (high), 20+ (very high)
    // Map to lightness: 70% (light) to 30% (dark)
    const maxComplexity = 25;  // Cap for scaling
    const normalizedComplexity = Math.min(complexity, maxComplexity) / maxComplexity;

    // Lightness goes from 65% (low complexity) to 35% (high complexity)
    const lightness = 65 - (normalizedComplexity * 30);

    // Saturation increases slightly with complexity (60% to 80%)
    const saturation = 60 + (normalizedComplexity * 20);

    return `hsl(${baseHue}, ${saturation}%, ${lightness}%)`;
}

// Get node fill color with complexity shading
function getNodeFillColor(d) {
    const nodeData = d.data;

    if (nodeData.type === 'directory') {
        // Orange (30°) if collapsed, Blue (210°) if expanded
        const hue = nodeData._children ? 30 : 210;
        // Directories aggregate complexity from children
        const avgComplexity = calculateAverageComplexity(nodeData);
        if (avgComplexity > 0) {
            const lightness = 55 - (Math.min(avgComplexity, 15) / 15) * 20;
            return `hsl(${hue}, 70%, ${lightness}%)`;
        }
        return nodeData._children ? '#f39c12' : '#3498db';
    } else if (nodeData.type === 'file') {
        // Gray files, but show complexity if available
        const avgComplexity = calculateAverageComplexity(nodeData);
        if (avgComplexity > 0) {
            // Gray hue (0° with 0 saturation) to slight red tint for complexity
            const saturation = Math.min(avgComplexity, 15) * 2;  // 0-30%
            const lightness = 70 - (Math.min(avgComplexity, 15) / 15) * 25;
            return `hsl(0, ${saturation}%, ${lightness}%)`;
        }
        return nodeData._children ? '#95a5a6' : '#ecf0f1';
    } else if (chunkTypes.includes(nodeData.type)) {
        // Purple (280°) for chunks, darker with higher complexity
        return getComplexityColor(d, 280);
    }

    return '#95a5a6';
}

// Calculate average complexity for a node (recursively for dirs/files)
function calculateAverageComplexity(node) {
    if (chunkTypes.includes(node.type)) {
        return node.complexity || 0;
    }

    const children = node.children || node._children || [];
    if (children.length === 0) return 0;

    let totalComplexity = 0;
    let count = 0;

    children.forEach(child => {
        if (chunkTypes.includes(child.type) && child.complexity) {
            totalComplexity += child.complexity;
            count++;
        } else {
            const childAvg = calculateAverageComplexity(child);
            if (childAvg > 0) {
                totalComplexity += childAvg;
                count++;
            }
        }
    });

    return count > 0 ? totalComplexity / count : 0;
}

// Get stroke color based on complexity - red outline for high complexity
function getNodeStrokeColor(d) {
    const nodeData = d.data;

    // Only chunks have direct complexity
    if (chunkTypes.includes(nodeData.type)) {
        const complexity = nodeData.complexity || 0;
        // Complexity thresholds:
        // 0-5: white (simple)
        // 5-10: orange (moderate)
        // 10+: red (complex)
        if (complexity >= 10) {
            return '#e74c3c';  // Red for high complexity
        } else if (complexity >= 5) {
            return '#f39c12';  // Orange for moderate
        }
        return '#fff';  // White for low complexity
    }

    // Files and directories: check average complexity of children
    if (nodeData.type === 'file' || nodeData.type === 'directory') {
        const avgComplexity = calculateAverageComplexity(nodeData);
        if (avgComplexity >= 10) {
            return '#e74c3c';  // Red
        } else if (avgComplexity >= 5) {
            return '#f39c12';  // Orange
        }
    }

    return '#fff';  // Default white
}

// Get stroke width based on complexity - thicker for high complexity
function getNodeStrokeWidth(d) {
    const nodeData = d.data;
    const complexity = chunkTypes.includes(nodeData.type)
        ? (nodeData.complexity || 0)
        : calculateAverageComplexity(nodeData);

    if (complexity >= 10) return 3;  // Thick red outline
    if (complexity >= 5) return 2.5;  // Medium orange outline
    return 2;  // Default
}

function getNodeRadius(d) {
    const nodeData = d.data;

    // Size configuration based on node type
    const dirMinRadius = 8;   // Min for directories
    const dirMaxRadius = 40;  // Max for directories
    const fileMinRadius = 6;  // Min for files
    const fileMaxRadius = 30; // Max for files
    const chunkMinRadius = sizeConfig.chunkMinRadius;  // From config
    const chunkMaxRadius = sizeConfig.chunkMaxRadius;  // From config

    // Directory nodes: size by file_count (logarithmic scale)
    if (nodeData.type === 'directory') {
        const fileCount = nodeData.file_count || 0;
        if (fileCount === 0) return dirMinRadius;

        // Logarithmic scale: log2(fileCount + 1) for smooth scaling
        // +1 to avoid log(0), gives range [1, log2(max_files+1)]
        const logCount = Math.log2(fileCount + 1);
        const maxLogCount = Math.log2(100 + 1);  // Assume max ~100 files per dir
        const normalized = Math.min(logCount / maxLogCount, 1.0);

        return dirMinRadius + (normalized * (dirMaxRadius - dirMinRadius));
    }

    // File nodes: size by total lines of code (sum of all chunks)
    if (nodeData.type === 'file') {
        const lineCount = nodeData._lineCount || 1;

        // Collapsed file+chunk: use chunk sizing (since it's really a chunk)
        if (nodeData.collapsed_chunk) {
            const minLines = 5;
            const maxLines = 150;

            let normalized;
            if (lineCount <= minLines) {
                normalized = 0;
            } else if (lineCount >= maxLines) {
                normalized = 1;
            } else {
                const logMin = Math.log(minLines);
                const logMax = Math.log(maxLines);
                const logCount = Math.log(lineCount);
                normalized = (logCount - logMin) / (logMax - logMin);
            }
            return chunkMinRadius + (normalized * (chunkMaxRadius - chunkMinRadius));
        }

        // Regular files: linear scaling based on total lines
        const minFileLines = 5;
        const maxFileLines = 300;

        let normalized;
        if (lineCount <= minFileLines) {
            normalized = 0;
        } else if (lineCount >= maxFileLines) {
            normalized = 1;
        } else {
            normalized = (lineCount - minFileLines) / (maxFileLines - minFileLines);
        }

        return fileMinRadius + (normalized * (fileMaxRadius - fileMinRadius));
    }

    // Chunk nodes: size by lines of code (absolute thresholds, not percentiles)
    // This ensures 330-line functions are ALWAYS big, regardless of codebase distribution
    if (chunkTypes.includes(nodeData.type)) {
        const lineCount = nodeData._lineCount || 1;

        // Absolute thresholds for intuitive sizing:
        // - 1-10 lines: small (imports, constants, simple functions)
        // - 10-50 lines: medium (typical functions)
        // - 50-150 lines: large (complex functions)
        // - 150+ lines: maximum (very large functions/classes)
        const minLines = 5;
        const maxLines = 150;  // Anything over 150 lines gets max size

        let normalized;
        if (lineCount <= minLines) {
            normalized = 0;
        } else if (lineCount >= maxLines) {
            normalized = 1;
        } else {
            // Logarithmic scaling for better visual distribution
            const logMin = Math.log(minLines);
            const logMax = Math.log(maxLines);
            const logCount = Math.log(lineCount);
            normalized = (logCount - logMin) / (logMax - logMin);
        }

        return chunkMinRadius + (normalized * (chunkMaxRadius - chunkMinRadius));
    }

    // Default fallback for other node types
    return sizeConfig.minRadius;
}

// ============================================================================
// VISUALIZATION RENDERING
// ============================================================================

function renderVisualization() {
    console.log('=== RENDER VISUALIZATION ===');
    console.log(`Current viz mode: ${currentVizMode}`);
    console.log(`Current layout: ${currentLayout}`);
    console.log(`Current grouping: ${currentGroupingMode}`);
    console.log(`Tree data exists: ${treeData !== null}`);
    if (treeData) {
        console.log(`Root node: ${treeData.name}, children: ${(treeData.children || []).length}, _children: ${(treeData._children || []).length}`);
    }

    // Clear existing content
    const graphElement = d3.select('#graph');
    console.log(`Graph element found: ${!graphElement.empty()}`);
    graphElement.selectAll('*').remove();

    // Dispatch to appropriate visualization
    if (currentVizMode === 'treemap') {
        console.log('Calling renderTreemap()...');
        renderTreemap();
    } else if (currentVizMode === 'sunburst') {
        console.log('Calling renderSunburst()...');
        renderSunburst();
    } else {
        // Default tree mode
        if (currentLayout === 'linear') {
            console.log('Calling renderLinearTree()...');
            renderLinearTree();
        } else {
            console.log('Calling renderCircularTree()...');
            renderCircularTree();
        }
    }
    console.log('=== END RENDER VISUALIZATION ===');
}

// ============================================================================
// LINEAR TREE LAYOUT
// ============================================================================

function renderLinearTree() {
    console.log('=== RENDER LINEAR TREE ===');
    const { width, height } = getViewportDimensions();
    console.log(`Viewport dimensions: ${width}x${height}`);

    const svg = d3.select('#graph')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create tree layout
    // For horizontal tree: size is [height, width] where height controls vertical spread
    const treeLayout = d3.tree()
        .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);

    console.log(`Tree layout size: ${height - margin.top - margin.bottom} x ${width - margin.left - margin.right}`);

    // Create hierarchy from tree data
    // D3 hierarchy automatically respects children vs _children
    console.log('Creating D3 hierarchy...');

    // DEBUG: Check if treeData children have content property BEFORE D3 processes them
    console.log('=== PRE-D3 HIERARCHY DEBUG ===');
    if (treeData.children && treeData.children.length > 0) {
        const firstChild = treeData.children[0];
        console.log('First root child:', firstChild.name, 'type:', firstChild.type);
        console.log('First child keys:', Object.keys(firstChild));
        console.log('First child has content:', 'content' in firstChild);

        // Find a chunk node in the tree
        function findFirstChunk(node) {
            if (chunkTypes.includes(node.type)) {
                return node;
            }
            if (node.children) {
                for (const child of node.children) {
                    const found = findFirstChunk(child);
                    if (found) return found;
                }
            }
            if (node._children) {
                for (const child of node._children) {
                    const found = findFirstChunk(child);
                    if (found) return found;
                }
            }
            return null;
        }

        const sampleChunk = findFirstChunk(treeData);
        if (sampleChunk) {
            console.log('Sample chunk node BEFORE D3:', sampleChunk.name, 'type:', sampleChunk.type);
            console.log('Sample chunk keys:', Object.keys(sampleChunk));
            console.log('Sample chunk has content:', 'content' in sampleChunk);
            console.log('Sample chunk content length:', sampleChunk.content ? sampleChunk.content.length : 0);
        }
    }
    console.log('=== END PRE-D3 HIERARCHY DEBUG ===');

    const root = d3.hierarchy(treeData, d => d.children);
    console.log(`Hierarchy created: ${root.descendants().length} nodes`);

    // DEBUG: Check if content is preserved AFTER D3 processes them
    console.log('=== POST-D3 HIERARCHY DEBUG ===');
    const debugDescendants = root.descendants();
    const chunkDescendants = debugDescendants.filter(d => chunkTypes.includes(d.data.type));
    console.log(`Found ${chunkDescendants.length} chunk nodes in D3 hierarchy`);
    if (chunkDescendants.length > 0) {
        const firstChunkD3 = chunkDescendants[0];
        console.log('First chunk in D3 hierarchy:', firstChunkD3.data.name, 'type:', firstChunkD3.data.type);
        console.log('First chunk d.data keys:', Object.keys(firstChunkD3.data));
        console.log('First chunk has content in d.data:', 'content' in firstChunkD3.data);
        console.log('First chunk content length:', firstChunkD3.data.content ? firstChunkD3.data.content.length : 0);
    }
    console.log('=== END POST-D3 HIERARCHY DEBUG ===');

    // Apply tree layout
    console.log('Applying tree layout...');
    treeLayout(root);
    console.log('Tree layout applied');

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 3])
        .on('zoom', (event) => {
            g.attr('transform', `translate(${margin.left},${margin.top}) ${event.transform}`);
        });

    svg.call(zoom);

    // Draw links
    const links = root.links();
    console.log(`Drawing ${links.length} links`);
    g.selectAll('.link')
        .data(links)
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x))
        .attr('fill', 'none')
        .attr('stroke', '#ccc')
        .attr('stroke-width', 1.5);

    // Draw nodes
    const descendants = root.descendants();
    console.log(`Drawing ${descendants.length} nodes`);
    const nodes = g.selectAll('.node')
        .data(descendants)
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.y},${d.x})`)
        .on('click', handleNodeClick)
        .style('cursor', 'pointer');

    console.log(`Created ${nodes.size()} node elements`);

    // Node circles - sized proportionally to content, colored by complexity
    nodes.append('circle')
        .attr('r', d => getNodeRadius(d))  // Dynamic size based on content
        .attr('fill', d => getNodeFillColor(d))  // Complexity-based coloring
        .attr('stroke', d => getNodeStrokeColor(d))  // Red/orange for high complexity
        .attr('stroke-width', d => getNodeStrokeWidth(d))
        .attr('class', d => {
            // Add complexity grade class if available
            const grade = d.data.complexity_grade || '';
            const hasSmells = (d.data.smell_count && d.data.smell_count > 0) || (d.data.smells && d.data.smells.length > 0);
            const classes = [];
            if (grade) classes.push(`grade-${grade}`);
            if (hasSmells) classes.push('has-smells');
            return classes.join(' ');
        });

    // Add external call arrow indicators (only for chunk nodes)
    nodes.each(function(d) {
        const node = d3.select(this);
        const nodeData = d.data;

        // Only add indicators for code chunks (functions, classes, methods)
        if (!chunkTypes.includes(nodeData.type)) return;

        const counts = getExternalCallCounts(nodeData);
        const radius = getNodeRadius(d);

        // Inbound arrow: ← before the node (functions from other files call this)
        if (counts.inbound > 0) {
            node.append('text')
                .attr('class', 'call-indicator inbound')
                .attr('x', -(radius + 8))
                .attr('y', 5)
                .attr('text-anchor', 'end')
                .attr('fill', '#58a6ff')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .attr('cursor', 'pointer')
                .text(counts.inbound > 1 ? `${counts.inbound}←` : '←')
                .append('title')
                .text(`Called by ${counts.inbound} external function(s):\n${counts.inboundNodes.map(n => n.name).join(', ')}`);
        }

        // Outbound arrow: → after the label (this calls functions in other files)
        if (counts.outbound > 0) {
            // Get approximate label width
            const labelText = nodeData.name || '';
            const labelWidth = labelText.length * 7;

            node.append('text')
                .attr('class', 'call-indicator outbound')
                .attr('x', radius + labelWidth + 16)
                .attr('y', 5)
                .attr('text-anchor', 'start')
                .attr('fill', '#f0883e')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .attr('cursor', 'pointer')
                .text(counts.outbound > 1 ? `→${counts.outbound}` : '→')
                .append('title')
                .text(`Calls ${counts.outbound} external function(s):\n${counts.outboundNodes.map(n => n.name).join(', ')}`);
        }
    });

    // Collect and draw external call lines
    collectExternalCallData();
    drawExternalCallLines(g, root);

    // Node labels - positioned to the right of node, left-aligned
    // Use transform to position text, as x attribute can have rendering issues
    const labels = nodes.append('text')
        .attr('class', 'node-label')
        .attr('transform', d => `translate(${getNodeRadius(d) + 6}, 0)`)
        .attr('dominant-baseline', 'middle')
        .attr('text-anchor', 'start')
        .text(d => d.data.name)
        .style('font-size', d => chunkTypes.includes(d.data.type) ? '15px' : '18px')
        .style('font-family', 'Arial, sans-serif')
        .style('fill', d => chunkTypes.includes(d.data.type) ? '#bb86fc' : '#adbac7')
        .style('cursor', 'pointer')
        .style('pointer-events', 'all')
        .on('click', handleNodeClick);

    console.log(`Created ${labels.size()} label elements`);
    console.log('=== END RENDER LINEAR TREE ===');
}

// ============================================================================
// CIRCULAR TREE LAYOUT
// ============================================================================

function renderCircularTree() {
    const { width, height } = getViewportDimensions();
    const svg = d3.select('#graph')
        .attr('width', width)
        .attr('height', height);

    const radius = Math.min(width, height) / 2 - 100;

    const g = svg.append('g')
        .attr('transform', `translate(${width/2},${height/2})`);

    // Create radial tree layout
    const treeLayout = d3.tree()
        .size([2 * Math.PI, radius])
        .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth);

    // Create hierarchy
    // D3 hierarchy automatically respects children vs _children
    const root = d3.hierarchy(treeData, d => d.children);

    // Apply layout
    treeLayout(root);

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 3])
        .on('zoom', (event) => {
            g.attr('transform', `translate(${width/2},${height/2}) ${event.transform}`);
        });

    svg.call(zoom);

    // Draw links
    g.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkRadial()
            .angle(d => d.x)
            .radius(d => d.y))
        .attr('fill', 'none')
        .attr('stroke', '#ccc')
        .attr('stroke-width', 1.5);

    // Draw nodes
    const nodes = g.selectAll('.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `
            rotate(${d.x * 180 / Math.PI - 90})
            translate(${d.y},0)
        `)
        .on('click', handleNodeClick)
        .style('cursor', 'pointer');

    // Node circles - sized proportionally to content, colored by complexity
    nodes.append('circle')
        .attr('r', d => getNodeRadius(d))  // Dynamic size based on content
        .attr('fill', d => getNodeFillColor(d))  // Complexity-based coloring
        .attr('stroke', d => getNodeStrokeColor(d))  // Red/orange for high complexity
        .attr('stroke-width', d => getNodeStrokeWidth(d))
        .attr('class', d => {
            // Add complexity grade class if available
            const grade = d.data.complexity_grade || '';
            const hasSmells = (d.data.smell_count && d.data.smell_count > 0) || (d.data.smells && d.data.smells.length > 0);
            const classes = [];
            if (grade) classes.push(`grade-${grade}`);
            if (hasSmells) classes.push('has-smells');
            return classes.join(' ');
        });

    // Add external call arrow indicators (only for chunk nodes)
    nodes.each(function(d) {
        const node = d3.select(this);
        const nodeData = d.data;

        if (!chunkTypes.includes(nodeData.type)) return;

        const counts = getExternalCallCounts(nodeData);
        const radius = getNodeRadius(d);

        // Inbound indicator
        if (counts.inbound > 0) {
            node.append('text')
                .attr('x', 0)
                .attr('y', -(radius + 8))
                .attr('text-anchor', 'middle')
                .attr('fill', '#58a6ff')
                .attr('font-size', '10px')
                .attr('font-weight', 'bold')
                .text(counts.inbound > 1 ? `↓${counts.inbound}` : '↓')
                .append('title')
                .text(`Called by ${counts.inbound} external function(s)`);
        }

        // Outbound indicator
        if (counts.outbound > 0) {
            node.append('text')
                .attr('x', 0)
                .attr('y', radius + 12)
                .attr('text-anchor', 'middle')
                .attr('fill', '#f0883e')
                .attr('font-size', '10px')
                .attr('font-weight', 'bold')
                .text(counts.outbound > 1 ? `↑${counts.outbound}` : '↑')
                .append('title')
                .text(`Calls ${counts.outbound} external function(s)`);
        }
    });

    // Node labels - positioned to the right of node, left-aligned
    // Use transform to position text, as x attribute can have rendering issues
    nodes.append('text')
        .attr('class', 'node-label')
        .attr('transform', d => {
            const offset = getNodeRadius(d) + 6;
            const rotate = d.x >= Math.PI ? 'rotate(180)' : '';
            return `translate(${offset}, 0) ${rotate}`;
        })
        .attr('dominant-baseline', 'middle')
        .attr('text-anchor', d => d.x >= Math.PI ? 'end' : 'start')
        .text(d => d.data.name)
        .style('font-size', d => chunkTypes.includes(d.data.type) ? '15px' : '18px')
        .style('font-family', 'Arial, sans-serif')
        .style('fill', d => chunkTypes.includes(d.data.type) ? '#bb86fc' : '#adbac7')
        .style('cursor', 'pointer')
        .style('pointer-events', 'all')
        .on('click', handleNodeClick);

    // Collect and draw external call lines (circular version)
    collectExternalCallData();
    drawExternalCallLinesCircular(g, root);
}

// ============================================================================
// INTERACTION HANDLERS
// ============================================================================

function handleNodeClick(event, d) {
    event.stopPropagation();

    const nodeData = d.data;

    console.log('=== NODE CLICK DEBUG ===');
    console.log(`Clicked node: ${nodeData.name} (type: ${nodeData.type}, id: ${nodeData.id})`);
    console.log(`Has children: ${nodeData.children ? nodeData.children.length : 0}`);
    console.log(`Has _children: ${nodeData._children ? nodeData._children.length : 0}`);

    if (nodeData.type === 'directory') {
        // Toggle directory: swap children <-> _children
        if (nodeData.children) {
            // Currently expanded - collapse it
            console.log('Collapsing directory');
            nodeData._children = nodeData.children;
            nodeData.children = null;
        } else if (nodeData._children) {
            // Currently collapsed - expand it
            console.log('Expanding directory');
            nodeData.children = nodeData._children;
            nodeData._children = null;
        }

        // Re-render to show/hide children
        renderVisualization();

        // Don't auto-open viewer panel for directories - just expand/collapse
    } else if (nodeData.type === 'file') {
        // Check if this file has a collapsed chunk (single chunk with no children)
        if (nodeData.collapsed_chunk) {
            console.log(`Collapsed file+chunk: ${nodeData.name}, showing content directly`);
            displayChunkContent({
                ...nodeData.collapsed_chunk,
                name: nodeData.collapsed_chunk.name,
                type: nodeData.collapsed_chunk.type
            });
            return;
        }

        // Check if this is a single-chunk file - skip to content directly
        const childrenArray = nodeData._children || nodeData.children;

        if (childrenArray && childrenArray.length === 1) {
            const onlyChild = childrenArray[0];

            if (chunkTypes.includes(onlyChild.type)) {
                console.log(`Single-chunk file: ${nodeData.name}, showing content directly`);

                // Expand the file visually (for tree consistency)
                if (nodeData._children) {
                    nodeData.children = nodeData._children;
                    nodeData._children = null;
                    renderVisualization();
                }

                // Auto-display the chunk content
                displayChunkContent(onlyChild);
                return; // Skip normal file toggle behavior
            }
        }

        // Continue with existing toggle logic for multi-chunk files
        // Toggle file: swap children <-> _children
        if (nodeData.children) {
            // Currently expanded - collapse it
            console.log('Collapsing file');
            nodeData._children = nodeData.children;
            nodeData.children = null;
        } else if (nodeData._children) {
            // Currently collapsed - expand it
            console.log('Expanding file');
            nodeData.children = nodeData._children;
            nodeData._children = null;
        } else {
            console.log('WARNING: File has neither children nor _children!');
        }

        // Re-render to show/hide children
        renderVisualization();

        // Don't auto-open viewer panel for files - just expand/collapse
    } else if (chunkTypes.includes(nodeData.type)) {
        // Chunks can have children too (e.g., imports -> functions, class -> methods)
        // If chunk has children, toggle expand/collapse
        if (nodeData.children || nodeData._children) {
            if (nodeData.children) {
                // Currently expanded - collapse it
                console.log(`Collapsing ${nodeData.type} chunk`);
                nodeData._children = nodeData.children;
                nodeData.children = null;
            } else if (nodeData._children) {
                // Currently collapsed - expand it
                console.log(`Expanding ${nodeData.type} chunk to show ${nodeData._children.length} children`);
                nodeData.children = nodeData._children;
                nodeData._children = null;
            }
            // Re-render to show/hide children
            renderVisualization();
        }

        // Also show chunk content in side panel
        console.log('Displaying chunk content');
        displayChunkContent(nodeData);
    }

    console.log('=== END NODE CLICK DEBUG ===');
}

function displayDirectoryInfo(dirData, addToHistory = true) {
    openViewerPanel();

    // Add to navigation history
    if (addToHistory) {
        addToNavHistory({type: 'directory', data: dirData});
    }

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    title.textContent = `📁 ${dirData.name}`;

    // Count children
    const children = dirData.children || dirData._children || [];
    const dirs = children.filter(c => c.type === 'directory').length;
    const files = children.filter(c => c.type === 'file').length;

    let html = '';

    // Navigation bar with breadcrumbs and back/forward
    html += renderNavigationBar(dirData);

    html += '<div class="viewer-section">';
    html += '<div class="viewer-section-title">Directory Information</div>';
    html += '<div class="viewer-info-grid">';
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Name:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(dirData.name)}</span>`;
    html += `</div>`;
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Subdirectories:</span>`;
    html += `<span class="viewer-info-value">${dirs}</span>`;
    html += `</div>`;
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Files:</span>`;
    html += `<span class="viewer-info-value">${files}</span>`;
    html += `</div>`;
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Total Items:</span>`;
    html += `<span class="viewer-info-value">${children.length}</span>`;
    html += `</div>`;
    html += '</div>';
    html += '</div>';

    if (children.length > 0) {
        html += '<div class="viewer-section">';
        html += '<div class="viewer-section-title">Contents</div>';
        html += '<div class="dir-list">';

        // Sort: directories first, then files
        const sortedChildren = [...children].sort((a, b) => {
            if (a.type === 'directory' && b.type !== 'directory') return -1;
            if (a.type !== 'directory' && b.type === 'directory') return 1;
            return a.name.localeCompare(b.name);
        });

        sortedChildren.forEach(child => {
            const icon = child.type === 'directory' ? '📁' : '📄';
            const type = child.type === 'directory' ? 'dir' : 'file';
            const childData = JSON.stringify(child).replace(/"/g, '&quot;');
            const clickHandler = child.type === 'directory'
                ? `navigateToDirectory(${childData})`
                : `navigateToFile(${childData})`;
            html += `<div class="dir-list-item clickable" onclick="${clickHandler}">`;
            html += `<span class="dir-icon">${icon}</span>`;
            html += `<span class="dir-name">${escapeHtml(child.name)}</span>`;
            html += `<span class="dir-type">${type}</span>`;
            html += `<span class="dir-arrow">→</span>`;
            html += `</div>`;
        });

        html += '</div>';
        html += '</div>';
    }

    content.innerHTML = html;

    // Hide section dropdown for directories (no code sections)
    populateSectionDropdown([]);
}

function displayFileInfo(fileData, addToHistory = true) {
    openViewerPanel();

    // Add to navigation history
    if (addToHistory) {
        addToNavHistory({type: 'file', data: fileData});
    }

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    title.textContent = `📄 ${fileData.name}`;

    // Get chunks
    const chunks = fileData.children || fileData._children || [];

    let html = '';

    // Navigation bar with breadcrumbs and back/forward
    html += renderNavigationBar(fileData);

    html += '<div class="viewer-section">';
    html += '<div class="viewer-section-title">File Information</div>';
    html += '<div class="viewer-info-grid">';
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Name:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(fileData.name)}</span>`;
    html += `</div>`;
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Chunks:</span>`;
    html += `<span class="viewer-info-value">${chunks.length}</span>`;
    html += `</div>`;
    if (fileData.path) {
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">Path:</span>`;
        html += `<span class="viewer-info-value" style="word-break: break-all;">${escapeHtml(fileData.path)}</span>`;
        html += `</div>`;
    }
    html += '</div>';
    html += '</div>';

    if (chunks.length > 0) {
        html += '<div class="viewer-section">';
        html += '<div class="viewer-section-title">Code Chunks</div>';
        html += '<div class="chunk-list">';

        chunks.forEach(chunk => {
            const icon = getChunkIcon(chunk.type);
            const chunkName = chunk.name || chunk.type || 'chunk';
            const lines = chunk.start_line && chunk.end_line
                ? `Lines ${chunk.start_line}-${chunk.end_line}`
                : 'Unknown lines';

            html += `<div class="chunk-list-item" onclick="displayChunkContent(${JSON.stringify(chunk).replace(/"/g, '&quot;')})">`;
            html += `<span class="chunk-icon">${icon}</span>`;
            html += `<div class="chunk-info">`;
            html += `<div class="chunk-name">${escapeHtml(chunkName)}</div>`;
            html += `<div class="chunk-meta">${lines} • ${chunk.type || 'code'}</div>`;
            html += `</div>`;
            html += `</div>`;
        });

        html += '</div>';
        html += '</div>';
    }

    content.innerHTML = html;

    // Hide section dropdown for files (no code sections in file view)
    populateSectionDropdown([]);
}

function displayChunkContent(chunkData, addToHistory = true) {
    openViewerPanel();

    // Expand path to node and highlight it in tree
    if (chunkData.id) {
        expandAndHighlightNode(chunkData.id);
    }

    // Add to navigation history
    if (addToHistory) {
        addToNavHistory({type: 'chunk', data: chunkData});
    }

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    const chunkName = chunkData.name || chunkData.type || 'Chunk';
    title.textContent = `${getChunkIcon(chunkData.type)} ${chunkName}`;

    let html = '';

    // Navigation bar with breadcrumbs and back/forward
    html += renderNavigationBar(chunkData);

    // Track sections for dropdown navigation
    const sections = [];

    // === ORDER: Docstring (comments), Code, Metadata ===

    // === 1. Docstring Section (Comments) ===
    if (chunkData.docstring) {
        sections.push({ id: 'docstring', label: '📖 Docstring' });
        html += '<div class="viewer-section" data-section="docstring">';
        html += '<div class="viewer-section-title">📖 Docstring</div>';
        html += `<div style="color: #8b949e; font-style: italic; padding: 8px 12px; background: #161b22; border-radius: 4px; white-space: pre-wrap;">${escapeHtml(chunkData.docstring)}</div>`;
        html += '</div>';
    }

    // === 2. Source Code Section ===
    if (chunkData.content) {
        sections.push({ id: 'source-code', label: '📝 Source Code' });
        html += '<div class="viewer-section" data-section="source-code">';
        html += '<div class="viewer-section-title">📝 Source Code</div>';
        const langClass = getLanguageClass(chunkData.file_path);
        html += `<pre><code class="hljs${langClass ? ' language-' + langClass : ''}">${escapeHtml(chunkData.content)}</code></pre>`;
        html += '</div>';
    } else {
        html += '<p style="color: #8b949e; padding: 20px; text-align: center;">No content available for this chunk.</p>';
    }

    // === 3. Metadata Section ===
    sections.push({ id: 'metadata', label: 'ℹ️ Metadata' });
    html += '<div class="viewer-section" data-section="metadata">';
    html += '<div class="viewer-section-title">ℹ️ Metadata</div>';
    html += '<div class="viewer-info-grid">';

    // Basic info
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Name:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(chunkName)}</span>`;
    html += `</div>`;

    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Type:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(chunkData.type || 'code')}</span>`;
    html += `</div>`;

    // File path (clickable - navigates to file node)
    if (chunkData.file_path) {
        const shortPath = chunkData.file_path.split('/').slice(-3).join('/');
        const escapedPath = escapeHtml(chunkData.file_path).replace(/'/g, "\\'");
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">File:</span>`;
        html += `<span class="viewer-info-value clickable" onclick="navigateToFileByPath('${escapedPath}')" title="Click to navigate to file: ${escapeHtml(chunkData.file_path)}">.../${escapeHtml(shortPath)}</span>`;
        html += `</div>`;
    }

    // Line numbers
    if (chunkData.start_line && chunkData.end_line) {
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">Lines:</span>`;
        html += `<span class="viewer-info-value">${chunkData.start_line} - ${chunkData.end_line} (${chunkData.end_line - chunkData.start_line + 1} lines)</span>`;
        html += `</div>`;
    }

    // Language
    if (chunkData.language) {
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">Language:</span>`;
        html += `<span class="viewer-info-value">${escapeHtml(chunkData.language)}</span>`;
        html += `</div>`;
    }

    // Complexity
    if (chunkData.complexity !== undefined && chunkData.complexity !== null) {
        const complexityColor = chunkData.complexity > 10 ? '#f85149' : chunkData.complexity > 5 ? '#d29922' : '#3fb950';
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">Complexity:</span>`;
        html += `<span class="viewer-info-value" style="color: ${complexityColor}">${chunkData.complexity.toFixed(1)}</span>`;
        html += `</div>`;
    }

    html += '</div>';
    html += '</div>';

    // === 4. External Calls & Callers Section (Cross-file references) ===
    const chunkId = chunkData.id;
    const currentFilePath = chunkData.file_path;

    if (chunkId) {
        // Find all caller relationships
        const allCallers = allLinks.filter(l => l.type === 'caller' && l.target === chunkId);
        const allCallees = allLinks.filter(l => l.type === 'caller' && l.source === chunkId);

        // Separate external (different file) from local (same file) relationships
        // Use Maps to deduplicate by node.id
        const externalCallersMap = new Map();
        const localCallersMap = new Map();
        allCallers.forEach(link => {
            const callerNode = allNodes.find(n => n.id === link.source);
            if (callerNode) {
                if (callerNode.file_path !== currentFilePath) {
                    if (!externalCallersMap.has(callerNode.id)) {
                        externalCallersMap.set(callerNode.id, { link, node: callerNode });
                    }
                } else {
                    if (!localCallersMap.has(callerNode.id)) {
                        localCallersMap.set(callerNode.id, { link, node: callerNode });
                    }
                }
            }
        });
        const externalCallers = Array.from(externalCallersMap.values());
        const localCallers = Array.from(localCallersMap.values());

        const externalCalleesMap = new Map();
        const localCalleesMap = new Map();
        allCallees.forEach(link => {
            const calleeNode = allNodes.find(n => n.id === link.target);
            if (calleeNode) {
                if (calleeNode.file_path !== currentFilePath) {
                    if (!externalCalleesMap.has(calleeNode.id)) {
                        externalCalleesMap.set(calleeNode.id, { link, node: calleeNode });
                    }
                } else {
                    if (!localCalleesMap.has(calleeNode.id)) {
                        localCalleesMap.set(calleeNode.id, { link, node: calleeNode });
                    }
                }
            }
        });
        const externalCallees = Array.from(externalCalleesMap.values());
        const localCallees = Array.from(localCalleesMap.values());

        // === External Callers Section (functions from other files that call this) ===
        if (externalCallers.length > 0) {
            sections.push({ id: 'external-callers', label: '📥 External Callers' });
            html += '<div class="viewer-section" data-section="external-callers">';
            html += '<div class="viewer-section-title">📥 External Callers <span style="color: #8b949e; font-weight: normal;">(functions from other files calling this)</span></div>';
            html += '<div style="display: flex; flex-direction: column; gap: 6px;">';
            externalCallers.slice(0, 10).forEach(({ link, node }) => {
                const shortPath = node.file_path ? node.file_path.split('/').slice(-2).join('/') : '';
                html += `<div class="external-call-item" onclick="focusNodeInTree('${link.source}')" title="${escapeHtml(node.file_path || '')}">`;
                html += `<span class="external-call-icon">←</span>`;
                html += `<span class="external-call-name">${escapeHtml(node.name || node.id.substring(0, 8))}</span>`;
                html += `<span class="external-call-path">${escapeHtml(shortPath)}</span>`;
                html += `</div>`;
            });
            if (externalCallers.length > 10) {
                html += `<div style="color: #8b949e; font-size: 11px; padding-left: 20px;">+${externalCallers.length - 10} more external callers</div>`;
            }
            html += '</div></div>';
        }

        // === External Calls Section (functions in other files this calls) ===
        if (externalCallees.length > 0) {
            sections.push({ id: 'external-calls', label: '📤 External Calls' });
            html += '<div class="viewer-section" data-section="external-calls">';
            html += '<div class="viewer-section-title">📤 External Calls <span style="color: #8b949e; font-weight: normal;">(functions in other files this calls)</span></div>';
            html += '<div style="display: flex; flex-direction: column; gap: 6px;">';
            externalCallees.slice(0, 10).forEach(({ link, node }) => {
                const shortPath = node.file_path ? node.file_path.split('/').slice(-2).join('/') : '';
                html += `<div class="external-call-item" onclick="focusNodeInTree('${link.target}')" title="${escapeHtml(node.file_path || '')}">`;
                html += `<span class="external-call-icon">→</span>`;
                html += `<span class="external-call-name">${escapeHtml(node.name || node.id.substring(0, 8))}</span>`;
                html += `<span class="external-call-path">${escapeHtml(shortPath)}</span>`;
                html += `</div>`;
            });
            if (externalCallees.length > 10) {
                html += `<div style="color: #8b949e; font-size: 11px; padding-left: 20px;">+${externalCallees.length - 10} more external calls</div>`;
            }
            html += '</div></div>';
        }

        // === Local (Same-File) Relationships Section ===
        if (localCallers.length > 0 || localCallees.length > 0) {
            sections.push({ id: 'local-references', label: '🔗 Local References' });
            html += '<div class="viewer-section" data-section="local-references">';
            html += '<div class="viewer-section-title">🔗 Local References <span style="color: #8b949e; font-weight: normal;">(same file)</span></div>';

            if (localCallers.length > 0) {
                html += '<div style="margin-bottom: 8px;">';
                html += '<div style="color: #58a6ff; font-size: 11px; margin-bottom: 4px;">Called by:</div>';
                html += '<div style="display: flex; flex-wrap: wrap; gap: 4px;">';
                localCallers.slice(0, 8).forEach(({ link, node }) => {
                    html += `<span class="relationship-tag caller" onclick="focusNodeInTree('${link.source}')" title="${escapeHtml(node.name || '')}">${escapeHtml(node.name || node.id.substring(0, 8))}</span>`;
                });
                if (localCallers.length > 8) {
                    html += `<span style="color: #8b949e; font-size: 10px;">+${localCallers.length - 8} more</span>`;
                }
                html += '</div></div>';
            }

            if (localCallees.length > 0) {
                html += '<div>';
                html += '<div style="color: #f0883e; font-size: 11px; margin-bottom: 4px;">Calls:</div>';
                html += '<div style="display: flex; flex-wrap: wrap; gap: 4px;">';
                localCallees.slice(0, 8).forEach(({ link, node }) => {
                    html += `<span class="relationship-tag callee" onclick="focusNodeInTree('${link.target}')" title="${escapeHtml(node.name || '')}">${escapeHtml(node.name || node.id.substring(0, 8))}</span>`;
                });
                if (localCallees.length > 8) {
                    html += `<span style="color: #8b949e; font-size: 10px;">+${localCallees.length - 8} more</span>`;
                }
                html += '</div></div>';
            }

            html += '</div>';
        }

        // === Semantically Similar Section ===
        const semanticLinks = allLinks.filter(l => l.type === 'semantic' && l.source === chunkId);
        if (semanticLinks.length > 0) {
            sections.push({ id: 'semantic', label: '🧠 Semantically Similar' });
            html += '<div class="viewer-section" data-section="semantic">';
            html += '<div class="viewer-section-title">🧠 Semantically Similar</div>';
            html += '<div style="display: flex; flex-direction: column; gap: 4px;">';
            semanticLinks.slice(0, 5).forEach(link => {
                const similarNode = allNodes.find(n => n.id === link.target);
                if (similarNode) {
                    const similarity = (link.similarity * 100).toFixed(0);
                    const label = similarNode.name || similarNode.id.substring(0, 8);
                    html += `<div class="semantic-item" onclick="focusNodeInTree('${link.target}')" title="${escapeHtml(similarNode.file_path || '')}">`;
                    html += `<span class="semantic-score">${similarity}%</span>`;
                    html += `<span class="semantic-name">${escapeHtml(label)}</span>`;
                    html += `<span class="semantic-type">${similarNode.type || ''}</span>`;
                    html += `</div>`;
                }
            });
            html += '</div>';
            html += '</div>';
        }
    }

    content.innerHTML = html;

    // Apply syntax highlighting to code blocks
    content.querySelectorAll('pre code').forEach((block) => {
        if (typeof hljs !== 'undefined') {
            hljs.highlightElement(block);
        }
    });

    // Populate section dropdown for navigation
    populateSectionDropdown(sections);
}

// Focus on a node in the tree (expand path, scroll, highlight)
function focusNodeInTree(nodeId) {
    console.log(`Focusing on node in tree: ${nodeId}`);

    // Find the node in allNodes (the original data)
    const targetNodeData = allNodes.find(n => n.id === nodeId);
    if (!targetNodeData) {
        console.log(`Node ${nodeId} not found in allNodes`);
        return;
    }

    // Find the path to this node in the tree structure
    // We need to find and expand all ancestors to make the node visible
    const pathToNode = findPathToNode(treeData, nodeId);

    if (pathToNode.length > 0) {
        console.log(`Found path to node: ${pathToNode.map(n => n.name).join(' -> ')}`);

        // Expand all nodes along the path (except the target node itself)
        pathToNode.slice(0, -1).forEach(node => {
            if (node._children) {
                // Node is collapsed, expand it
                console.log(`Expanding ${node.name} to reveal path`);
                node.children = node._children;
                node._children = null;
            }
        });

        // Re-render the tree to show the expanded path
        renderVisualization();

        // After render, scroll to and highlight the target node
        setTimeout(() => {
            highlightNodeInTree(nodeId);
        }, 100);
    } else {
        console.log(`Path to node ${nodeId} not found in tree - it may be orphaned`);
    }

    // Display the content in the viewer panel
    if (chunkTypes.includes(targetNodeData.type)) {
        displayChunkContent(targetNodeData);
    } else if (targetNodeData.type === 'file') {
        displayFileInfo(targetNodeData);
    } else if (targetNodeData.type === 'directory') {
        displayDirectoryInfo(targetNodeData);
    }
}

// Find path from root to a specific node by ID
function findPathToNode(node, targetId, path = []) {
    if (!node) return [];

    // Add current node to path
    const currentPath = [...path, node];

    // Check if this is the target
    if (node.id === targetId) {
        return currentPath;
    }

    // Check visible children
    if (node.children) {
        for (const child of node.children) {
            const result = findPathToNode(child, targetId, currentPath);
            if (result.length > 0) return result;
        }
    }

    // Check hidden children
    if (node._children) {
        for (const child of node._children) {
            const result = findPathToNode(child, targetId, currentPath);
            if (result.length > 0) return result;
        }
    }

    return [];
}

// Expand path to a node and highlight it (without triggering content display)
function expandAndHighlightNode(nodeId) {
    console.log('=== EXPAND AND HIGHLIGHT ===');
    console.log('Target nodeId:', nodeId);

    // Find the path to this node in the tree structure
    const pathToNode = findPathToNode(treeData, nodeId);

    if (pathToNode.length > 0) {
        console.log('Found path:', pathToNode.map(n => n.name).join(' -> '));

        // Expand all nodes along the path (except the target node itself)
        let needsRerender = false;
        pathToNode.slice(0, -1).forEach(node => {
            if (node._children) {
                console.log('Expanding:', node.name);
                node.children = node._children;
                node._children = null;
                needsRerender = true;
            }
        });

        if (needsRerender) {
            renderVisualization();
        }

        // Highlight after a short delay to allow render to complete
        setTimeout(() => {
            highlightNodeInTree(nodeId);
        }, 50);
    } else {
        console.log('Path not found - trying direct highlight');
        highlightNodeInTree(nodeId);
    }
}

// Highlight and scroll to a node in the rendered tree
function highlightNodeInTree(nodeId, persistent = true) {
    console.log('=== HIGHLIGHT NODE ===');
    console.log('Looking for nodeId:', nodeId);

    // Remove any existing highlight
    d3.selectAll('.node-highlight').classed('node-highlight', false);
    if (persistent) {
        d3.selectAll('.node-selected').classed('node-selected', false);
    }

    // Find and highlight the target node in the rendered SVG
    const svg = d3.select('#graph');
    const allNodes = svg.selectAll('.node');
    console.log('Total nodes in SVG:', allNodes.size());

    const targetNode = allNodes.filter(d => d.data.id === nodeId);
    console.log('Matching nodes found:', targetNode.size());

    if (!targetNode.empty()) {
        // Add highlight class (persistent = orange glow that stays)
        if (persistent) {
            targetNode.classed('node-selected', true);
            console.log('Applied .node-selected class');
        } else {
            targetNode.classed('node-highlight', true);
            console.log('Applied .node-highlight class');
        }

        // Pulse the node circle - scale up from current size
        targetNode.select('circle')
            .transition()
            .duration(200)
            .attr('r', d => getNodeRadius(d) * 1.5)  // Grow 50%
            .transition()
            .duration(200)
            .attr('r', d => getNodeRadius(d) * 0.8)  // Shrink 20%
            .transition()
            .duration(200)
            .attr('r', d => getNodeRadius(d));       // Return to normal

        // Get the node's position for scrolling
        const nodeTransform = targetNode.attr('transform');
        const match = nodeTransform.match(/translate\\(([^,]+),([^)]+)\\)/);
        if (match) {
            const x = parseFloat(match[1]);
            const y = parseFloat(match[2]);

            // Pan the view to center on this node
            const { width, height } = getViewportDimensions();
            const zoom = d3.zoom().on('zoom', () => {});  // Get current zoom
            const svg = d3.select('#graph');

            // Calculate center offset
            const centerX = width / 2 - x;
            const centerY = height / 2 - y;

            // Apply smooth transition to center on node
            svg.transition()
                .duration(500)
                .call(
                    d3.zoom().transform,
                    d3.zoomIdentity.translate(centerX, centerY)
                );
        }

        console.log(`Highlighted node ${nodeId}`);
    } else {
        console.log(`Node ${nodeId} not found in rendered tree`);
    }
}

// Legacy function for backward compatibility
function focusNode(nodeId) {
    focusNodeInTree(nodeId);
}

function getChunkIcon(chunkType) {
    const icons = {
        'function': '⚡',
        'class': '🏛️',
        'method': '🔧',
        'code': '📝',
        'import': '📦',
        'comment': '💬',
        'docstring': '📖'
    };
    return icons[chunkType] || '📝';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Get language class for highlight.js based on file extension
function getLanguageClass(filePath) {
    if (!filePath) return '';
    const ext = filePath.split('.').pop().toLowerCase();
    const langMap = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'tsx': 'typescript',
        'jsx': 'javascript',
        'java': 'java',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'c': 'c',
        'cpp': 'cpp',
        'cc': 'cpp',
        'h': 'c',
        'hpp': 'cpp',
        'cs': 'csharp',
        'swift': 'swift',
        'kt': 'kotlin',
        'scala': 'scala',
        'sh': 'bash',
        'bash': 'bash',
        'zsh': 'bash',
        'sql': 'sql',
        'html': 'html',
        'htm': 'html',
        'css': 'css',
        'scss': 'scss',
        'less': 'less',
        'json': 'json',
        'yaml': 'yaml',
        'yml': 'yaml',
        'xml': 'xml',
        'md': 'markdown',
        'markdown': 'markdown',
        'toml': 'ini',
        'ini': 'ini',
        'cfg': 'ini',
        'lua': 'lua',
        'r': 'r',
        'dart': 'dart',
        'ex': 'elixir',
        'exs': 'elixir',
        'erl': 'erlang',
        'hs': 'haskell',
        'clj': 'clojure',
        'vim': 'vim',
        'dockerfile': 'dockerfile'
    };
    return langMap[ext] || '';
}

// ============================================================================
// LAYOUT TOGGLE
// ============================================================================

function toggleCallLines(show) {
    showCallLines = show;
    const lineGroup = d3.select('.external-lines-group');
    if (!lineGroup.empty()) {
        lineGroup.style('display', show ? 'block' : 'none');
    }
    console.log(`Call lines ${show ? 'shown' : 'hidden'}`);
}

function toggleLayout() {
    const toggleCheckbox = document.getElementById('layout-toggle');
    const labels = document.querySelectorAll('.toggle-label');

    // Update layout based on checkbox state
    currentLayout = toggleCheckbox.checked ? 'circular' : 'linear';

    // Update label highlighting
    labels.forEach((label, index) => {
        if (index === 0) {
            // Linear label (left)
            label.classList.toggle('active', currentLayout === 'linear');
        } else {
            // Circular label (right)
            label.classList.toggle('active', currentLayout === 'circular');
        }
    });

    console.log(`Layout switched to: ${currentLayout}`);
    renderVisualization();
}

// ============================================================================
// FILE TYPE FILTER
// ============================================================================

// Code file extensions
const codeExtensions = new Set([
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sh', '.bash', '.zsh', '.ps1', '.bat', '.sql', '.html', '.css', '.scss',
    '.sass', '.less', '.vue', '.svelte', '.astro', '.elm', '.clj', '.ex', '.exs',
    '.hs', '.ml', '.lua', '.pl', '.pm', '.m', '.mm', '.f', '.f90', '.for',
    '.asm', '.s', '.v', '.vhd', '.sv', '.nim', '.zig', '.d', '.dart', '.groovy',
    '.coffee', '.litcoffee', '.purs', '.rkt', '.scm', '.lisp', '.cl'
]);

// Doc file extensions
const docExtensions = new Set([
    '.md', '.markdown', '.rst', '.txt', '.adoc', '.asciidoc', '.org', '.tex',
    '.rtf', '.doc', '.docx', '.pdf', '.json', '.yaml', '.yml', '.toml', '.ini',
    '.cfg', '.conf', '.xml', '.csv', '.tsv', '.log', '.man', '.info', '.pod',
    '.rdoc', '.textile', '.wiki'
]);

function getFileType(filename) {
    if (!filename) return 'unknown';
    const ext = '.' + filename.split('.').pop().toLowerCase();
    if (codeExtensions.has(ext)) return 'code';
    if (docExtensions.has(ext)) return 'docs';
    return 'unknown';
}

function setFileFilter(filter) {
    currentFileFilter = filter;

    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    // Apply filter to the tree
    applyFileFilter();

    console.log(`File filter set to: ${filter}`);
}

function applyFileFilter() {
    if (!treeData) return;

    console.log('=== APPLYING FILE FILTER (VISIBILITY) ===');
    console.log('Current filter:', currentFileFilter);

    // Get all node elements in the visualization
    const nodeElements = d3.selectAll('.node');
    const linkElements = d3.selectAll('.link');

    if (currentFileFilter === 'all') {
        // Show everything
        nodeElements.style('display', null);
        linkElements.style('display', null);
        console.log('Showing all nodes');
        return;
    }

    // Build set of visible node IDs based on filter
    const visibleIds = new Set();

    // Recursive function to check if a node or any descendant matches filter
    function checkNodeAndDescendants(node) {
        let hasMatchingDescendant = false;

        // Check children (both visible and collapsed)
        const children = node.children || node._children || [];
        children.forEach(child => {
            if (checkNodeAndDescendants(child)) {
                hasMatchingDescendant = true;
            }
        });

        // Check if this node itself matches
        let matches = false;
        if (node.type === 'directory') {
            // Directories are visible if they have matching descendants
            matches = hasMatchingDescendant;
        } else if (node.type === 'file') {
            const fileType = getFileType(node.name);
            matches = (fileType === currentFileFilter) || (fileType === 'unknown');
        } else if (chunkTypes.includes(node.type)) {
            // Chunks match if their parent file matches
            // For simplicity, check file_path extension
            if (node.file_path) {
                const ext = node.file_path.substring(node.file_path.lastIndexOf('.'));
                const fileType = getFileType(node.file_path);
                matches = (fileType === currentFileFilter) || (fileType === 'unknown');
            }
        }

        if (matches || hasMatchingDescendant) {
            visibleIds.add(node.id);
            return true;
        }
        return false;
    }

    checkNodeAndDescendants(treeData);
    console.log(`Filter found ${visibleIds.size} visible nodes`);

    // Apply visibility to DOM elements
    nodeElements.style('display', function(d) {
        return visibleIds.has(d.data.id) ? null : 'none';
    });

    // Hide links where either end is hidden
    linkElements.style('display', function(d) {
        const sourceVisible = visibleIds.has(d.source.data.id);
        const targetVisible = visibleIds.has(d.target.data.id);
        return (sourceVisible && targetVisible) ? null : 'none';
    });

    console.log('=== FILTER COMPLETE (VISIBILITY) ===');
}

// ============================================================================
// VIEWER PANEL CONTROLS
// ============================================================================

let isViewerExpanded = false;

function openViewerPanel() {
    const panel = document.getElementById('viewer-panel');
    const container = document.getElementById('main-container');

    if (!isViewerOpen) {
        panel.classList.add('open');
        container.classList.add('viewer-open');
        isViewerOpen = true;

        // Re-render visualization to adjust to new viewport size
        setTimeout(() => {
            renderVisualization();
        }, 300); // Wait for transition
    }
}

function closeViewerPanel() {
    const panel = document.getElementById('viewer-panel');
    const container = document.getElementById('main-container');

    panel.classList.remove('open');
    panel.classList.remove('expanded');
    container.classList.remove('viewer-open');
    isViewerOpen = false;
    isViewerExpanded = false;

    // Update icon
    const icon = document.getElementById('expand-icon');
    if (icon) icon.textContent = '⬅';

    // Re-render visualization to adjust to new viewport size
    setTimeout(() => {
        renderVisualization();
    }, 300); // Wait for transition
}

function toggleViewerExpand() {
    const panel = document.getElementById('viewer-panel');
    const icon = document.getElementById('expand-icon');

    isViewerExpanded = !isViewerExpanded;

    if (isViewerExpanded) {
        panel.classList.add('expanded');
        if (icon) icon.textContent = '➡';
    } else {
        panel.classList.remove('expanded');
        if (icon) icon.textContent = '⬅';
    }

    // Don't re-render graph on expand - only affects panel width
    // Graph will adjust on close via closeViewerPanel()
}

function jumpToSection(sectionId) {
    if (!sectionId) return;

    const viewerContent = document.getElementById('viewer-content');
    const sectionElement = viewerContent.querySelector(`[data-section="${sectionId}"]`);

    if (sectionElement) {
        sectionElement.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Briefly highlight the section
        sectionElement.style.transition = 'background-color 0.3s';
        sectionElement.style.backgroundColor = 'rgba(88, 166, 255, 0.15)';
        setTimeout(() => {
            sectionElement.style.backgroundColor = '';
        }, 1000);
    }

    // Reset dropdown to default
    const dropdown = document.getElementById('section-dropdown');
    if (dropdown) dropdown.value = '';
}

function populateSectionDropdown(sections) {
    const dropdown = document.getElementById('section-dropdown');
    if (!dropdown) return;

    // Clear existing options except the first one
    dropdown.innerHTML = '<option value="">Jump to section...</option>';

    // Add section options
    sections.forEach(section => {
        const option = document.createElement('option');
        option.value = section.id;
        option.textContent = section.label;
        dropdown.appendChild(option);
    });

    // Show/hide dropdown based on whether we have sections
    const sectionNav = document.getElementById('section-nav');
    if (sectionNav) {
        sectionNav.style.display = sections.length > 1 ? 'block' : 'none';
    }
}

// ============================================================================
// NAVIGATION FUNCTIONS
// ============================================================================

function addToNavHistory(item) {
    // Remove any forward history when adding new item
    if (navigationIndex < navigationHistory.length - 1) {
        navigationHistory = navigationHistory.slice(0, navigationIndex + 1);
    }
    navigationHistory.push(item);
    navigationIndex = navigationHistory.length - 1;
    console.log(`Navigation history: ${navigationHistory.length} items, index: ${navigationIndex}`);
}

function goBack() {
    if (navigationIndex > 0) {
        navigationIndex--;
        const item = navigationHistory[navigationIndex];
        console.log(`Going back to: ${item.type} - ${item.data.name}`);
        if (item.type === 'directory') {
            displayDirectoryInfo(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        } else if (item.type === 'file') {
            displayFileInfo(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        } else if (item.type === 'chunk') {
            displayChunkContent(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        }
    }
}

function goForward() {
    if (navigationIndex < navigationHistory.length - 1) {
        navigationIndex++;
        const item = navigationHistory[navigationIndex];
        console.log(`Going forward to: ${item.type} - ${item.data.name}`);
        if (item.type === 'directory') {
            displayDirectoryInfo(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        } else if (item.type === 'file') {
            displayFileInfo(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        } else if (item.type === 'chunk') {
            displayChunkContent(item.data, false);  // false = don't add to history
            focusNodeInTree(item.data.id);
        }
    }
}

function navigateToDirectory(dirData) {
    console.log(`Navigating to directory: ${dirData.name}`);
    // Focus on the node in the tree (expand path and highlight)
    focusNodeInTree(dirData.id);
}

function navigateToFile(fileData) {
    console.log(`Navigating to file: ${fileData.name}`);
    // Focus on the node in the tree (expand path and highlight)
    focusNodeInTree(fileData.id);
}

// Navigate to a file by its file path (used when clicking on file paths in chunk metadata)
function navigateToFileByPath(filePath) {
    console.log(`Navigating to file by path: ${filePath}`);

    // Find the file node in allNodes that matches this path
    const fileNode = allNodes.find(n => {
        if (n.type !== 'file') return false;
        // Match against various path properties
        return n.path === filePath ||
               n.file_path === filePath ||
               (n.path && n.path.endsWith(filePath)) ||
               (n.file_path && n.file_path.endsWith(filePath)) ||
               filePath.endsWith(n.name);
    });

    let targetNode = fileNode;
    if (!targetNode) {
        // Try to find by just the filename
        const fileName = filePath.split('/').pop();
        targetNode = allNodes.find(n => n.type === 'file' && n.name === fileName);
    }

    if (!targetNode) {
        console.log(`File node not found for path: ${filePath}`);
        return;
    }

    console.log(`Found file node: ${targetNode.name} (id: ${targetNode.id})`);

    // Handle navigation based on current visualization mode
    if (currentVizMode === 'treemap' || currentVizMode === 'sunburst') {
        // Close the viewer panel first
        closeViewerPanel();

        // For treemap/sunburst, zoom to the file's parent directory
        // Find the parent directory by looking at the file path
        const parentPath = filePath.split('/').slice(0, -1).join('/');
        const parentDir = allNodes.find(n =>
            n.type === 'directory' &&
            (n.path === parentPath || n.file_path === parentPath ||
             (n.path && parentPath.endsWith(n.path)) ||
             (n.file_path && parentPath.endsWith(n.file_path)))
        );

        if (parentDir) {
            console.log(`Zooming to parent directory: ${parentDir.name}`);
            currentZoomRootId = parentDir.id;
        } else {
            // If no parent found, zoom to file itself (for file grouping mode)
            // or reset to root
            console.log(`Parent directory not found, resetting zoom`);
            currentZoomRootId = null;
        }

        renderVisualization();
    } else {
        // For tree mode, use the existing tree focus behavior
        focusNodeInTree(targetNode.id);
    }
}

function renderNavigationBar(currentItem) {
    let html = '<div class="navigation-bar">';

    // Back/Forward buttons
    const canGoBack = navigationIndex > 0;
    const canGoForward = navigationIndex < navigationHistory.length - 1;

    html += `<button class="nav-btn ${canGoBack ? '' : 'disabled'}" onclick="goBack()" ${canGoBack ? '' : 'disabled'} title="Go Back">←</button>`;
    html += `<button class="nav-btn ${canGoForward ? '' : 'disabled'}" onclick="goForward()" ${canGoForward ? '' : 'disabled'} title="Go Forward">→</button>`;

    // Breadcrumb trail
    html += '<div class="breadcrumb-trail">';

    // Build breadcrumb from path
    if (currentItem && currentItem.id) {
        const path = findPathToNode(treeData, currentItem.id);
        path.forEach((node, index) => {
            const isLast = index === path.length - 1;
            const clickable = !isLast;

            if (index > 0) {
                html += '<span class="breadcrumb-separator">/</span>';
            }

            if (clickable) {
                html += `<span class="breadcrumb-item clickable" onclick="focusNodeInTree('${node.id}')">${escapeHtml(node.name)}</span>`;
            } else {
                html += `<span class="breadcrumb-item current">${escapeHtml(node.name)}</span>`;
            }
        });
    }

    html += '</div>';
    html += '</div>';

    return html;
}

// ============================================================================
// SEARCH FUNCTIONALITY
// ============================================================================

let searchDebounceTimer = null;
let searchResults = [];
let selectedSearchIndex = -1;

function handleSearchInput(event) {
    const query = event.target.value.trim();

    // Debounce search - wait 150ms after typing stops
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
        performSearch(query);
    }, 150);
}

function handleSearchKeydown(event) {
    const resultsContainer = document.getElementById('search-results');

    switch(event.key) {
        case 'ArrowDown':
            event.preventDefault();
            if (searchResults.length > 0) {
                selectedSearchIndex = Math.min(selectedSearchIndex + 1, searchResults.length - 1);
                updateSearchSelection();
            }
            break;
        case 'ArrowUp':
            event.preventDefault();
            if (searchResults.length > 0) {
                selectedSearchIndex = Math.max(selectedSearchIndex - 1, 0);
                updateSearchSelection();
            }
            break;
        case 'Enter':
            event.preventDefault();
            if (selectedSearchIndex >= 0 && selectedSearchIndex < searchResults.length) {
                selectSearchResult(searchResults[selectedSearchIndex]);
            }
            break;
        case 'Escape':
            closeSearchResults();
            document.getElementById('search-input').blur();
            break;
    }
}

function performSearch(query) {
    const resultsContainer = document.getElementById('search-results');

    if (!query || query.length < 2) {
        closeSearchResults();
        return;
    }

    const lowerQuery = query.toLowerCase();

    // Search through all nodes (directories, files, and chunks)
    searchResults = allNodes
        .filter(node => {
            // Match against name
            const nameMatch = node.name && node.name.toLowerCase().includes(lowerQuery);
            // Match against file path
            const pathMatch = node.file_path && node.file_path.toLowerCase().includes(lowerQuery);
            // Match against ID (useful for finding specific chunks)
            const idMatch = node.id && node.id.toLowerCase().includes(lowerQuery);
            return nameMatch || pathMatch || idMatch;
        })
        .slice(0, 20);  // Limit to 20 results

    // Sort results: exact matches first, then by type priority
    const typePriority = { 'directory': 1, 'file': 2, 'class': 3, 'function': 4, 'method': 5 };
    searchResults.sort((a, b) => {
        // Exact name match gets highest priority
        const aExact = a.name && a.name.toLowerCase() === lowerQuery ? 0 : 1;
        const bExact = b.name && b.name.toLowerCase() === lowerQuery ? 0 : 1;
        if (aExact !== bExact) return aExact - bExact;

        // Then sort by type priority
        const aPriority = typePriority[a.type] || 10;
        const bPriority = typePriority[b.type] || 10;
        if (aPriority !== bPriority) return aPriority - bPriority;

        // Finally sort alphabetically
        return (a.name || '').localeCompare(b.name || '');
    });

    selectedSearchIndex = searchResults.length > 0 ? 0 : -1;
    renderSearchResults(query);
}

function renderSearchResults(query) {
    const resultsContainer = document.getElementById('search-results');

    if (searchResults.length === 0) {
        resultsContainer.innerHTML = '<div class="search-no-results">No results found</div>';
        resultsContainer.classList.add('visible');
        return;
    }

    let html = '';

    searchResults.forEach((node, index) => {
        const icon = getSearchResultIcon(node.type);
        const name = highlightMatch(node.name || node.id.substring(0, 20), query);
        const path = node.file_path ? node.file_path.split('/').slice(-3).join('/') : '';
        const type = node.type || 'unknown';
        const selected = index === selectedSearchIndex ? 'selected' : '';

        html += `<div class="search-result-item ${selected}"
                      data-index="${index}"
                      onclick="selectSearchResultByIndex(${index})"
                      onmouseenter="hoverSearchResult(${index})">`;
        html += `<span class="search-result-icon">${icon}</span>`;
        html += `<div class="search-result-info">`;
        html += `<div class="search-result-name">${name}</div>`;
        if (path) {
            html += `<div class="search-result-path">${escapeHtml(path)}</div>`;
        }
        html += `</div>`;
        html += `<span class="search-result-type">${type}</span>`;
        html += `</div>`;
    });

    html += '<div class="search-hint">↑↓ Navigate • Enter Select • Esc Close</div>';

    resultsContainer.innerHTML = html;
    resultsContainer.classList.add('visible');
}

function getSearchResultIcon(type) {
    const icons = {
        'directory': '📁',
        'file': '📄',
        'function': '⚡',
        'class': '🏛️',
        'method': '🔧',
        'module': '📦',
        'imports': '📦',
        'text': '📝',
        'code': '📝'
    };
    return icons[type] || '📄';
}

function highlightMatch(text, query) {
    if (!text || !query) return escapeHtml(text || '');

    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const index = lowerText.indexOf(lowerQuery);

    if (index === -1) return escapeHtml(text);

    const before = text.substring(0, index);
    const match = text.substring(index, index + query.length);
    const after = text.substring(index + query.length);

    return escapeHtml(before) + '<mark>' + escapeHtml(match) + '</mark>' + escapeHtml(after);
}

function updateSearchSelection() {
    const items = document.querySelectorAll('.search-result-item');
    items.forEach((item, index) => {
        item.classList.toggle('selected', index === selectedSearchIndex);
    });

    // Scroll selected item into view
    const selected = items[selectedSearchIndex];
    if (selected) {
        selected.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
}

function hoverSearchResult(index) {
    selectedSearchIndex = index;
    updateSearchSelection();
}

function selectSearchResultByIndex(index) {
    if (index >= 0 && index < searchResults.length) {
        selectSearchResult(searchResults[index]);
    }
}

function selectSearchResult(node) {
    console.log(`Search selected: ${node.name} (${node.type})`);

    // Close search dropdown
    closeSearchResults();

    // Clear input
    document.getElementById('search-input').value = '';

    // Focus on the node in the tree
    focusNodeInTree(node.id);
}

function closeSearchResults() {
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.classList.remove('visible');
    searchResults = [];
    selectedSearchIndex = -1;
}

// ============================================================================
// THEME TOGGLE
// ============================================================================

function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    // Update theme
    if (newTheme === 'light') {
        html.setAttribute('data-theme', 'light');
    } else {
        html.removeAttribute('data-theme');
    }

    // Save preference
    localStorage.setItem('theme', newTheme);

    // Update icon only
    const themeIcon = document.getElementById('theme-icon');

    if (newTheme === 'light') {
        themeIcon.textContent = '☀️';
    } else {
        themeIcon.textContent = '🌙';
    }

    console.log(`Theme toggled to: ${newTheme}`);
}

function loadThemePreference() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const html = document.documentElement;

    if (savedTheme === 'light') {
        html.setAttribute('data-theme', 'light');
        const themeIcon = document.getElementById('theme-icon');
        if (themeIcon) themeIcon.textContent = '☀️';
    }

    console.log(`Loaded theme preference: ${savedTheme}`);
}

// ============================================================================
// ANALYSIS REPORTS
// ============================================================================

function getComplexityGrade(complexity) {
    if (complexity === undefined || complexity === null) return 'N/A';
    if (complexity <= 5) return 'A';
    if (complexity <= 10) return 'B';
    if (complexity <= 15) return 'C';
    if (complexity <= 20) return 'D';
    return 'F';
}

function getGradeColor(grade) {
    const colors = {
        'A': '#2ea043',
        'B': '#1f6feb',
        'C': '#d29922',
        'D': '#f0883e',
        'F': '#da3633',
        'N/A': '#6e7681'
    };
    return colors[grade] || colors['N/A'];
}

function showComplexityReport() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '📊 Complexity Report';

    // Collect all chunk nodes with complexity data
    const chunksWithComplexity = [];

    function collectChunks(node) {
        if (chunkTypes.includes(node.type)) {
            const complexity = node.complexity !== undefined ? node.complexity : null;
            chunksWithComplexity.push({
                name: node.name,
                type: node.type,
                file_path: node.file_path || 'Unknown',
                start_line: node.start_line || 0,
                end_line: node.end_line || 0,
                complexity: complexity,
                grade: getComplexityGrade(complexity),
                node: node
            });
        }

        // Recursively process children
        const children = node.children || node._children || [];
        children.forEach(child => collectChunks(child));
    }

    // Start from treeData root
    if (treeData) {
        collectChunks(treeData);
    }

    // Calculate statistics
    const totalFunctions = chunksWithComplexity.length;
    const validComplexity = chunksWithComplexity.filter(c => c.complexity !== null);
    const avgComplexity = validComplexity.length > 0
        ? (validComplexity.reduce((sum, c) => sum + c.complexity, 0) / validComplexity.length).toFixed(2)
        : 'N/A';

    // Count by grade
    const gradeCounts = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0, 'N/A': 0
    };
    chunksWithComplexity.forEach(c => {
        gradeCounts[c.grade]++;
    });

    // Sort by complexity (highest first)
    const sortedChunks = [...chunksWithComplexity].sort((a, b) => {
        if (a.complexity === null) return 1;
        if (b.complexity === null) return -1;
        return b.complexity - a.complexity;
    });

    // Build HTML
    let html = '<div class="complexity-report">';

    // Summary Stats
    html += '<div class="complexity-summary">';
    html += '<div class="summary-grid">';
    html += `<div class="summary-card">
        <div class="summary-label">Total Functions</div>
        <div class="summary-value">${totalFunctions}</div>
    </div>`;
    html += `<div class="summary-card">
        <div class="summary-label">Average Complexity</div>
        <div class="summary-value">${avgComplexity}</div>
    </div>`;
    html += `<div class="summary-card">
        <div class="summary-label">With Complexity Data</div>
        <div class="summary-value">${validComplexity.length}</div>
    </div>`;
    html += '</div>';

    // Grade Distribution
    html += '<div class="grade-distribution">';
    html += '<div class="distribution-title">Grade Distribution</div>';
    html += '<div class="distribution-bars">';

    const maxCount = Math.max(...Object.values(gradeCounts));
    ['A', 'B', 'C', 'D', 'F', 'N/A'].forEach(grade => {
        const count = gradeCounts[grade];
        const percentage = totalFunctions > 0 ? (count / totalFunctions * 100) : 0;
        const barWidth = maxCount > 0 ? (count / maxCount * 100) : 0;
        html += `
            <div class="distribution-row">
                <div class="distribution-grade" style="color: ${getGradeColor(grade)}">${grade}</div>
                <div class="distribution-bar-container">
                    <div class="distribution-bar" style="width: ${barWidth}%; background: ${getGradeColor(grade)}"></div>
                </div>
                <div class="distribution-count">${count} (${percentage.toFixed(1)}%)</div>
            </div>
        `;
    });
    html += '</div></div>';
    html += '</div>';

    // Complexity Hotspots Table
    html += '<div class="complexity-hotspots">';
    html += '<h3 class="section-title">Complexity Hotspots</h3>';

    if (sortedChunks.length === 0) {
        html += '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No functions found with complexity data.</p>';
    } else {
        html += '<div class="hotspots-table-container">';
        html += '<table class="hotspots-table">';
        html += `
            <thead>
                <tr>
                    <th>Name</th>
                    <th>File</th>
                    <th>Lines</th>
                    <th>Complexity</th>
                    <th>Grade</th>
                </tr>
            </thead>
            <tbody>
        `;

        sortedChunks.forEach(chunk => {
            const lines = chunk.end_line > 0 ? chunk.end_line - chunk.start_line + 1 : 'N/A';
            const complexityDisplay = chunk.complexity !== null ? chunk.complexity.toFixed(1) : 'N/A';
            const gradeColor = getGradeColor(chunk.grade);

            // Get relative file path
            const fileName = chunk.file_path.split('/').pop() || chunk.file_path;
            const lineRange = chunk.start_line > 0 ? `L${chunk.start_line}-${chunk.end_line}` : '';

            html += `
                <tr class="hotspot-row" onclick='navigateToChunk(${JSON.stringify(chunk.name)})'>
                    <td class="hotspot-name">${escapeHtml(chunk.name)}</td>
                    <td class="hotspot-file" title="${escapeHtml(chunk.file_path)}">${escapeHtml(fileName)}</td>
                    <td class="hotspot-lines">${lines}</td>
                    <td class="hotspot-complexity">${complexityDisplay}</td>
                    <td class="hotspot-grade">
                        <span class="grade-badge" style="background: ${gradeColor}">${chunk.grade}</span>
                    </td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
    }

    html += '</div>'; // complexity-hotspots
    html += '</div>'; // complexity-report

    viewerContent.innerHTML = html;

    // Hide section dropdown for reports (no code sections)
    const sectionNav = document.getElementById('section-nav');
    if (sectionNav) {
        sectionNav.style.display = 'none';
    }
}

function navigateToChunk(chunkName) {
    // Find the chunk node in the tree
    function findChunk(node) {
        if (chunkTypes.includes(node.type) && node.name === chunkName) {
            return node;
        }
        const children = node.children || node._children || [];
        for (const child of children) {
            const found = findChunk(child);
            if (found) return found;
        }
        return null;
    }

    if (treeData) {
        const chunk = findChunk(treeData);
        if (chunk) {
            // Display the chunk content
            displayChunkContent(chunk);

            // Highlight the node in the visualization
            highlightNode(chunk.id);
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function highlightNode(nodeId) {
    // Remove existing highlights
    d3.selectAll('.node').classed('node-highlight', false);

    // Add highlight to the target node
    d3.selectAll('.node')
        .filter(d => d.data.id === nodeId)
        .classed('node-highlight', true);
}

// ============================================================================
// CODE SMELLS DETECTION AND REPORTING
// ============================================================================

function detectCodeSmells(nodes) {
    const smells = [];

    function analyzeNode(node) {
        // Only analyze code chunks
        if (!chunkTypes.includes(node.type)) {
            const children = node.children || node._children || [];
            children.forEach(child => analyzeNode(child));
            return;
        }

        const lineCount = (node.end_line && node.start_line)
            ? node.end_line - node.start_line + 1
            : 0;
        const complexity = node.complexity || 0;

        // 1. Long Method - Functions with > 50 lines
        if (node.type === 'function' || node.type === 'method') {
            if (lineCount > 100) {
                smells.push({
                    type: 'Long Method',
                    severity: 'error',
                    node: node,
                    details: `${lineCount} lines (very long)`
                });
            } else if (lineCount > 50) {
                smells.push({
                    type: 'Long Method',
                    severity: 'warning',
                    node: node,
                    details: `${lineCount} lines`
                });
            }
        }

        // 2. High Complexity - Functions with complexity > 15
        if ((node.type === 'function' || node.type === 'method') && complexity > 0) {
            if (complexity > 20) {
                smells.push({
                    type: 'High Complexity',
                    severity: 'error',
                    node: node,
                    details: `Complexity: ${complexity.toFixed(1)} (very complex)`
                });
            } else if (complexity > 15) {
                smells.push({
                    type: 'High Complexity',
                    severity: 'warning',
                    node: node,
                    details: `Complexity: ${complexity.toFixed(1)}`
                });
            }
        }

        // 3. Deep Nesting - Proxy using complexity > 20
        if ((node.type === 'function' || node.type === 'method') && complexity > 20) {
            smells.push({
                type: 'Deep Nesting',
                severity: complexity > 25 ? 'error' : 'warning',
                node: node,
                details: `Complexity: ${complexity.toFixed(1)} (likely deep nesting)`
            });
        }

        // 4. God Class - Classes with > 20 methods or > 500 lines
        if (node.type === 'class') {
            const children = node.children || node._children || [];
            const methodCount = children.filter(c => c.type === 'method').length;

            if (methodCount > 30 || lineCount > 800) {
                smells.push({
                    type: 'God Class',
                    severity: 'error',
                    node: node,
                    details: `${methodCount} methods, ${lineCount} lines (very large)`
                });
            } else if (methodCount > 20 || lineCount > 500) {
                smells.push({
                    type: 'God Class',
                    severity: 'warning',
                    node: node,
                    details: `${methodCount} methods, ${lineCount} lines`
                });
            }
        }

        // Recursively process children
        const children = node.children || node._children || [];
        children.forEach(child => analyzeNode(child));
    }

    if (nodes) {
        analyzeNode(nodes);
    }

    return smells;
}

function showCodeSmells() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '🔍 Code Smells';

    // Detect all code smells
    const allSmells = detectCodeSmells(treeData);

    // Count by type and severity
    const smellCounts = {
        'Long Method': { total: 0, warning: 0, error: 0 },
        'High Complexity': { total: 0, warning: 0, error: 0 },
        'Deep Nesting': { total: 0, warning: 0, error: 0 },
        'God Class': { total: 0, warning: 0, error: 0 }
    };

    let totalWarnings = 0;
    let totalErrors = 0;

    allSmells.forEach(smell => {
        smellCounts[smell.type].total++;
        smellCounts[smell.type][smell.severity]++;
        if (smell.severity === 'warning') totalWarnings++;
        if (smell.severity === 'error') totalErrors++;
    });

    // Build HTML
    let html = '<div class="code-smells-report">';

    // Summary Cards
    html += '<div class="smell-summary-grid">';
    html += `
        <div class="smell-summary-card">
            <div class="smell-card-header">
                <span class="smell-card-icon">🔍</span>
                <div class="smell-card-title">Total Smells</div>
            </div>
            <div class="smell-card-count">${allSmells.length}</div>
        </div>
        <div class="smell-summary-card warning">
            <div class="smell-card-header">
                <span class="smell-card-icon">⚠️</span>
                <div class="smell-card-title">Warnings</div>
            </div>
            <div class="smell-card-count" style="color: var(--warning)">${totalWarnings}</div>
        </div>
        <div class="smell-summary-card error">
            <div class="smell-card-header">
                <span class="smell-card-icon">🚨</span>
                <div class="smell-card-title">Errors</div>
            </div>
            <div class="smell-card-count" style="color: var(--error)">${totalErrors}</div>
        </div>
    `;
    html += '</div>';

    // Filters
    html += '<div class="smell-filters">';
    html += '<div class="filter-title">Filter by Type</div>';
    html += '<div class="filter-checkboxes">';

    Object.keys(smellCounts).forEach(type => {
        const count = smellCounts[type].total;
        html += `
            <div class="filter-checkbox-item">
                <input type="checkbox" id="filter-${type.replace(/\\s+/g, '-')}"
                       checked onchange="filterCodeSmells()">
                <label class="filter-checkbox-label" for="filter-${type.replace(/\\s+/g, '-')}">${type}</label>
                <span class="filter-checkbox-count">${count}</span>
            </div>
        `;
    });

    html += '</div></div>';

    // Smells Table
    html += '<div id="smells-table-wrapper">';
    html += '<h3 class="section-title">Detected Code Smells</h3>';

    if (allSmells.length === 0) {
        html += '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No code smells detected! Great job! 🎉</p>';
    } else {
        html += '<div class="smells-table-container">';
        html += '<table class="smells-table">';
        html += `
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Name</th>
                    <th>File</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody id="smells-table-body">
        `;

        // Sort by severity (error first) then by type
        const sortedSmells = [...allSmells].sort((a, b) => {
            if (a.severity !== b.severity) {
                return a.severity === 'error' ? -1 : 1;
            }
            return a.type.localeCompare(b.type);
        });

        sortedSmells.forEach(smell => {
            const fileName = smell.node.file_path ? smell.node.file_path.split('/').pop() : 'Unknown';
            const severityIcon = smell.severity === 'error' ? '🚨' : '⚠️';

            html += `
                <tr class="smell-row" data-smell-type="${smell.type.replace(/\\s+/g, '-')}"
                    onclick='navigateToChunk(${JSON.stringify(smell.node.name)})'>
                    <td><span class="smell-type-badge">${escapeHtml(smell.type)}</span></td>
                    <td><span class="severity-badge ${smell.severity}">${severityIcon} ${smell.severity.toUpperCase()}</span></td>
                    <td class="smell-name">${escapeHtml(smell.node.name)}</td>
                    <td class="smell-file" title="${escapeHtml(smell.node.file_path || '')}">${escapeHtml(fileName)}</td>
                    <td class="smell-details">${escapeHtml(smell.details)}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div>';
    }

    html += '</div>'; // smells-table-wrapper
    html += '</div>'; // code-smells-report

    viewerContent.innerHTML = html;

    // Hide section dropdown for reports (no code sections)
    const sectionNav = document.getElementById('section-nav');
    if (sectionNav) {
        sectionNav.style.display = 'none';
    }
}

function filterCodeSmells() {
    const rows = document.querySelectorAll('.smell-row');

    rows.forEach(row => {
        const smellType = row.getAttribute('data-smell-type');
        const checkbox = document.getElementById(`filter-${smellType}`);

        if (checkbox && checkbox.checked) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// ============================================================================
// DEPENDENCIES ANALYSIS AND REPORTING
// ============================================================================

function showDependencies() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '🔗 Code Structure';

    // Code file extensions (exclude docs like .md, .txt, .rst)
    const codeExtensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.ex', '.exs', '.clj', '.vue', '.svelte'];

    function isCodeFile(filePath) {
        const ext = filePath.substring(filePath.lastIndexOf('.')).toLowerCase();
        return codeExtensions.includes(ext);
    }

    // Build directory structure from nodes (code files only)
    const dirStructure = new Map();

    allNodes.forEach(node => {
        if (node.type === 'file' && node.file_path && isCodeFile(node.file_path)) {
            const parts = node.file_path.split('/');
            const fileName = parts.pop();
            const dirPath = parts.join('/') || '/';

            if (!dirStructure.has(dirPath)) {
                dirStructure.set(dirPath, []);
            }
            dirStructure.get(dirPath).push({
                name: fileName,
                path: node.file_path,
                chunks: allNodes.filter(n => n.file_path === node.file_path && n.type !== 'file').length
            });
        }
    });

    // Calculate stats
    const totalDirs = dirStructure.size;
    const totalFiles = Array.from(dirStructure.values()).reduce((sum, files) => sum + files.length, 0);
    const totalChunks = allNodes.filter(n => n.type !== 'file' && n.type !== 'directory').length;

    let html = `
        <div class="report-section">
            <h3>📁 Directory Overview</h3>
            <p style="color: var(--text-secondary); margin-bottom: 15px;">Showing code organization by directory structure.</p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${totalDirs}</div>
                    <div class="metric-label">Directories</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${totalFiles}</div>
                    <div class="metric-label">Files</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${totalChunks}</div>
                    <div class="metric-label">Code Chunks</div>
                </div>
            </div>
        </div>
        <div class="report-section">
            <h3>📂 Directory Structure</h3>
    `;

    // Sort directories
    const sortedDirs = Array.from(dirStructure.entries()).sort((a, b) => a[0].localeCompare(b[0]));

    sortedDirs.forEach(([dir, files]) => {
        const dirDisplay = dir === '/' ? 'Root' : dir;
        const totalChunksInDir = files.reduce((sum, f) => sum + f.chunks, 0);

        html += `
            <div class="dependency-item" style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 6px; border-left: 3px solid var(--accent-blue);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong style="color: var(--accent-blue);">📁 ${escapeHtml(dirDisplay)}</strong>
                    <span style="color: var(--text-secondary); font-size: 12px;">${files.length} files, ${totalChunksInDir} chunks</span>
                </div>
                <ul style="margin: 0; padding: 0 0 0 20px; list-style: none;">
        `;

        files.sort((a, b) => a.name.localeCompare(b.name)).forEach(file => {
            html += `<li style="padding: 4px 0; color: var(--text-primary);">
                📄 ${escapeHtml(file.name)}
                <span style="color: var(--text-secondary); font-size: 11px;">(${file.chunks} chunks)</span>
            </li>`;
        });

        html += `</ul></div>`;
    });

    html += '</div>';

    viewerContent.innerHTML = html;

    // Hide section dropdown for reports (no code sections)
    const sectionNav = document.getElementById('section-nav');
    if (sectionNav) {
        sectionNav.style.display = 'none';
    }
}

function buildFileDependencyGraph() {
    // Map: file_path -> { dependsOn: Set<file_path>, usedBy: Set<file_path> }
    const fileDeps = new Map();

    allLinks.forEach(link => {
        if (link.type === 'caller') {
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

function findCircularDeps(fileDeps) {
    // Simple cycle detection using DFS
    const cycles = [];
    const visited = new Set();
    const recStack = new Set();
    const pathStack = [];

    function dfs(filePath) {
        visited.add(filePath);
        recStack.add(filePath);
        pathStack.push(filePath);

        const deps = fileDeps.get(filePath);
        if (deps && deps.dependsOn) {
            for (const depFile of deps.dependsOn) {
                if (!visited.has(depFile)) {
                    dfs(depFile);
                } else if (recStack.has(depFile)) {
                    // Found a cycle
                    const cycleStartIndex = pathStack.indexOf(depFile);
                    if (cycleStartIndex !== -1) {
                        const cycle = pathStack.slice(cycleStartIndex);
                        cycle.push(depFile); // Complete the cycle
                        // Check if this cycle is already recorded
                        const cycleStr = cycle.sort().join('|');
                        if (!cycles.some(c => c.sort().join('|') === cycleStr)) {
                            cycles.push([...cycle]);
                        }
                    }
                }
            }
        }

        pathStack.pop();
        recStack.delete(filePath);
    }

    for (const filePath of fileDeps.keys()) {
        if (!visited.has(filePath)) {
            dfs(filePath);
        }
    }

    return cycles;
}

function toggleDependencyDetails(rowId, index) {
    const detailsRow = document.getElementById(`${rowId}-details`);
    const btn = document.querySelector(`#${rowId} .expand-btn`);

    if (detailsRow.style.display === 'none') {
        detailsRow.style.display = '';
        btn.textContent = '▲';
    } else {
        detailsRow.style.display = 'none';
        btn.textContent = '▼';
    }
}

// ============================================================================
// TRENDS / METRICS SNAPSHOT
// ============================================================================

function showTrends() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '📈 Codebase Metrics Snapshot';

    // Calculate metrics from current codebase
    const metrics = calculateCodebaseMetrics();

    // Build HTML
    let html = '<div class="trends-report">';

    // Snapshot Banner
    html += '<div class="snapshot-banner">';
    html += '<div class="snapshot-header">';
    html += '<div class="snapshot-icon">📊</div>';
    html += '<div class="snapshot-info">';
    html += '<div class="snapshot-title">Codebase Metrics Snapshot</div>';
    html += `<div class="snapshot-timestamp">Generated: ${new Date().toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    })}</div>`;
    html += '</div></div>';
    html += '<div class="snapshot-description">';
    html += 'This snapshot serves as the baseline for future trend tracking. ';
    html += 'With git history analysis, you could track how these metrics evolve over time.';
    html += '</div>';
    html += '</div>';

    // Key Metrics Cards
    html += '<div class="metrics-section">';
    html += '<h3 class="section-title">Key Metrics</h3>';
    html += '<div class="metrics-grid">';

    html += `<div class="metric-card">
        <div class="metric-icon">📝</div>
        <div class="metric-value">${metrics.totalLines.toLocaleString()}</div>
        <div class="metric-label">Lines of Code</div>
    </div>`;

    html += `<div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-value">${metrics.totalFunctions}</div>
        <div class="metric-label">Functions/Methods</div>
    </div>`;

    html += `<div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">${metrics.totalClasses}</div>
        <div class="metric-label">Classes</div>
    </div>`;

    html += `<div class="metric-card">
        <div class="metric-icon">📄</div>
        <div class="metric-value">${metrics.totalFiles}</div>
        <div class="metric-label">Files</div>
    </div>`;

    html += '</div></div>';

    // Code Health Score
    html += '<div class="health-section">';
    html += '<h3 class="section-title">Code Health Score</h3>';
    html += '<div class="health-card">';

    const healthInfo = getHealthScoreInfo(metrics.healthScore);
    html += `<div class="health-score-display">
        <div class="health-score-value">${metrics.healthScore}/100</div>
        <div class="health-score-label">${healthInfo.emoji} ${healthInfo.label}</div>
    </div>`;

    html += '<div class="health-progress-container">';
    html += `<div class="health-progress-bar" style="width: ${metrics.healthScore}%; background: ${healthInfo.color}"></div>`;
    html += '</div>';

    html += `<div class="health-description">${healthInfo.description}</div>`;
    html += '</div></div>';

    // Complexity Distribution
    html += '<div class="distribution-section">';
    html += '<h3 class="section-title">Complexity Distribution</h3>';
    html += '<div class="distribution-chart">';

    const complexityDist = metrics.complexityDistribution;
    const maxPct = Math.max(...Object.values(complexityDist));

    ['A', 'B', 'C', 'D', 'F'].forEach(grade => {
        const pct = complexityDist[grade] || 0;
        const barWidth = maxPct > 0 ? (pct / maxPct * 100) : 0;
        const color = getGradeColor(grade);
        const range = getComplexityRange(grade);

        html += `<div class="distribution-bar-row">
            <div class="distribution-bar-label">
                <span class="distribution-grade" style="color: ${color}">${grade}</span>
                <span class="distribution-range">${range}</span>
            </div>
            <div class="distribution-bar-container">
                <div class="distribution-bar-fill" style="width: ${barWidth}%; background: ${color}"></div>
            </div>
            <div class="distribution-bar-value">${pct.toFixed(1)}%</div>
        </div>`;
    });

    html += '</div></div>';

    // Function Size Distribution
    html += '<div class="size-distribution-section">';
    html += '<h3 class="section-title">Function Size Distribution</h3>';
    html += '<div class="distribution-chart">';

    const sizeDist = metrics.sizeDistribution;
    const maxSizePct = Math.max(...Object.values(sizeDist));

    [
        { key: 'small', label: 'Small (1-20 lines)', color: '#238636' },
        { key: 'medium', label: 'Medium (21-50 lines)', color: '#1f6feb' },
        { key: 'large', label: 'Large (51-100 lines)', color: '#d29922' },
        { key: 'veryLarge', label: 'Very Large (100+ lines)', color: '#da3633' }
    ].forEach(({ key, label, color }) => {
        const pct = sizeDist[key] || 0;
        const barWidth = maxSizePct > 0 ? (pct / maxSizePct * 100) : 0;

        html += `<div class="distribution-bar-row">
            <div class="distribution-bar-label">
                <span class="size-label">${label}</span>
            </div>
            <div class="distribution-bar-container">
                <div class="distribution-bar-fill" style="width: ${barWidth}%; background: ${color}"></div>
            </div>
            <div class="distribution-bar-value">${pct.toFixed(1)}%</div>
        </div>`;
    });

    html += '</div></div>';

    // Historical Trends Section
    html += '<div class="trends-section">';
    html += '<h3 class="section-title">📊 Historical Trends</h3>';

    // Check if trend data is available
    if (window.graphTrendData && window.graphTrendData.entries && window.graphTrendData.entries.length > 0) {
        const trendEntries = window.graphTrendData.entries;

        // Render trend charts
        html += '<div class="trends-container">';
        html += '<div id="health-score-chart" class="trend-chart"></div>';
        html += '<div id="complexity-chart" class="trend-chart"></div>';
        html += '<div id="files-chunks-chart" class="trend-chart"></div>';
        html += '</div>';

        html += `<div class="trend-info">Showing ${trendEntries.length} data points from ${trendEntries[0].date} to ${trendEntries[trendEntries.length - 1].date}</div>`;
    } else {
        // No trend data yet - show placeholder
        html += '<div class="future-placeholder">';
        html += '<div class="future-icon">📊</div>';
        html += '<div class="future-title">No Historical Data Yet</div>';
        html += '<div class="future-description">';
        html += 'Trend data will be collected automatically after each indexing operation. ';
        html += 'Run <code>mcp-vector-search index</code> to generate the first snapshot.';
        html += '</div>';
        html += '</div>';
    }

    html += '</div>'; // trends-section

    html += '</div>'; // trends-report

    viewerContent.innerHTML = html;

    // Hide section dropdown for reports (no code sections)
    const sectionNav = document.getElementById('section-nav');
    if (sectionNav) {
        sectionNav.style.display = 'none';
    }

    // Render D3 charts if trend data is available
    if (window.graphTrendData && window.graphTrendData.entries && window.graphTrendData.entries.length > 0) {
        renderTrendCharts(window.graphTrendData.entries);
    }
}

// ============================================================================
// REMEDIATION REPORT GENERATION
// ============================================================================

function generateRemediationReport() {
    // Gather complexity data (metrics are stored directly on nodes, not in a metrics object)
    const complexityData = [];
    allNodes.forEach(node => {
        if (node.complexity !== undefined && node.complexity !== null) {
            const complexity = node.complexity;
            let grade = 'A';
            if (complexity > 40) grade = 'F';
            else if (complexity > 30) grade = 'D';
            else if (complexity > 20) grade = 'C';
            else if (complexity > 10) grade = 'B';

            // Calculate lines from start_line and end_line
            const lines = (node.end_line && node.start_line) ? (node.end_line - node.start_line + 1) : 0;

            if (grade !== 'A' && grade !== 'B') {  // Only include C, D, F
                complexityData.push({
                    name: node.name || node.id,
                    file: node.file_path || 'Unknown',
                    type: node.type || 'unknown',
                    complexity: complexity,
                    grade: grade,
                    lines: lines
                });
            }
        }
    });

    // Gather code smell data
    const smells = [];
    allNodes.forEach(node => {
        const file = node.file_path || 'Unknown';
        const name = node.name || node.id;
        const lines = (node.end_line && node.start_line) ? (node.end_line - node.start_line + 1) : 0;
        const complexity = node.complexity || 0;
        const depth = node.depth || 0;

        // Long Method
        if (lines > 50) {
            smells.push({
                file, name,
                smell: 'Long Method',
                severity: lines > 100 ? 'error' : 'warning',
                detail: `${lines} lines (recommended: <50)`
            });
        }

        // High Complexity
        if (complexity > 15) {
            smells.push({
                file, name,
                smell: 'High Complexity',
                severity: complexity > 25 ? 'error' : 'warning',
                detail: `Complexity: ${complexity} (recommended: <15)`
            });
        }

        // Deep Nesting (using depth field)
        if (depth > 4) {
            smells.push({
                file, name,
                smell: 'Deep Nesting',
                severity: depth > 6 ? 'error' : 'warning',
                detail: `Depth: ${depth} (recommended: <4)`
            });
        }

        // God Class (for classes only)
        if (node.type === 'class' && lines > 300) {
            smells.push({
                file, name,
                smell: 'God Class',
                severity: 'error',
                detail: `${lines} lines - consider breaking into smaller classes`
            });
        }
    });

    // Sort by severity (errors first) then by file
    complexityData.sort((a, b) => {
        const gradeOrder = { F: 0, D: 1, C: 2 };
        return (gradeOrder[a.grade] || 99) - (gradeOrder[b.grade] || 99);
    });

    smells.sort((a, b) => {
        if (a.severity !== b.severity) {
            return a.severity === 'error' ? -1 : 1;
        }
        return a.file.localeCompare(b.file);
    });

    // Generate Markdown
    const date = new Date().toISOString().split('T')[0];
    let markdown = `# Code Remediation Report
Generated: ${date}

## Summary

- **High Complexity Items**: ${complexityData.length}
- **Code Smells Detected**: ${smells.length}
- **Critical Issues (Errors)**: ${smells.filter(s => s.severity === 'error').length}

---

## 🔴 Priority: High Complexity Code

These functions/methods have complexity scores that make them difficult to maintain and test.

| Grade | Name | File | Complexity | Lines |
|-------|------|------|------------|-------|
`;

    complexityData.forEach(item => {
        const gradeEmoji = item.grade === 'F' ? '🔴' : item.grade === 'D' ? '🟠' : '🟡';
        markdown += `| ${gradeEmoji} ${item.grade} | \\`${item.name}\\` | ${item.file} | ${item.complexity} | ${item.lines} |\\n`;
    });

    markdown += `
---

## 🔍 Code Smells

### Critical Issues (Errors)

`;

    const errors = smells.filter(s => s.severity === 'error');
    if (errors.length === 0) {
        markdown += '_No critical issues found._\\n';
    } else {
        markdown += '| Smell | Name | File | Detail |\\n|-------|------|------|--------|\\n';
        errors.forEach(s => {
            markdown += `| 🔴 ${s.smell} | \\`${s.name}\\` | ${s.file} | ${s.detail} |\\n`;
        });
    }

    markdown += `
### Warnings

`;

    const warnings = smells.filter(s => s.severity === 'warning');
    if (warnings.length === 0) {
        markdown += '_No warnings found._\\n';
    } else {
        markdown += '| Smell | Name | File | Detail |\\n|-------|------|------|--------|\\n';
        warnings.forEach(s => {
            markdown += `| 🟡 ${s.smell} | \\`${s.name}\\` | ${s.file} | ${s.detail} |\\n`;
        });
    }

    markdown += `
---

## Recommended Actions

1. **Start with Grade F items** - These have the highest complexity and are hardest to maintain
2. **Address Critical code smells** - God Classes and deeply nested code should be refactored
3. **Break down long methods** - Extract helper functions to reduce complexity
4. **Add tests before refactoring** - Ensure behavior is preserved

---

_Generated by MCP Vector Search Visualization_
`;

    // Save the file with dialog (or fallback to download)
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const defaultFilename = `remediation-report-${date}.md`;

    async function saveWithDialog() {
        try {
            // Use File System Access API if available (shows save dialog)
            if ('showSaveFilePicker' in window) {
                const handle = await window.showSaveFilePicker({
                    suggestedName: defaultFilename,
                    types: [{
                        description: 'Markdown files',
                        accept: { 'text/markdown': ['.md'] }
                    }]
                });
                const writable = await handle.createWritable();
                await writable.write(blob);
                await writable.close();
                return handle.name;
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                return null; // User cancelled
            }
            console.warn('Save dialog failed, falling back to download:', err);
        }

        // Fallback: standard download
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = defaultFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return defaultFilename;
    }

    saveWithDialog().then(savedFilename => {
        if (savedFilename === null) {
            // User cancelled - don't show confirmation
            return;
        }

        // Show confirmation
        openViewerPanel();
        document.getElementById('viewer-title').textContent = '📋 Report Saved';
        document.getElementById('viewer-content').innerHTML = `
            <div class="report-section">
                <h3>✅ Remediation Report Generated</h3>
                <p>The report has been saved as <code>${escapeHtml(savedFilename)}</code></p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">${complexityData.length}</div>
                    <div class="metric-label">Complexity Issues</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${smells.length}</div>
                    <div class="metric-label">Code Smells</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${errors.length}</div>
                    <div class="metric-label">Critical Errors</div>
                </div>
            </div>
            <p style="margin-top: 15px; color: var(--text-secondary);">Share this report with your team for prioritized remediation.</p>
        </div>
    `;

        // Hide section dropdown for reports (no code sections)
        const sectionNav = document.getElementById('section-nav');
        if (sectionNav) {
            sectionNav.style.display = 'none';
        }
    });
}

// Render trend line charts using D3
function renderTrendCharts(entries) {
    // Chart dimensions
    const margin = {top: 20, right: 30, bottom: 40, left: 50};
    const width = 600 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    // Parse dates
    const parseDate = d3.timeParse('%Y-%m-%d');
    entries.forEach(d => {
        d.parsedDate = parseDate(d.date);
    });

    // 1. Health Score Chart
    renderLineChart('#health-score-chart', entries, {
        title: 'Health Score Over Time',
        width, height, margin,
        yAccessor: d => d.metrics.health_score || 0,
        yLabel: 'Health Score',
        color: '#238636',
        yDomain: [0, 100]
    });

    // 2. Average Complexity Chart
    renderLineChart('#complexity-chart', entries, {
        title: 'Average Complexity Over Time',
        width, height, margin,
        yAccessor: d => d.metrics.avg_complexity || 0,
        yLabel: 'Avg Complexity',
        color: '#d29922',
        yDomain: [0, d3.max(entries, d => d.metrics.avg_complexity || 0) * 1.1]
    });

    // 3. Files and Chunks Chart (dual line)
    renderDualLineChart('#files-chunks-chart', entries, {
        title: 'Files and Chunks Over Time',
        width, height, margin,
        y1Accessor: d => d.metrics.total_files || 0,
        y2Accessor: d => d.metrics.total_chunks || 0,
        y1Label: 'Files',
        y2Label: 'Chunks',
        color1: '#1f6feb',
        color2: '#8957e5'
    });
}

// Render single line chart
function renderLineChart(selector, data, config) {
    const svg = d3.select(selector)
        .append('svg')
        .attr('width', config.width + config.margin.left + config.margin.right)
        .attr('height', config.height + config.margin.top + config.margin.bottom)
        .append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);

    // Add title
    svg.append('text')
        .attr('x', config.width / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text(config.title);

    // Create scales
    const xScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.parsedDate))
        .range([0, config.width]);

    const yScale = d3.scaleLinear()
        .domain(config.yDomain || [0, d3.max(data, config.yAccessor)])
        .range([config.height, 0]);

    // Create line generator
    const line = d3.line()
        .x(d => xScale(d.parsedDate))
        .y(d => yScale(config.yAccessor(d)))
        .curve(d3.curveMonotoneX);

    // Add X axis
    svg.append('g')
        .attr('transform', `translate(0,${config.height})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat('%b %d')))
        .selectAll('text')
        .style('font-size', '11px');

    // Add Y axis
    svg.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll('text')
        .style('font-size', '11px');

    // Add Y axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - config.margin.left + 10)
        .attr('x', 0 - (config.height / 2))
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(config.yLabel);

    // Add line path
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', config.color)
        .attr('stroke-width', 2)
        .attr('d', line);

    // Add dots
    svg.selectAll('.dot')
        .data(data)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.parsedDate))
        .attr('cy', d => yScale(config.yAccessor(d)))
        .attr('r', 4)
        .attr('fill', config.color)
        .style('cursor', 'pointer')
        .append('title')
        .text(d => `${d.date}: ${config.yAccessor(d).toFixed(1)}`);
}

// Render dual line chart (two Y axes)
function renderDualLineChart(selector, data, config) {
    const svg = d3.select(selector)
        .append('svg')
        .attr('width', config.width + config.margin.left + config.margin.right)
        .attr('height', config.height + config.margin.top + config.margin.bottom)
        .append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);

    // Add title
    svg.append('text')
        .attr('x', config.width / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', 'bold')
        .text(config.title);

    // Create scales
    const xScale = d3.scaleTime()
        .domain(d3.extent(data, d => d.parsedDate))
        .range([0, config.width]);

    const y1Scale = d3.scaleLinear()
        .domain([0, d3.max(data, config.y1Accessor) * 1.1])
        .range([config.height, 0]);

    const y2Scale = d3.scaleLinear()
        .domain([0, d3.max(data, config.y2Accessor) * 1.1])
        .range([config.height, 0]);

    // Create line generators
    const line1 = d3.line()
        .x(d => xScale(d.parsedDate))
        .y(d => y1Scale(config.y1Accessor(d)))
        .curve(d3.curveMonotoneX);

    const line2 = d3.line()
        .x(d => xScale(d.parsedDate))
        .y(d => y2Scale(config.y2Accessor(d)))
        .curve(d3.curveMonotoneX);

    // Add X axis
    svg.append('g')
        .attr('transform', `translate(0,${config.height})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat('%b %d')))
        .selectAll('text')
        .style('font-size', '11px');

    // Add Y1 axis (left)
    svg.append('g')
        .call(d3.axisLeft(y1Scale).ticks(5))
        .selectAll('text')
        .style('font-size', '11px')
        .style('fill', config.color1);

    // Add Y2 axis (right)
    svg.append('g')
        .attr('transform', `translate(${config.width},0)`)
        .call(d3.axisRight(y2Scale).ticks(5))
        .selectAll('text')
        .style('font-size', '11px')
        .style('fill', config.color2);

    // Add line 1
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', config.color1)
        .attr('stroke-width', 2)
        .attr('d', line1);

    // Add line 2
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', config.color2)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('d', line2);

    // Add legend
    const legend = svg.append('g')
        .attr('transform', `translate(${config.width - 120}, 10)`);

    legend.append('line')
        .attr('x1', 0).attr('x2', 20)
        .attr('y1', 5).attr('y2', 5)
        .attr('stroke', config.color1)
        .attr('stroke-width', 2);
    legend.append('text')
        .attr('x', 25).attr('y', 9)
        .style('font-size', '11px')
        .text(config.y1Label);

    legend.append('line')
        .attr('x1', 0).attr('x2', 20)
        .attr('y1', 20).attr('y2', 20)
        .attr('stroke', config.color2)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');
    legend.append('text')
        .attr('x', 25).attr('y', 24)
        .style('font-size', '11px')
        .text(config.y2Label);
}

function calculateCodebaseMetrics() {
    const metrics = {
        totalLines: 0,
        totalFunctions: 0,
        totalClasses: 0,
        totalFiles: 0,
        complexityDistribution: { A: 0, B: 0, C: 0, D: 0, F: 0 },
        sizeDistribution: { small: 0, medium: 0, large: 0, veryLarge: 0 },
        healthScore: 0
    };

    const chunksWithComplexity = [];

    function analyzeNode(node) {
        // Count files
        if (node.type === 'file') {
            metrics.totalFiles++;
        }

        // Count classes
        if (node.type === 'class') {
            metrics.totalClasses++;
        }

        // Count functions and methods
        if (node.type === 'function' || node.type === 'method') {
            metrics.totalFunctions++;

            // Calculate lines
            const lineCount = (node.end_line && node.start_line)
                ? node.end_line - node.start_line + 1
                : 0;
            metrics.totalLines += lineCount;

            // Size distribution
            if (lineCount <= 20) {
                metrics.sizeDistribution.small++;
            } else if (lineCount <= 50) {
                metrics.sizeDistribution.medium++;
            } else if (lineCount <= 100) {
                metrics.sizeDistribution.large++;
            } else {
                metrics.sizeDistribution.veryLarge++;
            }

            // Complexity distribution
            if (node.complexity !== undefined && node.complexity !== null) {
                const grade = getComplexityGrade(node.complexity);
                if (grade in metrics.complexityDistribution) {
                    metrics.complexityDistribution[grade]++;
                }
                chunksWithComplexity.push(node);
            }
        }

        // Recursively process children
        const children = node.children || node._children || [];
        children.forEach(child => analyzeNode(child));
    }

    if (treeData) {
        analyzeNode(treeData);
    }

    // Convert complexity counts to percentages
    const totalWithComplexity = chunksWithComplexity.length;
    if (totalWithComplexity > 0) {
        Object.keys(metrics.complexityDistribution).forEach(grade => {
            metrics.complexityDistribution[grade] =
                (metrics.complexityDistribution[grade] / totalWithComplexity) * 100;
        });
    }

    // Convert size counts to percentages
    const totalFuncs = metrics.totalFunctions;
    if (totalFuncs > 0) {
        Object.keys(metrics.sizeDistribution).forEach(size => {
            metrics.sizeDistribution[size] =
                (metrics.sizeDistribution[size] / totalFuncs) * 100;
        });
    }

    // Calculate health score
    metrics.healthScore = calculateHealthScore(chunksWithComplexity);

    return metrics;
}

function calculateHealthScore(chunks) {
    if (chunks.length === 0) return 100;

    let score = 0;
    chunks.forEach(chunk => {
        const grade = getComplexityGrade(chunk.complexity);
        const gradeScores = { A: 100, B: 80, C: 60, D: 40, F: 20 };
        score += gradeScores[grade] || 50;
    });

    return Math.round(score / chunks.length);
}

function getHealthScoreInfo(score) {
    if (score >= 80) {
        return {
            emoji: '🟢',
            label: 'Excellent',
            color: '#238636',
            description: 'Your codebase has excellent complexity distribution with most code in the A-B range.'
        };
    } else if (score >= 60) {
        return {
            emoji: '🟡',
            label: 'Good',
            color: '#d29922',
            description: 'Your codebase is in good shape, but could benefit from refactoring some complex functions.'
        };
    } else if (score >= 40) {
        return {
            emoji: '🟠',
            label: 'Needs Attention',
            color: '#f0883e',
            description: 'Your codebase has significant complexity issues that should be addressed soon.'
        };
    } else {
        return {
            emoji: '🔴',
            label: 'Critical',
            color: '#da3633',
            description: 'Your codebase has critical complexity issues requiring immediate refactoring.'
        };
    }
}

function getComplexityRange(grade) {
    const ranges = {
        'A': '1-5',
        'B': '6-10',
        'C': '11-15',
        'D': '16-20',
        'F': '21+'
    };
    return ranges[grade] || '';
}

function showComingSoon(reportName) {
    alert(`${reportName} - Coming Soon!\n\nThis feature will display detailed ${reportName.toLowerCase()} in a future release.`);
}

// ============================================================================
// VISUALIZATION MODE CONTROLS
// ============================================================================

function setVisualizationMode(mode) {
    if (mode === currentVizMode) return;

    currentVizMode = mode;
    currentZoomRootId = null;  // Reset zoom state

    // Update button states
    document.querySelectorAll('.viz-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Show/hide tree layout toggle (only for tree mode)
    const treeLayoutGroup = document.getElementById('tree-layout-group');
    if (treeLayoutGroup) {
        treeLayoutGroup.style.display = mode === 'tree' ? 'block' : 'none';
    }

    // Show/hide grouping mode toggle (only for treemap/sunburst)
    const groupingGroup = document.getElementById('grouping-mode-group');
    if (groupingGroup) {
        groupingGroup.style.display = (mode === 'treemap' || mode === 'sunburst') ? 'block' : 'none';
    }

    console.log(`Visualization mode changed to: ${mode}`);
    renderVisualization();
}

function toggleGroupingMode() {
    const toggle = document.getElementById('grouping-toggle');
    currentGroupingMode = toggle.checked ? 'ast' : 'file';

    // Update label highlighting
    const fileLabel = document.getElementById('grouping-label-file');
    const astLabel = document.getElementById('grouping-label-ast');
    if (fileLabel) fileLabel.classList.toggle('active', currentGroupingMode === 'file');
    if (astLabel) astLabel.classList.toggle('active', currentGroupingMode === 'ast');

    currentZoomRootId = null;  // Reset zoom when grouping changes
    console.log(`Grouping mode changed to: ${currentGroupingMode}`);
    renderVisualization();
}

// ============================================================================
// HIERARCHY TRANSFORMATION FUNCTIONS
// ============================================================================

function buildFileHierarchy() {
    // Return cached version if available
    if (cachedFileHierarchy) {
        console.log('Using cached file hierarchy');
        return cachedFileHierarchy;
    }

    console.time('buildFileHierarchy');

    // Build hierarchy from treeData (already structured by file/directory)
    // We need to ensure it's in D3-compatible format with value for sizing

    function processNode(node) {
        // Extract only needed properties (avoid full object cloning)
        const result = {
            name: node.name,
            id: node.id,
            type: node.type,
            file_path: node.file_path,
            complexity: node.complexity,
            lines_of_code: node.lines_of_code || (node.end_line && node.start_line ? node.end_line - node.start_line + 1 : 0),
            start_line: node.start_line,
            end_line: node.end_line,
            content: node.content,
            docstring: node.docstring,
            language: node.language
        };

        const children = node.children || node._children || [];
        if (children.length > 0) {
            result.children = children.map(child => processNode(child));
        }

        return result;
    }

    if (!treeData) {
        cachedFileHierarchy = { name: 'root', children: [] };
    } else {
        cachedFileHierarchy = processNode(treeData);
    }

    console.timeEnd('buildFileHierarchy');
    return cachedFileHierarchy;
}

function buildASTHierarchy() {
    // Return cached version if available
    if (cachedASTHierarchy) {
        console.log('Using cached AST hierarchy');
        return cachedASTHierarchy;
    }

    console.time('buildASTHierarchy');

    // Group all chunk nodes by: Language → Type (function/class/method) → Individual chunks
    // This gives a flatter view organized by code structure

    const byLanguage = new Map();

    // Language extension map (defined once outside loop)
    const langMap = {
        'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
        'tsx': 'TypeScript', 'jsx': 'JavaScript', 'java': 'Java',
        'go': 'Go', 'rs': 'Rust', 'rb': 'Ruby', 'php': 'PHP',
        'c': 'C', 'cpp': 'C++', 'cs': 'C#', 'swift': 'Swift'
    };

    // Use pre-cached chunk nodes (already filtered in initializeCaches)
    const chunkNodesToProcess = cachedChunkNodes || allNodes.filter(node => chunkTypes.includes(node.type));

    chunkNodesToProcess.forEach(node => {
        // Determine language from file extension
        let language = node.language || 'Unknown';
        if (!language || language === 'Unknown') {
            if (node.file_path) {
                const ext = node.file_path.split('.').pop().toLowerCase();
                language = langMap[ext] || ext.toUpperCase();
            }
        }

        if (!byLanguage.has(language)) {
            byLanguage.set(language, new Map());
        }

        const byType = byLanguage.get(language);
        const chunkType = node.type || 'code';

        if (!byType.has(chunkType)) {
            byType.set(chunkType, []);
        }

        byType.get(chunkType).push({
            name: node.name || node.id.substring(0, 20),
            id: node.id,
            type: node.type,
            file_path: node.file_path,
            complexity: node.complexity,
            lines_of_code: node.lines_of_code || (node.end_line && node.start_line ? node.end_line - node.start_line + 1 : 1),
            start_line: node.start_line,
            end_line: node.end_line,
            content: node.content,
            docstring: node.docstring,
            language: language
        });
    });

    // Convert to D3 hierarchy format
    const root = {
        name: 'Codebase',
        id: 'root',
        type: 'root',
        children: []
    };

    byLanguage.forEach((byType, language) => {
        const langNode = {
            name: language,
            id: `lang-${language}`,
            type: 'language',
            children: []
        };

        byType.forEach((chunks, chunkType) => {
            const typeNode = {
                name: capitalize(chunkType) + 's',
                id: `type-${language}-${chunkType}`,
                type: 'category',
                children: chunks
            };
            langNode.children.push(typeNode);
        });

        root.children.push(langNode);
    });

    cachedASTHierarchy = root;
    console.timeEnd('buildASTHierarchy');
    return cachedASTHierarchy;
}

// Build a Map of descendants for O(1) lookup by ID
function buildDescendantMap(root) {
    const map = new Map();
    root.descendants().forEach(d => {
        if (d.data.id) {
            map.set(d.data.id, d);
        }
    });
    return map;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Get complexity grade color for treemap/sunburst
function getComplexityColor(complexity) {
    if (complexity === undefined || complexity === null) return '#6e7681';  // gray for no data
    if (complexity <= 5) return '#238636';   // A - green
    if (complexity <= 10) return '#1f6feb';  // B - blue
    if (complexity <= 15) return '#d29922';  // C - yellow
    if (complexity <= 20) return '#f0883e';  // D - orange
    return '#da3633';                         // F - red
}

// Get color for a node based on its complexity or type
function getNodeColor(d) {
    // If it's a leaf node with complexity, use complexity color
    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        return getComplexityColor(d.data.complexity);
    }

    // Color by node type for non-leaf nodes
    const typeColors = {
        'directory': '#79c0ff',
        'file': '#58a6ff',
        'language': '#8957e5',
        'category': '#6e7681',
        'function': '#d29922',
        'method': '#8957e5',
        'class': '#1f6feb',
        'root': '#6e7681'
    };

    return typeColors[d.data.type] || '#6e7681';
}

// ============================================================================
// TREEMAP VISUALIZATION
// ============================================================================

function renderTreemap() {
    console.time('renderTreemap');
    const svg = d3.select('#graph');
    // Faster DOM clearing than selectAll('*').remove()
    svg.node().innerHTML = '';

    const { width, height } = getViewportDimensions();

    // Build hierarchy based on grouping mode
    const hierarchyData = currentGroupingMode === 'ast' ? buildASTHierarchy() : buildFileHierarchy();

    // Create D3 hierarchy
    const root = d3.hierarchy(hierarchyData)
        .sum(d => {
            // Use lines_of_code for sizing, minimum of 1 for visibility
            if (!d.children || d.children.length === 0) {
                return Math.max(d.lines_of_code || 1, 1);
            }
            return 0;
        })
        .sort((a, b) => b.value - a.value);

    vizHierarchy = root;

    // Build descendant map for O(1) lookup (replaces repeated linear search)
    const descendantMap = buildDescendantMap(root);

    // Find zoom root by ID if set (O(1) lookup instead of O(n))
    let displayRoot = root;
    if (currentZoomRootId) {
        const foundNode = descendantMap.get(currentZoomRootId);
        if (foundNode) {
            displayRoot = foundNode;
        } else {
            // Node not found, reset zoom
            currentZoomRootId = null;
        }
    }

    // Create treemap layout
    const treemap = d3.treemap()
        .size([width, height])
        .paddingTop(22)
        .paddingRight(3)
        .paddingBottom(3)
        .paddingLeft(3)
        .paddingInner(2)
        .round(true);

    treemap(displayRoot);

    // Create container group
    const g = svg.append('g')
        .attr('class', 'treemap-container');

    // Add breadcrumb if zoomed
    if (currentZoomRootId && displayRoot !== root) {
        renderTreemapBreadcrumb(svg, width, displayRoot, root);
    }

    // Create cells
    const cell = g.selectAll('g')
        .data(displayRoot.descendants())
        .join('g')
        .attr('transform', d => `translate(${d.x0},${d.y0})`);

    // Add rectangles
    cell.append('rect')
        .attr('class', 'treemap-cell')
        .attr('width', d => Math.max(0, d.x1 - d.x0))
        .attr('height', d => Math.max(0, d.y1 - d.y0))
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', 'rgba(0,0,0,0.3)')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('click', handleTreemapClick)
        .on('mouseover', handleTreemapHover)
        .on('mouseout', hideVizTooltip)
        .append('title')
        .text(d => `${d.data.name}\\n${d.value} lines`);

    // Add labels for cells large enough
    cell.filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 20)
        .append('text')
        .attr('class', 'treemap-label')
        .attr('x', 4)
        .attr('y', 14)
        .text(d => {
            const width = d.x1 - d.x0;
            const name = d.data.name;
            // Truncate if too long
            const maxChars = Math.floor(width / 7);
            return name.length > maxChars ? name.substring(0, maxChars - 1) + '…' : name;
        })
        .style('fill', '#fff')
        .style('font-size', '11px')
        .style('pointer-events', 'none');

    // Add value labels for larger cells
    cell.filter(d => (d.x1 - d.x0) > 60 && (d.y1 - d.y0) > 35 && !d.children)
        .append('text')
        .attr('class', 'treemap-value')
        .attr('x', 4)
        .attr('y', 26)
        .text(d => {
            const loc = d.value;
            const complexity = d.data.complexity;
            let text = `${loc} lines`;
            if (complexity !== undefined && complexity !== null) {
                text += ` (${getComplexityGrade(complexity)})`;
            }
            return text;
        })
        .style('fill', 'rgba(255,255,255,0.7)')
        .style('font-size', '9px')
        .style('pointer-events', 'none');

    // Create tooltip element if it doesn't exist
    ensureVizTooltip();

    console.timeEnd('renderTreemap');
    console.log(`Rendered ${displayRoot.descendants().length} treemap cells`);
}

function handleTreemapClick(event, d) {
    event.stopPropagation();

    // If it's a leaf node (code chunk), show the content
    if (!d.children || d.children.length === 0) {
        if (d.data.content || chunkTypes.includes(d.data.type)) {
            // Find the original node data
            // Use cached nodeIdMap for O(1) lookup instead of O(n) find
            const nodeData = (nodeIdMap && nodeIdMap.get(d.data.id)) || d.data;
            displayChunkContent(nodeData);
        }
        return;
    }

    // Store the path to this node for zoom (d.data.id or path based)
    // We need to store a way to find this node after re-building hierarchy
    currentZoomRootId = d.data.id;
    renderVisualization();
}

function handleTreemapHover(event, d) {
    const tooltip = document.getElementById('viz-tooltip');
    if (!tooltip) return;

    let html = `<div class="viz-tooltip-title">${escapeHtml(d.data.name)}</div>`;
    html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Type:</span> ${d.data.type}</div>`;
    html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Lines:</span> ${d.value}</div>`;

    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        const grade = getComplexityGrade(d.data.complexity);
        html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Complexity:</span> ${d.data.complexity.toFixed(1)} (${grade})</div>`;
    }

    if (d.data.file_path) {
        const shortPath = d.data.file_path.split('/').slice(-2).join('/');
        html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">File:</span> ${escapeHtml(shortPath)}</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.classList.add('visible');

    // Position tooltip
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width / 2) + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
}

function renderTreemapBreadcrumb(svg, width, currentNode, root) {
    // Build path from root to current node
    const path = [];
    let node = currentNode;
    while (node) {
        path.unshift(node);
        node = node.parent;
    }

    // Create breadcrumb container
    const breadcrumb = svg.append('g')
        .attr('class', 'viz-breadcrumb-container')
        .attr('transform', 'translate(10, 10)');

    // Background
    breadcrumb.append('rect')
        .attr('fill', 'rgba(13, 17, 23, 0.9)')
        .attr('stroke', 'var(--border-primary)')
        .attr('rx', 6)
        .attr('width', width - 20)
        .attr('height', 30);

    // Home icon
    breadcrumb.append('text')
        .attr('x', 12)
        .attr('y', 20)
        .attr('class', 'viz-breadcrumb-home')
        .text('🏠')
        .style('cursor', 'pointer')
        .on('click', () => {
            currentZoomRootId = null;
            renderVisualization();
        });

    let xPos = 35;
    path.forEach((n, i) => {
        // Separator
        if (i > 0) {
            breadcrumb.append('text')
                .attr('x', xPos)
                .attr('y', 20)
                .attr('class', 'viz-breadcrumb-separator')
                .text(' / ')
                .style('fill', 'var(--text-tertiary)');
            xPos += 20;
        }

        const isLast = i === path.length - 1;
        const text = breadcrumb.append('text')
            .attr('x', xPos)
            .attr('y', 20)
            .text(n.data.name)
            .style('fill', isLast ? 'var(--text-primary)' : 'var(--accent)')
            .style('font-size', '12px')
            .style('cursor', isLast ? 'default' : 'pointer');

        if (!isLast) {
            text.on('click', () => {
                currentZoomRootId = n.data.id;
                renderVisualization();
            });
        }

        xPos += n.data.name.length * 7 + 5;
    });
}

// ============================================================================
// SUNBURST VISUALIZATION
// ============================================================================

function renderSunburst() {
    console.time('renderSunburst');
    const svg = d3.select('#graph');
    // Faster DOM clearing than selectAll('*').remove()
    svg.node().innerHTML = '';

    const { width, height } = getViewportDimensions();
    const radius = Math.min(width, height) / 2 - 10;

    // Build hierarchy based on grouping mode
    const hierarchyData = currentGroupingMode === 'ast' ? buildASTHierarchy() : buildFileHierarchy();

    // Create D3 hierarchy
    const root = d3.hierarchy(hierarchyData)
        .sum(d => {
            if (!d.children || d.children.length === 0) {
                return Math.max(d.lines_of_code || 1, 1);
            }
            return 0;
        })
        .sort((a, b) => b.value - a.value);

    vizHierarchy = root;

    // Build descendant map for O(1) lookup (replaces repeated linear search)
    const descendantMap = buildDescendantMap(root);

    // Find zoom root by ID if set (O(1) lookup instead of O(n))
    let displayRoot = root;
    if (currentZoomRootId) {
        const foundNode = descendantMap.get(currentZoomRootId);
        if (foundNode) {
            displayRoot = foundNode;
        } else {
            currentZoomRootId = null;
        }
    }

    // Create partition layout
    const partition = d3.partition()
        .size([2 * Math.PI, radius]);

    partition(displayRoot);

    // Create arc generator
    const arc = d3.arc()
        .startAngle(d => d.x0)
        .endAngle(d => d.x1)
        .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
        .padRadius(radius / 2)
        .innerRadius(d => d.y0)
        .outerRadius(d => d.y1 - 1);

    // Create container group centered
    const g = svg.append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`);

    // Create arcs
    const arcs = g.selectAll('path')
        .data(displayRoot.descendants().filter(d => d.depth > 0))  // Exclude root
        .join('path')
        .attr('class', 'sunburst-arc')
        .attr('d', arc)
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', '#0d1117')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer')
        .on('click', handleSunburstClick)
        .on('mouseover', handleSunburstHover)
        .on('mouseout', hideVizTooltip);

    // Add title for tooltip
    arcs.append('title')
        .text(d => `${d.data.name}\\n${d.value} lines`);

    // Add labels for arcs large enough
    const labelArcs = displayRoot.descendants().filter(d => {
        const angle = d.x1 - d.x0;
        const radius = (d.y0 + d.y1) / 2;
        const arcLength = angle * radius;
        return d.depth > 0 && arcLength > 30 && (d.y1 - d.y0) > 15;
    });

    g.selectAll('text.sunburst-label')
        .data(labelArcs)
        .join('text')
        .attr('class', 'sunburst-label')
        .attr('transform', d => {
            const x = (d.x0 + d.x1) / 2 * 180 / Math.PI;
            const y = (d.y0 + d.y1) / 2;
            return `rotate(${x - 90}) translate(${y}, 0) rotate(${x < 180 ? 0 : 180})`;
        })
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .text(d => {
            const arcLength = (d.x1 - d.x0) * ((d.y0 + d.y1) / 2);
            const maxChars = Math.floor(arcLength / 7);
            const name = d.data.name;
            return name.length > maxChars ? name.substring(0, maxChars - 1) + '…' : name;
        })
        .style('fill', '#fff')
        .style('font-size', '10px')
        .style('pointer-events', 'none');

    // Add center label
    const centerText = displayRoot.data.name;
    const centerValue = displayRoot.value;

    g.append('text')
        .attr('class', 'sunburst-center-label')
        .attr('dy', '-0.3em')
        .text(centerText.length > 15 ? centerText.substring(0, 14) + '…' : centerText);

    g.append('text')
        .attr('class', 'sunburst-center-value')
        .attr('dy', '1em')
        .text(`${centerValue.toLocaleString()} lines`);

    // Add click handler on center to zoom out
    g.append('circle')
        .attr('r', displayRoot.y1 * 0.3)
        .attr('fill', 'transparent')
        .style('cursor', currentZoomRootId ? 'pointer' : 'default')
        .on('click', () => {
            if (currentZoomRootId && displayRoot.parent) {
                currentZoomRootId = displayRoot.parent === vizHierarchy ? null : displayRoot.parent.data.id;
                renderVisualization();
            }
        });

    // Create tooltip element if it doesn't exist
    ensureVizTooltip();

    console.timeEnd('renderSunburst');
    console.log(`Rendered ${displayRoot.descendants().length} sunburst arcs`);
}

function handleSunburstClick(event, d) {
    event.stopPropagation();

    // If it's a leaf node (code chunk), show the content
    if (!d.children || d.children.length === 0) {
        if (d.data.content || chunkTypes.includes(d.data.type)) {
            // Use cached nodeIdMap for O(1) lookup instead of O(n) find
            const nodeData = (nodeIdMap && nodeIdMap.get(d.data.id)) || d.data;
            displayChunkContent(nodeData);
        }
        return;
    }

    // Zoom into this node by ID
    currentZoomRootId = d.data.id;
    renderVisualization();
}

function handleSunburstHover(event, d) {
    const tooltip = document.getElementById('viz-tooltip');
    if (!tooltip) return;

    let html = `<div class="viz-tooltip-title">${escapeHtml(d.data.name)}</div>`;
    html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Type:</span> ${d.data.type}</div>`;
    html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Lines:</span> ${d.value}</div>`;

    if (d.data.complexity !== undefined && d.data.complexity !== null) {
        const grade = getComplexityGrade(d.data.complexity);
        html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">Complexity:</span> ${d.data.complexity.toFixed(1)} (${grade})</div>`;
    }

    if (d.data.file_path) {
        const shortPath = d.data.file_path.split('/').slice(-2).join('/');
        html += `<div class="viz-tooltip-row"><span class="viz-tooltip-label">File:</span> ${escapeHtml(shortPath)}</div>`;
    }

    tooltip.innerHTML = html;
    tooltip.classList.add('visible');

    // Position tooltip near mouse
    tooltip.style.left = (event.pageX + 10) + 'px';
    tooltip.style.top = (event.pageY - 10) + 'px';
}

// ============================================================================
// SHARED TOOLTIP FUNCTIONS
// ============================================================================

function ensureVizTooltip() {
    if (!document.getElementById('viz-tooltip')) {
        const tooltip = document.createElement('div');
        tooltip.id = 'viz-tooltip';
        tooltip.className = 'viz-tooltip';
        document.body.appendChild(tooltip);
    }
}

function hideVizTooltip() {
    const tooltip = document.getElementById('viz-tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Load data and initialize UI when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('=== PAGE INITIALIZATION ===');
    console.log('DOMContentLoaded event fired');

    // Load theme preference before anything else
    loadThemePreference();

    // Initialize toggle label highlighting for layout toggle
    const labels = document.querySelectorAll('#tree-layout-group .toggle-label');
    console.log(`Found ${labels.length} layout toggle labels`);
    if (labels[0]) {
        labels[0].classList.add('active');
        console.log('Activated first toggle label (linear mode)');
    }

    // Initialize grouping toggle label highlighting
    const groupingFileLabel = document.getElementById('grouping-label-file');
    if (groupingFileLabel) {
        groupingFileLabel.classList.add('active');
    }

    // Close search results when clicking outside
    document.addEventListener('click', (event) => {
        const searchContainer = document.querySelector('.search-container');
        if (searchContainer && !searchContainer.contains(event.target)) {
            closeSearchResults();
        }
    });

    // Load graph data
    console.log('Calling loadGraphData()...');
    loadGraphData();
    console.log('=== END PAGE INITIALIZATION ===');
});
"""
