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

// Navigation history for back/forward
let navigationHistory = [];
let navigationIndex = -1;

// Chunk types for code nodes (function, class, method, text, imports, module)
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];

// Size scaling configuration
const sizeConfig = {
    minRadius: 8,       // Minimum node radius (increased for readability)
    maxRadius: 20,      // Maximum node radius
    chunkMinRadius: 6,  // Minimum for chunks (increased for readability)
    chunkMaxRadius: 12  // Maximum for chunks
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

async function loadGraphData() {
    try {
        const response = await fetch('/api/graph');
        const data = await response.json();
        allNodes = data.nodes || [];
        allLinks = data.links || [];

        console.log(`Loaded ${allNodes.length} nodes and ${allLinks.length} links`);

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
        chunk_containment: 0  // undefined links = chunk-to-chunk (class -> method)
    };

    let relationshipsMatched = {
        dir_hierarchy: 0,
        dir_containment: 0,
        file_containment: 0,
        chunk_containment: 0
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
        } else if (linkType === undefined || linkType === 'undefined') {
            // Undefined links are chunk-to-chunk (e.g., class -> method)
            category = 'chunk_containment';
        } else {
            // Skip semantic, caller, and other non-hierarchical links
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
    console.log(`  chunk_containment: ${relationshipsMatched.chunk_containment}/${relationshipsProcessed.chunk_containment} matched`);
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
                    // Store info about the collapsed chunk
                    node.collapsed_chunk = {
                        type: onlyChild.type,
                        name: onlyChild.name,
                        id: onlyChild.id
                    };
                } else {
                    // Chunk has no children - just keep as is (will show chunk content on click)
                    console.log(`File ${node.name} has single chunk ${onlyChild.type} with no children - keeping as is`);
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

    // Collapse all child nodes of the root (but keep root's direct children visible initially)
    // This way, only the root level (first level) is visible, all deeper levels are collapsed
    if (treeData.children) {
        treeData.children.forEach(child => {
            // Collapse all descendants of each root child, but keep the root children themselves visible
            if (child.children && child.children.length > 0) {
                child.children.forEach(grandchild => collapseAll(grandchild));
                // Move children to _children to collapse
                child._children = child.children;
                child.children = null;
            }
        });
    }

    console.log('Tree structure built with all directories and files collapsed');

    // Calculate sizes for all nodes (for proportional node rendering)
    calculateNodeSizes(treeData);
    console.log('Node sizes calculated');

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

// Global variables for size scaling
let globalMinSize = Infinity;
let globalMaxSize = 0;

function calculateNodeSizes(node) {
    if (!node) return 0;

    // For chunks: use content length or line count
    if (chunkTypes.includes(node.type)) {
        const contentLen = (node.content || '').length;
        const lineCount = (node.start_line && node.end_line)
            ? node.end_line - node.start_line + 1
            : 0;
        // Prefer content length, fall back to line count * avg chars per line
        node._size = contentLen || (lineCount * 40);
        if (node._size > 0) {
            globalMinSize = Math.min(globalMinSize, node._size);
            globalMaxSize = Math.max(globalMaxSize, node._size);
        }
        return node._size;
    }

    // For files and directories: sum of children sizes
    const children = node.children || node._children || [];
    let totalSize = 0;

    children.forEach(child => {
        totalSize += calculateNodeSizes(child);
    });

    node._size = totalSize || 1;  // Minimum 1 for empty dirs/files

    if (node._size > 0) {
        globalMinSize = Math.min(globalMinSize, node._size);
        globalMaxSize = Math.max(globalMaxSize, node._size);
    }

    return node._size;
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

    // Build a map of node positions from the tree
    const nodePositions = new Map();
    root.descendants().forEach(d => {
        nodePositions.set(d.data.id, { x: d.x, y: d.y, node: d });
    });

    // Create a group for external call lines (behind nodes)
    let lineGroup = svg.select('.external-lines-group');
    if (lineGroup.empty()) {
        lineGroup = svg.insert('g', ':first-child')
            .attr('class', 'external-lines-group');
    }

    externalCallData.forEach(data => {
        const sourcePos = nodePositions.get(data.nodeId);
        if (!sourcePos) return;

        // Draw lines to inbound nodes (callers) - dashed blue
        data.inboundNodes.forEach(caller => {
            const targetPos = nodePositions.get(caller.id);
            if (targetPos) {
                lineGroup.append('path')
                    .attr('class', 'external-call-line inbound-line')
                    .attr('d', `M${targetPos.y},${targetPos.x} C${(targetPos.y + sourcePos.y)/2},${targetPos.x} ${(targetPos.y + sourcePos.y)/2},${sourcePos.x} ${sourcePos.y},${sourcePos.x}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#58a6ff')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.4)
                    .attr('pointer-events', 'none');
            }
        });

        // Draw lines to outbound nodes (callees) - dashed orange
        data.outboundNodes.forEach(callee => {
            const targetPos = nodePositions.get(callee.id);
            if (targetPos) {
                lineGroup.append('path')
                    .attr('class', 'external-call-line outbound-line')
                    .attr('d', `M${sourcePos.y},${sourcePos.x} C${(sourcePos.y + targetPos.y)/2},${sourcePos.x} ${(sourcePos.y + targetPos.y)/2},${targetPos.x} ${targetPos.y},${targetPos.x}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#f0883e')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4,2')
                    .attr('opacity', 0.4)
                    .attr('pointer-events', 'none');
            }
        });
    });
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

function getNodeRadius(d) {
    const nodeData = d.data;
    const nodeSize = nodeData._size || 1;

    // Determine min/max based on node type
    let minR, maxR;
    if (chunkTypes.includes(nodeData.type)) {
        minR = sizeConfig.chunkMinRadius;
        maxR = sizeConfig.chunkMaxRadius;
    } else {
        minR = sizeConfig.minRadius;
        maxR = sizeConfig.maxRadius;
    }

    // Logarithmic scaling for better distribution
    // Using log scale because file sizes can vary by orders of magnitude
    if (globalMaxSize <= globalMinSize) {
        return (minR + maxR) / 2;  // Default if no range
    }

    const logMin = Math.log(globalMinSize + 1);
    const logMax = Math.log(globalMaxSize + 1);
    const logSize = Math.log(nodeSize + 1);

    // Normalize to 0-1 range
    const normalized = (logSize - logMin) / (logMax - logMin);

    // Scale to radius range
    return minR + (normalized * (maxR - minR));
}

// ============================================================================
// VISUALIZATION RENDERING
// ============================================================================

function renderVisualization() {
    console.log('=== RENDER VISUALIZATION ===');
    console.log(`Current layout: ${currentLayout}`);
    console.log(`Tree data exists: ${treeData !== null}`);
    if (treeData) {
        console.log(`Root node: ${treeData.name}, children: ${(treeData.children || []).length}, _children: ${(treeData._children || []).length}`);
    }

    // Clear existing content
    const graphElement = d3.select('#graph');
    console.log(`Graph element found: ${!graphElement.empty()}`);
    graphElement.selectAll('*').remove();

    if (currentLayout === 'linear') {
        console.log('Calling renderLinearTree()...');
        renderLinearTree();
    } else {
        console.log('Calling renderCircularTree()...');
        renderCircularTree();
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
    const root = d3.hierarchy(treeData, d => d.children);
    console.log(`Hierarchy created: ${root.descendants().length} nodes`);

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
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

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
        .style('font-size', d => chunkTypes.includes(d.data.type) ? '10px' : '12px')
        .style('font-family', 'Arial, sans-serif')
        .style('fill', d => chunkTypes.includes(d.data.type) ? '#bb86fc' : '#adbac7');

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
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

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
        .style('font-size', d => chunkTypes.includes(d.data.type) ? '10px' : '12px')
        .style('font-family', 'Arial, sans-serif')
        .style('fill', d => chunkTypes.includes(d.data.type) ? '#bb86fc' : '#adbac7');
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

        // Also show directory info in viewer
        displayDirectoryInfo(nodeData);
    } else if (nodeData.type === 'file') {
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

        // Show file info in viewer
        displayFileInfo(nodeData);
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
}

function displayChunkContent(chunkData, addToHistory = true) {
    openViewerPanel();

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

    // === ORDER: Docstring (comments), Code, Metadata ===

    // === 1. Docstring Section (Comments) ===
    if (chunkData.docstring) {
        html += '<div class="viewer-section">';
        html += '<div class="viewer-section-title">📖 Docstring</div>';
        html += `<div style="color: #8b949e; font-style: italic; padding: 8px 12px; background: #161b22; border-radius: 4px; white-space: pre-wrap;">${escapeHtml(chunkData.docstring)}</div>`;
        html += '</div>';
    }

    // === 2. Source Code Section ===
    if (chunkData.content) {
        html += '<div class="viewer-section">';
        html += '<div class="viewer-section-title">📝 Source Code</div>';
        html += `<pre><code>${escapeHtml(chunkData.content)}</code></pre>`;
        html += '</div>';
    } else {
        html += '<p style="color: #8b949e; padding: 20px; text-align: center;">No content available for this chunk.</p>';
    }

    // === 3. Metadata Section ===
    html += '<div class="viewer-section">';
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

    // File path
    if (chunkData.file_path) {
        const shortPath = chunkData.file_path.split('/').slice(-3).join('/');
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">File:</span>`;
        html += `<span class="viewer-info-value" title="${escapeHtml(chunkData.file_path)}">.../${escapeHtml(shortPath)}</span>`;
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
            html += '<div class="viewer-section">';
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
            html += '<div class="viewer-section">';
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
            html += '<div class="viewer-section">';
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
            html += '<div class="viewer-section">';
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

// Highlight and scroll to a node in the rendered tree
function highlightNodeInTree(nodeId) {
    // Remove any existing highlight
    d3.selectAll('.node-highlight').classed('node-highlight', false);

    // Find and highlight the target node in the rendered SVG
    const svg = d3.select('#graph');
    const targetNode = svg.selectAll('.node')
        .filter(d => d.data.id === nodeId);

    if (!targetNode.empty()) {
        // Add highlight class
        targetNode.classed('node-highlight', true);

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

// ============================================================================
// LAYOUT TOGGLE
// ============================================================================

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
// VIEWER PANEL CONTROLS
// ============================================================================

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
    container.classList.remove('viewer-open');
    isViewerOpen = false;

    // Re-render visualization to adjust to new viewport size
    setTimeout(() => {
        renderVisualization();
    }, 300); // Wait for transition
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
// INITIALIZATION
// ============================================================================

// Load data and initialize UI when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('=== PAGE INITIALIZATION ===');
    console.log('DOMContentLoaded event fired');

    // Initialize toggle label highlighting
    const labels = document.querySelectorAll('.toggle-label');
    console.log(`Found ${labels.length} toggle labels`);
    if (labels[0]) {
        labels[0].classList.add('active');
        console.log('Activated first toggle label (linear mode)');
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
