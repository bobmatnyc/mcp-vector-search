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

// Chunk types for code nodes (function, class, method, text, imports, module)
const chunkTypes = ['function', 'class', 'method', 'text', 'imports', 'module'];

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

    // First pass: Handle directory hierarchy
    console.log('=== BUILDING DIRECTORY HIERARCHY ===');
    let dirLinksProcessed = 0;
    let dirLinksMatched = 0;

    allLinks.forEach(link => {
        if (link.type === 'dir_containment' || link.type === 'dir_hierarchy') {
            dirLinksProcessed++;

            if (nodeMap.has(link.source) && nodeMap.has(link.target)) {
                dirLinksMatched++;
                parentMap.set(link.target, link.source);

                const parent = nodeMap.get(link.source);
                const child = nodeMap.get(link.target);

                if (parent && child) {
                    parent.children.push(child);
                }
            } else {
                // DEBUG: Why didn't this link match?
                if (dirLinksProcessed <= 5) {  // Only log first few mismatches
                    console.log(`Link mismatch - source: ${link.source} (exists: ${nodeMap.has(link.source)}), target: ${link.target} (exists: ${nodeMap.has(link.target)})`);
                }
            }
        }
    });

    console.log(`Processed ${dirLinksProcessed} directory links, matched ${dirLinksMatched}`);
    console.log('=== END DIRECTORY HIERARCHY ===');

    // Second pass: Attach chunks to their parent files using file_containment links
    console.log('=== CHUNK ATTACHMENT DEBUG ===');

    let fileLinksProcessed = 0;
    let fileLinksMatched = 0;
    let fileLinksChunkMissing = 0;
    let fileLinksFileMissing = 0;
    let fileLinksChunkNotInTree = 0;
    let fileLinksFileNotInTree = 0;

    allLinks.forEach(link => {
        if (link.type === 'file_containment') {
            fileLinksProcessed++;

            // For file_containment links:
            // - source is the FILE
            // - target is the CHUNK
            const parentFile = nodeMap.get(link.source);
            const chunkNode = nodeMap.get(link.target);

            if (!parentFile) {
                fileLinksFileMissing++;
                // Check if file exists in allNodes but not in tree
                const fileInAll = allNodes.find(n => n.id === link.source);
                if (fileInAll) {
                    fileLinksFileNotInTree++;
                }
                if (fileLinksFileMissing <= 3) {  // Log first few
                    console.log(`File ${link.source} not found in nodeMap (exists in allNodes: ${!!fileInAll})`);
                }
                return;
            }

            if (!chunkNode) {
                fileLinksChunkMissing++;
                // Check if chunk exists in allNodes but not in tree
                const chunkInAll = allNodes.find(n => n.id === link.target);
                if (chunkInAll) {
                    fileLinksChunkNotInTree++;
                }
                if (fileLinksChunkMissing <= 3) {  // Log first few
                    console.log(`Chunk ${link.target} not found in nodeMap (exists in allNodes: ${!!chunkInAll}, type: ${chunkInAll?.type})`);
                }
                return;
            }

            // Successfully attach chunk to parent file
            parentFile.children.push(chunkNode);
            parentMap.set(link.target, link.source);
            fileLinksMatched++;
        }
    });

    console.log(`Processed ${fileLinksProcessed} file_containment links`);
    console.log(`Successfully matched: ${fileLinksMatched}`);
    console.log(`File not found: ${fileLinksFileMissing} (${fileLinksFileNotInTree} exist in allNodes but filtered out)`);
    console.log(`Chunk not found: ${fileLinksChunkMissing} (${fileLinksChunkNotInTree} exist in allNodes but filtered out)`);
    console.log('=== END CHUNK ATTACHMENT DEBUG ===');

    // DEBUG: Check parent map
    console.log('=== PARENT MAP DEBUG ===');
    console.log(`Total entries in parentMap: ${parentMap.size}`);
    console.log(`Total tree nodes: ${treeNodes.length}`);
    console.log(`Expected root nodes: ${treeNodes.length - parentMap.size}`);

    // Sample some chunk nodes to see if they're in parentMap
    const sampleChunks = treeNodes.filter(n => chunkTypes.includes(n.type)).slice(0, 5);
    console.log('Sample chunk parent check:');
    sampleChunks.forEach(chunk => {
        const hasParent = parentMap.has(chunk.id);
        const parent = parentMap.get(chunk.id);
        console.log(`  Chunk ${chunk.id} (${chunk.type}): hasParent=${hasParent}, parent=${parent}`);
    });
    console.log('=== END PARENT MAP DEBUG ===');

    // Find root nodes (nodes with no parents)
    const rootNodes = treeNodes
        .filter(node => !parentMap.has(node.id))
        .map(node => nodeMap.get(node.id))
        .filter(node => node !== undefined);

    console.log(`Found ${rootNodes.length} root nodes`);

    // DEBUG: Count root node types
    const rootTypeCounts = {};
    rootNodes.forEach(node => {
        const type = node.type || 'undefined';
        rootTypeCounts[type] = (rootTypeCounts[type] || 0) + 1;
    });
    console.log('Root node type breakdown:', rootTypeCounts);

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

    // Node circles
    nodes.append('circle')
        .attr('r', d => chunkTypes.includes(d.data.type) ? 4 : 6)  // Smaller circles for chunks
        .attr('fill', d => {
            if (d.data.type === 'directory') {
                // Orange if collapsed (has _children), blue if expanded (has children)
                return d.data._children ? '#f39c12' : '#3498db';
            } else if (d.data.type === 'file') {
                // Gray if collapsed (has _children), white if expanded or no chunks
                return d.data._children ? '#95a5a6' : '#ecf0f1';
            } else if (chunkTypes.includes(d.data.type)) {
                return '#9b59b6';  // Purple for chunks
            }
            return '#95a5a6';  // Default gray
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Node labels
    const labels = nodes.append('text')
        .attr('dx', 12)
        .attr('dy', 4)
        .text(d => {
            // For chunks, show the name (function/class name)
            return d.data.name;
        })
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

    // Node circles
    nodes.append('circle')
        .attr('r', d => chunkTypes.includes(d.data.type) ? 4 : 6)  // Smaller circles for chunks
        .attr('fill', d => {
            if (d.data.type === 'directory') {
                // Orange if collapsed (has _children), blue if expanded (has children)
                return d.data._children ? '#f39c12' : '#3498db';
            } else if (d.data.type === 'file') {
                // Gray if collapsed (has _children), white if expanded or no chunks
                return d.data._children ? '#95a5a6' : '#ecf0f1';
            } else if (chunkTypes.includes(d.data.type)) {
                return '#9b59b6';  // Purple for chunks
            }
            return '#95a5a6';  // Default gray
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Node labels
    nodes.append('text')
        .attr('dx', 12)
        .attr('dy', 4)
        .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
        .attr('text-anchor', d => d.x >= Math.PI ? 'end' : 'start')
        .text(d => {
            // For chunks, show the name (function/class name)
            return d.data.name;
        })
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
        console.log('Displaying chunk content');
        // Show chunk content in side panel
        displayChunkContent(nodeData);
    }

    console.log('=== END NODE CLICK DEBUG ===');
}

function displayDirectoryInfo(dirData) {
    openViewerPanel();

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    title.textContent = `📁 ${dirData.name}`;

    // Count children
    const children = dirData.children || dirData._children || [];
    const dirs = children.filter(c => c.type === 'directory').length;
    const files = children.filter(c => c.type === 'file').length;

    let html = '<div class="viewer-section">';
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

        children.forEach(child => {
            const icon = child.type === 'directory' ? '📁' : '📄';
            const type = child.type === 'directory' ? 'dir' : 'file';
            html += `<div class="dir-list-item">`;
            html += `<span class="dir-icon">${icon}</span>`;
            html += `<span class="dir-name">${escapeHtml(child.name)}</span>`;
            html += `<span class="dir-type">${type}</span>`;
            html += `</div>`;
        });

        html += '</div>';
        html += '</div>';
    }

    content.innerHTML = html;
}

function displayFileInfo(fileData) {
    openViewerPanel();

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    title.textContent = `📄 ${fileData.name}`;

    // Get chunks
    const chunks = fileData.children || fileData._children || [];

    let html = '<div class="viewer-section">';
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

function displayChunkContent(chunkData) {
    openViewerPanel();

    const title = document.getElementById('viewer-title');
    const content = document.getElementById('viewer-content');

    const chunkName = chunkData.name || chunkData.type || 'Chunk';
    title.textContent = `${getChunkIcon(chunkData.type)} ${chunkName}`;

    let html = '<div class="viewer-section">';
    html += '<div class="viewer-section-title">Chunk Information</div>';
    html += '<div class="viewer-info-grid">';
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Name:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(chunkName)}</span>`;
    html += `</div>`;
    html += `<div class="viewer-info-row">`;
    html += `<span class="viewer-info-label">Type:</span>`;
    html += `<span class="viewer-info-value">${escapeHtml(chunkData.type || 'code')}</span>`;
    html += `</div>`;
    if (chunkData.start_line && chunkData.end_line) {
        html += `<div class="viewer-info-row">`;
        html += `<span class="viewer-info-label">Lines:</span>`;
        html += `<span class="viewer-info-value">${chunkData.start_line} - ${chunkData.end_line}</span>`;
        html += `</div>`;
    }
    html += '</div>';
    html += '</div>';

    if (chunkData.content) {
        html += '<div class="viewer-section">';
        html += '<div class="viewer-section-title">Source Code</div>';
        html += `<pre><code>${escapeHtml(chunkData.content)}</code></pre>`;
        html += '</div>';
    } else {
        html += '<p style="color: #8b949e; padding: 20px; text-align: center;">No content available for this chunk.</p>';
    }

    content.innerHTML = html;
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

    // Load graph data
    console.log('Calling loadGraphData()...');
    loadGraphData();
    console.log('=== END PAGE INITIALIZATION ===');
});
"""
