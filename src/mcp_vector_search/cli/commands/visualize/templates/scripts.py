"""JavaScript code for the D3.js visualization.

This module contains all JavaScript functionality for the interactive code graph,
organized into logical sections for maintainability.
"""


def get_d3_initialization() -> str:
    """Get D3.js initialization and global variables.

    Returns:
        JavaScript string for D3.js setup
    """
    return """
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Create zoom behavior - allow more zoom out for larger nodes
        const zoom = d3.zoom()
            .scaleExtent([0.15, 4]) // Increased range from [0.1, 3] to allow better zoom out with larger nodes
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });

        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height)
            .call(zoom);

        const g = svg.append("g");
        const tooltip = d3.select("#tooltip");
        let simulation;
        let allNodes = [];
        let allLinks = [];
        let visibleNodes = new Set();
        let collapsedNodes = new Set();
        let highlightedNode = null;
        let rootNodes = [];  // Store root nodes for reset function
    """


def get_file_type_functions() -> str:
    """Get file type detection and icon functions.

    Returns:
        JavaScript string for file type handling
    """
    return """
        // Get file extension from path
        function getFileExtension(filePath) {
            if (!filePath) return '';
            const match = filePath.match(/\\.([^.]+)$/);
            return match ? match[1].toLowerCase() : '';
        }

        // Get SVG icon path for file type
        function getFileTypeIcon(node) {
            if (node.type === 'directory') {
                // Folder icon
                return 'M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z';
            }
            if (node.type === 'file') {
                const ext = getFileExtension(node.file_path);

                // Python files
                if (ext === 'py') {
                    return 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z';
                }
                // JavaScript/TypeScript
                if (ext === 'js' || ext === 'jsx' || ext === 'ts' || ext === 'tsx') {
                    return 'M3 3h18v18H3V3zm16 16V5H5v14h14zM7 7h2v2H7V7zm4 0h2v2h-2V7zm-4 4h2v2H7v-2zm4 0h6v2h-6v-2zm-4 4h10v2H7v-2z';
                }
                // Markdown
                if (ext === 'md' || ext === 'markdown') {
                    return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6zm10-10h-3v2h3v2h-3v2h3v2h-7V8h7v2z';
                }
                // JSON/YAML/Config files
                if (ext === 'json' || ext === 'yaml' || ext === 'yml' || ext === 'toml' || ext === 'ini' || ext === 'conf') {
                    return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm0 2l4 4h-4V4zM6 20V4h6v6h6v10H6zm4-4h4v2h-4v-2zm0-4h4v2h-4v-2z';
                }
                // Shell scripts
                if (ext === 'sh' || ext === 'bash' || ext === 'zsh') {
                    return 'M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V8h16v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2zM6 10h8v2H6v-2z';
                }
                // Generic code file
                return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm0 2l4 4h-4V4zM6 20V4h6v6h6v10H6zm3-4h6v2H9v-2zm0-4h6v2H9v-2z';
            }
            return null;
        }

        // Get color for file type icon
        function getFileTypeColor(node) {
            if (node.type === 'directory') return '#79c0ff';
            if (node.type === 'file') {
                const ext = getFileExtension(node.file_path);
                if (ext === 'py') return '#3776ab';  // Python blue
                if (ext === 'js' || ext === 'jsx') return '#f7df1e';  // JavaScript yellow
                if (ext === 'ts' || ext === 'tsx') return '#3178c6';  // TypeScript blue
                if (ext === 'md' || ext === 'markdown') return '#8b949e';  // Gray
                if (ext === 'json' || ext === 'yaml' || ext === 'yml') return '#90a4ae';  // Config gray
                if (ext === 'sh' || ext === 'bash' || ext === 'zsh') return '#4eaa25';  // Shell green
                return '#58a6ff';  // Default file color
            }
            return null;
        }
    """


def get_spacing_calculation_functions() -> str:
    """Get automatic spacing calculation functions.

    Returns:
        JavaScript string for spacing calculations

    Design Decision: Adaptive spacing based on graph density

    Rationale: Hardcoded 800px spacing doesn't adapt to viewport size or node count.
    Selected density-based formula (nodes/area) for automatic scaling across devices.

    Trade-offs:
    - Adaptability: Auto-scales for mobile to 4K displays vs. fixed spacing
    - Complexity: Requires calculation but prevents manual tuning per graph size
    - Performance: Minimal overhead (O(1) calculation) vs. simplicity of hardcoded value

    Alternatives Considered:
    1. Fixed spacing with manual overrides: Rejected - requires user intervention
    2. Viewport-only scaling: Rejected - doesn't account for node count
    3. Node-count-only scaling: Rejected - breaks on different screen sizes

    Extension Points: Mode parameter ('tight', 'balanced', 'loose') allows future
    customization. Bounds can be adjusted per graph size category if needed.

    Performance:
    - Time Complexity: O(1) - simple arithmetic operations
    - Space Complexity: O(1) - no data structures allocated
    - Expected Performance: <1ms per calculation on modern browsers

    Error Handling:
    - Zero nodes: Returns default 100px spacing
    - Invalid mode: Falls back to 'balanced' mode
    - Extreme viewports: Clamped by min/max bounds per size category
    """
    return """
        // Calculate adaptive spacing based on graph density
        function calculateAdaptiveSpacing(nodeCount, width, height, mode = 'balanced') {
            if (nodeCount === 0) return 100; // Guard clause

            const areaPerNode = (width * height) / nodeCount;
            const baseSpacing = Math.sqrt(areaPerNode);

            // Scale factors for different modes
            const modeScales = {
                'tight': 0.4,      // Dense packing
                'balanced': 0.6,   // Good default
                'loose': 0.8       // More breathing room
            };

            const scaleFactor = modeScales[mode] || 0.6;
            const calculatedSpacing = baseSpacing * scaleFactor;

            // Bounds based on graph size
            let minBound, maxBound;
            if (nodeCount < 50) {
                minBound = 150; maxBound = 400;
            } else if (nodeCount < 500) {
                minBound = 100; maxBound = 250;
            } else {
                minBound = 60; maxBound = 150;
            }

            return Math.max(minBound, Math.min(maxBound, calculatedSpacing));
        }

        // Calculate coordinated force parameters
        function calculateForceParameters(nodeCount, width, height, spacing) {
            const k = Math.sqrt(nodeCount / (width * height));

            return {
                linkDistance: Math.max(30, spacing * 0.25),
                chargeStrength: -10 / k,
                collideRadius: 30,
                centerStrength: 0.05 + (0.1 * k),
                radialStrength: 0.05 + (0.15 * k)
            };
        }
    """


def get_loading_spinner_functions() -> str:
    """Get loading spinner functions for async node operations.

    Returns:
        JavaScript string for loading spinner display

    Usage Examples:
        // Show spinner during async operation
        showNodeLoading(nodeId);
        try {
            await fetchNodeData(nodeId);
        } finally {
            hideNodeLoading(nodeId);
        }

    Common Use Cases:
    - Lazy-loading node data when expanding collapsed groups
    - Fetching additional details from backend
    - Loading file contents on demand
    - Any async operation tied to a specific node

    Error Case Handling:
    - Missing nodeId: Silently returns (no error thrown)
    - Invalid nodeId: Silently returns if node not found in DOM
    - Multiple calls: Safe to call showNodeLoading multiple times (removes old spinner)

    Performance:
    - Time Complexity: O(n) where n = number of visible nodes (D3 selection filter)
    - Space Complexity: O(1) - adds 2 SVG elements per node
    - Animation: CSS-based, hardware-accelerated transform
    """
    return """
        // Show loading spinner on a node
        function showNodeLoading(nodeId) {
            const node = svg.selectAll('.node')
                .filter(d => d.id === nodeId);

            if (node.empty()) return;

            const nodeData = node.datum();
            const x = nodeData.x || 0;
            const y = nodeData.y || 0;
            const radius = 30;

            // Remove any existing spinner first
            node.selectAll('.node-loading, .node-loading-overlay').remove();

            // Add semi-transparent overlay
            node.append('circle')
                .attr('class', 'node-loading-overlay')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', radius);

            // Add spinning circle
            node.append('circle')
                .attr('class', 'node-loading')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', radius * 0.7)
                .style('transform-origin', `${x}px ${y}px`);
        }

        // Hide loading spinner from a node
        function hideNodeLoading(nodeId) {
            const node = svg.selectAll('.node')
                .filter(d => d.id === nodeId);

            node.selectAll('.node-loading, .node-loading-overlay').remove();
        }
    """


def get_graph_visualization_functions() -> str:
    """Get main graph visualization functions.

    Returns:
        JavaScript string for graph rendering
    """
    return """
        // Helper function to calculate complexity-based color shading
        function getComplexityShade(baseColor, complexity) {
            if (!complexity || complexity === 0) return baseColor;

            // Convert hex to HSL for proper darkening
            const rgb = d3.rgb(baseColor);
            const hsl = d3.hsl(rgb);

            // Reduce lightness based on complexity (darker = more complex)
            // Complexity scale: 0-5 (low), 6-10 (medium), 11+ (high)
            // Max reduction: 40% for very complex functions
            const lightnessReduction = Math.min(complexity * 0.03, 0.4);
            hsl.l = Math.max(hsl.l - lightnessReduction, 0.1); // Don't go too dark

            return hsl.toString();
        }

        // Position ALL nodes in an adaptive initial layout
        function positionNodesCompactly(nodes) {
            const folders = nodes.filter(n => n.type === 'directory');
            const outliers = nodes.filter(n => n.type !== 'directory');

            // Calculate adaptive spacing for folders (grid layout)
            if (folders.length > 0) {
                const folderSpacing = calculateAdaptiveSpacing(folders.length, width, height, 'balanced');
                const cols = Math.ceil(Math.sqrt(folders.length));
                const startX = width / 2 - (cols * folderSpacing) / 2;
                const startY = height / 2 - (Math.ceil(folders.length / cols) * folderSpacing) / 2;

                folders.forEach((folder, i) => {
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    folder.x = startX + col * folderSpacing;
                    folder.y = startY + row * folderSpacing;
                    folder.fx = folder.x; // Fix position initially
                    folder.fy = folder.y;
                });
            }

            // Calculate adaptive radius for outliers (spiral layout)
            if (outliers.length > 0) {
                const clusterRadius = calculateAdaptiveSpacing(outliers.length, width * 0.6, height * 0.6, 'tight') * 2;
                outliers.forEach((node, i) => {
                    const angle = (i / outliers.length) * 2 * Math.PI;
                    const radius = clusterRadius * Math.sqrt(i / outliers.length);
                    node.x = width / 2 + radius * Math.cos(angle);
                    node.y = height / 2 + radius * Math.sin(angle);
                });
            }

            // Release fixed folder positions after settling
            setTimeout(() => {
                folders.forEach(folder => {
                    folder.fx = null;
                    folder.fy = null;
                });
            }, 1000);
        }

        function visualizeGraph(data) {
            g.selectAll("*").remove();

            allNodes = data.nodes;
            allLinks = data.links;

            // Find root nodes - start with only top-level nodes
            if (data.metadata && data.metadata.is_monorepo) {
                // In monorepos, subproject nodes are roots
                rootNodes = allNodes.filter(n => n.type === 'subproject');
            } else {
                // Regular projects: show root-level directories AND files
                const dirNodes = allNodes.filter(n => n.type === 'directory');
                const fileNodes = allNodes.filter(n => n.type === 'file');

                // Find minimum depth for directories and files
                const minDirDepth = dirNodes.length > 0
                    ? Math.min(...dirNodes.map(n => n.depth))
                    : Infinity;
                const minFileDepth = fileNodes.length > 0
                    ? Math.min(...fileNodes.map(n => n.depth))
                    : Infinity;

                // Include both root-level directories and root-level files
                rootNodes = [
                    ...dirNodes.filter(n => n.depth === minDirDepth),
                    ...fileNodes.filter(n => n.depth === minFileDepth)
                ];

                // Fallback to all files if nothing found
                if (rootNodes.length === 0) {
                    rootNodes = fileNodes;
                }
            }

            // Start with only root nodes visible, all collapsed
            visibleNodes = new Set(rootNodes.map(n => n.id));
            collapsedNodes = new Set(rootNodes.map(n => n.id));
            highlightedNode = null;

            // Initial render
            renderGraph();

            // Position folders compactly (same as reset view)
            const currentNodes = allNodes.filter(n => visibleNodes.has(n.id));
            positionNodesCompactly(currentNodes);

            // Zoom to fit (same as reset view)
            setTimeout(() => {
                zoomToFit(750);
            }, 300); // Slightly longer delay for positioning
        }

        function renderGraph() {
            const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
            const visibleLinks = allLinks.filter(l =>
                visibleNodes.has(l.source.id || l.source) &&
                visibleNodes.has(l.target.id || l.target)
            );

            simulation = d3.forceSimulation(visibleNodesList)
                .force("link", d3.forceLink(visibleLinks)
                    .id(d => d.id)
                    .distance(d => {
                        // MUCH shorter distances for compact packing
                        if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') {
                            return 40; // Drastically reduced from 60
                        }
                        if (d.is_cycle) return 80; // Reduced from 120
                        if (d.type === 'semantic') return 100; // Reduced from 150
                        return 60; // Reduced from 90
                    })
                    .strength(d => {
                        // STRONGER links to pull nodes much closer
                        if (d.type === 'dir_containment' || d.type === 'dir_hierarchy') {
                            return 0.8; // Increased from 0.6
                        }
                        if (d.is_cycle) return 0.4; // Increased from 0.3
                        if (d.type === 'semantic') return 0.3; // Increased from 0.2
                        return 0.7; // Increased from 0.5
                    })
                )
                .force("charge", d3.forceManyBody()
                    .strength(d => {
                        // ULTRA-LOW repulsion for maximum clustering
                        if (d.type === 'directory') {
                            return -30; // FURTHER REDUCED: -50 → -30 (40% less)
                        }
                        return -60; // FURTHER REDUCED: -100 → -60 (40% less)
                    })
                )
                .force("center", d3.forceCenter(width / 2, height / 2).strength(0.1)) // Explicit centering strength
                .force("radial", d3.forceRadial(100, width / 2, height / 2)
                    .strength(d => {
                        // Pull non-folder nodes toward center
                        if (d.type === 'directory') {
                            return 0; // Don't affect folders
                        }
                        return 0.1; // Gentle pull toward center for other nodes
                    })
                )
                .force("collision", d3.forceCollide()
                    .radius(d => {
                        // Collision radius to prevent overlap
                        if (d.type === 'directory') return 30;
                        if (d.type === 'file') return 26;
                        return 24;
                    })
                    .strength(1.0) // Maximum collision strength to prevent overlap
                )
                .velocityDecay(0.6)
                .alphaDecay(0.02)

            g.selectAll("*").remove();

            const link = g.append("g")
                .selectAll("line")
                .data(visibleLinks)
                .join("line")
                .attr("class", d => {
                    // Cycle links have highest priority
                    if (d.is_cycle) return "link cycle";
                    if (d.type === "dependency") return "link dependency";
                    if (d.type === "semantic") {
                        // Color based on similarity score
                        const sim = d.similarity || 0;
                        let simClass = "sim-very-low";
                        if (sim >= 0.8) simClass = "sim-high";
                        else if (sim >= 0.6) simClass = "sim-medium-high";
                        else if (sim >= 0.4) simClass = "sim-medium";
                        else if (sim >= 0.2) simClass = "sim-low";
                        return `link semantic ${simClass}`;
                    }
                    return "link";
                })
                .on("mouseover", showLinkTooltip)
                .on("mouseout", hideTooltip);

            const node = g.append("g")
                .selectAll("g")
                .data(visibleNodesList)
                .join("g")
                .attr("class", d => {
                    let classes = `node ${d.type}`;
                    if (highlightedNode && d.id === highlightedNode.id) {
                        classes += ' highlighted';
                    }
                    return classes;
                })
                .call(drag(simulation))
                .on("click", handleNodeClick)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            // Add shapes based on node type
            const isDocNode = d => ['docstring', 'comment'].includes(d.type);
            const isFileOrDir = d => d.type === 'file' || d.type === 'directory';

            // Add circles for regular code nodes (not files/dirs/docs)
            node.filter(d => !isDocNode(d) && !isFileOrDir(d))
                .append("circle")
                .attr("r", d => {
                    if (d.type === 'subproject') return 28; // Increased from 24
                    // Increase base size and complexity multiplier
                    return d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18; // Was 12 + complexity * 2, max 28, default 15
                })
                .attr("stroke", d => {
                    // Check if node has incoming caller/imports edges (dead code detection)
                    const hasIncoming = allLinks.some(l =>
                        (l.target.id || l.target) === d.id &&
                        (l.type === 'caller' || l.type === 'imports')
                    );
                    if (!hasIncoming && (d.type === 'function' || d.type === 'class' || d.type === 'method')) {
                        // Check if it's not an entry point (main, test, cli files)
                        const isEntryPoint = d.file_path && (
                            d.file_path.includes('main.py') ||
                            d.file_path.includes('__main__.py') ||
                            d.file_path.includes('cli.py') ||
                            d.file_path.includes('test_')
                        );
                        if (!isEntryPoint) {
                            return "#ff6b6b"; // Red border for potentially dead code
                        }
                    }
                    return hasChildren(d) ? "#ffffff" : "none";
                })
                .attr("stroke-width", d => {
                    const hasIncoming = allLinks.some(l =>
                        (l.target.id || l.target) === d.id &&
                        (l.type === 'caller' || l.type === 'imports')
                    );
                    if (!hasIncoming && (d.type === 'function' || d.type === 'class' || d.type === 'method')) {
                        const isEntryPoint = d.file_path && (
                            d.file_path.includes('main.py') ||
                            d.file_path.includes('__main__.py') ||
                            d.file_path.includes('cli.py') ||
                            d.file_path.includes('test_')
                        );
                        if (!isEntryPoint) {
                            return 3; // Thicker red border
                        }
                    }
                    return hasChildren(d) ? 2 : 0;
                })
                .style("fill", d => {
                    const baseColor = d.color || null;
                    if (!baseColor) return null;
                    return getComplexityShade(baseColor, d.complexity);
                });

            // Add rectangles for document nodes
            node.filter(d => isDocNode(d))
                .append("rect")
                .attr("width", d => {
                    const size = d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18; // Increased from 12/28/15
                    return size * 2;
                })
                .attr("height", d => {
                    const size = d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18; // Increased from 12/28/15
                    return size * 2;
                })
                .attr("x", d => {
                    const size = d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18; // Increased from 12/28/15
                    return -size;
                })
                .attr("y", d => {
                    const size = d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18; // Increased from 12/28/15
                    return -size;
                })
                .attr("rx", 2)  // Rounded corners
                .attr("ry", 2)
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => {
                    const baseColor = d.color || null;
                    if (!baseColor) return null;
                    return getComplexityShade(baseColor, d.complexity);
                });

            // Add SVG icons for file and directory nodes
            node.filter(d => isFileOrDir(d))
                .append("path")
                .attr("class", "file-icon")
                .attr("d", d => getFileTypeIcon(d))
                .attr("transform", d => {
                    const scale = d.type === 'directory' ? 2.2 : 1.8; // Increased from 1.8/1.5
                    return `translate(-12, -12) scale(${scale})`;
                })
                .style("color", d => getFileTypeColor(d))
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 1.5 : 0); // Slightly thicker stroke

            // Add expand/collapse indicator - positioned to the left of label
            node.filter(d => hasChildren(d))
                .append("text")
                .attr("class", "expand-indicator")
                .attr("x", d => {
                    const iconRadius = d.type === 'directory' ? 22 : (d.type === 'file' ? 18 : 18); // Increased from 18/15/15
                    return iconRadius + 5;  // Just right of the icon
                })
                .attr("y", 0)
                .attr("dy", "0.6em")
                .attr("text-anchor", "start")
                .style("font-size", "15px") // Slightly larger from 14px
                .style("font-weight", "bold")
                .style("fill", "#ffffff")
                .style("pointer-events", "none")
                .text(d => collapsedNodes.has(d.id) ? "+" : "−");

            // Add labels (show actual import statement for L1 nodes)
            node.append("text")
                .text(d => {
                    // L1 (depth 1) nodes are imports
                    if (d.depth === 1 && d.type !== 'directory' && d.type !== 'file') {
                        if (d.content) {
                            // Extract first line of import statement
                            const importLine = d.content.split('\\n')[0].trim();
                            // Truncate if too long (max 60 chars)
                            return importLine.length > 60 ? importLine.substring(0, 57) + '...' : importLine;
                        }
                        return d.name;  // Fallback to name if no content
                    }
                    return d.name;
                })
                .attr("x", d => {
                    const iconRadius = d.type === 'directory' ? 22 : (d.type === 'file' ? 18 : 18); // Increased from 18/15/15
                    const hasExpand = hasChildren(d);
                    return iconRadius + 8 + (hasExpand ? 24 : 0); // Slightly more offset for larger indicator
                })
                .attr("y", 0)
                .attr("dy", "0.6em")
                .attr("text-anchor", "start")
                .style("font-size", d => {
                    if (d.type === 'subproject') return "13px"; // Slightly larger from 12px
                    if (isFileOrDir(d)) return "12px"; // Larger from 11px
                    return "11px"; // Larger from 10px
                });

            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            updateStats({nodes: visibleNodesList, links: visibleLinks, metadata: {total_files: allNodes.length}});
        }

        function hasChildren(node) {
            return allLinks.some(l => (l.source.id || l.source) === node.id);
        }
    """


def get_zoom_and_navigation_functions() -> str:
    """Get zoom and navigation functions.

    Returns:
        JavaScript string for zoom and navigation
    """
    return """
        // Zoom to fit all visible nodes with appropriate padding
        function zoomToFit(duration = 750) {
            const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
            if (visibleNodesList.length === 0) return;

            // Calculate bounding box of visible nodes
            // Use MUCH more padding to ensure all nodes visible
            const padding = 120;
            let minX = Infinity, minY = Infinity;
            let maxX = -Infinity, maxY = -Infinity;

            visibleNodesList.forEach(d => {
                if (d.x !== undefined && d.y !== undefined) {
                    minX = Math.min(minX, d.x);
                    minY = Math.min(minY, d.y);
                    maxX = Math.max(maxX, d.x);
                    maxY = Math.max(maxY, d.y);
                }
            });

            // Add padding
            minX -= padding;
            minY -= padding;
            maxX += padding;
            maxY += padding;

            const boxWidth = maxX - minX;
            const boxHeight = maxY - minY;

            // Calculate scale to fit ALL nodes in viewport
            const scale = Math.min(
                width / boxWidth,
                height / boxHeight,
                2  // Max zoom level
            ) * 0.5;  // EVEN MORE ZOOM-OUT: 0.7 → 0.5 (29% more margin)

            // Calculate center translation
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const translateX = width / 2 - scale * centerX;
            const translateY = height / 2 - scale * centerY;

            // Apply zoom transform with animation
            svg.transition()
                .duration(duration)
                .call(
                    zoom.transform,
                    d3.zoomIdentity
                        .translate(translateX, translateY)
                        .scale(scale)
                );
        }

        function centerNode(node) {
            // Get current transform to maintain zoom level
            const transform = d3.zoomTransform(svg.node());

            // Calculate translation to center the node in LEFT portion of viewport
            // Position at 30% from left to avoid code pane on right side
            const x = -node.x * transform.k + width * 0.3;
            const y = -node.y * transform.k + height / 2;

            // Apply smooth animation to center the node
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(transform.k));
        }

        function resetView() {
            // Reset to root level nodes only
            visibleNodes = new Set(rootNodes.map(n => n.id));
            collapsedNodes = new Set(rootNodes.map(n => n.id));
            highlightedNode = null;

            // Clean up non-visible objects
            const pane = document.querySelector('.content-pane');
            if (pane) {
                pane.classList.remove('visible');
                // Clear pane content to free memory
                const content = pane.querySelector('.content-container');
                const footer = pane.querySelector('.footer-container');
                if (content) content.innerHTML = '';
                if (footer) footer.innerHTML = '';
            }

            // Remove any highlighted states
            d3.selectAll('.node circle, .node rect')
                .classed('highlighted', false)
                .classed('selected', false);

            // Re-render graph
            renderGraph();

            // Position folders compactly after rendering
            setTimeout(() => {
                const currentNodes = allNodes.filter(n => visibleNodes.has(n.id));
                positionNodesCompactly(currentNodes);
            }, 100);

            // Zoom to fit after positioning
            setTimeout(() => {
                zoomToFit(750);
            }, 300);
        }
    """


def get_interaction_handlers() -> str:
    """Get interaction handler functions (click, expand, collapse).

    Returns:
        JavaScript string for interaction handling
    """
    return """
        function handleNodeClick(event, d) {
            event.stopPropagation();

            // Always show content pane when clicking any node
            showContentPane(d);

            // If node has children, also toggle expansion
            if (hasChildren(d)) {
                const wasCollapsed = collapsedNodes.has(d.id);
                if (wasCollapsed) {
                    expandNode(d);
                } else {
                    collapseNode(d);
                }
                renderGraph();

                // After rendering and nodes have positions, zoom to fit ONLY visible nodes
                if (!wasCollapsed) {
                    setTimeout(() => {
                        simulation.alphaTarget(0);
                        zoomToFit(750);
                    }, 200);
                } else {
                    setTimeout(() => {
                        centerNode(d);
                    }, 200);
                }
            } else {
                setTimeout(() => {
                    centerNode(d);
                }, 100);
            }
        }

        function expandNode(node) {
            collapsedNodes.delete(node.id);

            // Find direct children
            const children = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            children.forEach(child => {
                visibleNodes.add(child.id);
                collapsedNodes.add(child.id); // Children start collapsed
            });
        }

        function collapseNode(node) {
            collapsedNodes.add(node.id);

            // Hide all descendants recursively
            function hideDescendants(parentId) {
                const children = allLinks
                    .filter(l => (l.source.id || l.source) === parentId)
                    .map(l => l.target.id || l.target);

                children.forEach(childId => {
                    visibleNodes.delete(childId);
                    collapsedNodes.delete(childId);
                    hideDescendants(childId);
                });
            }

            hideDescendants(node.id);
        }
    """


def get_tooltip_logic() -> str:
    """Get tooltip display logic with enhanced relationship types.

    Returns:
        JavaScript string for tooltip handling
    """
    return """
        function showTooltip(event, d) {
            // Extract first 2-3 lines of docstring for preview
            let docPreview = '';
            if (d.docstring) {
                const lines = d.docstring.split('\\n').filter(l => l.trim());
                const previewLines = lines.slice(0, 3).join(' ');
                const truncated = previewLines.length > 150 ? previewLines.substring(0, 147) + '...' : previewLines;
                docPreview = `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">${truncated}</div>`;
            }

            tooltip
                .style("display", "block")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .html(`
                    <div><strong>${d.name}</strong></div>
                    <div>Type: ${d.type}</div>
                    ${d.complexity ? `<div>Complexity: ${d.complexity.toFixed(1)}</div>` : ''}
                    ${d.start_line ? `<div>Lines: ${d.start_line}-${d.end_line}</div>` : ''}
                    <div>File: ${d.file_path}</div>
                    ${docPreview}
                `);
        }

        function showLinkTooltip(event, d) {
            const sourceName = allNodes.find(n => n.id === (d.source.id || d.source))?.name || 'Unknown';
            const targetName = allNodes.find(n => n.id === (d.target.id || d.target))?.name || 'Unknown';

            // Special tooltip for cycle links
            if (d.is_cycle) {
                tooltip
                    .style("display", "block")
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY + 10) + "px")
                    .html(`
                        <div style="color: #ff4444;"><strong>⚠️ Circular Dependency Detected</strong></div>
                        <div style="margin-top: 8px;">Path: ${sourceName} → ${targetName}</div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">
                            This indicates a circular call relationship that may lead to infinite recursion or tight coupling.
                        </div>
                    `);
                return;
            }

            // Tooltip content based on link type
            let typeLabel = '';
            let typeDescription = '';
            let extraInfo = '';

            switch(d.type) {
                case 'caller':
                    typeLabel = '📞 Function Call';
                    typeDescription = `${sourceName} calls ${targetName}`;
                    extraInfo = 'This is a direct function call relationship, the most common type of code dependency.';
                    break;
                case 'semantic':
                    typeLabel = '🔗 Semantic Similarity';
                    typeDescription = `${(d.similarity * 100).toFixed(1)}% similar`;
                    extraInfo = `These code chunks have similar meaning or purpose based on their content.`;
                    break;
                case 'imports':
                    typeLabel = '📦 Import Dependency';
                    typeDescription = `${sourceName} imports ${targetName}`;
                    extraInfo = 'This is an explicit import/dependency declaration.';
                    break;
                case 'file_containment':
                    typeLabel = '📄 File Contains';
                    typeDescription = `${sourceName} contains ${targetName}`;
                    extraInfo = 'This file contains the code chunk or function.';
                    break;
                case 'dir_containment':
                    typeLabel = '📁 Directory Contains';
                    typeDescription = `${sourceName} contains ${targetName}`;
                    extraInfo = 'This directory contains the file or subdirectory.';
                    break;
                case 'dir_hierarchy':
                    typeLabel = '🗂️ Directory Hierarchy';
                    typeDescription = `${sourceName} → ${targetName}`;
                    extraInfo = 'Parent-child directory structure relationship.';
                    break;
                case 'method':
                    typeLabel = '⚙️ Method Relationship';
                    typeDescription = `${sourceName} ↔ ${targetName}`;
                    extraInfo = 'Class method relationship.';
                    break;
                case 'module':
                    typeLabel = '📚 Module Relationship';
                    typeDescription = `${sourceName} ↔ ${targetName}`;
                    extraInfo = 'Module-level relationship.';
                    break;
                case 'dependency':
                    typeLabel = '🔀 Dependency';
                    typeDescription = `${sourceName} depends on ${targetName}`;
                    extraInfo = 'General code dependency relationship.';
                    break;
                default:
                    typeLabel = `🔗 ${d.type || 'Unknown'}`;
                    typeDescription = `${sourceName} → ${targetName}`;
                    extraInfo = 'Code relationship.';
            }

            tooltip
                .style("display", "block")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .html(`
                    <div><strong>${typeLabel}</strong></div>
                    <div style="margin-top: 4px;">${typeDescription}</div>
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">
                        ${extraInfo}
                    </div>
                `);
        }

        function hideTooltip() {
            tooltip.style("display", "none");
        }
    """


def get_drag_and_stats_functions() -> str:
    """Get drag behavior and stats update functions.

    Returns:
        JavaScript string for drag and stats
    """
    return """
        function drag(simulation) {
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }

        function updateStats(data) {
            const stats = d3.select("#stats");
            stats.html(`
                <div>Nodes: ${data.nodes.length}</div>
                <div>Links: ${data.links.length}</div>
                ${data.metadata ? `<div>Files: ${data.metadata.total_files || 'N/A'}</div>` : ''}
                ${data.metadata && data.metadata.is_monorepo ? `<div>Monorepo: ${data.metadata.subprojects.length} subprojects</div>` : ''}
            `);

            // Show subproject legend if monorepo
            if (data.metadata && data.metadata.is_monorepo && data.metadata.subprojects.length > 0) {
                const subprojectsLegend = d3.select("#subprojects-legend");
                const subprojectsList = d3.select("#subprojects-list");

                subprojectsLegend.style("display", "block");

                // Get subproject nodes with colors
                const subprojectNodes = allNodes.filter(n => n.type === 'subproject');

                subprojectsList.html(
                    subprojectNodes.map(sp =>
                        `<div class="legend-item">
                            <span class="legend-color" style="background: ${sp.color};"></span> ${sp.name}
                        </div>`
                    ).join('')
                );
            }
        }
    """


def get_breadcrumb_functions() -> str:
    """Get breadcrumb navigation functions for file/directory paths.

    Returns:
        JavaScript string for breadcrumb generation and navigation
    """
    return """
        // Generate breadcrumb navigation for a file/directory path
        function generateBreadcrumbs(node) {
            if (!node.file_path) return '';

            let nodePath = node.file_path;

            // Strip project root prefix to show relative paths
            // Look for 'mcp-vector-search' directory and strip everything before it
            const projectRootIndex = nodePath.indexOf('mcp-vector-search');
            if (projectRootIndex >= 0) {
                // Find the first '/' after 'mcp-vector-search' and take everything after it
                const afterProjectRoot = nodePath.indexOf('/', projectRootIndex);
                if (afterProjectRoot >= 0) {
                    nodePath = nodePath.substring(afterProjectRoot + 1);
                }
            }

            const segments = nodePath.split('/').filter(s => s.length > 0);

            if (segments.length === 0) return '';

            let breadcrumbHTML = '<div class="breadcrumb-nav">';
            breadcrumbHTML += '<span class="breadcrumb-root" onclick="navigateToRoot()">🏠 Root</span>';

            let currentPath = '';
            segments.forEach((segment, index) => {
                currentPath += (currentPath ? '/' : '') + segment;
                const isLast = (index === segments.length - 1);

                breadcrumbHTML += ' <span class="breadcrumb-separator">/</span> ';

                if (!isLast) {
                    // Parent directories are clickable
                    breadcrumbHTML += `<span class="breadcrumb-link" onclick="navigateToBreadcrumb('${currentPath}')">${segment}</span>`;
                } else {
                    // Current file/directory is not clickable (highlighted)
                    breadcrumbHTML += `<span class="breadcrumb-current">${segment}</span>`;
                }
            });

            breadcrumbHTML += '</div>';
            return breadcrumbHTML;
        }

        // Navigate to a breadcrumb link (find and highlight the node)
        function navigateToBreadcrumb(path) {
            // Try to find the directory node by path
            const targetNode = allNodes.find(n => n.file_path === path || n.dir_path === path);

            if (targetNode) {
                navigateToNode(targetNode);
            }
        }

        // Navigate to project root
        function navigateToRoot() {
            resetView();
        }
    """


def get_code_chunks_functions() -> str:
    """Get code chunks display functions for file viewer.

    Returns:
        JavaScript string for code chunks navigation

    Design Decision: Clickable code chunks section for file detail pane

    Rationale: Users need quick navigation to specific functions/classes within files.
    Selected list-based UI with line ranges and type badges for clarity.

    Trade-offs:
    - Performance: O(n) filtering per file vs. pre-indexing (chose simplicity)
    - UX: Shows all chunks vs. grouped by type (chose comprehensive view)
    - Visual: Icons vs. badges (chose both for maximum clarity)

    Alternatives Considered:
    1. Tree structure (functions under classes): Rejected - adds complexity
    2. Grouped by type: Rejected - line number order more intuitive
    3. Separate tab: Rejected - want chunks visible by default

    Extension Points: Can add filtering by chunk_type, search box, or grouping later.
    """
    return """
        // Get all code chunks for a given file
        function getCodeChunksForFile(filePath) {
            if (!filePath) return [];

            const chunks = allNodes.filter(n =>
                n.type === 'code' ||
                (n.file_path === filePath &&
                 ['function', 'class', 'method'].includes(n.type))
            ).filter(n => n.file_path === filePath || n.parent_file === filePath);

            // Sort by start_line
            return chunks.sort((a, b) =>
                (a.start_line || 0) - (b.start_line || 0)
            );
        }

        // Generate HTML for code chunks section
        function generateCodeChunksSection(filePath) {
            const chunks = getCodeChunksForFile(filePath);

            if (chunks.length === 0) {
                return ''; // No chunks, don't show section
            }

            let html = '<div class="code-chunks-section">';
            html += '<h4 class="section-header">Code Chunks (' + chunks.length + ')</h4>';
            html += '<div class="code-chunks-list">';

            chunks.forEach(chunk => {
                const icon = getChunkIcon(chunk.type);
                const lineRange = chunk.start_line ?
                    ` <span class="line-range">L${chunk.start_line}-${chunk.end_line || chunk.start_line}</span>` :
                    '';

                html += `
                    <div class="code-chunk-item" data-type="${chunk.type}" onclick="navigateToChunk('${chunk.id}')">
                        <span class="chunk-icon">${icon}</span>
                        <span class="chunk-name">${escapeHtml(chunk.name || 'unnamed')}</span>
                        ${lineRange}
                        <span class="chunk-type">${chunk.type || 'code'}</span>
                    </div>
                `;
            });

            html += '</div></div>';
            return html;
        }

        // Get icon for chunk type
        function getChunkIcon(chunkType) {
            const icons = {
                'function': '⚡',
                'class': '📦',
                'method': '🔧',
                'variable': '📊',
                'import': '📥',
                'export': '📤',
                'code': '📄'
            };
            return icons[chunkType] || '📄';
        }

        // Navigate to a code chunk (highlight and show details)
        function navigateToChunk(chunkId) {
            const chunk = allNodes.find(n => n.id === chunkId);
            if (chunk) {
                navigateToNode(chunk);
            }
        }

        // Navigate to a file by path (reload parent file view from code chunk)
        function navigateToFile(filePath) {
            const fileNode = allNodes.find(n => n.file_path === filePath && n.type === 'file');
            if (fileNode) {
                navigateToNode(fileNode);
            }
        }
    """


def get_content_pane_functions() -> str:
    """Get content pane display functions.

    Returns:
        JavaScript string for content pane
    """
    return """
        function showContentPane(node) {
            // Highlight the node
            highlightedNode = node;
            renderGraph();

            // Populate content pane
            const pane = document.getElementById('content-pane');
            const title = document.getElementById('pane-title');
            const meta = document.getElementById('pane-meta');
            const content = document.getElementById('pane-content');
            const footer = document.getElementById('pane-footer');

            // Generate and inject breadcrumbs at the top
            const breadcrumbs = generateBreadcrumbs(node);

            // Set title with actual import statement for L1 nodes
            if (node.depth === 1 && node.type !== 'directory' && node.type !== 'file') {
                if (node.content) {
                    const importLine = node.content.split('\\n')[0].trim();
                    title.innerHTML = breadcrumbs + importLine;
                } else {
                    title.innerHTML = breadcrumbs + `Import: ${node.name}`;
                }
            } else {
                title.innerHTML = breadcrumbs + node.name;
            }

            // Set metadata
            meta.textContent = node.type;

            // Build footer with annotations
            let footerHtml = '';
            if (node.language) {
                footerHtml += `<span class="footer-item"><span class="footer-label">Language:</span> ${node.language}</span>`;
            }
            footerHtml += `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;

            if (node.start_line !== undefined && node.end_line !== undefined) {
                const totalLines = node.end_line - node.start_line + 1;

                // Build line info string with optional non-doc code lines
                let lineInfo = `${node.start_line}-${node.end_line} (${totalLines} lines`;

                // Add non-documentation code lines if available
                if (node.non_doc_lines !== undefined && node.non_doc_lines > 0) {
                    lineInfo += `, ${node.non_doc_lines} code`;
                }

                lineInfo += ')';

                if (node.type === 'function' || node.type === 'class' || node.type === 'method') {
                    footerHtml += `<span class="footer-item"><span class="footer-label">Lines:</span> ${lineInfo}</span>`;
                } else if (node.type === 'file') {
                    footerHtml += `<span class="footer-item"><span class="footer-label">File Lines:</span> ${totalLines}</span>`;
                } else {
                    footerHtml += `<span class="footer-item"><span class="footer-label">Location:</span> ${lineInfo}</span>`;
                }

                if (node.complexity && node.complexity > 0) {
                    footerHtml += `<span class="footer-item"><span class="footer-label">Complexity:</span> ${node.complexity}</span>`;
                }
            }

            footer.innerHTML = footerHtml;

            // Display content based on node type
            if (node.type === 'directory') {
                showDirectoryContents(node, content, footer);
            } else if (node.type === 'file') {
                showFileContents(node, content);
            } else if (node.depth === 1 && node.type !== 'directory' && node.type !== 'file') {
                showImportDetails(node, content);
            } else {
                showCodeContent(node, content);
            }

            pane.classList.add('visible');
        }

        function showDirectoryContents(node, container, footer) {
            const children = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            if (children.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">Empty directory</p>';
                footer.innerHTML = `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;
                return;
            }

            const files = children.filter(n => n.type === 'file');
            const subdirs = children.filter(n => n.type === 'directory');
            const chunks = children.filter(n => n.type !== 'file' && n.type !== 'directory');

            let html = '<ul class="directory-list">';

            subdirs.forEach(child => {
                html += `
                    <li data-node-id="${child.id}">
                        <span class="item-icon">📁</span>
                        ${child.name}
                    </li>
                `;
            });

            files.forEach(child => {
                html += `
                    <li data-node-id="${child.id}">
                        <span class="item-icon">📄</span>
                        ${child.name}
                    </li>
                `;
            });

            chunks.forEach(child => {
                const icon = child.type === 'class' ? '🔷' : child.type === 'function' ? '⚡' : '📝';
                html += `
                    <li data-node-id="${child.id}">
                        <span class="item-icon">${icon}</span>
                        ${child.name}
                    </li>
                `;
            });

            html += '</ul>';
            container.innerHTML = html;

            const listItems = container.querySelectorAll('.directory-list li');
            listItems.forEach(item => {
                item.addEventListener('click', () => {
                    const nodeId = item.getAttribute('data-node-id');
                    const childNode = allNodes.find(n => n.id === nodeId);
                    if (childNode) {
                        showContentPane(childNode);
                    }
                });
            });

            footer.innerHTML = `
                <span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>
                <span class="footer-item"><span class="footer-label">Total:</span> ${children.length} items (${subdirs.length} directories, ${files.length} files, ${chunks.length} code chunks)</span>
            `;
        }

        function showFileContents(node, container) {
            const fileChunks = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            if (fileChunks.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">No code chunks found in this file</p>';
                return;
            }

            const sortedChunks = fileChunks
                .filter(c => c.content)
                .sort((a, b) => a.start_line - b.start_line);

            if (sortedChunks.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">File content not available</p>';
                return;
            }

            const fullContent = sortedChunks.map(c => c.content).join('\\n\\n');

            // Generate code chunks section HTML
            const codeChunksHtml = generateCodeChunksSection(node.file_path || node.id);

            container.innerHTML = `
                ${codeChunksHtml}
                <p style="color: #8b949e; font-size: 11px; margin-bottom: 12px; ${codeChunksHtml ? 'margin-top: 20px;' : ''}">
                    Contains ${fileChunks.length} code chunks
                </p>
                <pre><code>${escapeHtml(fullContent)}</code></pre>
            `;
        }

        function showImportDetails(node, container) {
            const importHtml = `
                <div class="import-details">
                    ${node.content ? `
                        <div style="margin-bottom: 16px;">
                            <div class="detail-label" style="margin-bottom: 8px;">Import Statement:</div>
                            <pre><code>${escapeHtml(node.content)}</code></pre>
                        </div>
                    ` : '<p style="color: #8b949e;">No import content available</p>'}
                </div>
            `;

            container.innerHTML = importHtml;
        }

        function parseDocstring(docstring) {
            if (!docstring) return { brief: '', sections: {} };

            const lines = docstring.split('\\n');
            const sections = {};
            let currentSection = 'brief';
            let currentContent = [];

            for (let line of lines) {
                const trimmed = line.trim();
                const sectionMatch = trimmed.match(/^(Args?|Returns?|Yields?|Raises?|Note|Notes|Example|Examples|See Also|Docs?|Parameters?):?$/i);

                if (sectionMatch) {
                    if (currentContent.length > 0) {
                        sections[currentSection] = currentContent.join('\\n').trim();
                    }
                    currentSection = sectionMatch[1].toLowerCase();
                    currentContent = [];
                } else {
                    currentContent.push(line);
                }
            }

            if (currentContent.length > 0) {
                sections[currentSection] = currentContent.join('\\n').trim();
            }

            return { brief: sections.brief || '', sections };
        }

        // Create linkable code with Python primitives bolded
        function createLinkableCode(code, currentNodeId) {
            // Build map of available nodes (functions, classes, methods)
            const nodeMap = new Map();
            allNodes.forEach(node => {
                if (node.type === 'function' || node.type === 'class' || node.type === 'method') {
                    nodeMap.set(node.name, node.id);
                }
            });

            // Escape HTML first
            let html = code
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Find and link function/class references
            // Match: word followed by '(' or preceded by 'class '
            const identifierRegex = /\\b([a-zA-Z_][a-zA-Z0-9_]*)\\s*(?=\\()|(?<=class\\s+)([a-zA-Z_][a-zA-Z0-9_]*)/g;

            // Collect matches to avoid overlapping replacements
            const matches = [];
            let match;
            while ((match = identifierRegex.exec(code)) !== null) {
                const name = match[1] || match[2];
                if (nodeMap.has(name) && nodeMap.get(name) !== currentNodeId) {
                    matches.push({
                        index: match.index,
                        length: name.length,
                        name: name,
                        nodeId: nodeMap.get(name)
                    });
                }
            }

            // Apply replacements in reverse order to preserve indices
            matches.reverse().forEach(m => {
                const before = html.substring(0, m.index);
                const linkText = html.substring(m.index, m.index + m.length);
                const after = html.substring(m.index + m.length);

                html = before +
                       `<span class="code-link" data-node-id="${m.nodeId}" title="Jump to ${m.name}">${linkText}</span>` +
                       after;
            });

            // Apply primitive bolding AFTER creating links
            html = boldPythonPrimitives(html);

            return html;
        }

        // Bold Python primitives (keywords and built-ins)
        function boldPythonPrimitives(html) {
            // Python keywords
            const keywords = [
                'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return',
                'import', 'from', 'try', 'except', 'finally', 'with', 'as',
                'async', 'await', 'yield', 'lambda', 'pass', 'break', 'continue',
                'raise', 'assert', 'del', 'global', 'nonlocal', 'is', 'in',
                'and', 'or', 'not'
            ];

            // Built-in types and functions
            const builtins = [
                'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
                'None', 'True', 'False', 'type', 'len', 'range', 'enumerate',
                'zip', 'map', 'filter', 'sorted', 'reversed', 'any', 'all',
                'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'print'
            ];

            // Combine all primitives
            const allPrimitives = [...keywords, ...builtins];

            // Create regex pattern: \b(keyword1|keyword2|...)\b
            // Use word boundaries to avoid matching parts of identifiers
            const pattern = new RegExp(`\\\\b(${allPrimitives.join('|')})\\\\b`, 'g');

            // Replace with bold version
            // Important: Skip content inside existing HTML tags (like code-link spans)
            const result = html.replace(pattern, (match) => {
                return `<strong style="color: #ff7b72; font-weight: 600;">${match}</strong>`;
            });

            return result;
        }

        function showCodeContent(node, container) {
            let html = '';

            const docInfo = parseDocstring(node.docstring);

            if (docInfo.brief && docInfo.brief.trim()) {
                html += `
                    <div style="margin-bottom: 16px; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px;">
                        <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px; font-weight: 600;">DESCRIPTION</div>
                        <pre style="margin: 0; padding: 0; background: transparent; border: none; white-space: pre-wrap;"><code>${escapeHtml(docInfo.brief)}</code></pre>
                    </div>
                `;
            }

            if (node.content) {
                // Use linkable code with primitives bolding
                const linkedCode = createLinkableCode(node.content, node.id);
                html += `<pre><code>${linkedCode}</code></pre>`;
            } else {
                html += '<p style="color: #8b949e;">No content available</p>';
            }

            container.innerHTML = html;

            // Add click handler for code links
            const codeLinks = container.querySelectorAll('.code-link');
            codeLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const nodeId = link.getAttribute('data-node-id');
                    const targetNode = allNodes.find(n => n.id === nodeId);
                    if (targetNode) {
                        navigateToNode(targetNode);
                    }
                });
            });

            const footer = document.getElementById('pane-footer');
            let footerHtml = '';

            if (node.language) {
                footerHtml += `<div class="footer-item"><span class="footer-label">Language:</span> <span class="footer-value">${node.language}</span></div>`;
            }
            footerHtml += `<div class="footer-item"><span class="footer-label">File:</span> <a href="#" class="file-path-link" onclick="navigateToFile('${node.file_path}'); return false;" style="color: #58a6ff; text-decoration: none; cursor: pointer;">${node.file_path}</a></div>`;
            if (node.start_line) {
                footerHtml += `<div class="footer-item"><span class="footer-label">Lines:</span> <span class="footer-value">${node.start_line}-${node.end_line}</span></div>`;
            }

            if (node.callers && node.callers.length > 0) {
                footerHtml += `<div class="footer-item" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d;">`;
                footerHtml += `<span class="footer-label">Called By:</span><br/>`;
                node.callers.forEach(caller => {
                    const fileName = caller.file.split('/').pop();
                    const callerDisplay = `${fileName}::${caller.name}`;
                    footerHtml += `<span class="footer-value" style="display: block; margin-left: 8px; margin-top: 4px;">
                        <a href="#" class="caller-link" data-chunk-id="${caller.chunk_id}" style="color: #58a6ff; text-decoration: none; cursor: pointer;">
                            • ${escapeHtml(callerDisplay)}
                        </a>
                    </span>`;
                });
                footerHtml += `</div>`;
            } else if (node.type === 'function' || node.type === 'method' || node.type === 'class') {
                footerHtml += `<div class="footer-item" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d;">`;
                footerHtml += `<span class="footer-label">Called By:</span> <span class="footer-value" style="font-style: italic; color: #6e7681;">(No external callers found)</span>`;
                footerHtml += `</div>`;
            }

            const sectionLabels = {
                'docs': 'Docs', 'doc': 'Docs',
                'args': 'Args', 'arg': 'Args',
                'parameters': 'Args', 'parameter': 'Args',
                'returns': 'Returns', 'return': 'Returns',
                'yields': 'Yields', 'yield': 'Yields',
                'raises': 'Raises', 'raise': 'Raises',
                'note': 'Note', 'notes': 'Note',
                'example': 'Example', 'examples': 'Example',
            };

            for (let [key, content] of Object.entries(docInfo.sections)) {
                if (key === 'brief') continue;

                const label = sectionLabels[key] || key.charAt(0).toUpperCase() + key.slice(1);
                const truncated = content.length > 200 ? content.substring(0, 197) + '...' : content;

                footerHtml += `<div class="footer-item"><span class="footer-label">${label}:</span> <span class="footer-value">${escapeHtml(truncated)}</span></div>`;
            }

            footer.innerHTML = footerHtml;

            const callerLinks = footer.querySelectorAll('.caller-link');
            callerLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const chunkId = link.getAttribute('data-chunk-id');
                    const callerNode = allNodes.find(n => n.id === chunkId);
                    if (callerNode) {
                        navigateToNode(callerNode);
                    }
                });
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function navigateToNode(targetNode) {
            if (!visibleNodes.has(targetNode.id)) {
                expandParentsToNode(targetNode);
                renderGraph();
            }

            showContentPane(targetNode);

            setTimeout(() => {
                if (targetNode.x !== undefined && targetNode.y !== undefined) {
                    const scale = 1.5;
                    const translateX = width * 0.3 - scale * targetNode.x;
                    const translateY = height / 2 - scale * targetNode.y;

                    svg.transition()
                        .duration(750)
                        .call(
                            zoom.transform,
                            d3.zoomIdentity
                                .translate(translateX, translateY)
                                .scale(scale)
                        );
                }
            }, 200);
        }

        function expandParentsToNode(targetNode) {
            const path = [];
            let current = targetNode;

            while (current) {
                path.unshift(current);
                const parentLink = allLinks.find(l =>
                    (l.target.id || l.target) === current.id &&
                    (l.type !== 'semantic' && l.type !== 'dependency')
                );
                if (parentLink) {
                    const parentId = parentLink.source.id || parentLink.source;
                    current = allNodes.find(n => n.id === parentId);
                } else {
                    break;
                }
            }

            path.forEach(node => {
                if (!visibleNodes.has(node.id)) {
                    visibleNodes.add(node.id);
                }
                if (collapsedNodes.has(node.id)) {
                    expandNode(node);
                }
            });
        }

        function closeContentPane() {
            const pane = document.getElementById('content-pane');
            pane.classList.remove('visible');

            highlightedNode = null;
            renderGraph();
        }
    """


def get_data_loading_logic() -> str:
    """Get data loading logic with progress indicators.

    Returns:
        JavaScript string for data loading
    """
    return """
        // Auto-load graph data on page load with progress tracking
        window.addEventListener('DOMContentLoaded', () => {
            const loadingEl = document.getElementById('loading');

            // Show initial loading message
            loadingEl.innerHTML = '<label style="color: #58a6ff;"><span class="spinner"></span>Loading graph data...</label><br>' +
                                 '<div style="margin-top: 8px; background: #21262d; border-radius: 4px; height: 20px; width: 250px; position: relative; overflow: hidden;">' +
                                 '<div id="progress-bar" style="background: #238636; height: 100%; width: 0%; transition: width 0.3s;"></div>' +
                                 '</div>' +
                                 '<small id="progress-text" style="color: #8b949e; margin-top: 4px; display: block;">Connecting...</small>';

            // Create abort controller for timeout
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout

            fetch("chunk-graph.json", { signal: controller.signal })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const contentLength = response.headers.get('content-length');
                    const total = contentLength ? parseInt(contentLength, 10) : 0;
                    let loaded = 0;

                    const progressBar = document.getElementById('progress-bar');
                    const progressText = document.getElementById('progress-text');

                    if (total > 0) {
                        const sizeMB = (total / (1024 * 1024)).toFixed(1);
                        progressText.textContent = `Downloading ${sizeMB}MB...`;
                    } else {
                        progressText.textContent = 'Downloading...';
                    }

                    // Create a new response with progress tracking
                    const reader = response.body.getReader();
                    const stream = new ReadableStream({
                        start(controller) {
                            function push() {
                                reader.read().then(({ done, value }) => {
                                    if (done) {
                                        controller.close();
                                        return;
                                    }

                                    loaded += value.byteLength;

                                    if (total > 0) {
                                        const percent = Math.round((loaded / total) * 100);
                                        progressBar.style.width = percent + '%';
                                        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                                        const totalMB = (total / (1024 * 1024)).toFixed(1);
                                        progressText.textContent = `Downloaded ${loadedMB}MB / ${totalMB}MB (${percent}%)`;
                                    } else {
                                        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                                        progressText.textContent = `Downloaded ${loadedMB}MB...`;
                                    }

                                    controller.enqueue(value);
                                    push();
                                }).catch(err => {
                                    console.error('Stream reading error:', err);
                                    controller.error(err);
                                });
                            }
                            push();
                        }
                    });

                    return new Response(stream);
                })
                .then(response => {
                    clearTimeout(timeout);

                    const progressText = document.getElementById('progress-text');
                    const progressBar = document.getElementById('progress-bar');
                    progressBar.style.width = '100%';
                    progressText.textContent = 'Parsing JSON data...';

                    return response.json();
                })
                .then(data => {
                    clearTimeout(timeout);
                    loadingEl.innerHTML = '<label style="color: #238636;">✓ Graph loaded successfully</label>';
                    setTimeout(() => loadingEl.style.display = 'none', 2000);
                    visualizeGraph(data);
                })
                .catch(err => {
                    clearTimeout(timeout);

                    let errorMsg = err.message;
                    if (err.name === 'AbortError') {
                        errorMsg = 'Loading timeout - file may be too large or server unresponsive';
                    }

                    loadingEl.innerHTML = `<label style="color: #f85149;">✗ Failed to load graph data</label><br>` +
                                         `<small style="color: #8b949e;">${errorMsg}</small><br>` +
                                         `<small style="color: #8b949e;">Run: mcp-vector-search visualize export</small>`;
                    console.error("Failed to load graph:", err);
                });
        });

        // Reset view button event handler
        document.getElementById('reset-view-btn').addEventListener('click', () => {
            resetView();
        });
    """


def get_all_scripts() -> str:
    """Get all JavaScript code combined.

    Returns:
        Complete JavaScript string for the visualization
    """
    return "".join(
        [
            get_d3_initialization(),
            get_file_type_functions(),
            get_spacing_calculation_functions(),
            get_loading_spinner_functions(),
            get_graph_visualization_functions(),
            get_zoom_and_navigation_functions(),
            get_interaction_handlers(),
            get_tooltip_logic(),
            get_drag_and_stats_functions(),
            get_breadcrumb_functions(),
            get_code_chunks_functions(),
            get_content_pane_functions(),
            get_data_loading_logic(),
        ]
    )
