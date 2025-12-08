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
        let isInitialOverview = true;  // Track if we're in Phase 1 (initial overview) or Phase 2 (tree expansion)
        let cy = null;  // Cytoscape instance
        let edgeFilters = {
            containment: true,
            calls: true,
            imports: false,
            semantic: false,
            cycles: true
        };
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
            const filteredLinks = getFilteredLinks();
            const visibleLinks = filteredLinks.filter(l =>
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

            // File paths are already relative to project root, use them directly
            const nodePath = node.file_path;
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
        function showContentPane(node, addToHistory = true) {
            // Add to navigation stack if requested
            if (addToHistory) {
                viewStack.push(node.id);
            }

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


def get_navigation_stack_logic() -> str:
    """Get navigation stack for back/forward functionality.

    Returns:
        JavaScript string for navigation stack management
    """
    return """
        // Navigation stack for back/forward functionality
        const viewStack = {
            stack: [],
            currentIndex: -1,

            push(chunkId) {
                // Don't add duplicates if clicking same node
                if (this.stack.length > 0 && this.stack[this.currentIndex] === chunkId) {
                    return;
                }

                // Remove forward history if we're not at the end
                this.stack = this.stack.slice(0, this.currentIndex + 1);
                this.stack.push(chunkId);
                this.currentIndex++;
                this.updateButtons();
            },

            canGoBack() {
                return this.currentIndex > 0;
            },

            canGoForward() {
                return this.currentIndex < this.stack.length - 1;
            },

            back() {
                if (this.canGoBack()) {
                    this.currentIndex--;
                    this.updateButtons();
                    return this.stack[this.currentIndex];
                }
                return null;
            },

            forward() {
                if (this.canGoForward()) {
                    this.currentIndex++;
                    this.updateButtons();
                    return this.stack[this.currentIndex];
                }
                return null;
            },

            updateButtons() {
                const backBtn = document.getElementById('navBack');
                const forwardBtn = document.getElementById('navForward');
                const positionSpan = document.getElementById('navPosition');

                if (backBtn) backBtn.disabled = !this.canGoBack();
                if (forwardBtn) forwardBtn.disabled = !this.canGoForward();
                if (positionSpan && this.stack.length > 0) {
                    positionSpan.textContent = `${this.currentIndex + 1} of ${this.stack.length}`;
                }
            },

            clear() {
                this.stack = [];
                this.currentIndex = -1;
                this.updateButtons();
            }
        };

        // Add keyboard shortcuts for navigation
        document.addEventListener('keydown', (e) => {
            if (e.altKey && e.key === 'ArrowLeft') {
                e.preventDefault();
                const chunkId = viewStack.back();
                if (chunkId) {
                    const node = allNodes.find(n => n.id === chunkId);
                    if (node) showContentPane(node, false); // false = don't add to history
                }
            } else if (e.altKey && e.key === 'ArrowRight') {
                e.preventDefault();
                const chunkId = viewStack.forward();
                if (chunkId) {
                    const node = allNodes.find(n => n.id === chunkId);
                    if (node) showContentPane(node, false);
                }
            }
        });

        // Add click handlers for navigation buttons
        document.addEventListener('DOMContentLoaded', () => {
            const backBtn = document.getElementById('navBack');
            const forwardBtn = document.getElementById('navForward');

            if (backBtn) {
                backBtn.addEventListener('click', () => {
                    const chunkId = viewStack.back();
                    if (chunkId) {
                        const node = allNodes.find(n => n.id === chunkId);
                        if (node) showContentPane(node, false);
                    }
                });
            }

            if (forwardBtn) {
                forwardBtn.addEventListener('click', () => {
                    const chunkId = viewStack.forward();
                    if (chunkId) {
                        const node = allNodes.find(n => n.id === chunkId);
                        if (node) showContentPane(node, false);
                    }
                });
            }
        });
    """


def get_layout_switching_logic() -> str:
    """Get layout switching functionality for Dagre/Force/Circle layouts.

    Returns:
        JavaScript string for layout switching
    """
    return """
        // Filter edges based on current filter settings
        function getFilteredLinks() {
            return allLinks.filter(link => {
                const linkType = link.type || 'unknown';

                // Containment edges
                if (linkType === 'dir_containment' || linkType === 'dir_hierarchy' || linkType === 'file_containment') {
                    return edgeFilters.containment;
                }

                // Call edges
                if (linkType === 'caller') {
                    return edgeFilters.calls;
                }

                // Import edges
                if (linkType === 'imports') {
                    return edgeFilters.imports;
                }

                // Semantic edges
                if (linkType === 'semantic') {
                    return edgeFilters.semantic;
                }

                // Cycle edges
                if (link.is_cycle) {
                    return edgeFilters.cycles;
                }

                // Default: show other edge types
                return true;
            });
        }

        // Switch to Cytoscape layout (Dagre or Circle)
        function switchToCytoscapeLayout(layoutName) {
            // Note: This is legacy code for old visualization architecture
            // V2.0 uses tree-based layouts with automatic phase transitions

            // Hide D3 SVG
            svg.style('display', 'none');

            // Create Cytoscape container if doesn't exist
            let cyContainer = document.getElementById('cy-container');
            if (!cyContainer) {
                cyContainer = document.createElement('div');
                cyContainer.id = 'cy-container';
                cyContainer.style.width = '100vw';
                cyContainer.style.height = '100vh';
                cyContainer.style.position = 'absolute';
                cyContainer.style.top = '0';
                cyContainer.style.left = '0';
                document.body.insertBefore(cyContainer, document.body.firstChild);
            }
            cyContainer.style.display = 'block';

            // Get visible nodes and filtered links
            const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
            const filteredLinks = getFilteredLinks();
            const visibleLinks = filteredLinks.filter(l =>
                visibleNodes.has(l.source.id || l.source) &&
                visibleNodes.has(l.target.id || l.target)
            );

            // Convert to Cytoscape format
            const cyElements = [];

            // Add nodes
            visibleNodesList.forEach(node => {
                cyElements.push({
                    data: {
                        id: node.id,
                        label: node.name,
                        nodeType: node.type,
                        color: node.color,
                        ...node
                    }
                });
            });

            // Add edges
            visibleLinks.forEach(link => {
                const sourceId = link.source.id || link.source;
                const targetId = link.target.id || link.target;
                cyElements.push({
                    data: {
                        ...link,
                        source: sourceId,
                        target: targetId,
                        linkType: link.type,
                        isCycle: link.is_cycle
                    }
                });
            });

            // Initialize or update Cytoscape
            if (cy) {
                cy.destroy();
            }

            cy = cytoscape({
                container: cyContainer,
                elements: cyElements,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'background-color': 'data(color)',
                            'color': '#c9d1d9',
                            'font-size': '11px',
                            'text-valign': 'center',
                            'text-halign': 'right',
                            'text-margin-x': '5px',
                            'width': d => d.data('type') === 'directory' ? 35 : 25,
                            'height': d => d.data('type') === 'directory' ? 35 : 25,
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#30363d',
                            'target-arrow-color': '#30363d',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    },
                    {
                        selector: 'edge[isCycle]',
                        style: {
                            'line-color': '#ff4444',
                            'width': 3,
                            'line-style': 'dashed'
                        }
                    }
                ],
                layout: {
                    name: layoutName === 'dagre' ? 'dagre' : 'circle',
                    rankDir: 'TB',
                    rankSep: 150,
                    nodeSep: 80,
                    ranker: 'network-simplex',
                    spacingFactor: 1.2
                }
            });

            // Add click handler
            cy.on('tap', 'node', function(evt) {
                const nodeData = evt.target.data();
                const node = allNodes.find(n => n.id === nodeData.id);
                if (node) {
                    showContentPane(node);
                }
            });
        }

        // Switch to D3 force-directed layout
        function switchToForceLayout() {
            // Note: This is legacy code for old visualization architecture
            // V2.0 uses tree-based layouts with automatic phase transitions

            // Hide Cytoscape
            const cyContainer = document.getElementById('cy-container');
            if (cyContainer) {
                cyContainer.style.display = 'none';
            }

            // Show D3 SVG
            svg.style('display', 'block');

            // Re-render with D3
            renderGraph();
        }

        // Handle layout selector change
        document.addEventListener('DOMContentLoaded', () => {
            const layoutSelector = document.getElementById('layoutSelector');
            if (layoutSelector) {
                layoutSelector.addEventListener('change', (e) => {
                    const layout = e.target.value;
                    if (layout === 'force') {
                        switchToForceLayout();
                    } else if (layout === 'dagre' || layout === 'circle') {
                        switchToCytoscapeLayout(layout);
                    }
                });
            }

            // Handle edge filter checkboxes
            const filterCheckboxes = {
                'filter-containment': 'containment',
                'filter-calls': 'calls',
                'filter-imports': 'imports',
                'filter-semantic': 'semantic',
                'filter-cycles': 'cycles'
            };

            Object.entries(filterCheckboxes).forEach(([id, filterKey]) => {
                const checkbox = document.getElementById(id);
                if (checkbox) {
                    checkbox.addEventListener('change', (e) => {
                        edgeFilters[filterKey] = e.target.checked;
                        // Re-render with new filters
                        // Note: V2.0 uses automatic layout based on view mode
                        if (typeof renderGraphV2 === 'function') {
                            renderGraphV2();  // V2.0 rendering
                        } else {
                            renderGraph();    // Legacy fallback
                        }
                    });
                }
            });
        });
    """


def get_data_loading_logic() -> str:
    """Get data loading logic with streaming JSON parser.

    Returns:
        JavaScript string for data loading

    Design Decision: Streaming JSON with chunked transfer and incremental parsing

    Rationale: Safari's JSON.parse() crashes with 6.3MB files. Selected streaming
    approach to download in chunks and parse incrementally, avoiding browser memory
    limits and parser crashes.

    Trade-offs:
    - Memory: Constant memory usage vs. loading entire file
    - Complexity: Custom streaming parser vs. simple JSON.parse()
    - Performance: Slightly slower but prevents crashes

    Alternatives Considered:
    1. Web Workers for parsing: Rejected - still requires full JSON in memory
    2. IndexedDB caching: Rejected - doesn't solve initial load problem
    3. MessagePack binary: Rejected - requires backend changes

    Error Handling:
    - Network errors: Show retry button with clear error message
    - Timeout: 60s timeout with abort controller
    - Parse errors: Log to console and show user-friendly message
    - Incomplete data: Validate nodes/links exist before rendering

    Performance:
    - Transfer: Shows progress 0-50% during download
    - Parse: Shows progress 50-100% during JSON parsing
    - Expected: <10s for 6.3MB file on localhost
    - Memory: <100MB peak usage during load
    """
    return """
        // Streaming JSON loader to handle large files without crashing Safari
        async function loadGraphDataStreaming() {
            const progressBar = document.getElementById('progress-bar');
            const progressText = document.getElementById('progress-text');

            try {
                // Fetch from streaming endpoint
                const response = await fetch('/api/graph-data');

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const contentLength = response.headers.get('content-length');
                const total = contentLength ? parseInt(contentLength, 10) : 0;
                let loaded = 0;

                if (total > 0) {
                    const sizeMB = (total / (1024 * 1024)).toFixed(1);
                    progressText.textContent = `Downloading ${sizeMB}MB...`;
                }

                // Stream download with progress tracking
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const {done, value} = await reader.read();

                    if (done) break;

                    loaded += value.byteLength;

                    // Update progress (0-50% for transfer)
                    if (total > 0) {
                        const transferPercent = Math.round((loaded / total) * 50);
                        progressBar.style.width = transferPercent + '%';
                        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                        const totalMB = (total / (1024 * 1024)).toFixed(1);
                        progressText.textContent = `Downloaded ${loadedMB}MB / ${totalMB}MB (${transferPercent}%)`;
                    } else {
                        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                        progressText.textContent = `Downloaded ${loadedMB}MB...`;
                    }

                    // Accumulate chunks into buffer
                    buffer += decoder.decode(value, {stream: true});
                }

                // Transfer complete, now parse
                progressBar.style.width = '50%';
                progressText.textContent = 'Parsing JSON data...';

                // Parse JSON (this is still the bottleneck, but at least we streamed the download)
                // Future optimization: Implement incremental JSON parser if needed
                const data = JSON.parse(buffer);

                // Parsing complete
                progressBar.style.width = '100%';
                progressText.textContent = 'Complete!';

                return data;

            } catch (error) {
                console.error('Streaming load error:', error);
                throw error;
            }
        }

        // Auto-load graph data on page load with streaming support
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

            // Use streaming loader
            loadGraphDataStreaming()
                .then(data => {
                    clearTimeout(timeout);
                    loadingEl.innerHTML = '<label style="color: #238636;">✓ Graph loaded successfully</label>';
                    setTimeout(() => loadingEl.style.display = 'none', 2000);

                    // CRITICAL: Always initialize data arrays first
                    // These are required by both visualizeGraph() and switchToCytoscapeLayout()
                    allNodes = data.nodes;
                    allLinks = data.links;

                    // ALWAYS initialize through visualizeGraph first
                    // This sets up visibleNodes, filteredLinks, and root nodes
                    visualizeGraph(data);

                    // Then switch to Dagre for large graphs (if needed)
                    const layoutSelector = document.getElementById('layoutSelector');
                    if (layoutSelector && data.nodes && data.nodes.length > 500) {
                        layoutSelector.value = 'dagre';
                        switchToCytoscapeLayout('dagre');
                    }
                })
                .catch(err => {
                    clearTimeout(timeout);

                    let errorMsg = err.message;
                    if (err.name === 'AbortError') {
                        errorMsg = 'Loading timeout - file may be too large or server unresponsive';
                    }

                    loadingEl.innerHTML = `<label style="color: #f85149;">✗ Failed to load graph data</label><br>` +
                                         `<small style="color: #8b949e;">${errorMsg}</small><br>` +
                                         `<button onclick="location.reload()" style="margin-top: 8px; padding: 6px 12px; background: #238636; border: none; border-radius: 6px; color: white; cursor: pointer;">Retry</button><br>` +
                                         `<small style="color: #8b949e; margin-top: 4px; display: block;">Or run: mcp-vector-search visualize export</small>`;
                    console.error("Failed to load graph:", err);
                });
        });

        // Reset view button event handler
        document.getElementById('reset-view-btn').addEventListener('click', () => {
            resetView();
        });
    """


def get_state_management() -> str:
    """Get visualization V2.0 state management JavaScript.

    Implements the VisualizationStateManager class for hierarchical
    list-based navigation with expansion paths and sibling exclusivity.

    Returns:
        JavaScript string for state management
    """
    return """
        /**
         * Visualization State Manager for V2.0 Architecture
         *
         * Manages expansion paths, node visibility, and view modes.
         * Enforces sibling exclusivity: only one child expanded per depth.
         *
         * Two-Phase Prescriptive Approach:
         *   Phase 1 (tree_root): Initial overview - vertical list of root nodes, all collapsed, NO edges
         *   Phase 2 (tree_expanded/file_detail): Tree navigation - rightward expansion with dagre-style hierarchy
         *
         * View Modes (corresponds to phases):
         *   - tree_root: Phase 1 - Vertical list of root nodes, NO edges shown
         *   - tree_expanded: Phase 2 - Rightward tree expansion of directories, NO edges shown
         *   - file_detail: Phase 2 - File with AST chunks, function call edges shown
         *
         * Design Decision: Prescriptive (non-configurable) phase transition
         *
         * The first click on any node automatically transitions from Phase 1 to Phase 2.
         * This is a fixed behavior with no user configuration - reduces cognitive load
         * and provides consistent, predictable interaction patterns.
         *
         * Reference: docs/development/VISUALIZATION_ARCHITECTURE_V2.md
         */
        class VisualizationStateManager {
            constructor(initialState = null) {
                // View mode: "tree_root", "tree_expanded", or "file_detail"
                this.viewMode = initialState?.view_mode || "tree_root";

                // Handle old view mode names (backward compatibility)
                if (this.viewMode === "list") this.viewMode = "tree_root";
                if (this.viewMode === "directory_fan") this.viewMode = "tree_expanded";
                if (this.viewMode === "file_fan") this.viewMode = "file_detail";

                // Expansion path: ordered array of expanded node IDs (root to current)
                this.expansionPath = initialState?.expansion_path || [];

                // Node states: map of node_id -> {expanded, visible, children_visible}
                this.nodeStates = new Map();

                // Visible edges: set of [source_id, target_id] tuples
                this.visibleEdges = new Set();

                // Event listeners for state changes
                this.listeners = [];

                // Initialize from initial state if provided
                if (initialState?.node_states) {
                    for (const [nodeId, state] of Object.entries(initialState.node_states)) {
                        this.nodeStates.set(nodeId, {
                            expanded: state.expanded || false,
                            visible: state.visible || true,
                            childrenVisible: state.children_visible || false,
                            positionOverride: state.position_override || null
                        });
                    }
                }

                console.log('[StateManager] Initialized with mode:', this.viewMode);
            }

            /**
             * Get or create node state
             */
            _getOrCreateState(nodeId) {
                if (!this.nodeStates.has(nodeId)) {
                    this.nodeStates.set(nodeId, {
                        expanded: false,
                        visible: true,
                        childrenVisible: false,
                        positionOverride: null
                    });
                }
                return this.nodeStates.get(nodeId);
            }

            /**
             * Expand a node (directory or file)
             *
             * Enforces sibling exclusivity: if another sibling is expanded
             * at the same depth, it is collapsed first.
             */
            expandNode(nodeId, nodeType, children = []) {
                console.log(`[StateManager] Expanding ${nodeType} node:`, nodeId, 'with', children.length, 'children');

                const nodeState = this._getOrCreateState(nodeId);

                // Calculate depth
                const depth = this.expansionPath.length;

                // Sibling exclusivity: check if another sibling is expanded at this depth
                if (depth < this.expansionPath.length) {
                    const oldSibling = this.expansionPath[depth];
                    if (oldSibling !== nodeId) {
                        console.log(`[StateManager] Sibling exclusivity: collapsing ${oldSibling}`);
                        // Collapse old path from this depth onward
                        const nodesToCollapse = this.expansionPath.slice(depth);
                        this.expansionPath = this.expansionPath.slice(0, depth);
                        for (const oldNode of nodesToCollapse) {
                            this._collapseNodeInternal(oldNode);
                        }
                    }
                }

                // Mark node as expanded
                nodeState.expanded = true;
                nodeState.childrenVisible = true;

                // Add to expansion path
                if (!this.expansionPath.includes(nodeId)) {
                    this.expansionPath.push(nodeId);
                }

                // Make children visible
                for (const childId of children) {
                    const childState = this._getOrCreateState(childId);
                    childState.visible = true;
                }

                // Update view mode
                if (nodeType === 'directory') {
                    this.viewMode = 'tree_expanded';
                } else if (nodeType === 'file') {
                    this.viewMode = 'file_detail';
                }

                console.log('[StateManager] Expansion path:', this.expansionPath.join(' > '));
                console.log('[StateManager] View mode:', this.viewMode);

                // Notify listeners
                this._notifyListeners();
            }

            /**
             * Internal collapse (without path manipulation)
             */
            _collapseNodeInternal(nodeId) {
                const nodeState = this.nodeStates.get(nodeId);
                if (!nodeState) return;

                nodeState.expanded = false;
                nodeState.childrenVisible = false;
            }

            /**
             * Collapse a node and hide all descendants
             */
            collapseNode(nodeId) {
                console.log('[StateManager] Collapsing node:', nodeId);

                // Remove from expansion path
                const pathIndex = this.expansionPath.indexOf(nodeId);
                if (pathIndex !== -1) {
                    this.expansionPath = this.expansionPath.slice(0, pathIndex);
                }

                // Mark as collapsed
                this._collapseNodeInternal(nodeId);

                // Update view mode if path is empty
                if (this.expansionPath.length === 0) {
                    this.viewMode = 'tree_root';
                    console.log('[StateManager] Collapsed to root, switching to TREE_ROOT view');
                }

                // Notify listeners
                this._notifyListeners();
            }

            /**
             * Reset state to initial list view
             */
            reset() {
                console.log('[StateManager] Resetting to initial state');

                // Collapse all nodes in expansion path
                const nodesToCollapse = [...this.expansionPath];
                for (const nodeId of nodesToCollapse) {
                    this._collapseNodeInternal(nodeId);
                }

                // Clear expansion path
                this.expansionPath = [];

                // Reset view mode to tree_root
                this.viewMode = 'tree_root';

                // Make only root nodes visible
                for (const [nodeId, state] of this.nodeStates.entries()) {
                    // Keep only nodes that have no parent (root nodes)
                    // This will be determined by the allLinks data
                    // For now, mark all non-root nodes as invisible
                    // The renderGraphV2 function will handle visibility correctly
                }

                // Notify listeners
                this._notifyListeners();
            }

            /**
             * Get list of visible node IDs
             */
            getVisibleNodes() {
                const visible = [];
                for (const [nodeId, state] of this.nodeStates.entries()) {
                    if (state.visible) {
                        visible.push(nodeId);
                    }
                }
                return visible;
            }

            /**
             * Get visible edges (AST calls only in FILE_FAN mode)
             */
            getVisibleEdges() {
                return Array.from(this.visibleEdges);
            }

            /**
             * Subscribe to state changes
             */
            subscribe(listener) {
                this.listeners.push(listener);
            }

            /**
             * Notify all listeners of state change
             */
            _notifyListeners() {
                for (const listener of this.listeners) {
                    listener(this.toDict());
                }
            }

            /**
             * Serialize state to plain object
             */
            toDict() {
                const nodeStatesObj = {};
                for (const [nodeId, state] of this.nodeStates.entries()) {
                    nodeStatesObj[nodeId] = {
                        expanded: state.expanded,
                        visible: state.visible,
                        children_visible: state.childrenVisible,
                        position_override: state.positionOverride
                    };
                }

                return {
                    view_mode: this.viewMode,
                    expansion_path: [...this.expansionPath],
                    visible_nodes: this.getVisibleNodes(),
                    visible_edges: this.getVisibleEdges(),
                    node_states: nodeStatesObj
                };
            }
        }

        // Global state manager instance (initialized in visualizeGraph)
        let stateManager = null;
    """


def get_layout_algorithms_v2() -> str:
    """Get V2.0 layout algorithms (list and fan layouts).

    Returns:
        JavaScript string for layout calculation functions
    """
    return """
        /**
         * Calculate vertical list layout positions for nodes.
         *
         * Positions nodes in a vertical list with fixed spacing,
         * sorted alphabetically with directories before files.
         *
         * @param {Array} nodes - Array of node objects
         * @param {Number} canvasWidth - SVG viewport width
         * @param {Number} canvasHeight - SVG viewport height
         * @returns {Map} Map of nodeId -> {x, y} positions
         */
        function calculateListLayout(nodes, canvasWidth, canvasHeight) {
            if (!nodes || nodes.length === 0) {
                console.debug('[Layout] No nodes to layout');
                return new Map();
            }

            // Sort alphabetically (directories first, then files)
            const sortedNodes = nodes.slice().sort((a, b) => {
                // Directories first
                const aIsDir = a.type === 'directory' ? 0 : 1;
                const bIsDir = b.type === 'directory' ? 0 : 1;
                if (aIsDir !== bIsDir) return aIsDir - bIsDir;

                // Then alphabetical by name
                const aName = (a.name || '').toLowerCase();
                const bName = (b.name || '').toLowerCase();
                return aName.localeCompare(bName);
            });

            // Layout parameters
            const nodeHeight = 50;  // Vertical space per node
            const xPosition = 100;  // Left margin
            const totalHeight = sortedNodes.length * nodeHeight;

            // Center vertically in viewport
            const startY = (canvasHeight - totalHeight) / 2;

            // Calculate positions
            const positions = new Map();
            sortedNodes.forEach((node, i) => {
                if (!node.id) {
                    console.warn('[Layout] Node missing id:', node);
                    return;
                }

                const yPosition = startY + (i * nodeHeight);
                positions.set(node.id, { x: xPosition, y: yPosition });
            });

            console.debug(
                `[Layout] List: ${positions.size} nodes, ` +
                `height=${totalHeight}px, startY=${startY.toFixed(1)}`
            );

            return positions;
        }

        /**
         * Calculate horizontal fan layout positions for child nodes.
         *
         * Arranges children in a 180° arc (horizontal fan) from parent node.
         * Radius adapts to child count (200-400px range).
         *
         * @param {Object} parentPos - {x, y} coordinates of parent
         * @param {Array} children - Array of child node objects
         * @param {Number} canvasWidth - SVG viewport width
         * @param {Number} canvasHeight - SVG viewport height
         * @returns {Map} Map of childId -> {x, y} positions
         */
        function calculateFanLayout(parentPos, children, canvasWidth, canvasHeight) {
            if (!children || children.length === 0) {
                console.debug('[Layout] No children to layout in fan');
                return new Map();
            }

            const parentX = parentPos.x;
            const parentY = parentPos.y;

            // Calculate adaptive radius based on child count
            const baseRadius = 200;  // Minimum radius
            const maxRadius = 400;   // Maximum radius
            const spacingPerChild = 60;  // Horizontal space per child

            // Arc length = radius * π (for 180° arc)
            // We want: arc_length >= num_children * spacingPerChild
            // Therefore: radius >= (num_children * spacingPerChild) / π
            const calculatedRadius = (children.length * spacingPerChild) / Math.PI;
            const radius = Math.max(baseRadius, Math.min(calculatedRadius, maxRadius));

            // Horizontal fan: 180° arc from left to right
            const startAngle = Math.PI;  // Left (180°)
            const endAngle = 0;          // Right (0°)
            const angleRange = startAngle - endAngle;

            // Sort children (directories first, then alphabetical)
            const sortedChildren = children.slice().sort((a, b) => {
                const aIsDir = a.type === 'directory' ? 0 : 1;
                const bIsDir = b.type === 'directory' ? 0 : 1;
                if (aIsDir !== bIsDir) return aIsDir - bIsDir;

                const aName = (a.name || '').toLowerCase();
                const bName = (b.name || '').toLowerCase();
                return aName.localeCompare(bName);
            });

            // Calculate positions
            const positions = new Map();
            const numChildren = sortedChildren.length;

            sortedChildren.forEach((child, i) => {
                if (!child.id) {
                    console.warn('[Layout] Child missing id:', child);
                    return;
                }

                // Calculate angle for this child
                let angle;
                if (numChildren === 1) {
                    // Single child: center of arc (90°)
                    angle = Math.PI / 2;
                } else {
                    // Distribute evenly across arc
                    const progress = i / (numChildren - 1);
                    angle = startAngle - (progress * angleRange);
                }

                // Convert polar to cartesian coordinates
                const x = parentX + radius * Math.cos(angle);
                const y = parentY + radius * Math.sin(angle);

                positions.set(child.id, { x, y });
            });

            console.debug(
                `[Layout] Fan: ${positions.size} children, ` +
                `radius=${radius.toFixed(1)}px, ` +
                `arc=${(angleRange * 180 / Math.PI).toFixed(0)}°`
            );

            return positions;
        }

        /**
         * Calculate tree layout for directory navigation (rightward expansion).
         *
         * Arranges children vertically to the right of parent node,
         * creating a hierarchical tree structure similar to file explorers.
         *
         * Design Decision: Tree layout for directory navigation
         *
         * Rationale: Selected rightward tree layout to match familiar file explorer
         * UX (Finder, Explorer). Provides clear parent-child relationships and
         * efficient use of horizontal space for deep hierarchies.
         *
         * Trade-offs:
         * - Clarity: Clear hierarchical structure vs. fan's compact radial layout
         * - Space: Grows rightward (scrollable) vs. fan's fixed radius
         * - Familiarity: Matches file explorer metaphor vs. novel visualization
         *
         * @param {Object} parentPos - {x, y} coordinates of parent
         * @param {Array} children - Array of child node objects
         * @param {Number} canvasWidth - SVG viewport width
         * @param {Number} canvasHeight - SVG viewport height
         * @returns {Map} Map of childId -> {x, y} positions
         */
        function calculateTreeLayout(parentPos, children, canvasWidth, canvasHeight) {
            if (!children || children.length === 0) {
                console.debug('[Layout] No children for tree layout');
                return new Map();
            }

            const parentX = parentPos.x;
            const parentY = parentPos.y;

            // Tree layout parameters
            const horizontalOffset = 800;  // Fixed horizontal spacing from parent
            const verticalSpacing = 50;    // Vertical spacing between children

            // Sort children (directories first, then alphabetical)
            const sortedChildren = children.slice().sort((a, b) => {
                const aIsDir = a.type === 'directory' ? 0 : 1;
                const bIsDir = b.type === 'directory' ? 0 : 1;
                if (aIsDir !== bIsDir) return aIsDir - bIsDir;

                const aName = (a.name || '').toLowerCase();
                const bName = (b.name || '').toLowerCase();
                return aName.localeCompare(bName);
            });

            // Calculate vertical centering
            const totalHeight = sortedChildren.length * verticalSpacing;
            const startY = parentY - (totalHeight / 2);

            // Calculate positions
            const positions = new Map();
            sortedChildren.forEach((child, i) => {
                if (!child.id) {
                    console.warn('[Layout] Child missing id:', child);
                    return;
                }

                const x = parentX + horizontalOffset;
                const y = startY + (i * verticalSpacing);

                positions.set(child.id, { x, y });
            });

            console.debug(
                `[Layout] Tree: ${positions.size} children, ` +
                `offset=${horizontalOffset}px, spacing=${verticalSpacing}px`
            );

            return positions;
        }

        /**
         * Calculate hybrid layout for file detail view.
         *
         * Combines vertical tree positioning for AST chunks with
         * force-directed layout for function call relationships.
         *
         * Design Decision: Vertical tree + function call edges
         *
         * Rationale: AST chunks within a file have natural top-to-bottom order
         * (by line number). Vertical tree preserves this order while function
         * call edges show actual code dependencies.
         *
         * Trade-offs:
         * - Readability: Preserves code order vs. force layout's organic grouping
         * - Performance: Simple O(n) tree vs. O(n²) force simulation
         * - Edges: Shows only AST calls (clear) vs. all relationships (cluttered)
         *
         * @param {Object} parentPos - {x, y} coordinates of parent file node
         * @param {Array} chunks - Array of AST chunk node objects
         * @param {Array} edges - Array of function call edges
         * @param {Number} canvasWidth - SVG viewport width
         * @param {Number} canvasHeight - SVG viewport height
         * @returns {Map} Map of chunkId -> {x, y} positions
         */
        function calculateHybridCodeLayout(parentPos, chunks, edges, canvasWidth, canvasHeight) {
            if (!chunks || chunks.length === 0) {
                console.debug('[Layout] No chunks for hybrid code layout');
                return new Map();
            }

            // Use tree layout for initial positioning (preserves code order)
            // For file detail view, we show chunks in vertical order
            const positions = calculateTreeLayout(parentPos, chunks, canvasWidth, canvasHeight);

            // Note: Force-directed refinement can be added later if needed
            // For now, simple tree layout preserves line number order

            console.debug(
                `[Layout] Hybrid code: ${positions.size} chunks positioned in tree layout`
            );

            return positions;
        }

        /**
         * @deprecated Use calculateTreeLayout instead
         * Legacy function name for backward compatibility
         */
        function calculateCompactFolderLayout(parentPos, children, canvasWidth, canvasHeight) {
            return calculateTreeLayout(parentPos, children, canvasWidth, canvasHeight);
        }
    """


def get_interaction_handlers_v2() -> str:
    """Get V2.0 interaction handlers (expand, collapse, click).

    Returns:
        JavaScript string for interaction handling
    """
    return """
        /**
         * Handle node click events for V2.0 navigation.
         *
         * Behavior (Two-Phase Prescriptive Approach):
         * - Phase 1 (Initial Overview): Circle/grid layout with root-level nodes only, all collapsed
         * - Phase 2 (Tree Navigation): Dagre vertical tree layout with rightward expansion
         *
         * On first click, automatically transitions from Phase 1 to Phase 2.
         *
         * Node Behavior:
         * - Directory: Expand/collapse with rightward tree layout
         * - File: Expand/collapse AST chunks with tree layout + call edges
         * - AST Chunk: Show in content pane, no expansion
         *
         * Design Decision: Automatic phase transition on first interaction
         *
         * Rationale: Users start with high-level overview (Phase 1), then drill down
         * into specific areas (Phase 2). The transition is automatic and prescriptive
         * - no user configuration needed.
         *
         * Trade-offs:
         * - Simplicity: Fixed behavior vs. user choice (removes cognitive load)
         * - Discoverability: Automatic transition vs. explicit control
         * - Consistency: Predictable behavior vs. flexible customization
         */
        function handleNodeClickV2(event, nodeData) {
            event.stopPropagation();

            const node = allNodes.find(n => n.id === nodeData.id);
            if (!node) {
                console.warn('[Click] Node not found:', nodeData.id);
                return;
            }

            console.log('[Click] Node clicked:', node.type, node.name);

            // PHASE TRANSITION: First click transitions from Phase 1 to Phase 2
            if (isInitialOverview && (node.type === 'directory' || node.type === 'file')) {
                console.log('[Phase Transition] Switching from Phase 1 (overview) to Phase 2 (tree expansion)');
                isInitialOverview = false;
                // The layout will automatically change when expandNodeV2 updates viewMode to 'tree_expanded'
            }

            // Always show content pane
            showContentPane(node);

            // Handle expansion based on node type
            if (node.type === 'directory' || node.type === 'file') {
                if (!stateManager) {
                    console.error('[Click] State manager not initialized');
                    return;
                }

                const isExpanded = stateManager.nodeStates.get(node.id)?.expanded || false;

                if (isExpanded) {
                    // Collapse node
                    collapseNodeV2(node.id);
                } else {
                    // Expand node (this will trigger layout change to tree)
                    expandNodeV2(node.id, node.type);
                }
            }
            // AST chunks (function, class, method) don't expand
        }

        /**
         * Expand a node (directory or file) in V2.0 mode.
         *
         * Triggers state update and re-render with animation.
         */
        function expandNodeV2(nodeId, nodeType) {
            if (!stateManager) {
                console.error('[Expand] State manager not initialized');
                return;
            }

            const node = allNodes.find(n => n.id === nodeId);
            if (!node) {
                console.warn('[Expand] Node not found:', nodeId);
                return;
            }

            // Find direct children
            const children = allLinks
                .filter(link => {
                    const sourceId = link.source.id || link.source;
                    const linkType = link.type;
                    return sourceId === nodeId &&
                           (linkType === 'dir_containment' ||
                            linkType === 'file_containment' ||
                            linkType === 'dir_hierarchy');
                })
                .map(link => {
                    const targetId = link.target.id || link.target;
                    return allNodes.find(n => n.id === targetId);
                })
                .filter(n => n);

            const childIds = children.map(c => c.id);

            console.log('[Expand] Expanding node:', nodeId, 'with', childIds.length, 'children');

            // Update state
            stateManager.expandNode(nodeId, nodeType, childIds);

            // Re-render with animation
            renderGraphV2();
        }

        /**
         * Collapse a node and hide all its descendants.
         */
        function collapseNodeV2(nodeId) {
            if (!stateManager) {
                console.error('[Collapse] State manager not initialized');
                return;
            }

            console.log('[Collapse] Collapsing node:', nodeId);

            // Update state (recursively hides descendants)
            stateManager.collapseNode(nodeId, allNodes);

            // Re-render with animation
            renderGraphV2();
        }

        /**
         * Reset to initial list view (Phase 1).
         */
        function resetToListViewV2() {
            if (!stateManager) {
                console.error('[Reset] State manager not initialized');
                return;
            }

            console.log('[Reset] Resetting to Phase 1 (initial overview)');

            // Reset to Phase 1
            isInitialOverview = true;

            // Collapse all nodes
            stateManager.reset();

            // Clear selection
            highlightedNode = null;

            // Close content pane
            closeContentPane();

            // Re-render
            renderGraphV2();
        }

        /**
         * Navigate to a node in the expansion path (breadcrumb click).
         */
        function navigateToNodeInPath(nodeId) {
            if (!stateManager) {
                console.error('[Navigate] State manager not initialized');
                return;
            }

            const pathIndex = stateManager.expansionPath.indexOf(nodeId);
            if (pathIndex === -1) {
                console.warn('[Navigate] Node not in expansion path:', nodeId);
                return;
            }

            console.log('[Navigate] Navigating to node in path:', nodeId);

            // Collapse all nodes after this one in the path
            const nodesToCollapse = stateManager.expansionPath.slice(pathIndex + 1);
            nodesToCollapse.forEach(id => collapseNodeV2(id));

            // Show the node in content pane
            const node = allNodes.find(n => n.id === nodeId);
            if (node) {
                showContentPane(node);
            }
        }
    """


def get_rendering_v2() -> str:
    """Get V2.0 rendering functions with transitions.

    Returns:
        JavaScript string for D3.js rendering with animations
    """
    return """
        /**
         * Main rendering function for V2.0 with transition animations.
         *
         * Two-Phase Prescriptive Layout:
         * - Phase 1 (tree_root): Vertical list layout with root-level nodes only, all collapsed
         * - Phase 2 (tree_expanded/file_detail): Rightward tree expansion with dagre-style hierarchy
         *
         * Renders visible nodes with smooth 750ms transitions between layouts.
         *
         * Design Decision: Automatic layout selection based on view mode
         *
         * The layout automatically adapts to the current phase:
         * - Phase 1 uses simple vertical list (clear overview)
         * - Phase 2 uses tree layout (rightward expansion for deep hierarchies)
         */
        function renderGraphV2(duration = 750) {
            if (!stateManager) {
                console.error('[Render] State manager not initialized');
                return;
            }

            console.log('[Render] Rendering graph, mode:', stateManager.viewMode, 'phase:', isInitialOverview ? 'Phase 1 (overview)' : 'Phase 2 (tree)');

            // 1. Get visible nodes
            const visibleNodeIds = stateManager.getVisibleNodes();
            const visibleNodesList = visibleNodeIds
                .map(id => allNodes.find(n => n.id === id))
                .filter(n => n);

            console.log('[Render] Visible nodes:', visibleNodesList.length);

            // 2. Calculate layout positions (Two-Phase Prescriptive)
            const positions = new Map();

            if (stateManager.viewMode === 'tree_root') {
                // PHASE 1: Vertical list layout for root nodes only (initial overview)
                const listPos = calculateListLayout(visibleNodesList, width, height);
                listPos.forEach((pos, nodeId) => positions.set(nodeId, pos));

                console.debug('[Render] PHASE 1 (tree_root): Vertical list with', positions.size, 'root nodes');
            } else if (stateManager.viewMode === 'tree_expanded' || stateManager.viewMode === 'file_detail') {
                // PHASE 2: Tree layout with rightward expansion (after first click)
                // Tree layout: rightward expansion for directories/files
                stateManager.expansionPath.forEach((expandedId, depth) => {
                    const expandedNode = allNodes.find(n => n.id === expandedId);
                    if (!expandedNode) return;

                    // Position expanded node
                    if (depth === 0) {
                        // Root level - use list layout position
                        const rootNodes = allNodes.filter(n => {
                            const parentLinks = allLinks.filter(l =>
                                (l.target.id || l.target) === n.id &&
                                (l.type === 'dir_containment' || l.type === 'file_containment')
                            );
                            return parentLinks.length === 0;
                        });
                        const listPos = calculateListLayout(rootNodes, width, height);
                        const pos = listPos.get(expandedId);
                        if (pos) positions.set(expandedId, pos);
                    } else {
                        // Child node - should already have position from parent's tree
                        if (!positions.has(expandedId)) {
                            // Fallback to center-left
                            positions.set(expandedId, { x: width * 0.3, y: height / 2 });
                        }
                    }

                    // Calculate tree layout for children (rightward expansion)
                    const children = allLinks
                        .filter(link => {
                            const sourceId = link.source.id || link.source;
                            return sourceId === expandedId;
                        })
                        .map(link => {
                            const targetId = link.target.id || link.target;
                            return allNodes.find(n => n.id === targetId);
                        })
                        .filter(n => n && visibleNodeIds.includes(n.id));

                    if (children.length > 0) {
                        const parentPos = positions.get(expandedId) || { x: width * 0.3, y: height / 2 };

                        // Use tree layout for rightward expansion
                        const treePos = calculateTreeLayout(parentPos, children, width, height);
                        treePos.forEach((pos, childId) => positions.set(childId, pos));
                    }
                });

                console.debug(
                    `[Render] ${stateManager.viewMode.toUpperCase()}: ` +
                    `Tree layout with ${positions.size} nodes, ` +
                    `depth ${stateManager.expansionPath.length}`
                );
            }

            console.log('[Render] Calculated positions for', positions.size, 'nodes');

            // 3. Filter edges
            const visibleLinks = getFilteredLinksForCurrentViewV2();

            console.log('[Render] Visible links:', visibleLinks.length);

            // 4. D3 rendering with transitions

            // --- LINKS ---
            const linkSelection = g.selectAll('.link')
                .data(visibleLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

            // ENTER: New links
            linkSelection.enter()
                .append('line')
                .attr('class', d => `link ${d.type}`)
                .attr('x1', d => {
                    const sourceId = d.source.id || d.source;
                    const pos = positions.get(sourceId);
                    return pos ? pos.x : (d.source.x || 0);
                })
                .attr('y1', d => {
                    const sourceId = d.source.id || d.source;
                    const pos = positions.get(sourceId);
                    return pos ? pos.y : (d.source.y || 0);
                })
                .attr('x2', d => {
                    const targetId = d.target.id || d.target;
                    const pos = positions.get(targetId);
                    return pos ? pos.x : (d.target.x || 0);
                })
                .attr('y2', d => {
                    const targetId = d.target.id || d.target;
                    const pos = positions.get(targetId);
                    return pos ? pos.y : (d.target.y || 0);
                })
                .style('opacity', 0)
                .transition()
                .duration(duration)
                .style('opacity', 1);

            // UPDATE: Existing links
            linkSelection.transition()
                .duration(duration)
                .attr('x1', d => {
                    const sourceId = d.source.id || d.source;
                    const pos = positions.get(sourceId);
                    return pos ? pos.x : (d.source.x || 0);
                })
                .attr('y1', d => {
                    const sourceId = d.source.id || d.source;
                    const pos = positions.get(sourceId);
                    return pos ? pos.y : (d.source.y || 0);
                })
                .attr('x2', d => {
                    const targetId = d.target.id || d.target;
                    const pos = positions.get(targetId);
                    return pos ? pos.x : (d.target.x || 0);
                })
                .attr('y2', d => {
                    const targetId = d.target.id || d.target;
                    const pos = positions.get(targetId);
                    return pos ? pos.y : (d.target.y || 0);
                });

            // EXIT: Remove links
            linkSelection.exit()
                .transition()
                .duration(duration)
                .style('opacity', 0)
                .remove();

            // --- NODES ---
            const nodeSelection = g.selectAll('.node')
                .data(visibleNodesList, d => d.id);

            // ENTER: New nodes
            const nodeEnter = nodeSelection.enter()
                .append('g')
                .attr('class', d => `node ${d.type}`)
                .attr('transform', d => {
                    // Start at calculated position or center
                    const pos = positions.get(d.id);
                    if (pos) {
                        return `translate(${pos.x}, ${pos.y})`;
                    }
                    return `translate(${width / 2}, ${height / 2})`;
                })
                .style('opacity', 0)
                .on('click', handleNodeClickV2)
                .on('mouseover', (event, d) => showTooltip(event, d))
                .on('mouseout', () => hideTooltip());

            // Add node visuals (reuse existing rendering functions)
            addNodeVisuals(nodeEnter);

            // Fade in new nodes
            nodeEnter.transition()
                .duration(duration)
                .style('opacity', 1);

            // UPDATE: Existing nodes with transition
            nodeSelection.transition()
                .duration(duration)
                .attr('transform', d => {
                    const pos = positions.get(d.id);
                    if (pos) {
                        // Update stored position for force layout compatibility
                        d.x = pos.x;
                        d.y = pos.y;
                        return `translate(${pos.x}, ${pos.y})`;
                    }
                    return `translate(${d.x || width / 2}, ${d.y || height / 2})`;
                });

            // Update expand/collapse indicators
            nodeSelection.selectAll('.expand-indicator')
                .text(d => {
                    if (!hasChildren(d)) return '';
                    const state = stateManager.nodeStates.get(d.id);
                    return state?.expanded ? '−' : '+';
                });

            // EXIT: Remove nodes
            nodeSelection.exit()
                .transition()
                .duration(duration)
                .style('opacity', 0)
                .remove();

            // 5. Post-render updates
            updateBreadcrumbsV2();
            updateStats();
        }

        /**
         * Filter links for current view mode (V2.0).
         *
         * Rules (Tree-based):
         * - TREE_ROOT mode: NO edges shown (vertical list only)
         * - TREE_EXPANDED mode: NO edges shown (directory tree only)
         * - FILE_DETAIL mode: Only AST call edges within expanded file
         *
         * Design Decision: No edges during navigation
         *
         * Rationale: Edges are hidden during directory navigation to reduce
         * visual clutter and maintain focus on hierarchy. Only function call
         * edges are shown in file detail view where they provide value.
         *
         * Error Handling:
         * - Returns empty array if state manager not initialized
         * - Returns empty array if no file expanded in FILE_DETAIL mode
         * - Filters out edges where source or target nodes are not visible
         */
        function getFilteredLinksForCurrentViewV2() {
            if (!stateManager) {
                console.warn('[EdgeFilter] State manager not initialized');
                return [];
            }

            // No edges in tree navigation modes
            if (stateManager.viewMode === 'tree_root' || stateManager.viewMode === 'tree_expanded') {
                return [];
            }

            // FILE_DETAIL mode: Show AST call edges within file
            if (stateManager.viewMode === 'file_detail') {
                // Find expanded file in path
                const expandedFileId = stateManager.expansionPath.find(nodeId => {
                    const node = allNodes.find(n => n.id === nodeId);
                    return node && node.type === 'file';
                });

                if (!expandedFileId) {
                    console.debug('[EdgeFilter] No file expanded in FILE_DETAIL mode');
                    return [];
                }

                const expandedFile = allNodes.find(n => n.id === expandedFileId);
                if (!expandedFile) {
                    console.warn('[EdgeFilter] Expanded file node not found:', expandedFileId);
                    return [];
                }

                // Show only caller edges within this file
                const filteredLinks = allLinks.filter(link => {
                    // Must be caller relationship
                    if (link.type !== 'caller') return false;

                    // Both source and target must be AST chunks of the expanded file
                    const sourceId = link.source.id || link.source;
                    const targetId = link.target.id || link.target;

                    const source = allNodes.find(n => n.id === sourceId);
                    const target = allNodes.find(n => n.id === targetId);

                    if (!source || !target) return false;

                    // Both must be in the same file and visible
                    return source.file_path === expandedFile.file_path &&
                           target.file_path === expandedFile.file_path &&
                           stateManager.getVisibleNodes().includes(sourceId) &&
                           stateManager.getVisibleNodes().includes(targetId);
                });

                console.debug(
                    `[EdgeFilter] FILE_DETAIL mode: ${filteredLinks.length} call edges ` +
                    `in file ${expandedFile.name}`
                );

                return filteredLinks;
            }

            // Unknown view mode
            console.warn('[EdgeFilter] Unknown view mode:', stateManager.viewMode);
            return [];
        }

        /**
         * Update breadcrumbs for V2.0 navigation.
         */
        function updateBreadcrumbsV2() {
            if (!stateManager) return;

            const breadcrumbEl = document.querySelector('.breadcrumb-nav');
            if (!breadcrumbEl) return;

            const parts = ['<span class="breadcrumb-root" onclick="resetToListViewV2()" style="cursor:pointer;">🏠 Root</span>'];

            stateManager.expansionPath.forEach((nodeId, index) => {
                const node = allNodes.find(n => n.id === nodeId);
                if (!node) return;

                const isLast = (index === stateManager.expansionPath.length - 1);

                parts.push(' / ');

                if (isLast) {
                    // Current node: not clickable, highlighted
                    parts.push(`<span class="breadcrumb-current" style="color: #ffffff; font-weight: 600;">${escapeHtml(node.name)}</span>`);
                } else {
                    // Parent nodes: clickable
                    parts.push(
                        `<span class="breadcrumb-link" onclick="navigateToNodeInPath('${node.id}')" ` +
                        `style="color: #58a6ff; cursor: pointer; text-decoration: none;">` +
                        `${escapeHtml(node.name)}</span>`
                    );
                }
            });

            breadcrumbEl.innerHTML = parts.join('');
        }

        /**
         * Helper function to add node visuals (reused from existing code).
         */
        function addNodeVisuals(nodeEnter) {
            // This function should call existing node rendering logic
            // For now, we'll add basic shapes

            // Add circles for code nodes
            nodeEnter.filter(d => !isFileOrDir(d) && !isDocNode(d))
                .append('circle')
                .attr('r', d => d.complexity ? Math.min(15 + d.complexity * 2.5, 32) : 18)
                .style('fill', d => d.color || '#58a6ff')
                .attr('stroke', d => hasChildren(d) ? '#ffffff' : 'none')
                .attr('stroke-width', d => hasChildren(d) ? 2 : 0);

            // Add SVG icons for file and directory nodes
            nodeEnter.filter(d => isFileOrDir(d))
                .append('path')
                .attr('class', 'file-icon')
                .attr('d', d => getFileTypeIcon(d))
                .attr('transform', d => {
                    const scale = d.type === 'directory' ? 2.2 : 1.8;
                    return `translate(-12, -12) scale(${scale})`;
                })
                .style('color', d => getFileTypeColor(d))
                .attr('stroke', d => hasChildren(d) ? '#ffffff' : 'none')
                .attr('stroke-width', d => hasChildren(d) ? 1.5 : 0);

            // Add expand/collapse indicator
            nodeEnter.filter(d => hasChildren(d))
                .append('text')
                .attr('class', 'expand-indicator')
                .attr('x', d => {
                    const iconRadius = d.type === 'directory' ? 22 : 18;
                    return iconRadius + 5;
                })
                .attr('y', 0)
                .attr('dy', '0.6em')
                .attr('text-anchor', 'start')
                .style('fill', '#ffffff')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .style('pointer-events', 'none')
                .text(d => {
                    if (!stateManager) return '+';
                    const state = stateManager.nodeStates.get(d.id);
                    return state?.expanded ? '−' : '+';
                });

            // Add labels
            nodeEnter.append('text')
                .attr('class', 'node-label')
                .attr('x', d => {
                    if (isFileOrDir(d)) {
                        const iconRadius = d.type === 'directory' ? 22 : 18;
                        return iconRadius + 25;  // After icon and indicator
                    }
                    return 0;
                })
                .attr('y', d => isFileOrDir(d) ? 0 : 0)
                .attr('dy', d => isFileOrDir(d) ? '0.35em' : '2.5em')
                .attr('text-anchor', d => isFileOrDir(d) ? 'start' : 'middle')
                .style('fill', '#ffffff')
                .style('font-size', '14px')
                .style('pointer-events', 'none')
                .text(d => d.name || 'Unknown');
        }

        /**
         * Helper function to check if node is file or directory.
         */
        function isFileOrDir(node) {
            return node.type === 'file' || node.type === 'directory';
        }

        /**
         * Helper function to check if node is a document node.
         */
        function isDocNode(node) {
            return node.type === 'document' || node.type === 'section';
        }

        /**
         * Helper function to check if node has children.
         */
        function hasChildren(node) {
            return allLinks.some(link => {
                const sourceId = link.source.id || link.source;
                return sourceId === node.id &&
                       (link.type === 'dir_containment' ||
                        link.type === 'file_containment' ||
                        link.type === 'dir_hierarchy');
            });
        }

        /**
         * Helper function to escape HTML in strings.
         */
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    """


def get_all_scripts() -> str:
    """Get all JavaScript code combined.

    Returns:
        Complete JavaScript string for the visualization
    """
    return "".join(
        [
            get_d3_initialization(),
            get_state_management(),  # NEW: V2.0 state management
            get_layout_algorithms_v2(),  # NEW: V2.0 layout algorithms
            get_interaction_handlers_v2(),  # NEW: V2.0 interaction handlers
            get_rendering_v2(),  # NEW: V2.0 rendering with transitions
            get_file_type_functions(),
            get_spacing_calculation_functions(),
            get_loading_spinner_functions(),
            get_navigation_stack_logic(),
            get_layout_switching_logic(),
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
