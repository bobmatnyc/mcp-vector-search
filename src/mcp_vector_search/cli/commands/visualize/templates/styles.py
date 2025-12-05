"""CSS styles for the visualization interface.

This module contains all CSS styling for the D3.js code graph visualization,
organized into logical sections for maintainability.
"""


def get_base_styles() -> str:
    """Get base styles for body and core layout.

    Returns:
        CSS string for base styling
    """
    return """
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }

        h1 { margin: 0 0 16px 0; font-size: 18px; }
        h3 { margin: 16px 0 8px 0; font-size: 14px; color: #8b949e; }
    """


def get_controls_styles() -> str:
    """Get styles for the control panel.

    Returns:
        CSS string for control panel styling
    """
    return """
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            min-width: 250px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .control-group {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 4px;
            font-size: 12px;
            color: #8b949e;
        }

        input[type="file"] {
            width: 100%;
            padding: 6px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 12px;
        }

        .legend {
            position: absolute;
            top: 70px;
            right: 20px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            max-width: 300px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .legend-category {
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #21262d;
        }

        .legend-category:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }

        .legend-title {
            font-weight: 600;
            color: #c9d1d9;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            padding-left: 8px;
        }

        .legend-item:last-child {
            margin-bottom: 0;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .stats {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #30363d;
            font-size: 12px;
            color: #8b949e;
        }
    """


def get_graph_styles() -> str:
    """Get styles for the graph SVG element.

    Returns:
        CSS string for graph styling
    """
    return """
        #graph {
            width: 100vw;
            height: 100vh;
        }
    """


def get_node_styles() -> str:
    """Get styles for graph nodes.

    Returns:
        CSS string for node styling including different node types
    """
    return """
        .node circle {
            cursor: pointer;
            stroke: #c9d1d9;
            stroke-width: 2px;
            pointer-events: all;
        }

        .node.module circle { fill: #238636; }
        .node.class circle { fill: #1f6feb; }
        .node.function circle { fill: #d29922; }
        .node.method circle { fill: #8957e5; }
        .node.code circle { fill: #6e7681; }
        .node.file circle {
            fill: none;
            stroke: #58a6ff;
            stroke-width: 2px;
            stroke-dasharray: 5,3;
            opacity: 0.6;
        }
        .node.directory circle {
            fill: none;
            stroke: #79c0ff;
            stroke-width: 2px;
            stroke-dasharray: 3,3;
            opacity: 0.5;
        }
        .node.subproject circle { fill: #da3633; stroke-width: 3px; }

        /* Non-code document nodes - squares */
        .node.docstring rect { fill: #8b949e; }
        .node.comment rect { fill: #6e7681; }
        .node rect {
            cursor: pointer;
            stroke: #c9d1d9;
            stroke-width: 2px;
            pointer-events: all;
        }

        /* File type icon styling */
        .node path.file-icon {
            fill: currentColor;
            stroke: none;
            pointer-events: all;
            cursor: pointer;
        }

        .node text {
            font-size: 14px;
            fill: #c9d1d9;
            text-anchor: middle;
            pointer-events: none;
            user-select: none;
        }

        .node.highlighted circle,
        .node.highlighted rect {
            stroke: #f0e68c;
            stroke-width: 3px;
            filter: drop-shadow(0 0 8px #f0e68c);
        }
    """


def get_link_styles() -> str:
    """Get styles for graph links (edges).

    Returns:
        CSS string for link styling including semantic similarity and cycles
    """
    return """
        .link {
            stroke: #30363d;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }

        .link.dependency {
            stroke: #d29922;
            stroke-opacity: 0.8;
            stroke-width: 2px;
            stroke-dasharray: 5,5;
        }

        /* Semantic relationship links - colored by similarity */
        .link.semantic {
            stroke-opacity: 0.7;
            stroke-dasharray: 4,4;
        }

        .link.semantic.sim-high { stroke: #00ff00; stroke-width: 4px; }
        .link.semantic.sim-medium-high { stroke: #88ff00; stroke-width: 3px; }
        .link.semantic.sim-medium { stroke: #ffff00; stroke-width: 2.5px; }
        .link.semantic.sim-low { stroke: #ffaa00; stroke-width: 2px; }
        .link.semantic.sim-very-low { stroke: #ff0000; stroke-width: 1.5px; }

        /* Circular dependency links - highest visual priority */
        .link.cycle {
            stroke: #ff4444 !important;
            stroke-width: 3px !important;
            stroke-dasharray: 8, 4;
            stroke-opacity: 0.8;
            animation: pulse-cycle 2s infinite;
        }

        @keyframes pulse-cycle {
            0%, 100% { stroke-opacity: 0.8; }
            50% { stroke-opacity: 1.0; }
        }
    """


def get_tooltip_styles() -> str:
    """Get styles for tooltips.

    Returns:
        CSS string for tooltip styling
    """
    return """
        .tooltip {
            position: absolute;
            padding: 12px;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid #30363d;
            border-radius: 6px;
            pointer-events: none;
            display: none;
            font-size: 12px;
            max-width: 300px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .caller-link {
            color: #58a6ff;
            text-decoration: none;
            cursor: pointer;
            transition: color 0.2s;
        }

        .caller-link:hover {
            color: #79c0ff;
            text-decoration: underline;
        }
    """


def get_content_pane_styles() -> str:
    """Get styles for the content pane (code viewer).

    Returns:
        CSS string for content pane styling
    """
    return """
        #content-pane {
            position: fixed;
            top: 0;
            right: 0;
            width: 600px;
            height: 100vh;
            background: rgba(13, 17, 23, 0.98);
            border-left: 1px solid #30363d;
            overflow-y: auto;
            box-shadow: -4px 0 24px rgba(0, 0, 0, 0.5);
            transform: translateX(100%);
            transition: transform 0.3s ease-in-out;
            z-index: 1000;
        }

        #content-pane.visible {
            transform: translateX(0);
        }

        #content-pane .pane-header {
            position: sticky;
            top: 0;
            background: rgba(13, 17, 23, 0.98);
            padding: 20px;
            border-bottom: 1px solid #30363d;
            z-index: 1;
        }

        #content-pane .pane-title {
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
            margin-bottom: 8px;
            padding-right: 30px;
        }

        #content-pane .pane-meta {
            font-size: 12px;
            color: #8b949e;
        }

        #content-pane .pane-footer {
            position: sticky;
            bottom: 0;
            background: rgba(13, 17, 23, 0.98);
            padding: 16px 20px;
            border-top: 1px solid #30363d;
            font-size: 11px;
            color: #8b949e;
            z-index: 1;
        }

        #content-pane .pane-footer .footer-item {
            display: block;
            margin-bottom: 8px;
        }

        #content-pane .pane-footer .footer-label {
            color: #c9d1d9;
            font-weight: 600;
            margin-right: 4px;
        }

        #content-pane .pane-footer .footer-value {
            color: #8b949e;
        }

        #content-pane .collapse-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            cursor: pointer;
            color: #8b949e;
            font-size: 24px;
            line-height: 1;
            background: none;
            border: none;
            padding: 0;
            transition: color 0.2s;
        }

        #content-pane .collapse-btn:hover {
            color: #c9d1d9;
        }

        #content-pane .pane-content {
            padding: 20px;
        }

        #content-pane pre {
            margin: 0;
            padding: 16px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.6;
        }

        #content-pane code {
            color: #c9d1d9;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        /* Code links for jumping to function/class definitions */
        #content-pane .code-link {
            color: #58a6ff;
            cursor: pointer;
            text-decoration: underline;
            text-decoration-style: dotted;
        }

        #content-pane .code-link:hover {
            color: #79c0ff;
            text-decoration-style: solid;
        }

        /* Bold primitives styling */
        #content-pane pre code strong {
            color: #ff7b72;
            font-weight: 600;
        }

        #content-pane .directory-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        #content-pane .directory-list li {
            padding: 8px 12px;
            margin: 4px 0;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            font-size: 12px;
            display: flex;
            align-items: center;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        #content-pane .directory-list li:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        #content-pane .directory-list .item-icon {
            margin-right: 8px;
            font-size: 14px;
        }

        #content-pane .directory-list .item-type {
            margin-left: auto;
            padding-left: 12px;
            font-size: 10px;
            color: #8b949e;
        }

        #content-pane .import-details {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
        }

        #content-pane .import-details .import-statement {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            color: #79c0ff;
            margin-bottom: 12px;
        }

        #content-pane .import-details .detail-row {
            font-size: 11px;
            color: #8b949e;
            margin: 4px 0;
        }

        #content-pane .import-details .detail-label {
            color: #c9d1d9;
            font-weight: 600;
        }
    """


def get_reset_button_styles() -> str:
    """Get styles for the reset view button.

    Returns:
        CSS string for reset button styling
    """
    return """
        #reset-view-btn {
            position: fixed;
            top: 20px;
            right: 460px;
            padding: 8px 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 100;
            transition: all 0.2s;
        }

        #reset-view-btn:hover {
            background: #30363d;
            border-color: #58a6ff;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
    """


def get_spinner_styles() -> str:
    """Get styles for the loading spinner animation.

    Returns:
        CSS string for spinner styling and animation
    """
    return """
        /* Loading spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
    """


def get_all_styles() -> str:
    """Get all CSS styles combined.

    Returns:
        Complete CSS string for the visualization
    """
    return "".join(
        [
            get_base_styles(),
            get_controls_styles(),
            get_graph_styles(),
            get_node_styles(),
            get_link_styles(),
            get_tooltip_styles(),
            get_content_pane_styles(),
            get_reset_button_styles(),
            get_spinner_styles(),
        ]
    )
