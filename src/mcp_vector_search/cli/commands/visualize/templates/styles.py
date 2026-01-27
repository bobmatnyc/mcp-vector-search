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
        /* CSS Variables for Theme Support */
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-tertiary: #6e7681;
            --border-primary: #30363d;
            --border-secondary: #21262d;
            --accent: #58a6ff;
            --accent-hover: #79c0ff;
            --success: #238636;
            --warning: #d29922;
            --error: #da3633;
            --shadow: rgba(0, 0, 0, 0.4);
        }

        [data-theme="light"] {
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #eaeef2;
            --text-primary: #24292f;
            --text-secondary: #57606a;
            --text-tertiary: #6e7781;
            --border-primary: #d0d7de;
            --border-secondary: #d8dee4;
            --accent: #0969da;
            --accent-hover: #0550ae;
            --success: #1a7f37;
            --warning: #9a6700;
            --error: #cf222e;
            --shadow: rgba(31, 35, 40, 0.15);
        }

        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        h1 { margin: 0 0 4px 0; font-size: 18px; color: var(--text-primary); }
        h3 { margin: 16px 0 8px 0; font-size: 14px; color: var(--text-secondary); }

        .version-badge {
            font-size: 10px;
            color: var(--text-tertiary);
            margin-bottom: 16px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            padding: 2px 6px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            display: inline-block;
        }
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
            background: var(--bg-primary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 16px;
            min-width: 250px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 8px 24px var(--shadow);
            z-index: 500;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .control-group {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 4px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        input[type="file"] {
            width: 100%;
            padding: 6px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 12px;
        }

        .legend {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            max-width: 300px;
            margin-top: 16px;
            box-shadow: 0 8px 24px var(--shadow);
        }

        .legend-category {
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-secondary);
        }

        .legend-category:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }

        .legend-title {
            font-weight: 600;
            color: var(--text-primary);
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

        /* Report buttons in sidebar */
        .report-btn {
            cursor: pointer;
            padding: 10px 12px !important;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            margin-bottom: 8px !important;
            transition: all 0.2s ease;
        }

        .report-btn:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
            transform: translateX(4px);
        }

        .report-btn:hover span {
            color: white !important;
        }

        .report-icon {
            margin-right: 10px;
            font-size: 16px;
        }

        .stats {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-primary);
            font-size: 12px;
            color: var(--text-secondary);
        }

        .toggle-switch-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
        }

        .toggle-label {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
            transition: color 0.2s ease;
        }

        .toggle-label.active {
            color: var(--accent);
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 48px;
            height: 24px;
            margin: 0;
        }

        .toggle-switch input {
            opacity: 0;
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            cursor: pointer;
            z-index: 2;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: var(--border-primary);
            transition: 0.3s;
            border-radius: 24px;
            border: 1px solid var(--border-primary);
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: var(--text-secondary);
            transition: 0.3s;
            border-radius: 50%;
        }

        .toggle-switch input:checked + .toggle-slider {
            background-color: var(--success);
            border-color: var(--success);
        }

        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
            background-color: var(--bg-primary);
        }

        .toggle-slider:hover {
            opacity: 0.8;
        }

        .toggle-switch input:checked + .toggle-slider:hover {
            opacity: 0.9;
        }

        /* Filter buttons */
        .filter-buttons {
            display: flex;
            gap: 4px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 4px;
        }

        .filter-btn {
            flex: 1;
            padding: 8px 12px;
            background: transparent;
            border: none;
            border-radius: 4px;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .filter-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .filter-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .filter-btn.active:hover {
            background: var(--accent-hover);
        }
    """


def get_graph_styles() -> str:
    """Get styles for the graph SVG element.

    Returns:
        CSS string for graph styling
    """
    return """
        #main-container {
            position: fixed;
            left: 0;
            top: 0;
            right: 0;
            bottom: 0;
            transition: right 0.3s ease-in-out;
        }

        #main-container.viewer-open {
            right: 450px;
        }

        /* When viewer is expanded, handled via JS style.right */

        #graph {
            width: 100%;
            height: 100%;
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
        .node.docstring rect:not(.hit-area) { fill: #8b949e; }
        .node.comment rect:not(.hit-area) { fill: #6e7681; }
        .node rect:not(.hit-area) {
            cursor: pointer;
            stroke: #c9d1d9;
            stroke-width: 2px;
            pointer-events: all;
        }

        /* Hit area for file/directory nodes - transparent clickable rectangle */
        .node rect.hit-area {
            fill: transparent;
            stroke: none;
            pointer-events: all;
            cursor: pointer;
        }

        /* Debug mode: uncomment to visualize hit areas */
        /* .node rect.hit-area { fill: rgba(255, 0, 0, 0.1); stroke: red; stroke-width: 1; } */

        /* File type icon styling */
        .node path.file-icon {
            fill: currentColor;
            stroke: none;
            pointer-events: none;
            cursor: pointer;
        }

        .node text {
            font-size: 14px;
            fill: #c9d1d9;
            /* text-anchor set by JS based on layout */
            pointer-events: none;
            user-select: none;
        }

        .node.highlighted circle,
        .node.highlighted rect {
            stroke: #f0e68c;
            stroke-width: 3px;
            filter: drop-shadow(0 0 8px #f0e68c);
        }

        /* Node loading spinner */
        .node-loading {
            stroke: #2196F3;
            stroke-width: 3;
            fill: none;
            animation: spin 1s linear infinite;
        }

        .node-loading-overlay {
            fill: rgba(255, 255, 255, 0.8);
            pointer-events: none;
        }
    """


def get_link_styles() -> str:
    """Get styles for graph links (edges).

    Returns:
        CSS string for link styling including semantic similarity and cycles
    """
    return """
        .link {
            fill: none;
            stroke: #c9d1d9;
            stroke-opacity: 0.8;
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
            background: var(--bg-primary);
            opacity: 0.95;
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            pointer-events: none;
            display: none;
            font-size: 12px;
            max-width: 300px;
            box-shadow: 0 8px 24px var(--shadow);
        }

        .caller-link {
            color: var(--accent);
            text-decoration: none;
            cursor: pointer;
            transition: color 0.2s;
        }

        .caller-link:hover {
            color: var(--accent-hover);
            text-decoration: underline;
        }
    """


def get_breadcrumb_styles() -> str:
    """Get styles for breadcrumb navigation.

    Returns:
        CSS string for breadcrumb styling
    """
    return """
        /* Breadcrumb navigation */
        .breadcrumb-nav {
            margin: 0 0 10px 0;
            padding: 8px 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.6;
            overflow-x: auto;
            white-space: nowrap;
        }

        .breadcrumb-root {
            color: var(--accent);
            cursor: pointer;
            font-weight: 500;
            transition: color 0.2s;
        }

        .breadcrumb-root:hover {
            color: var(--accent-hover);
            text-decoration: underline;
        }

        .breadcrumb-link {
            color: var(--accent);
            cursor: pointer;
            transition: color 0.2s;
        }

        .breadcrumb-link:hover {
            color: var(--accent-hover);
            text-decoration: underline;
        }

        .breadcrumb-separator {
            color: var(--text-tertiary);
            margin: 0 6px;
        }

        .breadcrumb-current {
            color: var(--text-primary);
            font-weight: 600;
        }
    """


def get_content_pane_styles() -> str:
    """Get styles for the viewer panel (code/file/directory viewer).

    Returns:
        CSS string for viewer panel styling
    """
    return """
        .viewer-panel {
            position: fixed;
            top: 0;
            right: 0;
            width: 450px;
            height: 100vh;
            background: var(--bg-primary);
            opacity: 0.98;
            border-left: 1px solid var(--border-primary);
            overflow-y: auto;
            box-shadow: -4px 0 24px var(--shadow);
            transform: translateX(100%);
            transition: transform 0.3s ease-in-out;
            z-index: 1000;
        }

        .viewer-panel.open {
            transform: translateX(0);
        }

        .viewer-panel.expanded {
            width: 70vw;
            max-width: 1200px;
        }

        .viewer-header {
            position: sticky;
            top: 0;
            background: var(--bg-primary);
            opacity: 0.98;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-primary);
            z-index: 1;
        }

        .viewer-header-buttons {
            position: absolute;
            top: 12px;
            right: 12px;
            display: flex;
            gap: 8px;
        }

        .viewer-expand-btn {
            cursor: pointer;
            color: var(--text-primary);
            font-size: 16px;
            line-height: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            padding: 6px 8px;
            transition: color 0.2s, background 0.2s, border-color 0.2s;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .viewer-expand-btn:hover {
            color: var(--accent);
            background: var(--border-primary);
            border-color: var(--accent);
        }

        .viewer-title {
            font-size: 16px;
            font-weight: bold;
            color: var(--accent);
            margin: 0 0 8px 0;
            padding-right: 80px;
        }

        .section-nav {
            margin-top: 8px;
        }

        .section-nav select {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
            cursor: pointer;
            width: 100%;
            max-width: 250px;
        }

        .section-nav select:hover {
            border-color: var(--accent);
        }

        .section-nav select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
        }

        .viewer-close-btn {
            cursor: pointer;
            color: var(--text-primary);
            font-size: 18px;
            line-height: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            padding: 6px 10px;
            transition: color 0.2s, background 0.2s, border-color 0.2s;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .viewer-close-btn:hover {
            color: var(--error);
            background: var(--border-primary);
            border-color: var(--error);
        }

        .viewer-content {
            padding: 20px;
        }

        .viewer-section {
            margin-bottom: 24px;
        }

        .viewer-section-title {
            font-size: 13px;
            font-weight: 600;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .viewer-info-grid {
            display: grid;
            gap: 8px;
        }

        .viewer-info-row {
            display: flex;
            font-size: 13px;
        }

        .viewer-info-label {
            color: #8b949e;
            min-width: 100px;
            font-weight: 500;
        }

        .viewer-info-value {
            color: #c9d1d9;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        .viewer-info-value.clickable {
            color: #58a6ff;
            cursor: pointer;
            text-decoration: underline;
            text-decoration-style: dotted;
            text-underline-offset: 2px;
        }

        .viewer-info-value.clickable:hover {
            color: #79c0ff;
            text-decoration-style: solid;
        }

        .viewer-content pre {
            margin: 0;
            padding: 16px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.6;
        }

        .viewer-content code {
            color: #c9d1d9;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        .chunk-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .chunk-list-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chunk-list-item:hover {
            background: #21262d;
            border-color: #58a6ff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        /* Relationship tags for callers/callees */
        .relationship-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .relationship-tag.caller {
            background: rgba(88, 166, 255, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(88, 166, 255, 0.3);
        }

        .relationship-tag.caller:hover {
            background: rgba(88, 166, 255, 0.3);
            border-color: #58a6ff;
        }

        .relationship-tag.callee {
            background: rgba(240, 136, 62, 0.15);
            color: #f0883e;
            border: 1px solid rgba(240, 136, 62, 0.3);
        }

        .relationship-tag.callee:hover {
            background: rgba(240, 136, 62, 0.3);
            border-color: #f0883e;
        }

        /* Semantic similarity items */
        .semantic-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .semantic-item:hover {
            background: #21262d;
            border-color: #a371f7;
        }

        .semantic-score {
            font-size: 11px;
            font-weight: 600;
            color: #a371f7;
            background: rgba(163, 113, 247, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
            min-width: 36px;
            text-align: center;
        }

        .semantic-name {
            flex: 1;
            color: #c9d1d9;
            font-size: 12px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .semantic-type {
            font-size: 10px;
            color: #8b949e;
            text-transform: uppercase;
        }

        /* External Calls/Callers Styles */
        .external-call-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .external-call-item:hover {
            background: #21262d;
            border-color: #58a6ff;
            transform: translateX(4px);
        }

        .external-call-icon {
            font-size: 14px;
            font-weight: bold;
            color: #58a6ff;
            width: 20px;
            text-align: center;
        }

        .external-call-name {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 12px;
            font-weight: 600;
            color: #58a6ff;
            flex-shrink: 0;
        }

        .external-call-path {
            font-size: 11px;
            color: #8b949e;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            text-align: right;
        }

        .chunk-icon {
            font-size: 16px;
            flex-shrink: 0;
        }

        .chunk-info {
            flex: 1;
            min-width: 0;
        }

        .chunk-name {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 13px;
            color: #c9d1d9;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .chunk-meta {
            font-size: 11px;
            color: #8b949e;
            margin-top: 2px;
        }

        .dir-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .dir-list-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .dir-list-item:hover {
            background: #21262d;
            border-color: #58a6ff;
        }

        .dir-icon {
            font-size: 16px;
            flex-shrink: 0;
        }

        .dir-name {
            flex: 1;
            font-size: 13px;
            color: #c9d1d9;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .dir-type {
            font-size: 11px;
            color: #8b949e;
            text-transform: uppercase;
        }

        .dir-arrow {
            color: #58a6ff;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.2s ease;
        }

        .dir-list-item:hover .dir-arrow {
            opacity: 1;
        }

        .dir-list-item.clickable {
            cursor: pointer;
        }

        /* Navigation bar styles */
        .navigation-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
            margin-bottom: 12px;
            border-bottom: 1px solid #30363d;
        }

        .nav-btn {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 4px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .nav-btn:hover:not(.disabled) {
            background: #30363d;
            border-color: #58a6ff;
        }

        .nav-btn.disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .breadcrumb-trail {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 4px;
            flex: 1;
            min-width: 0;
            overflow: hidden;
        }

        .breadcrumb-separator {
            color: #484f58;
            font-size: 12px;
        }

        .breadcrumb-item {
            font-size: 12px;
            color: #8b949e;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .breadcrumb-item.clickable {
            color: #58a6ff;
            cursor: pointer;
        }

        .breadcrumb-item.clickable:hover {
            text-decoration: underline;
        }

        .breadcrumb-item.current {
            color: #c9d1d9;
            font-weight: 500;
        }

        /* Node highlight animation - temporary focus */
        .node-highlight circle {
            stroke: #58a6ff !important;
            stroke-width: 4px !important;
            filter: drop-shadow(0 0 8px rgba(88, 166, 255, 0.8));
        }

        .node-highlight text {
            fill: #58a6ff !important;
            font-weight: bold !important;
        }

        /* Selected node - persistent highlight for node shown in viewer */
        .node-selected circle {
            stroke: #f0883e !important;
            stroke-width: 5px !important;
            filter: drop-shadow(0 0 12px rgba(240, 136, 62, 0.9));
            animation: selected-pulse 2s ease-in-out infinite;
        }

        .node-selected text {
            fill: #f0883e !important;
            font-weight: bold !important;
        }

        @keyframes selected-pulse {
            0%, 100% { filter: drop-shadow(0 0 12px rgba(240, 136, 62, 0.9)); }
            50% { filter: drop-shadow(0 0 20px rgba(240, 136, 62, 1.0)); }
        }
    """


def get_code_chunks_styles() -> str:
    """Get styles for code chunks section in file viewer.

    Returns:
        CSS string for code chunks styling
    """
    return """
        /* Code chunks section */
        .code-chunks-section {
            margin: 0 0 20px 0;
            padding: 15px;
            background: #161b22;
            border-radius: 6px;
            border: 1px solid #30363d;
        }

        .section-header {
            margin: 0 0 12px 0;
            font-size: 13px;
            font-weight: 600;
            color: #c9d1d9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .code-chunks-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .code-chunk-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .code-chunk-item:hover {
            background: #21262d;
            border-color: #58a6ff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .chunk-icon {
            font-size: 16px;
            flex-shrink: 0;
        }

        .chunk-name {
            flex: 1;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 13px;
            color: #c9d1d9;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .line-range {
            font-size: 11px;
            color: #8b949e;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            background: #161b22;
            padding: 2px 6px;
            border-radius: 3px;
            flex-shrink: 0;
        }

        .chunk-type {
            font-size: 11px;
            color: #ffffff;
            background: #6e7681;
            padding: 2px 8px;
            border-radius: 12px;
            text-transform: lowercase;
            flex-shrink: 0;
        }

        /* Type-specific colors for chunk badges */
        .code-chunk-item[data-type="function"] .chunk-type {
            background: #d29922;
        }

        .code-chunk-item[data-type="class"] .chunk-type {
            background: #1f6feb;
        }

        .code-chunk-item[data-type="method"] .chunk-type {
            background: #8957e5;
        }

        .code-chunk-item[data-type="code"] .chunk-type {
            background: #6e7681;
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


def get_theme_toggle_styles() -> str:
    """Get styles for the theme toggle button.

    Returns:
        CSS string for theme toggle styling
    """
    return """
        .theme-toggle-icon-btn {
            width: 36px;
            height: 36px;
            padding: 0;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }

        .theme-toggle-icon-btn:hover {
            background: var(--accent);
            border-color: var(--accent-hover);
            transform: scale(1.05);
        }

        .theme-toggle-icon-btn .theme-icon {
            font-size: 18px;
            line-height: 1;
        }

        /* Complexity grade colors for nodes */
        .grade-A { fill: #238636 !important; stroke: #2ea043; }
        .grade-B { fill: #1f6feb !important; stroke: #388bfd; }
        .grade-C { fill: #d29922 !important; stroke: #e0ac3a; }
        .grade-D { fill: #f0883e !important; stroke: #f59f5f; }
        .grade-F { fill: #da3633 !important; stroke: #f85149; }

        /* Code smell indicator - red border */
        .has-smells circle {
            stroke: var(--error) !important;
            stroke-width: 3px !important;
            stroke-dasharray: 5, 3;
        }

        /* Circular dependency indicator */
        .in-cycle circle {
            stroke: #ff4444 !important;
            stroke-width: 3px !important;
            animation: pulse-border 1.5s infinite;
        }

        @keyframes pulse-border {
            0%, 100% { stroke-opacity: 0.8; }
            50% { stroke-opacity: 1.0; }
        }
    """


def get_search_styles() -> str:
    """Get styles for the search box and results dropdown.

    Returns:
        CSS string for search styling
    """
    return """
        /* Search box styles */
        .search-container {
            position: relative;
            margin-bottom: 16px;
            z-index: 100;
        }

        #search-input {
            width: 100%;
            padding: 10px 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
            position: relative;
            z-index: 101;
            cursor: text;
        }

        #search-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
        }

        #search-input::placeholder {
            color: var(--text-secondary);
        }

        /* Search results dropdown */
        .search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 300px;
            overflow-y: auto;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-top: none;
            border-radius: 0 0 6px 6px;
            z-index: 1000;
            display: none;
            box-shadow: 0 4px 12px var(--shadow);
        }

        .search-results.visible {
            display: block;
        }

        .search-result-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            cursor: pointer;
            border-bottom: 1px solid var(--border-secondary);
            transition: background 0.15s;
        }

        .search-result-item:last-child {
            border-bottom: none;
        }

        .search-result-item:hover,
        .search-result-item.selected {
            background: var(--bg-tertiary);
        }

        .search-result-icon {
            font-size: 14px;
            flex-shrink: 0;
            width: 20px;
            text-align: center;
        }

        .search-result-info {
            flex: 1;
            min-width: 0;
            overflow: hidden;
        }

        .search-result-name {
            font-size: 13px;
            color: var(--text-primary);
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .search-result-name mark {
            background: rgba(88, 166, 255, 0.3);
            color: var(--accent);
            border-radius: 2px;
            padding: 0 2px;
        }

        .search-result-path {
            font-size: 11px;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin-top: 2px;
        }

        .search-result-type {
            font-size: 10px;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 10px;
            text-transform: uppercase;
            flex-shrink: 0;
        }

        .search-no-results {
            padding: 20px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
        }

        .search-hint {
            padding: 8px 12px;
            font-size: 11px;
            color: var(--text-tertiary);
            background: var(--bg-primary);
            border-top: 1px solid var(--border-secondary);
        }
    """


def get_complexity_report_styles() -> str:
    """Get styles for the complexity report.

    Returns:
        CSS string for complexity report styling
    """
    return """
        /* Complexity Report Styles */
        .complexity-report {
            padding: 0;
        }

        /* Summary Section */
        .complexity-summary {
            margin-bottom: 24px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .summary-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 16px;
            text-align: center;
        }

        .summary-label {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .summary-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent);
        }

        /* Grade Distribution */
        .grade-distribution {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 16px;
        }

        .distribution-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .distribution-bars {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .distribution-row {
            display: grid;
            grid-template-columns: 40px 1fr 100px;
            gap: 12px;
            align-items: center;
        }

        .distribution-grade {
            font-size: 14px;
            font-weight: bold;
            text-align: center;
        }

        .distribution-bar-container {
            background: var(--bg-tertiary);
            border-radius: 4px;
            height: 24px;
            overflow: hidden;
            border: 1px solid var(--border-primary);
        }

        .distribution-bar {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
            opacity: 0.8;
        }

        .distribution-count {
            font-size: 12px;
            color: var(--text-secondary);
            text-align: right;
        }

        /* Hotspots Section */
        .complexity-hotspots {
            margin-top: 24px;
        }

        .section-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .hotspots-table-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            overflow: hidden;
        }

        .hotspots-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .hotspots-table thead {
            background: var(--bg-tertiary);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .hotspots-table th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-primary);
        }

        .hotspots-table tbody tr {
            border-bottom: 1px solid var(--border-secondary);
        }

        .hotspots-table tbody tr:last-child {
            border-bottom: none;
        }

        .hotspot-row {
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .hotspot-row:hover {
            background: var(--bg-tertiary);
        }

        .hotspots-table td {
            padding: 12px;
            color: var(--text-primary);
        }

        .hotspot-name {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-weight: 500;
            color: var(--accent);
        }

        .hotspot-file {
            color: var(--text-secondary);
            font-size: 12px;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .hotspot-lines {
            text-align: center;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 12px;
        }

        .hotspot-complexity {
            text-align: center;
            font-weight: bold;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        .hotspot-grade {
            text-align: center;
        }

        .grade-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 11px;
            color: white;
            min-width: 28px;
            text-align: center;
        }
    """


def get_code_smells_styles() -> str:
    """Get styles for the code smells report.

    Returns:
        CSS string for code smells styling
    """
    return """
        /* Code Smells Report Styles */
        .code-smells-report {
            padding: 0;
        }

        /* Smell Type Filters */
        .smell-filters {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .filter-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .filter-checkboxes {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
        }

        .filter-checkbox-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .filter-checkbox-item:hover {
            background: var(--bg-primary);
            border-color: var(--accent);
        }

        .filter-checkbox-item input[type="checkbox"] {
            cursor: pointer;
            width: 16px;
            height: 16px;
        }

        .filter-checkbox-label {
            flex: 1;
            font-size: 12px;
            color: var(--text-primary);
            cursor: pointer;
        }

        .filter-checkbox-count {
            font-size: 11px;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            padding: 2px 8px;
            border-radius: 10px;
        }

        /* Smell Summary Cards */
        .smell-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }

        .smell-summary-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 16px;
        }

        .smell-summary-card.warning {
            border-left: 3px solid var(--warning);
        }

        .smell-summary-card.error {
            border-left: 3px solid var(--error);
        }

        .smell-card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .smell-card-icon {
            font-size: 18px;
        }

        .smell-card-title {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            flex: 1;
        }

        .smell-card-count {
            font-size: 24px;
            font-weight: bold;
            color: var(--text-primary);
        }

        /* Smells Table */
        .smells-table-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            overflow: hidden;
        }

        .smells-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .smells-table thead {
            background: var(--bg-tertiary);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .smells-table th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-primary);
        }

        .smells-table tbody tr {
            border-bottom: 1px solid var(--border-secondary);
        }

        .smells-table tbody tr:last-child {
            border-bottom: none;
        }

        .smell-row {
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .smell-row:hover {
            background: var(--bg-tertiary);
        }

        .smells-table td {
            padding: 12px;
            color: var(--text-primary);
        }

        .smell-type-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 500;
            font-size: 11px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            white-space: nowrap;
        }

        .severity-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 11px;
            min-width: 70px;
            justify-content: center;
        }

        .severity-badge.warning {
            background: rgba(210, 153, 34, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }

        .severity-badge.error {
            background: rgba(218, 54, 51, 0.2);
            color: var(--error);
            border: 1px solid var(--error);
        }

        .smell-name {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-weight: 500;
            color: var(--accent);
        }

        .smell-file {
            color: var(--text-secondary);
            font-size: 12px;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .smell-details {
            font-size: 12px;
            color: var(--text-secondary);
        }
    """


def get_dependencies_styles() -> str:
    """Get styles for the dependencies report.

    Returns:
        CSS string for dependencies styling
    """
    return """
        /* Dependencies Report Styles */
        .dependencies-report {
            padding: 0;
        }

        .dependency-summary {
            margin-bottom: 24px;
        }

        /* Circular Dependencies Warning */
        .circular-deps-warning {
            background: rgba(218, 54, 51, 0.1);
            border: 2px solid var(--error);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 24px;
        }

        .warning-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .warning-icon {
            font-size: 24px;
        }

        .warning-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--error);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .cycle-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-left: 36px;
        }

        .cycle-item {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 12px;
            color: var(--text-primary);
            padding: 8px 12px;
            background: var(--bg-secondary);
            border-left: 3px solid var(--error);
            border-radius: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* Dependencies Table */
        .dependencies-table-section {
            margin-top: 24px;
        }

        .dependencies-table-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            overflow: hidden;
        }

        .dependencies-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .dependencies-table thead {
            background: var(--bg-tertiary);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .dependencies-table th {
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-primary);
        }

        .dependencies-table th:nth-child(2),
        .dependencies-table th:nth-child(3),
        .dependencies-table th:nth-child(4) {
            text-align: center;
        }

        .dependencies-table th:nth-child(5) {
            width: 50px;
            text-align: center;
        }

        .dependencies-table tbody tr.dependency-row {
            border-bottom: 1px solid var(--border-secondary);
            transition: background 0.2s ease;
        }

        .dependencies-table tbody tr.dependency-row:hover {
            background: var(--bg-tertiary);
        }

        .dependencies-table tbody tr.dependency-row.in-cycle {
            background: rgba(218, 54, 51, 0.05);
        }

        .dependencies-table tbody tr.dependency-row.in-cycle:hover {
            background: rgba(218, 54, 51, 0.1);
        }

        .dependencies-table td {
            padding: 12px;
            color: var(--text-primary);
        }

        .dep-file {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-weight: 500;
            color: var(--accent);
        }

        .dep-count {
            text-align: center;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            color: var(--text-secondary);
        }

        .dep-total {
            text-align: center;
            font-weight: bold;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            color: var(--text-primary);
        }

        .dep-expand {
            text-align: center;
        }

        .expand-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            padding: 4px 10px;
            cursor: pointer;
            color: var(--text-primary);
            font-size: 12px;
            transition: all 0.2s ease;
        }

        .expand-btn:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-primary);
        }

        /* Dependency Details Row */
        .dependency-details {
            background: var(--bg-primary);
        }

        .dependency-details td {
            padding: 0 !important;
        }

        .dependency-details-content {
            padding: 16px 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .dependency-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 12px;
        }

        .dependency-section-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .dependency-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .dependency-item {
            display: inline-block;
            padding: 4px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            font-size: 11px;
            color: var(--text-primary);
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            transition: all 0.2s ease;
            cursor: default;
        }

        .dependency-item:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-primary);
        }

        .dependency-item-empty {
            color: var(--text-tertiary);
            font-size: 12px;
            font-style: italic;
        }

        .dependency-item-more {
            color: var(--text-secondary);
            font-size: 11px;
            padding: 4px 10px;
            font-style: italic;
        }
    """


def get_trends_styles() -> str:
    """Get styles for the trends/metrics snapshot report.

    Returns:
        CSS string for trends styling
    """
    return """
        /* Trends Report Styles */
        .trends-report {
            padding: 0;
        }

        /* Snapshot Banner */
        .snapshot-banner {
            background: var(--bg-secondary);
            border: 2px solid var(--accent);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }

        .snapshot-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 12px;
        }

        .snapshot-icon {
            font-size: 32px;
            flex-shrink: 0;
        }

        .snapshot-info {
            flex: 1;
        }

        .snapshot-title {
            font-size: 18px;
            font-weight: bold;
            color: var(--accent);
            margin-bottom: 4px;
        }

        .snapshot-timestamp {
            font-size: 13px;
            color: var(--text-secondary);
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        .snapshot-description {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Metrics Section */
        .metrics-section {
            margin-bottom: 24px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .metric-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.2s ease;
        }

        .metric-card:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow);
        }

        .metric-icon {
            font-size: 32px;
            margin-bottom: 12px;
        }

        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: var(--accent);
            margin-bottom: 8px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        .metric-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Health Score Section */
        .health-section {
            margin-bottom: 24px;
        }

        .health-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 24px;
        }

        .health-score-display {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 16px;
        }

        .health-score-value {
            font-size: 48px;
            font-weight: bold;
            color: var(--accent);
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        .health-score-label {
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .health-progress-container {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            height: 32px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        .health-progress-bar {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
            opacity: 0.9;
        }

        .health-description {
            font-size: 14px;
            color: var(--text-secondary);
            text-align: center;
            line-height: 1.6;
        }

        /* Distribution Section */
        .distribution-section,
        .size-distribution-section {
            margin-bottom: 24px;
        }

        .distribution-chart {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 20px;
        }

        .distribution-bar-row {
            display: grid;
            grid-template-columns: 180px 1fr 80px;
            gap: 16px;
            align-items: center;
            margin-bottom: 12px;
        }

        .distribution-bar-row:last-child {
            margin-bottom: 0;
        }

        .distribution-bar-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }

        .distribution-grade {
            font-size: 16px;
            font-weight: bold;
            min-width: 24px;
        }

        .distribution-range {
            font-size: 12px;
            color: var(--text-secondary);
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        .size-label {
            font-size: 13px;
            color: var(--text-primary);
        }

        .distribution-bar-container {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            height: 28px;
            overflow: hidden;
        }

        .distribution-bar-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
            opacity: 0.8;
        }

        .distribution-bar-value {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            text-align: right;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        /* Trends Chart Styles */
        .trends-section {
            margin-top: 32px;
        }

        .trends-container {
            display: flex;
            flex-direction: column;
            gap: 24px;
            margin-bottom: 16px;
        }

        .trend-chart {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
        }

        .trend-chart svg {
            display: block;
            margin: 0 auto;
        }

        .trend-info {
            font-size: 12px;
            color: var(--text-secondary);
            text-align: center;
            margin-top: 8px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }

        /* Future Section (fallback when no trend data) */
        .future-section {
            margin-top: 32px;
        }

        .future-placeholder {
            background: var(--bg-secondary);
            border: 2px dashed var(--border-primary);
            border-radius: 8px;
            padding: 40px;
            text-align: center;
        }

        .future-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }

        .future-title {
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        .future-description {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.8;
            max-width: 600px;
            margin: 0 auto;
        }

        .future-description code {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 12px;
        }

        .future-description ul {
            text-align: left;
            margin: 16px auto 0;
            padding-left: 20px;
            max-width: 400px;
        }

        .future-description li {
            margin-bottom: 8px;
        }
    """


def get_visualization_mode_styles() -> str:
    """Get styles for visualization mode selector and treemap/sunburst.

    Returns:
        CSS string for visualization mode controls and new viz types
    """
    return """
        /* Visualization Mode Selector */
        .viz-mode-buttons {
            display: flex;
            gap: 4px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            padding: 4px;
        }

        .viz-mode-btn {
            flex: 1;
            padding: 8px 10px;
            background: transparent;
            border: none;
            border-radius: 4px;
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .viz-mode-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .viz-mode-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .viz-mode-btn.active:hover {
            background: var(--accent-hover);
        }

        /* Treemap Styles */
        .treemap-container {
            width: 100%;
            height: 100%;
            position: relative;
        }

        .treemap-cell {
            position: absolute;
            overflow: hidden;
            cursor: pointer;
            transition: opacity 0.2s ease;
            box-sizing: border-box;
        }

        .treemap-cell:hover {
            opacity: 0.85;
        }

        .treemap-cell-inner {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            padding: 4px;
            box-sizing: border-box;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 2px;
        }

        .treemap-label {
            font-size: 11px;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.95);
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            pointer-events: none;
        }

        .treemap-label-parent {
            font-size: 13px;
            font-weight: 600;
            padding: 4px 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 3px 3px 0 0;
        }

        .treemap-value {
            font-size: 9px;
            color: rgba(255, 255, 255, 0.7);
            pointer-events: none;
        }

        /* Sunburst Styles */
        .sunburst-container {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .sunburst-arc {
            cursor: pointer;
            transition: opacity 0.2s ease;
        }

        .sunburst-arc:hover {
            opacity: 0.8;
        }

        .sunburst-label {
            pointer-events: none;
            font-size: 10px;
            fill: rgba(255, 255, 255, 0.9);
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        }

        .sunburst-center-label {
            font-size: 14px;
            font-weight: 600;
            fill: var(--text-primary);
            text-anchor: middle;
            dominant-baseline: middle;
        }

        .sunburst-center-value {
            font-size: 10px;
            fill: var(--text-secondary);
            text-anchor: middle;
        }

        /* Shared Breadcrumb Trail for Treemap/Sunburst */
        .viz-breadcrumb {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            background: rgba(13, 17, 23, 0.9);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            font-size: 12px;
            overflow-x: auto;
            white-space: nowrap;
        }

        .viz-breadcrumb-item {
            color: var(--accent);
            cursor: pointer;
            transition: color 0.2s;
        }

        .viz-breadcrumb-item:hover {
            color: var(--accent-hover);
            text-decoration: underline;
        }

        .viz-breadcrumb-separator {
            color: var(--text-tertiary);
        }

        .viz-breadcrumb-current {
            color: var(--text-primary);
            font-weight: 600;
        }

        .viz-breadcrumb-home {
            font-size: 14px;
            cursor: pointer;
        }

        .viz-breadcrumb-home:hover {
            color: var(--accent);
        }

        /* Complexity Grade Colors (for treemap/sunburst cells) */
        .grade-A { fill: #238636; background: #238636; }
        .grade-B { fill: #1f6feb; background: #1f6feb; }
        .grade-C { fill: #d29922; background: #d29922; }
        .grade-D { fill: #f0883e; background: #f0883e; }
        .grade-F { fill: #da3633; background: #da3633; }
        .grade-none { fill: #6e7681; background: #6e7681; }

        /* Tooltip for Treemap/Sunburst */
        .viz-tooltip {
            position: absolute;
            padding: 10px 14px;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid var(--border-primary);
            border-radius: 6px;
            font-size: 12px;
            color: var(--text-primary);
            pointer-events: none;
            z-index: 200;
            max-width: 300px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            opacity: 0;
            transition: opacity 0.15s ease;
        }

        .viz-tooltip.visible {
            opacity: 1;
        }

        .viz-tooltip-title {
            font-weight: 600;
            margin-bottom: 4px;
        }

        .viz-tooltip-row {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .viz-tooltip-label {
            color: var(--text-tertiary);
        }

        /* Animation for zoom transitions */
        @keyframes treemap-zoom {
            from { opacity: 0.5; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .treemap-zooming .treemap-cell {
            animation: treemap-zoom 0.3s ease-out;
        }

        @keyframes sunburst-zoom {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .sunburst-zooming path {
            animation: sunburst-zoom 0.4s ease-out;
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
            get_breadcrumb_styles(),
            get_content_pane_styles(),
            get_code_chunks_styles(),
            get_reset_button_styles(),
            get_spinner_styles(),
            get_theme_toggle_styles(),
            get_search_styles(),
            get_complexity_report_styles(),
            get_code_smells_styles(),
            get_dependencies_styles(),
            get_trends_styles(),
            get_visualization_mode_styles(),
        ]
    )
