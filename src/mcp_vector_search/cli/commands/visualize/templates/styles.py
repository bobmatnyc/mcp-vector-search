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
            z-index: 500;
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
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
            max-width: 300px;
            margin-top: 16px;
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

        .toggle-switch-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 10px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
        }

        .toggle-label {
            font-size: 13px;
            color: #8b949e;
            font-weight: 500;
            transition: color 0.2s ease;
        }

        .toggle-label.active {
            color: #58a6ff;
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
            background-color: #30363d;
            transition: 0.3s;
            border-radius: 24px;
            border: 1px solid #30363d;
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: #8b949e;
            transition: 0.3s;
            border-radius: 50%;
        }

        .toggle-switch input:checked + .toggle-slider {
            background-color: #238636;
            border-color: #2ea043;
        }

        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(24px);
            background-color: #ffffff;
        }

        .toggle-slider:hover {
            background-color: #3a424d;
        }

        .toggle-switch input:checked + .toggle-slider:hover {
            background-color: #2ea043;
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
            stroke: #8b949e;
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
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.6;
            overflow-x: auto;
            white-space: nowrap;
        }

        .breadcrumb-root {
            color: #58a6ff;
            cursor: pointer;
            font-weight: 500;
            transition: color 0.2s;
        }

        .breadcrumb-root:hover {
            color: #79c0ff;
            text-decoration: underline;
        }

        .breadcrumb-link {
            color: #58a6ff;
            cursor: pointer;
            transition: color 0.2s;
        }

        .breadcrumb-link:hover {
            color: #79c0ff;
            text-decoration: underline;
        }

        .breadcrumb-separator {
            color: #6e7681;
            margin: 0 6px;
        }

        .breadcrumb-current {
            color: #c9d1d9;
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
            background: rgba(13, 17, 23, 0.98);
            border-left: 1px solid #30363d;
            overflow-y: auto;
            box-shadow: -4px 0 24px rgba(0, 0, 0, 0.5);
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
            background: rgba(13, 17, 23, 0.98);
            padding: 16px 20px;
            border-bottom: 1px solid #30363d;
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
            color: #8b949e;
            font-size: 18px;
            line-height: 1;
            background: none;
            border: none;
            padding: 4px;
            transition: color 0.2s, background 0.2s;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
        }

        .viewer-expand-btn:hover {
            color: #c9d1d9;
            background: rgba(255, 255, 255, 0.1);
        }

        .viewer-title {
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
            margin: 0 0 8px 0;
            padding-right: 80px;
        }

        .section-nav {
            margin-top: 8px;
        }

        .section-nav select {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
            cursor: pointer;
            width: 100%;
            max-width: 250px;
        }

        .section-nav select:hover {
            border-color: #58a6ff;
        }

        .section-nav select:focus {
            outline: none;
            border-color: #58a6ff;
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
        }

        .viewer-close-btn {
            cursor: pointer;
            color: #8b949e;
            font-size: 28px;
            line-height: 1;
            background: none;
            border: none;
            padding: 0;
            transition: color 0.2s;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .viewer-close-btn:hover {
            color: #c9d1d9;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
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

        /* Node highlight animation */
        .node-highlight circle {
            stroke: #58a6ff !important;
            stroke-width: 3px !important;
            filter: drop-shadow(0 0 6px rgba(88, 166, 255, 0.6));
        }

        .node-highlight text {
            fill: #58a6ff !important;
            font-weight: bold !important;
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
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
            position: relative;
            z-index: 101;
            cursor: text;
        }

        #search-input:focus {
            border-color: #58a6ff;
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
        }

        #search-input::placeholder {
            color: #8b949e;
        }

        /* Search results dropdown */
        .search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 300px;
            overflow-y: auto;
            background: #161b22;
            border: 1px solid #30363d;
            border-top: none;
            border-radius: 0 0 6px 6px;
            z-index: 1000;
            display: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
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
            border-bottom: 1px solid #21262d;
            transition: background 0.15s;
        }

        .search-result-item:last-child {
            border-bottom: none;
        }

        .search-result-item:hover,
        .search-result-item.selected {
            background: #21262d;
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
            color: #c9d1d9;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .search-result-name mark {
            background: rgba(88, 166, 255, 0.3);
            color: #58a6ff;
            border-radius: 2px;
            padding: 0 2px;
        }

        .search-result-path {
            font-size: 11px;
            color: #8b949e;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin-top: 2px;
        }

        .search-result-type {
            font-size: 10px;
            color: #8b949e;
            background: #21262d;
            padding: 2px 6px;
            border-radius: 10px;
            text-transform: uppercase;
            flex-shrink: 0;
        }

        .search-no-results {
            padding: 20px;
            text-align: center;
            color: #8b949e;
            font-size: 13px;
        }

        .search-hint {
            padding: 8px 12px;
            font-size: 11px;
            color: #6e7681;
            background: #0d1117;
            border-top: 1px solid #21262d;
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
            get_search_styles(),
        ]
    )
