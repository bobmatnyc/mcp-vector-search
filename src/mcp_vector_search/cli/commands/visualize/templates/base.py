"""HTML template generation for the visualization.

This module combines CSS and JavaScript from other template modules
to generate the complete HTML page for the D3.js visualization.
"""

import time

from .scripts import get_all_scripts
from .styles import get_all_styles


def generate_html_template() -> str:
    """Generate the complete HTML template for visualization.

    Returns:
        Complete HTML string with embedded CSS and JavaScript
    """
    # Add timestamp for cache busting
    build_timestamp = int(time.time())

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Chunk Relationship Graph</title>
    <meta http-cache="no-cache, no-store, must-revalidate">
    <meta http-pragma="no-cache">
    <meta http-expires="0">
    <!-- Build: {build_timestamp} -->
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <style>
{get_all_styles()}
    </style>
</head>
<body>
    <div id="controls">
        <h1>🔍 Code Graph</h1>

        <div class="control-group" id="loading">
            <label>⏳ Loading graph data...</label>
        </div>

        <div class="control-group" id="layout-controls" style="display: none;">
            <h3 style="margin: 12px 0 8px 0;">Layout</h3>
            <select id="layoutSelector" style="width: 100%; padding: 6px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9; font-size: 12px;">
                <option value="force">Force-Directed</option>
                <option value="dagre">Hierarchical (Dagre)</option>
                <option value="circle">Circular</option>
            </select>
        </div>

        <div class="control-group" id="edge-filters" style="display: none;">
            <h3 style="margin: 12px 0 8px 0;">Edge Filters</h3>
            <div style="font-size: 12px;">
                <label style="display: block; margin-bottom: 6px; cursor: pointer;">
                    <input type="checkbox" id="filter-containment" checked style="margin-right: 6px;">
                    Containment (dir/file)
                </label>
                <label style="display: block; margin-bottom: 6px; cursor: pointer;">
                    <input type="checkbox" id="filter-calls" checked style="margin-right: 6px;">
                    Function Calls
                </label>
                <label style="display: block; margin-bottom: 6px; cursor: pointer;">
                    <input type="checkbox" id="filter-imports" style="margin-right: 6px;">
                    Imports
                </label>
                <label style="display: block; margin-bottom: 6px; cursor: pointer;">
                    <input type="checkbox" id="filter-semantic" style="margin-right: 6px;">
                    Semantic Links
                </label>
                <label style="display: block; margin-bottom: 6px; cursor: pointer;">
                    <input type="checkbox" id="filter-cycles" checked style="margin-right: 6px;">
                    Circular Dependencies
                </label>
            </div>
        </div>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-category">
                <div class="legend-title">Code Elements</div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#d29922"/>
                    </svg>
                    <span>Function</span>
                </div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#1f6feb"/>
                    </svg>
                    <span>Class</span>
                </div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#8957e5"/>
                    </svg>
                    <span>Method</span>
                </div>
            </div>

            <div class="legend-category">
                <div class="legend-title">File</div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.py (Python) 🐍</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.js (JavaScript) 📜</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.ts (TypeScript) 📜</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.md (Markdown) 📝</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.json (JSON) ⚙️</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.yaml (YAML) ⚙️</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span style="margin-right: 6px;">📄</span>
                    <span>.sh (Shell) 💻</span>
                </div>
            </div>

            <div class="legend-category">
                <div class="legend-title">Indicators</div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#d29922" stroke="#ff6b6b" stroke-width="2"/>
                    </svg>
                    <span>Dead Code (red border)</span>
                </div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <line x1="2" y1="8" x2="14" y2="8" stroke="#ff4444" stroke-width="2" stroke-dasharray="4,2"/>
                    </svg>
                    <span>Circular Dependency (red dashed)</span>
                </div>
            </div>
        </div>

        <div id="subprojects-legend" style="display: none;">
            <h3>Subprojects</h3>
            <div class="legend" id="subprojects-list"></div>
        </div>

        <div class="stats" id="stats"></div>
    </div>

    <svg id="graph"></svg>
    <div id="tooltip" class="tooltip"></div>

    <button id="reset-view-btn" title="Reset to home view">
        <span style="font-size: 18px;">🏠</span>
        <span>Reset View</span>
    </button>

    <div id="content-pane">
        <div class="pane-header">
            <button class="collapse-btn" onclick="closeContentPane()">×</button>
            <div class="code-viewer-nav">
                <button id="navBack" disabled title="Back (Alt+Left)">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M9.78 12.78a.75.75 0 0 1-1.06 0L4.47 8.53a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L6.06 8l3.72 3.72a.75.75 0 0 1 0 1.06Z"></path>
                    </svg>
                </button>
                <button id="navForward" disabled title="Forward (Alt+Right)">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"></path>
                    </svg>
                </button>
                <span id="navPosition"></span>
            </div>
            <div class="pane-title" id="pane-title"></div>
            <div class="pane-meta" id="pane-meta"></div>
        </div>
        <div class="pane-content" id="pane-content"></div>
        <div class="pane-footer" id="pane-footer"></div>
    </div>

    <script>
{get_all_scripts()}
    </script>
</body>
</html>"""
    return html


def inject_data(html: str, data: dict) -> str:
    """Inject graph data into HTML template (not currently used for static export).

    This function is provided for potential future use where data might be
    embedded directly in the HTML rather than loaded from a separate JSON file.

    Args:
        html: HTML template string
        data: Graph data dictionary

    Returns:
        HTML with embedded data
    """
    # For now, we load data from external JSON file
    # This function can be enhanced later if inline data embedding is needed
    return html
