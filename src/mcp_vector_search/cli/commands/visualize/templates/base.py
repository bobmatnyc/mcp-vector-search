"""HTML template generation for the visualization.

This module combines CSS and JavaScript from other template modules
to generate the complete HTML page for the D3.js visualization.
"""

from .scripts import get_all_scripts
from .styles import get_all_styles


def generate_html_template() -> str:
    """Generate the complete HTML template for visualization.

    Returns:
        Complete HTML string with embedded CSS and JavaScript
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Chunk Relationship Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
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
