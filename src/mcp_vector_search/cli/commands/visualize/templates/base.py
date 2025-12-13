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
    <!-- Highlight.js for syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
{get_all_styles()}
    </style>
</head>
<body>
    <div id="controls">
        <div class="search-container">
            <input type="text" id="search-input" placeholder="Search nodes..." oninput="handleSearchInput(event)" onkeydown="handleSearchKeydown(event)">
            <div id="search-results" class="search-results"></div>
        </div>

        <h1>🔍 Code Tree</h1>

        <div class="control-group">
            <label style="color: var(--text-primary); margin-bottom: 8px;">Layout Mode</label>
            <div class="toggle-switch-container">
                <span class="toggle-label">Linear</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="layout-toggle" onchange="toggleLayout()">
                    <span class="toggle-slider"></span>
                </label>
                <span class="toggle-label">Circular</span>
            </div>
        </div>

        <div class="control-group">
            <label style="color: var(--text-primary); margin-bottom: 8px;">Show Files</label>
            <div class="filter-buttons">
                <button class="filter-btn active" data-filter="all" onclick="setFileFilter('all')">All</button>
                <button class="filter-btn" data-filter="code" onclick="setFileFilter('code')">Code</button>
                <button class="filter-btn" data-filter="docs" onclick="setFileFilter('docs')">Docs</button>
            </div>
        </div>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-category">
                <div class="legend-title">Node Types</div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#3498db"/>
                    </svg>
                    <span>Directory (expanded)</span>
                </div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#f39c12"/>
                    </svg>
                    <span>Directory (collapsed)</span>
                </div>
                <div class="legend-item">
                    <svg width="16" height="16" style="margin-right: 8px;">
                        <circle cx="8" cy="8" r="6" fill="#95a5a6"/>
                    </svg>
                    <span>File</span>
                </div>
            </div>

            <div class="legend-category">
                <div class="legend-title">Call Relationships</div>
                <div class="legend-item">
                    <svg width="40" height="16" style="margin-right: 8px;">
                        <line x1="0" y1="8" x2="35" y2="8" stroke="#58a6ff" stroke-width="2" stroke-dasharray="4,2"/>
                        <text x="38" y="12" fill="#58a6ff" font-size="12">←</text>
                    </svg>
                    <span>Inbound calls (called by)</span>
                </div>
                <div class="legend-item">
                    <svg width="40" height="16" style="margin-right: 8px;">
                        <line x1="0" y1="8" x2="35" y2="8" stroke="#f0883e" stroke-width="2" stroke-dasharray="4,2"/>
                        <text x="38" y="12" fill="#f0883e" font-size="12">→</text>
                    </svg>
                    <span>Outbound calls (calls to)</span>
                </div>
                <div class="legend-item" style="margin-top: 8px;">
                    <label class="toggle-switch">
                        <input type="checkbox" id="show-call-lines" checked onchange="toggleCallLines(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                    <span style="margin-left: 8px;">Show call lines</span>
                </div>
            </div>

            <div class="legend-category">
                <div class="legend-title">Interactions</div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span>Click directory → expand/collapse</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span>Click file → view info</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span>Click chunk → view code</span>
                </div>
                <div class="legend-item" style="padding-left: 16px;">
                    <span>Scroll → zoom in/out</span>
                </div>
            </div>
        </div>

        <!-- Options Section -->
        <h3>Options</h3>
        <div class="legend" style="margin-top: 8px;">
            <div class="legend-category" style="border-bottom: none;">
                <!-- Theme Toggle -->
                <div class="legend-item" style="margin-bottom: 12px; padding: 0;">
                    <button class="theme-toggle-icon-btn" onclick="toggleTheme()" title="Toggle dark/light theme">
                        <span class="theme-icon" id="theme-icon">🌙</span>
                    </button>
                    <span style="margin-left: 8px; color: var(--text-secondary); font-size: 12px;">Theme</span>
                </div>

                <!-- Analysis Reports -->
                <div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showComplexityReport()">
                    <span style="margin-right: 8px;">📊</span>
                    <span>Complexity Report</span>
                </div>
                <div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showCodeSmells()">
                    <span style="margin-right: 8px;">🔍</span>
                    <span>Code Smells</span>
                </div>
                <div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 6px;" onclick="showDependencies()">
                    <span style="margin-right: 8px;">🔗</span>
                    <span>Dependencies</span>
                </div>
                <div class="legend-item" style="cursor: pointer; padding: 8px; background: var(--bg-tertiary); border-radius: 4px;" onclick="showTrends()">
                    <span style="margin-right: 8px;">📈</span>
                    <span>Trends</span>
                </div>
            </div>
        </div>

        <div class="stats" id="stats"></div>
    </div>

    <div id="main-container">
        <svg id="graph"></svg>
    </div>

    <div id="viewer-panel" class="viewer-panel">
        <div class="viewer-header">
            <div class="viewer-header-buttons">
                <button class="viewer-expand-btn" onclick="toggleViewerExpand()" title="Expand/Collapse panel">
                    <span id="expand-icon">⬅</span>
                </button>
                <button class="viewer-close-btn" onclick="closeViewerPanel()" title="Close panel">×</button>
            </div>
            <h2 class="viewer-title" id="viewer-title">Viewer</h2>
            <div class="section-nav" id="section-nav">
                <select id="section-dropdown" onchange="jumpToSection(this.value)" title="Jump to section">
                    <option value="">Jump to section...</option>
                </select>
            </div>
        </div>
        <div class="viewer-content" id="viewer-content">
            <p style="color: #8b949e; text-align: center; padding: 40px;">Select a node to view details</p>
        </div>
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
