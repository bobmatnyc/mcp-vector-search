"""HTML standalone report generator for structural code analysis results.

This module provides the HTMLReportGenerator class that creates self-contained,
interactive HTML reports from analysis data. Reports include:
- Interactive charts using Chart.js
- Responsive design for mobile/desktop
- Code syntax highlighting via Highlight.js
- Embedded CSS and JavaScript (only CDN for libraries)
- Grade-based color coding and visualizations

The generated HTML files are fully self-contained except for CDN dependencies,
making them easy to share and view without additional infrastructure.

Example:
    >>> from pathlib import Path
    >>> from mcp_vector_search.analysis.visualizer import JSONExporter, HTMLReportGenerator
    >>>
    >>> # Export analysis to schema format
    >>> exporter = JSONExporter(project_root=Path("/path/to/project"))
    >>> export = exporter.export(project_metrics)
    >>>
    >>> # Generate HTML report
    >>> html_gen = HTMLReportGenerator(title="My Project Analysis")
    >>> html_output = html_gen.generate(export)
    >>>
    >>> # Or write directly to file
    >>> html_path = html_gen.generate_to_file(export, Path("report.html"))
"""

from __future__ import annotations

import json
from pathlib import Path

from .d3_data import transform_for_d3
from .schemas import AnalysisExport


class HTMLReportGenerator:
    """Generates standalone HTML reports from analysis data.

    Creates self-contained HTML files with embedded styles and scripts,
    using CDN links only for Chart.js (visualizations) and Highlight.js
    (syntax highlighting).

    Attributes:
        title: Report title displayed in header and <title> tag
    """

    # CDN URLs for external libraries
    CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"
    HIGHLIGHT_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0"
    D3_JS_CDN = "https://d3js.org/d3.v7.min.js"

    def __init__(self, title: str = "Code Analysis Report"):
        """Initialize HTML report generator.

        Args:
            title: Title for the report (default: "Code Analysis Report")
        """
        self.title = title

    def generate(self, export: AnalysisExport) -> str:
        """Generate complete HTML report as a string.

        Args:
            export: Analysis export data in schema format

        Returns:
            Complete HTML document as string
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="{self.CHART_JS_CDN}"></script>
    <script src="{self.D3_JS_CDN}"></script>
    <link rel="stylesheet" href="{self.HIGHLIGHT_JS_CDN}/styles/github.min.css">
    <script src="{self.HIGHLIGHT_JS_CDN}/highlight.min.js"></script>
    {self._generate_styles()}
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <div class="a11y-controls" role="toolbar" aria-label="Accessibility controls">
        <button id="toggle-contrast" class="a11y-button" aria-pressed="false" title="Toggle high contrast mode">
            High Contrast
        </button>
        <button id="toggle-reduced-motion" class="a11y-button" aria-pressed="false" title="Toggle reduced motion preference">
            Reduce Motion
        </button>
    </div>
    <div id="main-content">
    {self._generate_header(export)}
    {self._generate_summary_section(export)}
    {self._generate_d3_graph_section(export)}
    {self._generate_complexity_chart(export)}
    {self._generate_grade_distribution(export)}
    {self._generate_smells_section(export)}
    {self._generate_files_table(export)}
    {self._generate_dependencies_section(export)}
    {self._generate_trends_section(export)}
    {self._generate_footer(export)}
    </div><!-- #main-content -->
    {self._generate_scripts(export)}
</body>
</html>"""

    def generate_to_file(self, export: AnalysisExport, output_path: Path) -> Path:
        """Generate HTML report and write to file.

        Args:
            export: Analysis export data
            output_path: Path where HTML file will be written

        Returns:
            Path to the created HTML file
        """
        html = self.generate(export)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _generate_styles(self) -> str:
        """Generate embedded CSS styles.

        Returns:
            <style> block with complete CSS
        """
        return """<style>
:root {
    --primary: #3b82f6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-700: #374151;
    --gray-900: #111827;
}

/* High Contrast Mode */
body.high-contrast {
    --primary: #0056b3;
    --gray-50: #ffffff;
    --gray-100: #e0e0e0;
    --gray-200: #c0c0c0;
    --gray-700: #000000;
    --gray-900: #000000;
    background: #ffffff;
    color: #000000;
}

body.high-contrast .card {
    border: 2px solid #000000;
}

body.high-contrast .node circle {
    stroke: #000000 !important;
    stroke-width: 2px !important;
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--gray-900);
    background: var(--gray-50);
    padding: 2rem;
}

.container { max-width: 1200px; margin: 0 auto; }

header {
    background: linear-gradient(135deg, var(--primary), #1d4ed8);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}

h1 { font-size: 2rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.5rem; margin-bottom: 1rem; color: var(--gray-700); }
h3 { font-size: 1.25rem; margin-bottom: 0.75rem; }

.card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.stat-card {
    background: var(--gray-100);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}

.stat-value { font-size: 2rem; font-weight: bold; color: var(--primary); }
.stat-label { font-size: 0.875rem; color: var(--gray-700); }

.grade-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.875rem;
}

.grade-a { background: #dcfce7; color: #166534; }
.grade-b { background: #dbeafe; color: #1e40af; }
.grade-c { background: #fef3c7; color: #92400e; }
.grade-d { background: #fed7aa; color: #9a3412; }
.grade-f { background: #fecaca; color: #991b1b; }

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--gray-200);
}

th { background: var(--gray-100); font-weight: 600; }

.chart-container {
    position: relative;
    height: 300px;
    margin: 1rem 0;
}

.health-bar {
    height: 8px;
    background: var(--gray-200);
    border-radius: 4px;
    overflow: hidden;
}

.health-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

footer {
    text-align: center;
    padding: 2rem;
    color: var(--gray-700);
    font-size: 0.875rem;
}

/* Accessibility Controls */
.a11y-controls {
    position: fixed;
    top: 1rem;
    right: 1rem;
    display: flex;
    gap: 0.5rem;
    z-index: 10000;
}

.a11y-button {
    padding: 0.5rem 1rem;
    background: white;
    border: 2px solid var(--gray-200);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 600;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.a11y-button:hover {
    background: var(--gray-100);
    border-color: var(--primary);
}

.a11y-button:focus {
    outline: 3px solid var(--primary);
    outline-offset: 2px;
}

.a11y-button.active {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
}

/* Skip Link */
.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: var(--primary);
    color: white;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 0 0 4px 0;
    z-index: 10001;
}

.skip-link:focus {
    top: 0;
}

/* Export Controls */
.export-controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--gray-200);
}

.export-button {
    padding: 0.5rem 1rem;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.export-button:hover {
    background: #1d4ed8;
}

.export-button:active {
    transform: translateY(1px);
}

.export-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Focus trap indicator */
.focus-trapped {
    outline: 3px solid var(--primary);
    outline-offset: 2px;
}

/* Screen reader only */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

/* Loading/Error States */
.error-message {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #991b1b;
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--gray-200);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
    body { padding: 1rem; }
    .stats-grid { grid-template-columns: 1fr; }
    h1 { font-size: 1.5rem; }
    .a11y-controls {
        position: relative;
        top: auto;
        right: auto;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
}

/* D3 Graph Styles */
.graph-dashboard-container {
    position: relative;
    width: 100%;
    margin-bottom: 1rem;
}

#d3-graph-container {
    position: relative;
    overflow: hidden;
    background: #fafafa;
    height: 700px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

#d3-graph {
    width: 100%;
    height: 100%;
}

/* Stats Toggle Button */
.stats-toggle-button {
    position: absolute;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
    padding: 0.5rem 1rem;
    background: white;
    border: 2px solid var(--gray-200);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 600;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.stats-toggle-button:hover {
    background: var(--gray-100);
    border-color: var(--primary);
}

.stats-toggle-button:focus {
    outline: 3px solid var(--primary);
    outline-offset: 2px;
}

.stats-toggle-button.active {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
}

/* Filter Controls Panel */
.filter-controls {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.filter-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.filter-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.filter-label {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--gray-700);
}

.filter-checkboxes {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.filter-checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.875rem;
    cursor: pointer;
}

.filter-checkbox-label input[type="checkbox"] {
    cursor: pointer;
}

.filter-input {
    padding: 0.5rem;
    border: 1px solid var(--gray-200);
    border-radius: 6px;
    font-size: 0.875rem;
    font-family: inherit;
}

.filter-input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.filter-select {
    padding: 0.5rem;
    border: 1px solid var(--gray-200);
    border-radius: 6px;
    font-size: 0.875rem;
    background: white;
    cursor: pointer;
}

.filter-select:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.filter-button {
    padding: 0.5rem 1rem;
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.filter-button:hover {
    background: var(--gray-200);
}

.filter-button:active {
    transform: translateY(1px);
}

.filter-actions {
    display: flex;
    gap: 0.5rem;
}

/* Node filtering states */
.node-filtered {
    opacity: 0.1 !important;
    transition: opacity 0.3s;
}

.link-filtered {
    opacity: 0.05 !important;
    stroke: var(--gray-200) !important;
    transition: opacity 0.3s, stroke 0.3s;
}

/* Search highlight */
.node-search-highlight {
    filter: drop-shadow(0 0 6px #fbbf24) !important;
}

/* Keyboard focus */
.node-focused {
    filter: drop-shadow(0 0 8px var(--primary)) !important;
}

.node circle:focus {
    outline: 2px solid var(--primary);
    outline-offset: 4px;
}

/* Dashboard Panels */
.dashboard-panels {
    position: fixed;
    top: 0;
    right: -400px;
    width: 400px;
    height: 100vh;
    background: white;
    box-shadow: -2px 0 8px rgba(0,0,0,0.1);
    z-index: 999;
    transition: right 0.3s ease-in-out;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.dashboard-panels.visible {
    right: 0;
}

.dashboard-panel {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    overflow-y: auto;
}

.dashboard-panel h3 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--gray-900);
}

.panel-stat {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--gray-100);
}

.panel-stat:last-child {
    border-bottom: none;
}

.panel-stat-label {
    color: var(--gray-700);
    font-size: 0.875rem;
}

.panel-stat-value {
    font-weight: 600;
    color: var(--gray-900);
}

.smell-badge-small {
    display: inline-block;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.smell-badge-error {
    background: #fecaca;
    color: #991b1b;
}

.smell-badge-warning {
    background: #fed7aa;
    color: #9a3412;
}

.smell-badge-info {
    background: #dbeafe;
    color: #1e40af;
}

.panel-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.panel-list-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--gray-100);
    font-size: 0.875rem;
}

.panel-list-item:last-child {
    border-bottom: none;
}

#node-detail-panel.hidden {
    display: none;
}

.detail-header {
    background: var(--gray-100);
    padding: 0.75rem;
    border-radius: 6px;
    margin-bottom: 0.75rem;
}

.detail-header-title {
    font-weight: 600;
    color: var(--gray-900);
    margin-bottom: 0.25rem;
}

.detail-header-path {
    font-size: 0.75rem;
    color: var(--gray-600);
    font-family: 'Monaco', 'Courier New', monospace;
}

.detail-section {
    margin-bottom: 1rem;
}

.detail-section-title {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--gray-700);
    margin-bottom: 0.5rem;
}

.circular-warning {
    color: #dc2626;
    font-weight: 600;
}

/* Node styles - complexity shading (darker = more complex) */
.node-complexity-low { fill: #f3f4f6; }
.node-complexity-moderate { fill: #9ca3af; }
.node-complexity-high { fill: #4b5563; }
.node-complexity-very-high { fill: #1f2937; }
.node-complexity-critical { fill: #111827; }

/* Node borders - smell severity (redder = worse) */
.smell-none { stroke: #e5e7eb; stroke-width: 1px; }
.smell-info { stroke: #fca5a5; stroke-width: 2px; }
.smell-warning { stroke: #f87171; stroke-width: 3px; }
.smell-error { stroke: #ef4444; stroke-width: 4px; }
.smell-critical { stroke: #dc2626; stroke-width: 5px; filter: drop-shadow(0 0 4px #dc2626); }

/* Edge styles */
.link {
    fill: none;
    stroke-opacity: 0.6;
}

.link-circular {
    fill: none;
    stroke: #dc2626;
    stroke-opacity: 0.8;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { stroke-opacity: 0.8; }
    50% { stroke-opacity: 0.3; }
}

/* Arrowhead marker */
.arrowhead {
    fill: #64748b;
}

.arrowhead-circular {
    fill: #dc2626;
}

/* Module cluster hulls */
.module-hull {
    fill-opacity: 0.1;
    stroke-width: 2px;
    stroke-opacity: 0.4;
}

/* Node hover effects */
.node-dimmed {
    opacity: 0.2;
    transition: opacity 0.3s;
}

.node-highlighted {
    opacity: 1;
    transition: opacity 0.3s;
}

.link-dimmed {
    opacity: 0.1;
    transition: opacity 0.3s;
}

.link-highlighted {
    opacity: 1;
    transition: opacity 0.3s;
}

/* Node labels */
.node-label {
    font-size: 10px;
    fill: #374151;
    text-anchor: middle;
    pointer-events: none;
}

/* Tooltip */
.d3-tooltip {
    position: absolute;
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    font-size: 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    pointer-events: none;
    z-index: 1000;
    max-width: 320px;
    opacity: 0;
    transition: opacity 0.2s;
}

.tooltip-header {
    font-weight: 700;
    font-size: 13px;
    color: var(--gray-900);
    margin-bottom: 6px;
}

.tooltip-subtitle {
    color: var(--gray-600);
    font-size: 11px;
    margin-bottom: 10px;
}

.tooltip-section {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #f3f4f6;
}

.tooltip-metric {
    display: flex;
    justify-content: space-between;
    margin: 4px 0;
    font-size: 11px;
}

.tooltip-metric-label {
    color: var(--gray-600);
}

.tooltip-metric-value {
    font-weight: 600;
    color: var(--gray-900);
}

.tooltip-bar {
    width: 100%;
    height: 6px;
    background: #f3f4f6;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
}

.tooltip-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}

.tooltip-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    margin-left: 4px;
}

/* Legend - Overlay in Upper Left */
.d3-legend-container {
    position: absolute;
    top: 1rem;
    left: 1rem;
    z-index: 900;
    max-width: 300px;
}

.d3-legend {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    backdrop-filter: blur(4px);
}

.legend-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
}

.legend-section:last-child {
    margin-bottom: 0;
}

.legend-title {
    font-weight: 600;
    font-size: 11px;
    color: var(--gray-700);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--gray-700);
}

.legend-circle {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
}

.legend-line {
    width: 20px;
    height: 2px;
    flex-shrink: 0;
}

/* Collapsible Legend */
.legend-toggle {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.75rem;
    color: var(--gray-700);
    margin-bottom: 0.5rem;
    width: 100%;
    text-align: left;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.legend-toggle:hover {
    background: var(--gray-50);
    border-color: var(--primary);
}

.legend-content.collapsed {
    display: none;
}

.count-badge {
    background: var(--gray-700);
    color: white;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    margin-left: 0.5rem;
}

@media (max-width: 768px) {
    #d3-graph-container { height: 500px; }
    .d3-legend-container {
        max-width: 250px;
    }
    .dashboard-panels {
        width: 100%;
        right: -100%;
    }
    .stats-toggle-button {
        font-size: 0.75rem;
        padding: 0.4rem 0.8rem;
    }
}
</style>"""

    def _generate_header(self, export: AnalysisExport) -> str:
        """Generate report header with metadata.

        Args:
            export: Analysis export containing metadata

        Returns:
            HTML header section
        """
        meta = export.metadata
        git_info = ""
        if meta.git_commit:
            git_info = (
                f"<p>Git: {meta.git_branch or 'unknown'} @ {meta.git_commit[:8]}</p>"
            )

        return f"""<div class="container">
<header>
    <h1>📊 {self.title}</h1>
    <p>Generated: {meta.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Project: {meta.project_root}</p>
    {git_info}
</header>"""

    def _generate_summary_section(self, export: AnalysisExport) -> str:
        """Generate summary statistics cards.

        Args:
            export: Analysis export with summary data

        Returns:
            HTML summary section
        """
        s = export.summary
        return f"""<section class="card">
    <h2>📈 Project Summary</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{s.total_files:,}</div>
            <div class="stat-label">Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s.total_functions:,}</div>
            <div class="stat-label">Functions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s.total_classes:,}</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s.total_lines:,}</div>
            <div class="stat-label">Lines of Code</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s.avg_complexity:.1f}</div>
            <div class="stat-label">Avg Complexity</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s.total_smells:,}</div>
            <div class="stat-label">Code Smells</div>
        </div>
    </div>
</section>"""

    def _generate_d3_graph_section(self, export: AnalysisExport) -> str:
        """Generate D3.js interactive dependency graph section.

        Args:
            export: Analysis export with files and dependencies

        Returns:
            HTML section with D3 graph container and legend
        """
        # Transform data for D3
        d3_data = transform_for_d3(export)
        d3_json = json.dumps(d3_data)
        summary = d3_data["summary"]

        # Generate filter controls HTML
        filter_controls_html = self._generate_filter_controls(d3_data)

        # Generate summary panel HTML
        summary_panel_html = self._generate_summary_panel(summary)

        # Generate detail panel HTML (initially hidden)
        detail_panel_html = self._generate_detail_panel()

        # Generate legend HTML with counts
        legend_html = self._generate_legend_with_counts(summary)

        return f"""<section class="card" id="graph-section">
    <h2>🔗 Interactive Dependency Graph</h2>
    <p style="color: var(--gray-700); margin-bottom: 1rem;">
        Explore file dependencies with interactive visualization. Node size reflects lines of code,
        fill color shows complexity (darker = more complex), and border color indicates code smells
        (redder = more severe). Click nodes for details, drag to rearrange, zoom and pan to explore.
        Use filters below to focus on specific aspects.
    </p>
    <!-- Screen reader announcements -->
    <div id="sr-announcements" class="sr-only" role="status" aria-live="polite" aria-atomic="true"></div>
    {filter_controls_html}
    <div class="graph-dashboard-container">
        <!-- Stats Toggle Button in Upper Right -->
        <button
            id="stats-toggle"
            class="stats-toggle-button"
            onclick="toggleStatsPanel()"
            aria-label="Toggle statistics panel"
            aria-expanded="false"
        >
            📊 Show Stats
        </button>

        <!-- Main Graph Container -->
        <div id="d3-graph-container" role="img" aria-label="Interactive dependency graph visualization">
            <!-- Legend Overlay in Upper Left -->
            {legend_html}

            <svg id="d3-graph" aria-label="Dependency graph"></svg>
            <div id="graph-error" class="error-message" style="display: none;" role="alert">
                <strong>Error:</strong> <span id="graph-error-message"></span>
            </div>
            <div id="graph-loading" style="display: none; text-align: center; padding: 2rem;">
                <div class="loading-spinner"></div>
                <p style="margin-top: 1rem; color: var(--gray-600);">Loading visualization...</p>
            </div>
        </div>

        <!-- Stats Panel (Hidden by Default, Slides in from Right) -->
        <div id="stats-panel-container" class="dashboard-panels" aria-hidden="true">
            <button
                class="stats-toggle-button"
                onclick="toggleStatsPanel()"
                aria-label="Close statistics panel"
                style="position: relative; top: 0; right: 0; margin-bottom: 1rem;"
            >
                ✕ Hide Stats
            </button>
            {summary_panel_html}
            {detail_panel_html}
        </div>
    </div>
    <script id="d3-graph-data" type="application/json">{d3_json}</script>
</section>"""

    def _generate_filter_controls(self, d3_data: dict) -> str:
        """Generate filter controls panel for the graph.

        Args:
            d3_data: D3 graph data with nodes and modules

        Returns:
            HTML for filter controls panel
        """
        # Get unique modules
        modules = set()
        for node in d3_data.get("nodes", []):
            if node.get("module"):
                modules.add(node["module"])
        modules = sorted(modules)

        module_options = "".join(
            f'<option value="{module}">{module}</option>' for module in modules
        )

        return f"""<div class="filter-controls">
    <div class="filter-row">
        <div class="filter-group">
            <label class="filter-label">Complexity Grade</label>
            <div class="filter-checkboxes">
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-grade-a" value="A" checked>
                    <span>A (0-5)</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-grade-b" value="B" checked>
                    <span>B (6-10)</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-grade-c" value="C" checked>
                    <span>C (11-20)</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-grade-d" value="D" checked>
                    <span>D (21-30)</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-grade-f" value="F" checked>
                    <span>F (31+)</span>
                </label>
            </div>
        </div>
        <div class="filter-group">
            <label class="filter-label">Code Smells</label>
            <div class="filter-checkboxes">
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-smell-none" value="none" checked>
                    <span>None</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-smell-info" value="info" checked>
                    <span>Info</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-smell-warning" value="warning" checked>
                    <span>Warning</span>
                </label>
                <label class="filter-checkbox-label">
                    <input type="checkbox" id="filter-smell-error" value="error" checked>
                    <span>Error</span>
                </label>
            </div>
        </div>
    </div>
    <div class="filter-row">
        <div class="filter-group">
            <label class="filter-label" for="filter-module">Module Filter</label>
            <select id="filter-module" class="filter-select" multiple size="3">
                <option value="" selected>All Modules</option>
                {module_options}
            </select>
        </div>
        <div class="filter-group">
            <label class="filter-label" for="filter-search">Search Files</label>
            <input
                type="text"
                id="filter-search"
                class="filter-input"
                placeholder="Search by file name..."
                autocomplete="off"
            >
        </div>
        <div class="filter-group" style="justify-content: flex-end;">
            <label class="filter-label" style="visibility: hidden;">Actions</label>
            <div class="filter-actions">
                <button class="filter-button" onclick="resetFilters()">
                    Reset Filters
                </button>
                <button class="filter-button" onclick="clearSelection()">
                    Clear Selection
                </button>
            </div>
        </div>
    </div>
    <div class="export-controls">
        <button class="export-button" onclick="exportAsPNG()" aria-label="Export graph as PNG image">
            📥 Export PNG
        </button>
        <button class="export-button" onclick="exportAsSVG()" aria-label="Export graph as SVG vector image">
            📥 Export SVG
        </button>
        <button class="export-button" onclick="copyShareLink()" aria-label="Copy shareable link with current filters">
            🔗 Copy Link
        </button>
    </div>
</div>"""

    def _generate_summary_panel(self, summary: dict) -> str:
        """Generate the metrics summary panel.

        Args:
            summary: Summary statistics dictionary

        Returns:
            HTML for the summary panel
        """
        grade_class = f"grade-{summary['complexity_grade'].lower()}"
        circular_class = (
            "circular-warning" if summary["circular_dependencies"] > 0 else ""
        )

        return f"""<div class="dashboard-panel" id="summary-panel">
    <h3>📊 Project Metrics</h3>
    <div class="panel-stat">
        <span class="panel-stat-label">Total Files</span>
        <span class="panel-stat-value">{summary["total_files"]:,}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Total Functions</span>
        <span class="panel-stat-value">{summary["total_functions"]:,}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Total Classes</span>
        <span class="panel-stat-value">{summary["total_classes"]:,}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Total LOC</span>
        <span class="panel-stat-value">{summary["total_lines"]:,}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Average Complexity</span>
        <span class="panel-stat-value">
            {summary["avg_complexity"]:.1f}
            <span class="grade-badge {grade_class}">{summary["complexity_grade"]}</span>
        </span>
    </div>
    <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid var(--gray-200);">
    <div class="detail-section-title">Code Smells</div>
    <div class="panel-stat">
        <span class="panel-stat-label">Total Smells</span>
        <span class="panel-stat-value">{summary["total_smells"]:,}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Errors</span>
        <span class="panel-stat-value">
            <span class="smell-badge-small smell-badge-error">{summary["error_count"]}</span>
        </span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Warnings</span>
        <span class="panel-stat-value">
            <span class="smell-badge-small smell-badge-warning">{summary["warning_count"]}</span>
        </span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Info</span>
        <span class="panel-stat-value">
            <span class="smell-badge-small smell-badge-info">{summary["info_count"]}</span>
        </span>
    </div>
    <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid var(--gray-200);">
    <div class="panel-stat">
        <span class="panel-stat-label">LOC Range</span>
        <span class="panel-stat-value">{summary["min_loc"]} - {summary["max_loc"]}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Median LOC</span>
        <span class="panel-stat-value">{summary["median_loc"]}</span>
    </div>
    <div class="panel-stat">
        <span class="panel-stat-label">Circular Dependencies</span>
        <span class="panel-stat-value {circular_class}">{summary["circular_dependencies"]}</span>
    </div>
</div>"""

    def _generate_detail_panel(self) -> str:
        """Generate the node detail panel (initially hidden).

        Returns:
            HTML for the detail panel
        """
        return """<div class="dashboard-panel hidden" id="node-detail-panel" role="dialog" aria-labelledby="detail-panel-title">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <h3 id="detail-panel-title">📄 Node Details</h3>
        <button class="detail-close-button filter-button" aria-label="Close detail panel" style="padding: 0.25rem 0.5rem;">✕</button>
    </div>
    <div id="node-detail-content">
        <p style="color: var(--gray-600); font-size: 0.875rem; text-align: center; padding: 2rem 0;">
            Click a node to view details
        </p>
    </div>
</div>"""

    def _generate_legend_with_counts(self, summary: dict) -> str:
        """Generate collapsible legend with node counts.

        Args:
            summary: Summary statistics with distribution data

        Returns:
            HTML for the legend section
        """
        complexity_dist = summary["complexity_distribution"]
        smell_dist = summary["smell_distribution"]

        return f"""<div class="d3-legend-container">
    <button class="legend-toggle" onclick="toggleLegend()">
        <span>Legend</span>
        <span id="legend-toggle-icon">▼</span>
    </button>
    <div class="d3-legend legend-content" id="legend-content">
        <div class="legend-section">
            <div class="legend-title">Complexity (Fill)</div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #f3f4f6; border: 1px solid #e5e7eb;"></div>
                <span>0-5 (Low)<span class="count-badge">{complexity_dist["low"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #9ca3af;"></div>
                <span>6-10 (Moderate)<span class="count-badge">{complexity_dist["moderate"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #4b5563;"></div>
                <span>11-20 (High)<span class="count-badge">{complexity_dist["high"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #1f2937;"></div>
                <span>21-30 (Very High)<span class="count-badge">{complexity_dist["very_high"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #111827;"></div>
                <span>31+ (Critical)<span class="count-badge">{complexity_dist["critical"]}</span></span>
            </div>
        </div>
        <div class="legend-section">
            <div class="legend-title">Code Smells (Border)</div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 1px solid #e5e7eb;"></div>
                <span>None<span class="count-badge">{smell_dist["none"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 2px solid #fca5a5;"></div>
                <span>Info<span class="count-badge">{smell_dist["info"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 3px solid #f87171;"></div>
                <span>Warning<span class="count-badge">{smell_dist["warning"]}</span></span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 4px solid #ef4444;"></div>
                <span>Error<span class="count-badge">{smell_dist["error"]}</span></span>
            </div>
        </div>
        <div class="legend-section">
            <div class="legend-title">Dependencies (Edges)</div>
            <div class="legend-item">
                <div class="legend-line" style="background: #64748b;"></div>
                <span>Normal</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background: #dc2626;"></div>
                <span>Circular</span>
            </div>
        </div>
        <div class="legend-section">
            <div class="legend-title">Size</div>
            <div class="legend-item">
                <div class="legend-circle" style="width: 8px; height: 8px; background: #9ca3af;"></div>
                <span>Fewer lines</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="width: 16px; height: 16px; background: #9ca3af;"></div>
                <span>More lines</span>
            </div>
        </div>
    </div>
</div>"""

    def _generate_complexity_chart(self, export: AnalysisExport) -> str:
        """Generate complexity distribution chart placeholder.

        Args:
            export: Analysis export data

        Returns:
            HTML section with canvas for Chart.js
        """
        return """<section class="card">
    <h2>📊 Complexity Distribution</h2>
    <div class="chart-container">
        <canvas id="complexityChart"></canvas>
    </div>
</section>"""

    def _generate_grade_distribution(self, export: AnalysisExport) -> str:
        """Generate grade distribution table.

        Args:
            export: Analysis export with file data

        Returns:
            HTML section with grade breakdown
        """
        # Calculate grades from files
        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for f in export.files:
            grade = self._get_grade(f.cognitive_complexity)
            grades[grade] += 1

        total = sum(grades.values()) or 1

        rows = []
        for grade, count in grades.items():
            pct = (count / total) * 100
            rows.append(
                f"""<tr>
                <td><span class="grade-badge grade-{grade.lower()}">{grade}</span></td>
                <td>{count:,}</td>
                <td>{pct:.1f}%</td>
            </tr>"""
            )

        return f"""<section class="card">
    <h2>🎯 Grade Distribution</h2>
    <table>
        <thead>
            <tr><th>Grade</th><th>Count</th><th>Percentage</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</section>"""

    def _generate_smells_section(self, export: AnalysisExport) -> str:
        """Generate code smells section.

        Args:
            export: Analysis export with smell data

        Returns:
            HTML section with top code smells
        """
        all_smells = []
        for f in export.files:
            for smell in f.smells:
                all_smells.append((f.path, smell))

        # Sort by severity
        severity_order = {"error": 0, "warning": 1, "info": 2}
        all_smells.sort(key=lambda x: severity_order.get(x[1].severity, 3))

        # Limit to top 20
        top_smells = all_smells[:20]

        if not top_smells:
            return """<section class="card">
    <h2>🔍 Code Smells</h2>
    <p>No code smells detected! 🎉</p>
</section>"""

        rows = []
        for path, smell in top_smells:
            severity_class = {
                "error": "grade-f",
                "warning": "grade-d",
                "info": "grade-b",
            }.get(smell.severity, "")
            rows.append(
                f"""<tr>
                <td><span class="grade-badge {severity_class}">{smell.severity}</span></td>
                <td>{smell.smell_type}</td>
                <td>{path}:{smell.line}</td>
                <td>{smell.message}</td>
            </tr>"""
            )

        return f"""<section class="card">
    <h2>🔍 Code Smells ({export.summary.total_smells:,} total)</h2>
    <table>
        <thead>
            <tr><th>Severity</th><th>Type</th><th>Location</th><th>Message</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</section>"""

    def _generate_files_table(self, export: AnalysisExport) -> str:
        """Generate files table sorted by complexity.

        Args:
            export: Analysis export with file data

        Returns:
            HTML section with top files by complexity
        """
        # Sort by complexity descending, limit to top 20
        sorted_files = sorted(
            export.files, key=lambda f: f.cognitive_complexity, reverse=True
        )[:20]

        rows = []
        for f in sorted_files:
            grade = self._get_grade(f.cognitive_complexity)
            rows.append(
                f"""<tr>
                <td>{f.path}</td>
                <td><span class="grade-badge grade-{grade.lower()}">{grade}</span></td>
                <td>{f.cognitive_complexity}</td>
                <td>{f.cyclomatic_complexity}</td>
                <td>{f.lines_of_code:,}</td>
                <td>{f.function_count}</td>
                <td>{len(f.smells)}</td>
            </tr>"""
            )

        return f"""<section class="card">
    <h2>📁 Top Files by Complexity</h2>
    <table>
        <thead>
            <tr>
                <th>File</th>
                <th>Grade</th>
                <th>Cognitive</th>
                <th>Cyclomatic</th>
                <th>LOC</th>
                <th>Functions</th>
                <th>Smells</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</section>"""

    def _generate_dependencies_section(self, export: AnalysisExport) -> str:
        """Generate dependencies section.

        Args:
            export: Analysis export with dependency data

        Returns:
            HTML section with dependency analysis (empty if no data)
        """
        if not export.dependencies:
            return ""

        deps = export.dependencies

        circular_html = ""
        if deps.circular_dependencies:
            cycles = []
            for cycle in deps.circular_dependencies[:10]:
                cycles.append(f"<li>{' → '.join(cycle.cycle)}</li>")
            circular_html = f"""
    <h3>⚠️ Circular Dependencies ({len(deps.circular_dependencies)})</h3>
    <ul>{"".join(cycles)}</ul>"""

        return f"""<section class="card">
    <h2>🔗 Dependencies</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(deps.edges):,}</div>
            <div class="stat-label">Total Imports</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(deps.circular_dependencies)}</div>
            <div class="stat-label">Circular Dependencies</div>
        </div>
    </div>
    {circular_html}
</section>"""

    def _generate_trends_section(self, export: AnalysisExport) -> str:
        """Generate trends section.

        Args:
            export: Analysis export with trend data

        Returns:
            HTML section with trends (empty if no data)
        """
        if not export.trends or not export.trends.metrics:
            return ""

        rows = []
        for trend in export.trends.metrics:
            direction_icon = {
                "improving": "📈",
                "worsening": "📉",
                "stable": "➡️",
            }.get(trend.trend_direction, "➡️")

            change = (
                f"{trend.change_percent:+.1f}%"
                if trend.change_percent is not None
                else "N/A"
            )

            rows.append(
                f"""<tr>
                <td>{trend.metric_name}</td>
                <td>{trend.current_value:.1f}</td>
                <td>{change}</td>
                <td>{direction_icon} {trend.trend_direction}</td>
            </tr>"""
            )

        return f"""<section class="card">
    <h2>📈 Trends</h2>
    <table>
        <thead>
            <tr><th>Metric</th><th>Current</th><th>Change</th><th>Trend</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</section>"""

    def _generate_footer(self, export: AnalysisExport) -> str:
        """Generate report footer.

        Args:
            export: Analysis export with metadata

        Returns:
            HTML footer section
        """
        return f"""<footer>
    <p>Generated by mcp-vector-search v{export.metadata.tool_version}</p>
    <p>Schema version: {export.metadata.version}</p>
</footer>
</div>"""

    def _generate_scripts(self, export: AnalysisExport) -> str:
        """Generate JavaScript for charts and interactivity.

        Args:
            export: Analysis export for chart data

        Returns:
            <script> block with JavaScript
        """
        # Calculate grade distribution for chart
        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for f in export.files:
            grade = self._get_grade(f.cognitive_complexity)
            grades[grade] += 1

        return f"""<script>
// ===== ACCESSIBILITY CONTROLS =====
(function() {{
    const contrastButton = document.getElementById('toggle-contrast');
    const motionButton = document.getElementById('toggle-reduced-motion');

    // High contrast toggle
    contrastButton?.addEventListener('click', () => {{
        const isActive = document.body.classList.toggle('high-contrast');
        contrastButton.classList.toggle('active', isActive);
        contrastButton.setAttribute('aria-pressed', isActive);
        announceToScreenReader(isActive ? 'High contrast mode enabled' : 'High contrast mode disabled');
    }});

    // Reduced motion toggle
    motionButton?.addEventListener('click', () => {{
        const isActive = document.body.classList.toggle('reduced-motion');
        motionButton.classList.toggle('active', isActive);
        motionButton.setAttribute('aria-pressed', isActive);
        announceToScreenReader(isActive ? 'Reduced motion enabled' : 'Reduced motion disabled');
    }});

    // Respect user's prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
        document.body.classList.add('reduced-motion');
        motionButton?.classList.add('active');
        motionButton?.setAttribute('aria-pressed', 'true');
    }}
}})();

// Screen reader announcement helper
function announceToScreenReader(message) {{
    const announcer = document.getElementById('sr-announcements');
    if (announcer) {{
        announcer.textContent = message;
        setTimeout(() => announcer.textContent = '', 100);
    }}
}}

// ===== EXPORT FUNCTIONS =====
function exportAsPNG() {{
    try {{
        const svg = document.getElementById('d3-graph');
        if (!svg) throw new Error('Graph not found');

        // Create canvas
        const canvas = document.createElement('canvas');
        const bbox = svg.getBBox();
        canvas.width = bbox.width;
        canvas.height = bbox.height;
        const ctx = canvas.getContext('2d');

        // Serialize SVG to data URL
        const svgData = new XMLSerializer().serializeToString(svg);
        const svgBlob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
        const url = URL.createObjectURL(svgBlob);

        // Load as image and draw to canvas
        const img = new Image();
        img.onload = () => {{
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            // Download
            canvas.toBlob(blob => {{
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'dependency-graph.png';
                a.click();
                URL.revokeObjectURL(url);
                announceToScreenReader('Graph exported as PNG');
            }});
        }};
        img.src = url;
    }} catch (error) {{
        console.error('PNG export failed:', error);
        alert('Failed to export PNG: ' + error.message);
    }}
}}

function exportAsSVG() {{
    try {{
        const svg = document.getElementById('d3-graph');
        if (!svg) throw new Error('Graph not found');

        const svgData = new XMLSerializer().serializeToString(svg);
        const blob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'dependency-graph.svg';
        a.click();
        URL.revokeObjectURL(url);
        announceToScreenReader('Graph exported as SVG');
    }} catch (error) {{
        console.error('SVG export failed:', error);
        alert('Failed to export SVG: ' + error.message);
    }}
}}

function copyShareLink() {{
    try {{
        // Encode current filter state in URL hash
        const filterState = window.graphState?.filterState || {{}};
        const hash = '#filters=' + btoa(JSON.stringify(filterState));
        const url = window.location.href.split('#')[0] + hash;

        navigator.clipboard.writeText(url).then(() => {{
            announceToScreenReader('Link copied to clipboard');
            alert('Shareable link copied to clipboard!');
        }});
    }} catch (error) {{
        console.error('Copy link failed:', error);
        alert('Failed to copy link: ' + error.message);
    }}
}}

// Load filters from URL hash
function loadFiltersFromURL() {{
    const hash = window.location.hash;
    if (hash.startsWith('#filters=')) {{
        try {{
            const encoded = hash.substring(9);
            const filterState = JSON.parse(atob(encoded));
            if (window.graphState) {{
                // Update filter UI elements
                if (filterState.grades) {{
                    document.querySelectorAll('[id^="filter-grade-"]').forEach(cb => {{
                        cb.checked = filterState.grades.includes(cb.value);
                    }});
                }}
                if (filterState.smells) {{
                    document.querySelectorAll('[id^="filter-smell-"]').forEach(cb => {{
                        cb.checked = filterState.smells.includes(cb.value);
                    }});
                }}
                if (filterState.modules) {{
                    const moduleSelect = document.getElementById('filter-module');
                    if (moduleSelect) {{
                        Array.from(moduleSelect.options).forEach(opt => {{
                            opt.selected = filterState.modules.includes(opt.value);
                        }});
                    }}
                }}
                if (filterState.search) {{
                    const searchInput = document.getElementById('filter-search');
                    if (searchInput) {{
                        searchInput.value = filterState.search;
                    }}
                }}

                // Trigger filter application
                const event = new Event('change');
                document.querySelector('[id^="filter-grade-"]')?.dispatchEvent(event);

                announceToScreenReader('Filters loaded from shared link');
            }}
        }} catch (error) {{
            console.error('Failed to load filters from URL:', error);
        }}
    }}
}}

// Initialize D3 Graph with error handling and performance optimizations
(function() {{
    const loadingEl = document.getElementById('graph-loading');
    const errorEl = document.getElementById('graph-error');
    const errorMsgEl = document.getElementById('graph-error-message');

    function showError(message) {{
        if (loadingEl) loadingEl.style.display = 'none';
        if (errorEl) errorEl.style.display = 'block';
        if (errorMsgEl) errorMsgEl.textContent = message;
        announceToScreenReader('Graph error: ' + message);
    }}

    function showLoading() {{
        if (loadingEl) loadingEl.style.display = 'block';
        if (errorEl) errorEl.style.display = 'none';
    }}

    function hideLoading() {{
        if (loadingEl) loadingEl.style.display = 'none';
    }}

    // Check if D3 loaded
    if (typeof d3 === 'undefined') {{
        showError('D3.js library failed to load. Please check your internet connection.');
        return;
    }}

    showLoading();

    const dataScript = document.getElementById('d3-graph-data');
    if (!dataScript) {{
        showError('Graph data not found.');
        return;
    }}

    let graphData;
    try {{
        graphData = JSON.parse(dataScript.textContent);
    }} catch (error) {{
        showError('Failed to parse graph data: ' + error.message);
        return;
    }}

    if (!graphData.nodes || graphData.nodes.length === 0) {{
        showError('No files in analysis. Add some code files to see the dependency graph.');
        return;
    }}

    hideLoading();

    const svg = d3.select("#d3-graph");
    const container = document.getElementById("d3-graph-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr("viewBox", [0, 0, width, height]);

    // ===== PERFORMANCE OPTIMIZATIONS =====
    const nodeCount = graphData.nodes.length;
    const LARGE_GRAPH_THRESHOLD = 100;
    const isLargeGraph = nodeCount > LARGE_GRAPH_THRESHOLD;

    // Disable animations for large graphs
    const animationDuration = isLargeGraph ? 0 :
        (document.body.classList.contains('reduced-motion') ? 0 : 600);

    // Throttle simulation updates for smooth 60fps during drag
    let lastTickTime = 0;
    const TICK_THROTTLE_MS = 16; // ~60fps

    // Debounce filter changes
    let filterDebounceTimer;
    const FILTER_DEBOUNCE_MS = 150;

    // LOD (Level of Detail) state
    let currentZoom = 1.0;
    const LOD_ZOOM_THRESHOLD_LOW = 0.5;
    const LOD_ZOOM_THRESHOLD_HIGH = 1.5;

    // Helper functions for visual encoding
    const complexityColor = (complexity) => {{
        if (complexity <= 5) return '#f3f4f6';
        if (complexity <= 10) return '#9ca3af';
        if (complexity <= 20) return '#4b5563';
        if (complexity <= 30) return '#1f2937';
        return '#111827';
    }};

    const smellBorder = (severity) => {{
        const borders = {{
            'none': {{ color: '#e5e7eb', width: 1 }},
            'info': {{ color: '#fca5a5', width: 2 }},
            'warning': {{ color: '#f87171', width: 3 }},
            'error': {{ color: '#ef4444', width: 4 }},
            'critical': {{ color: '#dc2626', width: 5 }}
        }};
        return borders[severity] || borders['none'];
    }};

    // Edge color scale based on coupling strength
    const maxCoupling = d3.max(graphData.links, d => d.coupling) || 1;
    const edgeColorScale = d3.scaleLinear()
        .domain([1, maxCoupling])
        .range(['#d1d5db', '#4b5563']);

    // Size scale for LOC (min 8px, max 40px)
    const maxLoc = d3.max(graphData.nodes, d => d.loc) || 100;
    const sizeScale = d3.scaleSqrt()
        .domain([0, maxLoc])
        .range([8, 40]);

    // Edge thickness scale (min 1px, max 4px)
    const edgeScale = d3.scaleLinear()
        .domain([1, maxCoupling])
        .range([1, 4]);

    // Force simulation
    const simulation = d3.forceSimulation(graphData.nodes)
        .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(d => sizeScale(d.loc) + 5));

    // Zoom behavior with LOD (Level of Detail)
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {{
            g.attr("transform", event.transform);

            // Update current zoom level
            const newZoom = event.transform.k;
            if (Math.abs(newZoom - currentZoom) > 0.1) {{
                currentZoom = newZoom;
                updateLOD(currentZoom);
            }}
        }});

    svg.call(zoom);

    // LOD update function
    function updateLOD(zoomLevel) {{
        // Use requestAnimationFrame for smooth rendering
        requestAnimationFrame(() => {{
            if (zoomLevel < LOD_ZOOM_THRESHOLD_LOW) {{
                // Zoom out: Hide labels, simplify nodes
                node.selectAll("text").style("display", "none");
                node.selectAll("circle")
                    .style("stroke-width", d => Math.min(smellBorder(d.smell_severity).width, 2));
                // Hide complexity details
                node.selectAll(".complexity-label").style("display", "none");
            }} else if (zoomLevel > LOD_ZOOM_THRESHOLD_HIGH) {{
                // Zoom in: Show additional details
                node.selectAll("text").style("display", "block");
                node.selectAll("circle")
                    .style("stroke-width", d => smellBorder(d.smell_severity).width);
                // Show complexity number in node
                node.each(function(d) {{
                    const nodeGroup = d3.select(this);
                    if (!nodeGroup.select(".complexity-label").node()) {{
                        nodeGroup.append("text")
                            .attr("class", "complexity-label")
                            .attr("dy", 4)
                            .attr("text-anchor", "middle")
                            .style("font-size", "10px")
                            .style("font-weight", "bold")
                            .style("fill", d.complexity > 20 ? "white" : "#374151")
                            .style("pointer-events", "none")
                            .text(d.complexity);
                    }}
                }});
            }} else {{
                // Normal zoom: Show labels, full styling
                node.selectAll("text").style("display", "block");
                node.selectAll("circle")
                    .style("stroke-width", d => smellBorder(d.smell_severity).width);
                node.selectAll(".complexity-label").style("display", "none");
            }}
        }});
    }}

    const g = svg.append("g");

    // Define arrowhead markers
    svg.append("defs").selectAll("marker")
        .data(["arrowhead", "arrowhead-circular"])
        .join("marker")
        .attr("id", d => d)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 20)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("class", d => d);

    // Draw module cluster hulls (background layer)
    const hullGroup = g.append("g").attr("class", "hulls");

    // Draw edges with curves and arrowheads
    const linkGroup = g.append("g");
    const link = linkGroup.selectAll("path")
        .data(graphData.links)
        .join("path")
        .attr("class", d => d.circular ? "link link-circular" : "link")
        .attr("stroke", d => d.circular ? "#dc2626" : edgeColorScale(d.coupling))
        .attr("stroke-width", d => edgeScale(d.coupling))
        .attr("marker-end", d => d.circular ? "url(#arrowhead-circular)" : "url(#arrowhead)");

    // Draw nodes with entrance animation
    const nodeGroup = g.append("g");
    const node = nodeGroup.selectAll("g")
        .data(graphData.nodes)
        .join("g")
        .attr("opacity", 0)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Node entrance animation (conditional based on graph size and motion preference)
    if (animationDuration > 0) {{
        node.transition()
            .duration(animationDuration)
            .delay((d, i) => i * 30)
            .attr("opacity", 1);
    }} else {{
        node.attr("opacity", 1);
    }}

    // Node circles with complexity fill and smell border
    node.append("circle")
        .attr("r", animationDuration > 0 ? 0 : d => sizeScale(d.loc))
        .attr("fill", d => complexityColor(d.complexity))
        .attr("stroke", d => smellBorder(d.smell_severity).color)
        .attr("stroke-width", d => smellBorder(d.smell_severity).width)
        .style("filter", d => d.smell_severity === 'critical' ? 'drop-shadow(0 0 4px #dc2626)' : null)
        .attr("tabindex", 0)
        .attr("role", "button")
        .attr("aria-label", d => `${{d.label}}, complexity ${{d.complexity}}, ${{d.smell_count}} code smells`);

    if (animationDuration > 0) {{
        node.selectAll("circle")
            .transition()
            .duration(animationDuration)
            .delay((d, i) => i * 30)
            .attr("r", d => sizeScale(d.loc));
    }}

    // Node labels
    node.append("text")
        .text(d => d.label)
        .attr("class", "node-label")
        .attr("dy", d => sizeScale(d.loc) + 12);

    // Tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "d3-tooltip");

    // Enhanced tooltip rendering
    const renderTooltip = (d, event) => {{
        const getGrade = (complexity) => {{
            if (complexity <= 5) return {{ grade: 'A', class: 'grade-a', color: '#22c55e' }};
            if (complexity <= 10) return {{ grade: 'B', class: 'grade-b', color: '#3b82f6' }};
            if (complexity <= 20) return {{ grade: 'C', class: 'grade-c', color: '#f59e0b' }};
            if (complexity <= 30) return {{ grade: 'D', class: 'grade-d', color: '#f97316' }};
            return {{ grade: 'F', class: 'grade-f', color: '#ef4444' }};
        }};

        const grade = getGrade(d.complexity);
        const smellColor = smellBorder(d.smell_severity).color;

        // Calculate complexity bar percentage (max 50 for visualization)
        const complexityPct = Math.min((d.complexity / 50) * 100, 100);
        const complexityBarColor = grade.color;

        // Count imports/imported-by
        const incomingCount = graphData.links.filter(l => l.target.id === d.id).length;
        const outgoingCount = d.imports ? d.imports.length : 0;

        tooltip.html(`
            <div class="tooltip-header">${{d.label}}</div>
            <div class="tooltip-subtitle">Module: ${{d.module}}</div>

            <div class="tooltip-metric">
                <span class="tooltip-metric-label">Complexity</span>
                <span class="tooltip-metric-value">
                    ${{d.complexity}}
                    <span class="tooltip-badge ${{grade.class}}">${{grade.grade}}</span>
                </span>
            </div>
            <div class="tooltip-bar">
                <div class="tooltip-bar-fill" style="width: ${{complexityPct}}%; background: ${{complexityBarColor}};"></div>
            </div>

            <div class="tooltip-section">
                <div class="tooltip-metric">
                    <span class="tooltip-metric-label">Lines of Code</span>
                    <span class="tooltip-metric-value">${{d.loc}}</span>
                </div>
                <div class="tooltip-metric">
                    <span class="tooltip-metric-label">Code Smells</span>
                    <span class="tooltip-metric-value" style="color: ${{smellColor}}">
                        ${{d.smell_count}}
                        <span class="tooltip-badge" style="background: ${{smellColor}}; color: white;">${{d.smell_severity}}</span>
                    </span>
                </div>
            </div>

            <div class="tooltip-section">
                <div class="tooltip-metric">
                    <span class="tooltip-metric-label">Imports</span>
                    <span class="tooltip-metric-value">${{outgoingCount}}</span>
                </div>
                <div class="tooltip-metric">
                    <span class="tooltip-metric-label">Imported by</span>
                    <span class="tooltip-metric-value">${{incomingCount}}</span>
                </div>
            </div>
        `);
    }};

    // Hover highlighting with connected nodes
    node.on("mouseenter", (event, d) => {{
        // Find connected nodes
        const connectedNodes = new Set([d.id]);
        graphData.links.forEach(link => {{
            if (link.source.id === d.id) connectedNodes.add(link.target.id);
            if (link.target.id === d.id) connectedNodes.add(link.source.id);
        }});

        // Dim unconnected nodes
        node.classed("node-dimmed", n => !connectedNodes.has(n.id))
            .classed("node-highlighted", n => connectedNodes.has(n.id));

        // Dim unconnected edges
        link.classed("link-dimmed", l => l.source.id !== d.id && l.target.id !== d.id)
            .classed("link-highlighted", l => l.source.id === d.id || l.target.id === d.id);

        // Show enhanced tooltip
        renderTooltip(d, event);
        tooltip.transition().duration(200).style("opacity", 1);
    }})
    .on("mousemove", (event) => {{
        // Tooltip follows cursor
        tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px");
    }})
    .on("mouseleave", () => {{
        // Remove dimming
        node.classed("node-dimmed", false).classed("node-highlighted", false);
        link.classed("link-dimmed", false).classed("link-highlighted", false);

        tooltip.transition().duration(500).style("opacity", 0);
    }})
    .on("click", (event, d) => {{
        event.stopPropagation();
        showNodeDetails(d, graphData);
    }});

    // Function to update module cluster hulls
    function updateHulls() {{
        if (!graphData.modules || graphData.modules.length === 0) return;

        const hullData = graphData.modules.map(module => {{
            const moduleNodes = graphData.nodes.filter(n =>
                module.node_ids.includes(n.id)
            );
            if (moduleNodes.length < 2) return null;

            // Calculate convex hull points
            const points = moduleNodes.map(n => [n.x, n.y]);
            const hull = d3.polygonHull(points);

            return hull ? {{ points: hull, color: module.color, name: module.name }} : null;
        }}).filter(h => h !== null);

        hullGroup.selectAll("path")
            .data(hullData)
            .join("path")
            .attr("class", "module-hull")
            .attr("d", d => {{
                // Add padding around hull
                const centroid = d3.polygonCentroid(d.points);
                const paddedPoints = d.points.map(p => {{
                    const dx = p[0] - centroid[0];
                    const dy = p[1] - centroid[1];
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const padding = 30;
                    return [
                        p[0] + (dx / dist) * padding,
                        p[1] + (dy / dist) * padding
                    ];
                }});
                return "M" + paddedPoints.join("L") + "Z";
            }})
            .attr("fill", d => d.color)
            .attr("stroke", d => d.color);
    }}

    // Simulation tick with throttling for performance
    simulation.on("tick", () => {{
        const now = Date.now();
        if (now - lastTickTime < TICK_THROTTLE_MS) {{
            return; // Skip this tick to maintain 60fps
        }}
        lastTickTime = now;

        // Use requestAnimationFrame for smooth rendering
        requestAnimationFrame(() => {{
            // Update curved edges
            link.attr("d", d => {{
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy) * 2;
                return `M${{d.source.x}},${{d.source.y}}A${{dr}},${{dr}} 0 0,1 ${{d.target.x}},${{d.target.y}}`;
            }});

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

            // Update hulls periodically (not every tick for performance)
            if (simulation.alpha() < 0.1) {{
                updateHulls();
            }}
        }});
    }});

    // Drag functions with throttling
    let dragThrottle;
    function dragstarted(event) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
        announceToScreenReader(`Dragging ${{event.subject.label}}`);
    }}

    function dragged(event) {{
        // Throttle drag updates for performance
        if (dragThrottle) clearTimeout(dragThrottle);
        dragThrottle = setTimeout(() => {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}, TICK_THROTTLE_MS);
    }}

    function dragended(event) {{
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
        announceToScreenReader(`Released ${{event.subject.label}}`);
    }}
    // ===== FILTER FUNCTIONALITY =====

    // Get complexity grade from complexity value
    const getComplexityGrade = (complexity) => {{
        if (complexity <= 5) return 'A';
        if (complexity <= 10) return 'B';
        if (complexity <= 20) return 'C';
        if (complexity <= 30) return 'D';
        return 'F';
    }};

    // Filter state
    const filterState = {{
        grades: new Set(['A', 'B', 'C', 'D', 'F']),
        smells: new Set(['none', 'info', 'warning', 'error']),
        modules: new Set(['']), // Empty string means "All Modules"
        search: ''
    }};

    // Apply filters to nodes and edges (with debounce)
    const applyFilters = () => {{
        // Debounce for performance
        if (filterDebounceTimer) clearTimeout(filterDebounceTimer);
        filterDebounceTimer = setTimeout(() => {{
            const visibleNodeIds = new Set();

            node.each((d, i, nodes) => {{
            const nodeGrade = getComplexityGrade(d.complexity);
            const nodeSmell = d.smell_severity || 'none';
            const nodeModule = d.module || '';

            // Check if node passes all filters
            const passesGrade = filterState.grades.has(nodeGrade);
            const passesSmell = filterState.smells.has(nodeSmell);
            const passesModule = filterState.modules.has('') || filterState.modules.has(nodeModule);
            const passesSearch = !filterState.search ||
                d.id.toLowerCase().includes(filterState.search.toLowerCase()) ||
                d.label.toLowerCase().includes(filterState.search.toLowerCase());

            const isVisible = passesGrade && passesSmell && passesModule && passesSearch;

            // Apply filtering class
            d3.select(nodes[i])
                .classed("node-filtered", !isVisible)
                .classed("node-search-highlight", passesSearch && filterState.search.length > 0);

            if (isVisible) {{
                visibleNodeIds.add(d.id);
            }}
        }});

            // Filter edges (only show if both source and target are visible)
            link.classed("link-filtered", l =>
                !visibleNodeIds.has(l.source.id) || !visibleNodeIds.has(l.target.id)
            );

            // Update hulls to only include visible nodes
            updateHulls();

            // Screen reader announcement
            const visibleCount = visibleNodeIds.size;
            const totalCount = graphData.nodes.length;
            announceToScreenReader(`Showing ${{visibleCount}} of ${{totalCount}} files`);
        }}, FILTER_DEBOUNCE_MS);
    }};

    // Update hulls to exclude filtered nodes
    const originalUpdateHulls = updateHulls;
    updateHulls = function() {{
        if (!graphData.modules || graphData.modules.length === 0) return;

        const hullData = graphData.modules.map(module => {{
            const moduleNodes = graphData.nodes.filter(n =>
                module.node_ids.includes(n.id) &&
                !d3.select(`[data-node-id="${{n.id}}"]`).classed("node-filtered")
            );
            if (moduleNodes.length < 2) return null;

            const points = moduleNodes.map(n => [n.x, n.y]);
            const hull = d3.polygonHull(points);

            return hull ? {{ points: hull, color: module.color, name: module.name }} : null;
        }}).filter(h => h !== null);

        hullGroup.selectAll("path")
            .data(hullData)
            .join("path")
            .attr("class", "module-hull")
            .attr("d", d => {{
                const centroid = d3.polygonCentroid(d.points);
                const paddedPoints = d.points.map(p => {{
                    const dx = p[0] - centroid[0];
                    const dy = p[1] - centroid[1];
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const padding = 30;
                    return [
                        p[0] + (dx / dist) * padding,
                        p[1] + (dy / dist) * padding
                    ];
                }});
                return "M" + paddedPoints.join("L") + "Z";
            }})
            .attr("fill", d => d.color)
            .attr("stroke", d => d.color);
    }};

    // Add data-node-id attribute to nodes for filtering
    node.attr("data-node-id", d => d.id);

    // Setup filter event listeners
    document.querySelectorAll('[id^="filter-grade-"]').forEach(checkbox => {{
        checkbox.addEventListener('change', (e) => {{
            const grade = e.target.value;
            if (e.target.checked) {{
                filterState.grades.add(grade);
            }} else {{
                filterState.grades.delete(grade);
            }}
            applyFilters();
        }});
    }});

    document.querySelectorAll('[id^="filter-smell-"]').forEach(checkbox => {{
        checkbox.addEventListener('change', (e) => {{
            const smell = e.target.value;
            if (e.target.checked) {{
                filterState.smells.add(smell);
            }} else {{
                filterState.smells.delete(smell);
            }}
            applyFilters();
        }});
    }});

    const moduleSelect = document.getElementById('filter-module');
    if (moduleSelect) {{
        moduleSelect.addEventListener('change', (e) => {{
            filterState.modules = new Set(
                Array.from(e.target.selectedOptions).map(opt => opt.value)
            );
            applyFilters();
        }});
    }}

    const searchInput = document.getElementById('filter-search');
    if (searchInput) {{
        searchInput.addEventListener('input', (e) => {{
            filterState.search = e.target.value;
            applyFilters();
        }});
    }}

    // Store references for keyboard navigation and filter state
    window.graphState = {{
        node,
        graphData,
        currentFocusIndex: -1,
        visibleNodes: [],
        filterState: filterState
    }};

    // Load filters from URL if present
    loadFiltersFromURL();

    // Announce graph loaded
    announceToScreenReader(`Dependency graph loaded with ${{nodeCount}} files`);

    // ===== KEYBOARD NAVIGATION =====

    // Get visible (non-filtered) nodes
    const getVisibleNodes = () => {{
        const visible = [];
        node.each((d, i, nodes) => {{
            if (!d3.select(nodes[i]).classed("node-filtered")) {{
                visible.push({{ data: d, element: nodes[i], index: i }});
            }}
        }});
        return visible;
    }};

    // Focus on a specific node
    const focusNode = (nodeIndex) => {{
        const visibleNodes = getVisibleNodes();
        if (visibleNodes.length === 0) return;

        // Clamp index
        nodeIndex = Math.max(0, Math.min(nodeIndex, visibleNodes.length - 1));
        window.graphState.currentFocusIndex = nodeIndex;

        // Remove previous focus
        node.classed("node-focused", false);

        // Add focus to current node
        const focusedNode = visibleNodes[nodeIndex];
        d3.select(focusedNode.element).classed("node-focused", true);

        // Scroll node into view (center of SVG)
        const transform = d3.zoomTransform(svg.node());
        const newTransform = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(transform.k)
            .translate(-focusedNode.data.x, -focusedNode.data.y);

        svg.transition()
            .duration(500)
            .call(zoom.transform, newTransform);

        return focusedNode;
    }};

    // Navigate to connected nodes
    const navigateToConnected = (direction) => {{
        const visibleNodes = getVisibleNodes();
        if (window.graphState.currentFocusIndex < 0 || visibleNodes.length === 0) return;

        const currentNode = visibleNodes[window.graphState.currentFocusIndex];
        const connectedIds = new Set();

        graphData.links.forEach(link => {{
            if (link.source.id === currentNode.data.id) {{
                connectedIds.add(link.target.id);
            }}
            if (link.target.id === currentNode.data.id) {{
                connectedIds.add(link.source.id);
            }}
        }});

        // Find next connected visible node
        const connectedVisible = visibleNodes.filter(n => connectedIds.has(n.data.id));
        if (connectedVisible.length > 0) {{
            const nextIndex = visibleNodes.indexOf(connectedVisible[0]);
            focusNode(nextIndex);
        }}
    }};

    // Global keyboard handler
    document.addEventListener('keydown', (e) => {{
        // Only handle if not in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {{
            return;
        }}

        const visibleNodes = getVisibleNodes();

        switch(e.key) {{
            case 'Tab':
                e.preventDefault();
                // Tab through visible nodes
                const nextIndex = (window.graphState.currentFocusIndex + 1) % visibleNodes.length;
                focusNode(nextIndex);
                break;

            case 'Enter':
            case ' ':
                e.preventDefault();
                // Select focused node (show detail panel)
                if (window.graphState.currentFocusIndex >= 0 && visibleNodes.length > 0) {{
                    const focusedNode = visibleNodes[window.graphState.currentFocusIndex];
                    showNodeDetails(focusedNode.data, graphData);
                }}
                break;

            case 'Escape':
                e.preventDefault();
                // Clear selection and focus
                node.classed("node-focused", false);
                window.graphState.currentFocusIndex = -1;
                const detailPanel = document.getElementById('node-detail-panel');
                if (detailPanel) {{
                    detailPanel.classList.add('hidden');
                }}
                break;

            case 'ArrowRight':
            case 'ArrowDown':
                e.preventDefault();
                // Next node or connected node
                if (e.shiftKey) {{
                    navigateToConnected('next');
                }} else {{
                    focusNode(window.graphState.currentFocusIndex + 1);
                }}
                break;

            case 'ArrowLeft':
            case 'ArrowUp':
                e.preventDefault();
                // Previous node
                focusNode(window.graphState.currentFocusIndex - 1);
                break;
        }}
    }});
}})();

// Helper function to reset all filters
function resetFilters() {{
    // Reset checkboxes
    document.querySelectorAll('[id^="filter-grade-"]').forEach(cb => cb.checked = true);
    document.querySelectorAll('[id^="filter-smell-"]').forEach(cb => cb.checked = true);

    // Reset module select
    const moduleSelect = document.getElementById('filter-module');
    if (moduleSelect) {{
        Array.from(moduleSelect.options).forEach(opt => {{
            opt.selected = opt.value === '';
        }});
    }}

    // Reset search
    const searchInput = document.getElementById('filter-search');
    if (searchInput) {{
        searchInput.value = '';
    }}

    // Trigger change events to apply filters
    if (window.graphState && window.graphState.node) {{
        const event = new Event('change');
        document.querySelector('[id^="filter-grade-"]')?.dispatchEvent(event);
    }}
}}

// Helper function to clear node selection
function clearSelection() {{
    const detailPanel = document.getElementById('node-detail-panel');
    if (detailPanel) {{
        detailPanel.classList.add('hidden');
    }}

    if (window.graphState && window.graphState.node) {{
        window.graphState.node.classed("node-focused", false);
        window.graphState.currentFocusIndex = -1;
    }}
}}

// Helper function to show node details in detail panel with focus trap
function showNodeDetails(nodeData, graphData) {{
    const detailPanel = document.getElementById('node-detail-panel');
    const detailContent = document.getElementById('node-detail-content');

    // Store last focused element for restoration
    if (!window.lastFocusedElement) {{
        window.lastFocusedElement = document.activeElement;
    }}

    // Get complexity grade
    const getGrade = (complexity) => {{
        if (complexity <= 5) return {{ grade: 'A', class: 'grade-a' }};
        if (complexity <= 10) return {{ grade: 'B', class: 'grade-b' }};
        if (complexity <= 20) return {{ grade: 'C', class: 'grade-c' }};
        if (complexity <= 30) return {{ grade: 'D', class: 'grade-d' }};
        return {{ grade: 'F', class: 'grade-f' }};
    }};

    const complexityGrade = getGrade(nodeData.complexity);
    const cyclomaticGrade = getGrade(nodeData.cyclomatic_complexity);

    // Find incoming edges (imported-by)
    const incomingEdges = graphData.links.filter(link => link.target.id === nodeData.id);
    const importedBy = incomingEdges.map(link => link.source.id || link.source);

    // Outgoing edges are in nodeData.imports
    const imports = nodeData.imports || [];

    // Render smells
    let smellsHtml = '<p style="color: var(--gray-600); font-size: 0.875rem;">No smells detected</p>';
    if (nodeData.smells && nodeData.smells.length > 0) {{
        const smellItems = nodeData.smells.map(smell => {{
            const badgeClass = smell.severity === 'error' ? 'smell-badge-error' :
                              smell.severity === 'warning' ? 'smell-badge-warning' :
                              'smell-badge-info';
            return `
                <div class="panel-list-item">
                    <span class="smell-badge-small ${{badgeClass}}">${{smell.severity}}</span>
                    <strong>${{smell.type}}</strong> (line ${{smell.line}})<br/>
                    <span style="font-size: 0.75rem; color: var(--gray-600);">${{smell.message}}</span>
                </div>
            `;
        }}).join('');
        smellsHtml = `<ul class="panel-list">${{smellItems}}</ul>`;
    }}

    // Render imports
    let importsHtml = '<p style="color: var(--gray-600); font-size: 0.875rem;">No imports</p>';
    if (imports.length > 0) {{
        const importItems = imports.map(imp =>
            `<li class="panel-list-item" style="font-family: 'Monaco', 'Courier New', monospace; font-size: 0.75rem;">${{imp}}</li>`
        ).join('');
        importsHtml = `<ul class="panel-list">${{importItems}}</ul>`;
    }}

    // Render imported-by
    let importedByHtml = '<p style="color: var(--gray-600); font-size: 0.875rem;">Not imported by any files</p>';
    if (importedBy.length > 0) {{
        const importedByItems = importedBy.map(file =>
            `<li class="panel-list-item" style="font-family: 'Monaco', 'Courier New', monospace; font-size: 0.75rem;">${{file}}</li>`
        ).join('');
        importedByHtml = `<ul class="panel-list">${{importedByItems}}</ul>`;
    }}

    // Build detail HTML
    detailContent.innerHTML = `
        <div class="detail-header">
            <div class="detail-header-title">${{nodeData.label}}</div>
            <div class="detail-header-path">${{nodeData.id}}</div>
        </div>

        <div class="detail-section">
            <div class="detail-section-title">Metrics</div>
            <div class="panel-stat">
                <span class="panel-stat-label">Cognitive Complexity</span>
                <span class="panel-stat-value">
                    ${{nodeData.complexity}}
                    <span class="grade-badge ${{complexityGrade.class}}">${{complexityGrade.grade}}</span>
                </span>
            </div>
            <div class="panel-stat">
                <span class="panel-stat-label">Cyclomatic Complexity</span>
                <span class="panel-stat-value">
                    ${{nodeData.cyclomatic_complexity}}
                    <span class="grade-badge ${{cyclomaticGrade.class}}">${{cyclomaticGrade.grade}}</span>
                </span>
            </div>
            <div class="panel-stat">
                <span class="panel-stat-label">Lines of Code</span>
                <span class="panel-stat-value">${{nodeData.loc}}</span>
            </div>
            <div class="panel-stat">
                <span class="panel-stat-label">Functions</span>
                <span class="panel-stat-value">${{nodeData.function_count}}</span>
            </div>
            <div class="panel-stat">
                <span class="panel-stat-label">Classes</span>
                <span class="panel-stat-value">${{nodeData.class_count}}</span>
            </div>
        </div>

        <div class="detail-section">
            <div class="detail-section-title">Code Smells (${{nodeData.smell_count}})</div>
            ${{smellsHtml}}
        </div>

        <div class="detail-section">
            <div class="detail-section-title">Imports (${{imports.length}})</div>
            ${{importsHtml}}
        </div>

        <div class="detail-section">
            <div class="detail-section-title">Imported By (${{importedBy.length}})</div>
            ${{importedByHtml}}
        </div>
    `;

    // Show the panel
    detailPanel.classList.remove('hidden');
    detailPanel.classList.add('focus-trapped');

    // Focus first focusable element in panel
    setTimeout(() => {{
        const firstFocusable = detailPanel.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) {{
            firstFocusable.focus();
        }}
    }}, 100);

    // Screen reader announcement
    announceToScreenReader(`Showing details for ${{nodeData.label}}`);

    // Add close button handler for focus restoration
    const closeButton = detailPanel.querySelector('.detail-close-button');
    if (closeButton) {{
        closeButton.addEventListener('click', () => {{
            detailPanel.classList.add('hidden');
            detailPanel.classList.remove('focus-trapped');
            if (window.lastFocusedElement) {{
                window.lastFocusedElement.focus();
                window.lastFocusedElement = null;
            }}
        }}, {{ once: true }});
    }}
}}

// Helper function to toggle legend
function toggleLegend() {{
    const legendContent = document.getElementById('legend-content');
    const toggleIcon = document.getElementById('legend-toggle-icon');

    if (legendContent.classList.contains('collapsed')) {{
        legendContent.classList.remove('collapsed');
        toggleIcon.textContent = '▼';
        announceToScreenReader('Legend expanded');
    }} else {{
        legendContent.classList.add('collapsed');
        toggleIcon.textContent = '▶';
        announceToScreenReader('Legend collapsed');
    }}
}}

// Helper function to toggle stats panel
function toggleStatsPanel() {{
    const statsPanel = document.getElementById('stats-panel-container');
    const toggleButton = document.getElementById('stats-toggle');

    if (!statsPanel || !toggleButton) return;

    const isVisible = statsPanel.classList.toggle('visible');
    toggleButton.classList.toggle('active', isVisible);
    toggleButton.setAttribute('aria-expanded', isVisible);
    statsPanel.setAttribute('aria-hidden', !isVisible);

    // Update button text
    toggleButton.textContent = isVisible ? '✕ Hide Stats' : '📊 Show Stats';

    // Announce to screen readers
    announceToScreenReader(isVisible ? 'Statistics panel opened' : 'Statistics panel closed');
}}

// Initialize complexity chart
const ctx = document.getElementById('complexityChart');
if (ctx) {{
    new Chart(ctx, {{
        type: 'doughnut',
        data: {{
            labels: ['A (Excellent)', 'B (Good)', 'C (Acceptable)', 'D (Needs Work)', 'F (Refactor)'],
            datasets: [{{
                data: [{grades["A"]}, {grades["B"]}, {grades["C"]}, {grades["D"]}, {grades["F"]}],
                backgroundColor: ['#22c55e', '#3b82f6', '#f59e0b', '#f97316', '#ef4444'],
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ position: 'right' }}
            }}
        }}
    }});
}}

// Initialize syntax highlighting
hljs.highlightAll();
</script>"""

    @staticmethod
    def _get_grade(complexity: int) -> str:
        """Get letter grade from complexity score.

        Uses standard complexity thresholds:
        - A: 0-5 (Excellent)
        - B: 6-10 (Good)
        - C: 11-20 (Acceptable)
        - D: 21-30 (Needs work)
        - F: 31+ (Refactor required)

        Args:
            complexity: Cognitive complexity score

        Returns:
            Letter grade (A, B, C, D, or F)
        """
        if complexity <= 5:
            return "A"
        elif complexity <= 10:
            return "B"
        elif complexity <= 20:
            return "C"
        elif complexity <= 30:
            return "D"
        else:
            return "F"
