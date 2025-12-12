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

@media (max-width: 768px) {
    body { padding: 1rem; }
    .stats-grid { grid-template-columns: 1fr; }
    h1 { font-size: 1.5rem; }
}

/* D3 Graph Styles */
#d3-graph-container {
    position: relative;
    overflow: hidden;
    background: #fafafa;
    height: 600px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

#d3-graph {
    width: 100%;
    height: 100%;
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
    stroke: #64748b;
    stroke-opacity: 0.6;
}

.link-circular {
    stroke: #dc2626;
    stroke-opacity: 0.8;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { stroke-opacity: 0.8; }
    50% { stroke-opacity: 0.3; }
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
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    font-size: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    pointer-events: none;
    z-index: 1000;
    max-width: 300px;
    opacity: 0;
    transition: opacity 0.2s;
}

/* Legend */
.d3-legend {
    display: flex;
    gap: 24px;
    padding: 16px;
    background: white;
    border-top: 1px solid #e5e7eb;
    flex-wrap: wrap;
}

.legend-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.legend-title {
    font-weight: 600;
    font-size: 12px;
    color: var(--gray-700);
    margin-bottom: 4px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--gray-700);
}

.legend-circle {
    width: 16px;
    height: 16px;
    border-radius: 50%;
}

.legend-line {
    width: 24px;
    height: 2px;
}

@media (max-width: 768px) {
    #d3-graph-container { height: 400px; }
    .d3-legend { flex-direction: column; gap: 16px; }
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

        return f"""<section class="card">
    <h2>🔗 Interactive Dependency Graph</h2>
    <p style="color: var(--gray-700); margin-bottom: 1rem;">
        Explore file dependencies with interactive visualization. Node size reflects lines of code,
        fill color shows complexity (darker = more complex), and border color indicates code smells
        (redder = more severe). Drag nodes to rearrange, zoom and pan to explore.
    </p>
    <div id="d3-graph-container">
        <svg id="d3-graph"></svg>
    </div>
    <div class="d3-legend">
        <div class="legend-section">
            <div class="legend-title">Complexity (Fill)</div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #f3f4f6; border: 1px solid #e5e7eb;"></div>
                <span>0-5 (Low)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #9ca3af;"></div>
                <span>6-10 (Moderate)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #4b5563;"></div>
                <span>11-20 (High)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #1f2937;"></div>
                <span>21-30 (Very High)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #111827;"></div>
                <span>31+ (Critical)</span>
            </div>
        </div>
        <div class="legend-section">
            <div class="legend-title">Code Smells (Border)</div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 1px solid #e5e7eb;"></div>
                <span>None</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 2px solid #fca5a5;"></div>
                <span>Info</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 3px solid #f87171;"></div>
                <span>Warning</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: white; border: 4px solid #ef4444;"></div>
                <span>Error</span>
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
    <script id="d3-graph-data" type="application/json">{d3_json}</script>
</section>"""

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
// Initialize D3 Graph
(function() {{
    const dataScript = document.getElementById('d3-graph-data');
    if (!dataScript) return;

    const graphData = JSON.parse(dataScript.textContent);
    if (!graphData.nodes || graphData.nodes.length === 0) return;

    const svg = d3.select("#d3-graph");
    const container = document.getElementById("d3-graph-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr("viewBox", [0, 0, width, height]);

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

    // Size scale for LOC (min 8px, max 40px)
    const maxLoc = d3.max(graphData.nodes, d => d.loc) || 100;
    const sizeScale = d3.scaleSqrt()
        .domain([0, maxLoc])
        .range([8, 40]);

    // Edge thickness scale (min 1px, max 4px)
    const maxCoupling = d3.max(graphData.links, d => d.coupling) || 1;
    const edgeScale = d3.scaleLinear()
        .domain([1, maxCoupling])
        .range([1, 4]);

    // Force simulation
    const simulation = d3.forceSimulation(graphData.nodes)
        .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(d => sizeScale(d.loc) + 5));

    // Zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => g.attr("transform", event.transform));

    svg.call(zoom);

    const g = svg.append("g");

    // Draw edges
    const link = g.append("g")
        .selectAll("line")
        .data(graphData.links)
        .join("line")
        .attr("class", d => d.circular ? "link link-circular" : "link")
        .attr("stroke-width", d => edgeScale(d.coupling));

    // Draw nodes
    const node = g.append("g")
        .selectAll("g")
        .data(graphData.nodes)
        .join("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Node circles with complexity fill and smell border
    node.append("circle")
        .attr("r", d => sizeScale(d.loc))
        .attr("fill", d => complexityColor(d.complexity))
        .attr("stroke", d => smellBorder(d.smell_severity).color)
        .attr("stroke-width", d => smellBorder(d.smell_severity).width)
        .style("filter", d => d.smell_severity === 'critical' ? 'drop-shadow(0 0 4px #dc2626)' : null);

    // Node labels
    node.append("text")
        .text(d => d.label)
        .attr("class", "node-label")
        .attr("dy", d => sizeScale(d.loc) + 12);

    // Tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "d3-tooltip");

    node.on("mouseenter", (event, d) => {{
        tooltip.transition().duration(200).style("opacity", 1);
        tooltip.html(`
            <strong>${{d.label}}</strong><br/>
            <span style="color:#6b7280">Module: ${{d.module}}</span><br/>
            <hr style="margin:8px 0;border:none;border-top:1px solid #e5e7eb">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
                <span>Complexity:</span><span><strong>${{d.complexity}}</strong></span>
                <span>LOC:</span><span>${{d.loc}}</span>
                <span>Smells:</span><span style="color:${{smellBorder(d.smell_severity).color}}">${{d.smell_count}} (${{d.smell_severity}})</span>
            </div>
        `)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px");
    }})
    .on("mouseleave", () => {{
        tooltip.transition().duration(500).style("opacity", 0);
    }});

    // Simulation tick
    simulation.on("tick", () => {{
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});

    // Drag functions
    function dragstarted(event) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }}

    function dragged(event) {{
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }}

    function dragended(event) {{
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }}
}})();

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
