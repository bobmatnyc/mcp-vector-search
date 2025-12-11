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

from pathlib import Path

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
    <link rel="stylesheet" href="{self.HIGHLIGHT_JS_CDN}/styles/github.min.css">
    <script src="{self.HIGHLIGHT_JS_CDN}/highlight.min.js"></script>
    {self._generate_styles()}
</head>
<body>
    {self._generate_header(export)}
    {self._generate_summary_section(export)}
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
