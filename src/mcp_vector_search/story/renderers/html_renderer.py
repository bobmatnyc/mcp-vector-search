"""Interactive HTML renderer for StoryIndex.

This module generates a self-contained HTML visualization with:
- D3.js timeline with commit density and key events
- Collapsible sidebar navigation with act-based structure
- Theme bars, contributor heatmap, tech stack badges
- Dark/light mode toggle with localStorage persistence
- Responsive design (desktop and mobile)
"""

from __future__ import annotations

from pathlib import Path

from ..models import StoryIndex


def render_html(story: StoryIndex, output_path: Path | None = None) -> str:
    """Render StoryIndex as interactive HTML.

    Args:
        story: The StoryIndex to render
        output_path: If provided, write to this file

    Returns:
        HTML string
    """
    html = _build_html(story)

    if output_path:
        output_path.write_text(html, encoding="utf-8")

    return html


def _build_html(story: StoryIndex) -> str:
    """Build complete HTML document."""
    story_json = _serialize_story_data(story)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{story.narrative.title or story.metadata.project_name} - Code Story</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
{_get_styles()}
    </style>
</head>
<body>
    {_get_html_body(story)}

    <script id="story-data" type="application/json">
{story_json}
    </script>

    <script>
{_get_scripts()}
    </script>
</body>
</html>"""


def _get_html_body(story: StoryIndex) -> str:
    """Generate HTML body structure."""
    project_name = story.narrative.title or story.metadata.project_name or "Code Story"
    subtitle = story.narrative.subtitle or ""

    return f"""    <div class="app-container">
        <header class="app-header">
            <div class="header-content">
                <h1 class="header-title">📖 {project_name}</h1>
                {f'<p class="header-subtitle">{subtitle}</p>' if subtitle else ""}
            </div>
            <button class="theme-toggle" id="theme-toggle" title="Toggle dark/light mode">
                <span id="theme-icon">🌙</span>
            </button>
        </header>

        <div class="main-layout">
            <aside class="sidebar" id="sidebar">
                <button class="sidebar-toggle" id="sidebar-toggle">☰</button>
                <nav class="sidebar-nav" id="sidebar-nav">
                    <a href="#overview" class="nav-link active">Overview</a>
                    <div id="dynamic-nav"></div>
                    <a href="#themes" class="nav-link">Themes</a>
                    <a href="#people" class="nav-link">Contributors</a>
                    <a href="#tech" class="nav-link">Tech Stack</a>
                    <a href="#sources" class="nav-link">Sources</a>
                </nav>
            </aside>

            <main class="main-content">
                <section id="overview" class="content-section">
                    <div class="timeline-container">
                        <h2>Project Timeline</h2>
                        <div id="timeline-viz" class="timeline-viz"></div>
                    </div>

                    {_render_executive_summary(story)}
                </section>

                <div id="narrative-content"></div>

                <section id="themes" class="content-section">
                    <h2>Themes</h2>
                    <div id="themes-viz"></div>
                </section>

                <section id="people" class="content-section">
                    <h2>Contributors</h2>
                    <div id="contributors-viz"></div>
                </section>

                <section id="tech" class="content-section">
                    <h2>Tech Stack</h2>
                    <div id="tech-stack-viz"></div>
                </section>

                <section id="sources" class="content-section">
                    <h2>Sources</h2>
                    <div id="sources-info"></div>
                </section>
            </main>
        </div>
    </div>"""


def _render_executive_summary(story: StoryIndex) -> str:
    """Render executive summary section."""
    if not story.narrative.executive_summary:
        return ""

    summary = story.narrative.executive_summary.replace("\n", "<br>")
    return f"""
                    <div class="executive-summary">
                        <h2>Executive Summary</h2>
                        <div class="summary-content">{summary}</div>
                    </div>"""


def _serialize_story_data(story: StoryIndex) -> str:
    """Convert StoryIndex to JSON for embedding."""
    return story.model_dump_json(indent=2)


def _get_styles() -> str:
    """Generate CSS styles."""
    return """
/* CSS Variables for Theming */
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
    --sidebar-width: 280px;
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

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    transition: background-color 0.3s ease, color 0.3s ease;
}

.app-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.app-header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-primary);
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
    transition: background-color 0.3s ease;
}

.header-content {
    flex: 1;
}

.header-title {
    font-size: 1.75rem;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.header-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
}

.theme-toggle {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.theme-toggle:hover {
    background: var(--accent);
    border-color: var(--accent);
    transform: scale(1.05);
}

/* Main Layout */
.main-layout {
    display: flex;
    flex: 1;
}

/* Sidebar */
.sidebar {
    width: var(--sidebar-width);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-primary);
    position: sticky;
    top: 80px;
    height: calc(100vh - 80px);
    overflow-y: auto;
    transition: transform 0.3s ease;
}

.sidebar-toggle {
    display: none;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin: 1rem;
    cursor: pointer;
    font-size: 1.25rem;
}

.sidebar-nav {
    padding: 1rem 0;
}

.nav-link {
    display: block;
    padding: 0.75rem 1.5rem;
    color: var(--text-secondary);
    text-decoration: none;
    transition: all 0.2s ease;
    border-left: 3px solid transparent;
}

.nav-link:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
}

.nav-link.active {
    color: var(--accent);
    background: var(--bg-tertiary);
    border-left-color: var(--accent);
    font-weight: 600;
}

/* Main Content */
.main-content {
    flex: 1;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

.content-section {
    margin-bottom: 3rem;
    scroll-margin-top: 100px;
}

.content-section h2 {
    font-size: 1.75rem;
    margin-bottom: 1.5rem;
    color: var(--text-primary);
    border-bottom: 2px solid var(--border-primary);
    padding-bottom: 0.5rem;
}

/* Timeline */
.timeline-container {
    margin-bottom: 2rem;
}

.timeline-viz {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 8px;
    padding: 1rem;
    min-height: 200px;
}

/* Executive Summary */
.executive-summary {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

.summary-content {
    color: var(--text-secondary);
    line-height: 1.8;
}

/* Act Sections */
.act-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
}

.act-header {
    margin-bottom: 1rem;
}

.act-number {
    font-size: 0.875rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.act-title {
    font-size: 1.5rem;
    color: var(--accent);
    margin: 0.5rem 0;
}

.act-date-range {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-style: italic;
}

.act-content {
    color: var(--text-primary);
    line-height: 1.8;
}

.act-content a {
    color: var(--accent);
    text-decoration: none;
}

.act-content a:hover {
    text-decoration: underline;
}

/* Evidence Citations */
.evidence-list {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-primary);
}

.evidence-title {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.evidence-item {
    display: inline-block;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    margin: 0.25rem;
    font-size: 0.75rem;
    font-family: monospace;
    color: var(--text-secondary);
}

/* Themes */
.theme-bars {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.theme-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.theme-bar:hover {
    border-color: var(--accent);
    transform: translateX(4px);
}

.theme-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.theme-name {
    font-weight: 600;
    color: var(--text-primary);
}

.theme-confidence {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}

.theme-confidence.high {
    background: rgba(35, 134, 54, 0.2);
    color: var(--success);
}

.theme-confidence.medium {
    background: rgba(210, 153, 34, 0.2);
    color: var(--warning);
}

.theme-confidence.low {
    background: rgba(110, 118, 129, 0.2);
    color: var(--text-tertiary);
}

.theme-description {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
}

.theme-bar-container {
    background: var(--bg-tertiary);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.theme-bar-fill {
    height: 100%;
    background: var(--success);
    transition: width 0.3s ease;
}

.theme-examples {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-primary);
    display: none;
}

.theme-bar.expanded .theme-examples {
    display: block;
}

/* Contributors */
.contributors-heatmap {
    overflow-x: auto;
}

.heatmap-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}

.heatmap-label {
    width: 150px;
    font-size: 0.875rem;
    color: var(--text-secondary);
    padding-right: 1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.heatmap-cells {
    display: flex;
    gap: 2px;
}

.heatmap-cell {
    width: 12px;
    height: 12px;
    background: var(--bg-tertiary);
    border-radius: 2px;
    transition: transform 0.2s ease;
}

.heatmap-cell:hover {
    transform: scale(1.5);
}

.heatmap-cell[data-level="0"] { background: var(--bg-tertiary); }
.heatmap-cell[data-level="1"] { background: rgba(88, 166, 255, 0.3); }
.heatmap-cell[data-level="2"] { background: rgba(88, 166, 255, 0.5); }
.heatmap-cell[data-level="3"] { background: rgba(88, 166, 255, 0.7); }
.heatmap-cell[data-level="4"] { background: rgba(88, 166, 255, 0.9); }

/* Tech Stack */
.tech-stack-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}

.tech-category {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    padding: 1rem;
}

.tech-category-title {
    font-size: 0.875rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.tech-badge {
    display: inline-block;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-primary);
    border-radius: 12px;
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
    color: var(--text-primary);
    transition: all 0.2s ease;
}

.tech-badge:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--bg-primary);
}

.tech-badge.language { border-left: 3px solid #1f6feb; }
.tech-badge.framework { border-left: 3px solid #238636; }
.tech-badge.tool { border-left: 3px solid #d29922; }
.tech-badge.library { border-left: 3px solid #8957e5; }

/* Sources */
.sources-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.source-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    padding: 1.5rem;
    text-align: center;
}

.source-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

.source-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tooltip */
.tooltip {
    position: absolute;
    padding: 0.75rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-primary);
    border-radius: 6px;
    font-size: 0.875rem;
    color: var(--text-primary);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s ease;
    box-shadow: 0 4px 12px var(--shadow);
    z-index: 1000;
}

.tooltip.visible {
    opacity: 0.95;
}

/* Responsive */
@media (max-width: 768px) {
    .sidebar {
        position: fixed;
        left: 0;
        top: 0;
        height: 100vh;
        transform: translateX(-100%);
        z-index: 200;
    }

    .sidebar.open {
        transform: translateX(0);
    }

    .sidebar-toggle {
        display: block;
    }

    .main-content {
        padding: 1rem;
    }

    .tech-stack-grid {
        grid-template-columns: 1fr;
    }

    .sources-grid {
        grid-template-columns: 1fr;
    }
}"""


def _get_scripts() -> str:
    """Generate JavaScript code."""
    return """
// Load story data
const storyData = JSON.parse(document.getElementById('story-data').textContent);

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') ||
                       (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    document.getElementById('theme-icon').textContent = theme === 'dark' ? '🌙' : '☀️';
}

document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

// Sidebar navigation
function initSidebar() {
    const toggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');

    toggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Close sidebar on mobile when clicking a link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        });
    });

    // Update active link on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
                });
            }
        });
    }, { threshold: 0.5 });

    document.querySelectorAll('.content-section').forEach(section => {
        observer.observe(section);
    });
}

// Render narrative acts
function renderNarrativeActs() {
    const dynamicNav = document.getElementById('dynamic-nav');
    const narrativeContent = document.getElementById('narrative-content');

    storyData.narrative.acts.forEach(act => {
        // Add to sidebar
        const navLink = document.createElement('a');
        navLink.href = `#act-${act.number}`;
        navLink.className = 'nav-link';
        navLink.textContent = `Act ${act.number}: ${act.title}`;
        dynamicNav.appendChild(navLink);

        // Add to content
        const section = document.createElement('section');
        section.id = `act-${act.number}`;
        section.className = 'content-section act-section';

        section.innerHTML = `
            <div class="act-header">
                <div class="act-number">Act ${act.number}</div>
                <h2 class="act-title">${act.title}</h2>
                ${act.date_range ? `<div class="act-date-range">${act.date_range}</div>` : ''}
            </div>
            <div class="act-content">${formatMarkdown(act.content)}</div>
            ${act.evidence.length > 0 ? `
                <div class="evidence-list">
                    <div class="evidence-title">Evidence:</div>
                    ${act.evidence.map(e => `<span class="evidence-item">${e}</span>`).join('')}
                </div>
            ` : ''}
        `;

        narrativeContent.appendChild(section);
    });
}

// Simple markdown formatter
function formatMarkdown(text) {
    return text
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
        .replace(/\\n\\n/g, '</p><p>')
        .replace(/\\n/g, '<br>')
        .replace(/^(.+)$/, '<p>$1</p>');
}

// Render timeline with D3
function renderTimeline() {
    const container = document.getElementById('timeline-viz');
    const width = container.clientWidth;
    const height = 200;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };

    const svg = d3.select('#timeline-viz')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Process data
    const events = storyData.visualization.timeline_events.map(e => ({
        ...e,
        date: new Date(e.date)
    }));

    const commitDensity = Object.entries(storyData.visualization.commit_density)
        .map(([date, count]) => ({ date: new Date(date), count }))
        .sort((a, b) => a.date - b.date);

    if (events.length === 0 && commitDensity.length === 0) {
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', 'var(--text-secondary)')
            .text('No timeline data available');
        return;
    }

    // Scales
    const allDates = [...events.map(e => e.date), ...commitDensity.map(d => d.date)];
    const xScale = d3.scaleTime()
        .domain(d3.extent(allDates))
        .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
        .domain([0, d3.max(commitDensity, d => d.count) || 1])
        .range([height - margin.bottom, margin.top]);

    // Area chart for commit density
    if (commitDensity.length > 0) {
        const area = d3.area()
            .x(d => xScale(d.date))
            .y0(height - margin.bottom)
            .y1(d => yScale(d.count))
            .curve(d3.curveMonotoneX);

        svg.append('path')
            .datum(commitDensity)
            .attr('fill', 'var(--accent)')
            .attr('opacity', 0.2)
            .attr('d', area);
    }

    // X axis
    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .attr('color', 'var(--text-secondary)');

    // Y axis
    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(3))
        .attr('color', 'var(--text-secondary)');

    // Event circles
    const eventColors = {
        commit: '#58a6ff',
        issue: '#238636',
        pull_request: '#8957e5',
        pivot: '#da3633',
        milestone: '#d29922'
    };

    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip');

    svg.selectAll('.event-circle')
        .data(events)
        .enter()
        .append('circle')
        .attr('class', 'event-circle')
        .attr('cx', d => xScale(d.date))
        .attr('cy', d => yScale(0))
        .attr('r', d => 3 + d.importance * 3)
        .attr('fill', d => eventColors[d.event_type] || '#8b949e')
        .attr('stroke', 'var(--bg-primary)')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseover', (event, d) => {
            tooltip
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .classed('visible', true)
                .html(`
                    <strong>${d.title}</strong><br>
                    ${d.date.toLocaleDateString()}<br>
                    ${d.description || ''}
                `);
        })
        .on('mouseout', () => {
            tooltip.classed('visible', false);
        });
}

// Render themes
function renderThemes() {
    const container = document.getElementById('themes-viz');
    const themes = storyData.narrative.themes;

    if (themes.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary)">No themes identified</p>';
        return;
    }

    const maxEvidence = Math.max(...themes.map(t => t.evidence_count), 1);

    container.innerHTML = '<div class="theme-bars"></div>';
    const themeBars = container.querySelector('.theme-bars');

    themes.forEach(theme => {
        const barPercentage = (theme.evidence_count / maxEvidence) * 100;
        const themeEl = document.createElement('div');
        themeEl.className = 'theme-bar';
        themeEl.innerHTML = `
            <div class="theme-bar-header">
                <span class="theme-name">${theme.name}</span>
                <span class="theme-confidence ${theme.confidence}">${theme.confidence.toUpperCase()}</span>
            </div>
            <div class="theme-description">${theme.description}</div>
            <div class="theme-bar-container">
                <div class="theme-bar-fill" style="width: ${barPercentage}%"></div>
            </div>
            ${theme.examples.length > 0 ? `
                <div class="theme-examples">
                    ${theme.examples.map(ex => `<div class="evidence-item">${ex}</div>`).join('')}
                </div>
            ` : ''}
        `;

        themeEl.addEventListener('click', () => {
            themeEl.classList.toggle('expanded');
        });

        themeBars.appendChild(themeEl);
    });
}

// Render contributors heatmap
function renderContributors() {
    const container = document.getElementById('contributors-viz');
    const heatmapData = storyData.visualization.activity_heatmap;

    if (Object.keys(heatmapData).length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary)">No contributor activity data</p>';
        return;
    }

    // Get top 10 contributors by total commits
    const topContributors = Object.entries(heatmapData)
        .map(([name, dates]) => ({
            name,
            total: Object.values(dates).reduce((sum, count) => sum + count, 0),
            dates
        }))
        .sort((a, b) => b.total - a.total)
        .slice(0, 10);

    // Get all unique dates
    const allDates = new Set();
    topContributors.forEach(c => {
        Object.keys(c.dates).forEach(date => allDates.add(date));
    });
    const sortedDates = Array.from(allDates).sort().slice(-52); // Last 52 weeks

    container.innerHTML = '<div class="contributors-heatmap"></div>';
    const heatmap = container.querySelector('.contributors-heatmap');

    topContributors.forEach(contributor => {
        const row = document.createElement('div');
        row.className = 'heatmap-row';

        const label = document.createElement('div');
        label.className = 'heatmap-label';
        label.textContent = contributor.name;
        row.appendChild(label);

        const cells = document.createElement('div');
        cells.className = 'heatmap-cells';

        const maxCount = Math.max(...Object.values(contributor.dates));

        sortedDates.forEach(date => {
            const count = contributor.dates[date] || 0;
            const level = maxCount > 0 ? Math.ceil((count / maxCount) * 4) : 0;

            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.setAttribute('data-level', level);
            cell.title = `${date}: ${count} commits`;
            cells.appendChild(cell);
        });

        row.appendChild(cells);
        heatmap.appendChild(row);
    });
}

// Render tech stack
function renderTechStack() {
    const container = document.getElementById('tech-stack-viz');
    const techStack = storyData.analysis.tech_stack;

    if (techStack.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary)">No tech stack data</p>';
        return;
    }

    // Group by category
    const grouped = techStack.reduce((acc, tech) => {
        const category = tech.category || 'other';
        if (!acc[category]) acc[category] = [];
        acc[category].push(tech);
        return acc;
    }, {});

    container.innerHTML = '<div class="tech-stack-grid"></div>';
    const grid = container.querySelector('.tech-stack-grid');

    Object.entries(grouped).forEach(([category, items]) => {
        const categoryEl = document.createElement('div');
        categoryEl.className = 'tech-category';
        categoryEl.innerHTML = `
            <div class="tech-category-title">${category}</div>
            <div class="tech-badges">
                ${items.map(tech =>
                    `<span class="tech-badge ${category}" title="${tech.evidence.join(', ')}">${tech.name}</span>`
                ).join('')}
            </div>
        `;
        grid.appendChild(categoryEl);
    });
}

// Render sources
function renderSources() {
    const container = document.getElementById('sources-info');
    const metadata = storyData.metadata;
    const extraction = storyData.extraction;

    container.innerHTML = `
        <div class="sources-grid">
            <div class="source-card">
                <div class="source-value">${metadata.total_commits}</div>
                <div class="source-label">Commits</div>
            </div>
            <div class="source-card">
                <div class="source-value">${metadata.total_contributors}</div>
                <div class="source-label">Contributors</div>
            </div>
            <div class="source-card">
                <div class="source-value">${extraction.issues.length}</div>
                <div class="source-label">Issues</div>
            </div>
            <div class="source-card">
                <div class="source-value">${extraction.pull_requests.length}</div>
                <div class="source-label">Pull Requests</div>
            </div>
            <div class="source-card">
                <div class="source-value">${metadata.total_files}</div>
                <div class="source-label">Files</div>
            </div>
            <div class="source-card">
                <div class="source-value">${extraction.docs.length}</div>
                <div class="source-label">Docs</div>
            </div>
        </div>
    `;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSidebar();
    renderNarrativeActs();
    renderTimeline();
    renderThemes();
    renderContributors();
    renderTechStack();
    renderSources();
});"""


def _markdown_to_html(markdown: str) -> str:
    """Convert markdown to HTML (simple implementation).

    For production, consider using a library like markdown-it or commonmark.
    This is a minimal implementation for basic formatting.
    """
    html = markdown
    html = html.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    html = html.replace("*", "<em>", 1).replace("*", "</em>", 1)
    html = html.replace("\n\n", "</p><p>")
    html = html.replace("\n", "<br>")
    html = f"<p>{html}</p>"
    return html
