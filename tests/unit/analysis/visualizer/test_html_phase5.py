"""Tests for Phase 5 enhancements to HTML report generator.

This module tests:
- Performance optimizations (throttling, debouncing, LOD)
- Accessibility features (ARIA, high contrast, reduced motion)
- Export functionality (PNG, SVG, share links)
- Error handling (D3 load failures, empty data)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mcp_vector_search.analysis.visualizer import (
    AnalysisExport,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    HTMLReportGenerator,
    MetricsSummary,
)


@pytest.fixture
def minimal_export() -> AnalysisExport:
    """Create minimal export for testing."""
    return AnalysisExport(
        metadata=ExportMetadata(
            version="1.0.0",
            generated_at=datetime.now(),
            tool_version="0.19.0",
            project_root="/test",
        ),
        summary=MetricsSummary(
            total_files=1,
            total_functions=1,
            total_classes=0,
            total_lines=10,
            avg_complexity=5.0,
            avg_cognitive_complexity=5.0,
            avg_nesting_depth=1.0,
            total_smells=0,
        ),
        files=[
            FileDetail(
                path="test.py",
                language="python",
                lines_of_code=10,
                cyclomatic_complexity=5,
                cognitive_complexity=5,
                max_nesting_depth=1,
                function_count=1,
                class_count=0,
                efferent_coupling=0,
                afferent_coupling=0,
            )
        ],
        dependencies=DependencyGraph(edges=[], circular_dependencies=[]),
    )


@pytest.fixture
def empty_export() -> AnalysisExport:
    """Create empty export for error handling tests."""
    return AnalysisExport(
        metadata=ExportMetadata(
            version="1.0.0",
            generated_at=datetime.now(),
            tool_version="0.19.0",
            project_root="/test",
        ),
        summary=MetricsSummary(
            total_files=0,
            total_functions=0,
            total_classes=0,
            total_lines=0,
            avg_complexity=0.0,
            avg_cognitive_complexity=0.0,
            avg_nesting_depth=0.0,
            total_smells=0,
        ),
        files=[],
        dependencies=DependencyGraph(edges=[], circular_dependencies=[]),
    )


class TestAccessibilityFeatures:
    """Test Phase 5 accessibility enhancements."""

    def test_skip_link_present(self, minimal_export: AnalysisExport) -> None:
        """Test skip link is present in HTML."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'class="skip-link"' in html
        assert 'href="#main-content"' in html
        assert "Skip to main content" in html

    def test_high_contrast_toggle(self, minimal_export: AnalysisExport) -> None:
        """Test high contrast mode toggle button."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'id="toggle-contrast"' in html
        assert 'aria-pressed="false"' in html
        assert "High Contrast" in html

    def test_reduced_motion_toggle(self, minimal_export: AnalysisExport) -> None:
        """Test reduced motion toggle button."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'id="toggle-reduced-motion"' in html
        assert 'aria-pressed="false"' in html
        assert "Reduce Motion" in html

    def test_screen_reader_announcements(self, minimal_export: AnalysisExport) -> None:
        """Test screen reader announcement region."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'id="sr-announcements"' in html
        assert 'role="status"' in html
        assert 'aria-live="polite"' in html

    def test_aria_labels_on_controls(self, minimal_export: AnalysisExport) -> None:
        """Test ARIA labels on interactive elements."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        # Export buttons
        assert 'aria-label="Export graph as PNG image"' in html
        assert 'aria-label="Export graph as SVG vector image"' in html
        assert 'aria-label="Copy shareable link with current filters"' in html

        # Graph container
        assert 'role="img"' in html
        assert 'aria-label="Interactive dependency graph visualization"' in html

    def test_detail_panel_dialog_role(self, minimal_export: AnalysisExport) -> None:
        """Test detail panel has dialog role."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'role="dialog"' in html
        assert 'aria-labelledby="detail-panel-title"' in html

    def test_high_contrast_css_classes(self, minimal_export: AnalysisExport) -> None:
        """Test high contrast CSS classes."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "body.high-contrast" in html
        assert ".high-contrast .card" in html
        assert ".high-contrast .node circle" in html

    def test_reduced_motion_css(self, minimal_export: AnalysisExport) -> None:
        """Test reduced motion CSS."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "@media (prefers-reduced-motion: reduce)" in html
        assert "animation-duration: 0.01ms" in html


class TestExportFunctionality:
    """Test Phase 5 export features."""

    def test_export_png_button(self, minimal_export: AnalysisExport) -> None:
        """Test PNG export button is present."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'onclick="exportAsPNG()"' in html
        assert "📥 Export PNG" in html

    def test_export_svg_button(self, minimal_export: AnalysisExport) -> None:
        """Test SVG export button is present."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'onclick="exportAsSVG()"' in html
        assert "📥 Export SVG" in html

    def test_copy_link_button(self, minimal_export: AnalysisExport) -> None:
        """Test copy link button is present."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'onclick="copyShareLink()"' in html
        assert "🔗 Copy Link" in html

    def test_export_png_function_defined(self, minimal_export: AnalysisExport) -> None:
        """Test exportAsPNG function is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "function exportAsPNG()" in html
        assert "canvas.toBlob" in html
        assert "dependency-graph.png" in html

    def test_export_svg_function_defined(self, minimal_export: AnalysisExport) -> None:
        """Test exportAsSVG function is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "function exportAsSVG()" in html
        assert "XMLSerializer" in html
        assert "dependency-graph.svg" in html

    def test_copy_share_link_function(self, minimal_export: AnalysisExport) -> None:
        """Test copyShareLink function is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "function copyShareLink()" in html
        assert "btoa(JSON.stringify(filterState))" in html
        assert "navigator.clipboard.writeText" in html

    def test_load_filters_from_url_function(
        self, minimal_export: AnalysisExport
    ) -> None:
        """Test loadFiltersFromURL function is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "function loadFiltersFromURL()" in html
        assert "hash.startsWith('#filters=')" in html
        assert "atob(encoded)" in html


class TestErrorHandling:
    """Test Phase 5 error handling."""

    def test_empty_data_error_message(self, empty_export: AnalysisExport) -> None:
        """Test error message for empty data."""
        generator = HTMLReportGenerator()
        html = generator.generate(empty_export)

        assert 'id="graph-error"' in html
        assert 'role="alert"' in html
        assert 'id="graph-error-message"' in html

    def test_loading_spinner(self, minimal_export: AnalysisExport) -> None:
        """Test loading spinner is present."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert 'id="graph-loading"' in html
        assert 'class="loading-spinner"' in html
        assert "Loading visualization..." in html

    def test_d3_load_failure_check(self, minimal_export: AnalysisExport) -> None:
        """Test D3 load failure detection."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "typeof d3 === 'undefined'" in html
        assert "D3.js library failed to load" in html

    def test_data_parse_error_handling(self, minimal_export: AnalysisExport) -> None:
        """Test data parse error handling."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "JSON.parse(dataScript.textContent)" in html
        assert "Failed to parse graph data" in html

    def test_empty_nodes_check(self, minimal_export: AnalysisExport) -> None:
        """Test empty nodes array check."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "graphData.nodes.length === 0" in html
        assert "No files in analysis" in html


class TestPerformanceOptimizations:
    """Test Phase 5 performance optimizations."""

    def test_large_graph_threshold(self, minimal_export: AnalysisExport) -> None:
        """Test large graph threshold is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "LARGE_GRAPH_THRESHOLD = 100" in html
        assert "isLargeGraph = nodeCount > LARGE_GRAPH_THRESHOLD" in html

    def test_animation_duration_conditional(
        self, minimal_export: AnalysisExport
    ) -> None:
        """Test animation duration is conditional."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "animationDuration = isLargeGraph ? 0" in html
        assert "classList.contains('reduced-motion')" in html

    def test_tick_throttle_constant(self, minimal_export: AnalysisExport) -> None:
        """Test simulation tick throttling."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "TICK_THROTTLE_MS = 16" in html
        assert "now - lastTickTime < TICK_THROTTLE_MS" in html

    def test_filter_debounce_constant(self, minimal_export: AnalysisExport) -> None:
        """Test filter debouncing."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "FILTER_DEBOUNCE_MS = 150" in html
        assert "filterDebounceTimer" in html
        assert "setTimeout(() =>" in html

    def test_request_animation_frame_usage(
        self, minimal_export: AnalysisExport
    ) -> None:
        """Test requestAnimationFrame is used."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "requestAnimationFrame" in html

    def test_drag_throttling(self, minimal_export: AnalysisExport) -> None:
        """Test drag operations are throttled."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "dragThrottle" in html
        assert "function dragged(event)" in html


class TestLevelOfDetail:
    """Test Phase 5 Level of Detail rendering."""

    def test_lod_zoom_thresholds(self, minimal_export: AnalysisExport) -> None:
        """Test LOD zoom thresholds are defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "LOD_ZOOM_THRESHOLD_LOW = 0.5" in html
        assert "LOD_ZOOM_THRESHOLD_HIGH = 1.5" in html

    def test_update_lod_function(self, minimal_export: AnalysisExport) -> None:
        """Test updateLOD function is defined."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "function updateLOD(zoomLevel)" in html

    def test_lod_zoom_out_behavior(self, minimal_export: AnalysisExport) -> None:
        """Test LOD behavior when zoomed out."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        # Zoom out: hide labels
        assert "zoomLevel < LOD_ZOOM_THRESHOLD_LOW" in html
        assert '.style("display", "none")' in html

    def test_lod_zoom_in_behavior(self, minimal_export: AnalysisExport) -> None:
        """Test LOD behavior when zoomed in."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        # Zoom in: show complexity labels
        assert "zoomLevel > LOD_ZOOM_THRESHOLD_HIGH" in html
        assert "complexity-label" in html

    def test_zoom_event_lod_trigger(self, minimal_export: AnalysisExport) -> None:
        """Test zoom events trigger LOD updates."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        assert "Math.abs(newZoom - currentZoom) > 0.1" in html
        assert "updateLOD(currentZoom)" in html


class TestIntegration:
    """Integration tests for Phase 5 features."""

    def test_full_report_generation(
        self, minimal_export: AnalysisExport, tmp_path: Path
    ) -> None:
        """Test full report generation with Phase 5 features."""
        generator = HTMLReportGenerator()
        output_path = tmp_path / "report.html"

        result_path = generator.generate_to_file(minimal_export, output_path)

        assert result_path.exists()
        html = result_path.read_text(encoding="utf-8")

        # Verify Phase 5 features are present
        assert 'class="skip-link"' in html
        assert 'id="toggle-contrast"' in html
        assert "exportAsPNG()" in html
        assert "LARGE_GRAPH_THRESHOLD" in html
        assert "updateLOD" in html

    def test_all_phase_features_integrated(
        self, minimal_export: AnalysisExport
    ) -> None:
        """Test all phases work together."""
        generator = HTMLReportGenerator()
        html = generator.generate(minimal_export)

        # Phase 1-2: Basic graph
        assert "d3-graph" in html

        # Phase 3: Filters
        assert "filter-grade-" in html

        # Phase 4: Enhanced tooltips
        assert "tooltip-header" in html

        # Phase 5: New features
        assert "exportAsPNG" in html
        assert "skip-link" in html
        assert "LOD_ZOOM_THRESHOLD" in html
