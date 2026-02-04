"""Tests for codebase profiler."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.codebase_profiler import (
    CodebaseProfiler,
    CodebaseSize,
    CodebaseType,
)


@pytest.fixture
def temp_project():
    """Create a temporary project directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create Python files
        for i in range(5):
            (project_root / f"file{i}.py").write_text(f"# Python file {i}")

        # Create JavaScript files
        for i in range(3):
            (project_root / f"file{i}.js").write_text(f"// JavaScript file {i}")

        # Create documentation files
        (project_root / "README.md").write_text("# Project README")
        (project_root / "CHANGELOG.md").write_text("# Changelog")

        yield project_root


def test_profile_small_codebase(temp_project):
    """Test profiling a small codebase."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile()

    # Should detect as small (< 1000 files)
    assert profile.size_category == CodebaseSize.SMALL
    assert profile.total_files == 10  # 5 py + 3 js + 2 md

    # Should detect as Python-heavy (50% Python)
    assert profile.codebase_type == CodebaseType.PYTHON

    # Check language distribution
    assert ".py" in profile.language_distribution
    assert ".js" in profile.language_distribution
    assert ".md" in profile.language_distribution

    # Verify percentages add to ~100
    total_pct = sum(profile.language_distribution.values())
    assert 99.0 <= total_pct <= 101.0  # Allow floating point error


def test_detect_size_category():
    """Test size category detection."""
    profiler = CodebaseProfiler(Path("/tmp"))

    assert profiler._detect_size_category(500) == CodebaseSize.SMALL
    assert profiler._detect_size_category(5000) == CodebaseSize.MEDIUM
    assert profiler._detect_size_category(25000) == CodebaseSize.LARGE
    assert profiler._detect_size_category(75000) == CodebaseSize.ENTERPRISE


def test_detect_codebase_type():
    """Test codebase type detection."""
    profiler = CodebaseProfiler(Path("/tmp"))

    # Python-heavy
    lang_dist = {".py": 70.0, ".js": 20.0, ".md": 10.0}
    assert profiler._detect_codebase_type(lang_dist) == CodebaseType.PYTHON

    # JavaScript-heavy
    lang_dist = {".js": 50.0, ".ts": 20.0, ".py": 30.0}
    assert profiler._detect_codebase_type(lang_dist) == CodebaseType.JAVASCRIPT

    # Java-heavy
    lang_dist = {".java": 80.0, ".xml": 20.0}
    assert profiler._detect_codebase_type(lang_dist) == CodebaseType.JAVA

    # Documentation-heavy
    lang_dist = {".md": 40.0, ".txt": 10.0, ".py": 50.0}
    assert profiler._detect_codebase_type(lang_dist) == CodebaseType.DOCUMENTATION

    # Mixed/polyglot
    lang_dist = {".py": 30.0, ".js": 30.0, ".java": 30.0, ".md": 10.0}
    assert profiler._detect_codebase_type(lang_dist) == CodebaseType.MIXED


def test_get_optimization_preset_small():
    """Test optimization preset for small codebase."""
    profiler = CodebaseProfiler(Path("/tmp"))
    profile = profiler.profile()
    profile.size_category = CodebaseSize.SMALL

    preset = profiler.get_optimization_preset(profile)

    assert preset.batch_size == 16
    assert preset.parallel_embeddings is False
    assert len(preset.file_extensions) == 0  # Index all files


def test_get_optimization_preset_medium():
    """Test optimization preset for medium codebase."""
    profiler = CodebaseProfiler(Path("/tmp"))
    profile = profiler.profile()
    profile.size_category = CodebaseSize.MEDIUM

    preset = profiler.get_optimization_preset(profile)

    assert preset.batch_size == 32
    assert preset.parallel_embeddings is True
    assert len(preset.file_extensions) == 0  # Index all files


def test_get_optimization_preset_large():
    """Test optimization preset for large codebase."""
    profiler = CodebaseProfiler(Path("/tmp"))
    profile = profiler.profile()
    profile.size_category = CodebaseSize.LARGE

    preset = profiler.get_optimization_preset(profile)

    assert preset.batch_size == 64
    assert preset.parallel_embeddings is True
    assert len(preset.file_extensions) > 0  # Code-only filtering


def test_get_optimization_preset_enterprise():
    """Test optimization preset for enterprise codebase."""
    profiler = CodebaseProfiler(Path("/tmp"))
    profile = profiler.profile()
    profile.size_category = CodebaseSize.ENTERPRISE

    preset = profiler.get_optimization_preset(profile)

    assert preset.batch_size == 128
    assert preset.parallel_embeddings is True
    assert len(preset.file_extensions) > 0  # Code-only filtering


def test_format_profile_summary(temp_project):
    """Test profile summary formatting."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile()

    summary = profiler.format_profile_summary(profile)

    assert "Small" in summary
    assert "Python" in summary
    assert str(profile.total_files) in summary


def test_format_optimization_summary(temp_project):
    """Test optimization summary formatting."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile()
    preset = profiler.get_optimization_preset(profile)

    summary = profiler.format_optimization_summary(profile, preset)

    assert "Batch size" in summary
    assert "Parallel embeddings" in summary
    assert "File filter" in summary


def test_profile_with_sampling(temp_project):
    """Test profiling with sampling."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile(max_sample=5)

    # Should mark as sampled since we limited to 5 files
    assert profile.sampled is True
    assert profile.total_files <= 5


def test_profile_full_scan(temp_project):
    """Test full scan profiling."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile(force_full_scan=True)

    # Should scan all files
    assert profile.sampled is False
    assert profile.total_files == 10


def test_profile_performance(temp_project):
    """Test that profiling completes quickly."""
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile()

    # Should complete in under 2 seconds (requirement)
    assert profile.scan_time_seconds < 2.0
