"""Codebase profiler for automatic optimization of indexing settings.

This module analyzes codebase characteristics (size, language distribution) and
recommends optimal indexing settings for performance.
"""

import os
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from loguru import logger


class CodebaseSize(StrEnum):
    """Codebase size categories based on file count."""

    SMALL = "small"  # < 1,000 files
    MEDIUM = "medium"  # 1,000 - 10,000 files
    LARGE = "large"  # 10,000 - 50,000 files
    ENTERPRISE = "enterprise"  # > 50,000 files


class CodebaseType(StrEnum):
    """Codebase type based on dominant language."""

    PYTHON = "python"  # >60% .py files
    JAVASCRIPT = "javascript"  # >60% .js/.ts/.tsx/.jsx files
    JAVA = "java"  # >60% .java files
    MIXED = "mixed"  # No dominant language (polyglot)
    DOCUMENTATION = "documentation"  # >30% .md/.txt/.rst files


# Code-only extensions (exclude docs, configs, assets)
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".m",
    ".vue",
    ".dart",
    ".lua",
    ".pl",
    ".sh",
    ".bash",
    ".sql",
}

DOCUMENTATION_EXTENSIONS = {".md", ".txt", ".rst", ".adoc", ".org"}


@dataclass
class OptimizationPreset:
    """Recommended indexing settings based on codebase profile."""

    batch_size: int
    parallel_embeddings: bool
    file_extensions: set[str]
    max_cache_size: int
    description: str


# Optimization presets by codebase size
OPTIMIZATION_PRESETS: dict[CodebaseSize, OptimizationPreset] = {
    CodebaseSize.SMALL: OptimizationPreset(
        batch_size=16,
        parallel_embeddings=False,
        file_extensions=set(),  # Index all files
        max_cache_size=500,
        description="Small codebase: conservative settings, index all files",
    ),
    CodebaseSize.MEDIUM: OptimizationPreset(
        batch_size=32,
        parallel_embeddings=True,
        file_extensions=set(),  # Index all files
        max_cache_size=2000,
        description="Medium codebase: balanced settings, parallel enabled",
    ),
    CodebaseSize.LARGE: OptimizationPreset(
        batch_size=64,
        parallel_embeddings=True,
        file_extensions=CODE_EXTENSIONS,  # Code-only filtering
        max_cache_size=10000,
        description="Large codebase: aggressive optimization, code-only files",
    ),
    CodebaseSize.ENTERPRISE: OptimizationPreset(
        batch_size=128,
        parallel_embeddings=True,
        file_extensions=CODE_EXTENSIONS,  # Code-only filtering
        max_cache_size=50000,
        description="Enterprise codebase: maximum optimization, code-only files",
    ),
}


@dataclass
class CodebaseProfile:
    """Profile of a codebase with size, type, and language distribution."""

    total_files: int
    size_category: CodebaseSize
    codebase_type: CodebaseType
    language_distribution: dict[str, float]  # Extension -> percentage
    top_languages: list[tuple[str, float]]  # Top 3 languages
    scan_time_seconds: float
    sampled: bool  # Whether this is a sample or full scan


class CodebaseProfiler:
    """Profiles codebases to detect size and type for auto-optimization."""

    # Maximum files to sample for fast profiling (< 2 seconds)
    MAX_SAMPLE_SIZE = 1000

    def __init__(self, project_root: Path) -> None:
        """Initialize codebase profiler.

        Args:
            project_root: Project root directory to profile
        """
        self.project_root = project_root

    def profile(
        self, max_sample: int | None = None, force_full_scan: bool = False
    ) -> CodebaseProfile:
        """Profile codebase to detect size and type.

        By default, samples first 1000 files for speed (< 2 seconds).
        Use force_full_scan=True for accurate counts on large projects.

        Args:
            max_sample: Maximum files to sample (default: 1000)
            force_full_scan: Force scanning all files (slower but accurate)

        Returns:
            CodebaseProfile with detected characteristics
        """
        start_time = time.perf_counter()

        if max_sample is None:
            max_sample = self.MAX_SAMPLE_SIZE

        # Scan filesystem with sampling
        file_extensions, total_files, sampled = self._scan_files(
            max_sample=max_sample if not force_full_scan else None
        )

        # Detect size category
        size_category = self._detect_size_category(total_files)

        # Compute language distribution
        total_count = sum(file_extensions.values())
        language_distribution = {
            ext: (count / total_count * 100)
            for ext, count in file_extensions.items()
            if total_count > 0
        }

        # Get top 3 languages
        top_languages = sorted(
            language_distribution.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Detect codebase type
        codebase_type = self._detect_codebase_type(language_distribution)

        scan_time = time.perf_counter() - start_time

        profile = CodebaseProfile(
            total_files=total_files,
            size_category=size_category,
            codebase_type=codebase_type,
            language_distribution=language_distribution,
            top_languages=top_languages,
            scan_time_seconds=scan_time,
            sampled=sampled,
        )

        logger.debug(
            f"Profiled codebase in {scan_time:.2f}s: {total_files} files, "
            f"{size_category.value} size, {codebase_type.value} type"
        )

        return profile

    def _scan_files(self, max_sample: int | None = None) -> tuple[Counter, int, bool]:
        """Scan filesystem and count files by extension.

        Args:
            max_sample: Maximum files to sample (None = scan all)

        Returns:
            Tuple of (extension_counter, total_files, sampled)
            where sampled=True if we stopped early
        """
        extension_counter: Counter = Counter()
        total_files = 0
        sampled = False

        try:
            for root, dirs, files in os.walk(self.project_root):
                # Skip common ignore directories for speed
                dirs[:] = [
                    d
                    for d in dirs
                    if d
                    not in {
                        ".git",
                        "node_modules",
                        ".venv",
                        "venv",
                        "__pycache__",
                        ".pytest_cache",
                        "dist",
                        "build",
                        ".next",
                        "target",
                    }
                ]

                for file in files:
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()

                    if ext:  # Only count files with extensions
                        extension_counter[ext] += 1
                        total_files += 1

                        # Early exit if sampling
                        if max_sample and total_files >= max_sample:
                            sampled = True
                            logger.debug(
                                f"Sampled {total_files} files for profiling (early exit)"
                            )
                            return extension_counter, total_files, sampled

        except Exception as e:
            logger.warning(f"Error during filesystem scan: {e}")

        return extension_counter, total_files, sampled

    def _detect_size_category(self, total_files: int) -> CodebaseSize:
        """Detect codebase size category from file count.

        Args:
            total_files: Total number of files

        Returns:
            CodebaseSize enum value
        """
        if total_files < 1000:
            return CodebaseSize.SMALL
        elif total_files < 10000:
            return CodebaseSize.MEDIUM
        elif total_files < 50000:
            return CodebaseSize.LARGE
        else:
            return CodebaseSize.ENTERPRISE

    def _detect_codebase_type(
        self, language_distribution: dict[str, float]
    ) -> CodebaseType:
        """Detect codebase type from language distribution.

        Args:
            language_distribution: Extension -> percentage mapping

        Returns:
            CodebaseType enum value
        """
        # Check for documentation-heavy codebases (>30% docs)
        doc_percentage = sum(
            pct
            for ext, pct in language_distribution.items()
            if ext in DOCUMENTATION_EXTENSIONS
        )
        if doc_percentage > 30.0:
            return CodebaseType.DOCUMENTATION

        # Check for Python-heavy (>60% .py)
        python_percentage = language_distribution.get(".py", 0.0)
        if python_percentage > 60.0:
            return CodebaseType.PYTHON

        # Check for JavaScript/TypeScript-heavy (>60% .js/.ts/.tsx/.jsx)
        js_percentage = sum(
            language_distribution.get(ext, 0.0)
            for ext in {".js", ".ts", ".tsx", ".jsx"}
        )
        if js_percentage > 60.0:
            return CodebaseType.JAVASCRIPT

        # Check for Java-heavy (>60% .java)
        java_percentage = language_distribution.get(".java", 0.0)
        if java_percentage > 60.0:
            return CodebaseType.JAVA

        # Default to mixed/polyglot
        return CodebaseType.MIXED

    def get_optimization_preset(self, profile: CodebaseProfile) -> OptimizationPreset:
        """Get recommended optimization preset for codebase profile.

        Args:
            profile: Codebase profile from profiling

        Returns:
            OptimizationPreset with recommended settings
        """
        preset = OPTIMIZATION_PRESETS[profile.size_category]

        # Log preset selection
        logger.info(
            f"Selected optimization preset: {profile.size_category.value} "
            f"({preset.description})"
        )

        return preset

    def format_profile_summary(self, profile: CodebaseProfile) -> str:
        """Format profile as human-readable summary.

        Args:
            profile: Codebase profile to format

        Returns:
            Formatted string with profile details
        """
        lines = [
            f"📊 Codebase Profile: {profile.size_category.value.capitalize()} ({profile.total_files:,} files)",
            f"   Type: {profile.codebase_type.value.capitalize()}",
        ]

        # Add language distribution
        if profile.top_languages:
            lang_str = ", ".join(
                f"{ext[1:].upper()} {pct:.0f}%" for ext, pct in profile.top_languages
            )
            lines.append(f"   Languages: {lang_str}")

        # Add sampling note
        if profile.sampled:
            lines.append(
                f"   (Sampled first {self.MAX_SAMPLE_SIZE:,} files in {profile.scan_time_seconds:.2f}s)"
            )
        else:
            lines.append(
                f"   (Full scan completed in {profile.scan_time_seconds:.2f}s)"
            )

        return "\n".join(lines)

    def format_optimization_summary(
        self,
        profile: CodebaseProfile,
        preset: OptimizationPreset,
        previous_batch_size: int | None = None,
    ) -> str:
        """Format optimization changes as human-readable summary.

        Args:
            profile: Codebase profile
            preset: Optimization preset to apply
            previous_batch_size: Previous batch size for comparison

        Returns:
            Formatted string with optimization details
        """
        lines = ["⚡ Auto-Optimizations Applied:"]

        # Batch size
        if previous_batch_size and previous_batch_size != preset.batch_size:
            lines.append(
                f"   - Batch size: {preset.batch_size} (was {previous_batch_size})"
            )
        else:
            lines.append(f"   - Batch size: {preset.batch_size}")

        # Parallel embeddings
        parallel_status = "enabled" if preset.parallel_embeddings else "disabled"
        lines.append(f"   - Parallel embeddings: {parallel_status}")

        # File filtering
        if preset.file_extensions:
            lines.append(
                f"   - File filter: code-only (excluding docs, configs, {len(preset.file_extensions)} extensions)"
            )
        else:
            lines.append("   - File filter: all files")

        # Cache size
        lines.append(f"   - Cache size: {preset.max_cache_size:,} embeddings")

        # Time estimate (rough heuristic: 0.1s per file with optimizations)
        estimated_seconds = profile.total_files * 0.1 / preset.batch_size
        if estimated_seconds < 60:
            time_estimate = f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            time_estimate = f"{estimated_seconds / 60:.0f} minutes"
        else:
            time_estimate = f"{estimated_seconds / 3600:.1f} hours"

        lines.append(f"   - Estimated time: ~{time_estimate}")

        return "\n".join(lines)
