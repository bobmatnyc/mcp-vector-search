"""Metrics collection for code chunks during indexing."""

from pathlib import Path
from typing import Any

from loguru import logger

from ..analysis.collectors.base import MetricCollector
from ..analysis.metrics import ChunkMetrics
from .models import CodeChunk

# Extension to language mapping for metric collection
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
}


class IndexerMetricsCollector:
    """Collects metrics for code chunks during indexing.

    This class encapsulates metric collection logic including estimation
    heuristics for complexity metrics when full TreeSitter traversal is
    not performed.
    """

    def __init__(self, collectors: list[MetricCollector] | None = None) -> None:
        """Initialize metrics collector.

        Args:
            collectors: List of metric collectors to run (defaults to all complexity collectors)
        """
        self.collectors = (
            collectors if collectors is not None else self._default_collectors()
        )

    def _default_collectors(self) -> list[MetricCollector]:
        """Return default set of metric collectors.

        Returns:
            List of all complexity collectors (cognitive, cyclomatic, nesting, parameters, methods)
        """
        from ..analysis.collectors.complexity import (
            CognitiveComplexityCollector,
            CyclomaticComplexityCollector,
            MethodCountCollector,
            NestingDepthCollector,
            ParameterCountCollector,
        )

        return [
            CognitiveComplexityCollector(),
            CyclomaticComplexityCollector(),
            NestingDepthCollector(),
            ParameterCountCollector(),
            MethodCountCollector(),
        ]

    def collect_metrics_for_chunks(
        self, chunks: list[CodeChunk], file_path: Path
    ) -> dict[str, Any] | None:
        """Collect metrics for all chunks in a file.

        Args:
            chunks: List of code chunks to collect metrics for
            file_path: Path to the file being analyzed

        Returns:
            Dictionary mapping chunk_id to metrics metadata, or None if collection fails
        """
        if not self.collectors:
            return None

        try:
            # Read source code
            source_code = file_path.read_bytes()

            # Detect language from file extension
            language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "unknown")

            # Collect metrics for each chunk
            chunk_metrics = {}
            for chunk in chunks:
                metrics = self.collect_metrics(chunk, source_code, language)
                if metrics:
                    chunk_metrics[chunk.chunk_id] = metrics.to_metadata()

            logger.debug(
                f"Collected metrics for {len(chunk_metrics)} chunks from {file_path}"
            )
            return chunk_metrics

        except Exception as e:
            logger.warning(f"Failed to collect metrics for {file_path}: {e}")
            return None

    def collect_metrics(
        self, chunk: CodeChunk, source_code: bytes, language: str
    ) -> ChunkMetrics | None:
        """Collect metrics for a code chunk.

        This is a simplified version that estimates metrics from chunk content
        without full TreeSitter traversal. Future implementation will use
        TreeSitter node traversal for accurate metric collection.

        Args:
            chunk: The parsed code chunk
            source_code: Raw source code bytes
            language: Programming language identifier

        Returns:
            ChunkMetrics for the chunk, or None if no metrics collected
        """
        # For now, create basic metrics from chunk content
        # TODO: Implement full TreeSitter traversal in Phase 2
        lines_of_code = chunk.line_count

        # Estimate complexity from simple heuristics
        content = chunk.content
        cognitive_complexity = self._estimate_cognitive_complexity(content)
        cyclomatic_complexity = self._estimate_cyclomatic_complexity(content)
        max_nesting_depth = self._estimate_nesting_depth(content)
        parameter_count = len(chunk.parameters) if chunk.parameters else 0

        metrics = ChunkMetrics(
            cognitive_complexity=cognitive_complexity,
            cyclomatic_complexity=cyclomatic_complexity,
            max_nesting_depth=max_nesting_depth,
            parameter_count=parameter_count,
            lines_of_code=lines_of_code,
        )

        return metrics

    def _estimate_cognitive_complexity(self, content: str) -> int:
        """Estimate cognitive complexity from content (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated cognitive complexity score
        """
        # Simple heuristic: count control flow keywords
        keywords = [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "case",
            "when",
        ]
        complexity = 0
        for keyword in keywords:
            complexity += content.count(f" {keyword} ")
            complexity += content.count(f"\t{keyword} ")
            complexity += content.count(f"\n{keyword} ")
        return complexity

    def _estimate_cyclomatic_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity from content (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated cyclomatic complexity score (minimum 1)
        """
        # Start with baseline of 1
        complexity = 1

        # Count decision points
        keywords = [
            "if",
            "elif",
            "for",
            "while",
            "case",
            "when",
            "&&",
            "||",
            "and",
            "or",
        ]
        for keyword in keywords:
            complexity += content.count(keyword)

        return complexity

    def _estimate_nesting_depth(self, content: str) -> int:
        """Estimate maximum nesting depth from indentation (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated maximum nesting depth
        """
        max_depth = 0
        for line in content.split("\n"):
            # Count leading whitespace (4 spaces or 1 tab = 1 level)
            leading = len(line) - len(line.lstrip())
            if "\t" in line[:leading]:
                depth = line[:leading].count("\t")
            else:
                depth = leading // 4
            max_depth = max(max_depth, depth)
        return max_depth
