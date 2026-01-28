"""Analysis operation handlers for MCP vector search server."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..analysis import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    MethodCountCollector,
    NestingDepthCollector,
    ParameterCountCollector,
    ProjectMetrics,
    SmellDetector,
    SmellSeverity,
)
from ..analysis.collectors.coupling import build_import_graph
from ..analysis.interpretation import AnalysisInterpreter, LLMContextExport
from ..config.thresholds import ThresholdConfig
from ..parsers.registry import ParserRegistry


class AnalysisHandlers:
    """Handlers for code analysis-related MCP tool operations."""

    def __init__(self, project_root: Path):
        """Initialize analysis handlers.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.parser_registry = ParserRegistry()

    async def handle_analyze_project(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_project tool call.

        Args:
            args: Tool call arguments containing threshold_preset, output_format, etc.

        Returns:
            CallToolResult with project analysis results or error
        """
        threshold_preset = args.get("threshold_preset", "standard")
        output_format = args.get("output_format", "summary")

        try:
            # Load threshold configuration
            threshold_config = self._get_threshold_config(threshold_preset)

            # Find analyzable files
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, self.parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Analyze files
            collectors = self._get_collectors()
            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, self.parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception as e:
                    logger.debug(f"Failed to analyze {file_path}: {e}")
                    continue

            project_metrics.compute_aggregates()

            # Detect code smells
            smell_detector = SmellDetector(thresholds=threshold_config)
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Format response
            response_text = self._format_project_analysis(
                project_metrics, all_smells, output_format
            )

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Project analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def handle_analyze_file(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_file tool call.

        Args:
            args: Tool call arguments containing file_path

        Returns:
            CallToolResult with file analysis results or error
        """
        file_path_str = args.get("file_path", "")

        if not file_path_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = self.project_root / file_path

            if not file_path.exists():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"File not found: {file_path_str}"
                        )
                    ],
                    isError=True,
                )

            # Analyze single file
            from ..cli.commands.analyze import _analyze_file

            collectors = self._get_collectors()
            file_metrics = await _analyze_file(
                file_path, self.parser_registry, collectors
            )

            if not file_metrics:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Unable to analyze file: {file_path_str}",
                        )
                    ],
                    isError=True,
                )

            # Detect smells
            smell_detector = SmellDetector()
            smells = smell_detector.detect_all(file_metrics, str(file_path))

            # Format response
            response_text = self._format_file_analysis(file_path, file_metrics, smells)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"File analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def handle_find_smells(self, args: dict[str, Any]) -> CallToolResult:
        """Handle find_smells tool call.

        Args:
            args: Tool call arguments containing smell_type, severity filters

        Returns:
            CallToolResult with code smells or error
        """
        smell_type_filter = args.get("smell_type")
        severity_filter = args.get("severity")

        try:
            # Run full project analysis
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, self.parser_registry, None
            )

            collectors = self._get_collectors()
            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, self.parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112
                    continue

            # Detect all smells
            smell_detector = SmellDetector()
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Apply filters
            filtered_smells = self._filter_smells(
                all_smells, smell_type_filter, severity_filter
            )

            # Format response
            response_text = self._format_smells(
                filtered_smells, smell_type_filter, severity_filter
            )

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Smell detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Smell detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def handle_get_complexity_hotspots(
        self, args: dict[str, Any]
    ) -> CallToolResult:
        """Handle get_complexity_hotspots tool call.

        Args:
            args: Tool call arguments containing limit

        Returns:
            CallToolResult with complexity hotspots or error
        """
        limit = args.get("limit", 10)

        try:
            # Run full project analysis
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, self.parser_registry, None
            )

            collectors = self._get_collectors()
            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, self.parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112
                    continue

            # Get top N complex files
            hotspots = project_metrics.get_hotspots(limit=limit)

            # Format response
            response_text = self._format_hotspots(hotspots)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Hotspot detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Hotspot detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def handle_check_circular_dependencies(
        self, args: dict[str, Any]
    ) -> CallToolResult:
        """Handle check_circular_dependencies tool call.

        Args:
            args: Tool call arguments (unused)

        Returns:
            CallToolResult with circular dependency cycles or error
        """
        try:
            # Find analyzable files
            from ..cli.commands.analyze import _find_analyzable_files

            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, self.parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Build import graph
            import_graph = build_import_graph(
                self.project_root, files_to_analyze, language="python"
            )

            # Convert to forward dependency graph
            forward_graph = self._build_forward_graph(import_graph, files_to_analyze)

            # Detect cycles
            cycles = self._find_cycles(forward_graph)

            # Format response
            response_text = self._format_circular_dependencies(cycles)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Circular dependency check failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Circular dependency check failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_interpret_analysis(self, args: dict[str, Any]) -> CallToolResult:
        """Handle interpret_analysis tool call.

        Args:
            args: Tool call arguments containing analysis_json, focus, verbosity

        Returns:
            CallToolResult with interpretation or error
        """
        analysis_json_str = args.get("analysis_json", "")
        focus = args.get("focus", "summary")
        verbosity = args.get("verbosity", "normal")

        if not analysis_json_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="analysis_json parameter is required")
                ],
                isError=True,
            )

        try:
            # Parse JSON input
            analysis_data = json.loads(analysis_json_str)

            # Convert to LLMContextExport
            export = LLMContextExport(**analysis_data)

            # Create interpreter and generate interpretation
            interpreter = AnalysisInterpreter()
            interpretation = interpreter.interpret(
                export, focus=focus, verbosity=verbosity
            )

            return CallToolResult(
                content=[TextContent(type="text", text=interpretation)]
            )

        except json.JSONDecodeError as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid JSON input: {str(e)}",
                    )
                ],
                isError=True,
            )
        except Exception as e:
            logger.error(f"Analysis interpretation failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Interpretation failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    def _get_collectors(self) -> list:
        """Get list of metric collectors."""
        return [
            CognitiveComplexityCollector(),
            CyclomaticComplexityCollector(),
            NestingDepthCollector(),
            ParameterCountCollector(),
            MethodCountCollector(),
        ]

    def _get_threshold_config(self, preset: str) -> ThresholdConfig:
        """Get threshold configuration based on preset."""
        if preset == "strict":
            config = ThresholdConfig()
            config.complexity.cognitive_a = 3
            config.complexity.cognitive_b = 7
            config.complexity.cognitive_c = 15
            config.complexity.cognitive_d = 20
            config.smells.long_method_lines = 30
            config.smells.high_complexity = 10
            config.smells.too_many_parameters = 3
            config.smells.deep_nesting_depth = 3
            return config
        elif preset == "relaxed":
            config = ThresholdConfig()
            config.complexity.cognitive_a = 7
            config.complexity.cognitive_b = 15
            config.complexity.cognitive_c = 25
            config.complexity.cognitive_d = 40
            config.smells.long_method_lines = 75
            config.smells.high_complexity = 20
            config.smells.too_many_parameters = 7
            config.smells.deep_nesting_depth = 5
            return config
        else:
            return ThresholdConfig()

    def _filter_smells(
        self,
        smells: list,
        smell_type_filter: str | None,
        severity_filter: str | None,
    ) -> list:
        """Filter smells by type and severity."""
        filtered = smells

        if smell_type_filter:
            filtered = [s for s in filtered if s.name == smell_type_filter]

        if severity_filter:
            severity_enum = SmellSeverity(severity_filter)
            filtered = [s for s in filtered if s.severity == severity_enum]

        return filtered

    def _build_forward_graph(
        self, import_graph: dict, files_to_analyze: list
    ) -> dict[str, list[str]]:
        """Build forward dependency graph from import graph."""
        forward_graph: dict[str, list[str]] = {}

        for file_path in files_to_analyze:
            file_str = str(file_path.relative_to(self.project_root))
            if file_str not in forward_graph:
                forward_graph[file_str] = []

            for module, importers in import_graph.items():
                for importer in importers:
                    importer_str = str(
                        Path(importer).relative_to(self.project_root)
                        if Path(importer).is_absolute()
                        else importer
                    )
                    if importer_str == file_str:
                        if module not in forward_graph[file_str]:
                            forward_graph[file_str].append(module)

        return forward_graph

    def _find_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
        """Find all cycles in the import graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    try:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycle_tuple = tuple(sorted(cycle))
                        if not any(tuple(sorted(c)) == cycle_tuple for c in cycles):
                            cycles.append(cycle)
                    except ValueError:
                        pass

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _format_project_analysis(
        self, project_metrics: ProjectMetrics, smells: list, output_format: str
    ) -> str:
        """Format project analysis results."""
        if output_format == "detailed":
            output = project_metrics.to_summary()
            output["smells"] = {
                "total": len(smells),
                "by_severity": {
                    "error": sum(
                        1 for s in smells if s.severity == SmellSeverity.ERROR
                    ),
                    "warning": sum(
                        1 for s in smells if s.severity == SmellSeverity.WARNING
                    ),
                    "info": sum(1 for s in smells if s.severity == SmellSeverity.INFO),
                },
            }
            return json.dumps(output, indent=2)

        # Summary format
        summary = project_metrics.to_summary()
        response_lines = [
            "# Project Analysis Summary\n",
            f"**Project Root:** {summary['project_root']}",
            f"**Total Files:** {summary['total_files']}",
            f"**Total Functions:** {summary['total_functions']}",
            f"**Total Classes:** {summary['total_classes']}",
            f"**Average File Complexity:** {summary['avg_file_complexity']}\n",
            "## Complexity Distribution",
        ]

        dist = summary["complexity_distribution"]
        for grade in ["A", "B", "C", "D", "F"]:
            response_lines.append(f"- Grade {grade}: {dist[grade]} chunks")

        response_lines.extend(
            [
                "\n## Health Metrics",
                f"- Average Health Score: {summary['health_metrics']['avg_health_score']:.2f}",
                f"- Files Needing Attention: {summary['health_metrics']['files_needing_attention']}",
                "\n## Code Smells",
                f"- Total: {len(smells)}",
                f"- Errors: {sum(1 for s in smells if s.severity == SmellSeverity.ERROR)}",
                f"- Warnings: {sum(1 for s in smells if s.severity == SmellSeverity.WARNING)}",
                f"- Info: {sum(1 for s in smells if s.severity == SmellSeverity.INFO)}",
            ]
        )

        return "\n".join(response_lines)

    def _format_file_analysis(self, file_path: Path, file_metrics, smells: list) -> str:
        """Format file analysis results."""
        response_lines = [
            f"# File Analysis: {file_path.name}\n",
            f"**Path:** {file_path}",
            f"**Total Lines:** {file_metrics.total_lines}",
            f"**Code Lines:** {file_metrics.code_lines}",
            f"**Comment Lines:** {file_metrics.comment_lines}",
            f"**Functions:** {file_metrics.function_count}",
            f"**Classes:** {file_metrics.class_count}",
            f"**Methods:** {file_metrics.method_count}\n",
            "## Complexity Metrics",
            f"- Total Complexity: {file_metrics.total_complexity}",
            f"- Average Complexity: {file_metrics.avg_complexity:.2f}",
            f"- Max Complexity: {file_metrics.max_complexity}",
            f"- Health Score: {file_metrics.health_score:.2f}\n",
        ]

        if smells:
            response_lines.append(f"## Code Smells ({len(smells)})\n")
            for smell in smells[:10]:
                response_lines.append(
                    f"- [{smell.severity.value.upper()}] {smell.name}: {smell.description}"
                )
            if len(smells) > 10:
                response_lines.append(f"\n... and {len(smells) - 10} more")
        else:
            response_lines.append("## Code Smells\n- None detected")

        return "\n".join(response_lines)

    def _format_smells(
        self,
        smells: list,
        smell_type_filter: str | None,
        severity_filter: str | None,
    ) -> str:
        """Format code smells results."""
        if not smells:
            filter_desc = []
            if smell_type_filter:
                filter_desc.append(f"type={smell_type_filter}")
            if severity_filter:
                filter_desc.append(f"severity={severity_filter}")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            return f"No code smells found{filter_str}"

        response_lines = [f"# Code Smells Found: {len(smells)}\n"]

        # Group by severity
        by_severity = {
            "error": [s for s in smells if s.severity == SmellSeverity.ERROR],
            "warning": [s for s in smells if s.severity == SmellSeverity.WARNING],
            "info": [s for s in smells if s.severity == SmellSeverity.INFO],
        }

        for severity_level in ["error", "warning", "info"]:
            severity_smells = by_severity[severity_level]
            if severity_smells:
                response_lines.append(
                    f"## {severity_level.upper()} ({len(severity_smells)})\n"
                )
                for smell in severity_smells[:20]:
                    response_lines.append(f"- **{smell.name}** at `{smell.location}`")
                    response_lines.append(f"  {smell.description}")
                    if smell.suggestion:
                        response_lines.append(f"  *Suggestion: {smell.suggestion}*")
                    response_lines.append("")

        return "\n".join(response_lines)

    def _format_hotspots(self, hotspots: list) -> str:
        """Format complexity hotspots results."""
        if not hotspots:
            return "No complexity hotspots found"

        response_lines = [f"# Top {len(hotspots)} Complexity Hotspots\n"]

        for i, file_metrics in enumerate(hotspots, 1):
            response_lines.extend(
                [
                    f"## {i}. {Path(file_metrics.file_path).name}",
                    f"**Path:** `{file_metrics.file_path}`",
                    f"**Average Complexity:** {file_metrics.avg_complexity:.2f}",
                    f"**Max Complexity:** {file_metrics.max_complexity}",
                    f"**Total Complexity:** {file_metrics.total_complexity}",
                    f"**Functions:** {file_metrics.function_count}",
                    f"**Health Score:** {file_metrics.health_score:.2f}\n",
                ]
            )

        return "\n".join(response_lines)

    def _format_circular_dependencies(self, cycles: list) -> str:
        """Format circular dependencies results."""
        if not cycles:
            return "No circular dependencies detected"

        response_lines = [f"# Circular Dependencies Found: {len(cycles)}\n"]

        for i, cycle in enumerate(cycles, 1):
            response_lines.append(f"## Cycle {i}")
            response_lines.append("```")
            for j, node in enumerate(cycle):
                if j < len(cycle) - 1:
                    response_lines.append(f"{node}")
                    response_lines.append("  ↓")
                else:
                    response_lines.append(f"{node} (back to {cycle[0]})")
            response_lines.append("```\n")

        return "\n".join(response_lines)
