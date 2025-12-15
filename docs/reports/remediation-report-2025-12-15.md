# Code Remediation Report
Generated: 2025-12-15

## Summary

- **High Complexity Items**: 86
- **Code Smells Detected**: 1465
- **Critical Issues (Errors)**: 916

---

## 🔴 Priority: High Complexity Code

These functions/methods have complexity scores that make them difficult to maintain and test.

| Grade | Name | File | Complexity | Lines |
|-------|------|------|------------|-------|
| 🟠 D | `SearchQualityAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 40 | 538 |
| 🟠 D | `BaselineManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 37 | 507 |
| 🟠 D | `EfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 39 | 232 |
| 🟠 D | `MetricsStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 33 | 574 |
| 🟠 D | `_process_answer_query` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 33 | 320 |
| 🟠 D | `run_chat_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 31 | 226 |
| 🟠 D | `VisualizationState` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | 32 | 361 |
| 🟠 D | `DirectoryIndex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | 34 | 306 |
| 🟠 D | `ProjectManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | 36 | 326 |
| 🟠 D | `RelationshipStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 31 | 280 |
| 🟠 D | `SchedulerManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | 31 | 320 |
| 🟠 D | `_extract_class_skeleton` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 39 | 129 |
| 🟠 D | `_extract_class_skeleton_regex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 31 | 144 |
| 🟠 D | `test_visualization` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_visualization_playwright.py | 33 | 340 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 27 | 199 |
| 🟡 C | `BuildManager` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | 22 | 243 |
| 🟡 C | `SearchPerformanceMonitor` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | 28 | 294 |
| 🟡 C | `DocumentationUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | 29 | 213 |
| 🟡 C | `VersionManager` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | 30 | 240 |
| 🟡 C | `_extract_import` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 28 | 84 |
| 🟡 C | `EnhancedJSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 28 | 326 |
| 🟡 C | `AnalysisInterpreter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 21 | 187 |
| 🟡 C | `print_baseline_comparison` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 26 | 198 |
| 🟡 C | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 26 | 413 |
| 🟡 C | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/trends.py | 25 | 231 |
| 🟡 C | `HTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 26 | 2857 |
| 🟡 C | `_find_analyzable_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 23 | 124 |
| 🟡 C | `_run_batch_indexing` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 26 | 241 |
| 🟡 C | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | 27 | 288 |
| 🟡 C | `reset_index` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | 27 | 158 |
| 🟡 C | `setup_llm_api_keys` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 29 | 208 |
| 🟡 C | `setup_openrouter_api_key` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 23 | 179 |
| 🟡 C | `_display_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 21 | 126 |
| 🟡 C | `SearchResultExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 26 | 272 |
| 🟡 C | `SearchHistory` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/history.py | 23 | 209 |
| 🟡 C | `print_search_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/output.py | 26 | 142 |
| 🟡 C | `ContextualSuggestionProvider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 30 | 339 |
| 🟡 C | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 22 | 90 |
| 🟡 C | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 21 | 68 |
| 🟡 C | `GitManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | 30 | 326 |
| 🟡 C | `GitHookManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git_hooks.py | 30 | 248 |
| 🟡 C | `_build_chunk_hierarchy` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 22 | 108 |
| 🟡 C | `_check_circular_dependencies` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 21 | 129 |
| 🟡 C | `HTMLContentParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 25 | 182 |
| 🟡 C | `HTMLParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 25 | 220 |
| 🟡 C | `_extract_phpdoc_regex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 23 | 50 |
| 🟡 C | `TextParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | 22 | 177 |
| 🟡 C | `extract_docstring` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/utils.py | 21 | 75 |
| 🟡 C | `TestCLICommands` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | 30 | 495 |
| 🟡 C | `run_breadcrumb_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_breadcrumb_fix.py | 22 | 198 |
| 🟡 C | `comprehensive_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_final_comprehensive.py | 30 | 262 |
| 🟡 C | `main` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer.py | 21 | 95 |
| 🟡 C | `TestChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_connection_pool.py | 25 | 368 |
| 🟡 C | `TestSemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_search.py | 21 | 436 |
| 🟡 C | `CommandBuilder` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 27 | 419 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 78 | 567 |
| 🔴 F | `ConnectionManager` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 102 | 1083 |
| 🔴 F | `executeCommandOnTab` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 43 | 385 |
| 🔴 F | `ChangesetManager` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 47 | 318 |
| 🔴 F | `HomebrewFormulaUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 67 | 610 |
| 🔴 F | `LCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | 45 | 313 |
| 🔴 F | `ConsoleReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 64 | 631 |
| 🔴 F | `MarkdownReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | 43 | 462 |
| 🔴 F | `run_analysis` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 59 | 388 |
| 🔴 F | `run_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | 55 | 287 |
| 🔴 F | `_run_smart_setup` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 45 | 232 |
| 🔴 F | `build_graph_data` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | 62 | 334 |
| 🔴 F | `create_app` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py | 67 | 499 |
| 🔴 F | `InteractiveSearchSession` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | 44 | 316 |
| 🔴 F | `ChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/connection_pool.py | 42 | 323 |
| 🔴 F | `ChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 105 | 881 |
| 🔴 F | `PooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 75 | 511 |
| 🔴 F | `SemanticIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 134 | 1236 |
| 🔴 F | `LLMClient` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 75 | 736 |
| 🔴 F | `SemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 112 | 1082 |
| 🔴 F | `MCPVectorSearchServer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 129 | 1417 |
| 🔴 F | `DartParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | 73 | 594 |
| 🔴 F | `JavaScriptParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | 81 | 607 |
| 🔴 F | `PHPParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 93 | 683 |
| 🔴 F | `PythonParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 134 | 771 |
| 🔴 F | `RubyParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | 82 | 667 |
| 🔴 F | `MonorepoDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/monorepo.py | 44 | 294 |
| 🔴 F | `ConfigManager` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 50 | 500 |
| 🔴 F | `MCPInstaller` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 50 | 592 |
| 🔴 F | `MCPInspector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 55 | 609 |
| 🔴 F | `PlatformDetector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 56 | 478 |

---

## 🔍 Code Smells

### Critical Issues (Errors)

| Smell | Name | File | Detail |
|-------|------|------|--------|
| 🔴 Deep Nesting | `CLAUDE_20251009_pre_mpm_init.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/CLAUDE_20251009_pre_mpm_init.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CLAUDE_MPM_INIT_SUMMARY_20251009.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/CLAUDE_MPM_INIT_SUMMARY_20251009.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `DEPLOY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/DEPLOY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ENGINEER_TASK.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/ENGINEER_TASK.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `compact_folder_layout_deliverables.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/compact_folder_layout_deliverables.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `compact_folder_layout_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/compact_folder_layout_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/README.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_compact_folders_implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_compact_folders_implementation.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_compact_layout_fixes.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_compact_layout_fixes.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_enhancements_code_review.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_enhancements_code_review.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_enhancements_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_enhancements_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_implementation_plan.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_implementation_plan.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_implementation_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_implementation_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_improvements_spec.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_improvements_spec.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_test_report.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_test_report.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_verification_report.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_verification_report.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `INSTALL.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/INSTALL.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `MPM_INIT_EXECUTIVE_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/MPM_INIT_EXECUTIVE_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `BREADCRUMB_BUG_TEST_REPORT.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/BREADCRUMB_BUG_TEST_REPORT.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `breadcrumb_navigation_implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/breadcrumb_navigation_implementation.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `code_chunks_feature_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/code_chunks_feature_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `code_chunks_user_guide.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/code_chunks_user_guide.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `CRITICAL_BUG_REPORT_visibleNodes_initialization.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/CRITICAL_BUG_REPORT_visibleNodes_initialization.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `cytoscape_fix_verification.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/cytoscape_fix_verification.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `DOCUMENTATION_REORGANIZATION_COMPLETE.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/DOCUMENTATION_REORGANIZATION_COMPLETE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_INTEGRATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/HOMEBREW_INTEGRATION_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_INTEGRATION_USAGE.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/HOMEBREW_INTEGRATION_USAGE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_TAP_UPDATE_STATUS.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/HOMEBREW_TAP_UPDATE_STATUS.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_TAP_UPDATE_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/HOMEBREW_TAP_UPDATE_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_TEST_RESULTS.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/HOMEBREW_TEST_RESULTS.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `INVESTIGATION_REPORT_empty_nodes.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/INVESTIGATION_REPORT_empty_nodes.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `LAYOUT_TEST_FAILURE_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/LAYOUT_TEST_FAILURE_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `llm-benchmark-implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/llm-benchmark-implementation.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `MCP_AUTO_INSTALLATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/MCP_AUTO_INSTALLATION_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `mcp_installer_migration_complete.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/mcp_installer_migration_complete.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `mcp_installer_migration_plan.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/mcp_installer_migration_plan.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `PERFORMANCE_OPTIMIZATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/PERFORMANCE_OPTIMIZATION_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `PROJECT_CLEANUP_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/PROJECT_CLEANUP_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `QUICK_VERIFICATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/QUICK_VERIFICATION_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/README.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `REORGANIZATION_PLAN.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/REORGANIZATION_PLAN.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `REORGANIZATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/REORGANIZATION_SUMMARY.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `submodule_integration_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/submodule_integration_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_automatic_spacing_spinner_implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_automatic_spacing_spinner_implementation.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_cli_fix_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_cli_fix_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_enhancements_implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_enhancements_implementation.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_enhancements_test_results.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_enhancements_test_results.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_enhancements_user_guide.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_enhancements_user_guide.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_optimization_summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization_optimization_summary.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `visualization-controls-diagnosis.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/visualization-controls-diagnosis.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test-suite-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/test-suite-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `testing-strategy.md` | /Users/masa/Projects/mcp-vector-search/docs/_archive/testing-strategy.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/advanced/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `troubleshooting.md` | /Users/masa/Projects/mcp-vector-search/docs/advanced/troubleshooting.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `database-optimization.md` | /Users/masa/Projects/mcp-vector-search/docs/architecture/database-optimization.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `indexing-workflow.md` | /Users/masa/Projects/mcp-vector-search/docs/architecture/indexing-workflow.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `performance.md` | /Users/masa/Projects/mcp-vector-search/docs/architecture/performance.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/architecture/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/configuration/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/CHANGELOG.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_INTEGRATION.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/HOMEBREW_INTEGRATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `HOMEBREW_QUICKSTART.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/HOMEBREW_QUICKSTART.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASES.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/RELEASES.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VERSIONING_WORKFLOW.md` | /Users/masa/Projects/mcp-vector-search/docs/deployment/VERSIONING_WORKFLOW.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `api.md` | /Users/masa/Projects/mcp-vector-search/docs/development/api.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `architecture.md` | /Users/masa/Projects/mcp-vector-search/docs/development/architecture.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ast_circular_dependency_fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/ast_circular_dependency_fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `BUGFIX_VISUALIZATION_DATA_INIT.md` | /Users/masa/Projects/mcp-vector-search/docs/development/BUGFIX_VISUALIZATION_DATA_INIT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-rust-panic-defense.md` | /Users/masa/Projects/mcp-vector-search/docs/development/chromadb-rust-panic-defense.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-rust-panic-recovery.md` | /Users/masa/Projects/mcp-vector-search/docs/development/chromadb-rust-panic-recovery.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `code-quality.md` | /Users/masa/Projects/mcp-vector-search/docs/development/code-quality.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `contributing.md` | /Users/masa/Projects/mcp-vector-search/docs/development/contributing.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `crash-diagnostics.md` | /Users/masa/Projects/mcp-vector-search/docs/development/crash-diagnostics.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `d3js-tree-integration.md` | /Users/masa/Projects/mcp-vector-search/docs/development/d3js-tree-integration.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `dependency-graph.txt` | /Users/masa/Projects/mcp-vector-search/docs/development/dependency-graph.txt | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `duplicate-node-rendering-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/duplicate-node-rendering-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `fix-visibleNodes-initialization.md` | /Users/masa/Projects/mcp-vector-search/docs/development/fix-visibleNodes-initialization.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `FULL_TREE_FAN_VISUALIZATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/FULL_TREE_FAN_VISUALIZATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `git-integration-implementation.md` | /Users/masa/Projects/mcp-vector-search/docs/development/git-integration-implementation.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `github-milestones-setup.md` | /Users/masa/Projects/mcp-vector-search/docs/development/github-milestones-setup.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `glob-pattern-fix-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/development/glob-pattern-fix-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `hybrid-visualization-frontend-integration.md` | /Users/masa/Projects/mcp-vector-search/docs/development/hybrid-visualization-frontend-integration.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `MCP_AUTO_INSTALLATION_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/MCP_AUTO_INSTALLATION_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-installation-bug-fix-2025-12-01.md` | /Users/masa/Projects/mcp-vector-search/docs/development/mcp-installation-bug-fix-2025-12-01.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-installer-platform-forcing-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/mcp-installer-platform-forcing-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `MILESTONES_QUICKSTART.md` | /Users/masa/Projects/mcp-vector-search/docs/development/MILESTONES_QUICKSTART.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `monorepo-detection-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/monorepo-detection-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `OPENAI_API_INTEGRATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/OPENAI_API_INTEGRATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `openrouter-setup-demo.md` | /Users/masa/Projects/mcp-vector-search/docs/development/openrouter-setup-demo.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `openrouter-setup-enhancement.md` | /Users/masa/Projects/mcp-vector-search/docs/development/openrouter-setup-enhancement.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `phase-5-implementation-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/development/phase-5-implementation-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `PHASE1_STATE_MANAGEMENT_COMPLETE.md` | /Users/masa/Projects/mcp-vector-search/docs/development/PHASE1_STATE_MANAGEMENT_COMPLETE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `pr-workflow-guide.md` | /Users/masa/Projects/mcp-vector-search/docs/development/pr-workflow-guide.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `project-organization.md` | /Users/masa/Projects/mcp-vector-search/docs/development/project-organization.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/development/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ROOT_BREADCRUMB_FIX.md` | /Users/masa/Projects/mcp-vector-search/docs/development/ROOT_BREADCRUMB_FIX.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `root-node-filtering-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/root-node-filtering-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `setup-api-key-interactive-prompt-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/development/setup-api-key-interactive-prompt-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `setup.md` | /Users/masa/Projects/mcp-vector-search/docs/development/setup.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `sprint-board.md` | /Users/masa/Projects/mcp-vector-search/docs/development/sprint-board.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `sprint-plan-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/development/sprint-plan-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `sprint-plan.md` | /Users/masa/Projects/mcp-vector-search/docs/development/sprint-plan.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `sprint-quickstart.md` | /Users/masa/Projects/mcp-vector-search/docs/development/sprint-quickstart.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `STREAMING_JSON_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/STREAMING_JSON_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `testing.md` | /Users/masa/Projects/mcp-vector-search/docs/development/testing.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TREE_LAYOUT_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/TREE_LAYOUT_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `tree-navigation-test-guide.md` | /Users/masa/Projects/mcp-vector-search/docs/development/tree-navigation-test-guide.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TWO_PHASE_LAYOUT_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-vector-search/docs/development/TWO_PHASE_LAYOUT_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `versioning.md` | /Users/masa/Projects/mcp-vector-search/docs/development/versioning.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_ARCHITECTURE_V2_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_ARCHITECTURE_V2_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_ARCHITECTURE_V2.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_ARCHITECTURE_V2.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_V2_CHECKLIST.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_V2_DIAGRAMS.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_DIAGRAMS.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_V2_IMPLEMENTATION_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_IMPLEMENTATION_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_V2_INDEX.md` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_INDEX.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `benchmark-output-example.md` | /Users/masa/Projects/mcp-vector-search/docs/examples/benchmark-output-example.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/examples/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `configuration.md` | /Users/masa/Projects/mcp-vector-search/docs/getting-started/configuration.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `first-steps.md` | /Users/masa/Projects/mcp-vector-search/docs/getting-started/first-steps.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `installation.md` | /Users/masa/Projects/mcp-vector-search/docs/getting-started/installation.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/getting-started/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `api-key-storage.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/api-key-storage.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CHAT_COMMAND_SETUP.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/CHAT_COMMAND_SETUP.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chat-command.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/chat-command.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `cli-usage.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/cli-usage.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `DEPLOYMENT.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/DEPLOYMENT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `file-watching.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/file-watching.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `git-aware-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/git-aware-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `indexing.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/indexing.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `llm-benchmarking.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/llm-benchmarking.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `MCP_AUTO_INSTALLATION.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/MCP_AUTO_INSTALLATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-integration.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/mcp-integration.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `searching.md` | /Users/masa/Projects/mcp-vector-search/docs/guides/searching.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `hyperdev-2025-12.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/hyperdev-2025-12.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `improvements.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/improvements.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `install-enhancements.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/install-enhancements.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `refactoring-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/refactoring-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `search-analysis-report.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/search-analysis-report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `search-bug-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/search-bug-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `search-improvement-plan.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/search-improvement-plan.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `search-timing-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/search-timing-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `similarity-calculation-fix.md` | /Users/masa/Projects/mcp-vector-search/docs/internal/similarity-calculation-fix.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp_vector_search_prd_updated.md` | /Users/masa/Projects/mcp-vector-search/docs/prd/mcp_vector_search_prd_updated.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/prd/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/projects/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `structural-code-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/projects/structural-code-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-rust-panic-fix-verification.md` | /Users/masa/Projects/mcp-vector-search/docs/qa/chromadb-rust-panic-fix-verification.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `architecture.md` | /Users/masa/Projects/mcp-vector-search/docs/reference/architecture.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `cli-commands.md` | /Users/masa/Projects/mcp-vector-search/docs/reference/cli-commands.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `features.md` | /Users/masa/Projects/mcp-vector-search/docs/reference/features.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `MCP_SETUP.md` | /Users/masa/Projects/mcp-vector-search/docs/reference/MCP_SETUP.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/reference/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `security_scan_report_2025-12-02.txt` | /Users/masa/Projects/mcp-vector-search/docs/reports/security_scan_report_2025-12-02.txt | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `analyze-command-implementation-research-2024-12-10.md` | /Users/masa/Projects/mcp-vector-search/docs/research/analyze-command-implementation-research-2024-12-10.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `automatic-setup-command-design-2025-11-30.md` | /Users/masa/Projects/mcp-vector-search/docs/research/automatic-setup-command-design-2025-11-30.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-baseline-storage-investigation.md` | /Users/masa/Projects/mcp-vector-search/docs/research/chromadb-baseline-storage-investigation.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-rust-panic-indexing-analysis-2025-12-10.md` | /Users/masa/Projects/mcp-vector-search/docs/research/chromadb-rust-panic-indexing-analysis-2025-12-10.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chromadb-rust-panic-investigation-2025-12-10.md` | /Users/masa/Projects/mcp-vector-search/docs/research/chromadb-rust-panic-investigation-2025-12-10.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chunking-behavior-analysis-2024-12-14.md` | /Users/masa/Projects/mcp-vector-search/docs/research/chunking-behavior-analysis-2024-12-14.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `chunking-behavior-analysis-2025-12-14.md` | /Users/masa/Projects/mcp-vector-search/docs/research/chunking-behavior-analysis-2025-12-14.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `class-chunking-behavior-analysis-2025-12-14.md` | /Users/masa/Projects/mcp-vector-search/docs/research/class-chunking-behavior-analysis-2025-12-14.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `claude-desktop-documentation-review-2025-12-02.md` | /Users/masa/Projects/mcp-vector-search/docs/research/claude-desktop-documentation-review-2025-12-02.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `claude-desktop-vs-code-installer-analysis-2025-12-02.md` | /Users/masa/Projects/mcp-vector-search/docs/research/claude-desktop-vs-code-installer-analysis-2025-12-02.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `click-handler-debug-instructions-2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/research/click-handler-debug-instructions-2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `code-graph-visualization-best-practices-2025-12-05.md` | /Users/masa/Projects/mcp-vector-search/docs/research/code-graph-visualization-best-practices-2025-12-05.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `cycle-detection-analysis-2025-12-06.md` | /Users/masa/Projects/mcp-vector-search/docs/research/cycle-detection-analysis-2025-12-06.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `cytoscape-edge-error-investigation.md` | /Users/masa/Projects/mcp-vector-search/docs/research/cytoscape-edge-error-investigation.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `d3-automatic-spacing-research-2025-12-05.md` | /Users/masa/Projects/mcp-vector-search/docs/research/d3-automatic-spacing-research-2025-12-05.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `dotfile-configuration-research-2025-12-08.md` | /Users/masa/Projects/mcp-vector-search/docs/research/dotfile-configuration-research-2025-12-08.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `gitignore-auto-update-implementation-2025-11-25.md` | /Users/masa/Projects/mcp-vector-search/docs/research/gitignore-auto-update-implementation-2025-11-25.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `homebrew-token-investigation-2025-11-25.md` | /Users/masa/Projects/mcp-vector-search/docs/research/homebrew-token-investigation-2025-11-25.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `issue-14-code-smell-detection-requirements.md` | /Users/masa/Projects/mcp-vector-search/docs/research/issue-14-code-smell-detection-requirements.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `issue-17-diff-aware-analysis-research.md` | /Users/masa/Projects/mcp-vector-search/docs/research/issue-17-diff-aware-analysis-research.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `issue-18-baseline-comparison-requirements.md` | /Users/masa/Projects/mcp-vector-search/docs/research/issue-18-baseline-comparison-requirements.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `llm-controller-architecture-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/research/llm-controller-architecture-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-installation-bug-analysis-2025-12-01.md` | /Users/masa/Projects/mcp-vector-search/docs/research/mcp-installation-bug-analysis-2025-12-01.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-vector-search-structural-analysis-design.md` | /Users/masa/Projects/mcp-vector-search/docs/research/mcp-vector-search-structural-analysis-design.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `phase-4-visualization-export-plan.md` | /Users/masa/Projects/mcp-vector-search/docs/research/phase-4-visualization-export-plan.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `phase1-completion-status-2025-12-11.md` | /Users/masa/Projects/mcp-vector-search/docs/research/phase1-completion-status-2025-12-11.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `phase3-cross-file-analysis-requirements.md` | /Users/masa/Projects/mcp-vector-search/docs/research/phase3-cross-file-analysis-requirements.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `phase3-implementation-review-2025-12-11.md` | /Users/masa/Projects/mcp-vector-search/docs/research/phase3-implementation-review-2025-12-11.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `pre-release-readiness-check-2025-12-02.md` | /Users/masa/Projects/mcp-vector-search/docs/research/pre-release-readiness-check-2025-12-02.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `QUICK_FIX.md` | /Users/masa/Projects/mcp-vector-search/docs/research/QUICK_FIX.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/docs/research/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `sarif-output-format-requirements-2025-12-11.md` | /Users/masa/Projects/mcp-vector-search/docs/research/sarif-output-format-requirements-2025-12-11.md | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `json` | /Users/masa/Projects/mcp-vector-search/docs/research/sarif-output-format-requirements-2025-12-11.md | 135 lines (recommended: <50) |
| 🔴 Deep Nesting | `search-filter-file-path-issue-analysis-2025-12-08.md` | /Users/masa/Projects/mcp-vector-search/docs/research/search-filter-file-path-issue-analysis-2025-12-08.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `search-filtering-boilerplate-names-2025-12-10.md` | /Users/masa/Projects/mcp-vector-search/docs/research/search-filtering-boilerplate-names-2025-12-10.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `setup-command-design-2025-11-25.md` | /Users/masa/Projects/mcp-vector-search/docs/research/setup-command-design-2025-11-25.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `setup-installation-mcp-integration-analysis-2025-11-30.md` | /Users/masa/Projects/mcp-vector-search/docs/research/setup-installation-mcp-integration-analysis-2025-11-30.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `single-child-chain-collapsing-analysis.md` | /Users/masa/Projects/mcp-vector-search/docs/research/single-child-chain-collapsing-analysis.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_bugs_found.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_bugs_found.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_debug_report.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_debug_report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_layout_test_report.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_layout_test_report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_test_report.json` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_test_report.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_test_report.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_test_report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization_uat_report.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization_uat_report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization-architecture-analysis-2025-12-06.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization-architecture-analysis-2025-12-06.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization-layout-configuration-analysis-2025-12-08.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization-layout-configuration-analysis-2025-12-08.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization-layout-confusion-analysis-2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization-layout-confusion-analysis-2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualization-server-startup-performance-issue-2025-12-08.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualization-server-startup-performance-issue-2025-12-08.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualizer-filtering-analysis-2025-12-04.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualizer-filtering-analysis-2025-12-04.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `visualizer-issue-analysis-2025-12-03.md` | /Users/masa/Projects/mcp-vector-search/docs/research/visualizer-issue-analysis-2025-12-03.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CHUNK_TREE_EXPANSION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/CHUNK_TREE_EXPANSION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CHUNK_TREE_VERIFICATION_CHECKLIST.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/CHUNK_TREE_VERIFICATION_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CLICK_HANDLER_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/CLICK_HANDLER_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CLICK_HANDLER_FIX_VERIFICATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/CLICK_HANDLER_FIX_VERIFICATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_BROKEN_UAT_REPORT_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_BROKEN_UAT_REPORT_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_COLLAPSE_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_COLLAPSE_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_CONNECTOR_LINES_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_CONNECTOR_LINES_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_FIXES_VERIFICATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_FIXES_VERIFICATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_LAYOUT_IMPLEMENTATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_LAYOUT_IMPLEMENTATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_SPACING_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_SPACING_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `D3_TREE_VERIFICATION_CHECKLIST.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/D3_TREE_VERIFICATION_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `DUPLICATE_CHUNKTYPES_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/DUPLICATE_CHUNKTYPES_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `FILE_CLICK_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/FILE_CLICK_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `github-milestones-setup-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/github-milestones-setup-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `HIERARCHY_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/HIERARCHY_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `issue-18-baseline-comparison-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/issue-18-baseline-comparison-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `NULL_CHECK_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/NULL_CHECK_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `py-mcp-installer-integration-test-results.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/py-mcp-installer-integration-test-results.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `root-node-filtering-fix-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/root-node-filtering-fix-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `SIMPLE_D3_TREE_REWRITE_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/SIMPLE_D3_TREE_REWRITE_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `structural-analysis-github-issues-summary.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/structural-analysis-github-issues-summary.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `structural-analysis-quick-reference.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/structural-analysis-quick-reference.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TEXT_LABEL_COLOR_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TEXT_LABEL_COLOR_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TEXT_LABEL_VERIFICATION_CHECKLIST.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TEXT_LABEL_VERIFICATION_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TOGGLE_SWITCH_IMPLEMENTATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TOGGLE_SWITCH_IMPLEMENTATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TREE_COMPACT_SPACING_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TREE_COMPACT_SPACING_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TREE_FIX_VERIFICATION_CHECKLIST.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TREE_FIX_VERIFICATION_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TREE_HIERARCHY_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TREE_HIERARCHY_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TREE_LAYOUT_VISUAL_IMPROVEMENTS_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/TREE_LAYOUT_VISUAL_IMPROVEMENTS_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUAL_IMPROVEMENTS_VERIFICATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/VISUAL_IMPROVEMENTS_VERIFICATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_FIX_VERIFICATION_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/VISUALIZATION_FIX_VERIFICATION_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_NODE_CLICK_FIX_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/VISUALIZATION_NODE_CLICK_FIX_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VISUALIZATION_VIEWER_PANEL_2025-12-09.md` | /Users/masa/Projects/mcp-vector-search/docs/summaries/VISUALIZATION_VIEWER_PANEL_2025-12-09.md | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `demonstrate_connection_pooling` | /Users/masa/Projects/mcp-vector-search/examples/connection_pooling_example.py | 265 lines (recommended: <50) |
| 🔴 Long Method | `demonstrate_semi_automatic_reindexing` | /Users/masa/Projects/mcp-vector-search/examples/semi_automatic_reindexing_demo.py | 262 lines (recommended: <50) |
| 🔴 Deep Nesting | `background-enhanced.js` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `PortSelector` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 199 lines (recommended: <50) |
| 🔴 Long Method | `ConnectionManager` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 1083 lines (recommended: <50) |
| 🔴 High Complexity | `ConnectionManager` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | Complexity: 102 (recommended: <15) |
| 🔴 God Class | `ConnectionManager` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 1083 lines - consider breaking into smaller classes |
| 🔴 Long Method | `connectToBackend` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 118 lines (recommended: <50) |
| 🔴 Long Method | `_setupMessageHandler` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 110 lines (recommended: <50) |
| 🔴 Long Method | `setupWebSocketHandlers` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 140 lines (recommended: <50) |
| 🔴 Long Method | `executeCommandOnTab` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 385 lines (recommended: <50) |
| 🔴 High Complexity | `executeCommandOnTab` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | Complexity: 43 (recommended: <15) |
| 🔴 Deep Nesting | `content.js` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/content.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/manifest.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `popup-enhanced.js` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/popup-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Long Method | `PerformanceAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/analyze_search_bottlenecks.py | 255 lines (recommended: <50) |
| 🔴 Long Method | `benchmark_model` | /Users/masa/Projects/mcp-vector-search/scripts/benchmark_llm_models.py | 146 lines (recommended: <50) |
| 🔴 Long Method | `Changeset` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 104 lines (recommended: <50) |
| 🔴 Long Method | `ChangesetManager` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 318 lines (recommended: <50) |
| 🔴 High Complexity | `ChangesetManager` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | Complexity: 47 (recommended: <15) |
| 🔴 God Class | `ChangesetManager` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 318 lines - consider breaking into smaller classes |
| 🔴 Long Method | `BuildManager` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | 243 lines (recommended: <50) |
| 🔴 Long Method | `migrate_metrics` | /Users/masa/Projects/mcp-vector-search/scripts/migrate_chromadb_metrics.py | 136 lines (recommended: <50) |
| 🔴 Long Method | `PerformanceMonitor` | /Users/masa/Projects/mcp-vector-search/scripts/monitor_search_performance.py | 118 lines (recommended: <50) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/quick_search_timing.py | 221 lines (recommended: <50) |
| 🔴 Long Method | `SearchTimingTestSuite` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 495 lines (recommended: <50) |
| 🔴 God Class | `SearchTimingTestSuite` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 495 lines - consider breaking into smaller classes |
| 🔴 Long Method | `SearchPerformanceMonitor` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | 294 lines (recommended: <50) |
| 🔴 High Complexity | `SearchPerformanceMonitor` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | Complexity: 28 (recommended: <15) |
| 🔴 Long Method | `SearchQualityAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 538 lines (recommended: <50) |
| 🔴 High Complexity | `SearchQualityAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | Complexity: 40 (recommended: <15) |
| 🔴 God Class | `SearchQualityAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 538 lines - consider breaking into smaller classes |
| 🔴 Long Method | `generate_quality_report` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 106 lines (recommended: <50) |
| 🔴 Deep Nesting | `mcp-vector-search.sh` | /Users/masa/Projects/mcp-vector-search/scripts/setup/mcp-vector-search.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `setup-alias.sh` | /Users/masa/Projects/mcp-vector-search/scripts/setup/setup-alias.sh | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `DocumentationUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | 213 lines (recommended: <50) |
| 🔴 High Complexity | `DocumentationUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | Complexity: 29 (recommended: <15) |
| 🔴 Long Method | `HomebrewFormulaUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 610 lines (recommended: <50) |
| 🔴 High Complexity | `HomebrewFormulaUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | Complexity: 67 (recommended: <15) |
| 🔴 God Class | `HomebrewFormulaUpdater` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 610 lines - consider breaking into smaller classes |
| 🔴 Long Method | `update_formula` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 112 lines (recommended: <50) |
| 🔴 Long Method | `commit_and_push` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 107 lines (recommended: <50) |
| 🔴 Long Method | `VersionManager` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | 240 lines (recommended: <50) |
| 🔴 High Complexity | `VersionManager` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | Complexity: 30 (recommended: <15) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | 117 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `comparator.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `BaselineComparator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | 304 lines (recommended: <50) |
| 🔴 God Class | `BaselineComparator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | 304 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `manager.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `BaselineManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 507 lines (recommended: <50) |
| 🔴 High Complexity | `BaselineManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | Complexity: 37 (recommended: <15) |
| 🔴 God Class | `BaselineManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 507 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `base.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/base.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MetricCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/base.py | 111 lines (recommended: <50) |
| 🔴 Deep Nesting | `cohesion.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `LCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | 313 lines (recommended: <50) |
| 🔴 High Complexity | `LCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | Complexity: 45 (recommended: <15) |
| 🔴 God Class | `LCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | 313 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `complexity.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `CognitiveComplexityCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 141 lines (recommended: <50) |
| 🔴 Long Method | `ParameterCountCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 112 lines (recommended: <50) |
| 🔴 Deep Nesting | `coupling.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `CircularDependencyDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 155 lines (recommended: <50) |
| 🔴 Long Method | `EfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 232 lines (recommended: <50) |
| 🔴 High Complexity | `EfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | Complexity: 39 (recommended: <15) |
| 🔴 High Complexity | `_extract_import` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | Complexity: 28 (recommended: <15) |
| 🔴 Long Method | `AfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 128 lines (recommended: <50) |
| 🔴 Long Method | `InstabilityCalculator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 149 lines (recommended: <50) |
| 🔴 Deep Nesting | `halstead.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/halstead.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `HalsteadMetrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/halstead.py | 117 lines (recommended: <50) |
| 🔴 Long Method | `HalsteadCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/halstead.py | 224 lines (recommended: <50) |
| 🔴 Deep Nesting | `smells.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/smells.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `SmellDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/smells.py | 253 lines (recommended: <50) |
| 🔴 Long Method | `detect` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/smells.py | 108 lines (recommended: <50) |
| 🔴 Deep Nesting | `debt.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/debt.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TechnicalDebtEstimator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/debt.py | 336 lines (recommended: <50) |
| 🔴 God Class | `TechnicalDebtEstimator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/debt.py | 336 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `interpretation.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `EnhancedJSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 326 lines (recommended: <50) |
| 🔴 High Complexity | `EnhancedJSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | Complexity: 28 (recommended: <15) |
| 🔴 God Class | `EnhancedJSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 326 lines - consider breaking into smaller classes |
| 🔴 Long Method | `AnalysisInterpreter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 187 lines (recommended: <50) |
| 🔴 Deep Nesting | `metrics.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/metrics.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ChunkMetrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/metrics.py | 104 lines (recommended: <50) |
| 🔴 Long Method | `FileMetrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/metrics.py | 107 lines (recommended: <50) |
| 🔴 Long Method | `ProjectMetrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/metrics.py | 141 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `console.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `ConsoleReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 631 lines (recommended: <50) |
| 🔴 High Complexity | `ConsoleReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | Complexity: 64 (recommended: <15) |
| 🔴 God Class | `ConsoleReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 631 lines - consider breaking into smaller classes |
| 🔴 Long Method | `print_instability` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 123 lines (recommended: <50) |
| 🔴 Long Method | `print_baseline_comparison` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 198 lines (recommended: <50) |
| 🔴 High Complexity | `print_baseline_comparison` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | Complexity: 26 (recommended: <15) |
| 🔴 Deep Nesting | `markdown.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MarkdownReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | 462 lines (recommended: <50) |
| 🔴 High Complexity | `MarkdownReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | Complexity: 43 (recommended: <15) |
| 🔴 God Class | `MarkdownReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | 462 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_build_analysis_markdown` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | 180 lines (recommended: <50) |
| 🔴 Long Method | `_build_fixes_markdown` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | 192 lines (recommended: <50) |
| 🔴 Deep Nesting | `sarif.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/sarif.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `SARIFReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/sarif.py | 347 lines (recommended: <50) |
| 🔴 God Class | `SARIFReporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/sarif.py | 347 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `metrics_store.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MetricsStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 574 lines (recommended: <50) |
| 🔴 High Complexity | `MetricsStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | Complexity: 33 (recommended: <15) |
| 🔴 God Class | `MetricsStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 574 lines - consider breaking into smaller classes |
| 🔴 Long Method | `save_project_snapshot` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 101 lines (recommended: <50) |
| 🔴 Deep Nesting | `schema.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/schema.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `trend_tracker.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 413 lines (recommended: <50) |
| 🔴 High Complexity | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | Complexity: 26 (recommended: <15) |
| 🔴 God Class | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 413 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `trends.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/trends.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/trends.py | 231 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `d3_data.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/d3_data.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `exporter.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `JSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | 426 lines (recommended: <50) |
| 🔴 God Class | `JSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | 426 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `html_report.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `HTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 2857 lines (recommended: <50) |
| 🔴 High Complexity | `HTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | Complexity: 26 (recommended: <15) |
| 🔴 God Class | `HTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 2857 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_generate_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 877 lines (recommended: <50) |
| 🔴 Long Method | `_generate_filter_controls` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 111 lines (recommended: <50) |
| 🔴 Long Method | `_generate_scripts` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 1175 lines (recommended: <50) |
| 🔴 Deep Nesting | `schemas.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/schemas.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `analyze.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 284 lines (recommended: <50) |
| 🔴 Long Method | `run_analysis` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 388 lines (recommended: <50) |
| 🔴 High Complexity | `run_analysis` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | Complexity: 59 (recommended: <15) |
| 🔴 Long Method | `_find_analyzable_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 124 lines (recommended: <50) |
| 🔴 Deep Nesting | `auto_index.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/auto_index.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `chat.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `chat_main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 168 lines (recommended: <50) |
| 🔴 Long Method | `run_chat_with_intent` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 121 lines (recommended: <50) |
| 🔴 Long Method | `run_chat_answer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 127 lines (recommended: <50) |
| 🔴 Long Method | `run_chat_analyze` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 178 lines (recommended: <50) |
| 🔴 Long Method | `_process_answer_query` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 320 lines (recommended: <50) |
| 🔴 High Complexity | `_process_answer_query` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | Complexity: 33 (recommended: <15) |
| 🔴 Long Method | `run_chat_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 226 lines (recommended: <50) |
| 🔴 High Complexity | `run_chat_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | Complexity: 31 (recommended: <15) |
| 🔴 Deep Nesting | `config.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/config.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `demo.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/demo.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `demo` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/demo.py | 330 lines (recommended: <50) |
| 🔴 Deep Nesting | `index.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 113 lines (recommended: <50) |
| 🔴 Long Method | `_run_batch_indexing` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 241 lines (recommended: <50) |
| 🔴 High Complexity | `_run_batch_indexing` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | Complexity: 26 (recommended: <15) |
| 🔴 Deep Nesting | `init.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | 288 lines (recommended: <50) |
| 🔴 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | Complexity: 27 (recommended: <15) |
| 🔴 Long Method | `run_init_setup` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | 143 lines (recommended: <50) |
| 🔴 Deep Nesting | `install_old.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | 234 lines (recommended: <50) |
| 🔴 Long Method | `demo` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | 140 lines (recommended: <50) |
| 🔴 Deep Nesting | `install.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 175 lines (recommended: <50) |
| 🔴 Long Method | `_install_to_platform` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 146 lines (recommended: <50) |
| 🔴 Long Method | `install_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 160 lines (recommended: <50) |
| 🔴 Deep Nesting | `mcp.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `install_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 145 lines (recommended: <50) |
| 🔴 Long Method | `test_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 108 lines (recommended: <50) |
| 🔴 Deep Nesting | `reset.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `reset_index` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | 158 lines (recommended: <50) |
| 🔴 High Complexity | `reset_index` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | Complexity: 27 (recommended: <15) |
| 🔴 Long Method | `check_health` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | 124 lines (recommended: <50) |
| 🔴 Deep Nesting | `search.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `search_main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | 273 lines (recommended: <50) |
| 🔴 Long Method | `run_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | 287 lines (recommended: <50) |
| 🔴 High Complexity | `run_search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | Complexity: 55 (recommended: <15) |
| 🔴 Deep Nesting | `setup.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `setup_llm_api_keys` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 208 lines (recommended: <50) |
| 🔴 High Complexity | `setup_llm_api_keys` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | Complexity: 29 (recommended: <15) |
| 🔴 Long Method | `_setup_single_provider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 128 lines (recommended: <50) |
| 🔴 Long Method | `setup_openrouter_api_key` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 179 lines (recommended: <50) |
| 🔴 Long Method | `_run_smart_setup` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 232 lines (recommended: <50) |
| 🔴 High Complexity | `_run_smart_setup` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | Complexity: 45 (recommended: <15) |
| 🔴 Deep Nesting | `status.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 112 lines (recommended: <50) |
| 🔴 Long Method | `show_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 131 lines (recommended: <50) |
| 🔴 Long Method | `_display_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 126 lines (recommended: <50) |
| 🔴 Long Method | `_print_metrics_summary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 121 lines (recommended: <50) |
| 🔴 Deep Nesting | `uninstall.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/uninstall.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `uninstall_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/uninstall.py | 134 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/__init__.py | Depth: 10 (recommended: <4) |
| 🔴 Deep Nesting | `cli.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/cli.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `_export_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/cli.py | 101 lines (recommended: <50) |
| 🔴 Long Method | `serve` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/cli.py | 123 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/exporters/__init__.py | Depth: 11 (recommended: <4) |
| 🔴 Deep Nesting | `html_exporter.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/exporters/html_exporter.py | Depth: 11 (recommended: <4) |
| 🔴 Deep Nesting | `json_exporter.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/exporters/json_exporter.py | Depth: 11 (recommended: <4) |
| 🔴 Deep Nesting | `graph_builder.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `build_graph_data` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | 334 lines (recommended: <50) |
| 🔴 High Complexity | `build_graph_data` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | Complexity: 62 (recommended: <15) |
| 🔴 Deep Nesting | `layout_engine.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `calculate_fan_layout` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | 131 lines (recommended: <50) |
| 🔴 Long Method | `calculate_tree_layout` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | 105 lines (recommended: <50) |
| 🔴 Deep Nesting | `server.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `create_app` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py | 499 lines (recommended: <50) |
| 🔴 High Complexity | `create_app` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py | Complexity: 67 (recommended: <15) |
| 🔴 Deep Nesting | `state_manager.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `VisualizationState` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | 361 lines (recommended: <50) |
| 🔴 High Complexity | `VisualizationState` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | Complexity: 32 (recommended: <15) |
| 🔴 God Class | `VisualizationState` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | 361 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/__init__.py | Depth: 11 (recommended: <4) |
| 🔴 Deep Nesting | `base.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py | Depth: 11 (recommended: <4) |
| 🔴 Long Method | `generate_html_template` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py | 191 lines (recommended: <50) |
| 🔴 Deep Nesting | `scripts.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py | Depth: 11 (recommended: <4) |
| 🔴 Long Method | `get_all_scripts` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py | 4273 lines (recommended: <50) |
| 🔴 Deep Nesting | `styles.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | Depth: 11 (recommended: <4) |
| 🔴 Long Method | `get_controls_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 223 lines (recommended: <50) |
| 🔴 Long Method | `get_content_pane_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 507 lines (recommended: <50) |
| 🔴 Long Method | `get_code_chunks_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 102 lines (recommended: <50) |
| 🔴 Long Method | `get_search_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 142 lines (recommended: <50) |
| 🔴 Long Method | `get_complexity_report_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 212 lines (recommended: <50) |
| 🔴 Long Method | `get_code_smells_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 231 lines (recommended: <50) |
| 🔴 Long Method | `get_dependencies_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 243 lines (recommended: <50) |
| 🔴 Long Method | `get_trends_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 315 lines (recommended: <50) |
| 🔴 Deep Nesting | `watch.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/watch.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `didyoumean.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/didyoumean.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `export.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `SearchResultExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 272 lines (recommended: <50) |
| 🔴 High Complexity | `SearchResultExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | Complexity: 26 (recommended: <15) |
| 🔴 Deep Nesting | `history.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/history.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `SearchHistory` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/history.py | 209 lines (recommended: <50) |
| 🔴 Deep Nesting | `interactive.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `InteractiveSearchSession` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | 316 lines (recommended: <50) |
| 🔴 High Complexity | `InteractiveSearchSession` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | Complexity: 44 (recommended: <15) |
| 🔴 God Class | `InteractiveSearchSession` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | 316 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `main.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/main.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `output.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/output.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `print_search_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/output.py | 142 lines (recommended: <50) |
| 🔴 High Complexity | `print_search_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/output.py | Complexity: 26 (recommended: <15) |
| 🔴 Deep Nesting | `suggestions.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ContextualSuggestionProvider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 339 lines (recommended: <50) |
| 🔴 High Complexity | `ContextualSuggestionProvider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | Complexity: 30 (recommended: <15) |
| 🔴 God Class | `ContextualSuggestionProvider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 339 lines - consider breaking into smaller classes |
| 🔴 Long Method | `get_workflow_suggestions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 116 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `constants.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/constants.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `defaults.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `settings.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `thresholds.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/thresholds.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ThresholdConfig` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/thresholds.py | 178 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `auto_indexer.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/auto_indexer.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `AutoIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/auto_indexer.py | 197 lines (recommended: <50) |
| 🔴 Deep Nesting | `boilerplate.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/boilerplate.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `BoilerplateFilter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/boilerplate.py | 101 lines (recommended: <50) |
| 🔴 Deep Nesting | `config_utils.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/config_utils.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ConfigManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/config_utils.py | 163 lines (recommended: <50) |
| 🔴 Deep Nesting | `connection_pool.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/connection_pool.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/connection_pool.py | 323 lines (recommended: <50) |
| 🔴 High Complexity | `ChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/connection_pool.py | Complexity: 42 (recommended: <15) |
| 🔴 God Class | `ChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/connection_pool.py | 323 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `database.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 881 lines (recommended: <50) |
| 🔴 High Complexity | `ChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Complexity: 105 (recommended: <15) |
| 🔴 God Class | `ChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 881 lines - consider breaking into smaller classes |
| 🔴 Long Method | `initialize` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 152 lines (recommended: <50) |
| 🔴 Long Method | `_detect_and_recover_corruption` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 113 lines (recommended: <50) |
| 🔴 Long Method | `PooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 511 lines (recommended: <50) |
| 🔴 High Complexity | `PooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Complexity: 75 (recommended: <15) |
| 🔴 God Class | `PooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 511 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `directory_index.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `DirectoryIndex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | 306 lines (recommended: <50) |
| 🔴 High Complexity | `DirectoryIndex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | Complexity: 34 (recommended: <15) |
| 🔴 God Class | `DirectoryIndex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | 306 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `embeddings.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `EmbeddingCache` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py | 118 lines (recommended: <50) |
| 🔴 Deep Nesting | `exceptions.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/exceptions.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `factory.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/factory.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ComponentFactory` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/factory.py | 171 lines (recommended: <50) |
| 🔴 Deep Nesting | `git_hooks.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git_hooks.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `GitHookManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git_hooks.py | 248 lines (recommended: <50) |
| 🔴 High Complexity | `GitHookManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git_hooks.py | Complexity: 30 (recommended: <15) |
| 🔴 Deep Nesting | `git.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `GitManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | 326 lines (recommended: <50) |
| 🔴 High Complexity | `GitManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | Complexity: 30 (recommended: <15) |
| 🔴 God Class | `GitManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | 326 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `indexer.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `SemanticIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 1236 lines (recommended: <50) |
| 🔴 High Complexity | `SemanticIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | Complexity: 134 (recommended: <15) |
| 🔴 God Class | `SemanticIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 1236 lines - consider breaking into smaller classes |
| 🔴 Long Method | `index_project` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 159 lines (recommended: <50) |
| 🔴 Long Method | `index_files_with_progress` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 104 lines (recommended: <50) |
| 🔴 Long Method | `_build_chunk_hierarchy` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 108 lines (recommended: <50) |
| 🔴 Deep Nesting | `llm_client.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `LLMClient` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 736 lines (recommended: <50) |
| 🔴 High Complexity | `LLMClient` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | Complexity: 75 (recommended: <15) |
| 🔴 God Class | `LLMClient` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 736 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `models.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `CodeChunk` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py | 119 lines (recommended: <50) |
| 🔴 Long Method | `SearchResult` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py | 141 lines (recommended: <50) |
| 🔴 Deep Nesting | `project.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ProjectManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | 326 lines (recommended: <50) |
| 🔴 High Complexity | `ProjectManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | Complexity: 36 (recommended: <15) |
| 🔴 God Class | `ProjectManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | 326 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `relationships.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `RelationshipStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 280 lines (recommended: <50) |
| 🔴 High Complexity | `RelationshipStore` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | Complexity: 31 (recommended: <15) |
| 🔴 Deep Nesting | `scheduler.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `SchedulerManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | 320 lines (recommended: <50) |
| 🔴 High Complexity | `SchedulerManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | Complexity: 31 (recommended: <15) |
| 🔴 God Class | `SchedulerManager` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | 320 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `search.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `SemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 1082 lines (recommended: <50) |
| 🔴 High Complexity | `SemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | Complexity: 112 (recommended: <15) |
| 🔴 God Class | `SemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 1082 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_rerank_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 109 lines (recommended: <50) |
| 🔴 Deep Nesting | `watcher.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/watcher.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `CodeFileHandler` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/watcher.py | 102 lines (recommended: <50) |
| 🔴 Long Method | `FileWatcher` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/watcher.py | 152 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__main__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/__main__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `server.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `MCPVectorSearchServer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 1417 lines (recommended: <50) |
| 🔴 High Complexity | `MCPVectorSearchServer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | Complexity: 129 (recommended: <15) |
| 🔴 God Class | `MCPVectorSearchServer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 1417 lines - consider breaking into smaller classes |
| 🔴 Long Method | `get_tools` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 248 lines (recommended: <50) |
| 🔴 Long Method | `_analyze_project` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 132 lines (recommended: <50) |
| 🔴 Long Method | `_analyze_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 105 lines (recommended: <50) |
| 🔴 Long Method | `_find_smells` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 120 lines (recommended: <50) |
| 🔴 Long Method | `_check_circular_dependencies` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 129 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `base.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/base.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `BaseParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/base.py | 236 lines (recommended: <50) |
| 🔴 Deep Nesting | `dart.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `DartParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | 594 lines (recommended: <50) |
| 🔴 High Complexity | `DartParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | Complexity: 73 (recommended: <15) |
| 🔴 God Class | `DartParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | 594 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | 185 lines (recommended: <50) |
| 🔴 Deep Nesting | `html.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `HTMLContentParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 182 lines (recommended: <50) |
| 🔴 Long Method | `HTMLParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 220 lines (recommended: <50) |
| 🔴 Deep Nesting | `javascript.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `JavaScriptParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | 607 lines (recommended: <50) |
| 🔴 High Complexity | `JavaScriptParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | Complexity: 81 (recommended: <15) |
| 🔴 God Class | `JavaScriptParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | 607 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_regex_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | 150 lines (recommended: <50) |
| 🔴 Deep Nesting | `php.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `PHPParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 683 lines (recommended: <50) |
| 🔴 High Complexity | `PHPParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | Complexity: 93 (recommended: <15) |
| 🔴 God Class | `PHPParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 683 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 194 lines (recommended: <50) |
| 🔴 Deep Nesting | `python.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `PythonParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 771 lines (recommended: <50) |
| 🔴 High Complexity | `PythonParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | Complexity: 134 (recommended: <15) |
| 🔴 God Class | `PythonParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 771 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_extract_class_skeleton` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 129 lines (recommended: <50) |
| 🔴 High Complexity | `_extract_class_skeleton` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | Complexity: 39 (recommended: <15) |
| 🔴 Long Method | `_extract_class_skeleton_regex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 144 lines (recommended: <50) |
| 🔴 High Complexity | `_extract_class_skeleton_regex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | Complexity: 31 (recommended: <15) |
| 🔴 Deep Nesting | `registry.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/registry.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ParserRegistry` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/registry.py | 160 lines (recommended: <50) |
| 🔴 Deep Nesting | `ruby.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `RubyParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | 667 lines (recommended: <50) |
| 🔴 High Complexity | `RubyParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | Complexity: 82 (recommended: <15) |
| 🔴 God Class | `RubyParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | 667 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | 177 lines (recommended: <50) |
| 🔴 Deep Nesting | `text.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TextParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | 177 lines (recommended: <50) |
| 🔴 Deep Nesting | `utils.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/utils.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `gitignore_updater.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore_updater.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ensure_gitignore_entry` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore_updater.py | 205 lines (recommended: <50) |
| 🔴 Deep Nesting | `gitignore.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `GitignoreParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore.py | 121 lines (recommended: <50) |
| 🔴 Deep Nesting | `monorepo.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/monorepo.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `MonorepoDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/monorepo.py | 294 lines (recommended: <50) |
| 🔴 High Complexity | `MonorepoDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/monorepo.py | Complexity: 44 (recommended: <15) |
| 🔴 Deep Nesting | `timing.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/timing.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `PerformanceProfiler` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/timing.py | 199 lines (recommended: <50) |
| 🔴 Deep Nesting | `version.py` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/version.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `temp_project_dir` | /Users/masa/Projects/mcp-vector-search/tests/conftest.py | 146 lines (recommended: <50) |
| 🔴 Long Method | `mock_database` | /Users/masa/Projects/mcp-vector-search/tests/conftest.py | 118 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_cli_commands.py` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestCLICommands` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | 495 lines (recommended: <50) |
| 🔴 High Complexity | `TestCLICommands` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | Complexity: 30 (recommended: <15) |
| 🔴 God Class | `TestCLICommands` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | 495 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_boilerplate_filtering.py` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_boilerplate_filtering.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestBoilerplateFilteringIntegration` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_boilerplate_filtering.py | 258 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_indexing_workflow.py` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_indexing_workflow.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestIndexingWorkflow` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_indexing_workflow.py | 406 lines (recommended: <50) |
| 🔴 God Class | `TestIndexingWorkflow` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_indexing_workflow.py | 406 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_init_gitignore.py` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_init_gitignore.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestProjectInitGitignore` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_init_gitignore.py | 201 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_setup_integration.py` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestSetupCompleteWorkflow` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 178 lines (recommended: <50) |
| 🔴 Long Method | `TestSetupMCPIntegration` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 131 lines (recommended: <50) |
| 🔴 Long Method | `TestSetupEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 212 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_visualization_v2_flow.py` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_visualization_v2_flow.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `analyze-command-test-report.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/analyze-command-test-report.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `apply_fix.sh` | /Users/masa/Projects/mcp-vector-search/tests/manual/apply_fix.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `breadcrumb_test_results.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/breadcrumb_test_results.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `d3-tree-collapse-test.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/d3-tree-collapse-test.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `debug_loading_timing.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_loading_timing.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `debug_network.js` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_network.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `debug_visualization_simple.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_visualization_simple.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `debug_visualizer.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_visualizer.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `debug_visualizer` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_visualizer.py | 127 lines (recommended: <50) |
| 🔴 Deep Nesting | `file-click-debug-checklist.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/file-click-debug-checklist.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `inspect_visualization_controls.js` | /Users/masa/Projects/mcp-vector-search/tests/manual/inspect_visualization_controls.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `chromium` | /Users/masa/Projects/mcp-vector-search/tests/manual/inspect_visualization_controls.js | 210 lines (recommended: <50) |
| 🔴 Deep Nesting | `investigate_visualization.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/investigate_visualization.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `investigate` | /Users/masa/Projects/mcp-vector-search/tests/manual/investigate_visualization.py | 189 lines (recommended: <50) |
| 🔴 Deep Nesting | `package-lock.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/package-lock.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `package.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/package.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_analyze_git_integration.sh` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_analyze_git_integration.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_api_key_obfuscation.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_api_key_obfuscation.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_breadcrumb_fix.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_breadcrumb_fix.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `run_breadcrumb_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_breadcrumb_fix.py | 198 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_cli_integration.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_cli_integration.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_compact_folder_layout.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_compact_folder_layout.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `create_test_instructions` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_compact_folder_layout.py | 121 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_controls_safari.sh` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_controls_safari.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_cycle_detection.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_cycle_detection.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_d3_integration.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_d3_integration.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_file_scan_ewtn.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_file_scan_ewtn.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_final_comprehensive.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_final_comprehensive.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `comprehensive_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_final_comprehensive.py | 262 lines (recommended: <50) |
| 🔴 High Complexity | `comprehensive_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_final_comprehensive.py | Complexity: 30 (recommended: <15) |
| 🔴 Deep Nesting | `test_full_index_ewtn.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_full_index_ewtn.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_gitignore_ewtn.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_gitignore_ewtn.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_glob_pattern_filtering.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_glob_pattern_filtering.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_glob_filtering` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_glob_pattern_filtering.py | 107 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_graph_large.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_large.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_graph_medium.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_medium.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_graph_small.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_small.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_graph_visualization_playwright.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_visualization_playwright.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_visualization` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_visualization_playwright.py | 340 lines (recommended: <50) |
| 🔴 High Complexity | `test_visualization` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_graph_visualization_playwright.py | Complexity: 33 (recommended: <15) |
| 🔴 Deep Nesting | `test_instructions.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_instructions.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_mcp_auto_install.sh` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_mcp_auto_install.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_mcp_installer_platform_fix.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_mcp_installer_platform_fix.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_openrouter_setup.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_openrouter_setup.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_root_breadcrumb_reset.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_root_breadcrumb_reset.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_root_breadcrumb_reset` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_root_breadcrumb_reset.py | 280 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_root_detection_direct.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_root_detection_direct.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_visualization.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualization.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_visualization` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualization.py | 109 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_visualizer_detailed.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer_detailed.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_visualizer_detailed` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer_detailed.py | 112 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_visualizer_line_by_line.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer_line_by_line.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_visualizer.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_visualizer` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer.py | 120 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_with_cdp.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_with_cdp.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test-null-checks.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/test-null-checks.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verification_report.json` | /Users/masa/Projects/mcp-vector-search/tests/manual/verification_report.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verify_data_initialization_fix.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_data_initialization_fix.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verify_root_filtering.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_root_filtering.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verify_streaming_load.sh` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_streaming_load.sh | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verify_visualization.py` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_visualization.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `verify_visualization` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_visualization.py | 198 lines (recommended: <50) |
| 🔴 Deep Nesting | `verify_with_node.js` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_with_node.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `verify` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_with_node.js | 165 lines (recommended: <50) |
| 🔴 Deep Nesting | `verify_with_wait.js` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_with_wait.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `verify-tree-connector-lines.md` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify-tree-connector-lines.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ast_test_javascript.js` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_javascript.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `User` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_javascript.js | 105 lines (recommended: <50) |
| 🔴 Deep Nesting | `ast_test_python.py` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_python.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ast_test_typescript.ts` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_typescript.ts | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_ast_features.py` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_dart_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_dart_parser.py | 223 lines (recommended: <50) |
| 🔴 Long Method | `test_dart_widget_patterns` | /Users/masa/Projects/mcp-vector-search/tests/test_dart_parser.py | 111 lines (recommended: <50) |
| 🔴 Long Method | `test_html_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_html_parser.py | 163 lines (recommended: <50) |
| 🔴 Long Method | `TestMCPIntegration` | /Users/masa/Projects/mcp-vector-search/tests/test_mcp_integration.py | 182 lines (recommended: <50) |
| 🔴 Long Method | `test_php_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_php_parser.py | 304 lines (recommended: <50) |
| 🔴 Long Method | `test_php_laravel_patterns` | /Users/masa/Projects/mcp-vector-search/tests/test_php_parser.py | 207 lines (recommended: <50) |
| 🔴 Long Method | `test_ruby_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_ruby_parser.py | 257 lines (recommended: <50) |
| 🔴 Long Method | `test_ruby_rails_patterns` | /Users/masa/Projects/mcp-vector-search/tests/test_ruby_parser.py | 132 lines (recommended: <50) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/tests/test_simple.py | 141 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_comparator.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_comparator.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestBaselineComparator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_comparator.py | 345 lines (recommended: <50) |
| 🔴 God Class | `TestBaselineComparator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_comparator.py | 345 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_manager.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_manager.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestBaselineManager` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_manager.py | 371 lines (recommended: <50) |
| 🔴 God Class | `TestBaselineManager` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_manager.py | 371 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_cohesion.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_cohesion.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestLCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_cohesion.py | 452 lines (recommended: <50) |
| 🔴 God Class | `TestLCOM4Calculator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_cohesion.py | 452 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_halstead.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_halstead.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestHalsteadCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_halstead.py | 358 lines (recommended: <50) |
| 🔴 God Class | `TestHalsteadCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_halstead.py | 358 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_sarif.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/reporters/test_sarif.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestSARIFReporter` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/reporters/test_sarif.py | 469 lines (recommended: <50) |
| 🔴 God Class | `TestSARIFReporter` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/reporters/test_sarif.py | 469 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_metrics_store.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestProjectSnapshotSaving` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 133 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_trend_tracker.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestGetTrends` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 128 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_collectors.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_collectors.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestCollectorContext` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_collectors.py | 132 lines (recommended: <50) |
| 🔴 Long Method | `TestMetricCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_collectors.py | 247 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_complexity_collectors.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestCognitiveComplexityCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 176 lines (recommended: <50) |
| 🔴 Long Method | `TestCyclomaticComplexityCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 140 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_coupling.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestEfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 151 lines (recommended: <50) |
| 🔴 Long Method | `TestCircularDependencyDetector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 244 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_debt.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestTechnicalDebtEstimator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | 516 lines (recommended: <50) |
| 🔴 God Class | `TestTechnicalDebtEstimator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | 516 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_interpretation.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestEnhancedJSONExporter` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | 106 lines (recommended: <50) |
| 🔴 Long Method | `TestAnalysisInterpreter` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | 168 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_metrics.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_metrics.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestChunkMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_metrics.py | 121 lines (recommended: <50) |
| 🔴 Long Method | `TestFileMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_metrics.py | 153 lines (recommended: <50) |
| 🔴 Long Method | `TestProjectMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_metrics.py | 312 lines (recommended: <50) |
| 🔴 God Class | `TestProjectMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_metrics.py | 312 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_smells.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_smells.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestSmellDetector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_smells.py | 350 lines (recommended: <50) |
| 🔴 God Class | `TestSmellDetector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_smells.py | 350 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_d3_visualization.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `sample_export` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 109 lines (recommended: <50) |
| 🔴 Long Method | `TestExtractCircularPaths` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 103 lines (recommended: <50) |
| 🔴 Long Method | `TestCreateEdges` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 112 lines (recommended: <50) |
| 🔴 Long Method | `TestSummaryStats` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 103 lines (recommended: <50) |
| 🔴 Long Method | `TestModuleGrouping` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 172 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_exporter.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_exporter.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_html_phase5.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_phase5.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_html_report.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_report.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestHTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_report.py | 638 lines (recommended: <50) |
| 🔴 God Class | `TestHTMLReportGenerator` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_report.py | 638 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_schemas.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestAnalysisExport` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 150 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_analyze_exit_codes.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_exit_codes.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestExitCodes` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_exit_codes.py | 216 lines (recommended: <50) |
| 🔴 Long Method | `TestFilterSmellsBySeverity` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_exit_codes.py | 166 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_analyze_git.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_git.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestAnalyzeGitIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_git.py | 171 lines (recommended: <50) |
| 🔴 Long Method | `TestGitIntegrationEndToEnd` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_git.py | 131 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_analyze.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `TestAnalyzeCommand` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze.py | 179 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_chat_analyze.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_chat_analyze.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestRunChatAnalyze` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_chat_analyze.py | 254 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_search_quality_filters.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_search_quality_filters.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestQualityFilterIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_search_quality_filters.py | 276 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_signal_handlers.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_signal_handlers.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_status.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_status.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestStatusMetricsCommand` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_status.py | 110 lines (recommended: <50) |
| 🔴 Long Method | `TestMetricsOutput` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_status.py | 134 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_setup_api_key.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup_api_key.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestSetupOpenRouterApiKey` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup_api_key.py | 200 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_setup.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestSetupCommand` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 337 lines (recommended: <50) |
| 🔴 God Class | `TestSetupCommand` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 337 lines - consider breaking into smaller classes |
| 🔴 Long Method | `TestSetupErrorHandling` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 157 lines (recommended: <50) |
| 🔴 Long Method | `TestSetupEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 118 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/config/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_thresholds.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/config/test_thresholds.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestThresholdConfigSerialization` | /Users/masa/Projects/mcp-vector-search/tests/unit/config/test_thresholds.py | 107 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_boilerplate.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_boilerplate.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestBoilerplateFilter` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_boilerplate.py | 244 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_connection_pool.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_connection_pool.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_connection_pool.py | 368 lines (recommended: <50) |
| 🔴 God Class | `TestChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_connection_pool.py | 368 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_corruption_recovery.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_corruption_recovery.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestCorruptionRecovery` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_corruption_recovery.py | 299 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_database_metrics.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestDatabaseMetricsSupport` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 319 lines (recommended: <50) |
| 🔴 God Class | `TestDatabaseMetricsSupport` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 319 lines - consider breaking into smaller classes |
| 🔴 Long Method | `TestMigrationScript` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 103 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_database.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | 222 lines (recommended: <50) |
| 🔴 Long Method | `TestPooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | 381 lines (recommended: <50) |
| 🔴 God Class | `TestPooledChromaVectorDatabase` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | 381 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_git.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_git.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_indexer_collectors.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestIndexFileWithMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 183 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_indexer.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestSemanticIndexer` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer.py | 457 lines (recommended: <50) |
| 🔴 God Class | `TestSemanticIndexer` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer.py | 457 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_llm_client_intent.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_llm_client_intent.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestIntentDetection` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_llm_client_intent.py | 179 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_models.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_models.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_quality_ranking.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_quality_ranking.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestQualityScoreCalculation` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_quality_ranking.py | 181 lines (recommended: <50) |
| 🔴 Long Method | `TestQualityAwareRanking` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_quality_ranking.py | 251 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_relationships.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_relationships.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_search.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_search.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestSemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_search.py | 436 lines (recommended: <50) |
| 🔴 God Class | `TestSemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_search.py | 436 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `test_analysis_tools.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_analysis_tools.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `test_interpret_analysis.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_interpret_analysis.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestInterpretAnalysisTool` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_interpret_analysis.py | 186 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_graph_builder_ast.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_graph_builder_ast.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_layout_engine.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestFanLayout` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 130 lines (recommended: <50) |
| 🔴 Long Method | `TestTreeLayout` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 154 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_mcp_install_auto_detection.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_mcp_install_auto_detection.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_monorepo.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_monorepo.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestMonorepoDetector` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_monorepo.py | 122 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_state_manager.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_state_manager.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestVisualizationState` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_state_manager.py | 281 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_trends.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_trends.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_gitignore_updater.py` | /Users/masa/Projects/mcp-vector-search/tests/unit/utils/test_gitignore_updater.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestEnsureGitignoreEntry` | /Users/masa/Projects/mcp-vector-search/tests/unit/utils/test_gitignore_updater.py | 270 lines (recommended: <50) |
| 🔴 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/CHANGELOG.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CONTRIBUTING.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/CONTRIBUTING.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ARCHITECTURE.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/ARCHITECTURE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/design/SUMMARY.md | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `DIAGRAMS.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `IMPLEMENTATION-PLAN.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/IMPLEMENTATION-PLAN.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `PROJECT-STRUCTURE.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/PROJECT-STRUCTURE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `QUICK-REFERENCE.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/QUICK-REFERENCE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/README.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-server-installation-patterns-2025-12-05.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/research/mcp-server-installation-patterns-2025-12-05.md | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `phase3_demo.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/examples/phase3_demo.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASING.md` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/RELEASING.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `manage_version.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/scripts/manage_version.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `release.sh` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/scripts/release.sh | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `setup.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/setup.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `command_builder.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `CommandBuilder` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 419 lines (recommended: <50) |
| 🔴 High Complexity | `CommandBuilder` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | Complexity: 27 (recommended: <15) |
| 🔴 God Class | `CommandBuilder` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 419 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `config_manager.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `ConfigManager` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 500 lines (recommended: <50) |
| 🔴 High Complexity | `ConfigManager` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | Complexity: 50 (recommended: <15) |
| 🔴 God Class | `ConfigManager` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 500 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `exceptions.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/exceptions.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `installation_strategy.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `NativeCLIStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 242 lines (recommended: <50) |
| 🔴 Long Method | `JSONManipulationStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 141 lines (recommended: <50) |
| 🔴 Long Method | `TOMLManipulationStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 126 lines (recommended: <50) |
| 🔴 Deep Nesting | `installer.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MCPInstaller` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 592 lines (recommended: <50) |
| 🔴 High Complexity | `MCPInstaller` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | Complexity: 50 (recommended: <15) |
| 🔴 God Class | `MCPInstaller` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 592 lines - consider breaking into smaller classes |
| 🔴 Long Method | `install_server` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 141 lines (recommended: <50) |
| 🔴 Deep Nesting | `mcp_inspector.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MCPInspector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 609 lines (recommended: <50) |
| 🔴 High Complexity | `MCPInspector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | Complexity: 55 (recommended: <15) |
| 🔴 God Class | `MCPInspector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 609 lines - consider breaking into smaller classes |
| 🔴 Long Method | `inspect` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 117 lines (recommended: <50) |
| 🔴 Long Method | `validate_server` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 110 lines (recommended: <50) |
| 🔴 Deep Nesting | `platform_detector.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `PlatformDetector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 478 lines (recommended: <50) |
| 🔴 High Complexity | `PlatformDetector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | Complexity: 56 (recommended: <15) |
| 🔴 God Class | `PlatformDetector` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 478 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/__init__.py | Depth: 10 (recommended: <4) |
| 🔴 Deep Nesting | `claude_code.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/claude_code.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `ClaudeCodeStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/claude_code.py | 201 lines (recommended: <50) |
| 🔴 Deep Nesting | `codex.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/codex.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `CodexStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/codex.py | 159 lines (recommended: <50) |
| 🔴 Deep Nesting | `cursor.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/cursor.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `CursorStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platforms/cursor.py | 169 lines (recommended: <50) |
| 🔴 Deep Nesting | `types.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/types.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `utils.py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/utils.py | Depth: 9 (recommended: <4) |

### Warnings

| Smell | Name | File | Detail |
|-------|------|------|--------|
| 🟡 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-vector-search/CHANGELOG.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `check_duplicates.py` | /Users/masa/Projects/mcp-vector-search/check_duplicates.py | Depth: 5 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/check_duplicates.py | 65 lines (recommended: <50) |
| 🟡 Deep Nesting | `CLAUDE.md` | /Users/masa/Projects/mcp-vector-search/CLAUDE.md | Depth: 5 (recommended: <4) |
| 🟡 Long Method | `Change` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_compact_layout_fixes.md | 51 lines (recommended: <50) |
| 🟡 Long Method | `switch` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_implementation_plan.md | 51 lines (recommended: <50) |
| 🟡 Long Method | `switch` | /Users/masa/Projects/mcp-vector-search/docs/_archive/implementation-reports/visualization_improvements_spec.md | 51 lines (recommended: <50) |
| 🟡 Long Method | `docs` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/DOCUMENTATION_REORGANIZATION_COMPLETE.md | 65 lines (recommended: <50) |
| 🟡 Long Method | `docs` | /Users/masa/Projects/mcp-vector-search/docs/_archive/summaries/REORGANIZATION_PLAN.md | 62 lines (recommended: <50) |
| 🟡 Long Method | `STEP` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_DIAGRAMS.md | 78 lines (recommended: <50) |
| 🟡 Long Method | `USER` | /Users/masa/Projects/mcp-vector-search/docs/development/VISUALIZATION_V2_DIAGRAMS.md | 60 lines (recommended: <50) |
| 🟡 Deep Nesting | `DOCUMENTATION-STANDARDS.md` | /Users/masa/Projects/mcp-vector-search/docs/DOCUMENTATION-STANDARDS.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `index.md` | /Users/masa/Projects/mcp-vector-search/docs/index.md | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `mcp` | /Users/masa/Projects/mcp-vector-search/docs/prd/mcp_vector_search_prd_updated.md | 63 lines (recommended: <50) |
| 🟡 Long Method | `analyze_app` | /Users/masa/Projects/mcp-vector-search/docs/research/analyze-command-implementation-research-2024-12-10.md | 72 lines (recommended: <50) |
| 🟡 Long Method | `mcp` | /Users/masa/Projects/mcp-vector-search/docs/research/automatic-setup-command-design-2025-11-30.md | 54 lines (recommended: <50) |
| 🟡 Long Method | `File` | /Users/masa/Projects/mcp-vector-search/docs/research/issue-18-baseline-comparison-requirements.md | 75 lines (recommended: <50) |
| 🟡 Long Method | `json` | /Users/masa/Projects/mcp-vector-search/docs/research/issue-18-baseline-comparison-requirements.md | 88 lines (recommended: <50) |
| 🟡 Long Method | `Command` | /Users/masa/Projects/mcp-vector-search/docs/research/visualizer-filtering-analysis-2025-12-04.md | 57 lines (recommended: <50) |
| 🟡 Deep Nesting | `connection_pooling_example.py` | /Users/masa/Projects/mcp-vector-search/examples/connection_pooling_example.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `export_analysis_json.py` | /Users/masa/Projects/mcp-vector-search/examples/export_analysis_json.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `create_sample_project_metrics` | /Users/masa/Projects/mcp-vector-search/examples/export_analysis_json.py | 55 lines (recommended: <50) |
| 🟡 Deep Nesting | `generate_html_report.py` | /Users/masa/Projects/mcp-vector-search/examples/generate_html_report.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/examples/generate_html_report.py | 68 lines (recommended: <50) |
| 🟡 Deep Nesting | `metrics_store_usage.py` | /Users/masa/Projects/mcp-vector-search/examples/metrics_store_usage.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `create_sample_metrics` | /Users/masa/Projects/mcp-vector-search/examples/metrics_store_usage.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/examples/metrics_store_usage.py | 73 lines (recommended: <50) |
| 🟡 Deep Nesting | `semi_automatic_reindexing_demo.py` | /Users/masa/Projects/mcp-vector-search/examples/semi_automatic_reindexing_demo.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `threshold_config_demo.py` | /Users/masa/Projects/mcp-vector-search/examples/threshold_config_demo.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `PortSelector` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `disconnectBackend` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 61 lines (recommended: <50) |
| 🟡 High Complexity | `_setupMessageHandler` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_setupCloseHandler` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 64 lines (recommended: <50) |
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 87 lines (recommended: <50) |
| 🟡 Long Method | `connectToServer` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/background-enhanced.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `updateBackendList` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/popup-enhanced.js | 68 lines (recommended: <50) |
| 🟡 Long Method | `handleConnectToBackend` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/popup-enhanced.js | 90 lines (recommended: <50) |
| 🟡 Long Method | `handleScanBackends` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/popup-enhanced.js | 84 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-vector-search/mcp-browser-extensions/chrome/Readability.js | 59 lines (recommended: <50) |
| 🟡 Deep Nesting | `mcp-vector-search-analysis-fixes.md` | /Users/masa/Projects/mcp-vector-search/mcp-vector-search-analysis-fixes.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `mcp-vector-search-analysis.md` | /Users/masa/Projects/mcp-vector-search/mcp-vector-search-analysis.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `MAKEFILE_EXTRACTION_REPORT.md` | /Users/masa/Projects/mcp-vector-search/project-template/MAKEFILE_EXTRACTION_REPORT.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/project-template/README.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `STRUCTURE.md` | /Users/masa/Projects/mcp-vector-search/project-template/STRUCTURE.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `TEMPLATE_README.md` | /Users/masa/Projects/mcp-vector-search/project-template/TEMPLATE_README.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `VERIFICATION.md` | /Users/masa/Projects/mcp-vector-search/project-template/VERIFICATION.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/README.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/scripts/__init__.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `add_issue_dependencies.sh` | /Users/masa/Projects/mcp-vector-search/scripts/add_issue_dependencies.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `analyze_search_bottlenecks.py` | /Users/masa/Projects/mcp-vector-search/scripts/analyze_search_bottlenecks.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `PerformanceAnalyzer` | /Users/masa/Projects/mcp-vector-search/scripts/analyze_search_bottlenecks.py | Complexity: 19 (recommended: <15) |
| 🟡 Deep Nesting | `benchmark_llm_models.py` | /Users/masa/Projects/mcp-vector-search/scripts/benchmark_llm_models.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `print_summary` | /Users/masa/Projects/mcp-vector-search/scripts/benchmark_llm_models.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `run_benchmarks` | /Users/masa/Projects/mcp-vector-search/scripts/benchmark_llm_models.py | 95 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/benchmark_llm_models.py | 69 lines (recommended: <50) |
| 🟡 Deep Nesting | `build.sh` | /Users/masa/Projects/mcp-vector-search/scripts/build.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `changeset.py` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `consume` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `_update_changelog` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 60 lines (recommended: <50) |
| 🟡 High Complexity | `_update_changelog` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | Complexity: 17 (recommended: <15) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/changeset.py | 76 lines (recommended: <50) |
| 🟡 Deep Nesting | `comprehensive_build.py` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `BuildManager` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `run_integration_tests` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | 86 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/scripts/comprehensive_build.py | Complexity: 19 (recommended: <15) |
| 🟡 Deep Nesting | `deploy-test.sh` | /Users/masa/Projects/mcp-vector-search/scripts/deploy-test.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `dev-build.py` | /Users/masa/Projects/mcp-vector-search/scripts/dev-build.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/dev-build.py | 84 lines (recommended: <50) |
| 🟡 Deep Nesting | `dev-setup.py` | /Users/masa/Projects/mcp-vector-search/scripts/dev-setup.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `dev-test.sh` | /Users/masa/Projects/mcp-vector-search/scripts/dev-test.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `fix_linting.py` | /Users/masa/Projects/mcp-vector-search/scripts/fix_linting.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/fix_linting.py | 55 lines (recommended: <50) |
| 🟡 Deep Nesting | `full_release.sh` | /Users/masa/Projects/mcp-vector-search/scripts/full_release.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `generate_favicon.py` | /Users/masa/Projects/mcp-vector-search/scripts/generate_favicon.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `create_network_graph_icon` | /Users/masa/Projects/mcp-vector-search/scripts/generate_favicon.py | 74 lines (recommended: <50) |
| 🟡 Deep Nesting | `HOMEBREW_FORMULA_SUMMARY.md` | /Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_FORMULA_SUMMARY.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `HOMEBREW_QUICKSTART.md` | /Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_QUICKSTART.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `HOMEBREW_UPDATE_QUICKREF.md` | /Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_UPDATE_QUICKREF.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `HOMEBREW_WORKFLOW.md` | /Users/masa/Projects/mcp-vector-search/scripts/HOMEBREW_WORKFLOW.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `migrate_chromadb_metrics.py` | /Users/masa/Projects/mcp-vector-search/scripts/migrate_chromadb_metrics.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/migrate_chromadb_metrics.py | 72 lines (recommended: <50) |
| 🟡 Deep Nesting | `monitor_search_performance.py` | /Users/masa/Projects/mcp-vector-search/scripts/monitor_search_performance.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/monitor_search_performance.py | 54 lines (recommended: <50) |
| 🟡 Deep Nesting | `publish.sh` | /Users/masa/Projects/mcp-vector-search/scripts/publish.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `quick_search_timing.py` | /Users/masa/Projects/mcp-vector-search/scripts/quick_search_timing.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README_FAVICON.md` | /Users/masa/Projects/mcp-vector-search/scripts/README_FAVICON.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README_HOMEBREW_FORMULA.md` | /Users/masa/Projects/mcp-vector-search/scripts/README_HOMEBREW_FORMULA.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README_SUBMODULES.md` | /Users/masa/Projects/mcp-vector-search/scripts/README_SUBMODULES.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-vector-search/scripts/README.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `reorganize-docs.sh` | /Users/masa/Projects/mcp-vector-search/scripts/reorganize-docs.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `run_search_timing_tests.py` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `SearchTimingTestSuite` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_generate_medium_python_file` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `_generate_large_python_file` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `_generate_typescript_file` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `_generate_markdown_file` | /Users/masa/Projects/mcp-vector-search/scripts/run_search_timing_tests.py | 67 lines (recommended: <50) |
| 🟡 Deep Nesting | `run_tests.py` | /Users/masa/Projects/mcp-vector-search/scripts/run_tests.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/run_tests.py | 76 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/scripts/run_tests.py | Complexity: 17 (recommended: <15) |
| 🟡 Deep Nesting | `search_performance_monitor.py` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `run_quick_check` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `run_stress_test` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `check_search_quality` | /Users/masa/Projects/mcp-vector-search/scripts/search_performance_monitor.py | 56 lines (recommended: <50) |
| 🟡 Deep Nesting | `search_quality_analyzer.py` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `_create_quality_test_cases` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 90 lines (recommended: <50) |
| 🟡 Long Method | `test_edge_cases` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | 52 lines (recommended: <50) |
| 🟡 High Complexity | `generate_quality_report` | /Users/masa/Projects/mcp-vector-search/scripts/search_quality_analyzer.py | Complexity: 17 (recommended: <15) |
| 🟡 Deep Nesting | `setup_github_milestones.py` | /Users/masa/Projects/mcp-vector-search/scripts/setup_github_milestones.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/setup_github_milestones.py | 61 lines (recommended: <50) |
| 🟡 Deep Nesting | `setup_milestones.sh` | /Users/masa/Projects/mcp-vector-search/scripts/setup_milestones.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `setup-dev-mcp.sh` | /Users/masa/Projects/mcp-vector-search/scripts/setup-dev-mcp.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `update_docs.py` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `update_readme_version` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `update_claude_recent_activity` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | 73 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/update_docs.py | 68 lines (recommended: <50) |
| 🟡 Deep Nesting | `update_homebrew_formula.py` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `fetch_pypi_info` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 74 lines (recommended: <50) |
| 🟡 Long Method | `setup_tap_repository` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `run` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.py | 66 lines (recommended: <50) |
| 🟡 Deep Nesting | `update_homebrew_formula.sh` | /Users/masa/Projects/mcp-vector-search/scripts/update_homebrew_formula.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `update_submodules.sh` | /Users/masa/Projects/mcp-vector-search/scripts/update_submodules.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `verify-install.sh` | /Users/masa/Projects/mcp-vector-search/scripts/verify-install.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `version_manager.py` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `update_changelog` | /Users/masa/Projects/mcp-vector-search/scripts/version_manager.py | 61 lines (recommended: <50) |
| 🟡 Deep Nesting | `wait_and_update_homebrew.sh` | /Users/masa/Projects/mcp-vector-search/scripts/wait_and_update_homebrew.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `workflow.sh` | /Users/masa/Projects/mcp-vector-search/scripts/workflow.sh | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `BaselineComparator` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `compare` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `_compare_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `_classify_change` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/comparator.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `save_baseline` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `load_baseline` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `_get_git_info` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `_serialize_aggregate_metrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `_deserialize_project_metrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/baseline/manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `UnionFind` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `_calculate_class_cohesion` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/cohesion.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `collect_node` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `CyclomaticComplexityCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `NestingDepthCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `MethodCountCollector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/complexity.py | 94 lines (recommended: <50) |
| 🟡 Long Method | `ImportGraph` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `is_stdlib_module` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 81 lines (recommended: <50) |
| 🟡 Long Method | `_extract_import` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `build_import_graph` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/coupling.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `from_counts` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/halstead.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `_is_operand` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/collectors/halstead.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `DebtSummary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/debt.py | 74 lines (recommended: <50) |
| 🟡 Long Method | `_create_summary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/debt.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `_compute_threshold_comparisons` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 58 lines (recommended: <50) |
| 🟡 High Complexity | `AnalysisInterpreter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `_interpret_summary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/interpretation.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `print_distribution` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `print_hotspots` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `print_smells` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `print_recommendations` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/console.py | 90 lines (recommended: <50) |
| 🟡 High Complexity | `_build_fixes_markdown` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/markdown.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `generate_sarif` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/sarif.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `_smell_to_result` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/reporters/sarif.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `ProjectSnapshot` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 67 lines (recommended: <50) |
| 🟡 Long Method | `save_file_metrics` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `get_file_history` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `get_trends` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `_get_git_info` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/metrics_store.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `TrendData` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `get_trends` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 92 lines (recommended: <50) |
| 🟡 Long Method | `calculate_trend_direction` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `_find_regressions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `_find_improvements` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/storage/trend_tracker.py | 74 lines (recommended: <50) |
| 🟡 High Complexity | `TrendTracker` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/trends.py | Complexity: 25 (recommended: <15) |
| 🟡 Long Method | `compute_metrics_from_stats` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/trends.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `D3Node` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/d3_data.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `transform_for_d3` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/d3_data.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `_create_summary_stats` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/d3_data.py | 99 lines (recommended: <50) |
| 🟡 High Complexity | `JSONExporter` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | Complexity: 20 (recommended: <15) |
| 🟡 Long Method | `_create_summary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | 100 lines (recommended: <50) |
| 🟡 Long Method | `_convert_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/exporter.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `_generate_d3_graph_section` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 80 lines (recommended: <50) |
| 🟡 Long Method | `_generate_summary_panel` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `_generate_legend_with_counts` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `_generate_smells_section` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/html_report.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `MetricsSummary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/schemas.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `FileDetail` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/analysis/visualizer/schemas.py | 55 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | Complexity: 17 (recommended: <15) |
| 🟡 High Complexity | `_find_analyzable_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | Complexity: 23 (recommended: <15) |
| 🟡 Long Method | `_analyze_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py | 86 lines (recommended: <50) |
| 🟡 Long Method | `_show_auto_index_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/auto_index.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `_configure_auto_index` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/auto_index.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `_setup_auto_indexing` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/auto_index.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `ChatSession` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 87 lines (recommended: <50) |
| 🟡 Long Method | `_display_rich_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/chat.py | 81 lines (recommended: <50) |
| 🟡 Long Method | `set` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/config.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `reset` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/config.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `_parse_config_value` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/config.py | 56 lines (recommended: <50) |
| 🟡 High Complexity | `_parse_config_value` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/config.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `run_indexing` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `reindex_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `_reindex_entire_project` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `_reindex_single_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `init_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `configure_mcp_for_tool` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `setup_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | 96 lines (recommended: <50) |
| 🟡 Long Method | `print_next_steps` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | 51 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `detect_all_platforms` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 73 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | Complexity: 18 (recommended: <15) |
| 🟡 High Complexity | `_install_to_platform` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | Complexity: 18 (recommended: <15) |
| 🟡 High Complexity | `install_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `mcp_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 92 lines (recommended: <50) |
| 🟡 Long Method | `list_platforms` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `create_project_claude_config` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `configure_codex_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 76 lines (recommended: <50) |
| 🟡 High Complexity | `configure_codex_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `configure_auggie` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `configure_claude_code` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `configure_codex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `configure_gemini` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `list_tools` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `remove_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `show_mcp_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py | 89 lines (recommended: <50) |
| 🟡 Long Method | `reset_all` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/reset.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `search_similar_cmd` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | 72 lines (recommended: <50) |
| 🟡 Long Method | `search_context_cmd` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `register_with_claude_cli` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `scan_project_file_extensions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 64 lines (recommended: <50) |
| 🟡 High Complexity | `_setup_single_provider` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `setup_openrouter_api_key` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | Complexity: 23 (recommended: <15) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py | 57 lines (recommended: <50) |
| 🟡 High Complexity | `_display_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `perform_health_check` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `check_mcp_integration` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `show_metrics_summary` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `_output_metrics_json` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `check_dependencies` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py | 60 lines (recommended: <50) |
| 🟡 High Complexity | `uninstall_mcp` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/uninstall.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `extract_chunk_name` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `detect_cycles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | 67 lines (recommended: <50) |
| 🟡 Long Method | `apply_state` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `calculate_list_layout` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | 79 lines (recommended: <50) |
| 🟡 Long Method | `calculate_compact_folder_layout` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `_build_tree_levels` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/layout_engine.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `start_visualization_server` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/server.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `expand_node` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `collapse_node` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/state_manager.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `get_base_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `get_node_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 92 lines (recommended: <50) |
| 🟡 Long Method | `get_breadcrumb_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `get_theme_toggle_styles` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `watch_main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/watch.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `_watch_async` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/watch.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `EnhancedDidYouMeanTyper` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/didyoumean.py | 94 lines (recommended: <50) |
| 🟡 Long Method | `add_common_suggestions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/didyoumean.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `export_to_json` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `export_to_csv` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `export_to_markdown` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `export_summary_table` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/export.py | 74 lines (recommended: <50) |
| 🟡 High Complexity | `SearchHistory` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/history.py | Complexity: 23 (recommended: <15) |
| 🟡 Long Method | `_filter_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | 90 lines (recommended: <50) |
| 🟡 High Complexity | `_filter_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/interactive.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/main.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `cli_with_suggestions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/main.py | 70 lines (recommended: <50) |
| 🟡 High Complexity | `cli_with_suggestions` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/main.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `get_project_state` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `get_next_steps` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/suggestions.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `ProjectConfig` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py | 76 lines (recommended: <50) |
| 🟡 High Complexity | `ThresholdConfig` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/thresholds.py | Complexity: 18 (recommended: <15) |
| 🟡 High Complexity | `AutoIndexer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/auto_indexer.py | Complexity: 20 (recommended: <15) |
| 🟡 Long Method | `get_staleness_info` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/auto_indexer.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `VectorDatabase` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 98 lines (recommended: <50) |
| 🟡 Long Method | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 90 lines (recommended: <50) |
| 🟡 High Complexity | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 99 lines (recommended: <50) |
| 🟡 Long Method | `get_stats` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 85 lines (recommended: <50) |
| 🟡 Long Method | `get_all_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 68 lines (recommended: <50) |
| 🟡 High Complexity | `_detect_and_recover_corruption` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `_recover_from_corruption` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 68 lines (recommended: <50) |
| 🟡 High Complexity | `add_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `get_stats` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `get_all_chunks` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `rebuild_from_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/directory_index.py | 95 lines (recommended: <50) |
| 🟡 Long Method | `BatchEmbeddingProcessor` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py | 89 lines (recommended: <50) |
| 🟡 Long Method | `process_batch` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `create_standard_components` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/factory.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `GitChangeDetector` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git_hooks.py | 85 lines (recommended: <50) |
| 🟡 Long Method | `get_changed_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | 87 lines (recommended: <50) |
| 🟡 Long Method | `get_diff_files` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/git.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `__init__` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 98 lines (recommended: <50) |
| 🟡 High Complexity | `index_project` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | Complexity: 20 (recommended: <15) |
| 🟡 Long Method | `index_file` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `_should_ignore_path` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | 59 lines (recommended: <50) |
| 🟡 High Complexity | `_should_ignore_path` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | Complexity: 16 (recommended: <15) |
| 🟡 High Complexity | `_build_chunk_hierarchy` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `__init__` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 85 lines (recommended: <50) |
| 🟡 High Complexity | `__init__` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | Complexity: 17 (recommended: <15) |
| 🟡 Long Method | `generate_search_queries` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `analyze_and_rank_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `_chat_completion` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `_parse_ranking_response` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 88 lines (recommended: <50) |
| 🟡 Long Method | `detect_intent` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `stream_chat_completion` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 97 lines (recommended: <50) |
| 🟡 High Complexity | `stream_chat_completion` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `generate_answer` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `chat_with_tools` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py | 72 lines (recommended: <50) |
| 🟡 Long Method | `Directory` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/models.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `initialize` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py | 81 lines (recommended: <50) |
| 🟡 Long Method | `extract_chunk_name` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `compute_and_store` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `_compute_semantic_relationships` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `_compute_caller_relationships` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/relationships.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `_install_cron_job` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `_install_windows_task` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `_search_with_retry` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 95 lines (recommended: <50) |
| 🟡 Long Method | `search` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 85 lines (recommended: <50) |
| 🟡 Long Method | `_get_adaptive_threshold` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 72 lines (recommended: <50) |
| 🟡 High Complexity | `_rerank_results` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | Complexity: 17 (recommended: <15) |
| 🟡 Long Method | `analyze_query` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 72 lines (recommended: <50) |
| 🟡 Long Method | `suggest_related_queries` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `_enhance_with_file_context` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `_calculate_result_quality` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py | 52 lines (recommended: <50) |
| 🟡 High Complexity | `CodeFileHandler` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/watcher.py | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `initialize` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `_search_code` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `_get_project_status` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `_search_similar` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 89 lines (recommended: <50) |
| 🟡 Long Method | `_search_context` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 74 lines (recommended: <50) |
| 🟡 High Complexity | `_find_smells` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | Complexity: 17 (recommended: <15) |
| 🟡 Long Method | `_get_complexity_hotspots` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 76 lines (recommended: <50) |
| 🟡 High Complexity | `_check_circular_dependencies` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `_interpret_analysis` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `_calculate_complexity` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/base.py | 97 lines (recommended: <50) |
| 🟡 Long Method | `_create_chunk` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/base.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `_extract_chunks_from_tree` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | 59 lines (recommended: <50) |
| 🟡 High Complexity | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/dart.py | Complexity: 17 (recommended: <15) |
| 🟡 High Complexity | `HTMLContentParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | Complexity: 25 (recommended: <15) |
| 🟡 Long Method | `handle_starttag` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 56 lines (recommended: <50) |
| 🟡 High Complexity | `HTMLParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | Complexity: 25 (recommended: <15) |
| 🟡 Long Method | `parse_content` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `_merge_small_sections` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/html.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `_extract_chunks_from_tree` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/javascript.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `_extract_chunks_from_tree` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | 76 lines (recommended: <50) |
| 🟡 High Complexity | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | Complexity: 18 (recommended: <15) |
| 🟡 High Complexity | `_extract_phpdoc_regex` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/php.py | Complexity: 23 (recommended: <15) |
| 🟡 Long Method | `_extract_chunks_from_tree` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `_fallback_parse` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/python.py | 94 lines (recommended: <50) |
| 🟡 Long Method | `_extract_chunks_from_tree` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/ruby.py | 71 lines (recommended: <50) |
| 🟡 High Complexity | `TextParser` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `parse_content` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `_extract_paragraphs` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/text.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `extract_docstring` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/utils.py | 75 lines (recommended: <50) |
| 🟡 High Complexity | `extract_docstring` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/utils.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `find_code_blocks_with_patterns` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/utils.py | 53 lines (recommended: <50) |
| 🟡 High Complexity | `ensure_gitignore_entry` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore_updater.py | Complexity: 20 (recommended: <15) |
| 🟡 Long Method | `GitignorePattern` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore.py | 89 lines (recommended: <50) |
| 🟡 Long Method | `matches` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/gitignore.py | 55 lines (recommended: <50) |
| 🟡 High Complexity | `PerformanceProfiler` | /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/utils/timing.py | Complexity: 19 (recommended: <15) |
| 🟡 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-vector-search/tests/__init__.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `conftest.py` | /Users/masa/Projects/mcp-vector-search/tests/conftest.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_search_command_with_glob_pattern` | /Users/masa/Projects/mcp-vector-search/tests/e2e/test_cli_commands.py | 62 lines (recommended: <50) |
| 🟡 High Complexity | `TestIndexingWorkflow` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_indexing_workflow.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `python_project` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `test_setup_complete_workflow` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `TestSetupPerformance` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 67 lines (recommended: <50) |
| 🟡 Long Method | `TestSetupConfiguration` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py | 80 lines (recommended: <50) |
| 🟡 Long Method | `TestDirectoryExpansion` | /Users/masa/Projects/mcp-vector-search/tests/integration/test_visualization_v2_flow.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `timing_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_loading_timing.py | 73 lines (recommended: <50) |
| 🟡 Long Method | `debug_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/debug_visualization_simple.py | 97 lines (recommended: <50) |
| 🟡 High Complexity | `run_breadcrumb_test` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_breadcrumb_fix.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `create_test_graph_data` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_compact_folder_layout.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `test_file_scan` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_file_scan_ewtn.py | 81 lines (recommended: <50) |
| 🟡 Long Method | `test_full_index` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_full_index_ewtn.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `test_gitignore` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_gitignore_ewtn.py | 65 lines (recommended: <50) |
| 🟡 High Complexity | `test_glob_filtering` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_glob_pattern_filtering.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `test_root_node_detection` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_root_detection_direct.py | 84 lines (recommended: <50) |
| 🟡 High Complexity | `test_root_node_detection` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_root_detection_direct.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `test_with_debugger` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer_line_by_line.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer.py | 95 lines (recommended: <50) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_visualizer.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `test_with_cdp` | /Users/masa/Projects/mcp-vector-search/tests/manual/test_with_cdp.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_root_filtering.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `verify` | /Users/masa/Projects/mcp-vector-search/tests/manual/verify_with_wait.js | 78 lines (recommended: <50) |
| 🟡 Long Method | `AuthenticationManager` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_javascript.js | 96 lines (recommended: <50) |
| 🟡 Long Method | `User` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_python.py | 73 lines (recommended: <50) |
| 🟡 Long Method | `AuthenticationManager` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/ast_test_python.py | 87 lines (recommended: <50) |
| 🟡 Long Method | `test_python_parser` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | 99 lines (recommended: <50) |
| 🟡 High Complexity | `test_python_parser` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `test_javascript_parser` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `test_typescript_parser` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `test_hierarchy_building` | /Users/masa/Projects/mcp-vector-search/tests/sample_code/test_ast_features.py | 56 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_basic_functionality.py` | /Users/masa/Projects/mcp-vector-search/tests/test_basic_functionality.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_indexing_and_search` | /Users/masa/Projects/mcp-vector-search/tests/test_basic_functionality.py | 56 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_dart_parser.py` | /Users/masa/Projects/mcp-vector-search/tests/test_dart_parser.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `test_html_parser.py` | /Users/masa/Projects/mcp-vector-search/tests/test_html_parser.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_html_edge_cases` | /Users/masa/Projects/mcp-vector-search/tests/test_html_parser.py | 79 lines (recommended: <50) |
| 🟡 Long Method | `test_html_semantic_structure` | /Users/masa/Projects/mcp-vector-search/tests/test_html_parser.py | 90 lines (recommended: <50) |
| 🟡 Long Method | `test_html_chunk_merging` | /Users/masa/Projects/mcp-vector-search/tests/test_html_parser.py | 62 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_js_parser.py` | /Users/masa/Projects/mcp-vector-search/tests/test_js_parser.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_javascript_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_js_parser.py | 97 lines (recommended: <50) |
| 🟡 Long Method | `test_typescript_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_js_parser.py | 90 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_mcp_integration.py` | /Users/masa/Projects/mcp-vector-search/tests/test_mcp_integration.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `test_php_parser.py` | /Users/masa/Projects/mcp-vector-search/tests/test_php_parser.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `test_php_parser` | /Users/masa/Projects/mcp-vector-search/tests/test_php_parser.py | Complexity: 17 (recommended: <15) |
| 🟡 Deep Nesting | `test_ruby_parser.py` | /Users/masa/Projects/mcp-vector-search/tests/test_ruby_parser.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `test_search_performance.py` | /Users/masa/Projects/mcp-vector-search/tests/test_search_performance.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `SearchPerformanceTester` | /Users/masa/Projects/mcp-vector-search/tests/test_search_performance.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `performance_tester` | /Users/masa/Projects/mcp-vector-search/tests/test_search_performance.py | 56 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_simple.py` | /Users/masa/Projects/mcp-vector-search/tests/test_simple.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_basic_functionality` | /Users/masa/Projects/mcp-vector-search/tests/test_simple.py | 64 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_version_reindex.py` | /Users/masa/Projects/mcp-vector-search/tests/test_version_reindex.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `test_auto_reindex_integration` | /Users/masa/Projects/mcp-vector-search/tests/test_version_reindex.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `test_compare_multiple_files` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/baseline/test_comparator.py | 79 lines (recommended: <50) |
| 🟡 Long Method | `TestHalsteadMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/collectors/test_halstead.py | 98 lines (recommended: <50) |
| 🟡 Long Method | `TestMetricsStoreInitialization` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 88 lines (recommended: <50) |
| 🟡 Long Method | `TestFileMetricsSaving` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 100 lines (recommended: <50) |
| 🟡 Long Method | `TestCompleteSnapshot` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 81 lines (recommended: <50) |
| 🟡 Long Method | `TestHistoryQueries` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `TestTrendAnalysis` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_metrics_store.py | 99 lines (recommended: <50) |
| 🟡 Long Method | `create_project_metrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `TestCalculateTrendDirection` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `TestRegressionAlerts` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `TestTrendDataProperties` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `TestEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/storage/test_trend_tracker.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `test_collector_workflow` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_collectors.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `TestGetNodeTypes` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `TestNestingDepthCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 96 lines (recommended: <50) |
| 🟡 Long Method | `TestParameterCountCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 91 lines (recommended: <50) |
| 🟡 Long Method | `TestMethodCountCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_complexity_collectors.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `TestAfferentCouplingCollector` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 95 lines (recommended: <50) |
| 🟡 Long Method | `TestCouplingMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `TestCollectorIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `TestImportGraph` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `TestBuildImportGraphFromDict` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_coupling.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `TestDebtSummary` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `test_debt_summary_to_dict` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | 73 lines (recommended: <50) |
| 🟡 Long Method | `test_create_summary_aggregations` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_debt.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `TestThresholdContext` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `sample_llm_export` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | 90 lines (recommended: <50) |
| 🟡 Long Method | `TestIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/test_interpretation.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `TestD3Node` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `TestCreateNode` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `TestCalculateWorstSeverity` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 99 lines (recommended: <50) |
| 🟡 Long Method | `TestTransformForD3` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `test_create_summary_stats_complete` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestEnhancedD3Node` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `test_create_module_groups_with_clusters` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_d3_visualization.py | 87 lines (recommended: <50) |
| 🟡 Long Method | `TestJSONExporter` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_exporter.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `TestMetadataCreation` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_exporter.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestSummaryCreation` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_exporter.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `TestEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_exporter.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `TestAccessibilityFeatures` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_phase5.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `TestExportFunctionality` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_phase5.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `TestPerformanceOptimizations` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_phase5.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `test_filter_module_options` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_html_report.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `TestMetricsSummary` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 83 lines (recommended: <50) |
| 🟡 Long Method | `TestFunctionMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `TestSmellLocation` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestFileDetail` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `TestModelDumpMethods` | /Users/masa/Projects/mcp-vector-search/tests/unit/analysis/visualizer/test_schemas.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `test_git_diff_files_baseline` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze_git.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `TestConsoleReporter` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/commands/test_analyze.py | 79 lines (recommended: <50) |
| 🟡 Long Method | `test_analyze_basic_query` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_chat_analyze.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `test_analyze_fallback_on_stream_error` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_chat_analyze.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `test_analyze_uses_advanced_model` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_chat_analyze.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `TestGradeFilterParser` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_search_quality_filters.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `test_combined_filters` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_search_quality_filters.py | 80 lines (recommended: <50) |
| 🟡 Long Method | `TestQualityScoreCalculation` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_search_quality_filters.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `TestStatusMetricsIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_status.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `test_metrics_with_code_smells_and_complexity` | /Users/masa/Projects/mcp-vector-search/tests/unit/cli/test_status.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `TestApiKeySetupIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup_api_key.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `TestScanProjectFileExtensions` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 97 lines (recommended: <50) |
| 🟡 Long Method | `test_setup_configures_all_platforms` | /Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `TestThresholdConfig` | /Users/masa/Projects/mcp-vector-search/tests/unit/config/test_thresholds.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `TestThresholdConfigYAML` | /Users/masa/Projects/mcp-vector-search/tests/unit/config/test_thresholds.py | 95 lines (recommended: <50) |
| 🟡 High Complexity | `TestChromaConnectionPool` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_connection_pool.py | Complexity: 25 (recommended: <15) |
| 🟡 Long Method | `test_exponential_backoff_timing` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_corruption_recovery.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `test_search_filter_by_complexity_grade` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `test_search_filter_by_smell_count` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `test_search_filter_by_complexity_range` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `test_multiple_chunks_same_file` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `TestMetricsIntegration` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database_metrics.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `test_sqlite_corruption_detection` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `test_rust_panic_recovery` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_database.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `TestIndexerCollectorInitialization` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `TestCollectMetrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `TestEstimateComplexity` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `test_index_file_collects_metrics` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `test_index_file_with_no_collectors` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `test_index_file_handles_metric_collection_errors` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_indexer_collectors.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestIntentDetectionEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_llm_client_intent.py | 78 lines (recommended: <50) |
| 🟡 Long Method | `TestCodeChunk` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_models.py | 91 lines (recommended: <50) |
| 🟡 Long Method | `TestSearchResult` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_models.py | 86 lines (recommended: <50) |
| 🟡 Long Method | `test_pure_quality_ranking_weight_one` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_quality_ranking.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `test_balanced_ranking_weight_0_3` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_quality_ranking.py | 87 lines (recommended: <50) |
| 🟡 High Complexity | `TestSemanticSearchEngine` | /Users/masa/Projects/mcp-vector-search/tests/unit/core/test_search.py | Complexity: 21 (recommended: <15) |
| 🟡 Long Method | `TestAnalyzeProject` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_analysis_tools.py | 74 lines (recommended: <50) |
| 🟡 Long Method | `TestAnalyzeFile` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_analysis_tools.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `sample_analysis_json` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_interpret_analysis.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `TestIntegrationWorkflow` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_interpret_analysis.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `test_cli_to_mcp_workflow` | /Users/masa/Projects/mcp-vector-search/tests/unit/mcp/test_interpret_analysis.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `TestListLayout` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 82 lines (recommended: <50) |
| 🟡 Long Method | `TestCompactFolderLayout` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `TestBuildTreeLevels` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `TestEdgeCases` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_layout_engine.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestEndToEndScenarios` | /Users/masa/Projects/mcp-vector-search/tests/unit/test_mcp_install_auto_detection.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `User` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 97 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 66 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 52 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 62 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 72 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 67 lines (recommended: <50) |
| 🟡 Long Method | `Operation` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 72 lines (recommended: <50) |
| 🟡 Long Method | `MCPInstaller` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/DIAGRAMS.md | 57 lines (recommended: <50) |
| 🟡 Long Method | `py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/IMPLEMENTATION-PLAN.md | 69 lines (recommended: <50) |
| 🟡 Long Method | `py` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/docs/PROJECT-STRUCTURE.md | 88 lines (recommended: <50) |
| 🟡 Long Method | `build_command` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `detect_best_method` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `write` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `add_server` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `validate` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `migrate_legacy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `InstallationStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 68 lines (recommended: <50) |
| 🟡 High Complexity | `NativeCLIStrategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `__init__` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 62 lines (recommended: <50) |
| 🟡 High Complexity | `install_server` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_select_strategy` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `InspectionReport` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `detect` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `detect_for_platform` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `detect_claude_code` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `detect_claude_desktop` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `validate_json_structure` | /Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/utils.py | 51 lines (recommended: <50) |
| 🟡 Deep Nesting | `visualize` | src/mcp_vector_search/cli/commands/visualize | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `exporters` | src/mcp_vector_search/cli/commands/visualize/exporters | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `templates` | src/mcp_vector_search/cli/commands/visualize/templates | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `platforms` | vendor/py-mcp-installer-service/src/py_mcp_installer/platforms | Depth: 5 (recommended: <4) |

---

## Recommended Actions

1. **Start with Grade F items** - These have the highest complexity and are hardest to maintain
2. **Address Critical code smells** - God Classes and deeply nested code should be refactored
3. **Break down long methods** - Extract helper functions to reduce complexity
4. **Add tests before refactoring** - Ensure behavior is preserved

---

_Generated by MCP Vector Search Visualization_
