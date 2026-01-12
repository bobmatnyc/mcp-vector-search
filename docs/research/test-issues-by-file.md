# Test Issues - Detailed File Analysis

**Generated:** 2026-01-12
**Purpose:** Comprehensive list of test files with quality issues

## Files with Critical Issues

### tests/unit/commands/test_setup.py
- **Size:** 913 lines
- **Issue:** 676 lines (74%) are skipped test code
- **Reason:** "Tests need update for py_mcp_installer PlatformInfo data structure"
- **Affected:** Lines 238-914
- **Test classes skipped:**
  - `TestSetupCommand` - 10 tests (lines 238-578)
  - `TestSetupErrorHandling` - 6 tests (lines 585-746)
  - `TestSetupVerboseMode` - 1 test (lines 752-786)
  - `TestSetupEdgeCases` - 7 tests (lines 793-914)
- **Recommendation:** DELETE skipped sections or UPDATE entire file
- **Priority:** CRITICAL

### tests/integration/test_setup_integration.py
- **Size:** 876 lines
- **Issue:** 5 integration tests skipped
- **Reason:** Same PlatformInfo migration issue
- **Affected tests:**
  - Line 196: `test_setup_fresh_project_creates_index`
  - Line 226: `test_setup_reconfigures_existing_project`
  - Line 263: `test_setup_with_force_flag_reindexes`
  - Line 292: `test_setup_detects_languages`
  - Line 324: `test_setup_handles_errors_gracefully`
- **Recommendation:** DELETE or UPDATE all skipped tests
- **Priority:** CRITICAL

### tests/test_simple.py
- **Size:** 253 lines
- **Issue:** Duplicate of test_basic_functionality.py
- **Content:**
  - `test_basic_functionality()` - async test (lines 23-87)
  - `main()` - 140 lines of duplicate implementation (lines 90-230)
  - `main_wrapper()` - CLI entry point, not a proper test (lines 233-248)
- **Recommendation:** DELETE entire file
- **Priority:** CRITICAL

### tests/unit/cli/test_chat_analyze.py
- **Size:** ~400 lines
- **Issues:**
  1. 3 tests properly skipped (integration tests)
  2. 10+ empty test placeholders with only `pass` (lines 313-368)
- **Empty tests:**
  - `test_analyze_markdown_files` (line 313)
  - `test_analyze_javascript_project` (line 319)
  - `test_analyze_typescript_project` (line 324)
  - `test_analyze_rust_project` (line 329)
  - `test_analyze_go_project` (line 334)
  - `test_analyze_mixed_language_project` (line 344)
  - `test_analyze_with_custom_metrics` (line 350)
  - `test_analyze_with_baseline_comparison` (line 356)
  - `test_analyze_with_trend_analysis` (line 362)
  - `test_analyze_error_handling` (line 368)
- **Recommendation:** DELETE empty placeholders (lines 313-368)
- **Priority:** HIGH

### tests/e2e/test_cli_commands.py
- **Issue:** 2 tests skipped
- **Reason:** "CliRunner doesn't support concurrent operations well"
- **Skipped tests:**
  - `test_concurrent_indexing`
  - `test_concurrent_search`
- **Recommendation:** DELETE - concurrency not critical for CLI
- **Priority:** MEDIUM

## Files Requiring Refactoring

### tests/unit/analysis/visualizer/test_d3_visualization.py
- **Size:** 1,131 lines (LARGEST test file)
- **Issue:** Too large, poor organization
- **Recommendation:** Split into 4 files:
  - test_d3_layout.py (layout algorithms)
  - test_d3_interactions.py (user interactions)
  - test_d3_rendering.py (SVG generation)
  - test_d3_data_binding.py (data structures)
- **Priority:** MEDIUM

### tests/unit/analysis/test_coupling.py
- **Size:** 1,004 lines
- **Issues:**
  1. Very large file
  2. Contains skipped tests
- **Recommendation:** Split into 3 files:
  - test_coupling_metrics.py
  - test_coupling_detection.py
  - test_coupling_reporting.py
- **Priority:** MEDIUM

### tests/unit/analysis/visualizer/test_html_report.py
- **Size:** 825 lines
- **Issue:** Large file with multiple concerns
- **Recommendation:** Split into:
  - test_html_structure.py
  - test_html_styles.py
  - test_html_data.py
- **Priority:** LOW

### tests/unit/analysis/visualizer/test_schemas.py
- **Size:** 817 lines
- **Issue:** Large file testing multiple schemas
- **Recommendation:** Split by schema type
- **Priority:** LOW

## Misplaced Tests

### tests/manual/test_api_key_obfuscation.py
- **Size:** 61 lines
- **Issue:** Pure unit test in manual directory
- **Current location:** tests/manual/
- **Should be:** tests/unit/commands/test_api_key_obfuscation.py
- **Recommendation:** MOVE to unit tests
- **Priority:** HIGH

### tests/unit/cli/test_chat_analyze.py (skipped tests)
- **Issue:** Integration tests in unit directory
- **Tests with 6+ mocks:** Lines 53-100, 102-150, 152-200
- **Reason for skip:** "Integration test - requires full mocking"
- **Recommendation:** MOVE to tests/integration/test_chat_analyze_integration.py
- **Priority:** MEDIUM

## Manual Test Directory Issues

### tests/manual/ - Debug Artifacts

**39 non-test files to delete:**

**HTML files (8):**
- bug_visualization.html
- debug_console_inspector.html
- debug_hierarchy.html
- debug-d3-tree-simple.html
- test-link-structure.html
- test_visualization.html
- verify_cytoscape_fix.html
- verify_with_node.js (JavaScript)

**Shell scripts (5):**
- apply_fix.sh
- test_analyze_git_integration.sh
- test_controls_safari.sh
- test_mcp_auto_install.sh
- verify_streaming_load.sh

**Screenshots (2 + directory):**
- screenshot_after_wait.png
- screenshot_verification.png
- screenshots/ (13 files inside)

**Documentation/Reports (5):**
- analyze-command-test-report.md
- d3-tree-collapse-test.md
- file-click-debug-checklist.md
- test-null-checks.md
- verify-tree-connector-lines.md

**JSON files (4):**
- breadcrumb_test_results.json
- test_graph_large.json
- test_graph_medium.json
- test_graph_small.json
- verification_report.json

**JavaScript files (3):**
- debug_network.js
- inspect_visualization_controls.js
- verify_with_node.js
- verify_with_wait.js

**AppleScript files (2):**
- check_controls.scpt
- get_selector_details.scpt
- test_css_fix.scpt

**Node.js config (3):**
- package.json
- package-lock.json
- node_modules/ (should be in .gitignore)

**Python debug scripts (6):**
- debug_loading_timing.py
- debug_visualization_simple.py
- debug_visualizer.py
- investigate_visualization.py
- verify_data_initialization_fix.py
- verify_root_filtering.py
- verify_visualization.py

**Recommendation:** DELETE all non-.py files, add patterns to .gitignore

## Test Organization Issues

### Root tests/ directory vs tests/unit/

**Duplicate organization:**

Files in `tests/` root that should be in subdirectories:

1. **Parser tests** (should be in tests/unit/parsers/):
   - test_dart_parser.py (320 lines)
   - test_html_parser.py (489 lines)
   - test_js_parser.py (163 lines)
   - test_php_parser.py (551 lines)
   - test_ruby_parser.py (352 lines)

2. **Integration tests** (already have tests/integration/):
   - test_basic_functionality.py (168 lines) - could be integration
   - test_mcp_integration.py (206 lines) - should be in tests/integration/
   - test_version_reindex.py (313 lines) - should be in tests/integration/

3. **Performance tests** (should be in tests/performance/):
   - test_search_performance.py (410 lines)

**Recommendation:** Reorganize into proper subdirectories

## Files with Weak Assertions

### Parser Tests (Needs Review)

All parser tests should be reviewed for assertion quality:

1. **test_dart_parser.py** (320 lines)
   - Check: Tests beyond "doesn't crash"
   - Review: AST structure validation
   - Review: Error handling tests

2. **test_html_parser.py** (489 lines)
   - Check: Proper HTML structure validation
   - Review: Edge case coverage

3. **test_js_parser.py** (163 lines)
   - Check: JavaScript-specific syntax validation
   - Review: Modern JS features (ES6+)

4. **test_php_parser.py** (551 lines)
   - Check: PHP version compatibility tests
   - Review: Namespace handling

5. **test_ruby_parser.py** (352 lines)
   - Check: Ruby-specific syntax validation
   - Review: Metaprogramming edge cases

**Recommendation:** Code review each parser test file for meaningful assertions

## Performance Test Issues

### tests/test_search_performance.py
- **Issue:** Custom timing instead of pytest-benchmark
- **Lines:** 410 lines
- **Recommendation:** Refactor to use pytest-benchmark plugin
- **Priority:** LOW

### tests/manual/test_multiprocessing_performance.py
- **Issue:** Manual benchmark in wrong directory
- **Recommendation:** Convert to pytest-benchmark and move to tests/performance/
- **Priority:** LOW

## Summary Statistics

**Critical Issues (Immediate Action Required):**
- 3 files with skipped/dead code (1,542 lines total)
- 1 duplicate file (253 lines)
- 1 file with empty test placeholders (10+ tests)

**High Priority Issues:**
- 39 debug artifact files in manual directory
- 1 misplaced unit test
- 10+ empty test methods

**Medium Priority Issues:**
- 4 large test files needing split (>800 lines each)
- 2 skipped e2e tests
- 3 integration tests in unit directory

**Low Priority Issues:**
- 5 parser test files needing assertion review
- 2 performance test files needing standardization
- Test organization improvements

**Total Impact:**
- Remove: 1,000+ lines of dead code
- Improve: 2,000+ lines through refactoring
- Clean up: 39 non-test artifact files
