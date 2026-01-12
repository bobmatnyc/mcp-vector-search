# Test Suite Cleanup - Prioritized Action Items

**Generated:** 2026-01-12
**Status:** Ready for implementation
**Impact:** Remove 1,000+ lines of dead code, improve test quality

## Critical Priority (Do First)

### 1. Remove Skipped Test Dead Code (Est. 2-3 hours)

**Impact:** Delete 800+ lines of unmaintained code

#### File: tests/unit/commands/test_setup.py
```bash
# Current: 913 lines, 676 lines skipped (74% dead code)
# Target: Either update or delete entirely

# Option A: Delete skipped test classes (RECOMMENDED)
# Remove lines 238-914 (4 test classes):
# - TestSetupCommand (10 tests)
# - TestSetupErrorHandling (6 tests)
# - TestSetupVerboseMode (1 test)
# - TestSetupEdgeCases (7 tests)

# Option B: Update for PlatformInfo structure
# Requires significant refactoring to use py_mcp_installer.detect_all_platforms()
# Estimate: 4-6 hours of work
```

**Decision needed:** Delete vs Update?
- These tests have been skipped since py_mcp_installer integration
- Current setup functionality works without these tests
- **Recommendation:** DELETE and create new tests if setup issues arise

#### File: tests/integration/test_setup_integration.py
```bash
# Current: 876 lines, 5 tests skipped
# Same issue as above - PlatformInfo migration

# Delete skipped tests or entire file if all critical tests are skipped
```

**Commands:**
```bash
# Review what's skipped
grep -n "@pytest.mark.skip" tests/unit/commands/test_setup.py
grep -n "@pytest.mark.skip" tests/integration/test_setup_integration.py

# After decision, remove skipped sections
# (Manual editing required to preserve any non-skipped tests)
```

### 2. Delete Duplicate Test File (Est. 15 minutes)

**File:** tests/test_simple.py (253 lines)

**Issue:** Duplicates test_basic_functionality.py

**Evidence:**
```python
# test_simple.py has:
- test_basic_functionality() - covers ComponentFactory.create_standard_components()
- main() function - 140 lines of duplicate test logic
- main_wrapper() - CLI entry point (not a test)

# test_basic_functionality.py has:
- test_project_initialization() - covers ProjectManager
- test_indexing_and_search() - covers indexing workflow

# Both test the same functionality with different approaches
```

**Action:**
```bash
# 1. Verify test_basic_functionality.py covers all scenarios
pytest tests/test_basic_functionality.py -v

# 2. Delete test_simple.py
rm tests/test_simple.py

# 3. Verify test suite still passes
pytest tests/ -v --ignore=tests/manual --ignore=tests/e2e -x
```

### 3. Implement or Delete Empty Test Placeholders (Est. 30 minutes)

**File:** tests/unit/cli/test_chat_analyze.py

**Issue:** 10+ test methods with only `pass` statement

**Empty tests (lines 313-368):**
```python
def test_analyze_markdown_files(self): pass
def test_analyze_javascript_project(self): pass
def test_analyze_typescript_project(self): pass
def test_analyze_rust_project(self): pass
def test_analyze_go_project(self): pass
def test_analyze_mixed_language_project(self): pass
def test_analyze_with_custom_metrics(self): pass
def test_analyze_with_baseline_comparison(self): pass
def test_analyze_with_trend_analysis(self): pass
def test_analyze_error_handling(self): pass
```

**Decision:**
- These have been placeholders for months
- Chat analyze functionality works without them
- **Recommendation:** DELETE empty tests

**Action:**
```bash
# Delete lines 313-368 in test_chat_analyze.py
# Keep the 3 skipped tests at top (they have @pytest.mark.skip with reason)
```

## High Priority (This Week)

### 4. Clean Up Manual Test Directory (Est. 1 hour)

**Issue:** 70 files in tests/manual/, most are debug artifacts

**Breakdown:**
- 31 actual test files (.py)
- 39 debug artifacts (.sh, .html, .png, .json, .md, .js, .scpt)

**Debug artifacts to delete:**
```
tests/manual/
├── *.html (8 files) - debug visualizations
├── *.png (2 files) - screenshots
├── *.json (4 files) - test data
├── *.md (5 files) - investigation notes
├── *.sh (5 files) - shell scripts
├── *.js (3 files) - debug scripts
├── *.scpt (2 files) - AppleScript
├── screenshots/ (directory) - 13 files
├── node_modules/ (directory) - should be in .gitignore
└── package*.json (2 files) - Playwright config
```

**Actions:**
```bash
cd tests/manual/

# Delete debug artifacts
rm -f *.html *.png *.json *.md *.sh *.js *.scpt
rm -rf screenshots/

# Add to .gitignore if not already there
echo "tests/manual/*.html" >> .gitignore
echo "tests/manual/*.png" >> .gitignore
echo "tests/manual/*.json" >> .gitignore
echo "tests/manual/*.md" >> .gitignore
echo "tests/manual/screenshots/" >> .gitignore
echo "tests/manual/node_modules/" >> .gitignore

# Review remaining 31 .py files for move/delete candidates
ls *.py
```

### 5. Move Misplaced Unit Test (Est. 10 minutes)

**File:** tests/manual/test_api_key_obfuscation.py

**Issue:** This is a proper unit test, not a manual test

**Evidence:**
```python
#!/usr/bin/env python3
"""Manual test for API key obfuscation logic."""

def test_obfuscation():
    """Test API key obfuscation with various inputs."""
    test_cases = [...]
    # Pure function testing, no I/O, no manual verification
```

**Action:**
```bash
# Move to proper location
mv tests/manual/test_api_key_obfuscation.py \
   tests/unit/commands/test_api_key_obfuscation.py

# Update imports if needed (should work as-is)

# Run to verify
pytest tests/unit/commands/test_api_key_obfuscation.py -v
```

## Medium Priority (Next Sprint)

### 6. Split Large Test Files (Est. 3-4 hours)

Files over 800 lines should be split into focused modules:

#### test_d3_visualization.py (1,131 lines)
```bash
# Split into:
tests/unit/analysis/visualizer/
├── test_d3_layout.py        # Layout algorithm tests
├── test_d3_interactions.py  # Click/drag/zoom tests
├── test_d3_rendering.py     # SVG generation tests
└── test_d3_data_binding.py  # D3 data structure tests
```

#### test_coupling.py (1,004 lines)
```bash
# Split into:
tests/unit/analysis/
├── test_coupling_metrics.py     # Coupling calculation
├── test_coupling_detection.py   # Coupling pattern detection
└── test_coupling_reporting.py   # Coupling report generation
```

#### test_html_report.py (825 lines)
```bash
# Split into:
tests/unit/analysis/visualizer/
├── test_html_structure.py   # HTML template tests
├── test_html_styles.py      # CSS generation tests
└── test_html_data.py        # Data injection tests
```

**Strategy:**
1. Identify logical test groupings (usually by test class)
2. Create new test files
3. Move test classes to appropriate files
4. Update imports
5. Verify all tests still pass

### 7. Fix or Delete Skipped E2E Tests (Est. 1 hour)

**File:** tests/e2e/test_cli_commands.py

**Skipped tests:**
```python
@pytest.mark.skip(reason="CliRunner doesn't support concurrent operations well")
def test_concurrent_indexing(self):
    pass

@pytest.mark.skip(reason="CliRunner doesn't support concurrent operations well")
def test_concurrent_search(self):
    pass
```

**Options:**
1. Use subprocess instead of CliRunner for concurrent tests
2. Delete tests if concurrency not critical for CLI
3. Move to integration tests with proper async handling

**Recommendation:** Delete - CLI concurrency is not a primary use case

### 8. Move Integration Tests from Unit Directory (Est. 30 minutes)

**File:** tests/unit/cli/test_chat_analyze.py (remaining tests)

**Issue:** Tests with 6+ mocks are integration tests

**Evidence:**
```python
@pytest.mark.skip(reason="Integration test - requires full mocking")
def test_analyze_basic_query(self):
    with (
        patch("mcp_vector_search.core.llm_client.LLMClient"),
        patch("mcp_vector_search.core.project.ProjectManager"),
        patch("mcp_vector_search.core.config_utils.get_openai_api_key"),
        # ... 4 more patches
    ):
```

**Action:**
```bash
# Move to integration tests
mv tests/unit/cli/test_chat_analyze.py \
   tests/integration/test_chat_analyze_integration.py

# Update test markers
# Remove @pytest.mark.skip if implementing proper integration setup
```

## Low Priority (Future Improvements)

### 9. Standardize Performance Tests (Est. 2 hours)

**Current state:**
- test_search_performance.py uses custom timing
- tests/manual/test_multiprocessing_performance.py uses custom benchmarks

**Goal:** Use pytest-benchmark plugin consistently

**Example refactor:**
```python
# Before
async def test_basic_search_timing(performance_tester):
    start = time.perf_counter()
    result = await search_engine.search(query)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

# After
def test_basic_search_timing(benchmark):
    result = benchmark(lambda: search_engine.search(query))
    assert benchmark.stats.mean < 1.0
```

### 10. Review Parser Test Quality (Est. 1-2 hours)

**Files to review:**
- test_dart_parser.py (320 lines)
- test_html_parser.py (489 lines)
- test_js_parser.py (163 lines)
- test_php_parser.py (459 lines)
- test_ruby_parser.py (352 lines)

**Check for:**
- Assertions beyond "parsing doesn't crash"
- Edge case coverage
- Error handling validation
- AST structure verification

## Implementation Checklist

### Week 1: Critical Priority
- [ ] Decide: Delete or update test_setup.py skipped tests
- [ ] Delete test_simple.py
- [ ] Delete empty test placeholders in test_chat_analyze.py
- [ ] Run full test suite to verify: `pytest tests/ --ignore=tests/manual --ignore=tests/e2e`
- [ ] Commit: "Clean up skipped and duplicate tests"

### Week 2: High Priority
- [ ] Clean up tests/manual/ debug artifacts
- [ ] Move test_api_key_obfuscation.py to unit tests
- [ ] Update .gitignore for manual test artifacts
- [ ] Run full test suite to verify
- [ ] Commit: "Clean up manual test directory"

### Week 3: Medium Priority
- [ ] Split test_d3_visualization.py
- [ ] Split test_coupling.py
- [ ] Fix or delete skipped e2e tests
- [ ] Run full test suite to verify
- [ ] Commit: "Refactor large test files"

### Week 4: Polish
- [ ] Move integration tests from unit/
- [ ] Standardize performance tests
- [ ] Review parser test quality
- [ ] Document test organization standards
- [ ] Commit: "Improve test organization and quality"

## Expected Outcomes

After completing all actions:

**Lines of Code:**
- Remove: 1,000+ lines (skipped tests, duplicates, dead code)
- Improve: 2,000+ lines (split large files, better organization)

**Test Quality:**
- All tests either passing or properly marked as expected failures
- Clear separation: unit / integration / e2e / manual
- Faster test execution (fewer redundant tests)
- Better test discoverability (smaller, focused files)

**Maintainability:**
- No dead code waiting for migration
- Clear test organization
- Standard patterns for performance testing
- Clean manual test directory

## Notes

- All changes should be committed incrementally
- Run test suite after each major change
- Consider creating issues for each priority level
- Coordinate with team on test_setup.py decision (delete vs update)
