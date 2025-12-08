# Breadcrumb Root Display Bug - Test Report

**Date**: December 6, 2025
**Tester**: Web QA Agent
**Test Type**: UAT + Playwright Automated Testing
**Environment**: http://localhost:8080 (visualization server)

---

## Executive Summary

**VERDICT**: ❌ **FAIL - Bug Still Present**

The breadcrumb root display bug is **NOT FIXED**. The visualization continues to show:

> **🏠 Root / tests / manual**

Instead of the expected correct path from the actual project root (mcp-vector-search).

---

## 1. Initial Page Load Test

### Test Objective
Verify the breadcrumb displays correctly on initial page load.

### Results
- **Status**: ❌ FAIL
- **Initial Breadcrumb**: Could not find breadcrumb element on initial load
- **After Node Click**: "🏠 Root / tests / manual"

### Evidence
![Initial Load Screenshot](../manual/screenshots/01_initial_load.png)

### Analysis
The breadcrumb element is not immediately visible on page load, but becomes visible after clicking a node. When visible, it incorrectly shows "tests/manual" as the root level.

---

## 2. Node Navigation Test

### Test Objective
Verify breadcrumb updates correctly when clicking different nodes in the graph.

### Results
- **Status**: ⚠️ PARTIAL PASS
- **Nodes Tested**: 2 visible nodes
- **Breadcrumb Structure**: Maintains "🏠 Root" at the beginning ✅
- **Path Correctness**: Incorrect - shows "tests/manual" ❌

### Detailed Results

| Node Clicked | Breadcrumb Displayed | Parts | Root Maintained? |
|--------------|---------------------|-------|------------------|
| "+" | 🏠 Root / tests / manual | ["🏠 Root", "tests", "manual"] | ✅ Yes |
| "manual" | 🏠 Root / tests / manual | ["🏠 Root", "tests", "manual"] | ✅ Yes |

### Evidence
- [Node Click 1 Screenshot](../manual/screenshots/02_node_click_1.png)
- [Node Click 2 Screenshot](../manual/screenshots/02_node_click_2.png)

### Analysis
While the breadcrumb correctly maintains "🏠 Root" at the beginning, the paths displayed are incorrect. The breadcrumb shows "tests/manual" as if it's the complete path from the project root, when it should show the actual file paths relative to mcp-vector-search.

---

## 3. Root Button Functionality Test

### Test Objective
Verify clicking the "🏠 Root" button resets the view.

### Results
- **Status**: ✅ PASS
- **Button Clickable**: Yes
- **View Reset**: Yes
- **Breadcrumb After Click**: "🏠 Root / tests / manual"

### Evidence
![Root Button Click Screenshot](../manual/screenshots/03_root_button_click.png)

### Analysis
The Root button is functional and clickable, but the view still shows the incorrect "tests/manual" path even after reset.

---

## 4. Console Errors

### Test Objective
Monitor browser console for JavaScript errors during testing.

### Results
- **Status**: ❌ FAIL - Critical Error Detected
- **Errors Found**: 1

### Error Details

```
Type: page_error
Message: Maximum call stack size exceeded
Location: null
```

### Analysis
A stack overflow error occurred during node interaction, indicating a potential infinite recursion or circular reference in the JavaScript code.

---

## 5. Root Cause Analysis

### Investigation Summary

After comprehensive testing and code analysis, the root cause has been identified:

#### The Problem

1. **Monorepo Detection False Positive**: The monorepo detector (`MonorepoDetector`) is incorrectly identifying `tests/manual` as a "subproject" because it contains a `package.json` file.

2. **File**: `/Users/masa/Projects/mcp-vector-search/tests/manual/package.json`
   ```json
   {
     "name": "manual",
     "version": "1.0.0",
     "description": "",
     "main": "inspect_visualization_controls.js",
     ...
   }
   ```

3. **Subproject Node Creation**: In `graph_builder.py` (lines 197-214), when subprojects are detected, they are created as **depth-0 nodes**:
   ```python
   if subprojects:
       for sp_name, sp_data in subprojects.items():
           node = {
               "id": f"subproject_{sp_name}",
               "name": sp_name,
               "type": "subproject",
               "file_path": sp_data["path"] or "",  # "tests/manual"
               "depth": 0,  # ← THIS IS THE PROBLEM
               ...
           }
   ```

4. **Graph Data Evidence**: The generated `chunk-graph.json` shows:
   ```json
   {
     "id": "subproject_manual",
     "name": "manual",
     "type": "subproject",
     "file_path": "tests/manual",
     "depth": 0  // ← Treated as root level
   }
   ```

5. **Breadcrumb Generation**: The JavaScript breadcrumb generation is actually **CORRECT**. It's using `node.file_path` directly, which contains "tests/manual". The JavaScript code was already fixed and is working as intended.

### Why the JavaScript Fix Didn't Work

The JavaScript fix applied earlier removed logic that tried to strip 'mcp-vector-search' from paths. However, this fix cannot solve the problem because:

1. The **data itself** is incorrect - `tests/manual` is marked as a depth-0 subproject
2. The visualization has no way to know that "tests/manual" should be prefixed with the actual project root path
3. The breadcrumb is faithfully displaying what the data tells it - a root-level subproject named "manual"

### The Real Issue

This is **NOT** a visualization or breadcrumb bug. This is a **data generation bug** in the indexing/graph building pipeline:

- **Location**: `src/mcp_vector_search/utils/monorepo.py` - Monorepo detection logic
- **Location**: `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` - Subproject node creation
- **Cause**: Presence of `package.json` in `tests/manual/` triggers false positive monorepo detection
- **Effect**: `tests/manual` is treated as a top-level subproject, making all paths relative to it

---

## 6. Recommendations

### Immediate Fix Options

#### Option A: Remove package.json from tests/manual (Quick Fix)
- **Action**: Delete `/Users/masa/Projects/mcp-vector-search/tests/manual/package.json`
- **Pros**: Immediate fix, prevents false positive
- **Cons**: May break manual test scripts that depend on it

#### Option B: Improve Monorepo Detection Logic (Proper Fix)
- **Action**: Update `MonorepoDetector` to exclude test directories
- **File**: `src/mcp_vector_search/utils/monorepo.py`
- **Logic**: Add exclusion rules for:
  - `tests/**`
  - `test/**`
  - `examples/**`
  - Other non-subproject directories with package.json

#### Option C: Adjust Subproject Depth Calculation (Alternative Fix)
- **Action**: Calculate proper depth for subproject nodes based on their path
- **File**: `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`
- **Logic**:
  ```python
  "depth": len(Path(sp_data["path"]).parts)  # tests/manual -> depth 2
  ```

### Recommended Approach

**Implement Option B** (Improved Monorepo Detection) as the proper long-term solution:

1. Update `MonorepoDetector.detect_subprojects()` to filter out test directories
2. Add configuration option for excluded patterns
3. Re-index the project to regenerate graph data
4. Re-test visualization

---

## 7. Test Artifacts

### Generated Files

- **Test Script**: `/Users/masa/Projects/mcp-vector-search/tests/manual/test_breadcrumb_fix.py`
- **Test Results**: `/Users/masa/Projects/mcp-vector-search/tests/manual/breadcrumb_test_results.json`
- **Screenshots**: `/Users/masa/Projects/mcp-vector-search/tests/manual/screenshots/`
  - `01_initial_load.png` (96 KB)
  - `02_node_click_1.png` (101 KB)
  - `02_node_click_2.png` (101 KB)
  - `03_root_button_click.png` (101 KB)

### Test Data

```json
{
  "initial_breadcrumb": "ERROR: Timeout waiting for breadcrumb element",
  "node_navigation_results": [
    {
      "node_label": "+",
      "breadcrumb": "🏠 Root / tests / manual",
      "breadcrumb_parts": ["🏠 Root", "tests", "manual"]
    },
    {
      "node_label": "manual",
      "breadcrumb": "🏠 Root / tests / manual",
      "breadcrumb_parts": ["🏠 Root", "tests", "manual"]
    }
  ],
  "root_button_works": true,
  "console_errors": [
    {
      "type": "page_error",
      "text": "Maximum call stack size exceeded"
    }
  ],
  "passed": false
}
```

---

## 8. Success Criteria (Not Met)

| Criteria | Status | Notes |
|----------|--------|-------|
| Page loads without errors | ❌ FAIL | Stack overflow error detected |
| Breadcrumb shows "🏠 Root" correctly | ⚠️ PARTIAL | Shows Root but wrong path |
| Clicking nodes updates breadcrumb | ✅ PASS | Breadcrumb updates correctly |
| No subdirectory as root level | ❌ FAIL | tests/manual treated as root |
| Breadcrumb paths are correct | ❌ FAIL | Missing project context |
| "🏠 Root" button resets view | ✅ PASS | Button works correctly |

**Overall**: 2/6 criteria met - **TEST FAILED**

---

## 9. Next Steps

1. **Choose Fix Strategy**: Decide between Option A (quick) or Option B (proper)
2. **Implement Fix**: Apply chosen fix to codebase
3. **Re-index Project**: Run `mcp-vector-search index` to regenerate data
4. **Re-run Visualization**: Run `mcp-vector-search visualize` to regenerate graph
5. **Re-test**: Run `uv run python tests/manual/test_breadcrumb_fix.py` again
6. **Verify**: Confirm breadcrumb now shows actual project structure

---

## 10. Technical Details

### Test Environment
- **Browser**: Chromium (Playwright)
- **Viewport**: 1920x1080
- **Server**: HTTP localhost:8080
- **Visualization Mode**: `--code-only`
- **Graph Data**: Loaded from `/chunk-graph.json`

### Test Automation
- **Framework**: Playwright for Python
- **Test Script**: Comprehensive automated test with screenshots
- **Execution Time**: ~15 seconds
- **Exit Code**: 1 (failure)

---

**Report Generated**: December 6, 2025
**Test Execution**: Automated with Playwright
**Report Location**: `/docs/summaries/BREADCRUMB_BUG_TEST_REPORT.md`
