# Manual Test Plan: Null Check Validation

## Purpose
Verify that all null checks prevent JavaScript crashes when data is undefined or invalid.

## Test Setup
```bash
# Start the visualization server
cd /Users/masa/Projects/mcp-vector-search
mcp-vector-search visualize
```

## Test Cases

### Test 1: Normal Operation (Valid Data)
**Objective**: Verify fix doesn't break normal operation

**Steps**:
1. Navigate to http://localhost:8000
2. Wait for graph to load
3. Click on nodes to expand
4. Verify no console errors

**Expected**:
- ✅ Graph loads successfully
- ✅ Nodes are visible and interactive
- ✅ No JavaScript errors in console
- ✅ Stats panel shows node/link counts

---

### Test 2: Simulate Invalid Data (Empty Object)
**Objective**: Verify graceful handling of empty data

**Steps**:
1. Open browser DevTools (Console tab)
2. Before page loads, add breakpoint in `loadGraphDataStreaming()` at line ~2200
3. When breakpoint hits, modify return to `return {}`
4. Resume execution

**Alternative (Easier)**:
1. Temporarily modify `server.py` to return `{"nodes": [], "links": []}`
2. Reload page

**Expected**:
- ✅ Error message: "Invalid graph data: missing nodes or links"
- ✅ Or: "Failed to initialize graph" with Retry button
- ✅ No JavaScript crash
- ✅ Console shows helpful error log

---

### Test 3: Simulate Network Error
**Objective**: Verify handling of fetch failures

**Steps**:
1. Open browser DevTools (Network tab)
2. Set throttling to "Offline"
3. Reload page

**Expected**:
- ✅ Loading timeout message appears
- ✅ Error caught by .catch() handler
- ✅ Retry button available
- ✅ No uncaught promise rejection

---

### Test 4: Simulate Partial Data (Missing Nodes)
**Objective**: Verify validation catches missing nodes

**Steps**:
1. Modify server to return `{"links": [], "metadata": {}}`
2. Reload page

**Expected**:
- ✅ Validation catches missing nodes
- ✅ Error: "Invalid graph data: missing nodes or links"
- ✅ User sees error message
- ✅ No attempt to access data.nodes

---

### Test 5: Simulate Partial Data (Missing Links)
**Objective**: Verify validation catches missing links

**Steps**:
1. Modify server to return `{"nodes": [], "metadata": {}}`
2. Reload page

**Expected**:
- ✅ Validation catches missing links
- ✅ Error: "Invalid graph data: missing nodes or links"
- ✅ User sees error message
- ✅ No attempt to access data.links

---

### Test 6: Console Error Logging
**Objective**: Verify helpful debug information is logged

**Steps**:
1. Trigger any invalid data scenario
2. Check browser console

**Expected**:
- ✅ Console shows: "[Init V2] Invalid data received: {data}"
- ✅ Or: "[visualizeGraph] Invalid data received: {data}"
- ✅ Error includes actual data value for debugging
- ✅ Stack trace available if error thrown

---

## Verification Checklist

### Code Coverage
- [x] `visualizeGraph()` - Line 339 null check
- [x] `initializeVisualizationV2()` - Line 3306 null check
- [x] Promise chain - Line 2241 validation
- [x] `updateStats()` - Line 1055 null check (already existed)

### Error Paths
- [x] Undefined data → Error message
- [x] Null data → Error message
- [x] Missing nodes → Error message
- [x] Missing links → Error message
- [x] Empty object → Error message

### User Experience
- [x] Clear error messages (not JavaScript errors)
- [x] Retry button available
- [x] Loading state handled properly
- [x] No white screen of death

### Developer Experience
- [x] Console logs for debugging
- [x] Error includes actual data value
- [x] Function name in error message
- [x] Helpful error text

---

## Automated Test (Future)

```javascript
// tests/visualization/test-null-checks.test.js
describe('Null Check Protection', () => {
  test('visualizeGraph rejects undefined data', () => {
    const consoleSpy = jest.spyOn(console, 'error');
    visualizeGraph(undefined);
    expect(consoleSpy).toHaveBeenCalledWith(
      '[visualizeGraph] Invalid data received:',
      undefined
    );
  });

  test('initializeVisualizationV2 shows error UI for null data', () => {
    const loadingEl = document.getElementById('loading');
    initializeVisualizationV2(null);
    expect(loadingEl.innerHTML).toContain('Failed to initialize graph');
    expect(loadingEl.style.display).toBe('block');
  });

  test('promise chain throws on invalid data', async () => {
    const invalidData = {};
    await expect(
      Promise.resolve(invalidData).then(data => {
        if (!data || !data.nodes || !data.links) {
          throw new Error('Invalid graph data: missing nodes or links');
        }
      })
    ).rejects.toThrow('Invalid graph data');
  });
});
```

---

## Success Criteria

### Must Have
1. ✅ No "Cannot read properties of undefined" errors
2. ✅ User sees helpful error messages
3. ✅ Retry button works
4. ✅ Normal operation unaffected

### Nice to Have
1. ⚠️ Specific error messages for different failure modes
2. ⚠️ Auto-retry with exponential backoff
3. ⚠️ Partial rendering if some data is valid

---

## Rollback Plan

If issues are found:
```bash
git checkout HEAD -- src/mcp_vector_search/cli/commands/visualize/templates/scripts.py
```

Or revert specific changes:
- Remove null checks added at lines 339, 2241, 3306
- Keep existing updateStats check at line 1055

---

**Test Date**: 2025-12-09
**Tester**: [Name]
**Result**: [PASS/FAIL]
**Notes**: [Any observations]
