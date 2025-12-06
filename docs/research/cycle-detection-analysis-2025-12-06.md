# Circular Dependency Detection Analysis

**Date:** 2025-12-06
**Investigator:** Claude (Research Agent)
**Objective:** Understand why legitimate patterns like "main calling main" (recursion) are being flagged as cycles

---

## Executive Summary

The cycle detection code in `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` uses a three-color DFS algorithm that **correctly identifies true cycles** in the call graph. However, it has a **critical filter at line 134** that is supposed to exclude self-loops but may have edge cases.

**Key Finding:** The algorithm includes logic to filter out self-loops (`if len(set(cycle_nodes)) > 1`), but the issue is likely **not in the detection algorithm itself**, but rather in **how the caller relationships are being built**. The problem stems from the "external callers" relationship mapping (lines 367-412) which is **file-based**, not function-based.

---

## 1. Cycle Detection Implementation

### Location
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/graph_builder.py`
**Function:** `detect_cycles()` (lines 94-160)

### Algorithm: Three-Color DFS

The implementation uses a **three-color marking algorithm** for cycle detection:

```python
def detect_cycles(chunks: list, caller_map: dict) -> list[list[str]]:
    """Detect TRUE cycles in the call graph using DFS with three-color marking.

    Uses three-color marking to distinguish between:
    - WHITE (0): Unvisited node, not yet explored
    - GRAY (1): Currently exploring, node is in the current DFS path
    - BLACK (2): Fully explored, all descendants processed

    A cycle exists when we encounter a GRAY node during traversal, which means
    we've found a back edge to a node currently in the exploration path.
    """
    cycles_found = []
    white, gray, black = 0, 1, 2
    color = {chunk.chunk_id or chunk.id: white for chunk in chunks}
```

**Key Points:**
- **WHITE (0):** Unvisited node
- **GRAY (1):** Currently being explored (in the current DFS path)
- **BLACK (2):** Fully explored (all descendants processed)
- **Cycle Detection:** When a GRAY node is encountered → back edge detected → cycle found

### Self-Loop Filter (Line 134)

```python
if color.get(node_id, white) == gray:
    # Found a TRUE cycle! Node is in current path
    try:
        cycle_start = path.index(node_id)
        cycle_nodes = path[cycle_start:] + [node_id]  # Include back edge
        # Only record if cycle length > 1 (avoid self-loops unless intentional)
        if len(set(cycle_nodes)) > 1:  # ← CRITICAL FILTER
            cycles_found.append(cycle_nodes)
    except ValueError:
        pass  # Node not in path (shouldn't happen)
    return
```

**Analysis:**
- Line 134 checks `if len(set(cycle_nodes)) > 1`
- This should filter out A → A (self-loops) because `set([A, A])` has length 1
- **However:** This filter only works if the cycle is truly `[A, A]`
- If the path contains duplicates elsewhere, this check may not work as intended

---

## 2. Root Cause: External Caller Relationship Building

### The Real Problem (Lines 367-412)

The cycle detection algorithm is working correctly, but the **input data (caller_map)** is built incorrectly for detecting function-level recursion.

```python
# Compute external caller relationships
console.print("[cyan]Computing external caller relationships...[/cyan]")
caller_map = {}  # Map chunk_id -> list of caller info

for chunk in code_chunks:
    chunk_id = chunk.chunk_id or chunk.id
    file_path = str(chunk.file_path)
    function_name = chunk.function_name or chunk.class_name

    if not function_name:
        continue

    # Search for other chunks that reference this function/class name
    for other_chunk in chunks:
        other_file_path = str(other_chunk.file_path)

        # Only track EXTERNAL callers (different file)  ← PROBLEM HERE
        if other_file_path == file_path:
            continue  # ← Skips same-file calls (including recursion)

        # Check if the other chunk's content mentions this function/class
        if function_name in other_chunk.content:
            other_chunk_id = other_chunk.chunk_id or other_chunk.id
            # ... store caller info ...
```

**Critical Issue:**
- **Line 383:** `if other_file_path == file_path: continue`
- This **explicitly excludes same-file calls**
- **Legitimate recursion** (function calling itself in the same file) is **NOT tracked**
- **False positive scenario:** If `main()` in `file_a.py` calls `main()` in `file_b.py`, and vice versa, this creates a cycle

---

## 3. Where Cycles Are Marked (Lines 419-436)

```python
# Mark cycle links
cycle_links = []
if cycles:
    console.print(f"[yellow]⚠ Found {len(cycles)} circular dependencies[/yellow]")

    # For each cycle, create links marking the cycle
    for cycle in cycles:
        # Create links for the cycle path: A → B → C → A
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]  # Wrap around to form cycle
            cycle_links.append(
                {
                    "source": source,
                    "target": target,
                    "type": "caller",
                    "is_cycle": True,  # ← Field set here
                }
            )
```

**Data Structure:**
- Cycles are stored as links with `"is_cycle": True` field
- These links are added to the graph's `links` array (line 547)

---

## 4. Visualization Layer (scripts.py)

### Cycle Display (Lines 406, 459, 852-895)

**Force-Directed Layout (D3.js):**
```javascript
.force("link", d3.forceLink(visibleLinks)
    .id(d => d.id)
    .distance(d => {
        // ...
        if (d.is_cycle) return 80; // Reduced from 120
        // ...
    })
    .strength(d => {
        // ...
        if (d.is_cycle) return 0.4; // Increased from 0.3
        // ...
    })
)
```

**Link Styling (Line 459):**
```javascript
.attr("class", d => {
    // Cycle links have highest priority
    if (d.is_cycle) return "link cycle";
    // ...
})
```

**Tooltip Warning (Lines 882-895):**
```javascript
if (d.is_cycle) {
    tooltip
        .style("display", "block")
        .html(`
            <div style="color: #ff4444;"><strong>⚠️ Circular Dependency Detected</strong></div>
            <div style="margin-top: 8px;">Path: ${sourceName} → ${targetName}</div>
            <div style="margin-top: 8px; ...">
                This indicates a circular call relationship that may lead to
                infinite recursion or tight coupling.
            </div>
        `);
    return;
}
```

---

## 5. False Positive Patterns

### Pattern 1: Same Function Name Across Files

**Scenario:**
```
file_a.py:
    def main():
        from file_b import main as main_b
        main_b()  # Calls file_b.main()

file_b.py:
    def main():
        from file_a import main as main_a
        main_a()  # Calls file_a.main()
```

**Result:**
- Caller map: `{file_a.main: [file_b.main], file_b.main: [file_a.main]}`
- Detected cycle: `[file_a.main, file_b.main, file_a.main]`
- **This IS a legitimate cycle** (mutual recursion across files)

### Pattern 2: True Recursion (Same File)

**Scenario:**
```
utils.py:
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)  # Recursive call
```

**Result:**
- **Line 383 skips this:** `if other_file_path == file_path: continue`
- **NOT tracked in caller_map**
- **NOT detected as cycle** (correctly, since it's intentional recursion)

### Pattern 3: False Positive - Name Collision

**Scenario:**
```
cli.py:
    def main():
        setup()
        run()

server.py:
    def main():
        start_server()
        # No reference to cli.main()
```

**Result:**
- If `server.py` contains the string "main" in a docstring or comment
- Line 387: `if function_name in other_chunk.content:` ← Simple substring match
- **False positive:** Detects "main" mentioned in comments as a "call"

---

## 6. Recommendations for Filtering Legitimate Patterns

### A. Exclude True Self-Loops (Already Implemented)

**Current Filter (Line 134):**
```python
if len(set(cycle_nodes)) > 1:
    cycles_found.append(cycle_nodes)
```

**Status:** ✅ Already filters A → A patterns
**Issue:** May not handle edge cases where path has duplicates

**Improved Filter:**
```python
# Only record if cycle length > 1 (avoid self-loops)
unique_nodes = set(cycle_nodes[:-1])  # Exclude the back edge duplicate
if len(unique_nodes) > 1:
    cycles_found.append(cycle_nodes)
```

### B. Improve Caller Detection (String Matching Issues)

**Current Issue (Line 387):**
```python
if function_name in other_chunk.content:
```

**Problem:** Simple substring matching causes false positives

**Recommended Approach:**
1. **Use AST parsing** to find actual function calls (not comments/strings)
2. **Import resolution** to track which functions are actually imported
3. **Call graph construction** based on AST, not string matching

**Example with AST:**
```python
import ast

def find_function_calls(code: str) -> set[str]:
    """Extract actual function calls from code using AST."""
    try:
        tree = ast.parse(code)
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return calls
    except SyntaxError:
        return set()

# Usage:
actual_calls = find_function_calls(other_chunk.content)
if function_name in actual_calls:
    # This is a real call, not just a mention
```

### C. Track Same-File Relationships Separately

**Recommendation:**
- Keep external caller tracking as-is (cross-file dependencies)
- Add **internal caller tracking** for same-file calls (recursion detection)
- Classify cycles based on file boundaries:
  - **Cross-file cycles:** True architectural issues (flag as warnings)
  - **Same-file cycles:** Usually intentional recursion (suppress or mark differently)

**Implementation:**
```python
external_caller_map = {}  # Cross-file calls (existing)
internal_caller_map = {}  # Same-file calls (new)

for chunk in code_chunks:
    for other_chunk in chunks:
        if other_file_path == file_path:
            # Internal call (same file)
            if chunk_id not in internal_caller_map:
                internal_caller_map[chunk_id] = []
            internal_caller_map[chunk_id].append(...)
        else:
            # External call (different file)
            if chunk_id not in external_caller_map:
                external_caller_map[chunk_id] = []
            external_caller_map[chunk_id].append(...)

# Detect cycles only in external_caller_map
cycles = detect_cycles(chunks, external_caller_map)
```

### D. Add Cycle Classification

**Recommendation:** Classify cycles by type and filter accordingly

```python
def classify_cycle(cycle: list[str], chunks: list) -> str:
    """Classify a cycle as self-loop, same-file, or cross-file."""
    if len(set(cycle[:-1])) == 1:
        return "self-loop"  # A → A

    chunk_map = {c.chunk_id or c.id: c for c in chunks}
    files = {chunk_map[node_id].file_path for node_id in cycle[:-1]}

    if len(files) == 1:
        return "same-file"  # A → B → A (same file)
    else:
        return "cross-file"  # A → B → A (different files)

# Usage:
for cycle in cycles:
    cycle_type = classify_cycle(cycle, chunks)

    if cycle_type == "self-loop":
        continue  # Skip, likely recursion
    elif cycle_type == "same-file":
        # Mark as info, not warning
        cycle_links.append({..., "severity": "info"})
    else:
        # Mark as warning (true architectural issue)
        cycle_links.append({..., "severity": "warning"})
```

---

## 7. Test Cases for Validation

### Test Case 1: True Recursion (Should NOT Flag)

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Legitimate recursion
```

**Expected:** No cycle detected (same-file, intentional)

### Test Case 2: Mutual Recursion (Should FLAG)

```python
# file_a.py
def is_even(n: int) -> bool:
    if n == 0:
        return True
    return is_odd(n - 1)

# file_b.py
def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)
```

**Expected:** Cycle detected (cross-file, architectural issue)

### Test Case 3: False Positive - Name Collision (Should NOT Flag)

```python
# cli.py
def main():
    """Main entry point."""
    pass

# server.py
def start():
    """Start the main server process."""  # "main" in docstring
    pass
```

**Expected:** No cycle detected (no actual call relationship)

---

## 8. Impact Assessment

### Current Behavior
- **Self-loops (A → A):** ✅ Correctly filtered by line 134
- **Cross-file cycles:** ✅ Correctly detected (e.g., A.main ↔ B.main)
- **Same-file recursion:** ✅ Correctly ignored (line 383 skip)
- **False positives:** ❌ Possible due to substring matching (line 387)

### Proposed Improvements Priority

**HIGH PRIORITY:**
1. Replace substring matching with AST-based call detection
2. Add cycle classification (self-loop, same-file, cross-file)
3. Filter or downgrade same-file cycles to "info" level

**MEDIUM PRIORITY:**
4. Track internal caller relationships separately
5. Add test cases for recursion patterns

**LOW PRIORITY:**
6. Improve self-loop filter robustness (line 134 edge cases)

---

## 9. Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Cycle Detection | `graph_builder.py` | 94-160 | DFS three-color algorithm |
| Self-Loop Filter | `graph_builder.py` | 134 | Filters A → A patterns |
| Caller Map Building | `graph_builder.py` | 367-412 | Tracks external callers |
| Same-File Skip | `graph_builder.py` | 383 | Excludes same-file calls |
| Substring Matching | `graph_builder.py` | 387 | Detects function mentions |
| Cycle Marking | `graph_builder.py` | 419-436 | Creates cycle links |
| Visualization | `scripts.py` | 406, 459, 852-895 | Renders cycles in UI |

---

## 10. Conclusion

The cycle detection algorithm is **theoretically sound** and uses industry-standard three-color DFS. The issue is **not in the detection logic**, but rather in:

1. **Input Data Quality:** Substring matching (line 387) causes false positives
2. **Classification Gap:** No distinction between cross-file cycles (bad) and same-file recursion (expected)
3. **Visualization Context:** All cycles shown with same severity (warning red)

**Recommended Next Steps:**
1. Implement AST-based call detection to replace substring matching
2. Add cycle classification to distinguish architectural issues from recursion
3. Update visualization to show different severity levels (error/warning/info)
4. Add comprehensive test suite for recursion patterns

---

## Appendix: Algorithm Complexity

- **Time Complexity:** O(V + E) where V = nodes (functions), E = edges (calls)
- **Space Complexity:** O(V) for color map and recursion stack
- **Correctness:** Proven algorithm for directed graph cycle detection
- **False Positive Rate:** High due to string matching, low with AST parsing

---

**Research completed:** 2025-12-06
**Files analyzed:** 2 core files, ~2500 lines of code
**Key insight:** Algorithm is correct, input data needs refinement
