# Dead Code Analyzer: Large Function Chunking Investigation

**Date**: 2026-02-18
**Investigator**: Research Agent
**Status**: ✅ ROOT CAUSE IDENTIFIED (Different from Hypothesis)

---

## Executive Summary

**Hypothesis (DISPROVEN)**: Large functions like `run_analysis` (410 lines) get split into multiple chunks, causing the dead code analyzer to miss function calls in continuation chunks that lack `function_name` metadata.

**Actual Root Cause (CONFIRMED)**: The entry point detector incorrectly identifies `analyze_app` (a `typer.Typer()` object) as an entry point instead of `analyze_callback` (the actual decorated function). Since `analyze_app` is not a function, it has zero outgoing calls in the call graph, making all code unreachable.

---

## Investigation Methodology

### 1. Hypothesis Testing: Do Large Functions Get Split?

**Test**: Parse a 414-line function similar to `run_analysis` to verify chunking behavior.

```python
# Test code with 414-line function
def run_analysis():
    """A large function to test chunking."""
    line_3 = "line 3"
    # ... 410 lines of code ...
    _print_trends(data)  # Call at line 413
    return result
```

**Result**:
```
Number of chunks created: 1
Total lines in function: 414

Chunk 1:
  Type: function
  Function name: run_analysis
  Lines: 1-415 (415 lines)
  Has '_print_trends' in content: True
```

**Conclusion**: ❌ **Hypothesis DISPROVEN**. The Python parser creates a **single chunk** for the entire function regardless of size. There is no splitting at all.

---

### 2. Database State Verification

**Query**: Check if `_print_trends` exists in the LanceDB database.

**Results**:
```
Total chunks in database: 40,548
Chunks with function_name='_print_trends': 3

Function: _print_trends
  File: /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/analyze.py
  Lines: 982-1070
  Type: function
```

**Conclusion**: ✅ `_print_trends` **exists** in the database with `function_name` properly set.

---

### 3. Call Graph Building Verification

**Query**: Check if `run_analysis` calls `_print_trends` in the call graph.

**Results**:
```
Call graph entries: 24

run_analysis calls 58 functions
✅ _print_trends IS in run_analysis's callees!

_print_trends is a node in call graph (calls 9 functions)
```

**Conclusion**: ✅ The call graph **correctly captures** the `run_analysis` → `_print_trends` call.

---

### 4. Entry Point Detection Analysis

**Query**: Check which entry points are detected in `analyze.py`.

**File Structure**:
```python
# src/mcp_vector_search/cli/commands/analyze.py

analyze_app = typer.Typer(help="📈 Analyze code complexity and quality")

@analyze_app.callback(invoke_without_command=True)
def analyze_callback(ctx: typer.Context, quick: bool = False) -> None:
    """Analyze code complexity and quality."""
    # ...
    run_analysis(...)  # Calls run_analysis
```

**Entry Points Detected**:
```
Entry points detected in analyze.py: 1
❌ run_analysis is NOT an entry point
Sample entry points: ['analyze_app']
```

**Problem Identified**: The entry point detector identifies **`analyze_app`** (a `typer.Typer()` object) as an entry point, but:
- `analyze_app` is **NOT a function** - it's a class instance
- It has **zero outgoing calls** in the call graph
- The actual entry point should be **`analyze_callback`** (the decorated function)

---

### 5. Reachability Simulation

**Test**: Simulate reachability analysis starting from different entry points.

**Results**:
```
Starting from 'analyze_app':
  Reachable functions: 1
  '_print_trends' reachable: False
  'run_analysis' reachable: False

Starting from 'analyze_callback':
  Reachable functions: 106
  '_print_trends' reachable: True
  'run_analysis' reachable: True
```

**Conclusion**: Starting from the wrong entry point (`analyze_app`) makes **all code unreachable**, while starting from the correct entry point (`analyze_callback`) correctly identifies reachable code.

---

## Root Cause: Entry Point Detection Bug

### The Problem

**File**: `src/mcp_vector_search/analysis/entry_points.py`

The `EntryPointDetector` identifies Typer CLI patterns but detects the **variable assignment** instead of the **decorated function**:

```python
# DETECTED AS ENTRY POINT (WRONG):
analyze_app = typer.Typer(help="...")  # ❌ This is a Typer object, not a function

# SHOULD BE DETECTED AS ENTRY POINT (CORRECT):
@analyze_app.callback(invoke_without_command=True)
def analyze_callback(...):  # ✅ This is the actual entry point function
    run_analysis(...)
```

### Why This Causes False Positives

1. **Entry point**: `analyze_app` (Typer object)
2. **Call graph**: `analyze_app` has **zero outgoing calls** (not a function)
3. **Reachability**: Nothing is reachable from `analyze_app`
4. **Result**: All functions appear dead, including `_print_trends`

---

## Evidence Summary

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Large functions split? | Maybe (hypothesis) | No, single chunk | ❌ Hypothesis wrong |
| `_print_trends` in database? | Yes | Yes (3 copies) | ✅ Correct |
| `_print_trends` has function_name? | Yes | Yes | ✅ Correct |
| `run_analysis` calls `_print_trends`? | Yes | Yes | ✅ Correct |
| Call graph captures call? | Yes | Yes | ✅ Correct |
| Entry point is `analyze_callback`? | Yes | No, detects `analyze_app` | ❌ **BUG HERE** |
| Reachability from `analyze_app`? | Should reach all | Reaches nothing | ❌ **BUG HERE** |
| Reachability from `analyze_callback`? | Should reach all | Reaches all | ✅ Would work |

---

## Recommended Fix

### Option 1: Fix Entry Point Detection for Typer Decorators

**Location**: `src/mcp_vector_search/analysis/entry_points.py`

**Current Detection** (simplified):
```python
# Detects Typer app creation
if "typer.Typer" in line:
    # Extracts 'analyze_app' from: analyze_app = typer.Typer(...)
    entry_points.append(EntryPoint(name="analyze_app", ...))  # ❌ Wrong
```

**Proposed Fix**:
```python
# Detect Typer decorator patterns
if re.match(r'@\w+\.(command|callback)', line):
    # Extract function name from next line: def analyze_callback(...)
    next_line = get_next_line()
    func_match = re.match(r'def\s+(\w+)\s*\(', next_line)
    if func_match:
        entry_points.append(EntryPoint(
            name=func_match.group(1),  # ✅ "analyze_callback"
            type=EntryPointType.CLI_COMMAND,
            ...
        ))
```

### Option 2: Detect Both and Link Them

Detect `analyze_app` but follow the decorator chain to find the actual function:
1. Detect `analyze_app = typer.Typer(...)`
2. Detect `@analyze_app.callback()` or `@analyze_app.command()`
3. Extract the decorated function name
4. Use **function name** as entry point, not variable name

---

## Testing the Fix

### Before Fix
```bash
$ mcp-vector-search analyze dead-code --min-confidence high
# Expected: False positives including _print_trends
```

### After Fix
```bash
$ mcp-vector-search analyze dead-code --min-confidence high
# Expected: _print_trends correctly identified as reachable
```

### Verification Script
```python
from pathlib import Path
from mcp_vector_search.analysis.entry_points import EntryPointDetector

detector = EntryPointDetector()
analyze_file = Path("src/mcp_vector_search/cli/commands/analyze.py")

with open(analyze_file) as f:
    code = f.read()

entry_points = detector.detect_from_file(analyze_file, code)

# Check results
print(f"Entry points: {[ep.name for ep in entry_points]}")

# Should include:
# - analyze_callback (main callback)
# - run_analysis (if using @analyze_app.command)
# - run_dead_code_analysis (if using @analyze_app.command)

# Should NOT include:
# - analyze_app (Typer object, not a function)
```

---

## Additional Findings

### Duplicate Chunks

There are **3 copies** of both `_print_trends` and `run_analysis` in the database:

```
_print_trends chunks:
  - Lines 982-1070 (duplicate 1)
  - Lines 982-1070 (duplicate 2)
  - Lines 1065-1153 (different line range)

run_analysis chunks:
  - Lines 353-761 (version 1)
  - Lines 353-761 (duplicate)
  - Lines 412-820 (version 2)
```

This suggests potential issues with:
- Index rebuilding without proper cleanup
- Multiple indexing passes creating duplicates
- Version control causing different snapshots

**Impact**: Minor - the call graph handles duplicates correctly, and reachability analysis still works. However, this should be investigated separately to ensure database integrity.

---

## Conclusion

**Original Hypothesis**: ❌ DISPROVEN
Large functions are **NOT split** into chunks. The Python parser creates a single chunk per function regardless of size.

**Actual Root Cause**: ✅ CONFIRMED
The entry point detector identifies **`analyze_app`** (a Typer object) instead of **`analyze_callback`** (the actual function), causing all code to appear unreachable.

**Fix Required**: Update `EntryPointDetector` to:
1. Detect Typer decorator patterns (`@app.callback()`, `@app.command()`)
2. Extract the **decorated function name** as the entry point
3. Stop detecting Typer object variable names as entry points

**Priority**: HIGH - This bug causes widespread false positives in dead code analysis for any project using Typer CLI.

---

## Related Issues

- **Venv False Positives** (documented in `dead-code-analyzer-venv-false-positives-2026-02-18.md`): Entry point detector scans `.venv-mcp` files, creating additional false positives. This is a separate bug but compounds the impact.

- **Duplicate Chunks**: Database contains duplicate entries for the same functions. This should be investigated to ensure index integrity, though it doesn't affect the current bug.

---

## Appendix: Test Commands

```bash
# Test large function parsing
source .venv/bin/activate
python3 << 'EOF'
# Test code from investigation
EOF

# Check database state
python3 << 'EOF'
from mcp_vector_search.core.lancedb_backend import LanceVectorDatabase
# Query code from investigation
EOF

# Simulate reachability
python3 << 'EOF'
# Reachability simulation from investigation
EOF
```
