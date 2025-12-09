# Text Label Color Fix - December 9, 2025

## Problem
Text labels in D3 tree visualization were hard to read on dark background:
- Regular node labels: Black (`#000`) - invisible on dark background
- Chunk labels: Dark purple (`#7d3c98`) - difficult to read on dark background
- Background: Dark (`#0d1117`)

## Solution
Updated text label colors to be light colored for visibility:

### Changes Made

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

1. **Linear Layout Labels** (Line 270):
   - Regular nodes: `#000` → `#adbac7` (light gray, matches theme)
   - Chunk nodes: `#7d3c98` → `#bb86fc` (lighter purple, more visible)

2. **Circular Layout Labels** (Line 367):
   - Regular nodes: `#000` → `#adbac7` (light gray, matches theme)
   - Chunk nodes: `#7d3c98` → `#bb86fc` (lighter purple, more visible)

3. **Chunk Content Panel** (Lines 413-419):
   - Title: Added `color: #c9d1d9` (light gray)
   - Metadata text: `#666` → `#8b949e` (lighter gray)
   - Labels (Type, Lines): Added `color: #c9d1d9` (light gray for emphasis)
   - Code background: `#f5f5f5` → `#0d1117` with `border: 1px solid #30363d` (dark theme)
   - Code text: Added `color: #c9d1d9` (light gray)

### Color Palette
- Background: `#0d1117` (dark gray)
- Body text: `#c9d1d9` (light gray)
- Connector lines: `#ccc` (light gray)
- **Regular node labels**: `#adbac7` (light gray - NEW)
- **Chunk node labels**: `#bb86fc` (light purple - NEW)

## Design Decision
- Regular node labels use light gray (`#adbac7`) for consistency with overall theme
- Chunk labels use lighter purple (`#bb86fc`) to maintain visual distinction while being readable
- Colors chosen to complement dark background and match connector line styling

## Testing
To verify:
1. Run `mcp-vector-search visualize`
2. Check that all text labels are clearly visible on dark background
3. Verify both linear and circular layouts show readable text
4. Confirm chunk labels maintain purple color identity but are visible

## Net LOC Impact
- **Lines Changed**: 9 (two `.style('fill', ...)` statements + chunk panel inline styles)
- **Net Impact**: 0 (pure replacement, no additions)

## Related Files
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py` - CSS theme colors
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` - D3 visualization logic
