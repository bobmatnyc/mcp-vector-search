# Index Command KG Stats Display - Implementation Summary

## Changes Made

### 1. Added `print_kg_stats()` function to `output.py`
**File**: `src/mcp_vector_search/cli/output.py`

- New function to display Knowledge Graph statistics in a formatted table
- Shows entity counts (code entities, doc sections)
- Shows relationship counts (calls, imports, inherits, contains)
- Uses Rich Table formatting with green values for visual consistency

### 2. Updated `_run_batch_indexing()` in `index.py`
**File**: `src/mcp_vector_search/cli/commands/index.py`

#### Added KG Stats Display Logic (after line 952)
- Checks if KG database exists
- Loads KG and fetches stats if available
- Displays stats table if entities > 0
- Shows hint to build KG if not yet built
- Properly closes KG connection

#### Updated "Next Steps" Section (around line 997)
- Conditionally shows KG build command if KG not built
- Shows KG query commands if KG already built

## Expected Output

### When KG is Built
```
✓ Processed 17,025 files (109,137 searchable chunks created)

                Index Statistics
┌────────────────┬──────────────────────────────────┐
│ Total Files    │ 17,025                           │
│ Indexed Files  │ 17,025                           │
│ Total Chunks   │ 109,137                          │
│ Languages      │ java: 85691, javascript: 6684    │
└────────────────┴──────────────────────────────────┘

         Knowledge Graph Statistics
┌──────────────────────┬─────────────────────────────┐
│ Total Entities       │ 2,341                       │
│   Code Entities      │ 2,205                       │
│   Doc Sections       │ 136                         │
│ Total Relationships  │ 7,087                       │
│   Calls              │ 3,421                       │
│   Imports            │ 1,892                       │
│   Inherits           │ 359                         │
│   Contains           │ 1,250                       │
└──────────────────────┴─────────────────────────────┘

💡 Ready to Search
  mcp-vector-search search 'your query' - Try semantic search
  mcp-vector-search kg stats - View graph statistics
  mcp-vector-search kg query "ClassName" - Find related entities
```

### When KG is NOT Built
```
✓ Processed 17,025 files (109,137 searchable chunks created)

                Index Statistics
┌────────────────┬──────────────────────────────────┐
│ Total Files    │ 17,025                           │
│ Indexed Files  │ 17,025                           │
│ Total Chunks   │ 109,137                          │
│ Languages      │ java: 85691, javascript: 6684    │
└────────────────┴──────────────────────────────────┘

💡 Run 'mcp-vector-search kg build' to enable graph queries

💡 Ready to Search
  mcp-vector-search search 'your query' - Try semantic search
  mcp-vector-search kg build - Build knowledge graph for advanced queries
```

## Implementation Details

### Error Handling
- Wrapped KG loading in try/except to gracefully handle cases where KG is not available
- Uses `logger.debug()` for errors to avoid cluttering user output
- Falls back to showing hint if KG stats cannot be loaded

### Performance Considerations
- Only loads KG if directory exists (cheap filesystem check)
- Only initializes KG connection if entities > 0
- Properly closes KG connection after reading stats

### User Experience
- Clear visual separation with blank lines
- Consistent table formatting between index stats and KG stats
- Progressive disclosure: shows KG build hint when not built, shows query commands when built
- Uses emoji (💡) for hints to draw attention

## Testing
- Syntax validated with `python -m py_compile`
- Logic tested with mock data showing correct formatting
- Import structure verified

## Files Modified
1. `src/mcp_vector_search/cli/output.py` - Added `print_kg_stats()` function
2. `src/mcp_vector_search/cli/commands/index.py` - Added KG stats display and conditional next steps
