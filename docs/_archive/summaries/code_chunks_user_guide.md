# Code Chunks Navigation - User Guide

## What is the Code Chunks Section?

When you click on a file in the visualization, a detail pane opens on the right side showing information about that file. The **Code Chunks** section is a new feature that displays a clickable list of all functions, classes, and methods in that file, making it easy to navigate to specific code sections.

## Visual Example

```
┌─────────────────────────────────────────────────────────┐
│ 🏠 Root / src / mcp_vector_search / core               │ ← Breadcrumbs
│ database.py                                             │ ← File name
│ file                                                    │ ← Type
├─────────────────────────────────────────────────────────┤
│ CODE CHUNKS (5)                                         │ ← Section header
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ⚡ __init__           L12-25     function           │ │ ← Clickable chunk
│ │ ⚡ initialize         L27-45     function           │ │
│ │ 📦 ChromaVectorDB    L48-250    class              │ │
│ │ 🔧 get_all_chunks    L75-120    method             │ │
│ │ 🔧 search            L145-200   method             │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Contains 5 code chunks                                  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ from pathlib import Path                            │ │
│ │ from typing import List, Optional                   │ │
│ │                                                     │ │
│ │ class ChromaVectorDB:                               │ │
│ │     """Vector database using ChromaDB."""           │ │
│ │     ...                                             │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Language: python                                        │ ← Footer
│ File: src/mcp_vector_search/core/database.py          │
└─────────────────────────────────────────────────────────┘
```

## How to Use

### Step 1: Open the Visualization
```bash
mcp-vector-search visualize serve
```

This opens the visualization in your browser.

### Step 2: Click a File Node
- Navigate the graph using mouse drag or zoom
- Click on any **file icon** (📄) in the graph
- The detail pane will slide in from the right

### Step 3: Navigate to Code Chunks
- Look for the **"CODE CHUNKS"** section at the top of the detail pane
- You'll see a list of all functions, classes, and methods in that file
- Each item shows:
  - **Icon** (⚡ function, 📦 class, 🔧 method)
  - **Name** of the chunk
  - **Line range** (e.g., L12-25)
  - **Type badge** (color-coded)

### Step 4: Click a Chunk
- Click any chunk in the list
- The graph will:
  1. Highlight that chunk node
  2. Center the view on it
  3. Show the chunk's details in the detail pane

## Visual Design

### Icons
- ⚡ **Function** - Lightning bolt (orange/gold badge)
- 📦 **Class** - Package box (blue badge)
- 🔧 **Method** - Wrench/tool (purple badge)
- 📄 **Code** - Document (gray badge)

### Color Coding
Badges use colors matching the graph nodes:
- **Function**: Orange/gold (#d29922)
- **Class**: Blue (#1f6feb)
- **Method**: Purple (#8957e5)
- **Code**: Gray (#6e7681)

### Hover Effects
When you hover over a chunk:
- Background changes from dark to lighter gray
- Border color changes to blue
- Subtle shadow appears
- Cursor changes to pointer

## Keyboard Shortcuts

Currently, navigation is mouse-only. Use these techniques:
- **Click** chunk to navigate
- **Scroll** the detail pane to see all chunks
- **Close** detail pane with × button

## Tips & Tricks

### 1. Quickly Find Functions
The list is sorted by line number, so functions appear in the order they're defined in the file.

### 2. See Code Context
After clicking a chunk, the detail pane shows:
- Full code of that function/class
- Documentation
- Line numbers
- Related information

### 3. Navigate Between Related Code
Use the graph edges (lines) to see:
- Which functions call this one
- What this function depends on
- Semantic similarity to other code

### 4. Empty Files
If a file has no code chunks (comments only, empty file, etc.), the Code Chunks section won't appear.

## Troubleshooting

### "No code chunks found"
This means:
- The file is empty
- The file only contains comments/imports
- The indexer hasn't parsed the file yet
- Solution: Re-run `mcp-vector-search index`

### Chunks section not visible
Make sure:
1. You clicked a **file node** (not a directory or function)
2. The file has been indexed
3. The visualization was regenerated after updating the code

### Click doesn't navigate
If clicking a chunk does nothing:
1. Check browser console for errors (F12)
2. Ensure the chunk exists in the graph
3. Try refreshing the page

## Comparison: Before vs. After

### Before (Without Code Chunks)
```
1. Click file → See breadcrumbs and full content
2. Scroll through content to find function
3. Manually search for function name
4. No quick navigation
```

### After (With Code Chunks)
```
1. Click file → See breadcrumbs, chunks list, and content
2. Scan chunk list (sorted, with line numbers)
3. Click chunk → Instantly navigate to it
4. See highlighted node in graph
```

## Advanced Usage

### Finding Dead Code
Chunks with no incoming edges (not called by anything) appear with a **red border** in the graph. Check the Code Chunks section to see which functions might be unused.

### Exploring Class Structure
For classes, you'll see:
- 📦 The class itself
- 🔧 All methods within it (sorted by line number)

This gives you a quick overview of class structure.

### Comparing Similar Files
Open two files side-by-side (in separate browser tabs) and compare their Code Chunks sections to see structural differences.

## What's Next?

Future enhancements might include:
- Search/filter chunks by name
- Group by type (all functions, all classes)
- Complexity indicators
- Keyboard navigation

---

**Questions or Issues?**
See the main documentation or file an issue on GitHub.
