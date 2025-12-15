# Visualization Report Buttons Analysis

**Date:** 2025-12-15
**Researcher:** Research Agent
**Project:** MCP Vector Search - Visualization Feature
**Context:** Investigation of report generation buttons in visualization UI

## Executive Summary

The visualization feature currently has **4 report buttons** in the sidebar:
1. **Complexity Report** (📊) - FULLY IMPLEMENTED
2. **Code Smells Report** (🔍) - FULLY IMPLEMENTED
3. **Dependencies Report** (🔗) - FULLY IMPLEMENTED
4. **Trends Report** (📈) - FULLY IMPLEMENTED

**User Expectation:** A single "Complexity & Smell Report" button that combines complexity hotspots with code smell detection to show recommended files/classes to improve.

**Current Gap:** No combined report exists. The complexity and code smell reports are separate and require clicking two different buttons.

## Current Implementation Details

### File Location
- **Primary file:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
- **Size:** 4,404 lines
- **Recent changes:** Modified filter link visibility (per git status)

### Report Buttons (base.py, lines 140-160)

```html
<h3>📋 Reports</h3>
<div class="legend" style="margin-top: 8px;">
    <div class="legend-category" style="border-bottom: none;">
        <div class="legend-item report-btn" onclick="showComplexityReport()">
            <span class="report-icon">📊</span>
            <span>Complexity</span>
        </div>
        <div class="legend-item report-btn" onclick="showCodeSmells()">
            <span class="report-icon">🔍</span>
            <span>Code Smells</span>
        </div>
        <div class="legend-item report-btn" onclick="showDependencies()">
            <span class="report-icon">🔗</span>
            <span>Dependencies</span>
        </div>
        <div class="legend-item report-btn" onclick="showTrends()">
            <span class="report-icon">📈</span>
            <span>Trends</span>
        </div>
    </div>
</div>
```

### 1. Complexity Report (showComplexityReport - Line 3082)

**Function:** `showComplexityReport()` (lines 3082-3234)

**Key Features:**
- Collects all code chunks with complexity data
- Calculates statistics: total functions, average complexity, grade distribution
- Uses complexity grading system: A (≤5), B (≤10), C (≤15), D (≤20), F (>20)
- Displays complexity hotspots table sorted by highest complexity first

**Data Structure:**
```javascript
chunksWithComplexity = [
    {
        name: "function_name",
        type: "function|method|class",
        file_path: "/path/to/file.py",
        start_line: 10,
        end_line: 50,
        complexity: 15.2,
        grade: "C",
        node: <original_node_reference>
    }
]
```

**Output Sections:**
1. **Summary Stats:** Total functions, average complexity, functions with data
2. **Grade Distribution:** Visual bars showing A-F grade counts
3. **Hotspots Table:** All functions sorted by complexity (columns: Name, File, Lines, Complexity, Grade)

**Helper Functions:**
- `getComplexityGrade(complexity)` - Converts complexity score to A-F grade (line 3061)
- `getGradeColor(grade)` - Returns color for each grade (line 3070)
- `navigateToChunk(chunkName)` - Navigates to code chunk in visualization (line 3236)

### 2. Code Smells Report (showCodeSmells - Line 3380)

**Function:** `showCodeSmells()` (lines 3380-3511)

**Key Features:**
- Detects 4 types of code smells via `detectCodeSmells()` function
- Counts by type and severity (warning vs error)
- Provides interactive filtering by smell type
- Displays smells table with clickable rows to navigate to code

**Smell Types Detected:**
1. **Long Method:** Lines > 100 (warning), > 200 (error)
2. **High Complexity:** Complexity > 15 (warning), > 20 (error)
3. **Deep Nesting:** Max nesting > 4 (warning), > 6 (error)
4. **God Class:** Methods > 30 or lines > 800 (error)

**Data Structure:**
```javascript
allSmells = [
    {
        type: "Long Method|High Complexity|Deep Nesting|God Class",
        severity: "warning|error",
        node: <node_reference>,
        details: "Descriptive message (e.g., '250 lines')"
    }
]
```

**Output Sections:**
1. **Summary Cards:** Total smells, warnings, errors
2. **Filter Checkboxes:** Filter by smell type
3. **Smells Table:** All detected smells (columns: Type, Severity, Name, File, Details)

**Helper Functions:**
- `detectCodeSmells(nodes)` - Analyzes tree and returns array of smell objects (line 3282)
- `filterCodeSmells()` - Client-side filtering based on checkboxes (line 3513)

### 3. Dependencies Report (showDependencies - Line 3532)

**Function:** `showDependencies()` (lines 3532-3714)

**Key Features:**
- Builds file-level dependency graph from caller links
- Detects circular dependencies
- Shows most connected files
- Lists dependencies and dependents for each file

**Helper Functions:**
- `buildFileDependencyGraph()` - Constructs file dependency map (line 3715)
- `findCircularDeps(fileDeps)` - Detects circular dependency cycles (line 3747)

### 4. Trends Report (showTrends - Line 3810)

**Function:** `showTrends()` (lines 3810-3992)

**Key Features:**
- Calculates codebase metrics snapshot
- Shows complexity distribution, health score, file/function counts
- Provides baseline for future trend tracking
- Notes that historical trend data not yet implemented

**Helper Functions:**
- `calculateCodebaseMetrics()` - Computes all codebase statistics (line 4223)
- `calculateHealthScore(chunks)` - Generates health score from complexity grades (line 4311)
- `getHealthScoreInfo(score)` - Returns health status info (emoji, label, color) (line 4324)

## What's Missing: Combined Complexity & Smell Report

### User Expectation
A single report that answers: **"Which files/classes should I improve first?"**

This would combine:
- High complexity functions/methods (from Complexity Report)
- Detected code smells (from Code Smells Report)
- Prioritized recommendations based on both metrics

### Proposed Implementation

**Option 1: Add Fifth Button (Recommended)**
- Add new button: "🎯 Improvement Priorities" or "🔧 Fix Recommendations"
- Function: `showImprovementPriorities()`
- Combines data from both reports into actionable recommendations

**Option 2: Replace Complexity Button**
- Change "Complexity" button to "Complexity & Smells"
- Merge both reports into single view with tabs or sections

**Option 3: Add Dropdown/Menu**
- Convert report buttons to dropdown menu
- Include combined report as option

### Recommended Approach: Option 1 (Fifth Button)

**Button HTML (add to base.py lines 143-159):**
```html
<div class="legend-item report-btn" onclick="showImprovementPriorities()">
    <span class="report-icon">🎯</span>
    <span>Priorities</span>
</div>
```

**Function Implementation (add to scripts.py after line 3511):**
```javascript
function showImprovementPriorities() {
    openViewerPanel();

    const viewerTitle = document.getElementById('viewer-title');
    const viewerContent = document.getElementById('viewer-content');

    viewerTitle.textContent = '🎯 Improvement Priorities';

    // 1. Get complexity data
    const chunksWithComplexity = [];
    function collectChunks(node) {
        if (chunkTypes.includes(node.type)) {
            const complexity = node.complexity !== undefined ? node.complexity : null;
            chunksWithComplexity.push({
                name: node.name,
                type: node.type,
                file_path: node.file_path || 'Unknown',
                complexity: complexity,
                grade: getComplexityGrade(complexity),
                node: node
            });
        }
        const children = node.children || node._children || [];
        children.forEach(child => collectChunks(child));
    }
    if (treeData) {
        collectChunks(treeData);
    }

    // 2. Get code smells
    const allSmells = detectCodeSmells(treeData);

    // 3. Combine and score
    const fileScores = new Map();

    // Score from complexity (F=5 points, D=4, C=3, B=2, A=1)
    chunksWithComplexity.forEach(chunk => {
        const filePath = chunk.file_path;
        if (!fileScores.has(filePath)) {
            fileScores.set(filePath, {
                path: filePath,
                complexityScore: 0,
                smellScore: 0,
                totalScore: 0,
                issues: []
            });
        }
        const score = fileScores.get(filePath);
        const gradePoints = {'F': 5, 'D': 4, 'C': 3, 'B': 2, 'A': 1, 'N/A': 0};
        score.complexityScore += gradePoints[chunk.grade] || 0;
        if (chunk.grade === 'F' || chunk.grade === 'D') {
            score.issues.push({
                type: 'complexity',
                name: chunk.name,
                details: `Complexity ${chunk.complexity.toFixed(1)} (Grade ${chunk.grade})`
            });
        }
    });

    // Score from smells (error=3 points, warning=1 point)
    allSmells.forEach(smell => {
        const filePath = smell.node.file_path;
        if (!fileScores.has(filePath)) {
            fileScores.set(filePath, {
                path: filePath,
                complexityScore: 0,
                smellScore: 0,
                totalScore: 0,
                issues: []
            });
        }
        const score = fileScores.get(filePath);
        const smellPoints = smell.severity === 'error' ? 3 : 1;
        score.smellScore += smellPoints;
        score.issues.push({
            type: 'smell',
            name: smell.node.name,
            details: `${smell.type}: ${smell.details}`
        });
    });

    // Calculate total scores
    fileScores.forEach((score, path) => {
        score.totalScore = score.complexityScore + score.smellScore;
    });

    // Sort files by total score (highest first)
    const sortedFiles = Array.from(fileScores.values())
        .filter(f => f.totalScore > 0)
        .sort((a, b) => b.totalScore - a.totalScore)
        .slice(0, 20); // Top 20 files

    // 4. Build HTML
    let html = '<div class="priorities-report">';

    // Summary
    html += '<div class="priorities-summary">';
    html += '<p>Files ranked by improvement priority (complexity + code smells)</p>';
    html += `<div class="summary-stats">
        <span>${sortedFiles.length} files need attention</span>
        <span>${allSmells.filter(s => s.severity === 'error').length} critical issues</span>
    </div>`;
    html += '</div>';

    // Top files table
    html += '<div class="priorities-table-container">';
    html += '<table class="priorities-table">';
    html += `
        <thead>
            <tr>
                <th>Rank</th>
                <th>File</th>
                <th>Complexity</th>
                <th>Smells</th>
                <th>Score</th>
                <th>Issues</th>
            </tr>
        </thead>
        <tbody>
    `;

    sortedFiles.forEach((file, index) => {
        const fileName = file.path.split('/').pop() || file.path;
        const rankEmoji = index < 3 ? '🔴' : (index < 10 ? '🟡' : '🟢');

        html += `
            <tr class="priority-row" onclick='navigateToFileByPath(${JSON.stringify(file.path)})'>
                <td>${rankEmoji} ${index + 1}</td>
                <td class="priority-file" title="${escapeHtml(file.path)}">${escapeHtml(fileName)}</td>
                <td>${file.complexityScore}</td>
                <td>${file.smellScore}</td>
                <td><strong>${file.totalScore}</strong></td>
                <td class="priority-issues">
                    <details>
                        <summary>${file.issues.length} issues</summary>
                        <ul>
                            ${file.issues.map(issue => `
                                <li><strong>${escapeHtml(issue.name)}</strong>: ${escapeHtml(issue.details)}</li>
                            `).join('')}
                        </ul>
                    </details>
                </td>
            </tr>
        `;
    });

    html += '</tbody></table></div>';
    html += '</div>'; // priorities-report

    viewerContent.innerHTML = html;
}
```

## Files to Modify

### 1. base.py (Add Button)
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py`

**Location:** After line 150 (inside the Reports section)

**Change:** Add new button entry:
```html
<div class="legend-item report-btn" onclick="showImprovementPriorities()">
    <span class="report-icon">🎯</span>
    <span>Priorities</span>
</div>
```

### 2. scripts.py (Add Function)
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`

**Location:** After line 3511 (after `filterCodeSmells()` function)

**Change:** Add complete `showImprovementPriorities()` function (see Recommended Approach above)

### 3. styles.py (Optional - Add Styling)
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/styles.py`

**Location:** End of file

**Change:** Add CSS for priorities report:
```css
.priorities-report {
    padding: 20px;
}

.priorities-summary {
    margin-bottom: 20px;
    padding: 15px;
    background: var(--bg-secondary);
    border-radius: 8px;
}

.summary-stats {
    display: flex;
    gap: 20px;
    margin-top: 10px;
    font-size: 14px;
    color: var(--text-secondary);
}

.priorities-table-container {
    overflow-x: auto;
}

.priorities-table {
    width: 100%;
    border-collapse: collapse;
}

.priorities-table th,
.priorities-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.priority-row {
    cursor: pointer;
}

.priority-row:hover {
    background: var(--hover-bg);
}

.priority-file {
    color: var(--text-primary);
    font-weight: 500;
}

.priority-issues details {
    cursor: pointer;
}

.priority-issues summary {
    color: var(--link-color);
}

.priority-issues ul {
    margin: 8px 0 0 0;
    padding-left: 20px;
    font-size: 13px;
}

.priority-issues li {
    margin: 4px 0;
}
```

## Code Snippets - Current Report Functions

### Complexity Report Data Collection
```javascript
// From showComplexityReport() - lines 3092-3116
function collectChunks(node) {
    if (chunkTypes.includes(node.type)) {
        const complexity = node.complexity !== undefined ? node.complexity : null;
        chunksWithComplexity.push({
            name: node.name,
            type: node.type,
            file_path: node.file_path || 'Unknown',
            start_line: node.start_line || 0,
            end_line: node.end_line || 0,
            complexity: complexity,
            grade: getComplexityGrade(complexity),
            node: node
        });
    }
    const children = node.children || node._children || [];
    children.forEach(child => collectChunks(child));
}
```

### Code Smells Detection
```javascript
// From detectCodeSmells() - lines 3282-3378
function detectCodeSmells(nodes) {
    const smells = [];

    function analyzeNode(node) {
        if (!chunkTypes.includes(node.type)) {
            const children = node.children || node._children || [];
            children.forEach(child => analyzeNode(child));
            return;
        }

        const lineCount = (node.end_line && node.start_line)
            ? node.end_line - node.start_line + 1
            : 0;
        const complexity = node.complexity || 0;

        // Long Method
        if (lineCount > 200) {
            smells.push({
                type: 'Long Method',
                severity: 'error',
                node: node,
                details: `${lineCount} lines (very long)`
            });
        } else if (lineCount > 100) {
            smells.push({
                type: 'Long Method',
                severity: 'warning',
                node: node,
                details: `${lineCount} lines (long)`
            });
        }

        // High Complexity
        if (complexity > 20) {
            smells.push({
                type: 'High Complexity',
                severity: 'error',
                node: node,
                details: `Complexity ${complexity.toFixed(1)} (very high)`
            });
        } else if (complexity > 15) {
            smells.push({
                type: 'High Complexity',
                severity: 'warning',
                node: node,
                details: `Complexity ${complexity.toFixed(1)} (high)`
            });
        }

        // Deep Nesting
        if (node.max_nesting_level > 6) {
            smells.push({
                type: 'Deep Nesting',
                severity: 'error',
                node: node,
                details: `${node.max_nesting_level} levels (very deep)`
            });
        } else if (node.max_nesting_level > 4) {
            smells.push({
                type: 'Deep Nesting',
                severity: 'warning',
                node: node,
                details: `${node.max_nesting_level} levels (deep)`
            });
        }

        // God Class
        if (node.type === 'class') {
            const children = node.children || node._children || [];
            const methodCount = children.filter(c => c.type === 'method').length;

            if (methodCount > 30 || lineCount > 800) {
                smells.push({
                    type: 'God Class',
                    severity: 'error',
                    node: node,
                    details: `${methodCount} methods, ${lineCount} lines (very large)`
                });
            }
        }

        const children = node.children || node._children || [];
        children.forEach(child => analyzeNode(child));
    }

    analyzeNode(nodes);
    return smells;
}
```

### Complexity Grading
```javascript
// From getComplexityGrade() - lines 3061-3068
function getComplexityGrade(complexity) {
    if (complexity === undefined || complexity === null) return 'N/A';
    if (complexity <= 5) return 'A';
    if (complexity <= 10) return 'B';
    if (complexity <= 15) return 'C';
    if (complexity <= 20) return 'D';
    return 'F';
}

// From getGradeColor() - lines 3070-3080
function getGradeColor(grade) {
    const colors = {
        'A': '#2ea043',
        'B': '#1f6feb',
        'C': '#d29922',
        'D': '#f0883e',
        'F': '#da3633',
        'N/A': '#8b949e'
    };
    return colors[grade] || colors['N/A'];
}
```

## Analysis Summary

### Strengths of Current Implementation
- ✅ Both complexity and code smell reports are fully functional
- ✅ Well-structured data collection and analysis
- ✅ Interactive navigation to specific code chunks
- ✅ Clear visual presentation with tables and filtering
- ✅ Consistent styling and UX patterns

### Gaps
- ❌ No combined/prioritized view
- ❌ User must click two separate buttons to understand improvement priorities
- ❌ No scoring/ranking system to prioritize files
- ❌ Cannot see which files have BOTH high complexity AND code smells

### Implementation Complexity
**Estimated Effort:** LOW to MEDIUM

**Reasons:**
1. All necessary data collection functions exist
2. Clear patterns to follow from existing reports
3. No new data sources needed
4. HTML/CSS patterns well-established

**Steps:**
1. Add button to base.py (1 line of HTML)
2. Add `showImprovementPriorities()` function to scripts.py (~150 lines)
3. Optional: Add CSS styling to styles.py (~80 lines)
4. Test with existing visualization data

**Dependencies:**
- Requires both `detectCodeSmells()` and complexity collection to work
- Uses existing helper functions: `getComplexityGrade()`, `navigateToFileByPath()`, `escapeHtml()`

## Recommendations

1. **Implement Option 1** (Fifth button - "🎯 Priorities")
   - Least disruptive to existing UI
   - Provides clear value without replacing existing reports
   - Easy to test and iterate

2. **Scoring Algorithm** should weight both complexity and smells:
   - Complexity: F=5, D=4, C=3, B=2, A=1 points per function
   - Smells: Error=3, Warning=1 points per smell
   - Aggregate at file level for prioritization

3. **Top N Display** (recommend 20 files max):
   - Prevents overwhelming the user
   - Focuses on highest-impact improvements
   - Can add pagination if needed later

4. **Make Issues Expandable**:
   - Use `<details>` element for per-file issue lists
   - Allows quick scanning while providing drill-down capability

## Next Steps

1. Create implementation ticket/issue
2. Add button to base.py
3. Implement `showImprovementPriorities()` function in scripts.py
4. Add CSS styling (optional but recommended)
5. Test with real codebase visualization
6. Gather user feedback on prioritization algorithm

## Files Analyzed
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/base.py` (226 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py` (4,404 lines - strategic sampling)

## Memory Usage Statistics
- Files read: 2 (1 full, 1 sampled via grep/offset reads)
- Total lines processed: ~500 lines (strategic sampling)
- Grep searches: 6 targeted pattern searches
- Tool availability: Vector search available, MCP ticketer available

---

**Research Complete:** 2025-12-15
**Recommendation:** Implement fifth "Priorities" button with combined complexity + smell scoring algorithm
