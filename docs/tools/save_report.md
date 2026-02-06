# save_report Tool

## Overview

The `save_report` tool saves analysis or search results as markdown files with metadata headers for documentation purposes.

## Usage

```typescript
{
  name: "save_report",
  arguments: {
    content: string,            // Required: Markdown content to save
    report_type?: string,       // Optional: Type of report (default: "custom")
    output_path?: string,       // Optional: Custom output path
    filename_prefix?: string    // Optional: Prefix for auto-generated filename
  }
}
```

## Parameters

### content (required)
- **Type**: `string`
- **Description**: Markdown content to save in the report file

### report_type (optional)
- **Type**: `string`
- **Enum**: `"analysis"`, `"search"`, `"smells"`, `"hotspots"`, `"custom"`
- **Default**: `"custom"`
- **Description**: Type of report, used for filename generation

### output_path (optional)
- **Type**: `string`
- **Description**: Custom output path (file or directory)
  - If a file path (has extension): saves to that exact file
  - If a directory path (no extension): auto-generates filename in that directory
  - If not provided: uses `reports/` directory in project root

### filename_prefix (optional)
- **Type**: `string`
- **Description**: Prefix for auto-generated filenames (e.g., `"project"` → `project_analysis_20260206_143022.md`)

## File Naming Convention

When filename is auto-generated:
```
{prefix_}{report_type}_{YYYYMMDD_HHMMSS}.md
```

Examples:
- `custom_20260206_143022.md` (default)
- `project_analysis_20260206_143022.md` (with prefix)
- `smells_20260206_143022.md` (specific type)

## Metadata Header

All saved reports include a YAML frontmatter metadata header:

```markdown
---
generated: 2026-02-06 14:30:22
project: /absolute/path/to/project
type: analysis
---

{original content}
```

## Examples

### Example 1: Basic Usage
Save a report with default settings:

```typescript
{
  name: "save_report",
  arguments: {
    content: "# Analysis Results\n\n## Summary\nCode analysis completed successfully."
  }
}
```

Output: `reports/custom_20260206_143022.md`

### Example 2: Typed Report with Prefix
Save an analysis report with custom prefix:

```typescript
{
  name: "save_report",
  arguments: {
    content: "# Project Analysis\n\n**Health Score**: 85\n**Issues**: 3",
    report_type: "analysis",
    filename_prefix: "myproject"
  }
}
```

Output: `reports/myproject_analysis_20260206_143022.md`

### Example 3: Custom Directory
Save to a custom directory:

```typescript
{
  name: "save_report",
  arguments: {
    content: "# Code Smells\n\n- Long methods: 5\n- Complex functions: 3",
    report_type: "smells",
    output_path: "docs/reports"
  }
}
```

Output: `docs/reports/smells_20260206_143022.md`

### Example 4: Specific Filename
Save with a specific filename:

```typescript
{
  name: "save_report",
  arguments: {
    content: "# Complexity Hotspots\n\n1. module.py - Complexity: 42",
    report_type: "hotspots",
    output_path: "reports/latest_hotspots.md"
  }
}
```

Output: `reports/latest_hotspots.md`

### Example 5: Search Results
Save search results:

```typescript
{
  name: "save_report",
  arguments: {
    content: "# Search Results: Authentication\n\n**Query**: user authentication\n**Matches**: 12\n\n...",
    report_type: "search",
    filename_prefix: "auth"
  }
}
```

Output: `reports/auth_search_20260206_143022.md`

## Response

### Success Response
```typescript
{
  content: [{
    type: "text",
    text: "Report saved successfully to: /absolute/path/to/report.md"
  }],
  isError: false
}
```

### Error Responses

#### Missing Content
```typescript
{
  content: [{
    type: "text",
    text: "content parameter is required"
  }],
  isError: true
}
```

#### Permission Denied
```typescript
{
  content: [{
    type: "text",
    text: "Permission denied writing to /path/to/file.md: [error details]"
  }],
  isError: true
}
```

#### General Failure
```typescript
{
  content: [{
    type: "text",
    text: "Failed to save report: [error details]"
  }],
  isError: true
}
```

## Use Cases

### Documentation Workflow
1. Run analysis: `analyze_project`
2. Save results: `save_report` with analysis output
3. Commit to repository for tracking over time

### Code Review Preparation
1. Find code smells: `find_smells`
2. Get complexity hotspots: `get_complexity_hotspots`
3. Save combined report for team review

### Search Result Persistence
1. Run semantic search: `search_code`
2. Save interesting results for documentation
3. Reference in team discussions

### Regular Health Checks
1. Schedule periodic analysis runs
2. Save timestamped reports
3. Track code quality metrics over time

## Integration with Other Tools

### With analyze_project
```typescript
// 1. Analyze project
const analysis = await call_tool("analyze_project", {
  output_format: "detailed"
});

// 2. Save results
await call_tool("save_report", {
  content: formatAnalysisAsMarkdown(analysis),
  report_type: "analysis",
  filename_prefix: "weekly"
});
```

### With find_smells
```typescript
// 1. Find code smells
const smells = await call_tool("find_smells", {
  severity: "error"
});

// 2. Save critical issues
await call_tool("save_report", {
  content: formatSmellsAsMarkdown(smells),
  report_type: "smells",
  output_path: "docs/code-quality"
});
```

### With search_code
```typescript
// 1. Search for patterns
const results = await call_tool("search_code", {
  query: "authentication logic"
});

// 2. Document findings
await call_tool("save_report", {
  content: formatSearchResults(results),
  report_type: "search",
  filename_prefix: "auth-review"
});
```

## Best Practices

### Organizing Reports
- Use `reports/` for temporary/working reports
- Use `docs/reports/` for committed documentation
- Use descriptive filename prefixes for easier identification

### Content Formatting
- Start with H1 heading for report title
- Include metadata (date, query, filters) in content
- Use markdown formatting for readability
- Add actionable recommendations

### Version Control
- Commit important reports to track changes
- Use consistent naming for periodic reports
- Add `.gitignore` entries for temporary reports

### Automation
- Integrate with CI/CD for automated quality reports
- Schedule periodic health check reports
- Alert on threshold violations in reports

## Error Handling

The tool handles various error scenarios:
- **Directory creation**: Automatically creates missing directories
- **Permission issues**: Reports clear error messages
- **Path validation**: Handles relative and absolute paths
- **Encoding**: Uses UTF-8 for proper unicode support

## File System Behavior

- **Directory creation**: Automatically creates parent directories
- **File overwriting**: Overwrites existing files without warning
- **Permissions**: Requires write access to output directory
- **Encoding**: Files are written with UTF-8 encoding

## Limitations

- No file size limits enforced (be mindful with large reports)
- No automatic compression or archiving
- No built-in report versioning (use git)
- No validation of markdown syntax
