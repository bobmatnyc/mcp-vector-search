# GitHub Milestones and Issue Dependencies Setup

This document provides instructions for setting up GitHub milestones and issue dependencies for the Structural Code Analysis project.

## Overview

The project is organized into 5 milestones corresponding to the rollout phases:

1. **v0.17.0 - Core Metrics** (Week 1-2): Issues #1-11
2. **v0.18.0 - Quality Gates** (Week 3): Issues #12-18
3. **v0.19.0 - Cross-File Analysis** (Week 4): Issues #19-26
4. **v0.20.0 - Visualization Export** (Week 5): Issues #27-33
5. **v0.21.0 - Search Integration** (Week 8+): Issues #34-37

## Quick Start

### Prerequisites

1. GitHub CLI (`gh`) installed and authenticated
2. Repository access to `bobmatnyc/mcp-vector-search`
3. Permissions to create milestones and edit issues

### Authentication Check

```bash
# Check authentication status
gh auth status

# If not authenticated or using wrong account:
gh auth login

# Select the account with write access to the repository
```

### Running the Setup Scripts

```bash
# Navigate to project root
cd /Users/masa/Projects/mcp-vector-search

# Step 1: Create milestones and assign issues
./scripts/setup_milestones.sh

# Step 2: Add dependency information to issues
./scripts/add_issue_dependencies.sh
```

## Milestone Details

### Milestone 1: v0.17.0 - Core Metrics
- **Duration**: 2 weeks
- **Due Date**: Calculated as +2 weeks from creation
- **Description**: Tier 1 collectors integrated into indexer, extended chunk metadata in ChromaDB, analyze --quick command, basic console reporter
- **Issues**: #1, #2, #3, #4, #5, #6, #7, #8, #9, #10, #11

### Milestone 2: v0.18.0 - Quality Gates
- **Duration**: 1 week (Week 3)
- **Due Date**: Calculated as +3 weeks from creation
- **Description**: Threshold configuration system, SARIF output for CI integration, --fail-on-smell exit codes, diff-aware analysis
- **Issues**: #12, #13, #14, #15, #16, #17, #18

### Milestone 3: v0.19.0 - Cross-File Analysis
- **Duration**: 1 week (Week 4)
- **Due Date**: Calculated as +4 weeks from creation
- **Description**: Tier 4 collectors (afferent coupling, circular deps), dependency graph construction, SQLite metrics store, trend tracking
- **Issues**: #19, #20, #21, #22, #23, #24, #25, #26

### Milestone 4: v0.20.0 - Visualization Export
- **Duration**: 1 week (Week 5)
- **Due Date**: Calculated as +5 weeks from creation
- **Description**: JSON export for visualizer, all chart data schemas finalized, HTML standalone report, documentation
- **Issues**: #27, #28, #29, #30, #31, #32, #33

### Milestone 5: v0.21.0 - Search Integration
- **Duration**: Future (Week 8+)
- **Due Date**: Calculated as +8 weeks from creation
- **Description**: Quality-aware search ranking and filtering, MCP tool exposure
- **Issues**: #34, #35, #36, #37

## Dependency Graph

### Critical Path

The critical path for the project follows this sequence:

```
#2 (dataclasses)
  → #8 (integrate collectors)
    → #10 (analyze CLI)
      → #14 (code smells)
        → #35 (quality filters)
          → #36 (quality ranking)
            → #37 (MCP tools)
```

### Phase 1 Dependencies (Core Metrics)

```
#2 (Core dataclasses) [START]
  ├─→ #3 (Cognitive complexity)
  ├─→ #4 (Cyclomatic complexity)
  ├─→ #5 (Nesting depth)
  ├─→ #6 (Parameter count)
  ├─→ #7 (Method count)
  ├─→ #8 (Integrate collectors)
  └─→ #9 (ChromaDB schema)

#8 (Integrate collectors) [REQUIRES: #2, #3, #4, #5, #6, #7]
  └─→ #10 (Analyze CLI)

#9 (ChromaDB schema) [REQUIRES: #2]
  └─→ #10 (Analyze CLI)

#10 (Analyze CLI) [REQUIRES: #8, #9]
  └─→ #11 (Console reporter)
```

### Phase 2 Dependencies (Quality Gates)

```
#13 (Thresholds) [REQUIRES: #2]
  └─→ #14 (Code smells)

#14 (Code smells) [REQUIRES: #8, #13]
  └─→ #15 (SARIF output)
  └─→ #16 (Fail-on-smell)

#15 (SARIF output) [REQUIRES: #10, #14]
  └─→ #16 (Fail-on-smell)

#17 (Diff-aware) [REQUIRES: #10]
  └─→ #18 (Baseline)
```

### Phase 3 Dependencies (Cross-File Analysis)

```
#20 (Efferent coupling) [REQUIRES: #2, #8]
  ├─→ #21 (Afferent coupling)
  ├─→ #22 (Instability index)
  └─→ #23 (Circular deps)

#21 (Afferent coupling) [REQUIRES: #20]
  └─→ #22 (Instability index)

#24 (SQLite store) [REQUIRES: #2]
  ├─→ #25 (Trend tracking)
  ├─→ #32 (Tech debt)
  └─→ #33 (Status --metrics)

#26 (LCOM4) [REQUIRES: #2, #8]
```

### Phase 4 Dependencies (Visualization Export)

```
#28 (JSON schema) [REQUIRES: #2]
  └─→ #29 (JSON exporter)

#29 (JSON exporter) [REQUIRES: #28, #10]
  └─→ #30 (HTML report)

#31 (Halstead) [REQUIRES: #2, #8]

#32 (Tech debt) [REQUIRES: #14, #24]

#33 (Status --metrics) [REQUIRES: #10, #24]
```

### Phase 5 Dependencies (Search Integration)

```
#35 (Quality filters) [REQUIRES: #10, #14]
  ├─→ #36 (Quality ranking)
  └─→ #37 (MCP tools)

#37 (MCP tools) [REQUIRES: #10, #35]
```

## Dependency Summary by Issue

| Issue | Title | Blocked By | Blocks |
|-------|-------|------------|--------|
| #2 | Core dataclasses | None | #3, #4, #5, #6, #7, #8, #9 |
| #3 | Cognitive complexity | #2 | #8 |
| #4 | Cyclomatic complexity | #2 | #8 |
| #5 | Nesting depth | #2 | #8 |
| #6 | Parameter count | #2 | #8 |
| #7 | Method count | #2 | #8 |
| #8 | Integrate collectors | #2, #3, #4, #5, #6, #7 | #10, #14, #20, #26, #31 |
| #9 | ChromaDB schema | #2 | #10 |
| #10 | Analyze CLI | #8, #9 | #11, #14, #15, #17, #29, #33, #35 |
| #11 | Console reporter | #10 | None |
| #13 | Thresholds | #2 | #14 |
| #14 | Code smells | #8, #13 | #15, #16, #32, #35 |
| #15 | SARIF output | #10, #14 | #16 |
| #16 | Fail-on-smell | #14, #15 | None |
| #17 | Diff-aware | #10 | #18 |
| #18 | Baseline | #17 | None |
| #20 | Efferent coupling | #2, #8 | #21, #22, #23 |
| #21 | Afferent coupling | #20 | #22 |
| #22 | Instability index | #20, #21 | None |
| #23 | Circular deps | #20 | None |
| #24 | SQLite store | #2 | #25, #32, #33 |
| #25 | Trend tracking | #24 | None |
| #26 | LCOM4 | #2, #8 | None |
| #28 | JSON schema | #2 | #29 |
| #29 | JSON exporter | #28, #10 | #30 |
| #30 | HTML report | #29 | None |
| #31 | Halstead | #2, #8 | None |
| #32 | Tech debt | #14, #24 | None |
| #33 | Status --metrics | #10, #24 | None |
| #35 | Quality filters | #10, #14 | #36, #37 |
| #36 | Quality ranking | #35 | None |
| #37 | MCP tools | #10, #35 | None |

## Manual Setup (Alternative)

If the scripts don't work or you prefer manual setup:

### Create Milestones via GitHub UI

1. Navigate to: https://github.com/bobmatnyc/mcp-vector-search/milestones
2. Click "New milestone"
3. Create each milestone with the details above
4. Assign issues to milestones by editing each issue

### Add Dependencies via Issue Edit

For each issue:
1. Go to the issue page
2. Click "Edit" on the issue description
3. Scroll to the end of the description
4. Add the dependencies section:

```markdown
## Dependencies

**Blocked by:** #2, #3
**Blocks:** #10, #14
```

## Troubleshooting

### Authentication Issues

```bash
# Clear existing auth and re-authenticate
gh auth logout
gh auth login

# Select GitHub.com and follow prompts
```

### Permission Errors

Ensure your GitHub account has:
- Write access to the repository
- Permission to create milestones
- Permission to edit issues

### Script Errors

If the scripts fail:
1. Check that `gh` CLI is installed: `gh --version`
2. Verify authentication: `gh auth status`
3. Test API access: `gh api repos/bobmatnyc/mcp-vector-search`
4. Review error messages for specific issues

## Verification

After running the scripts:

1. **Check milestones**: https://github.com/bobmatnyc/mcp-vector-search/milestones
   - Should see 5 milestones with correct dates
   - Each milestone should have issues assigned

2. **Check issues**: https://github.com/bobmatnyc/mcp-vector-search/issues
   - Each issue should have a milestone
   - Each issue should have a "Dependencies" section

3. **Verify dependency graph**:
   - Use GitHub's issue view to see relationships
   - Check that blockers are accurate

## Next Steps

After setup:
1. Review the dependency graph for accuracy
2. Start work on #2 (no dependencies)
3. Use GitHub's project boards to track progress
4. Update issue status as work progresses

## References

- [GitHub REST API - Milestones](https://docs.github.com/en/rest/issues/milestones)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [Project Design Document](/Users/masa/Projects/mcp-vector-search/docs/research/mcp-vector-search-structural-analysis-design.md)

---

**Last Updated**: December 9, 2025
