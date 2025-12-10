# PR Workflow Guide

## Branch Naming Convention

All feature branches should follow this pattern:

```
feature/<issue-number>-<short-description>
```

### Examples

| Issue | Branch Name |
|-------|-------------|
| #2 Create metric dataclasses | `feature/2-metric-dataclasses` |
| #3 Cognitive Complexity Collector | `feature/3-cognitive-complexity` |
| #10 Analyze CLI command | `feature/10-analyze-cli` |
| #14 Code smell detection | `feature/14-code-smell-detection` |

## PR Workflow States

The project uses a Kanban-style workflow:

| State | Description | Action |
|-------|-------------|--------|
| 📋 **Backlog** | Issue exists but not ready to start | Dependencies not met |
| 🎯 **Ready** | Dependencies met, ready to implement | Create branch, start work |
| 🔧 **In Progress** | Actively being developed | Commits being made |
| 👀 **In Review** | PR created, awaiting review | Awaiting approval |
| ✅ **Done** | PR merged, issue closed | Complete |

## Creating a PR

### 1. Check Dependencies

Before starting work, verify all blocking issues are complete:

```bash
# View issue with dependencies
gh issue view <issue-number> --repo bobmatnyc/mcp-vector-search
```

Look for the "Dependencies" section in the issue body.

### 2. Create Feature Branch

```bash
# Create and checkout feature branch
git checkout -b feature/<issue-number>-<short-description>

# Example
git checkout -b feature/2-metric-dataclasses
```

### 3. Link Issue to Branch

The branch name automatically links to the issue via the issue number.

### 4. Develop with Commits

```bash
# Make commits referencing the issue
git commit -m "feat(analysis): add ChunkMetrics dataclass

Implements the core dataclass for storing chunk-level metrics.

Refs #2"
```

### 5. Create Pull Request

```bash
# Push branch
git push -u origin feature/2-metric-dataclasses

# Create PR linking to issue
gh pr create --title "feat(analysis): add metric dataclasses and interfaces" \
  --body "## Summary
Implements the core data structures for the metrics collection system.

## Changes
- Added ChunkMetrics dataclass
- Added FileMetrics dataclass
- Added ProjectMetrics dataclass
- Added MetricCollector abstract base class

## Testing
- [ ] Unit tests added
- [ ] All tests pass

Closes #2"
```

### 6. Request Review

The PR will automatically move the issue to "In Review" when opened.

### 7. Merge and Close

When the PR is approved and merged:
- Issue automatically closes (via "Closes #X")
- Workflow moves to "Done"

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code refactoring |
| `docs` | Documentation |
| `test` | Adding tests |
| `chore` | Maintenance |

### Scopes for This Project

| Scope | Area |
|-------|------|
| `analysis` | Metrics collection system |
| `collectors` | Individual metric collectors |
| `cli` | CLI commands |
| `reporters` | Output reporters |
| `viz` | Visualization export |
| `db` | Database/storage |

### Examples

```bash
git commit -m "feat(collectors): implement cognitive complexity collector

Implements SonarQube's cognitive complexity algorithm with nesting penalties.

- Added CognitiveComplexityCollector class
- Handles Python, JavaScript, TypeScript
- Returns breakdown by category

Refs #3"
```

## Dependency Management

### Checking What's Ready

Issues in "Ready" state have no blockers:

```bash
# List ready issues
gh issue list --repo bobmatnyc/mcp-vector-search \
  --label "enhancement" \
  --json number,title,body \
  --jq '.[] | select(.body | contains("Blocked by:** None")) | "\(.number): \(.title)"'
```

### Critical Path

The critical path determines the minimum timeline:

```
#2 → #8 → #10 → #14 → #35 → #37
```

Prioritize issues on the critical path to avoid blocking downstream work.

## Automation

### GitHub Actions

PRs automatically trigger:
- Linting (Ruff, Black)
- Type checking (Mypy)
- Tests (Pytest)
- Security scanning (Bandit)

### Auto-Close Issues

Use keywords in PR body to auto-close issues:
- `Closes #X`
- `Fixes #X`
- `Resolves #X`

## Quick Reference

```bash
# Start work on issue #2
git checkout -b feature/2-metric-dataclasses
# ... make changes ...
git add .
git commit -m "feat(analysis): add ChunkMetrics dataclass

Refs #2"
git push -u origin feature/2-metric-dataclasses
gh pr create --title "feat(analysis): add metric dataclasses" --body "Closes #2"
```

## Links

- [Project Board](https://github.com/users/bobmatnyc/projects/13)
- [Milestones](https://github.com/bobmatnyc/mcp-vector-search/milestones)
- [All Issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
- [Design Document](../research/mcp-vector-search-structural-analysis-design.md)
