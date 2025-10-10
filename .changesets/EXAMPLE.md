# Changeset System - Complete Example

This example demonstrates the full changeset workflow for the mcp-vector-search project.

## Scenario: Adding a New Feature

You're adding Java language support to the project. Here's how you'd use the changeset system:

### 1. Start Development

```bash
# Create a feature branch
git checkout -b feat/java-support

# Make your code changes
# ... edit files, add Java parser, etc.
```

### 2. Add a Changeset

```bash
# Add a changeset describing your changes
make changeset-add TYPE=minor DESC="feat: add Java language support"

# Or use the script directly
python3 scripts/changeset.py add --type minor --description "feat: add Java language support"
```

This creates: `.changesets/YYYYMMDD-HHMMSS-feat-add-java-language-support.md`

### 3. Edit the Changeset Details

Edit the generated file to add comprehensive details:

```markdown
---
type: minor
---

## Summary
Add Java language support with AST-aware parsing

## Details
- Implement JavaParser class using tree-sitter
- Extract classes, interfaces, methods, and annotations
- Support JavaDoc comment parsing
- Add .java and .jav file extension support
- Include Spring Framework patterns

## Impact
Users can now:
- Search Java codebases semantically
- Find Java classes, methods, and interfaces
- Search across 9 languages (up from 8)

## Breaking Changes
None

## Related
- Issue: #42
- Inspired by existing PHP and Ruby parsers
```

### 4. View All Pending Changesets

```bash
# See what changes are queued for next release
make changeset-view

# Output:
# Pending Changesets (1):
#
# [MINOR] (1 changes)
#   • Add Java language support with AST-aware parsing
#     File: 20251009-123456-feat-add-java-language-support.md
```

### 5. Validate Changesets

```bash
# Ensure all changesets are properly formatted
make changeset-validate

# Output:
# ✓ 20251009-123456-feat-add-java-language-support.md: valid (minor)
# All changesets are valid!
```

### 6. Commit Your Changes

```bash
# Commit code and changeset together
git add .
git commit -m "feat: implement Java language support"
git push origin feat/java-support
```

### 7. Release (Maintainer Workflow)

When ready to release:

```bash
# Run release workflow (automatically consumes changesets)
make release-minor

# This will:
# 1. Bump version (0.7.1 → 0.8.0)
# 2. Increment build number
# 3. Consume all changesets → update CHANGELOG.md
# 4. Update README.md and CLAUDE.md version badges
# 5. Create git commit and tag
# 6. Build distribution packages
```

The CHANGELOG.md will be automatically updated with:

```markdown
## [0.8.0] - 2025-10-09

### Added
- **Add Java language support with AST-aware parsing**
  - Implement JavaParser class using tree-sitter
  - Extract classes, interfaces, methods, and annotations
  - Support JavaDoc comment parsing
  - Add .java and .jav file extension support
  - Include Spring Framework patterns
```

### 8. Publish Release

```bash
# Publish to PyPI
make publish

# Push to GitHub
make git-push
```

## Quick Reference

### Common Commands

```bash
# Add changeset for a bug fix
make changeset-add TYPE=patch DESC="fix: resolve search timeout issue"

# Add changeset for a new feature
make changeset-add TYPE=minor DESC="feat: add code completion support"

# Add changeset for breaking change
make changeset-add TYPE=major DESC="feat!: redesign search API"

# View pending changesets
make changeset-view

# Validate all changesets
make changeset-validate

# Test documentation update (dry-run)
DRY_RUN=1 make docs-update

# Update only README.md
make docs-update-readme

# Release workflow
make release-patch   # Bug fixes
make release-minor   # New features
make release-major   # Breaking changes
```

### Change Types

- **patch**: Bug fixes, small improvements (0.7.1 → 0.7.2)
- **minor**: New features, non-breaking changes (0.7.1 → 0.8.0)
- **major**: Breaking changes (0.7.1 → 1.0.0)

### File Structure

```
.changesets/
├── README.md                           # Documentation
├── template.md                         # Template for new changesets
├── EXAMPLE.md                          # This file
└── YYYYMMDD-HHMMSS-description.md     # Individual changesets
```

### Integration Points

1. **Version Manager** (`scripts/version_manager.py`): Handles version bumping
2. **Changeset Manager** (`scripts/changeset.py`): Manages changeset lifecycle
3. **Docs Updater** (`scripts/update_docs.py`): Updates documentation
4. **Makefile**: Orchestrates the complete workflow

## Benefits

✅ **Structured Release Notes**: Human-curated, meaningful changelog entries
✅ **Automated Updates**: Documentation stays in sync automatically
✅ **Version Control Friendly**: Text files in git, easy to review
✅ **Developer Friendly**: Simple CLI, clear workflow
✅ **Maintainer Friendly**: One command for complete release

## Troubleshooting

### No changesets found during release

Add at least one changeset before releasing:

```bash
make changeset-add TYPE=patch DESC="release: version bump"
```

### Changeset validation failed

Check the YAML frontmatter format:

```yaml
---
type: patch  # Must be patch, minor, or major
---
```

### Documentation not updating

Ensure the alpha release line exists in README.md:

```markdown
> ⚠️ **Alpha Release (vX.Y.Z)**: ...
```

## Advanced Usage

### Manual Changeset Consumption

```bash
# Consume changesets for specific version
python3 scripts/changeset.py consume --version 0.8.0

# Dry-run to preview
python3 scripts/changeset.py consume --version 0.8.0 --dry-run
```

### Manual Documentation Update

```bash
# Update all docs
python3 scripts/update_docs.py --version 0.8.0 --type minor

# Update specific files
python3 scripts/update_docs.py --version 0.8.0 --readme-only
python3 scripts/update_docs.py --version 0.8.0 --claude-only
```

### Editing Changesets

Simply edit the markdown file:

```bash
# Edit a changeset
vim .changesets/YYYYMMDD-HHMMSS-description.md

# Delete a changeset
rm .changesets/YYYYMMDD-HHMMSS-description.md
```

## Best Practices

1. **Add changesets early**: Create changeset when you start a feature
2. **One changeset per feature**: Keep changes focused and atomic
3. **Write for users**: Describe impact, not implementation
4. **Document breaking changes**: Always explain migration path
5. **Link to issues/PRs**: Add context with Related section
6. **Validate before committing**: Run `make changeset-validate`
