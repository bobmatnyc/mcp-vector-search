# Changesets

This directory contains changeset files that describe changes for upcoming releases.

## What are Changesets?

Changesets are a way to declare changes to your codebase with intent. Each changeset:
- Describes what changed
- Specifies the type of change (patch, minor, major)
- Gets consumed during release to update CHANGELOG.md

## Workflow

### 1. Add a Changeset

When you make a change, add a changeset:

```bash
# Using Make
make changeset-add TYPE=patch DESC="fix: resolve search bug"

# Or directly
python3 scripts/changeset.py add --type patch --description "fix: resolve search bug"
```

**Change Types:**
- `patch`: Bug fixes, minor improvements (0.7.1 → 0.7.2)
- `minor`: New features, non-breaking changes (0.7.1 → 0.8.0)
- `major`: Breaking changes (0.7.1 → 1.0.0)

### 2. View Pending Changesets

```bash
# Using Make
make changeset-view

# Or directly
python3 scripts/changeset.py list
```

### 3. Consume Changesets (During Release)

Changesets are automatically consumed during release:

```bash
# Release workflow automatically consumes changesets
make release-patch   # Consumes all changesets, updates CHANGELOG
make release-minor
make release-major
```

Or manually:

```bash
python3 scripts/changeset.py consume --version 0.7.2
```

## Changeset File Format

Changesets are stored as markdown files: `.changesets/YYYYMMDD-HHMMSS-slug.md`

Example:
```markdown
---
type: patch
---

## Summary
Fix search performance regression

## Details
- Optimize query expansion algorithm
- Add caching for frequent searches
- Reduce database connection overhead

## Breaking Changes
None
```

## Best Practices

1. **One changeset per PR/feature**: Keep changes focused
2. **Clear descriptions**: Explain what and why, not how
3. **User-facing language**: Write for users, not developers
4. **Breaking changes**: Always document breaking changes
5. **Add early**: Create changeset when you start working

## Examples

### Bug Fix (Patch)
```bash
make changeset-add TYPE=patch DESC="fix: correct HTML parser attribute extraction"
```

### New Feature (Minor)
```bash
make changeset-add TYPE=minor DESC="feat: add Java language support"
```

### Breaking Change (Major)
```bash
make changeset-add TYPE=major DESC="feat!: redesign search API with async support"
```

## Integration with Release Process

The release workflow automatically:
1. Collects all pending changesets
2. Groups changes by type (Added, Changed, Fixed, Breaking)
3. Updates CHANGELOG.md with formatted entries
4. Deletes consumed changeset files
5. Commits changes with version tag

## Manual Operations

### Edit a Changeset
Simply edit the markdown file in `.changesets/` directory.

### Delete a Changeset
Remove the file from `.changesets/` directory.

### Validate Changesets
```bash
python3 scripts/changeset.py validate
```

## Directory Structure

```
.changesets/
├── README.md                    # This file
├── template.md                  # Template for new changesets
└── YYYYMMDD-HHMMSS-slug.md     # Individual changeset files
```

## Why Changesets?

**Benefits:**
- ✅ Structured changelog entries
- ✅ Clear change history
- ✅ Automated release notes
- ✅ Version bump guidance
- ✅ Team collaboration on release notes
- ✅ No manual CHANGELOG editing

**Comparison to alternatives:**
- **Conventional Commits**: Changesets provide richer context
- **Manual CHANGELOG**: Changesets automate the process
- **Auto-generated**: Changesets give human-curated content

## Troubleshooting

### No changesets to consume
If you see "No changesets found", add changesets before releasing:
```bash
make changeset-add TYPE=patch DESC="release: version bump"
```

### Changeset validation failed
Check the changeset file format:
- YAML frontmatter with `type` field
- Valid type: patch, minor, or major
- Markdown content after frontmatter

### Duplicate changeset entries
Each changeset file should describe a unique change. Merge similar changesets.

## Reference

- [Changesets Documentation](https://github.com/changesets/changesets)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
