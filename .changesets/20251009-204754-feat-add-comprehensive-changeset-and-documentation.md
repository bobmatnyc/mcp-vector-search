---
type: minor  # patch | minor | major
---

## Summary
Add comprehensive changeset and documentation update system for automated release management

## Details
- Created .changesets/ directory with README and template
- Implemented scripts/changeset.py for changeset management (add, list, consume, validate)
- Implemented scripts/update_docs.py for automated documentation updates
- Updated Makefile with changeset and docs-update targets
- Integrated changeset consumption into release workflow
- Added automatic README.md and CLAUDE.md version updates

## Impact
Developers can now:
- Add changesets to describe changes during development
- Automatically generate structured CHANGELOG entries
- Update documentation version badges automatically during releases
- Track changes in a human-readable, version-control friendly format

## Breaking Changes
None - this is a new feature that enhances the existing release workflow

## Related
- Inspired by changesets/changesets
- Integrates with existing Makefile release workflow
- Complements version_manager.py
