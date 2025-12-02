# Pre-Release Readiness Check - v0.14.5

**Date**: 2025-12-02
**Current Version**: 0.14.4
**Next Version**: 0.14.5 (proposed)
**Type**: Breaking Change - Claude Desktop Support Removal

## Executive Summary

🟡 **NOT READY FOR RELEASE** - Uncommitted changes require review and commit before release.

**Status Overview**:
- ✅ All changes properly tracked in git
- ❌ **6 modified files** not staged for commit
- ❌ **2 untracked research documents** not committed
- 🟡 CHANGELOG requires update to reflect Claude Desktop removal
- ✅ Documentation changes comprehensive and consistent
- ✅ Code changes complete (install/uninstall commands)

## Git Status Analysis

### Modified Files (Not Staged)

**Total**: 6 files with uncommitted changes

1. **README.md** - Removed Claude Desktop references
   - Removed install example for Claude Desktop
   - Updated platform list in MCP integration section

2. **docs/getting-started/installation.md** - Removed Claude Desktop documentation
   - Removed installation example
   - Updated MCP integration types list

3. **docs/guides/mcp-integration.md** - Removed Claude Desktop from integration guide
   - Removed installation command
   - Updated supported platforms list

4. **docs/reference/cli-commands.md** - Removed Claude Desktop from CLI reference
   - Removed from detected platforms list
   - Updated installation examples

5. **src/mcp_vector_search/cli/commands/install.py** - Removed Claude Desktop installer
   - Removed `claude-desktop` from SUPPORTED_PLATFORMS dict
   - Removed `@install_app.command("claude-desktop")` function
   - Updated help text to remove Claude Desktop from platform list

6. **src/mcp_vector_search/cli/commands/uninstall.py** - Removed Claude Desktop uninstaller
   - Removed `claude-desktop` from SUPPORTED_PLATFORMS dict
   - Removed `@uninstall_app.command("claude-desktop")` function
   - Updated help text to remove Claude Desktop from platform list

### Untracked Files

**Total**: 2 research documents (should be committed for traceability)

1. **docs/research/claude-desktop-documentation-review-2025-12-02.md** (610 lines)
   - Analysis of Claude Desktop documentation presence
   - Recommendation to remove Claude Desktop support
   - Justification for breaking change

2. **docs/research/claude-desktop-vs-code-installer-analysis-2025-12-02.md** (631 lines)
   - Technical analysis of installer architecture
   - Impact assessment for removal
   - Migration path documentation

## CHANGELOG Status

### Current State (Unreleased Section)

**Line 14** mentions Claude Desktop in feature list:
```markdown
- **Multi-Platform MCP**: Configures all installed MCP platforms (Claude Code, Cursor, Windsurf, VS Code, Claude Desktop)
```

### Required Updates

**CRITICAL**: CHANGELOG must be updated to reflect **BREAKING CHANGE**:

1. **Add new section for v0.14.5** (or move to appropriate version number)
2. **Add BREAKING CHANGES section** documenting Claude Desktop removal:
   ```markdown
   ### BREAKING CHANGES
   - **Claude Desktop Support Removed**: The `mcp-vector-search install claude-desktop` and
     `mcp-vector-search uninstall claude-desktop` commands have been removed.
     - **Reason**: Claude Desktop uses global configuration which conflicts with project-scoped workflow
     - **Migration Path**: Use Claude Code (project-scoped) or Cursor/Windsurf/VS Code (global alternatives)
     - **Impacted Users**: Users who installed via `install claude-desktop` must manually remove configuration
     - **Manual Cleanup**: Remove `mcp-vector-search` entry from `~/Library/Application Support/Claude/claude_desktop_config.json`
   ```

3. **Update Unreleased section line 14** to remove Claude Desktop:
   ```markdown
   - **Multi-Platform MCP**: Configures all installed MCP platforms (Claude Code, Cursor, Windsurf, VS Code)
   ```

## Documentation Consistency Check

### ✅ Consistent Removals Across All Files

**Platform Lists Updated**:
- README.md: ✅ Removed
- docs/getting-started/installation.md: ✅ Removed
- docs/guides/mcp-integration.md: ✅ Removed
- docs/reference/cli-commands.md: ✅ Removed
- src/mcp_vector_search/cli/commands/install.py: ✅ Removed
- src/mcp_vector_search/cli/commands/uninstall.py: ✅ Removed

**Remaining Platform Support**:
- Claude Code (project-scoped) - PRIMARY
- Cursor (global)
- Windsurf (global)
- VS Code (global)

### 🔍 Files That Should Be Checked

**Additional files to verify**:
1. `docs/RELEASES.md` - May contain version-specific Claude Desktop mentions
2. `tests/` directory - May have Claude Desktop installer tests to remove
3. Any example configs or templates referencing Claude Desktop

## Version Bump Recommendation

**Current**: v0.14.4
**Recommended**: v0.15.0 (MAJOR change due to breaking change)

**Justification**:
- Removal of CLI commands (`install claude-desktop`, `uninstall claude-desktop`) = Breaking Change
- Users with existing Claude Desktop configurations will need manual migration
- Follows semantic versioning: MAJOR.MINOR.PATCH

**Alternative**: v0.14.5 if project is pre-1.0 and uses 0.MAJOR.MINOR convention

## Pre-Commit Checklist

### Required Actions Before Release

1. **Review Changes**:
   ```bash
   git diff  # Review all modified files
   ```

2. **Stage Documentation Changes**:
   ```bash
   git add README.md
   git add docs/getting-started/installation.md
   git add docs/guides/mcp-integration.md
   git add docs/reference/cli-commands.md
   ```

3. **Stage Code Changes**:
   ```bash
   git add src/mcp_vector_search/cli/commands/install.py
   git add src/mcp_vector_search/cli/commands/uninstall.py
   ```

4. **Stage Research Documents**:
   ```bash
   git add docs/research/claude-desktop-documentation-review-2025-12-02.md
   git add docs/research/claude-desktop-vs-code-installer-analysis-2025-12-02.md
   ```

5. **Update CHANGELOG**:
   ```bash
   # Edit docs/CHANGELOG.md to add BREAKING CHANGES section
   git add docs/CHANGELOG.md
   ```

6. **Update Version**:
   ```bash
   # Edit src/mcp_vector_search/__init__.py
   # Change __version__ = "0.14.4" to __version__ = "0.15.0"
   git add src/mcp_vector_search/__init__.py
   ```

7. **Verify Tests**:
   ```bash
   # Check if any tests reference Claude Desktop
   grep -r "claude-desktop" tests/
   grep -r "claude_desktop" tests/

   # Remove or update affected tests
   # Run test suite
   pytest
   ```

8. **Commit Changes**:
   ```bash
   git commit -m "feat!: remove Claude Desktop support (BREAKING CHANGE)

   - Remove install/uninstall commands for Claude Desktop
   - Update all documentation to remove Claude Desktop references
   - Add migration guide for affected users

   BREAKING CHANGE: Claude Desktop integration removed.

   Users must migrate to:
   - Claude Code (recommended, project-scoped)
   - Cursor/Windsurf/VS Code (global alternatives)

   Manual cleanup required:
   Remove mcp-vector-search from ~/Library/Application Support/Claude/claude_desktop_config.json

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

9. **Tag Release**:
   ```bash
   git tag -a v0.15.0 -m "Version 0.15.0 - Remove Claude Desktop support"
   ```

10. **Push to Remote**:
    ```bash
    git push origin main
    git push origin v0.15.0
    ```

## Testing Requirements

### Manual Testing Checklist

Before release, verify:

1. **Installation Commands**:
   ```bash
   mcp-vector-search install list  # Should NOT show claude-desktop
   mcp-vector-search install claude-desktop  # Should return error or "command not found"
   ```

2. **Uninstallation Commands**:
   ```bash
   mcp-vector-search uninstall list  # Should NOT show claude-desktop
   mcp-vector-search uninstall claude-desktop  # Should return error or "command not found"
   ```

3. **Help Text**:
   ```bash
   mcp-vector-search install --help  # Should NOT mention claude-desktop
   mcp-vector-search uninstall --help  # Should NOT mention claude-desktop
   ```

4. **Setup Command**:
   ```bash
   mcp-vector-search setup  # Should NOT attempt Claude Desktop detection/configuration
   ```

### Automated Testing

Run full test suite:
```bash
pytest -v
pytest tests/cli/  # Focus on CLI tests
pytest tests/integration/  # Integration tests
```

## Risk Assessment

### High Risk Areas

1. **User Impact**: Users with existing Claude Desktop configurations will be broken
   - **Mitigation**: Clear migration guide in CHANGELOG and documentation
   - **Severity**: MEDIUM (global config won't break, but won't update)

2. **Documentation Completeness**: May have missed some references
   - **Mitigation**: Full-text search for "claude-desktop" and "Claude Desktop"
   - **Severity**: LOW (can be patched in v0.15.1)

3. **Test Coverage**: Unknown if tests exist for Claude Desktop installer
   - **Mitigation**: Search and remove affected tests
   - **Severity**: MEDIUM (CI may fail if tests not updated)

### Low Risk Areas

1. **Code Changes**: Clean removal, no complex refactoring
2. **Backward Compatibility**: Only affects one optional feature
3. **Dependencies**: No dependency changes required

## Success Criteria

**Release is ready when**:

✅ All modified files committed to git
✅ CHANGELOG updated with BREAKING CHANGES section
✅ Version bumped to v0.15.0 (or appropriate version)
✅ Research documents committed for traceability
✅ All tests passing
✅ Manual testing completed
✅ Migration guide documented
✅ No references to Claude Desktop remain in code/docs

## Additional Verification Commands

```bash
# Find any remaining Claude Desktop references
grep -r "claude-desktop" . --exclude-dir=.git --exclude-dir=docs/research
grep -r "claude_desktop" . --exclude-dir=.git --exclude-dir=docs/research
grep -r "Claude Desktop" . --exclude-dir=.git --exclude-dir=docs/research

# Check for config references
grep -r "claude_desktop_config.json" . --exclude-dir=.git

# Verify SUPPORTED_PLATFORMS
grep -A20 "SUPPORTED_PLATFORMS" src/mcp_vector_search/cli/commands/install.py
grep -A20 "SUPPORTED_PLATFORMS" src/mcp_vector_search/cli/commands/uninstall.py
```

## Migration Guide for Users

**For documentation/release notes**:

### Migrating from Claude Desktop to Recommended Platforms

If you previously used `mcp-vector-search install claude-desktop`, follow these steps:

1. **Choose your migration path**:
   - **Claude Code** (recommended): Project-scoped, works per-project
   - **Cursor/Windsurf/VS Code**: Global configuration alternatives

2. **Install new platform**:
   ```bash
   # Recommended: Claude Code
   cd /path/to/your/project
   mcp-vector-search install claude-code

   # Or: Global alternative
   mcp-vector-search install cursor
   ```

3. **Manual cleanup** (remove old Claude Desktop config):
   ```bash
   # Backup your config
   cp ~/Library/Application\ Support/Claude/claude_desktop_config.json \
      ~/Library/Application\ Support/Claude/claude_desktop_config.json.backup

   # Edit and remove mcp-vector-search server entry
   # File: ~/Library/Application Support/Claude/claude_desktop_config.json
   # Remove the "mcp-vector-search" or "mcp" entry from "mcpServers" section
   ```

4. **Restart your platform**:
   - Claude Code: Restart Claude CLI
   - Cursor: Restart Cursor IDE
   - Windsurf: Restart Windsurf
   - VS Code: Reload VS Code window

## Conclusion

**RELEASE STATUS**: 🟡 **NOT READY**

**Required Actions**:
1. Update CHANGELOG with BREAKING CHANGES section
2. Bump version to v0.15.0
3. Search for and remove any Claude Desktop tests
4. Commit all changes with proper breaking change notation
5. Run full test suite
6. Perform manual testing

**Estimated Time to Release Ready**: 30-60 minutes

**Recommendation**: Do NOT release until all items in Pre-Commit Checklist are completed.

---

**Files Modified**: 6
**Files Untracked**: 2
**Breaking Changes**: 1 (Claude Desktop removal)
**Migration Complexity**: LOW (manual config cleanup only)
**User Impact**: MEDIUM (affects Claude Desktop users only)
