# Release Process & Best Practices

## 🎯 Release Philosophy

MCP Vector Search follows a structured release process that ensures quality, reliability, and clear communication with users. Our release strategy balances rapid iteration with stability guarantees.

---

## 🔄 Release Workflow

### 1. Pre-Release Planning

#### Version Planning
```bash
# Determine version bump based on changes
# - Bug fixes only → PATCH (0.0.3 → 0.0.4)
# - New features → MINOR (0.0.3 → 0.1.0)
# - Breaking changes → MAJOR (0.0.3 → 1.0.0)

# Review unreleased changes
git log --oneline v0.0.3..HEAD
```

#### Quality Assurance
```bash
# Run comprehensive tests
./scripts/dev-test.sh

# Test local deployment
./scripts/deploy-test.sh

# Manual testing checklist
# - CLI commands work correctly
# - Search results are relevant
# - File watching detects changes
# - Configuration management works
# - Error handling is graceful
```

### 2. Release Preparation

#### Update Version
```python
# src/mcp_vector_search/__init__.py
__version__ = "0.0.4"
```

#### Update Documentation
```bash
# Update CHANGELOG.md
# Move items from [Unreleased] to new version section
# Add release date and version number

# Update README.md if needed
# - New features or usage examples
# - Updated installation instructions
# - Performance improvements

# Update API documentation if needed
# - New functions or classes
# - Changed interfaces
# - Deprecated features
```

#### Create Release Notes
```markdown
# Create RELEASE_NOTES.md for this version
## v0.0.4 - Bug Fixes & Improvements

### 🐛 Bug Fixes
- Fixed handling of empty files during indexing
- Resolved search result ordering issues
- Improved error messages for invalid file types

### 🚀 Improvements
- Enhanced file watching performance
- Better memory usage during large indexing operations
- Improved CLI output formatting

### 📖 Documentation
- Updated API documentation
- Added troubleshooting guide
- Improved installation instructions

### 🙏 Contributors
Thanks to all contributors who helped with this release!
```

### 3. Release Execution

#### Commit and Tag
```bash
# Commit version changes
git add src/mcp_vector_search/__init__.py docs/CHANGELOG.md
git commit -m "bump: version 0.0.4

- Updated version to 0.0.4
- Updated changelog with release notes
- Prepared for release"

# Create annotated tag
git tag -a v0.0.4 -m "Release v0.0.4 - Bug Fixes & Improvements

See CHANGELOG.md for detailed changes."

# Push changes and tags
git push origin main
git push origin v0.0.4
```

#### Publish to PyPI
```bash
# Use automated script
./scripts/publish.sh

# Or manual process:
# 1. Clean previous builds
rm -rf dist/ build/ *.egg-info

# 2. Build package
uv run python -m build

# 3. Check package
uv run twine check dist/*

# 4. Upload to PyPI
uv run twine upload dist/*
```

#### Create GitHub Release
```bash
# Using GitHub CLI
gh release create v0.0.4 \
  --title "v0.0.4 - Bug Fixes & Improvements" \
  --notes-file RELEASE_NOTES.md \
  --prerelease  # For alpha/beta releases

# Or manually on GitHub web interface
# 1. Go to repository releases page
# 2. Click "Create a new release"
# 3. Select tag v0.0.4
# 4. Add release title and notes
# 5. Mark as pre-release if applicable
# 6. Publish release
```

### 4. Post-Release Activities

#### Verification
```bash
# Wait 2-3 minutes for PyPI to process
# Test installation from PyPI
pip install mcp-vector-search==0.0.4 --upgrade

# Verify functionality
mcp-vector-search version
mcp-vector-search --help

# Test in clean environment
python -m venv test-env
source test-env/bin/activate
pip install mcp-vector-search==0.0.4
mcp-vector-search version
```

#### Communication
```bash
# Update project status
# - GitHub repository description
# - PyPI project description
# - Documentation badges

# Announce release
# - GitHub Discussions
# - Social media (Twitter, LinkedIn)
# - Developer communities (Reddit, Discord)
# - Email to interested users
```

---

## 📋 Release Checklists

### Pre-Release Checklist
- [ ] All tests pass (`./scripts/dev-test.sh`)
- [ ] Local deployment works (`./scripts/deploy-test.sh`)
- [ ] Manual testing completed
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Release notes prepared
- [ ] Breaking changes documented
- [ ] Migration guide created (if needed)

### Release Checklist
- [ ] Version committed and tagged
- [ ] Package built successfully
- [ ] Package uploaded to PyPI
- [ ] GitHub release created
- [ ] Release notes published
- [ ] Installation verified
- [ ] Functionality tested

### Post-Release Checklist
- [ ] PyPI installation verified
- [ ] GitHub release visible
- [ ] Documentation updated
- [ ] Community notified
- [ ] Next version planning started

---

## 🎨 Release Notes Best Practices

### Structure
```markdown
## v0.0.4 - Release Title

### 🎉 Highlights
Brief summary of major changes or achievements.

### ✨ New Features
- Feature 1: Description with user benefit
- Feature 2: Description with usage example

### 🐛 Bug Fixes
- Fix 1: What was broken and how it's fixed
- Fix 2: Impact on user experience

### 🚀 Improvements
- Performance improvement with metrics
- UX enhancement with before/after

### 📖 Documentation
- New guides or tutorials
- Updated API documentation

### ⚠️ Breaking Changes
- Clear description of what changed
- Migration instructions
- Timeline for deprecation

### 🙏 Contributors
Recognition of contributors and community members.
```

### Writing Guidelines

#### User-Focused Language
```markdown
# Good: User-focused
- Added support for Go language parsing
- Fixed search results not showing for large files
- Improved indexing speed by 50%

# Bad: Technical jargon
- Implemented Go AST parser
- Resolved buffer overflow in file reader
- Optimized vector embedding generation
```

#### Clear Impact Description
```markdown
# Good: Clear impact
- **Search Accuracy**: Improved semantic search relevance by 30%
- **Performance**: Reduced memory usage during indexing by 40%
- **Usability**: Added progress bars for long-running operations

# Bad: Vague description
- Various improvements
- Bug fixes
- Performance enhancements
```

#### Examples and Context
```markdown
# Good: With examples
- **New Language Support**: Added Rust parsing with support for `impl` blocks and `trait` definitions
  ```bash
  mcp-vector-search index --language rust
  ```

# Bad: Without context
- Added Rust support
```

---

## 🏷️ Release Types

### Alpha Releases (0.0.x)
**Purpose**: Early feedback, rapid iteration
**Frequency**: Weekly or bi-weekly
**Stability**: Unstable, breaking changes expected

**Release Notes Focus**:
- New experimental features
- Known limitations
- Feedback requests
- Breaking changes

**Example**: v0.0.3, v0.0.4, v0.0.5

### Beta Releases (0.x.0)
**Purpose**: Feature freeze, stabilization
**Frequency**: Monthly
**Stability**: Mostly stable, minor breaking changes possible

**Release Notes Focus**:
- Feature completeness
- Stability improvements
- Performance benchmarks
- Migration guides

**Example**: v0.1.0, v0.2.0 (planned)

### Stable Releases (1.x.x)
**Purpose**: Production use
**Frequency**: Quarterly for major, monthly for minor
**Stability**: Stable, backward compatibility guaranteed

**Release Notes Focus**:
- Reliability improvements
- Security updates
- Long-term support information
- Enterprise features

**Example**: v1.0.0, v1.1.0 (future)

### Patch Releases (x.x.Z)
**Purpose**: Critical bug fixes
**Frequency**: As needed
**Stability**: Same as base version

**Release Notes Focus**:
- Critical bug fixes
- Security patches
- Minimal changes
- Upgrade recommendations

**Example**: v1.0.1, v1.0.2 (future)

---

## 📊 Release Metrics

### Success Metrics
- **Download Count**: PyPI download statistics
- **GitHub Stars**: Community interest indicator
- **Issue Resolution**: Bug fix effectiveness
- **User Feedback**: Satisfaction and feature requests

### Quality Metrics
- **Test Coverage**: Percentage of code covered by tests
- **Bug Reports**: Number of issues opened post-release
- **Performance**: Benchmarks vs. previous versions
- **Documentation**: Completeness and clarity

### Tracking Tools
```bash
# PyPI download stats
pip install pypistats
pypistats recent mcp-vector-search

# GitHub metrics
gh api repos/bobmatnyc/mcp-vector-search \
  --jq '.stargazers_count, .forks_count, .open_issues_count'

# Release-specific metrics
gh api repos/bobmatnyc/mcp-vector-search/releases/latest \
  --jq '.download_count, .published_at'
```

---

## 🚨 Hotfix Process

### When to Hotfix
- Critical security vulnerabilities
- Data corruption bugs
- Complete feature failures
- Performance regressions > 50%

### Hotfix Workflow
```bash
# 1. Create hotfix branch from release tag
git checkout -b hotfix/v0.0.4-security v0.0.3

# 2. Apply minimal fix
# ... make changes ...

# 3. Test thoroughly
./scripts/dev-test.sh

# 4. Update version (patch increment)
# v0.0.3 → v0.0.4

# 5. Commit and tag
git commit -m "fix: critical security vulnerability"
git tag v0.0.4

# 6. Release immediately
./scripts/publish.sh
gh release create v0.0.4 --title "v0.0.4 - Security Fix"

# 7. Merge back to main
git checkout main
git merge hotfix/v0.0.4-security
```

---

## 📅 Release Schedule

### Current Schedule (Alpha Phase)
- **Patch releases**: As needed for critical fixes
- **Minor releases**: Every 2-4 weeks
- **Major releases**: When breaking changes accumulate

### Planned Schedule (Post-1.0)
- **Patch releases**: Monthly or as needed
- **Minor releases**: Quarterly
- **Major releases**: Annually

### Release Calendar
```
2024 Q1: v0.0.x → v0.1.0 (Beta)
2024 Q2: v0.1.x → v1.0.0 (Stable)
2024 Q3: v1.1.0 (Feature release)
2024 Q4: v1.2.0 (Feature release)
```

---

## 🔧 Automation Opportunities

### GitHub Actions (Future)
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and publish to PyPI
        run: ./scripts/publish.sh
      - name: Create GitHub release
        run: gh release create ${{ github.ref_name }} --generate-notes
```

### Automated Changelog
```bash
# Generate changelog from conventional commits
npx conventional-changelog-cli -p angular -i CHANGELOG.md -s

# Or use GitHub's auto-generated release notes
gh release create v0.0.4 --generate-notes
```

---

## 📚 Resources

### Tools
- **[GitHub CLI](https://cli.github.com/)** - Release management
- **[Twine](https://twine.readthedocs.io/)** - PyPI uploads
- **[bump2version](https://github.com/c4urself/bump2version)** - Version management
- **[conventional-changelog](https://github.com/conventional-changelog/conventional-changelog)** - Automated changelogs

### Best Practices
- **[Semantic Release](https://semantic-release.gitbook.io/)** - Automated releases
- **[Keep a Changelog](https://keepachangelog.com/)** - Changelog format
- **[GitHub Release Guide](https://docs.github.com/en/repositories/releasing-projects-on-github)** - GitHub releases
- **[PyPI Publishing Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)** - Python packaging
