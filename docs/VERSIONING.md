# Versioning Guidelines

## 🎯 Semantic Versioning

MCP Vector Search follows [Semantic Versioning 2.0.0](https://semver.org/) (SemVer) for version numbering. This ensures predictable and meaningful version numbers for users and developers.

---

## 📊 Version Format

### Structure
```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

### Examples
- `0.0.3` - Alpha release
- `0.1.0` - Beta release  
- `1.0.0` - Stable release
- `1.2.3` - Patch release
- `2.0.0-alpha.1` - Pre-release
- `1.0.0+20231201` - Build metadata

---

## 🔢 Version Components

### MAJOR Version (X.y.z)
**Increment when**: Making incompatible API changes

**Examples**:
- Changing CLI command structure
- Removing or renaming public APIs
- Changing configuration file format
- Breaking changes to search results format

```bash
# Example: v1.0.0 → v2.0.0
# Breaking change: CLI command restructure
mcp-vector-search search "query"     # v1.x
mcp-vector-search query "search"     # v2.x (breaking)
```

### MINOR Version (x.Y.z)
**Increment when**: Adding functionality in backward-compatible manner

**Examples**:
- Adding new CLI commands
- Adding new language parsers
- Adding new configuration options
- Adding new search features

```bash
# Example: v1.2.0 → v1.3.0
# New feature: Go language support
mcp-vector-search index --language go
```

### PATCH Version (x.y.Z)
**Increment when**: Making backward-compatible bug fixes

**Examples**:
- Fixing search accuracy issues
- Fixing file parsing errors
- Fixing CLI output formatting
- Performance improvements

```bash
# Example: v1.2.3 → v1.2.4
# Bug fix: Handle empty files gracefully
```

---

## 🚀 Release Stages

### Alpha Releases (0.0.x)
**Purpose**: Early development, experimental features
**Stability**: Unstable, breaking changes expected
**Audience**: Early adopters, contributors

**Characteristics**:
- Rapid iteration
- Breaking changes allowed
- Limited documentation
- Basic functionality

**Example**: `0.0.3` (current)

### Beta Releases (0.x.0)
**Purpose**: Feature-complete, stabilizing
**Stability**: Mostly stable, minor breaking changes possible
**Audience**: Early users, testers

**Characteristics**:
- Feature freeze
- Bug fixes and polish
- Comprehensive testing
- Complete documentation

**Example**: `0.1.0` (planned)

### Stable Releases (1.x.x)
**Purpose**: Production-ready
**Stability**: Stable, backward compatibility guaranteed
**Audience**: All users

**Characteristics**:
- Semantic versioning strictly followed
- Comprehensive testing
- Long-term support
- Migration guides for breaking changes

**Example**: `1.0.0` (future)

---

## 📋 Version Management Process

### 1. Version Planning
```bash
# Check current version
uv run python -c "from mcp_vector_search import __version__; print(__version__)"

# Plan next version based on changes
# - Bug fixes only → patch (0.0.3 → 0.0.4)
# - New features → minor (0.0.3 → 0.1.0)  
# - Breaking changes → major (0.0.3 → 1.0.0)
```

### 2. Version Update
```python
# Update version in src/mcp_vector_search/__init__.py
__version__ = "0.0.4"
```

### 3. Changelog Update
```markdown
# Update CHANGELOG.md
## [0.0.4] - 2024-01-15

### Fixed
- Handle empty files gracefully during indexing
- Fix search results ordering by similarity score

### Changed
- Improve error messages for invalid file types
```

### 4. Git Tagging
```bash
# Commit version changes
git add src/mcp_vector_search/__init__.py CHANGELOG.md
git commit -m "bump: version 0.0.4"

# Create and push tag
git tag v0.0.4
git push origin main --tags
```

### 5. Release Creation
```bash
# Publish to PyPI
./scripts/publish.sh

# Create GitHub release
gh release create v0.0.4 --title "v0.0.4 - Bug Fixes" --notes-file RELEASE_NOTES.md
```

---

## 📝 Changelog Management

### Format
We follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features in development

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Now removed features

### Fixed
- Bug fixes

### Security
- Vulnerability fixes

## [0.0.4] - 2024-01-15

### Fixed
- Handle empty files gracefully during indexing
- Fix search results ordering by similarity score

### Changed
- Improve error messages for invalid file types

## [0.0.3] - 2024-01-10

### Added
- Initial alpha release
- Python, JavaScript, TypeScript parsing
- Semantic search functionality
- File watching capabilities
- CLI interface with rich output
```

### Categories
- **Added**: New features
- **Changed**: Changes in existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

---

## 🔄 Release Workflow

### Pre-release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Breaking changes documented
- [ ] Migration guide created (if needed)

### Release Process
```bash
# 1. Update version and changelog
vim src/mcp_vector_search/__init__.py
vim CHANGELOG.md

# 2. Test release candidate
./scripts/dev-test.sh
./scripts/deploy-test.sh

# 3. Commit and tag
git add .
git commit -m "bump: version 0.0.4"
git tag v0.0.4

# 4. Publish
./scripts/publish.sh
git push origin main --tags

# 5. Create GitHub release
gh release create v0.0.4 --generate-notes
```

---

## 🎯 Version Strategy by Component

### CLI Interface
- **Major**: Command structure changes
- **Minor**: New commands or options
- **Patch**: Bug fixes, output improvements

### Core APIs
- **Major**: Breaking API changes
- **Minor**: New APIs, optional parameters
- **Patch**: Bug fixes, performance improvements

### Configuration
- **Major**: Breaking config format changes
- **Minor**: New configuration options
- **Patch**: Default value changes, validation improvements

### Language Support
- **Major**: Removing language support
- **Minor**: Adding new languages
- **Patch**: Improving existing parsers

---

## 📊 Backward Compatibility

### Compatibility Promise
- **Alpha (0.0.x)**: No compatibility guarantees
- **Beta (0.x.0)**: Best effort compatibility
- **Stable (1.x.x)**: Strict backward compatibility

### Breaking Change Guidelines
```markdown
# Breaking changes must include:
1. Clear documentation of the change
2. Migration guide with examples
3. Deprecation warnings (when possible)
4. Timeline for removal

# Example migration guide:
## Migration from v1.x to v2.x

### CLI Changes
**Old**: `mcp-vector-search search "query"`
**New**: `mcp-vector-search query "search"`

**Migration**: Update your scripts to use the new command structure.

### Configuration Changes
**Old**: `search.max_results`
**New**: `search.limit`

**Migration**: Rename the configuration key in your config file.
```

---

## 🔍 Version Detection

### Runtime Version Check
```python
# In code
from mcp_vector_search import __version__
print(f"MCP Vector Search v{__version__}")

# CLI
mcp-vector-search version
mcp-vector-search --version
```

### Programmatic Access
```python
import pkg_resources

def get_version():
    """Get installed package version."""
    try:
        return pkg_resources.get_distribution("mcp-vector-search").version
    except pkg_resources.DistributionNotFound:
        return "unknown"
```

---

## 📈 Version History

### Release Timeline
```
v0.0.1 - Initial prototype (not released)
v0.0.2 - Internal testing (not released)  
v0.0.3 - First public alpha (2024-01-10)
v0.0.4 - Bug fixes (planned)
v0.1.0 - Beta release (planned Q1 2024)
v1.0.0 - Stable release (planned Q2 2024)
```

### Version Milestones
- **v0.0.3**: First public release, basic functionality
- **v0.1.0**: Feature-complete beta, comprehensive testing
- **v1.0.0**: Production-ready, MCP integration
- **v2.0.0**: Advanced features, plugin system

---

## 🛠️ Development Versions

### Development Builds
```bash
# Development version format
0.0.4.dev0+g1234567  # Development build
0.0.4rc1             # Release candidate
```

### Version Bumping Tools
```bash
# Manual version update
vim src/mcp_vector_search/__init__.py

# Or use bump2version (optional)
pip install bump2version
bump2version patch  # 0.0.3 → 0.0.4
bump2version minor  # 0.0.3 → 0.1.0
bump2version major  # 0.0.3 → 1.0.0
```

---

## 📚 Resources

### Standards
- **[Semantic Versioning](https://semver.org/)** - Version numbering standard
- **[Keep a Changelog](https://keepachangelog.com/)** - Changelog format
- **[Calendar Versioning](https://calver.org/)** - Alternative versioning scheme

### Tools
- **[bump2version](https://github.com/c4urself/bump2version)** - Version bumping tool
- **[semantic-release](https://semantic-release.gitbook.io/)** - Automated releases
- **[conventional-changelog](https://github.com/conventional-changelog/conventional-changelog)** - Automated changelog

### Best Practices
- **[Python Packaging](https://packaging.python.org/guides/distributing-packages-using-setuptools/)** - Python packaging guide
- **[Release Engineering](https://en.wikipedia.org/wiki/Release_engineering)** - Release management practices
