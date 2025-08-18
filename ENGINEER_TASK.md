# Engineer Task: Fix Import Bug and Create Consolidated Versioning System

## Priority: CRITICAL - Immediate fixes needed

## Task 1: Fix Import Bug (IMMEDIATE)
**File**: `src/mcp_vector_search/core/factory.py`
- Change import from `EmbeddingFunction` to `CodeBERTEmbeddingFunction`
- This is causing immediate runtime failures

## Task 2: Fix Version Number (IMMEDIATE)
**File**: `src/mcp_vector_search/__init__.py`
- Set `__version__ = "4.0.3"`
- Set `__build__ = 280`

## Task 3: Create Comprehensive Makefile
Replace existing Makefile with complete workflow system.

### Core Development Targets
```makefile
help         # Show all available commands
dev          # Install for development (uv sync)
test         # Run full test suite
lint         # Run linting checks (ruff, mypy)
format       # Format code (black, isort)
clean        # Clean build artifacts
```

### Version Management Targets
```makefile
version-show    # Display current version
version-patch   # Bump patch (4.0.3 → 4.0.4)
version-minor   # Bump minor (4.0.3 → 4.1.0)
version-major   # Bump major (4.0.3 → 5.0.0)
build-increment # Increment build number only
```

### Release Workflow Targets
```makefile
release-patch   # Full release with patch bump
release-minor   # Full release with minor bump
release-major   # Full release with major bump
publish         # Publish to PyPI
publish-test    # Publish to test PyPI
```

### Required Features
- Color-coded output (green=success, yellow=warning, red=error)
- Pre-flight checks (clean git, tests pass)
- Dry-run mode: `DRY_RUN=1 make release-patch`
- Automatic git operations (commit, tag, push)
- Error handling and rollback
- Self-documenting with help text

## Task 4: Create Version Manager Script
**File**: `scripts/version_manager.py`

### Required Functions
```python
def read_version() -> tuple[str, int]:
    """Read version and build from __init__.py"""
    
def write_version(version: str, build: int):
    """Write version and build to __init__.py"""
    
def bump_version(current: str, bump_type: str) -> str:
    """Bump version according to semver"""
    
def update_changelog(version: str):
    """Add new version section to CHANGELOG.md"""
    
def git_operations(version: str, dry_run: bool):
    """Handle git commit, tag, push"""
```

### CLI Interface
```bash
python scripts/version_manager.py --bump patch
python scripts/version_manager.py --show
python scripts/version_manager.py --set 4.0.3 --build 280
python scripts/version_manager.py --dry-run
```

## Task 5: Update pyproject.toml
- Configure dynamic versioning from `__init__.py`
- Ensure proper build system setup
- Verify all metadata is complete

## Task 6: Add Deprecation Notices
Add notices to old scripts (don't delete):
- `scripts/build.sh` - Add header comment
- `scripts/dev-build.py` - Add runtime warning
- `scripts/publish.sh` - Add deprecation notice

## Implementation Requirements
1. Fix bugs FIRST (Tasks 1 & 2)
2. Test each component thoroughly
3. Maintain backward compatibility
4. Use clean code principles
5. Make everything self-documenting

## Success Criteria
- [ ] Import bug fixed and code runs
- [ ] Version shows as 4.0.3 build 280
- [ ] `make help` shows all commands
- [ ] `make release-patch` completes full cycle
- [ ] Version management is automated
- [ ] Old scripts have deprecation notices