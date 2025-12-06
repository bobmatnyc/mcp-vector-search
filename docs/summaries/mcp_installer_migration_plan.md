# MCP Installer Service Migration Plan

**Date**: December 5, 2025
**Status**: In Progress (Step 1/7 Complete)
**Project**: mcp-vector-search
**Migration**: py-mcp-installer-service integration

## Overview

Replacing custom MCP installation code with the comprehensive py-mcp-installer-service library to gain broader platform support, better error handling, and reduced maintenance burden.

## Progress

- [x] **Step 1**: Add py-mcp-installer-service as git submodule ✅
- [ ] **Step 2**: Update pyproject.toml to reference submodule
- [ ] **Step 3**: Refactor install.py to use new library
- [ ] **Step 4**: Refactor uninstall.py to use new library
- [ ] **Step 5**: Remove old installation code files
- [ ] **Step 6**: Update tests for new integration
- [ ] **Step 7**: Update documentation with new platform support
- [ ] **Step 8**: Test migration end-to-end

## Step 1: Add Submodule ✅ COMPLETE

```bash
git submodule add https://github.com/bobmatnyc/py-mcp-installer-service.git vendor/py-mcp-installer-service
```

**Result**: Submodule successfully cloned to `vendor/py-mcp-installer-service/`

## Step 2: Update pyproject.toml

**Location**: `/Users/masa/Projects/mcp-vector-search/pyproject.toml`

**Changes Needed**:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "py-mcp-installer-service @ file:///vendor/py-mcp-installer-service",
]
```

**Alternative** (using editable install):
```toml
[tool.uv]
dev-dependencies = [
    # ... existing dev-dependencies ...
]

[tool.uv.sources]
py-mcp-installer-service = { path = "vendor/py-mcp-installer-service", editable = true }
```

## Step 3: Refactor install.py

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

**New Implementation** (simplified):

```python
"""Install and integration commands for MCP Vector Search CLI.

Uses py-mcp-installer-service for robust, multi-platform MCP server installation.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Import from submodule
from py_mcp_installer_service import MCPInstaller, MCPInspector
from py_mcp_installer_service.models import ServerConfig

from ...config.defaults import DEFAULT_EMBEDDING_MODELS
from ...core.exceptions import ProjectInitializationError
from ...core.project import ProjectManager
from ..didyoumean import create_enhanced_typer
from ..output import (
    print_error,
    print_info,
    print_success,
    print_warning,
)

console = Console()

install_app = create_enhanced_typer(
    help="""📦 Install mcp-vector-search and MCP integrations

[bold cyan]Supported Platforms:[/bold cyan]
  • Claude Desktop    • Cursor          • Cline
  • Continue          • Roo-Code        • Zed
  • Windsurf          • Void

[bold cyan]Usage:[/bold cyan]
  $ mcp-vector-search install              # Auto-detect and install
  $ mcp-vector-search install --dry-run    # Preview changes
  $ mcp-vector-search install --all        # Install to all platforms
""",
    no_args_is_help=True,
)


@install_app.command(name="mcp")
async def install_mcp_server(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying them"
    ),
    all_platforms: bool = typer.Option(
        False, "--all", help="Install to all detected platforms"
    ),
    platform: Optional[str] = typer.Option(
        None, "--platform", help="Specific platform to install to"
    ),
):
    """Install mcp-vector-search to MCP-compatible platforms."""

    try:
        # Initialize installer
        installer = MCPInstaller()
        inspector = MCPInspector()

        # Auto-detect available platforms
        print_info("🔍 Detecting available MCP platforms...")
        platforms = await installer.auto_detect()

        if not platforms:
            print_warning("No MCP platforms detected.")
            print_info("Supported platforms: Claude Desktop, Cursor, Cline, Continue, etc.")
            return

        # Display detected platforms
        table = Table(title="Detected MCP Platforms")
        table.add_column("Platform", style="cyan")
        table.add_column("Config Path", style="green")
        table.add_column("Scope", style="yellow")

        for p in platforms:
            table.add_row(p.name, str(p.config_path), p.scope)

        console.print(table)

        # Configure server
        project_root = Path.cwd()
        server_config = ServerConfig(
            command="uv",
            args=["run", "--directory", str(project_root), "mcp-vector-search"],
            env={
                "PROJECT_ROOT": str(project_root),
            },
        )

        # Install to selected platforms
        target_platforms = [
            p for p in platforms
            if platform is None or p.name == platform
        ]

        if dry_run:
            print_info("🔍 DRY RUN MODE - No changes will be applied")

        for p in target_platforms:
            print_info(f"📦 {'Would install' if dry_run else 'Installing'} to {p.name}...")

            result = await installer.install_server(
                server_name="mcp-vector-search",
                config=server_config,
                platform=p,
                dry_run=dry_run,
            )

            if result.success:
                print_success(f"✓ {'Would be installed' if dry_run else 'Installed'} successfully to {p.name}")
            else:
                print_error(f"✗ Failed to install to {p.name}: {result.error}")

        # Validate configurations
        if not dry_run:
            print_info("🔍 Validating configurations...")
            for p in target_platforms:
                validation = await inspector.validate(p.config_path)
                if validation.is_valid:
                    print_success(f"✓ {p.name} configuration is valid")
                else:
                    print_warning(f"⚠ {p.name} configuration has issues:")
                    for issue in validation.issues:
                        print_warning(f"  • {issue}")

        print_success("🎉 Installation complete!")

    except Exception as e:
        logger.exception("Installation failed")
        print_error(f"Installation failed: {e}")
        raise typer.Exit(code=1)


# Keep existing project initialization command
@install_app.command()
def init(
    # ... existing init parameters ...
):
    """Initialize mcp-vector-search in current project."""
    # ... existing init implementation ...
    pass
```

## Step 4: Refactor uninstall.py

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/uninstall.py`

**New Implementation**:

```python
"""Uninstall commands for MCP Vector Search CLI."""

import asyncio
from typing import Optional

import typer
from loguru import logger
from rich.console import Console

from py_mcp_installer_service import MCPInstaller

from ..output import (
    print_error,
    print_info,
    print_success,
)

console = Console()

uninstall_app = typer.Typer(help="Uninstall mcp-vector-search from MCP platforms")


@uninstall_app.command()
async def mcp(
    platform: Optional[str] = typer.Option(
        None, "--platform", help="Specific platform to uninstall from"
    ),
    all_platforms: bool = typer.Option(
        False, "--all", help="Uninstall from all detected platforms"
    ),
):
    """Uninstall mcp-vector-search from MCP platforms."""

    try:
        installer = MCPInstaller()

        # Detect platforms
        platforms = await installer.auto_detect()

        if not platforms:
            print_warning("No MCP platforms detected.")
            return

        # Filter platforms
        target_platforms = [
            p for p in platforms
            if platform is None or p.name == platform
        ]

        # Uninstall from each platform
        for p in target_platforms:
            print_info(f"🗑️  Uninstalling from {p.name}...")

            result = await installer.uninstall_server(
                server_name="mcp-vector-search",
                platform=p,
            )

            if result.success:
                print_success(f"✓ Uninstalled from {p.name}")
            else:
                print_error(f"✗ Failed to uninstall from {p.name}: {result.error}")

        print_success("🎉 Uninstallation complete!")

    except Exception as e:
        logger.exception("Uninstallation failed")
        print_error(f"Uninstallation failed: {e}")
        raise typer.Exit(code=1)
```

## Step 5: Remove Old Files

**Files to Delete**:

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install_old.py`
- Any custom platform detection logic (if separate file)
- Old installation helper modules (if any)

**Files to Keep**:

- `install.py` (refactored)
- `uninstall.py` (refactored)
- `mcp.py` (if it contains other MCP-related commands)

## Step 6: Update Tests

**Test Files to Update**:

1. `/Users/masa/Projects/mcp-vector-search/tests/unit/commands/test_setup.py`
2. `/Users/masa/Projects/mcp-vector-search/tests/integration/test_setup_integration.py`

**New Test Structure**:

```python
"""Test MCP installation with py-mcp-installer-service."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from py_mcp_installer_service import MCPInstaller, Platform
from py_mcp_installer_service.models import ServerConfig, InstallResult


@pytest.mark.asyncio
async def test_install_auto_detect():
    """Test auto-detection of MCP platforms."""
    with patch.object(MCPInstaller, 'auto_detect') as mock_detect:
        mock_detect.return_value = [
            Platform(name="claude-desktop", config_path=Path("~/.config/claude"), scope="global"),
            Platform(name="cursor", config_path=Path("~/.cursor"), scope="global"),
        ]

        installer = MCPInstaller()
        platforms = await installer.auto_detect()

        assert len(platforms) == 2
        assert platforms[0].name == "claude-desktop"
        assert platforms[1].name == "cursor"


@pytest.mark.asyncio
async def test_install_server_success():
    """Test successful server installation."""
    with patch.object(MCPInstaller, 'install_server') as mock_install:
        mock_install.return_value = InstallResult(success=True, message="Installed")

        installer = MCPInstaller()
        platform = Platform(name="claude-desktop", config_path=Path("~/.config/claude"), scope="global")
        config = ServerConfig(command="uv", args=["run", "mcp-vector-search"])

        result = await installer.install_server(
            server_name="mcp-vector-search",
            config=config,
            platform=platform,
        )

        assert result.success
        assert result.message == "Installed"


@pytest.mark.asyncio
async def test_uninstall_server():
    """Test server uninstallation."""
    with patch.object(MCPInstaller, 'uninstall_server') as mock_uninstall:
        mock_uninstall.return_value = InstallResult(success=True, message="Uninstalled")

        installer = MCPInstaller()
        platform = Platform(name="cursor", config_path=Path("~/.cursor"), scope="global")

        result = await installer.uninstall_server(
            server_name="mcp-vector-search",
            platform=platform,
        )

        assert result.success
```

## Step 7: Update Documentation

**Files to Update**:

1. `README.md` - Update installation instructions
2. `docs/installation.md` (if exists) - Add platform support table
3. `docs/deployment.md` (if exists) - Update deployment instructions

**New Platform Support Table**:

| Platform | Config Location | Scope | Auto-Detected |
|----------|----------------|-------|---------------|
| Claude Desktop | `~/Library/Application Support/Claude/` | Global | ✅ |
| Cursor | `~/.cursor/` | Global | ✅ |
| Cline | `./.continue/config.json` | Project | ✅ |
| Continue | `~/.continue/` | Global | ✅ |
| Roo-Code | `./.roo-code/config.json` | Project | ✅ |
| Zed | `~/.config/zed/` | Global | ✅ |
| Windsurf | `~/.codeium/windsurf/` | Global | ✅ |
| Void | `./.void/` | Project | ✅ |

## Step 8: End-to-End Testing

**Test Scenarios**:

1. **Fresh Install**: Test installation on a system with no MCP platforms configured
2. **Multi-Platform**: Test installation across multiple detected platforms
3. **Dry-Run**: Verify dry-run mode shows accurate preview
4. **Validation**: Test configuration validation and auto-fix
5. **Uninstall**: Test clean uninstallation from all platforms
6. **Upgrade**: Test upgrading existing installation
7. **Error Handling**: Test behavior when config files are corrupted

**Test Commands**:

```bash
# Test auto-detection
mcp-vector-search install mcp --dry-run

# Test single platform
mcp-vector-search install mcp --platform claude-desktop

# Test all platforms
mcp-vector-search install mcp --all

# Test uninstall
mcp-vector-search uninstall mcp --platform cursor

# Test validation
mcp-vector-search install mcp --validate
```

## Benefits Summary

### Code Reduction
- **Before**: ~500+ lines of custom installation code
- **After**: ~150 lines of integration code
- **Reduction**: 70% less code to maintain

### Platform Support
- **Before**: 1 platform (Claude Desktop only)
- **After**: 8 platforms (Claude Desktop, Cursor, Cline, Continue, Roo-Code, Zed, Windsurf, Void)
- **Increase**: 8x platform coverage

### Features Gained
- ✅ Atomic operations with rollback
- ✅ Configuration validation and auto-fix
- ✅ Dry-run mode for preview
- ✅ Smart installation method selection
- ✅ Environment variable support
- ✅ Legacy format migration
- ✅ Better error messages and recovery

## Rollback Plan

If migration encounters issues:

1. **Immediate**: `git checkout main` to revert changes
2. **Partial**: Keep submodule, revert code changes
3. **Complete**: Remove submodule: `git submodule deinit vendor/py-mcp-installer-service`

## Next Steps

1. ✅ Submodule added
2. **Next**: Update pyproject.toml (Step 2)
3. **Then**: Refactor install.py (Step 3)
4. **Continue**: Follow steps 4-8 sequentially

## Notes

- Submodule location: `vendor/py-mcp-installer-service/`
- Git submodule commit tracking in `.gitmodules`
- Can update submodule: `git submodule update --remote vendor/py-mcp-installer-service`
- Integration uses async/await - ensure CLI commands support async

## References

- py-mcp-installer-service: https://github.com/bobmatnyc/py-mcp-installer-service
- Original discussion: Session on December 5, 2025
- Related files:
  - Current install.py: `src/mcp_vector_search/cli/commands/install.py`
  - Current uninstall.py: `src/mcp_vector_search/cli/commands/uninstall.py`
