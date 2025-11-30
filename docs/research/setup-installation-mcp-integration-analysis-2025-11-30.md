# Setup/Installation Flow and Claude Code MCP Integration Research

**Date**: 2025-11-30
**Project**: mcp-vector-search
**Focus**: Current setup/install flow, MCP integration, system detection capabilities

---

## Executive Summary

mcp-vector-search has evolved a sophisticated setup system with three primary entry points:

1. **`setup` command** - Zero-config smart setup (recommended)
2. **`install` command** - Advanced manual setup with MCP integration
3. **`init` command** - Basic initialization (legacy)

The tool supports MCP integration with 5 platforms (Claude Code, Claude Desktop, Cursor, Windsurf, VS Code) using a unified configuration approach. System detection capabilities exist for **project languages** and **MCP platforms**, but **OS-level detection** is minimal (only basic platform checks via `pathlib.Path`).

---

## 1. Current Setup/Installation Workflow

### 1.1 Primary Entry Point: `setup` Command

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py`

**Zero-Config Smart Setup** - The recommended installation method:

```bash
mcp-vector-search setup [--force] [--verbose]
```

**Workflow Steps** (lines 230-444):

1. **Detection & Analysis Phase** (lines 243-300)
   - Detects project root (via `ProjectManager`)
   - Detects programming languages (`project_manager.detect_languages()`)
   - Scans for file extensions with **2-second timeout** (`scan_project_file_extensions()`, lines 84-148)
   - Detects installed MCP platforms (`detect_installed_platforms()`)

2. **Smart Configuration Phase** (lines 303-323)
   - Selects file extensions (detected or defaults)
   - Chooses optimal embedding model (`select_optimal_embedding_model()`, lines 150-168)
   - Sets similarity threshold (default: 0.5)

3. **Initialization Phase** (lines 325-339)
   - Calls `project_manager.initialize()` with selected settings
   - Creates `.mcp-vector-search/` directory
   - Saves `config.json`

4. **Indexing Phase** (lines 342-360)
   - Runs `run_indexing()` from index command
   - Displays progress
   - **Non-blocking**: Continues even if indexing fails

5. **MCP Integration Phase** (lines 363-404)
   - Configures detected platforms (or fallback to `claude-code`)
   - Calls `configure_platform()` for each detected platform
   - **Graceful degradation**: Logs failures but continues

6. **Completion** (lines 407-443)
   - Displays summary of configured components
   - Shows next steps

**Key Features**:
- **Idempotent**: Safe to run multiple times
- **Fast**: Timeout-protected file scanning (2 seconds max)
- **Team-friendly**: Creates `.mcp.json` for version control
- **Zero input required**

---

### 1.2 Advanced Entry Point: `install` Command

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

**Manual Setup** - For users needing fine-grained control:

```bash
mcp-vector-search install [OPTIONS]
# OR
mcp-vector-search install <platform>
```

**Main Command** (lines 270-446):
```bash
mcp-vector-search install \
  --extensions .py,.js,.ts \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --similarity-threshold 0.5 \
  --auto-index \
  --with-mcp \
  --force
```

**Workflow**:
1. Check if already initialized (unless `--force`)
2. Parse file extensions from `--extensions` or use defaults
3. Initialize project with `ProjectManager.initialize()`
4. Auto-index if `--auto-index` (default: True)
5. Install MCP integrations if `--with-mcp`

**Platform-Specific Subcommands** (lines 454-625):
- `install claude-code` - Project-scoped (`.mcp.json`)
- `install cursor` - Global (`~/.cursor/mcp.json`)
- `install windsurf` - Global (`~/.codeium/windsurf/mcp_config.json`)
- `install claude-desktop` - Global (`~/Library/Application Support/Claude/claude_desktop_config.json`)
- `install vscode` - Global (`~/.vscode/mcp.json`)

**Platform Configuration** (lines 83-114):
```python
SUPPORTED_PLATFORMS = {
    "claude-code": {
        "name": "Claude Code",
        "config_path": ".mcp.json",  # Project-scoped
        "description": "Claude Code with project-scoped configuration",
        "scope": "project",
    },
    "claude-desktop": {
        "name": "Claude Desktop",
        "config_path": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "scope": "global",
    },
    # ... other platforms
}
```

---

### 1.3 Legacy Entry Point: `init` Command

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py` (not shown in full)

**Basic Initialization** - Minimal setup:

```bash
mcp-vector-search init [--extensions .py,.js] [--force]
```

**Workflow**:
1. Create `.mcp-vector-search/` directory
2. Initialize vector database
3. Save `config.json`
4. **No automatic indexing**
5. **No MCP integration**

**Note**: This is the simplest option but requires manual follow-up steps.

---

## 2. Claude Code MCP Integration Details

### 2.1 Native Claude Code Command Format

**CRITICAL**: The current implementation uses a **custom JSON-based approach**, NOT the native Claude Code CLI command.

**Expected Native Format** (from user requirement):
```bash
claude mcp add {label} {command path} {arguments}
```

**Current Implementation** (lines 187-194 in `install.py`):
```python
config["mcpServers"][server_name] = {
    "type": "stdio",
    "command": "uv",
    "args": ["run", "mcp-vector-search", "mcp"],
    "env": {
        "MCP_ENABLE_FILE_WATCHING": "true" if enable_watch else "false"
    },
}
```

**Gap**: The tool creates `.mcp.json` manually instead of using `claude mcp add` CLI.

---

### 2.2 MCP Server Configuration

**Configuration Generator** (lines 140-173):

```python
def get_mcp_server_config(
    project_root: Path,
    platform: str,
    enable_watch: bool = True,
) -> dict[str, Any]:
    """Generate MCP server configuration for a platform."""
    config: dict[str, Any] = {
        "command": "uv",
        "args": ["run", "mcp-vector-search", "mcp"],
        "env": {
            "MCP_ENABLE_FILE_WATCHING": "true" if enable_watch else "false",
        },
    }

    # Platform-specific adjustments
    if platform in ("claude-code", "cursor", "windsurf", "vscode"):
        config["type"] = "stdio"

    # Only add cwd for global-scope platforms
    if SUPPORTED_PLATFORMS[platform]["scope"] == "global":
        config["cwd"] = str(project_root.absolute())

    return config
```

**Key Points**:
- Uses `uv run` for compatibility (not direct Python path)
- Adds `"type": "stdio"` for Claude Code
- Includes `cwd` for global platforms (not project-scoped)
- File watching controlled via environment variable

---

### 2.3 MCP Configuration Files by Platform

| Platform | Scope | Config Path | Format |
|----------|-------|-------------|--------|
| Claude Code | Project | `.mcp.json` | JSON |
| Claude Desktop | Global | `~/Library/Application Support/Claude/claude_desktop_config.json` | JSON |
| Cursor | Global | `~/.cursor/mcp.json` | JSON |
| Windsurf | Global | `~/.codeium/windsurf/mcp_config.json` | JSON |
| VS Code | Global | `~/.vscode/mcp.json` | JSON |

**Configuration Structure** (all platforms):
```json
{
  "mcpServers": {
    "mcp-vector-search": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "mcp-vector-search", "mcp"],
      "env": {
        "MCP_ENABLE_FILE_WATCHING": "true"
      }
    }
  }
}
```

---

### 2.4 MCP Integration Workflow

**Platform Configuration** (lines 198-263):

```python
def configure_platform(
    platform: str,
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
    force: bool = False,
) -> bool:
    """Configure MCP integration for a specific platform."""
    try:
        config_path = get_platform_config_path(platform, project_root)

        # Create backup if file exists
        if config_path.exists():
            backup_path = config_path.with_suffix(config_path.suffix + ".backup")
            shutil.copy2(config_path, backup_path)

            # Load existing config
            with open(config_path) as f:
                config = json.load(f)

            # Check if server already exists
            if "mcpServers" in config and server_name in config["mcpServers"]:
                if not force:
                    # Warn and return False
                    return False
        else:
            # Create new config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {}

        # Add server configuration
        server_config = get_mcp_server_config(project_root, platform, enable_watch)
        config["mcpServers"][server_name] = server_config

        # Write configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Failed to configure {platform}: {e}")
        return False
```

**Safety Features**:
- Creates `.backup` files before modification
- Checks for existing server configuration
- Requires `--force` to overwrite
- Creates config directories if missing

---

## 3. System Detection Capabilities

### 3.1 Language Detection

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py`

**Method**: `detect_languages()` (line 226)

```python
def detect_languages(self) -> list[str]:
    """Detect programming languages in the project."""
    # Implementation details not shown in excerpts
    # Returns: List of detected language names (e.g., ["Python", "JavaScript", "TypeScript"])
```

**Usage**:
- Called by `setup` command during detection phase
- Used to select optimal embedding model
- Based on file extension analysis

---

### 3.2 Platform Detection

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

**Method**: `detect_installed_platforms()` (lines 176-195)

```python
def detect_installed_platforms() -> dict[str, Path]:
    """Detect which MCP platforms are installed on the system."""
    detected = {}

    for platform, info in SUPPORTED_PLATFORMS.items():
        # For project-scoped platforms, always include them
        if info["scope"] == "project":
            detected[platform] = Path(info["config_path"])
            continue

        # For global platforms, check if config directory exists
        config_path = Path(info["config_path"]).expanduser()
        if config_path.parent.exists():
            detected[platform] = config_path

    return detected
```

**Detection Strategy**:
- **Project-scoped** (Claude Code): Always included
- **Global platforms**: Checks if config directory exists (not the file itself)
- **Example**: If `~/.cursor/` exists, Cursor is "detected"

**Limitation**: This is a **directory existence check**, not a true installation check.

---

### 3.3 File Extension Scanning

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py`

**Method**: `scan_project_file_extensions()` (lines 84-148)

```python
def scan_project_file_extensions(
    project_root: Path,
    timeout: float = 2.0,
) -> list[str] | None:
    """Scan project for unique file extensions with timeout."""
    extensions: set[str] = set()
    start_time = time.time()
    file_count = 0

    try:
        project_manager = ProjectManager(project_root)

        for path in project_root.rglob("*"):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.debug(f"File extension scan timed out after {timeout}s")
                return None

            # Skip directories and ignored paths
            if not path.is_file():
                continue

            if project_manager._should_ignore_path(path, is_directory=False):
                continue

            # Get extension
            ext = path.suffix
            if ext:
                language = get_language_from_extension(ext)
                if language != "text" or ext in [".txt", ".md", ".rst"]:
                    extensions.add(ext)

            file_count += 1

        return sorted(extensions) if extensions else None

    except Exception as e:
        logger.debug(f"File extension scan failed: {e}")
        return None
```

**Key Features**:
- **Timeout protection**: 2-second max (configurable)
- **Respects gitignore**: Uses `ProjectManager._should_ignore_path()`
- **Language filtering**: Only includes known extensions
- **Graceful degradation**: Returns `None` on timeout/error

---

### 3.4 OS Detection

**Current State**: **Minimal OS-level detection**

**Evidence**:
1. **Path expansion**: Uses `Path.expanduser()` for `~` (works cross-platform)
2. **Platform-specific paths**: Hardcoded macOS paths in `SUPPORTED_PLATFORMS`:
   ```python
   "config_path": "~/Library/Application Support/Claude/claude_desktop_config.json"
   ```
3. **No explicit OS checks**: No usage of `platform.system()` or `sys.platform`

**Gap**: The tool assumes macOS paths for Claude Desktop. Windows/Linux paths would differ:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

---

## 4. Documentation Coverage

### 4.1 README.md

**File**: `/Users/masa/Projects/mcp-vector-search/README.md`

**Setup Documentation** (lines 66-143):
- ✅ Covers `setup` command (lines 66-109)
- ✅ Covers `install` command (lines 111-156)
- ✅ Lists all MCP platforms (lines 129-143)
- ✅ Shows zero-config workflow (lines 68-100)
- ✅ Explains team-friendly `.mcp.json` (lines 91-92, 232)

**MCP Integration** (lines 129-156):
```bash
# Add Claude Code integration (project-scoped)
mcp-vector-search install claude-code

# Add Cursor IDE integration (global)
mcp-vector-search install cursor

# Add Claude Desktop integration (global)
mcp-vector-search install claude-desktop

# See all available platforms
mcp-vector-search install list
```

---

### 4.2 Installation Documentation

**File**: `/Users/masa/Projects/mcp-vector-search/docs/getting-started/installation.md`

**Coverage**:
- ✅ System requirements (lines 1-27)
- ✅ Quick installation (lines 30-68)
- ✅ Zero-config setup (lines 70-103)
- ✅ Advanced setup options (lines 105-143)
- ✅ MCP integration commands (lines 145-189)
- ✅ Verification steps (lines 197-217)
- ✅ Troubleshooting (lines 377-442)

**MCP Integration Section** (lines 145-195):
```bash
# Add MCP Integration
mcp-vector-search install claude-code    # Project-scoped
mcp-vector-search install cursor         # Global
mcp-vector-search install claude-desktop # Global
mcp-vector-search install windsurf       # Global
mcp-vector-search install vscode         # Global

# See all available platforms
mcp-vector-search install list

# Remove MCP Integration
mcp-vector-search uninstall claude-code
mcp-vector-search uninstall --all
mcp-vector-search uninstall list
```

---

### 4.3 MCP Integration Guide

**File**: `/Users/masa/Projects/mcp-vector-search/docs/guides/mcp-integration.md`

**Coverage**:
- ✅ Overview (lines 1-13)
- ❌ **Outdated commands**: References `mcp install` instead of `install claude-code` (lines 14-46)
- ✅ MCP commands reference (lines 48-104)
- ✅ Available MCP tools (lines 120-163)
- ✅ Usage examples (lines 165-210)
- ✅ Troubleshooting (lines 211-245)
- ✅ Configuration (lines 247-255)
- ✅ Security considerations (lines 257-262)

**Gap**: Documentation uses deprecated `mcp` subcommand approach instead of direct `install <platform>`.

---

## 5. Current Gaps & Opportunities

### 5.1 Claude Code CLI Integration Gap

**Current**: Manual `.mcp.json` creation via JSON manipulation

**Expected**: Native Claude Code CLI usage
```bash
claude mcp add mcp-vector-search uv run mcp-vector-search mcp
```

**Impact**:
- Custom JSON approach works but bypasses official Claude tooling
- May not benefit from Claude Code's validation/error handling
- Could break with future Claude Code updates

**Recommendation**: Investigate `claude mcp add` command and migrate if available.

---

### 5.2 OS Detection Gaps

**Current**: Hardcoded macOS paths for Claude Desktop

**Missing**:
- Windows path detection (`%APPDATA%`)
- Linux path detection (`~/.config/`)
- Cross-platform path resolution

**Recommendation**: Add OS detection logic:
```python
import platform

def get_claude_desktop_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":  # macOS
        return Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser()
    elif system == "Windows":
        return Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path("~/.config/claude/claude_desktop_config.json").expanduser()
    else:
        raise ValueError(f"Unsupported OS: {system}")
```

---

### 5.3 Platform Detection Improvements

**Current**: Directory existence checks (not true installation checks)

**Enhancement**: Verify actual installation:
```python
def is_claude_code_installed() -> bool:
    """Check if Claude Code CLI is actually installed."""
    return shutil.which("claude") is not None

def is_cursor_installed() -> bool:
    """Check if Cursor is installed (macOS)."""
    cursor_app = Path("/Applications/Cursor.app")
    return cursor_app.exists()
```

---

### 5.4 Documentation Inconsistencies

**Issues**:
1. **MCP integration guide** uses deprecated `mcp install` command
2. **README** shows modern `install claude-code` but doesn't explain migration
3. **No migration guide** for users upgrading from old `mcp` subcommands

**Recommendation**: Update docs to consistently show new `install <platform>` approach.

---

## 6. File Locations Summary

### 6.1 Setup/Install Code

| Component | File Path | Lines |
|-----------|-----------|-------|
| Setup command | `src/mcp_vector_search/cli/commands/setup.py` | 1-448 |
| Install command | `src/mcp_vector_search/cli/commands/install.py` | 1-679 |
| MCP command (deprecated) | `src/mcp_vector_search/cli/commands/mcp.py` | 1-1183 |
| Init command | `src/mcp_vector_search/cli/commands/init.py` | (not analyzed) |

---

### 6.2 Core Infrastructure

| Component | File Path | Key Functions |
|-----------|-----------|---------------|
| Project Manager | `src/mcp_vector_search/core/project.py` | `detect_languages()`, `_detect_project_root()` |
| Database | `src/mcp_vector_search/core/database.py` | `_detect_and_recover_corruption()` |
| Indexer | `src/mcp_vector_search/core/indexer.py` | Uses monorepo detection |
| Monorepo Detector | `src/mcp_vector_search/utils/monorepo.py` | `detect_subprojects()` |

---

### 6.3 Documentation

| Document | File Path | Coverage |
|----------|-----------|----------|
| README | `README.md` | Setup, install, MCP integration |
| Installation Guide | `docs/getting-started/installation.md` | Detailed setup steps |
| MCP Integration | `docs/guides/mcp-integration.md` | MCP commands (some outdated) |
| Package Config | `pyproject.toml` | Dependencies, entry points |

---

## 7. Command Examples with Line References

### 7.1 Zero-Config Setup

**Command**:
```bash
mcp-vector-search setup --verbose
```

**Flow** (`setup.py`):
1. Line 263: `languages = project_manager.detect_languages()`
2. Line 275: `detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)`
3. Line 289: `detected_platforms = detect_installed_platforms()`
4. Line 330: `project_manager.initialize(...)`
5. Line 350: `await run_indexing(...)`
6. Line 377: `success = configure_platform(...)`

---

### 7.2 Advanced Install with MCP

**Command**:
```bash
mcp-vector-search install --with-mcp --extensions .py,.js,.ts
```

**Flow** (`install.py`):
1. Line 346: Check project root from context
2. Line 357: Check if already initialized
3. Line 366: Parse extensions
4. Line 383: `project_manager.initialize(...)`
5. Line 397: Run indexing if `--auto-index`
6. Line 411: `with_mcp` triggers platform configuration
7. Line 412: `detected = detect_installed_platforms()`
8. Line 416: `configure_platform(platform, project_root, enable_watch=True)`

---

### 7.3 Platform-Specific Install

**Command**:
```bash
mcp-vector-search install claude-code --watch
```

**Flow** (`install.py`):
1. Line 454: `@install_app.command("claude-code")`
2. Line 474: Get project root
3. Line 484: `success = configure_platform("claude-code", project_root, enable_watch=True)`
4. Lines 187-194: Configuration created in `.mcp.json`

---

## 8. System Detection in Detail

### 8.1 "Detected Systems" Interpretation

**Context**: The requirement mentions "detected systems" but doesn't clarify what "system" means.

**Possible Interpretations**:

1. **Operating System** (macOS, Windows, Linux)
   - **Current**: Minimal (only `Path.expanduser()`)
   - **Evidence**: Hardcoded macOS paths in `SUPPORTED_PLATFORMS`

2. **MCP Platforms** (Claude Code, Cursor, etc.)
   - **Current**: Basic directory checks (`detect_installed_platforms()`)
   - **Implementation**: Lines 176-195 in `install.py`

3. **Programming Languages** (Python, JavaScript, TypeScript)
   - **Current**: Full implementation (`project_manager.detect_languages()`)
   - **Implementation**: Line 226 in `core/project.py`

4. **Monorepo Systems** (npm workspaces, Lerna, pnpm, nx)
   - **Current**: Comprehensive detection
   - **Implementation**: `utils/monorepo.py` (multiple detection methods)

**Most Likely**: "Detected systems" refers to **MCP platforms** and **programming languages** in the context of the setup command.

---

### 8.2 System Detection Flow Diagram

```
┌─────────────────────────────────────────────┐
│         mcp-vector-search setup             │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────┐
│    1. Detect Project Characteristics       │
│    - Languages (Python, JS, TS, etc.)      │
│    - File extensions (.py, .js, .ts)       │
│    - Project root (git, package.json)      │
└───────────────┬────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────┐
│    2. Detect MCP Platforms                 │
│    - Check ~/.cursor/ exists → Cursor      │
│    - Check ~/.codeium/windsurf/ → Windsurf │
│    - Always include Claude Code (project)  │
└───────────────┬────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────┐
│    3. Select Optimal Configuration         │
│    - Embedding model (code vs general)     │
│    - File extensions (detected or defaults)│
│    - Similarity threshold (0.5 default)    │
└───────────────┬────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────┐
│    4. Initialize + Index                   │
│    - Create .mcp-vector-search/            │
│    - Index codebase                        │
└───────────────┬────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────┐
│    5. Configure Detected MCP Platforms     │
│    - Create .mcp.json (Claude Code)        │
│    - Update ~/.cursor/mcp.json (Cursor)    │
│    - Update other platform configs         │
└────────────────────────────────────────────┘
```

---

## 9. Recommendations

### 9.1 Immediate Improvements

1. **Add OS Detection**
   - Implement cross-platform path resolution for Claude Desktop
   - Use `platform.system()` to handle Windows/Linux

2. **Enhance Platform Detection**
   - Check for actual executables (`shutil.which("claude")`)
   - Verify application installation (not just config directories)

3. **Update Documentation**
   - Replace deprecated `mcp install` with `install <platform>`
   - Add migration guide for legacy commands
   - Document cross-platform paths

---

### 9.2 Future Enhancements

1. **Native Claude CLI Integration**
   - Investigate `claude mcp add` command
   - Migrate from manual JSON to native CLI if available

2. **Advanced System Detection**
   - Detect Python environment (venv, conda, pipenv)
   - Detect package managers (pip, uv, poetry)
   - Detect CI/CD platforms (GitHub Actions, GitLab CI)

3. **Configuration Validation**
   - Add `setup verify` command
   - Check MCP server health
   - Validate embedding model downloads

---

## 10. Conclusion

**Current State**:
- **Strong**: Zero-config setup, language detection, monorepo support
- **Good**: MCP integration with 5 platforms, graceful degradation
- **Weak**: OS detection, native Claude CLI usage, platform verification

**Setup Flow**:
1. `setup` (recommended) → Auto-detect everything, one command
2. `install` (advanced) → Manual control, MCP integration
3. `init` (legacy) → Minimal setup, requires follow-up

**MCP Integration**:
- Custom JSON-based configuration (not native `claude mcp add`)
- Supports 5 platforms with project/global scoping
- File watching controlled via environment variable
- Backup/restore safety for existing configs

**System Detection**:
- **Languages**: ✅ Full implementation
- **MCP Platforms**: ⚠️ Directory checks only
- **OS**: ❌ Minimal (hardcoded macOS paths)
- **Monorepos**: ✅ Comprehensive

**Key Gaps**:
1. No native `claude mcp add` usage
2. macOS-only paths for Claude Desktop
3. Platform detection via directory existence (not true installation checks)
4. Documentation inconsistencies (deprecated `mcp` subcommand)

---

## Appendix: Key Code Snippets

### A. Platform Configuration (install.py:187-194)

```python
config["mcpServers"][server_name] = {
    "type": "stdio",
    "command": "uv",
    "args": ["run", "mcp-vector-search", "mcp"],
    "env": {
        "MCP_ENABLE_FILE_WATCHING": "true" if enable_watch else "false"
    },
}
```

### B. Platform Detection (install.py:176-195)

```python
def detect_installed_platforms() -> dict[str, Path]:
    """Detect which MCP platforms are installed on the system."""
    detected = {}

    for platform, info in SUPPORTED_PLATFORMS.items():
        # For project-scoped platforms, always include them
        if info["scope"] == "project":
            detected[platform] = Path(info["config_path"])
            continue

        # For global platforms, check if config directory exists
        config_path = Path(info["config_path"]).expanduser()
        if config_path.parent.exists():
            detected[platform] = config_path

    return detected
```

### C. File Extension Scanning with Timeout (setup.py:84-148)

```python
def scan_project_file_extensions(
    project_root: Path,
    timeout: float = 2.0,
) -> list[str] | None:
    """Scan project for unique file extensions with timeout."""
    extensions: set[str] = set()
    start_time = time.time()

    try:
        project_manager = ProjectManager(project_root)

        for path in project_root.rglob("*"):
            # Check timeout
            if time.time() - start_time > timeout:
                return None

            # Skip ignored paths
            if project_manager._should_ignore_path(path, is_directory=False):
                continue

            # Collect extensions
            ext = path.suffix
            if ext:
                language = get_language_from_extension(ext)
                if language != "text" or ext in [".txt", ".md", ".rst"]:
                    extensions.add(ext)

        return sorted(extensions) if extensions else None
    except Exception:
        return None
```

---

**End of Research Document**
