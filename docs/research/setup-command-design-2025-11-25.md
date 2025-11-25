# Smart Setup Command Design Research

**Research Date**: 2025-11-25
**Research Type**: Command Design & Architecture
**Status**: Complete

## Executive Summary

This research analyzes the existing `init` and `install` commands to design a new zero-configuration `setup` command that combines both workflows intelligently without requiring user input.

**Key Findings**:
- `init` and `install` have **significant overlap** (80%+ functionality)
- Auto-detection is already **partially implemented** but underutilized
- MCP platform detection can be **fully automated**
- File extension detection already exists via `detect_languages()`
- Smart defaults enable **true zero-config experience**

**Recommendation**: Create `setup` command as a smart orchestrator that combines init + install + auto-detection with comprehensive logging.

---

## 1. Current Command Analysis

### 1.1 Init Command (`init.py`)

**Primary Function**: Initialize project for semantic search

**What It Does**:
1. ✅ Creates `.mcp-vector-search/` directory for vector database
2. ✅ Detects programming languages via `ProjectManager.detect_languages()`
3. ✅ Generates project configuration (file extensions, embedding model, thresholds)
4. ✅ Auto-indexes codebase (optional, default: enabled)
5. ✅ Sets up MCP integration for Claude Code (optional, default: enabled)
6. ✅ Configures auto-indexing/file watching (optional, default: enabled)
7. ✅ Creates `.mcp.json` in project root (project-scoped, shareable)

**Parameters & Defaults**:
```python
--extensions: str | None = None  # Default: DEFAULT_FILE_EXTENSIONS (auto-detected)
--embedding-model: str = "sentence-transformers/all-MiniLM-L6-v2"
--similarity-threshold: float = 0.5
--force: bool = False
--auto-index: bool = True  # Automatically index after init
--mcp: bool = True  # Install Claude Code MCP integration
--auto-indexing: bool = True  # Enable file watching
```

**User Choices Required**: None (all flags have sensible defaults)

**Auto-Detection Capabilities**:
- ✅ **Project root**: Detects via `.git`, `.mcp-vector-search`, `pyproject.toml`, `package.json`, etc.
- ✅ **Languages**: Scans file extensions to detect Python, JavaScript, TypeScript, etc.
- ✅ **File types**: Uses `DEFAULT_FILE_EXTENSIONS` if not specified
- ⚠️ **Partial**: Could scan actual files to auto-detect extensions in use

**Current Workflow**:
```
init
├─ Check if already initialized (exit if yes, unless --force)
├─ Parse file extensions (or use defaults)
├─ Initialize ProjectManager
│  ├─ Create .mcp-vector-search/ directory
│  ├─ Detect languages (scan file extensions)
│  ├─ Save config.json
│  └─ Update .gitignore
├─ Auto-index codebase (if --auto-index, default: true)
├─ Install MCP integration (if --mcp, default: true)
│  ├─ Create .mcp.json with Claude Code config
│  └─ Setup auto-indexing (if --auto-indexing)
└─ Display success + next steps
```

### 1.2 Install Command (`install.py`)

**Primary Function**: Install mcp-vector-search + MCP integrations for multiple platforms

**What It Does**:
1. ✅ Same initialization as `init` (creates vector DB, config)
2. ✅ Auto-indexes codebase (optional, default: enabled)
3. ✅ Detects installed MCP platforms (Claude Code, Cursor, Windsurf, etc.)
4. ✅ Configures MCP integration for **all detected platforms** (if `--with-mcp`)
5. ✅ Supports platform-specific subcommands (`install claude-code`, etc.)

**Parameters & Defaults**:
```python
--extensions: str | None = None
--embedding-model: str = "sentence-transformers/all-MiniLM-L6-v2"
--similarity-threshold: float = 0.5
--auto-index: bool = True
--with-mcp: bool = False  # ⚠️ Default: FALSE (different from init!)
--force: bool = False
```

**Platform Detection** (`detect_installed_platforms()`):
```python
SUPPORTED_PLATFORMS = {
    "claude-code": {"config_path": ".mcp.json", "scope": "project"},
    "claude-desktop": {"config_path": "~/Library/Application Support/Claude/...", "scope": "global"},
    "cursor": {"config_path": "~/.cursor/mcp.json", "scope": "global"},
    "windsurf": {"config_path": "~/.codeium/windsurf/mcp_config.json", "scope": "global"},
    "vscode": {"config_path": "~/.vscode/mcp.json", "scope": "global"},
}
```

**Detection Logic**:
- Project-scoped platforms (Claude Code): **Always include** (no filesystem check needed)
- Global platforms: Check if `config_path.parent.exists()`
- Returns: `dict[platform_name, config_path]`

**Current Workflow**:
```
install
├─ Check if already initialized
├─ Initialize project (same as init)
├─ Auto-index codebase (if --auto-index)
├─ Install MCP integrations (if --with-mcp)
│  ├─ Detect installed platforms
│  ├─ For each detected platform:
│  │  └─ Configure MCP server in platform config
│  └─ Handle errors gracefully
└─ Display success + next steps
```

### 1.3 Key Differences: Init vs Install

| Feature | `init` | `install` |
|---------|--------|-----------|
| **Primary Focus** | Single-platform setup (Claude Code) | Multi-platform installation |
| **MCP Default** | `--mcp=True` (enabled) | `--with-mcp=False` (disabled) |
| **Platform Support** | Claude Code only | Claude Code, Cursor, Windsurf, VS Code, Claude Desktop |
| **Detection** | No platform detection | Auto-detects installed platforms |
| **Command Style** | Simple, single command | Command with subcommands (`install claude-code`) |
| **User Intent** | "Set up this project" | "Install on specific platform(s)" |

**Overlap**: ~80% of code is identical (initialization, indexing, config)

---

## 2. Auto-Detection Opportunities

### 2.1 Already Implemented ✅

1. **Project Root Detection** (`ProjectManager._detect_project_root()`):
   - Walks up directory tree looking for:
     - `.git` (Git repository)
     - `.mcp-vector-search` (already initialized)
     - `pyproject.toml` (Python project)
     - `package.json` (Node.js project)
     - `Cargo.toml` (Rust project)
     - `go.mod` (Go project)
     - `pom.xml`, `build.gradle` (Java project)

2. **Language Detection** (`ProjectManager.detect_languages()`):
   - Scans all source files in project
   - Maps file extensions to languages via `LANGUAGE_MAPPINGS`
   - Returns sorted list of detected languages

3. **Platform Detection** (`detect_installed_platforms()`):
   - Checks filesystem for installed AI tools
   - Returns platforms with existing config directories

### 2.2 Can Be Enhanced 🔧

1. **File Extension Auto-Detection**:
   - Currently uses `DEFAULT_FILE_EXTENSIONS` (38 extensions)
   - Could scan project to find **actually used extensions**
   - Example: Python project might only need `.py`, `.pyi`, `.txt`, `.md`

2. **Git Repository Detection**:
   - Check if project is a Git repo (`(project_root / ".git").exists()`)
   - Auto-configure `.gitignore` patterns

3. **MCP Platform Priority**:
   - Detect which platform user likely wants based on:
     - Running processes (is Claude Code running?)
     - Recently modified config files
     - Environment variables

### 2.3 New Detection Capabilities 🆕

1. **Project Type Inference**:
   ```python
   def infer_project_type(project_root: Path) -> str:
       if (project_root / "pyproject.toml").exists():
           return "Python"
       elif (project_root / "package.json").exists():
           return "JavaScript/TypeScript"
       elif (project_root / "Cargo.toml").exists():
           return "Rust"
       # etc.
   ```

2. **Optimal Embedding Model Selection**:
   ```python
   OPTIMAL_MODELS = {
       "Python": "sentence-transformers/all-MiniLM-L6-v2",
       "JavaScript": "sentence-transformers/all-MiniLM-L6-v2",
       "Mixed": "sentence-transformers/all-mpnet-base-v2",  # More precise
   }
   ```

3. **Smart File Extension Pruning**:
   - Scan project for unique extensions
   - Only index extensions that **actually exist**
   - Reduces indexing time for specialized projects

---

## 3. Proposed Smart Setup Command

### 3.1 Design Principles

1. **Zero User Input**: No prompts, no required flags
2. **Intelligent Defaults**: Auto-detect everything possible
3. **Comprehensive Logging**: Show what's happening and why
4. **Idempotent**: Safe to run multiple times
5. **Graceful Degradation**: Continue on errors, log warnings
6. **Speed Optimized**: Parallel detection where possible

### 3.2 Workflow Design

```
setup (zero-config command)
│
├─ [Phase 1: Detection & Analysis] (parallel where possible)
│  ├─ Detect project root (auto)
│  ├─ Check if already initialized
│  │  └─ If yes: Skip init, go to MCP setup (unless --force)
│  ├─ Scan for file extensions in use (cache for 1min)
│  ├─ Detect languages (existing function)
│  ├─ Detect project type (Python/JS/Rust/etc.)
│  ├─ Check Git repository status
│  └─ Detect installed MCP platforms
│
├─ [Phase 2: Smart Configuration]
│  ├─ Select optimal embedding model based on project type
│  ├─ Determine file extensions to index
│  │  └─ Use detected extensions OR defaults if scan too slow
│  ├─ Calculate optimal similarity threshold
│  └─ Choose platforms to configure
│
├─ [Phase 3: Initialization]
│  ├─ Create .mcp-vector-search/ directory
│  ├─ Generate config with detected settings
│  ├─ Update .gitignore (if Git repo)
│  └─ Display configuration summary
│
├─ [Phase 4: Indexing]
│  ├─ Index codebase with progress bar
│  ├─ Show statistics (files indexed, languages found)
│  └─ Enable auto-indexing/file watching
│
├─ [Phase 5: MCP Integration]
│  ├─ For each detected platform:
│  │  ├─ Configure MCP server
│  │  ├─ Test configuration (basic validation)
│  │  └─ Report success/failure
│  └─ Prioritize Claude Code (project-scoped)
│
└─ [Phase 6: Completion]
   ├─ Display comprehensive summary
   ├─ Show what was configured
   ├─ Provide next steps
   └─ Suggest verification command
```

### 3.3 Smart Defaults Strategy

| Setting | Detection Method | Fallback |
|---------|-----------------|----------|
| **Project Root** | Walk up tree for indicators | `Path.cwd()` |
| **File Extensions** | Scan unique extensions in project | `DEFAULT_FILE_EXTENSIONS` |
| **Languages** | Existing `detect_languages()` | Empty list |
| **Embedding Model** | Based on project type | `all-MiniLM-L6-v2` |
| **Similarity Threshold** | Based on primary language | `0.5` |
| **MCP Platforms** | Detect via filesystem | Claude Code only |
| **Auto-Index** | Always enabled | `True` |
| **File Watching** | Always enabled | `True` |

### 3.4 Implementation Approach

**Option A: Wrapper Around Existing Commands** (Recommended)
```python
async def setup_command(
    ctx: typer.Context,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """🚀 Smart zero-config setup (combines init + install)."""

    # Phase 1: Detection
    print_info("🔍 Analyzing your project...")
    project_root = detect_project_root()
    platforms = detect_installed_platforms()
    languages = detect_languages(project_root)
    extensions = scan_file_extensions(project_root)  # NEW

    # Phase 2: Smart Configuration
    config = SmartConfig(
        project_root=project_root,
        extensions=extensions or DEFAULT_FILE_EXTENSIONS,
        languages=languages,
        platforms=platforms,
    )

    # Log detected settings
    print_panel(config.summary())

    # Phase 3-4: Initialize + Index (reuse existing functions)
    await run_init_setup(
        project_root=config.project_root,
        file_extensions=config.extensions,
        embedding_model=config.embedding_model,
        mcp=False,  # We'll handle MCP separately
        auto_index=True,
        force=force,
    )

    # Phase 5: MCP Integration for all detected platforms
    if platforms:
        print_info(f"🔗 Configuring {len(platforms)} MCP platform(s)...")
        for platform in platforms:
            try:
                configure_platform(platform, project_root, enable_watch=True)
            except Exception as e:
                print_warning(f"Failed to configure {platform}: {e}")

    # Phase 6: Completion
    print_success("🎉 Setup complete!")
    print_next_steps([...])
```

**Option B: Unified Implementation** (More work, cleaner)
- Refactor common initialization logic into shared functions
- Create new `setup.py` with optimized workflow
- Deprecate `init` and `install` in favor of `setup`

### 3.5 New Functions Needed

```python
def scan_file_extensions(
    project_root: Path,
    timeout: float = 2.0  # Prevent slow scans
) -> list[str] | None:
    """Scan project for unique file extensions (fast with timeout)."""
    extensions = set()
    start_time = time.time()

    try:
        for path in project_root.rglob("*"):
            if time.time() - start_time > timeout:
                return None  # Timeout, use defaults

            if path.is_file() and not should_ignore(path):
                ext = path.suffix
                if ext and ext in LANGUAGE_MAPPINGS:
                    extensions.add(ext)
    except Exception:
        return None

    return sorted(extensions) if extensions else None


def select_optimal_embedding_model(languages: list[str]) -> str:
    """Select best embedding model based on project languages."""
    if not languages:
        return DEFAULT_EMBEDDING_MODELS["code"]

    # Use precise model for multi-language projects
    if len(languages) > 3:
        return DEFAULT_EMBEDDING_MODELS["precise"]

    return DEFAULT_EMBEDDING_MODELS["code"]


def should_configure_platform(platform: str, project_root: Path) -> bool:
    """Determine if we should configure a platform."""
    # Always configure Claude Code (project-scoped)
    if platform == "claude-code":
        return True

    # For global platforms, check if they exist
    config_path = get_platform_config_path(platform, project_root)
    return config_path.parent.exists()
```

---

## 4. Edge Cases & Error Handling

### 4.1 Edge Cases

| Scenario | Handling Strategy |
|----------|------------------|
| **Already initialized** | Skip init, only configure MCP (log: "Already initialized") |
| **No MCP platforms detected** | Configure Claude Code only (most common) |
| **Very large codebase** | Use timeout for extension scanning, fall back to defaults |
| **No Git repo** | Skip .gitignore updates, continue normally |
| **Permission errors** | Log warning, continue with next step |
| **Network issues** | Skip embedding model download, use cached |
| **Corrupted config** | With `--force`, reinitialize; otherwise, error and suggest `--force` |

### 4.2 Idempotency Guarantees

```python
# Safe to run multiple times
def setup_is_idempotent():
    # 1. Check existing state
    if is_initialized() and not force:
        print_info("Project already initialized, configuring MCP...")
        # Skip to MCP setup
        return configure_mcp_only()

    # 2. Backup existing config
    if config_exists():
        backup_config()

    # 3. Proceed with setup
    # ...
```

### 4.3 Error Recovery

```python
class SetupPhase(Enum):
    DETECTION = "detection"
    CONFIG = "configuration"
    INIT = "initialization"
    INDEX = "indexing"
    MCP = "mcp_integration"

def setup_with_recovery():
    failed_phases = []

    try:
        # Phase 1: Detection
        detection_result = detect_project()
    except Exception as e:
        failed_phases.append((SetupPhase.DETECTION, e))
        # Use fallback defaults
        detection_result = use_fallback_detection()

    # Continue with other phases...
    # If critical phase fails, abort gracefully
    # If optional phase fails, log warning and continue

    return SetupResult(
        success=len(failed_phases) == 0,
        failed_phases=failed_phases,
        configured_platforms=platforms,
    )
```

---

## 5. User Experience Design

### 5.1 Command Interface

```bash
# Primary use case: zero-config setup
$ mcp-vector-search setup
🔍 Analyzing your project...
  ✅ Detected: Python project (pyproject.toml)
  ✅ Found 3 languages: Python, Markdown, JSON
  ✅ Scanning file extensions... (127 .py files, 12 .md files)
  ✅ Detected 1 MCP platform: Claude Code

🚀 Initializing mcp-vector-search...
  ✅ Created .mcp-vector-search/ directory
  ✅ Generated configuration (127 files to index)
  ✅ Updated .gitignore

🔍 Indexing codebase...
  [████████████████████] 127/127 files
  ✅ Indexed 127 files in 3.2s

🔗 Configuring MCP integration...
  ✅ Claude Code: Configured (.mcp.json created)

🎉 Setup complete!

Next steps:
  • Open Claude Code in this directory
  • Try: "Search for authentication functions"
  • Check status: mcp-vector-search status

Configuration summary saved to .mcp-vector-search/setup.log
```

### 5.2 Verbose Mode

```bash
$ mcp-vector-search setup --verbose
[DEBUG] Detecting project root...
[DEBUG] Found .git at /path/to/project
[DEBUG] Using project root: /path/to/project
[DEBUG] Scanning for file extensions...
[DEBUG] Found extensions: .py (127 files), .md (12 files), .json (8 files)
[DEBUG] Detected languages: ['python', 'markdown', 'json']
[DEBUG] Selected embedding model: sentence-transformers/all-MiniLM-L6-v2
[DEBUG] Scanning for MCP platforms...
[DEBUG] Found: claude-code (.mcp.json)
[DEBUG] Not found: cursor (~/.cursor/mcp.json directory doesn't exist)
...
```

### 5.3 Comparison with Other Tools

**Similar "zero-config" patterns**:

1. **Vercel** (`vercel`):
   - Detects framework (Next.js, React, etc.)
   - Auto-configures build settings
   - No user input required

2. **Turborepo** (`turbo`):
   - Detects monorepo structure
   - Auto-generates `turbo.json`
   - Smart caching configuration

3. **Vite** (`npm create vite@latest`):
   - Template selection (only user choice)
   - Auto-installs dependencies
   - Generates optimal config

**Our approach**: Combine detection (like Vercel) + multi-platform support (unique) + zero prompts (like Turborepo).

---

## 6. Implementation Recommendations

### 6.1 Code Structure

```
src/mcp_vector_search/cli/commands/
├─ setup.py              # NEW: Smart setup command
├─ init.py               # KEEP: For advanced users who want control
├─ install.py            # KEEP: For platform-specific installation
├─ _setup_utils.py       # NEW: Shared utilities (detection, config)
└─ _setup_phases.py      # NEW: Phase implementations
```

### 6.2 Shared Utilities (`_setup_utils.py`)

```python
@dataclass
class ProjectAnalysis:
    """Results of project analysis."""
    project_root: Path
    project_type: str  # "Python", "JavaScript", "Rust", etc.
    languages: list[str]
    detected_extensions: list[str] | None
    installed_platforms: dict[str, Path]
    is_git_repo: bool
    file_count_estimate: int

def analyze_project(project_root: Path) -> ProjectAnalysis:
    """Analyze project and return comprehensive metadata."""
    # Run all detection in parallel
    # ...

def generate_smart_config(analysis: ProjectAnalysis) -> dict:
    """Generate optimal configuration from analysis."""
    # ...
```

### 6.3 Migration Path

**Phase 1**: Implement `setup` command (new)
- Add to CLI
- Use existing functions from `init.py` and `install.py`
- Add detection enhancements

**Phase 2**: Update documentation
- Make `setup` the recommended command
- Keep `init` and `install` for advanced use cases
- Add migration guide

**Phase 3** (optional): Deprecation
- Mark `init` and `install` as "advanced commands"
- Add deprecation warnings (in 6+ months)
- Eventually consolidate into `setup` with flags

---

## 7. Testing Strategy

### 7.1 Test Scenarios

```python
def test_setup_fresh_project():
    """Test setup in a brand new project."""
    # Should initialize, index, and configure MCP

def test_setup_already_initialized():
    """Test setup in already initialized project."""
    # Should skip init, only configure MCP

def test_setup_no_mcp_platforms():
    """Test setup when no platforms detected."""
    # Should configure Claude Code only

def test_setup_partial_failure():
    """Test setup when one phase fails."""
    # Should continue and report failures

def test_setup_timeout_detection():
    """Test setup in very large codebase."""
    # Should timeout and use defaults

def test_setup_idempotency():
    """Test running setup multiple times."""
    # Should be safe and produce same result
```

### 7.2 Performance Benchmarks

Target performance:
- Small project (<100 files): < 5 seconds
- Medium project (100-1000 files): < 15 seconds
- Large project (1000-10000 files): < 60 seconds
- Detection timeout: 2 seconds max

---

## 8. Summary & Next Steps

### 8.1 Key Recommendations

1. ✅ **Create `setup` command** as smart orchestrator
2. ✅ **Reuse existing functions** from `init` and `install` (avoid duplication)
3. ✅ **Add smart detection** for file extensions (with timeout)
4. ✅ **Configure all detected platforms** (not just Claude Code)
5. ✅ **Comprehensive logging** to show users what's happening
6. ✅ **Make it idempotent** (safe to run multiple times)
7. ✅ **Keep `init` and `install`** for advanced users

### 8.2 Implementation Checklist

**Phase 1: Core Setup Command**
- [ ] Create `setup.py` with basic structure
- [ ] Implement `analyze_project()` function
- [ ] Implement `scan_file_extensions()` with timeout
- [ ] Add project type detection
- [ ] Create smart config generator

**Phase 2: Integration**
- [ ] Integrate with existing `run_init_setup()`
- [ ] Add platform configuration loop
- [ ] Implement comprehensive logging
- [ ] Add error recovery

**Phase 3: Polish**
- [ ] Add `--verbose` flag
- [ ] Create setup summary output
- [ ] Write unit tests
- [ ] Update documentation
- [ ] Add to main CLI

### 8.3 Alternative Approaches Considered

**❌ Rejected: Interactive Prompts**
- Reason: Not zero-config, adds friction
- Users want fast, opinionated setup

**❌ Rejected: Deprecate `init` and `install` immediately**
- Reason: Advanced users need granular control
- Keep for CI/CD and scripting

**✅ Accepted: Wrapper + Enhancement Pattern**
- Reuses existing code (low risk)
- Adds intelligence on top
- Easy to test and maintain

---

## 9. Code Examples

### 9.1 Minimal Setup Command (Prototype)

```python
@setup_app.callback()
async def main(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
) -> None:
    """🚀 Smart zero-config setup for mcp-vector-search.

    Automatically detects your project type, languages, and installed MCP platforms,
    then configures everything with sensible defaults. No user input required!

    Examples:
        # Basic setup (recommended)
        $ mcp-vector-search setup

        # Force re-initialization
        $ mcp-vector-search setup --force

        # Verbose output for debugging
        $ mcp-vector-search setup --verbose
    """
    try:
        # Phase 1: Detection
        console.print("[bold blue]🔍 Analyzing your project...[/bold blue]")

        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        # Check existing state
        already_init = project_manager.is_initialized()
        if already_init and not force:
            print_info("✅ Project already initialized")
            print_info("Configuring MCP integrations...")

        # Detect languages and platforms
        languages = project_manager.detect_languages() if not already_init else []
        platforms = detect_installed_platforms()

        if verbose:
            print_info(f"Project root: {project_root}")
            print_info(f"Languages: {', '.join(languages) if languages else 'detecting...'}")
            print_info(f"Platforms: {', '.join(platforms.keys())}")

        # Phase 2: Initialize (if needed)
        if not already_init or force:
            console.print("\n[bold blue]🚀 Initializing mcp-vector-search...[/bold blue]")

            await run_init_setup(
                project_root=project_root,
                file_extensions=None,  # Use defaults
                mcp=False,  # We handle MCP separately
                auto_index=True,
                auto_indexing=True,
                force=force,
            )

        # Phase 3: MCP Integration
        if platforms:
            console.print(f"\n[bold blue]🔗 Configuring {len(platforms)} MCP platform(s)...[/bold blue]")

            for platform_name in platforms:
                try:
                    success = configure_platform(
                        platform=platform_name,
                        project_root=project_root,
                        enable_watch=True,
                        force=force,
                    )
                    if verbose and success:
                        print_success(f"✅ {platform_name} configured")
                except Exception as e:
                    print_warning(f"⚠️  {platform_name}: {e}")
        else:
            # No platforms detected, configure Claude Code as default
            console.print("\n[bold blue]🔗 Configuring Claude Code...[/bold blue]")
            configure_platform("claude-code", project_root, enable_watch=True, force=force)

        # Phase 4: Completion
        console.print("\n[bold green]🎉 Setup complete![/bold green]")

        next_steps = [
            "Open Claude Code in this directory",
            "Try: 'Search for authentication functions'",
            "[cyan]mcp-vector-search status[/cyan] - Check project status",
        ]
        print_next_steps(next_steps, title="Ready to Use")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print_error(f"Setup failed: {e}")
        raise typer.Exit(1)
```

---

## 10. Memory Updates

**Project-Specific Learnings**:

1. **Command Architecture**:
   - `init` and `install` have 80%+ overlap in functionality
   - `init` defaults to MCP enabled, `install` defaults to disabled
   - Both support same initialization parameters

2. **Detection Capabilities**:
   - Project root detection already implemented (walks directory tree)
   - Language detection via `detect_languages()` scans file extensions
   - Platform detection via `detect_installed_platforms()` checks filesystem
   - File extension detection could be added with timeout (2s max)

3. **Auto-Detection Pattern**:
   - `ProjectManager._detect_project_root()` uses common indicators (.git, pyproject.toml, etc.)
   - Can be extended for file type scanning
   - Should have timeouts to prevent slow scans on large codebases

4. **Smart Defaults Strategy**:
   - Use detected values when available
   - Fall back to `DEFAULT_FILE_EXTENSIONS` if detection times out
   - Select embedding model based on project type
   - Always enable auto-indexing and file watching

5. **Implementation Approach**:
   - Recommended: Create `setup` as wrapper around existing `run_init_setup()`
   - Reuse `detect_installed_platforms()` from `install.py`
   - Add new `scan_file_extensions()` with 2s timeout
   - Keep `init` and `install` for advanced users

---

**Files Analyzed**:
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py` (646 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py` (679 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py` (1183 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py` (351 lines)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py` (201 lines)

**Research Methodology**:
- Read and analyzed 5 core files (3060 total lines)
- Identified common patterns and duplication
- Mapped auto-detection capabilities
- Designed zero-config workflow
- Documented edge cases and UX patterns

**Research Duration**: ~15 minutes (file reading + analysis + documentation)
