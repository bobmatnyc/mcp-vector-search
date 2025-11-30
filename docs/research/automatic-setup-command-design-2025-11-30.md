# Automatic Setup Command Design - Using claude mcp add

**Research Date**: 2025-11-30
**Researcher**: Research Agent
**Status**: Design Complete
**Context**: Design automatic setup command using native `claude mcp add` instead of manual `.mcp.json` creation

---

## Executive Summary

This research identifies the correct MCP server entry point, native `claude mcp add` command syntax, and designs an automatic setup flow that combines `init` + `index` + MCP registration into a single hands-off command.

**Key Findings**:
1. ✅ MCP server entry point is `python -m mcp_vector_search.mcp.server`
2. ✅ Correct `claude mcp add` syntax: `claude mcp add --transport stdio <name> -- <command> [args]`
3. ✅ Current `.mcp.json` manual creation should be replaced with native CLI
4. ✅ Platform detection strategy defined for graceful fallback
5. ✅ Automatic setup flow designed with error handling

---

## 1. MCP Server Entry Point Verification

### Current Entry Points

**Primary Entry Point**: `mcp_vector_search.mcp.server`
- **Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py`
- **Module Path**: `python -m mcp_vector_search.mcp.server`
- **Alternative**: Direct execution via `__main__.py` (calls same `run_mcp_server` function)

**Evidence from server.py**:
```python
# Line 686-734
async def run_mcp_server(
    project_root: Path | None = None,
    enable_file_watching: bool | None = None
) -> None:
    """Run the MCP server using stdio transport."""
    server = create_mcp_server(project_root, enable_file_watching)

    init_options = InitializationOptions(
        server_name="mcp-vector-search",
        server_version="0.4.0",
        capabilities=ServerCapabilities(tools={"listChanged": True}, logging={}),
    )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    # ... error handling

if __name__ == "__main__":
    # Allow specifying project root as command line argument
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Check for file watching flag in command line args
    enable_file_watching = None
    if "--no-watch" in sys.argv:
        enable_file_watching = False
        # ...

    asyncio.run(run_mcp_server(project_root, enable_file_watching))
```

**Current Usage in mcp.py CLI (Line 123)**:
```python
def get_mcp_server_command(project_root: Path, enable_file_watching: bool = True) -> str:
    """Get the command to run the MCP server."""
    python_exe = sys.executable
    watch_flag = "" if enable_file_watching else " --no-watch"
    return f"{python_exe} -m mcp_vector_search.mcp.server{watch_flag} {project_root}"
```

**Command-Line Arguments**:
- **Position 1** (`sys.argv[1]`): Project root path
- **Flag**: `--no-watch` (disable file watching)
- **Flag**: `--watch` (enable file watching)
- **Default**: File watching enabled if not specified

**Entry Point from pyproject.toml (Line 76-77)**:
```toml
[project.scripts]
mcp-vector-search = "mcp_vector_search.cli.main:cli_with_suggestions"
```

### Server Capabilities

**MCP Tools Provided** (Lines 142-275):
1. `search_code` - Search for code using semantic similarity
2. `search_similar` - Find code similar to a specific file or function
3. `search_context` - Search for code based on contextual description
4. `get_project_status` - Get project indexing status and statistics
5. `index_project` - Index or reindex the project codebase

---

## 2. Claude MCP Add Command Syntax

### Official Syntax (from Web Search)

**Basic Format**:
```bash
claude mcp add --transport stdio <name> -- <command> [args...]
```

**Key Points**:
- `--transport stdio` - Specifies local stdio transport (required for local servers)
- `<name>` - Server identifier (e.g., "mcp-vector-search")
- `--` - Separator between Claude options and server command
- `<command> [args...]` - Actual command to run MCP server

**Environment Variables**:
```bash
claude mcp add --transport stdio <name> \
  --env KEY1=value1 --env KEY2=value2 \
  -- <command> [args...]
```

**Scope Options**:
- `--scope user` - Add to user-level configuration (global)
- Default: Project-scoped (`.mcp.json` in project root)

### Real-World Examples

**Example 1: Airtable MCP Server**
```bash
claude mcp add --transport stdio airtable \
  --env AIRTABLE_API_KEY=YOUR_KEY \
  -- npx -y airtable-mcp-server
```

**Example 2: Weather API (HTTP Transport)**
```bash
claude mcp add --transport http weather-api https://api.weather.com/mcp \
  --header "Authorization: Bearer token"
```

**Example 3: JSON Configuration (Alternative)**
```bash
claude mcp add-json weather-api '{
  "type":"http",
  "url":"https://api.weather.com/mcp",
  "headers":{"Authorization":"Bearer token"}
}'
```

### Current mcp-vector-search Usage (from MCP_SETUP.md)

**Current Manual JSON Config** (Lines 18-26):
```bash
claude mcp add-json mcp-vector-search '{
  "command": "/Users/masa/Projects/managed/mcp-vector-search/.venv/bin/python",
  "args": ["-m", "mcp_vector_search.mcp.server", "/Users/masa/Projects/managed/mcp-vector-search"],
  "cwd": "/Users/masa/Projects/managed/mcp-vector-search",
  "env": {
    "MCP_DEBUG": "1"
  }
}'
```

**Problems with Current Approach**:
1. ❌ Hardcoded absolute paths (not portable)
2. ❌ Uses `add-json` instead of native `add` command
3. ❌ Requires manual path specification
4. ❌ Not user-friendly for non-technical users

---

## 3. Proposed Native Command Syntax

### Option 1: Using `uv run` (Recommended)

**Command**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- uv run mcp-vector-search mcp
```

**Why This Works**:
- `uv run mcp-vector-search mcp` - Uses the CLI entry point defined in pyproject.toml
- BUT: Current CLI doesn't have a `serve` subcommand that runs MCP server
- **Problem**: The `mcp` subcommand is for MCP *configuration*, not *serving*

### Option 2: Using `uv run -m` (Correct Approach)

**Command**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}
```

**Why This Works**:
- `uv run python` - Runs Python from the uv-managed environment
- `-m mcp_vector_search.mcp.server` - Executes the MCP server module
- `{PROJECT_ROOT}` - Passed as sys.argv[1] to the server

### Option 3: Using Direct Python Path (Fallback)

**Command**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- python -m mcp_vector_search.mcp.server {PROJECT_ROOT}
```

**Why This Works**:
- Assumes `mcp-vector-search` is installed in PATH via pipx/uv tool
- Simpler command syntax
- Works if package is globally available

---

## 4. Automatic Setup Flow Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    mcp-vector-search setup                  │
│                     (Single Command)                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │  1. Platform Detection & Validation   │
         │     - Check claude CLI availability   │
         │     - Verify uv/pipx installation    │
         │     - Detect OS (Windows/Unix)       │
         └───────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │  2. Project Initialization (if needed)│
         │     - Run init workflow              │
         │     - Auto-detect languages/exts     │
         │     - Create .mcp-vector-search/     │
         └───────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │  3. Index Project (if needed)        │
         │     - Run indexing workflow          │
         │     - Show progress                  │
         │     - Verify index creation          │
         └───────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │  4. MCP Registration                 │
         │     ┌─────────────────────────────┐  │
         │     │ If claude CLI available:    │  │
         │     │ - Use claude mcp add        │  │
         │     │ - Auto-detect best command  │  │
         │     │ - Set environment vars      │  │
         │     └─────────────────────────────┘  │
         │     ┌─────────────────────────────┐  │
         │     │ If claude CLI unavailable:  │  │
         │     │ - Create .mcp.json manually │  │
         │     │ - Show manual setup guide   │  │
         │     └─────────────────────────────┘  │
         └───────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │  5. Verification & User Guidance      │
         │     - Test MCP server startup        │
         │     - Show next steps                │
         │     - Display available tools        │
         └───────────────────────────────────────┘
```

### Detailed Implementation Plan

#### Step 1: Platform Detection

```python
def detect_platform_capabilities() -> PlatformCapabilities:
    """Detect available tools and platform configuration."""

    # Check Claude CLI
    claude_cmd = shutil.which("claude")
    claude_available = False
    if claude_cmd:
        try:
            result = subprocess.run(
                [claude_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            claude_available = result.returncode == 0
        except Exception:
            pass

    # Check uv
    uv_available = shutil.which("uv") is not None

    # Check pipx
    pipx_available = shutil.which("pipx") is not None

    # Detect OS
    is_windows = sys.platform == "win32"
    is_wsl = "microsoft" in platform.uname().release.lower()

    return PlatformCapabilities(
        claude_cli_available=claude_available,
        uv_available=uv_available,
        pipx_available=pipx_available,
        is_windows=is_windows,
        is_wsl=is_wsl,
        python_executable=sys.executable
    )
```

#### Step 2: Command Generation

```python
def generate_mcp_server_command(
    platform: PlatformCapabilities,
    project_root: Path,
    enable_file_watching: bool = True
) -> str:
    """Generate the optimal MCP server command based on platform."""

    project_root_str = str(project_root.resolve())

    # Priority 1: uv run (most reliable)
    if platform.uv_available:
        cmd = f"uv run python -m mcp_vector_search.mcp.server"

        # Windows requires special handling
        if platform.is_windows and not platform.is_wsl:
            cmd = f"cmd /c {cmd}"

    # Priority 2: pipx (if uv not available)
    elif platform.pipx_available:
        # Assume package is installed via pipx
        cmd = f"{platform.python_executable} -m mcp_vector_search.mcp.server"

    # Priority 3: Direct python (fallback)
    else:
        cmd = f"python -m mcp_vector_search.mcp.server"

    # Add project root argument
    cmd += f" {project_root_str}"

    # Add file watching flag
    if not enable_file_watching:
        cmd += " --no-watch"

    return cmd
```

#### Step 3: MCP Registration with Native CLI

```python
def register_mcp_server_native(
    server_name: str,
    project_root: Path,
    enable_file_watching: bool = True,
    scope: str = "project"  # or "user"
) -> bool:
    """Register MCP server using native claude mcp add command."""

    # Detect platform
    platform = detect_platform_capabilities()

    if not platform.claude_cli_available:
        print_warning("Claude CLI not available, falling back to manual .mcp.json")
        return register_mcp_server_manual(server_name, project_root, enable_file_watching)

    # Generate server command
    server_cmd = generate_mcp_server_command(platform, project_root, enable_file_watching)

    # Parse command into parts (before and after project root)
    # Example: "uv run python -m mcp_vector_search.mcp.server /path/to/project"
    cmd_parts = server_cmd.split()

    # Find project_root position and split
    project_root_str = str(project_root.resolve())
    if project_root_str in cmd_parts:
        idx = cmd_parts.index(project_root_str)
        base_cmd = cmd_parts[:idx]
        args = cmd_parts[idx:]  # ["/path/to/project"]
    else:
        base_cmd = cmd_parts
        args = []

    # Build claude mcp add command
    claude_cmd = ["claude", "mcp", "add", "--transport", "stdio", server_name]

    # Add environment variables
    file_watching_val = "true" if enable_file_watching else "false"
    claude_cmd.extend(["--env", f"MCP_ENABLE_FILE_WATCHING={file_watching_val}"])

    # Add scope if user-level
    if scope == "user":
        claude_cmd.append("--scope")
        claude_cmd.append("user")

    # Add separator and command
    claude_cmd.append("--")
    claude_cmd.extend(base_cmd)
    claude_cmd.extend(args)

    # Execute
    try:
        print_info(f"Registering MCP server: {' '.join(claude_cmd)}")
        result = subprocess.run(
            claude_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print_success(f"✅ MCP server '{server_name}' registered successfully!")
            return True
        else:
            print_error(f"Registration failed: {result.stderr}")
            print_warning("Falling back to manual .mcp.json creation")
            return register_mcp_server_manual(server_name, project_root, enable_file_watching)

    except subprocess.TimeoutExpired:
        print_error("Registration timed out")
        return False
    except Exception as e:
        print_error(f"Registration error: {e}")
        return register_mcp_server_manual(server_name, project_root, enable_file_watching)
```

#### Step 4: Automatic Setup Command

```python
@setup_app.command("setup")
def automatic_setup(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server"
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing"
    ),
    scope: str = typer.Option(
        "project",
        "--scope",
        help="Registration scope: 'project' or 'user'"
    ),
    skip_index: bool = typer.Option(
        False,
        "--skip-index",
        help="Skip automatic indexing"
    ),
) -> None:
    """🚀 Automatic setup: init + index + MCP registration (one command).

    This command combines all setup steps into a single hands-off workflow:
    1. Initialize project (if not already initialized)
    2. Index codebase (if not already indexed or --skip-index)
    3. Register MCP server with Claude CLI (or create .mcp.json)
    4. Verify setup and show next steps

    Examples:
        $ mcp-vector-search setup
        $ mcp-vector-search setup --skip-index
        $ mcp-vector-search setup --scope user
        $ mcp-vector-search setup --no-watch
    """

    console.print("\n[bold blue]🚀 Starting Automatic Setup[/bold blue]\n")

    project_root = ctx.obj.get("project_root") or Path.cwd()
    project_manager = ProjectManager(project_root)
    enable_file_watching = not no_watch

    # Step 1: Platform Detection
    console.print("[cyan]Step 1/5:[/cyan] Detecting platform capabilities...")
    platform = detect_platform_capabilities()

    console.print(f"  • Claude CLI: {'✅' if platform.claude_cli_available else '❌'}")
    console.print(f"  • uv: {'✅' if platform.uv_available else '❌'}")
    console.print(f"  • OS: {sys.platform}")

    # Step 2: Initialize Project
    console.print("\n[cyan]Step 2/5:[/cyan] Initializing project...")
    if not project_manager.is_initialized():
        # Run init workflow with auto-detection
        from .init import run_init_workflow

        try:
            run_init_workflow(
                project_root=project_root,
                auto_detect=True,
                show_wizard=False
            )
            print_success("✅ Project initialized")
        except Exception as e:
            print_error(f"Initialization failed: {e}")
            raise typer.Exit(1)
    else:
        print_info("Project already initialized, skipping...")

    # Step 3: Index Project
    console.print("\n[cyan]Step 3/5:[/cyan] Indexing project...")
    if not skip_index:
        from .index import run_indexing

        try:
            # Check if already indexed
            config = project_manager.load_config()
            if not (config.index_path / "chroma.sqlite3").exists():
                run_indexing(
                    project_root=project_root,
                    force_reindex=False,
                    show_progress=True
                )
                print_success("✅ Project indexed")
            else:
                print_info("Index already exists, skipping...")
        except Exception as e:
            print_error(f"Indexing failed: {e}")
            raise typer.Exit(1)
    else:
        print_info("Indexing skipped (--skip-index)")

    # Step 4: MCP Registration
    console.print("\n[cyan]Step 4/5:[/cyan] Registering MCP server...")

    success = register_mcp_server_native(
        server_name=server_name,
        project_root=project_root,
        enable_file_watching=enable_file_watching,
        scope=scope
    )

    if not success:
        print_error("MCP registration failed")
        raise typer.Exit(1)

    # Step 5: Verification
    console.print("\n[cyan]Step 5/5:[/cyan] Verifying setup...")

    # Test server startup
    server_cmd = generate_mcp_server_command(platform, project_root, enable_file_watching)
    test_process = subprocess.Popen(
        server_cmd.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        test_process.wait(timeout=5)
        print_success("✅ MCP server can start successfully")
    except subprocess.TimeoutExpired:
        test_process.terminate()
        print_success("✅ MCP server is responsive")

    # Show next steps
    console.print("\n[bold green]🎉 Setup Complete![/bold green]\n")

    console.print("[bold]Next Steps:[/bold]")
    console.print("1. Open your project in Claude Code")
    console.print("2. Use semantic search tools in your conversation")
    console.print("3. Try: 'Search for authentication logic'")

    console.print("\n[bold]Available MCP Tools:[/bold]")
    console.print("  • search_code - Search using semantic similarity")
    console.print("  • search_similar - Find similar code")
    console.print("  • search_context - Search by description")
    console.print("  • get_project_status - View index status")
    console.print("  • index_project - Reindex codebase")

    if enable_file_watching:
        console.print("\n[green]✅ File watching enabled[/green] - Changes auto-indexed")
    else:
        console.print("\n[yellow]⚠️  File watching disabled[/yellow] - Manual reindex required")
```

---

## 5. Error Handling & Fallback Chain

### Fallback Strategy

```
1st Priority: claude mcp add (native CLI)
    ├─ If claude CLI available
    │  └─ Use subprocess to call claude mcp add
    │
    └─ If fails or unavailable
       │
       2nd Priority: Manual .mcp.json creation
          ├─ Create .mcp.json in project root
          ├─ Use optimal command based on platform
          └─ Show manual verification steps
          │
          └─ If .mcp.json creation fails
             │
             3rd Priority: Show manual instructions
                └─ Display exact command user should run
                └─ Provide troubleshooting guide
```

### Error Handling Code

```python
class MCPRegistrationError(Exception):
    """Raised when MCP registration fails."""
    pass

def register_mcp_server_with_fallback(
    server_name: str,
    project_root: Path,
    enable_file_watching: bool = True,
    scope: str = "project"
) -> tuple[bool, str]:
    """
    Register MCP server with automatic fallback chain.

    Returns:
        (success: bool, method: str)
        method can be: "claude-cli", "manual-json", "manual-instructions"
    """

    # Try native Claude CLI
    try:
        if register_mcp_server_native(server_name, project_root, enable_file_watching, scope):
            return (True, "claude-cli")
    except Exception as e:
        print_warning(f"Native registration failed: {e}")

    # Try manual .mcp.json creation
    try:
        if register_mcp_server_manual(server_name, project_root, enable_file_watching):
            return (True, "manual-json")
    except Exception as e:
        print_warning(f"Manual .mcp.json creation failed: {e}")

    # Show manual instructions as last resort
    show_manual_setup_instructions(server_name, project_root, enable_file_watching)
    return (False, "manual-instructions")

def show_manual_setup_instructions(
    server_name: str,
    project_root: Path,
    enable_file_watching: bool
) -> None:
    """Show manual setup instructions when automatic methods fail."""

    platform = detect_platform_capabilities()
    server_cmd = generate_mcp_server_command(platform, project_root, enable_file_watching)

    console.print("\n[bold red]⚠️  Automatic registration failed[/bold red]")
    console.print("\n[bold]Manual Setup Required:[/bold]")
    console.print("\n1. Run this command:")
    console.print(f"\n   [cyan]{server_cmd}[/cyan]")

    console.print("\n2. Or add to .mcp.json manually:")

    json_config = {
        "mcpServers": {
            server_name: {
                "type": "stdio",
                "command": server_cmd.split()[0],
                "args": server_cmd.split()[1:],
                "env": {
                    "MCP_ENABLE_FILE_WATCHING": "true" if enable_file_watching else "false"
                }
            }
        }
    }

    console.print(f"\n[dim]{json.dumps(json_config, indent=2)}[/dim]")

    console.print("\n[bold]Troubleshooting:[/bold]")
    console.print("  • Ensure Claude CLI is installed: https://claude.ai/download")
    console.print("  • Check mcp-vector-search is in PATH")
    console.print("  • Verify project is initialized: mcp-vector-search status")
```

---

## 6. Platform-Specific Considerations

### Windows

**Command Wrapper Required**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  -- cmd /c uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}
```

**Detection**:
```python
if sys.platform == "win32" and not is_wsl():
    cmd_prefix = ["cmd", "/c"]
else:
    cmd_prefix = []
```

### macOS/Linux

**Standard Command**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  -- uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}
```

### WSL (Windows Subsystem for Linux)

**Detection**:
```python
def is_wsl() -> bool:
    """Detect if running in WSL."""
    return "microsoft" in platform.uname().release.lower()
```

**Command**: Same as Linux (no `cmd /c` wrapper needed)

---

## 7. Implementation Roadmap

### Phase 1: Core Implementation (Week 1)
- [ ] Implement `detect_platform_capabilities()`
- [ ] Implement `generate_mcp_server_command()`
- [ ] Implement `register_mcp_server_native()`
- [ ] Add fallback to manual `.mcp.json` creation

### Phase 2: Setup Command (Week 1)
- [ ] Create `setup` command in CLI
- [ ] Integrate init workflow
- [ ] Integrate index workflow
- [ ] Add MCP registration step

### Phase 3: Error Handling (Week 2)
- [ ] Implement fallback chain
- [ ] Add manual instruction display
- [ ] Add verification tests
- [ ] Add troubleshooting guide

### Phase 4: Testing (Week 2)
- [ ] Test on macOS
- [ ] Test on Linux
- [ ] Test on Windows (native)
- [ ] Test on WSL
- [ ] Test with/without Claude CLI
- [ ] Test with/without uv/pipx

### Phase 5: Documentation (Week 3)
- [ ] Update README.md with `setup` command
- [ ] Create setup troubleshooting guide
- [ ] Add platform-specific notes
- [ ] Update MCP_SETUP.md

---

## 8. Testing Strategy

### Unit Tests

```python
def test_detect_platform_capabilities():
    """Test platform detection logic."""
    caps = detect_platform_capabilities()

    assert isinstance(caps.claude_cli_available, bool)
    assert isinstance(caps.uv_available, bool)
    assert isinstance(caps.pipx_available, bool)
    assert caps.python_executable == sys.executable

def test_generate_mcp_server_command_with_uv():
    """Test command generation with uv available."""
    platform = PlatformCapabilities(
        claude_cli_available=True,
        uv_available=True,
        pipx_available=False,
        is_windows=False,
        is_wsl=False,
        python_executable="/usr/bin/python"
    )

    cmd = generate_mcp_server_command(
        platform,
        Path("/home/user/project"),
        enable_file_watching=True
    )

    assert "uv run python" in cmd
    assert "mcp_vector_search.mcp.server" in cmd
    assert "/home/user/project" in cmd
    assert "--no-watch" not in cmd

def test_generate_mcp_server_command_windows():
    """Test command generation on Windows."""
    platform = PlatformCapabilities(
        claude_cli_available=True,
        uv_available=True,
        pipx_available=False,
        is_windows=True,
        is_wsl=False,
        python_executable="C:\\Python\\python.exe"
    )

    cmd = generate_mcp_server_command(
        platform,
        Path("C:\\Users\\user\\project"),
        enable_file_watching=False
    )

    assert "cmd /c" in cmd
    assert "--no-watch" in cmd
```

### Integration Tests

```python
@pytest.mark.integration
def test_setup_command_full_workflow(tmp_path):
    """Test complete setup workflow."""

    # Run setup command
    result = subprocess.run(
        ["mcp-vector-search", "setup", "--skip-index"],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

    # Verify project initialization
    assert (tmp_path / ".mcp-vector-search" / "config.json").exists()

    # Verify MCP registration (either .mcp.json or claude CLI)
    mcp_json = tmp_path / ".mcp.json"

    if mcp_json.exists():
        with open(mcp_json) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "mcp-vector-search" in config["mcpServers"]

@pytest.mark.integration
def test_setup_command_with_claude_cli_mock(tmp_path, mocker):
    """Test setup with mocked Claude CLI."""

    # Mock shutil.which to return claude
    mocker.patch("shutil.which", return_value="/usr/bin/claude")

    # Mock subprocess.run for claude --version
    mocker.patch("subprocess.run", side_effect=[
        # First call: claude --version
        subprocess.CompletedProcess(args=[], returncode=0, stdout="1.0.0"),
        # Second call: claude mcp add
        subprocess.CompletedProcess(args=[], returncode=0, stdout="Success"),
    ])

    # Run setup
    result = subprocess.run(
        ["mcp-vector-search", "setup", "--skip-index"],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "registered successfully" in result.stdout
```

---

## 9. User Experience Examples

### Scenario 1: First-Time Setup (Claude CLI Available)

```bash
$ cd my-project
$ mcp-vector-search setup

🚀 Starting Automatic Setup

Step 1/5: Detecting platform capabilities...
  • Claude CLI: ✅
  • uv: ✅
  • OS: darwin

Step 2/5: Initializing project...
Auto-detected languages: python, typescript
Auto-detected file extensions: .py, .ts, .tsx
✅ Project initialized

Step 3/5: Indexing project...
Indexing: 100%|████████████████████| 247/247 files
✅ 1,234 code chunks indexed
✅ Project indexed

Step 4/5: Registering MCP server...
Registering: claude mcp add --transport stdio mcp-vector-search --env MCP_ENABLE_FILE_WATCHING=true -- uv run python -m mcp_vector_search.mcp.server /Users/user/my-project
✅ MCP server 'mcp-vector-search' registered successfully!

Step 5/5: Verifying setup...
✅ MCP server can start successfully

🎉 Setup Complete!

Next Steps:
1. Open your project in Claude Code
2. Use semantic search tools in your conversation
3. Try: 'Search for authentication logic'

Available MCP Tools:
  • search_code - Search using semantic similarity
  • search_similar - Find similar code
  • search_context - Search by description
  • get_project_status - View index status
  • index_project - Reindex codebase

✅ File watching enabled - Changes auto-indexed
```

### Scenario 2: Setup Without Claude CLI

```bash
$ cd my-project
$ mcp-vector-search setup

🚀 Starting Automatic Setup

Step 1/5: Detecting platform capabilities...
  • Claude CLI: ❌
  • uv: ✅
  • OS: linux

Step 2/5: Initializing project...
✅ Project initialized

Step 3/5: Indexing project...
✅ Project indexed

Step 4/5: Registering MCP server...
⚠️  Claude CLI not available, falling back to manual .mcp.json
✅ Created .mcp.json with MCP server configuration

Step 5/5: Verifying setup...
✅ MCP server can start successfully

🎉 Setup Complete!

Note: Claude CLI not detected. Created .mcp.json manually.
To use with Claude CLI, install from: https://claude.ai/download

Next Steps:
1. Open your project in Claude Code
2. Claude Code will automatically detect .mcp.json
3. Use semantic search tools in your conversation

✅ File watching enabled - Changes auto-indexed
```

### Scenario 3: Setup with Errors

```bash
$ cd my-project
$ mcp-vector-search setup

🚀 Starting Automatic Setup

Step 1/5: Detecting platform capabilities...
  • Claude CLI: ✅
  • uv: ❌
  • OS: darwin

Step 2/5: Initializing project...
✅ Project initialized

Step 3/5: Indexing project...
✅ Project indexed

Step 4/5: Registering MCP server...
⚠️  Native registration failed: uv command not found
⚠️  Falling back to manual .mcp.json
✅ Created .mcp.json with MCP server configuration

Step 5/5: Verifying setup...
✅ MCP server can start successfully

🎉 Setup Complete!

⚠️  Recommendations:
  • Install uv for better package management: curl -LsSf https://astral.sh/uv/install.sh | sh
  • Alternatively, install via pipx: pipx install mcp-vector-search

Next Steps:
1. Open your project in Claude Code
2. Use semantic search tools in your conversation

✅ File watching enabled - Changes auto-indexed
```

---

## 10. Recommendations

### Immediate Actions
1. ✅ **Implement `setup` command** - Combine init + index + MCP registration
2. ✅ **Use native `claude mcp add`** - Replace manual `.mcp.json` creation
3. ✅ **Add platform detection** - Auto-detect Claude CLI, uv, pipx availability
4. ✅ **Implement fallback chain** - Graceful degradation when tools unavailable

### Future Enhancements
1. **Interactive mode** - Ask user for confirmation at each step
2. **Custom server names** - Allow user to specify MCP server identifier
3. **Multiple project support** - Register same server for multiple projects
4. **Health checks** - Periodic MCP server connectivity tests
5. **Auto-update mechanism** - Detect when server command needs updating

### Migration Path
1. **Deprecate `mcp init`** - Point users to `setup` command
2. **Keep `mcp claude-code`** - For manual configuration if needed
3. **Update documentation** - Emphasize `setup` as primary method
4. **Add warnings** - Notify users about deprecated manual methods

---

## 11. Appendix

### A. Platform Detection Pseudocode

```
FUNCTION detect_platform_capabilities():
    claude_cli_available = check_command_exists("claude") AND test_claude_version()
    uv_available = check_command_exists("uv")
    pipx_available = check_command_exists("pipx")

    is_windows = sys.platform == "win32"
    is_wsl = "microsoft" in platform.uname().release.lower()

    python_executable = sys.executable

    RETURN PlatformCapabilities(
        claude_cli_available,
        uv_available,
        pipx_available,
        is_windows,
        is_wsl,
        python_executable
    )

FUNCTION check_command_exists(command_name):
    RETURN shutil.which(command_name) IS NOT None

FUNCTION test_claude_version():
    TRY:
        result = subprocess.run(["claude", "--version"], timeout=5)
        RETURN result.returncode == 0
    EXCEPT Exception:
        RETURN False
```

### B. Command Generation Decision Tree

```
IF uv_available:
    base_command = "uv run python -m mcp_vector_search.mcp.server"

    IF is_windows AND NOT is_wsl:
        command = f"cmd /c {base_command} {project_root}"
    ELSE:
        command = f"{base_command} {project_root}"

ELSE IF pipx_available:
    # Assume package installed via pipx
    command = f"{python_executable} -m mcp_vector_search.mcp.server {project_root}"

ELSE:
    # Fallback to system python
    command = f"python -m mcp_vector_search.mcp.server {project_root}"

IF NOT enable_file_watching:
    command += " --no-watch"

RETURN command
```

### C. Registration Fallback Chain

```
1. TRY native claude mcp add:
   - Check claude CLI available
   - Generate optimal command
   - Execute: claude mcp add --transport stdio <name> -- <command>
   - IF success: RETURN (True, "claude-cli")
   - IF failure: CONTINUE to step 2

2. TRY manual .mcp.json creation:
   - Generate server config JSON
   - Write to project_root / ".mcp.json"
   - Verify file written successfully
   - IF success: RETURN (True, "manual-json")
   - IF failure: CONTINUE to step 3

3. SHOW manual instructions:
   - Display exact command to run
   - Show JSON config to copy
   - Provide troubleshooting guide
   - RETURN (False, "manual-instructions")
```

### D. Environment Variables

**MCP_ENABLE_FILE_WATCHING**:
- Default: `true`
- Values: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- Purpose: Enable/disable automatic file watching and reindexing

**MCP_DEBUG** (future):
- Default: `false`
- Purpose: Enable debug logging for MCP server

### E. Related Files

**Files Modified**:
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py` (new)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py` (update)
- `/Users/masa/Projects/mcp-vector-search/docs/reference/MCP_SETUP.md` (update)

**Files Referenced**:
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py` (MCP server entry point)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/__main__.py` (Alternative entry point)
- `/Users/masa/Projects/mcp-vector-search/pyproject.toml` (Package configuration)

---

## Conclusion

This research provides a complete design for an automatic setup command that:

1. ✅ Uses native `claude mcp add` instead of manual `.mcp.json` creation
2. ✅ Combines init + index + MCP registration into single command
3. ✅ Detects platform capabilities and adapts accordingly
4. ✅ Provides graceful fallback when tools are unavailable
5. ✅ Offers excellent user experience with clear guidance

**Next Steps**:
- Implement `setup` command following the design
- Add comprehensive error handling
- Test on multiple platforms (macOS, Linux, Windows, WSL)
- Update documentation to recommend `setup` as primary method
- Consider deprecating manual `mcp init` in favor of `setup`

**Command to Use**:
```bash
claude mcp add --transport stdio mcp-vector-search \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}
```

---

**Research Completed**: 2025-11-30
**Status**: Design Complete, Ready for Implementation
**Confidence Level**: High (verified against web search, codebase analysis, and current documentation)
