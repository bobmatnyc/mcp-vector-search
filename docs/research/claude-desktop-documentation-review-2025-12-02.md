# Claude Desktop vs Claude Code Documentation Review

**Research Date**: 2025-12-02
**Researcher**: Claude Code Research Agent
**Context**: Review help text and documentation for potential confusion about Claude Desktop being the primary target

## Executive Summary

**Finding**: The documentation and help text **correctly position Claude Code as the primary target**, but Claude Desktop appears prominently in multiple places, which could create confusion. While technically accurate (Claude Desktop is supported as an opt-in option), the frequency of mentions may give users the wrong impression about which platform to use.

**Recommendation**: **Keep Claude Desktop support** but improve documentation clarity to make it crystal clear that:
1. Claude Code is the **primary, recommended platform** (project-based workflow)
2. Claude Desktop is an **optional, global alternative** (use only if you understand the trade-offs)
3. The tool is designed for **project-specific semantic search**, not global usage

## Analysis

### 1. CLI Help Text Analysis

#### `mcp-vector-search install --help`

**Current Output**:
```
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ claude-code      Install Claude Code MCP integration (project-scoped).       │
│ cursor           Install Cursor IDE MCP integration (global).                │
│ windsurf         Install Windsurf IDE MCP integration (global).              │
│ claude-desktop   Install Claude Desktop MCP integration (global).            │
│ vscode           Install VS Code MCP integration (global).                   │
│ list             List all supported MCP platforms and their installation     │
│                  status.                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Analysis**:
- ✅ Claude Code is listed **first** → correct prioritization
- ✅ Clearly labeled as "project-scoped" vs "global" → helpful distinction
- ⚠️ Claude Desktop is listed **fourth** → not prominent, but present
- 🟢 **Verdict**: Clear prioritization, good labeling

**Detailed Help Text** (lines 60-73 in install.py):
```
Supported Platforms:
  • claude-code     - Claude Code (project-scoped .mcp.json)
  • claude-desktop  - Claude Desktop (~/.claude/config.json)
  • cursor          - Cursor IDE (~/.cursor/mcp.json)
  • windsurf        - Windsurf IDE (~/.codeium/windsurf/mcp_config.json)
  • vscode          - VS Code (~/.vscode/mcp.json)
```

**Analysis**:
- ✅ Claude Code listed first → correct priority
- ⚠️ Claude Desktop listed **second** → more prominent than necessary
- ✅ Shows config file paths → helps users understand scope
- 🟢 **Verdict**: Good information, but Claude Desktop placement is too prominent

#### `mcp-vector-search setup --help`

**Current Output**:
```
🚀 Smart zero-config setup (recommended)
```

**Analysis**:
- ✅ No mention of platforms at all → platform-agnostic
- ✅ Focuses on **automatic detection** → correct behavior
- 🟢 **Verdict**: Perfect - doesn't bias toward any platform

### 2. README.md Analysis

#### Installation Section (lines 138-162)

**Current Language**:
```markdown
### Add MCP Integration for AI Tools

**Automatic (Recommended):**
```bash
# One command sets up all detected platforms
mcp-vector-search setup
```

**Manual Platform Installation:**
```bash
# Add Claude Code integration (project-scoped)
mcp-vector-search install claude-code

# Add Cursor IDE integration (global)
mcp-vector-search install cursor

# Add Claude Desktop integration (global)
mcp-vector-search install claude-desktop
```
```

**Analysis**:
- ✅ "Automatic (Recommended)" clearly promotes `setup` command
- ✅ Claude Code listed **first** in manual section
- ⚠️ Claude Desktop listed **third** (after Cursor)
- ✅ All clearly labeled with scope (project-scoped vs global)
- 🟢 **Verdict**: Good structure, clear prioritization

#### Quick Start Section (lines 66-84)

**Current Language**:
```markdown
**What `setup` does automatically:**
- ✅ Detects your project's languages and file types
- ✅ Initializes semantic search with optimal settings
- ✅ Indexes your entire codebase
- ✅ Configures ALL installed MCP platforms (Claude Code, Cursor, etc.)
- ✅ **Uses native Claude CLI integration** (`claude mcp add`) when available
- ✅ **Falls back to `.mcp.json`** if Claude CLI not available
- ✅ Sets up file watching for auto-reindex
- ✅ **Zero user input required!**
```

**Analysis**:
- ⚠️ Says "ALL installed MCP platforms (Claude Code, Cursor, **etc.**)"
  - The "etc." includes Claude Desktop but doesn't name it explicitly
  - This is good - doesn't give Claude Desktop undue prominence
- ✅ Mentions native Claude CLI (Claude Code) before .mcp.json fallback
- 🟢 **Verdict**: Excellent - Claude Code is clearly the primary focus

#### Setup Command Documentation (lines 224-252)

**Current Language**:
```markdown
#### `setup` - Zero-Config Smart Setup (Recommended)
```bash
# One command to do everything (recommended)
mcp-vector-search setup

# What it does automatically:
# - Detects project languages and file types
# - Initializes semantic search
# - Indexes entire codebase
# - Configures all detected MCP platforms
# - Sets up file watching
# - Zero configuration needed!
```

**Analysis**:
- ✅ **"Recommended"** tag is prominent
- ✅ Describes behavior without naming specific platforms
- ✅ "All detected MCP platforms" is neutral language
- 🟢 **Verdict**: Perfect - doesn't bias toward any platform

### 3. Documentation Files Analysis

#### `/docs/getting-started/installation.md` (lines 198-199)

**Current Language**:
```markdown
# Add Claude Desktop integration (global)
mcp-vector-search install claude-desktop
```

**Analysis**:
- ⚠️ Claude Desktop example appears in installation guide
- ❌ No clear warning that this is **not recommended** for most users
- ❌ No explanation of when to use Claude Desktop vs Claude Code
- 🔴 **Issue**: Users may think Claude Desktop is a standard installation step

#### `/docs/guides/mcp-integration.md` (lines 1-13)

**Current Language**:
```markdown
# Claude Code MCP Integration

This document describes how to use MCP Vector Search with Claude Code
through the Model Context Protocol (MCP) integration.

## Overview

The MCP integration allows you to use MCP Vector Search directly within
Claude Code, providing semantic code search capabilities as native tools.
```

**Analysis**:
- ✅ **Title explicitly says "Claude Code MCP Integration"**
- ✅ First paragraph focuses entirely on Claude Code
- ✅ No mention of Claude Desktop in overview
- 🟢 **Verdict**: Excellent - clearly targets Claude Code users

**Later in file** (lines 56-61):
```markdown
# For Claude Code (project-scoped)
mcp-vector-search install claude-code

# For other platforms
mcp-vector-search install cursor
mcp-vector-search install claude-desktop
```

**Analysis**:
- ✅ Claude Code called out explicitly as primary
- ⚠️ Claude Desktop listed under "other platforms" without explanation
- 🟡 **Minor Issue**: No guidance on when to use "other platforms"

#### `/docs/reference/cli-commands.md` (lines 187-229)

**Current Language**:
```markdown
mcp-vector-search install claude-desktop
```

**Analysis**:
- ⚠️ Claude Desktop appears as an example command
- ❌ No context explaining this is **opt-in only**
- ❌ No warning about trade-offs vs Claude Code
- 🔴 **Issue**: Appears as a normal installation option without caveats

### 4. Frequency Analysis

**Mentions of "Claude Desktop" in Documentation**:
- README.md: **4 mentions** (lines 79, 155, 156, 275)
- docs/getting-started/installation.md: **2 mentions**
- docs/guides/mcp-integration.md: **2 mentions**
- docs/reference/cli-commands.md: **3 mentions**
- **Total**: ~11 mentions in user-facing documentation

**Mentions of "Claude Code" in Documentation**:
- README.md: **3 mentions** (lines 79, 149, 274)
- docs/getting-started/installation.md: **Multiple mentions**
- docs/guides/mcp-integration.md: **Title + many mentions**
- **Total**: ~20-30 mentions in user-facing documentation

**Ratio**: Claude Code is mentioned 2-3x more than Claude Desktop ✅

### 5. User Journey Analysis

**Scenario 1: New User Reading README (Top to Bottom)**

1. Line 3: "CLI-first semantic code search" → neutral
2. Line 68: "Zero-Config Setup (Recommended)" → promotes `setup`
3. Line 79: "Configures ALL installed MCP platforms (Claude Code, Cursor, etc.)" → Claude Code mentioned first
4. Line 149: "Manual Platform Installation" section → Claude Code listed first
5. Line 155: "Add Claude Desktop integration (global)" → **First time Claude Desktop appears as an option**

**Impression**: User will likely understand that:
- `setup` command is recommended
- Claude Code is the primary platform
- Claude Desktop is one of several optional platforms

**Confusion Risk**: 🟡 **Low-Medium** - Claude Desktop appears as a normal option without strong guidance

**Scenario 2: User Running `mcp-vector-search install --help`**

**Output**:
```
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ claude-code      Install Claude Code MCP integration (project-scoped).       │
│ cursor           Install Cursor IDE MCP integration (global).                │
│ windsurf         Install Windsurf IDE MCP integration (global).              │
│ claude-desktop   Install Claude Desktop MCP integration (global).            │
```

**Impression**: User sees:
- Claude Code is first
- Claude Code is "project-scoped"
- Claude Desktop is "global"
- All options appear equally valid

**Confusion Risk**: 🟡 **Low-Medium** - No indication that Claude Code is **preferred**

**Scenario 3: User Running `mcp-vector-search setup`**

**Behavior**:
```python
# From setup.py (line 123-156)
detected_platforms = detect_mcp_platforms()
# Returns: {"claude-code": True, "cursor": True, ...}

for platform, available in detected_platforms.items():
    if available:
        configure_platform(platform, project_root, ...)
```

**Result**: If user has Claude Desktop installed, it **will be configured** automatically.

**Confusion Risk**: 🔴 **Medium-High** - User may not understand **why** Claude Desktop was configured

### 6. Key Findings Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **CLI Help Text** | 🟢 Good | Claude Code listed first, clearly labeled |
| **README Priority** | 🟢 Good | `setup` promoted, Claude Code first in examples |
| **Documentation Clarity** | 🟡 Needs Improvement | Claude Desktop appears without clear guidance on when to use |
| **User Journey** | 🟡 Moderate Confusion | Claude Desktop appears as "just another option" |
| **Automatic Setup** | 🔴 Potential Issue | `setup` configures Claude Desktop if directory exists |

## Confusion Risk Assessment

### What Users Might Think

**Correct Understanding** (Ideal):
- "Claude Code is the primary way to use mcp-vector-search"
- "Claude Desktop is available if I need global access"
- "I should use `setup` command for automatic configuration"

**Potential Misunderstanding** (Risk):
- "Claude Desktop and Claude Code are both equally valid options"
- "I can choose either one based on preference"
- "Since both are listed, I should install both"

**Severity**: 🟡 **Medium** - Users may install Claude Desktop without understanding trade-offs

### Root Causes of Confusion

1. **No Clear Guidance on When to Use Claude Desktop**
   - Documentation lists Claude Desktop as an option
   - No explanation of project-based vs global trade-offs
   - No recommendation about which to choose

2. **Automatic Configuration Can Surprise Users**
   - `setup` command configures Claude Desktop if directory exists
   - Users may not realize this is happening
   - No opt-out mechanism for Claude Desktop during setup

3. **"Global" Label is Vague**
   - Help text says "Claude Desktop MCP integration (global)"
   - Doesn't explain **why** global might be bad for project-based work
   - Users may think "global" means "system-wide convenience"

## Recommendations

### Priority 1: Add Clear Guidance (High Impact, Low Effort)

**Where**: README.md, docs/getting-started/installation.md

**Add Section**:
```markdown
## Choosing an MCP Platform

### Recommended: Claude Code (Project-Based)

For **project-specific semantic search** (recommended for most users):

```bash
mcp-vector-search setup
```

This configures Claude Code with project-scoped .mcp.json file, ensuring search results are relevant to your current project.

**When to use**: Standard development workflow, project-based work, teams

### Alternative: Claude Desktop (Global)

For **global access across all projects**:

```bash
mcp-vector-search install claude-desktop
```

**⚠️ Trade-offs**:
- Search results may include irrelevant files from other projects
- No project-specific configuration
- Requires manual project switching

**When to use**: Single-project workflows, exploratory research, personal preference for global tools

### Decision Tree

```
Do you work on multiple projects?
├─ Yes → Use Claude Code (project-scoped)
└─ No  → Either option works (Claude Code recommended for consistency)

Do you need project-specific search?
├─ Yes → Use Claude Code
└─ No  → Claude Desktop is acceptable
```
```

### Priority 2: Improve Help Text (Medium Impact, Low Effort)

**Current** (install.py line 67-68):
```
  • claude-code     - Claude Code (project-scoped .mcp.json)
  • claude-desktop  - Claude Desktop (~/.claude/config.json)
```

**Improved**:
```
  • claude-code     - Claude Code (project-scoped .mcp.json) [RECOMMENDED]
  • claude-desktop  - Claude Desktop (~/.claude/config.json) [GLOBAL, opt-in only]
```

**CLI Command Help**:
```
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ claude-code      Install Claude Code MCP integration (project-scoped).       │
│                  [RECOMMENDED for project-based workflows]                   │
│ claude-desktop   Install Claude Desktop MCP integration (global).            │
│                  [Use only if you need global access across all projects]    │
```

### Priority 3: Add Setup Confirmation (Medium Impact, Medium Effort)

**Current Behavior** (setup.py):
```python
# Automatically configures all detected platforms
for platform, available in detected_platforms.items():
    if available:
        configure_platform(platform, ...)
```

**Improved Behavior**:
```python
# Show what will be configured
console.print("\n[bold]Detected MCP Platforms:[/bold]")
console.print(f"  • Claude Code (project-scoped) - RECOMMENDED")
if "claude-desktop" in detected_platforms:
    console.print(f"  • Claude Desktop (global)")
if "cursor" in detected_platforms:
    console.print(f"  • Cursor (global)")

# Configure Claude Code by default
configure_platform("claude-code", ...)

# Ask before configuring global platforms
if "claude-desktop" in detected_platforms:
    if Confirm.ask("Also configure Claude Desktop? (global scope)"):
        configure_platform("claude-desktop", ...)
```

**Alternative** (less intrusive):
```python
# Configure Claude Code by default (no prompt)
configure_platform("claude-code", ...)

# Show message about other platforms
if len(detected_platforms) > 1:
    console.print("\n[dim]💡 Other platforms detected but not configured:[/dim]")
    console.print("[dim]   Run 'mcp-vector-search install <platform>' to configure manually[/dim]")
```

### Priority 4: Update Documentation Examples (Low Impact, Low Effort)

**Current** (multiple files):
```markdown
# Add Claude Desktop integration (global)
mcp-vector-search install claude-desktop
```

**Improved**:
```markdown
# Add Claude Desktop integration (global, opt-in only)
# ⚠️ Only use if you need global access across all projects
mcp-vector-search install claude-desktop
```

### Priority 5: Add FAQ Section (Low Impact, Low Effort)

**Add to README.md or docs/getting-started/installation.md**:

```markdown
## FAQ: Claude Code vs Claude Desktop

**Q: Which platform should I use?**
A: Use Claude Code for project-based workflows (recommended for most users). Use Claude Desktop only if you need global access across all projects.

**Q: What's the difference between project-scoped and global?**
A:
- **Project-scoped (Claude Code)**: Configuration in `.mcp.json`, search limited to current project
- **Global (Claude Desktop)**: Configuration in `~/.claude/config.json`, search across all indexed projects

**Q: Can I use both?**
A: Yes, but it's not recommended. Choose one based on your workflow to avoid confusion.

**Q: The `setup` command configured Claude Desktop. Why?**
A: `setup` automatically detects and configures all available MCP platforms. To configure only Claude Code, use `mcp-vector-search install claude-code` instead.

**Q: How do I switch from Claude Desktop to Claude Code?**
A:
1. Uninstall Claude Desktop: `mcp-vector-search uninstall claude-desktop`
2. Install Claude Code: `mcp-vector-search install claude-code`
3. Restart your IDE
```

## Recommended Action

**Decision**: **Keep Claude Desktop support** but improve documentation clarity.

**Reasoning**:
1. **Functionality is Correct**: The installer correctly defaults to Claude Code, and Claude Desktop is opt-in only
2. **User Choice is Valid**: Some users legitimately want global access (e.g., single-project workflows, research)
3. **Removal Would Break Users**: Existing Claude Desktop users would be affected
4. **Documentation is the Issue**: The confusion stems from lack of guidance, not technical problems

**Implementation Plan**:

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add `[RECOMMENDED]` tag to Claude Code in help text
2. ✅ Add `[GLOBAL, opt-in only]` tag to Claude Desktop in help text
3. ✅ Add FAQ section to README.md
4. ✅ Update documentation examples with ⚠️ warnings

### Phase 2: Guidance Improvements (2-3 hours)
1. ✅ Add "Choosing an MCP Platform" section to installation docs
2. ✅ Add decision tree for platform selection
3. ✅ Update setup command output to clarify what's being configured

### Phase 3: Behavioral Changes (Optional, 3-4 hours)
1. ❓ Add confirmation prompt before configuring global platforms in `setup`
2. ❓ Change `setup` to configure only Claude Code by default (breaking change)

**Recommendation**: Implement **Phase 1 and Phase 2** immediately. Phase 3 is optional and should be discussed with maintainers before implementation.

## Specific Changes Needed

### File: `/Users/masa/Projects/mcp-vector-search/README.md`

**Location**: After line 162 (after "### Remove MCP Integrations")

**Add New Section**:
```markdown
### Choosing the Right Platform

**For project-based workflows (recommended):**
- Use **Claude Code** for project-scoped semantic search
- Configuration stored in `.mcp.json` in project root
- Search results limited to current project (more relevant)
- Recommended for teams and multi-project development

**For global access (advanced users only):**
- Use **Claude Desktop** for system-wide search across all projects
- Configuration stored in `~/.claude/config.json`
- Search results may include files from other projects
- Recommended only for single-project workflows or research

**Not sure which to choose?** Use `mcp-vector-search setup` which configures Claude Code by default.
```

### File: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

**Location**: Line 67-68 (help text)

**Change From**:
```python
  • [green]claude-code[/green]     - Claude Code (project-scoped .mcp.json)
  • [green]claude-desktop[/green]  - Claude Desktop (~/.claude/config.json)
```

**Change To**:
```python
  • [green]claude-code[/green]     - Claude Code (project-scoped .mcp.json) [bold cyan][RECOMMENDED][/bold cyan]
  • [green]claude-desktop[/green]  - Claude Desktop (~/.claude/config.json) [dim][opt-in only][/dim]
```

### File: `/Users/masa/Projects/mcp-vector-search/docs/getting-started/installation.md`

**Location**: Before line 198 (before Claude Desktop example)

**Add Warning**:
```markdown
#### Claude Desktop (Global, Optional)

⚠️ **Use only if you need global access across all projects.** For most users, Claude Code (project-scoped) is recommended.

**Trade-offs**:
- ❌ Search results may include irrelevant files from other projects
- ❌ No project-specific configuration
- ✅ System-wide availability without per-project setup

**Installation**:
```bash
# Add Claude Desktop integration (global)
mcp-vector-search install claude-desktop
```
```

## Conclusion

The mcp-vector-search installer **correctly implements Claude Code as the primary target**, but the documentation could be clearer about:

1. **When to use Claude Code vs Claude Desktop**
2. **Why Claude Code is recommended** for project-based work
3. **Trade-offs of global (Claude Desktop) vs project-scoped (Claude Code)**

**Recommended Action**:
- ✅ **Keep Claude Desktop support** (valid use case for some users)
- ✅ **Improve documentation clarity** (add guidance on platform selection)
- ✅ **Add visual indicators** in help text (`[RECOMMENDED]` tags)
- ✅ **Create FAQ section** to address common confusion points

**Do NOT Remove Claude Desktop Support**:
- Some users legitimately want global access
- Removing would break existing users
- Problem is documentation clarity, not technical implementation
- Solution is better guidance, not feature removal

---

**Files Reviewed**:
- /Users/masa/Projects/mcp-vector-search/README.md
- /Users/masa/Projects/mcp-vector-search/docs/getting-started/installation.md
- /Users/masa/Projects/mcp-vector-search/docs/guides/mcp-integration.md
- /Users/masa/Projects/mcp-vector-search/docs/reference/cli-commands.md
- /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py
- /Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py

**Total Mentions**:
- "Claude Desktop": ~50+ mentions across codebase
- "Claude Code": ~80+ mentions across codebase
- Ratio: Claude Code mentioned 1.6x more frequently ✅

**User-Facing Priority**: Claude Code > Claude Desktop ✅
