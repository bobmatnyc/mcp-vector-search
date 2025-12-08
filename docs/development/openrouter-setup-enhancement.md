# OpenRouter API Key Setup Enhancement

**Date**: December 8, 2025
**Status**: Completed
**Related Issue**: Enhance setup command with OpenRouter API key configuration

## Summary

Enhanced the `mcp-vector-search setup` command to automatically check for OpenRouter API key configuration and provide clear setup instructions when the key is not found. This improves the onboarding experience for users who want to use the chat command.

## Changes Made

### 1. Added OpenRouter API Key Check Function

**File**: `src/mcp_vector_search/cli/commands/setup.py`

Added new function `setup_openrouter_api_key()` that:
- Checks if `OPENROUTER_API_KEY` environment variable is set
- Returns `True` if configured, `False` otherwise
- Displays helpful setup instructions when not configured
- Shows success message when API key is found

**Implementation**:
```python
def setup_openrouter_api_key() -> bool:
    """Check and optionally set up OpenRouter API key for chat command.

    Returns:
        True if API key is configured, False otherwise
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if api_key:
        print_success("   ✅ OpenRouter API key found")
        print_info("      Chat command is ready to use!")
        return True

    # API key not found - show setup instructions
    print_info("   ℹ️  OpenRouter API key not found")
    print_info("")
    print_info("   The 'chat' command uses AI to answer questions about your code.")
    print_info("   It requires an OpenRouter API key (free tier available).")
    print_info("")
    print_info("   [bold cyan]To enable the chat command:[/bold cyan]")
    print_info("   1. Get a free API key: [cyan]https://openrouter.ai/keys[/cyan]")
    print_info("   2. Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
    print_info("      [yellow]export OPENROUTER_API_KEY='your-key-here'[/yellow]")
    print_info("   3. Reload your shell: [yellow]source ~/.bashrc[/yellow]")
    print_info("")
    print_info("   [dim]💡 You can skip this for now - search still works![/dim]")

    return False
```

### 2. Integrated into Setup Flow

**File**: `src/mcp_vector_search/cli/commands/setup.py`

Added new Phase 6 in the setup workflow:
- Runs after MCP integration configuration
- Calls `setup_openrouter_api_key()`
- Tracks result and updates summary accordingly
- Adds chat command to next steps if configured

**Changes**:
- Added Phase 6: OpenRouter API Key Setup (Optional)
- Updated completion summary to include API key status
- Conditionally adds chat command to "Next Steps" section
- Imports `os` module for environment variable access

### 3. Enhanced Main Help Text

**File**: `src/mcp_vector_search/cli/main.py`

Completely rewrote the main CLI help text to:
- Feature chat command prominently in "QUICK START" section
- Add dedicated "AI CHAT SETUP" section with clear instructions
- Include practical examples of both search and chat commands
- Organize commands by importance (main commands vs. more commands)

**New Help Structure**:
```
🔍 MCP Vector Search - Semantic Code Search CLI

QUICK START:
  mcp-vector-search setup           # One-time setup (recommended)
  mcp-vector-search search "query"  # Search by meaning
  mcp-vector-search chat "question" # Ask AI about your code

MAIN COMMANDS:
  setup, search, chat, status, visualize

AI CHAT SETUP:
  1. Get key: https://openrouter.ai/keys
  2. Set: export OPENROUTER_API_KEY='your-key'

EXAMPLES:
  [Practical examples for both search and chat]

MORE COMMANDS:
  [Secondary commands: install, uninstall, init, demo, etc.]
```

## User Experience Flow

### Scenario 1: User Without API Key

```bash
$ mcp-vector-search setup

# ... initialization and indexing ...

🤖 Chat Command Setup (Optional)...
ℹ️  OpenRouter API key not found

   The 'chat' command uses AI to answer questions about your code.
   It requires an OpenRouter API key (free tier available).

   To enable the chat command:
   1. Get a free API key: https://openrouter.ai/keys
   2. Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):
      export OPENROUTER_API_KEY='your-key-here'
   3. Reload your shell: source ~/.bashrc

   💡 You can skip this for now - search still works!

🎉 Setup Complete!

What was set up:
  ✅ Vector database initialized
  ✅ Codebase indexed and searchable
  ✅ 1 MCP platform(s) configured
  ✅ File watching enabled

🚀 Ready to Use
  1. mcp-vector-search search 'your query' - Search your code
  2. mcp-vector-search status - Check project status
```

### Scenario 2: User With API Key

```bash
$ export OPENROUTER_API_KEY='sk-or-...'
$ mcp-vector-search setup

# ... initialization and indexing ...

🤖 Chat Command Setup (Optional)...
✅ OpenRouter API key found
   Chat command is ready to use!

🎉 Setup Complete!

What was set up:
  ✅ Vector database initialized
  ✅ Codebase indexed and searchable
  ✅ 1 MCP platform(s) configured
  ✅ File watching enabled
  ✅ OpenRouter API configured for chat command

🚀 Ready to Use
  1. mcp-vector-search search 'your query' - Search your code
  2. mcp-vector-search chat 'question' - Ask AI about your code
  3. mcp-vector-search status - Check project status
```

## Testing

### Manual Testing

Created test script: `tests/manual/test_openrouter_setup.py`

**Test Coverage**:
1. ✅ Function returns `False` when API key not set
2. ✅ Function returns `True` when API key is set
3. ✅ Proper output formatting and messages
4. ✅ No exceptions raised

**Test Results**:
```bash
$ uv run python tests/manual/test_openrouter_setup.py

================================================================================
TEST 1: API Key NOT Set
================================================================================
ℹ️  OpenRouter API key not found
[... instructions display ...]
✓ Test passed: Returns False when API key not set

================================================================================
TEST 2: API Key SET
================================================================================
✅ OpenRouter API key found
✓ Test passed: Returns True when API key is set

================================================================================
ALL TESTS PASSED!
================================================================================
```

### Code Quality Checks

All quality checks passed:

```bash
# Formatting
$ uv run black src/mcp_vector_search/cli/commands/setup.py src/mcp_vector_search/cli/main.py --check
All done! ✨ 🍰 ✨
2 files would be left unchanged.

# Linting
$ uv run ruff check src/mcp_vector_search/cli/commands/setup.py src/mcp_vector_search/cli/main.py
All checks passed!

# Type Checking
$ uv run mypy src/mcp_vector_search/cli/commands/setup.py --ignore-missing-imports
Success: no issues found
```

## Files Modified

1. **src/mcp_vector_search/cli/commands/setup.py**
   - Added `import os` for environment variable access
   - Added `setup_openrouter_api_key()` function (38 lines)
   - Integrated function into setup workflow (Phase 6)
   - Updated completion summary and next steps
   - Net LOC impact: +45 lines

2. **src/mcp_vector_search/cli/main.py**
   - Rewrote main app help text for better UX
   - Reorganized command presentation
   - Added "AI CHAT SETUP" section
   - Added practical examples
   - Net LOC impact: +10 lines

3. **CHANGELOG.md**
   - Added entry for unreleased changes
   - Documented all enhancements

4. **tests/manual/test_openrouter_setup.py** (NEW)
   - Created manual test script
   - Validates function behavior
   - Net LOC impact: +65 lines

## Benefits

### For New Users
- Clear guidance during first-time setup
- No confusion about why chat command isn't working
- Direct links to get API keys
- Step-by-step instructions

### For Existing Users
- Non-intrusive (optional phase)
- Confirmation that chat is ready if configured
- No changes to workflow if API key already set

### For Documentation
- Self-documenting through CLI help
- Reduced support burden
- Clear distinction between search (no API key) and chat (requires API key)

## Design Decisions

### 1. Optional Phase (Not Required)

**Decision**: Make OpenRouter setup optional, not blocking
**Rationale**:
- Search functionality works without API key
- Users may not want/need chat immediately
- Reduces friction during initial setup

### 2. Environment Variable Only

**Decision**: Only check environment variable, don't prompt for input
**Rationale**:
- Consistent with common CLI patterns
- Secure (no plaintext in config files)
- Works well with shell profiles
- No interactive prompts to interrupt automation

### 3. Informational Output

**Decision**: Use `print_info()` for instructions, not warnings
**Rationale**:
- Not an error condition
- Not a warning (it's expected for new users)
- Informational guidance appropriate

### 4. Phase Placement

**Decision**: Run after MCP configuration, before completion
**Rationale**:
- Logical flow: setup core functionality first
- Non-critical feature doesn't block main setup
- Natural place for "optional enhancements"

## Future Enhancements

Potential improvements for future versions:

1. **API Key Validation**
   - Test connection to OpenRouter
   - Validate key format
   - Check account credits/limits

2. **Model Selection Guidance**
   - Suggest optimal model based on use case
   - Explain trade-offs (speed vs. quality)
   - Show pricing information

3. **Interactive Setup**
   - Optionally prompt for API key during setup
   - Automatically add to shell profile
   - Test immediately after configuration

4. **Alternative Providers**
   - Support for other LLM providers (Anthropic, OpenAI)
   - Multiple provider configuration
   - Automatic fallback between providers

## Conclusion

Successfully implemented OpenRouter API key setup guidance in the `setup` command with:
- ✅ Non-intrusive user experience
- ✅ Clear, actionable instructions
- ✅ Proper integration into setup flow
- ✅ Enhanced main help text
- ✅ Comprehensive testing
- ✅ All quality checks passed

The enhancement improves discoverability of the chat command and reduces confusion for new users while maintaining a smooth setup experience.
