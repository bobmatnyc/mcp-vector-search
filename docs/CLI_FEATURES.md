# CLI Features

## "Did You Mean" Command Suggestions

The mcp-vector-search CLI includes intelligent command suggestions to help users when they make typos or use similar commands.

### How It Works

When you type a command that doesn't exist, the CLI will:

1. **Suggest similar commands** using fuzzy matching
2. **Provide common alternatives** for known typos
3. **List all available commands** if no close match is found

### Examples

```bash
# Typo in "search"
$ mcp-vector-search serach "authentication"
No such command 'serach'. Did you mean 'search'?

# Typo in "index"
$ mcp-vector-search indx
No such command 'indx'. Did you mean 'index'?

# Common abbreviations
$ mcp-vector-search stat
No such command 'stat'. Did you mean 'status'?

# Multiple suggestions
$ mcp-vector-search conf
No such command 'conf'. Did you mean 'config'?
```

### Common Typo Mappings

The CLI recognizes these common typos and abbreviations:

| Typo/Abbreviation | Suggested Command |
|-------------------|-------------------|
| `serach`, `seach`, `searh` | `search` |
| `indx`, `idx` | `index` |
| `stat`, `stats`, `info` | `status` |
| `conf`, `cfg`, `setting`, `settings` | `config` |
| `initialize`, `setup`, `start` | `init` |
| `monitor` | `watch` |
| `auto`, `automatic` | `auto-index` |
| `claude`, `server` | `mcp` |
| `example` | `demo` |
| `check`, `health` | `doctor` |
| `ver` | `version` |
| `h` | `--help` |

### Subcommand Support

The "did you mean" functionality also works for subcommands:

```bash
# MCP subcommands
$ mcp-vector-search mcp instal
No such command 'instal'. Did you mean 'install'?

# Search subcommands (legacy)
$ mcp-vector-search search-legacy simlar
No such command 'simlar'. Did you mean 'similar'?
```

### Technical Implementation

The feature uses the `click-didyoumean` package, which provides:

- **Fuzzy string matching** using difflib
- **Customizable similarity thresholds**
- **Integration with Click/Typer** command groups
- **Extensible suggestion system**

### Disabling Suggestions

If you prefer not to see suggestions, you can:

1. Use the exact command names
2. Use `--help` to see available commands
3. Set environment variable `CLICK_DIDYOUMEAN_DISABLE=1`

### Benefits

- **Improved user experience** - Less frustration with typos
- **Faster learning curve** - Discover commands through suggestions
- **Reduced documentation lookup** - Get hints directly in the CLI
- **Consistent with modern tools** - Similar to Git, Docker, etc.
