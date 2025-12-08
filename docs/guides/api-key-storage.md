# OpenRouter API Key Storage Guide

This guide explains how to securely store and manage your OpenRouter API key for the `mcp-vector-search chat` command.

## Overview

The chat command uses AI (via OpenRouter) to answer questions about your code. It requires an API key, which can be provided in two ways:

1. **Environment variable** (recommended for security)
2. **Local config file** (convenient for development)

## Priority Order

The system checks for the API key in this order:

1. `OPENROUTER_API_KEY` environment variable (**highest priority**)
2. Local config file `.mcp-vector-search/config.json`

## Option 1: Environment Variable (Recommended)

**Best for:**
- Production environments
- Shared workstations
- Maximum security

**Setup:**

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export OPENROUTER_API_KEY='your-key-here'

# Reload your shell
source ~/.bashrc  # or source ~/.zshrc
```

**Pros:**
- Not stored in files (won't accidentally commit to git)
- Easy to change without modifying files
- Standard practice for API keys

**Cons:**
- Needs to be set in each shell/environment
- Requires shell profile modification

## Option 2: Local Config File (Convenient)

**Best for:**
- Personal development machines
- Quick setup
- Projects where you're the only user

**Setup:**

```bash
# Interactive setup (recommended)
mcp-vector-search setup --save-api-key

# Or manually edit config file
# File: .mcp-vector-search/config.json
# Add: "openrouter_api_key": "your-key-here"
```

**Pros:**
- Automatic loading in project directory
- No shell configuration needed
- Persists across terminal sessions

**Cons:**
- Stored as plain text (with 0600 permissions)
- Project-specific (needs setup per project)
- Risk of accidental exposure if sharing config files

## Security Features

### File Permissions
- Config file automatically set to `0600` (owner read/write only)
- Directory created with `0700` permissions (owner only)

### Gitignore Protection
- `.mcp-vector-search/` directory is gitignored by default
- API keys are never committed to version control

### Logging Safety
- API keys are masked in logs (only last 4 characters shown)
- Example log: `Set openrouter_api_key = ****1234`

### Sensitive Key Detection
- Keys named `api_key`, `token`, `secret`, etc. are automatically masked

## Usage Examples

### Check Current API Key Status

```bash
# Run setup to check status (won't overwrite existing key)
mcp-vector-search setup

# Output will show:
# ✅ OpenRouter API key found (ends with 1234)
#    Source: Environment variable
# OR
#    Source: Config file (/path/to/.mcp-vector-search/config.json)
```

### Save API Key Interactively

```bash
mcp-vector-search setup --save-api-key
```

You'll be prompted to:
1. Get an API key from https://openrouter.ai/keys
2. Paste the key (it will be saved securely)

### Remove API Key from Config

```bash
# Manually edit config file and remove the "openrouter_api_key" line
# Or delete the entire .mcp-vector-search/ directory and re-init
```

### Override Config with Environment Variable

```bash
# Config file has one key, but you want to use a different one temporarily
export OPENROUTER_API_KEY='different-key-here'
mcp-vector-search chat "your question"

# Environment variable takes priority over config file
```

## Getting an API Key

1. Visit https://openrouter.ai/keys
2. Sign up for a free account
3. Create a new API key
4. Copy the key (starts with `sk-or-...`)
5. Save using one of the methods above

**Free tier available!** OpenRouter offers free credits for testing.

## Troubleshooting

### "OpenRouter API key not found"

**Solution:**
Check if the key is set:

```bash
# Check environment variable
echo $OPENROUTER_API_KEY

# Check config file
cat .mcp-vector-search/config.json | grep openrouter_api_key
```

### "Permission denied" when saving to config

**Solution:**
Ensure the directory is writable:

```bash
chmod 700 .mcp-vector-search/
chmod 600 .mcp-vector-search/config.json
```

### Config file exists but key not loading

**Solution:**
Verify JSON format:

```bash
# Pretty-print config to check for errors
python -m json.tool .mcp-vector-search/config.json
```

### Want to use different keys for different projects

**Solution:**
Use the config file approach - each project has its own `.mcp-vector-search/config.json`:

```bash
cd project1/
mcp-vector-search setup --save-api-key  # Enter key for project 1

cd ../project2/
mcp-vector-search setup --save-api-key  # Enter key for project 2
```

### Want to use same key across all projects

**Solution:**
Use the environment variable approach - set once in shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENROUTER_API_KEY='your-key-here'
```

## Best Practices

### ✅ DO

- Use environment variables for production/shared environments
- Use config files for personal development machines
- Keep API keys secret (never commit to git)
- Rotate API keys periodically
- Use separate keys for different purposes (dev vs prod)

### ❌ DON'T

- Commit `.mcp-vector-search/config.json` with API keys to git
- Share API keys in chat/email (regenerate instead)
- Use production keys in development
- Hardcode API keys in scripts or code
- Store API keys in public repositories

## Configuration File Format

The config file is JSON format with project settings and optional API key:

```json
{
  "project_root": "/path/to/project",
  "index_path": "/path/to/project/.mcp-vector-search",
  "file_extensions": [".py", ".js", ".ts"],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "similarity_threshold": 0.5,
  "openrouter_api_key": "sk-or-v1-..."
}
```

**Note:** The `openrouter_api_key` field is optional and only needed for the chat command.

## Security Considerations

### File-Based Storage

**What we do:**
- Store in project-local directory (not global)
- Set restrictive file permissions (0600)
- Mask keys in logs and output

**What we don't do:**
- Encrypt keys at rest (relies on OS file permissions)
- Use OS keyring/keychain (complexity vs. benefit trade-off)
- Support key management systems (out of scope)

**Why this approach:**
- Simple and cross-platform
- No external dependencies
- Suitable for development use case
- Environment variables available for higher security

### Threat Model

**Protected against:**
- Accidental git commits (gitignored)
- Other users on same machine (file permissions)
- Accidental disclosure in logs (masked)

**NOT protected against:**
- Root/admin users (they can read all files)
- Malware with user-level access
- Physical access to unlocked machine
- Backup/sync tools that copy files

**For higher security needs:**
- Use environment variables only
- Use secrets management system (Vault, AWS Secrets Manager, etc.)
- Use short-lived tokens with rotation

## Related Commands

```bash
# Setup with API key prompt
mcp-vector-search setup --save-api-key

# Run chat command (uses saved key)
mcp-vector-search chat "where is the search function?"

# Check project status (shows if API key is configured)
mcp-vector-search setup

# View config file location
ls -la .mcp-vector-search/config.json
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/mcp-vector-search/issues
- Documentation: https://github.com/yourusername/mcp-vector-search/docs
- OpenRouter Docs: https://openrouter.ai/docs
