# Chat Command Setup Guide

## Overview

The `mcp-vector-search chat` command uses AI to answer natural language questions about your code. It supports both **OpenAI** and **OpenRouter** APIs.

## Quick Setup

### Option 1: OpenAI (Recommended)

1. **Get API Key**: https://platform.openai.com/api-keys
2. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. **Test It**:
   ```bash
   mcp-vector-search chat "where is similarity_threshold set?"
   ```

### Option 2: OpenRouter

1. **Get API Key**: https://openrouter.ai/keys
2. **Set Environment Variable**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```
3. **Test It**:
   ```bash
   mcp-vector-search chat "where is similarity_threshold set?"
   ```

### Option 3: Interactive Setup

Run the setup wizard:
```bash
mcp-vector-search setup
```

Follow the prompts to configure your preferred provider.

## Usage Examples

### Basic Usage

```bash
# Ask where something is defined
mcp-vector-search chat "where is the LLMClient class defined?"

# Ask how something works
mcp-vector-search chat "how does the indexing process work?"

# Find implementation details
mcp-vector-search chat "show me the search ranking algorithm"
```

### Provider Selection

```bash
# Auto-detect (OpenAI preferred if both keys set)
mcp-vector-search chat "question"

# Force OpenAI
mcp-vector-search chat "question" --provider openai

# Force OpenRouter
mcp-vector-search chat "question" --provider openrouter
```

### Custom Models

```bash
# Use OpenAI's GPT-4o (more powerful, slower)
mcp-vector-search chat "complex question" --model gpt-4o

# Use Claude 3.5 Sonnet via OpenRouter
mcp-vector-search chat "complex question" --model anthropic/claude-3.5-sonnet
```

### Advanced Options

```bash
# Limit number of results
mcp-vector-search chat "find auth code" --limit 3

# Increase timeout for slow queries
mcp-vector-search chat "complex question" --timeout 60

# JSON output for scripting
mcp-vector-search chat "question" --json
```

## Configuration

### Environment Variables

**Priority Order** (highest to lowest):
1. `OPENAI_API_KEY` - OpenAI API key
2. `OPENROUTER_API_KEY` - OpenRouter API key

If both are set, **OpenAI is used by default** (can be overridden with `--provider` flag).

### Config File

API keys can also be stored in `~/.mcp-vector-search/config.json`:

```json
{
  "openai_api_key": "sk-...",
  "openrouter_api_key": "sk-or-...",
  "preferred_llm_provider": "openai"
}
```

**Set via Interactive Setup:**
```bash
mcp-vector-search setup
# Choose: "Configure LLM API Keys"
```

**Security**: Config file has `0600` permissions (owner read/write only).

## Provider Comparison

| Feature | OpenAI | OpenRouter |
|---------|--------|------------|
| **Default Model** | `gpt-4o-mini` | `anthropic/claude-3-haiku` |
| **Speed** | Very Fast | Very Fast |
| **Cost per 1M tokens** | ~$0.15 | ~$0.25 |
| **Setup URL** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **Pro** | Direct API, widely used | Access to Claude models |
| **Con** | Only OpenAI models | Proxy service |

**Recommendation**: Use **OpenAI** for cost and simplicity. Use **OpenRouter** if you need Claude models or unified billing.

## Troubleshooting

### No API Key Found

**Error**:
```
No LLM API key found.
```

**Solution**:
1. Get an API key (see Quick Setup above)
2. Set environment variable or run `mcp-vector-search setup`

### Invalid API Key

**Error**:
```
Invalid OpenAI API key. Please check OPENAI_API_KEY environment variable.
```

**Solution**:
1. Verify your API key is correct
2. Check if key has sufficient credits/quota
3. Regenerate key if needed

### Provider Not Available

**Error**:
```
OpenAI provider specified but OPENAI_API_KEY not found.
```

**Solution**:
- Either set the required API key, or
- Remove `--provider openai` to auto-detect, or
- Use `--provider openrouter` if you have that key

### Rate Limit Exceeded

**Error**:
```
OpenAI API rate limit exceeded. Please wait and try again.
```

**Solution**:
1. Wait a few seconds and retry
2. Upgrade your API plan if hitting limits frequently
3. Switch to different provider temporarily

### Request Timeout

**Error**:
```
LLM request timed out after 30.0 seconds.
```

**Solution**:
- Increase timeout: `--timeout 60`
- Simplify your question
- Check network connection

## How It Works

The chat command uses AI to:

1. **Generate Search Queries**: Converts your natural language question into 2-3 targeted search queries
2. **Search Codebase**: Executes semantic search against your indexed code
3. **Analyze Results**: Uses AI to rank and explain the most relevant results
4. **Display**: Shows code snippets with explanations

**Example Flow**:
```
User: "where is similarity_threshold set?"

AI generates queries:
  1. "similarity_threshold default value"
  2. "similarity_threshold configuration"
  3. "SemanticSearchEngine init threshold"

Searches codebase → 15 results found

AI analyzes → Top 5 most relevant

Displays results with explanations
```

## Best Practices

### Writing Good Questions

✅ **Good Questions**:
- "where is the LLMClient class defined?"
- "how does the indexing process work?"
- "show me error handling in the search engine"
- "what authentication methods are used?"

❌ **Poor Questions**:
- "fix this" (too vague)
- "everything about X" (too broad)
- "yes or no" (not leveraging semantic search)

### Tips for Better Results

1. **Be Specific**: "where is X defined" beats "what is X"
2. **Use Technical Terms**: "authentication flow" beats "login stuff"
3. **Ask About Implementation**: "how does X work" gets code-level explanations
4. **Iterate**: If results aren't great, rephrase your question

### Cost Management

- **Use Defaults**: Default models (`gpt-4o-mini`, `claude-3-haiku`) are very cheap (~$0.15-0.25 per 1M tokens)
- **Limit Results**: Use `--limit 3` for simpler questions
- **Cache Results**: Same questions use same API calls (consider implementing caching)
- **Monitor Usage**: Check your API dashboard regularly

## Advanced Configuration

### Set Preferred Provider

If you have both API keys, set a preference:

```bash
mcp-vector-search setup
# Choose: "Set preferred provider"
# Select: OpenAI or OpenRouter
```

This avoids needing `--provider` flag every time.

### Model Selection

**Fast & Cheap** (default):
- OpenAI: `gpt-4o-mini`
- OpenRouter: `anthropic/claude-3-haiku`

**More Powerful**:
- OpenAI: `gpt-4o` or `gpt-4-turbo`
- OpenRouter: `anthropic/claude-3.5-sonnet`

```bash
# Override default model
mcp-vector-search chat "question" --model gpt-4o
```

### Timeout Configuration

Default timeout is 30 seconds. Increase for complex queries:

```bash
mcp-vector-search chat "complex question" --timeout 60
```

## Migration from OpenRouter-only

If you were using OpenRouter before, **nothing changes**:

```bash
# Old setup (still works)
export OPENROUTER_API_KEY="sk-or-..."
mcp-vector-search chat "question"
```

**To add OpenAI**:
```bash
# Keep OpenRouter, add OpenAI
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."

# Now uses OpenAI by default (you can override)
mcp-vector-search chat "question"
mcp-vector-search chat "question" --provider openrouter
```

## FAQ

**Q: Which provider is better?**
A: For most users, **OpenAI** is simpler and slightly cheaper. Use **OpenRouter** if you specifically want Claude models.

**Q: Can I use both?**
A: Yes! Set both API keys and use `--provider` flag to choose, or set a preferred provider in config.

**Q: Will this break existing OpenRouter usage?**
A: No, it's fully backward compatible. Existing users don't need to change anything.

**Q: How much does it cost?**
A: Very cheap. Default models cost ~$0.15-0.25 per 1M tokens. A typical query uses ~1-5k tokens = $0.0002-0.001 per question.

**Q: Can I use local models?**
A: Not yet, but it's planned. For now, use OpenAI or OpenRouter.

**Q: What if I don't want to use the chat command?**
A: That's fine! The regular `mcp-vector-search search` command works without any API keys.

## Support

- **Documentation**: See `docs/development/OPENAI_API_INTEGRATION.md`
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
