# OpenAI API Integration

**Date**: December 8, 2025
**Status**: ✅ Completed

## Overview

Added OpenAI API support alongside OpenRouter for the `chat` command, giving users flexibility to choose their preferred LLM provider.

## What Changed

### 1. **Core LLM Client** (`src/mcp_vector_search/core/llm_client.py`)

**Changes:**
- Added support for both OpenAI and OpenRouter APIs
- Auto-detection of provider based on available API keys
- Provider-specific defaults:
  - OpenAI: `gpt-4o-mini` (comparable to `claude-3-haiku`)
  - OpenRouter: `anthropic/claude-3-haiku`

**Key Features:**
- **Provider Selection Priority**:
  1. Explicit `provider` parameter
  2. Preferred provider from config
  3. Auto-detect: OpenAI if available, otherwise OpenRouter
- **Backward Compatibility**: Existing `api_key` parameter still works (assumes OpenRouter)
- **Environment Variables**:
  - `OPENAI_API_KEY` for OpenAI
  - `OPENROUTER_API_KEY` for OpenRouter

**New Class Attributes:**
```python
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openrouter": "anthropic/claude-3-haiku"
}

API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions"
}
```

**Constructor Signature:**
```python
def __init__(
    self,
    api_key: str | None = None,  # Deprecated
    model: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
    provider: LLMProvider | None = None,  # NEW
    openai_api_key: str | None = None,  # NEW
    openrouter_api_key: str | None = None,  # NEW
) -> None:
```

### 2. **Configuration Storage** (`src/mcp_vector_search/core/config_utils.py`)

**New Functions:**
- `get_openai_api_key(config_dir)` - Get OpenAI key from env or config
- `save_openai_api_key(api_key, config_dir)` - Save OpenAI key to config
- `delete_openai_api_key(config_dir)` - Delete OpenAI key from config
- `get_preferred_llm_provider(config_dir)` - Get preferred provider
- `save_preferred_llm_provider(provider, config_dir)` - Set preferred provider

**Priority Order:**
1. Environment variable (e.g., `OPENAI_API_KEY`)
2. Config file (`~/.mcp-vector-search/config.json`)

### 3. **Settings Schema** (`src/mcp_vector_search/config/settings.py`)

**New Fields:**
```python
openai_api_key: str | None = Field(
    default=None,
    description="OpenAI API key for chat command (optional, can also use env var)"
)
preferred_llm_provider: str | None = Field(
    default=None,
    description="Preferred LLM provider: 'openai' or 'openrouter' (auto-detect if not set)"
)
```

### 4. **Chat Command** (`src/mcp_vector_search/cli/commands/chat.py`)

**New Options:**
- `--provider`: Force specific provider (`openai` or `openrouter`)

**Example Usage:**
```bash
# Auto-detect provider (OpenAI preferred if both keys set)
mcp-vector-search chat "where is similarity_threshold set?"

# Force OpenAI
mcp-vector-search chat "question" --provider openai

# Force OpenRouter
mcp-vector-search chat "question" --provider openrouter

# Custom model
mcp-vector-search chat "question" --model gpt-4o
mcp-vector-search chat "question" --model anthropic/claude-3.5-sonnet
```

**Updated Help Text:**
- Now mentions both OpenAI and OpenRouter
- Shows examples for both providers
- Clarifies auto-detection behavior

### 5. **Setup Command** (`src/mcp_vector_search/cli/commands/setup.py`)

**New Function:** `setup_llm_api_keys(project_root, interactive=True)`

**Interactive Setup Menu:**
```
LLM API Key Setup

The 'chat' command uses AI to answer questions about your code.
You can use OpenAI or OpenRouter (or both).

Current Configuration:
  • OpenAI: sk-or...abc1 (config file)
  • OpenRouter: not configured
  • Preferred: openai

Options:
  1. Configure OpenAI (recommended, fast & cheap)
  2. Configure OpenRouter
  3. Set preferred provider
  4. Skip / Keep current

Select option (1-4):
```

**Helper Function:** `_setup_single_provider()`
- Handles setup for individual providers
- Supports adding, updating, and clearing API keys
- Shows obfuscated keys for security

## Configuration Storage

API keys are stored in `~/.mcp-vector-search/config.json`:

```json
{
  "openai_api_key": "sk-...",
  "openrouter_api_key": "sk-or-...",
  "preferred_llm_provider": "openai"
}
```

**Security:**
- File permissions: `0600` (owner read/write only)
- Keys are obfuscated in UI: `sk-or...abc1234`
- Environment variables take precedence

## Provider Comparison

| Feature | OpenAI | OpenRouter |
|---------|--------|------------|
| **Default Model** | `gpt-4o-mini` | `anthropic/claude-3-haiku` |
| **Speed** | Fast | Fast |
| **Cost** | Very cheap | Very cheap |
| **Setup** | https://platform.openai.com/api-keys | https://openrouter.ai/keys |
| **Environment Variable** | `OPENAI_API_KEY` | `OPENROUTER_API_KEY` |

**Model Equivalents:**
- OpenRouter `anthropic/claude-3-haiku` ≈ OpenAI `gpt-4o-mini`
- OpenRouter `anthropic/claude-3.5-sonnet` ≈ OpenAI `gpt-4o`

## Migration Guide

### From OpenRouter-only to Both Providers

**Before (OpenRouter only):**
```bash
export OPENROUTER_API_KEY="sk-or-..."
mcp-vector-search chat "question"
```

**After (with OpenAI support):**
```bash
# Option A: Use OpenAI instead
export OPENAI_API_KEY="sk-..."
mcp-vector-search chat "question"  # Auto-detects OpenAI

# Option B: Use both (OpenAI preferred by default)
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
mcp-vector-search chat "question"  # Uses OpenAI

# Option C: Use both, force OpenRouter
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
mcp-vector-search chat "question" --provider openrouter
```

### Backward Compatibility

**Old code still works:**
```python
from mcp_vector_search.core.llm_client import LLMClient

# Old way (still supported, assumes OpenRouter)
client = LLMClient(api_key="sk-or-...")

# New way (explicit provider)
client = LLMClient(
    openrouter_api_key="sk-or-...",
    provider="openrouter"
)
```

## Testing

### Manual Testing Checklist

- [ ] **OpenAI API Key Setup**
  - [ ] Environment variable: `export OPENAI_API_KEY="sk-..."`
  - [ ] Config file: `mcp-vector-search setup` → Option 1
  - [ ] Verify chat works: `mcp-vector-search chat "test"`

- [ ] **OpenRouter API Key Setup**
  - [ ] Environment variable: `export OPENROUTER_API_KEY="sk-or-..."`
  - [ ] Config file: `mcp-vector-search setup` → Option 2
  - [ ] Verify chat works: `mcp-vector-search chat "test"`

- [ ] **Provider Selection**
  - [ ] Auto-detect (OpenAI preferred if both set)
  - [ ] Force OpenAI: `--provider openai`
  - [ ] Force OpenRouter: `--provider openrouter`
  - [ ] Set preferred in setup: Option 3

- [ ] **Error Handling**
  - [ ] No API key → Clear error message with instructions
  - [ ] Invalid provider → Validation error
  - [ ] Wrong API key → 401 error with helpful message

- [ ] **Backward Compatibility**
  - [ ] Existing OpenRouter users can continue without changes
  - [ ] Old `api_key` parameter still works

### Example Test Commands

```bash
# Test OpenAI
export OPENAI_API_KEY="your-key"
mcp-vector-search chat "where is the similarity_threshold parameter set?"

# Test OpenRouter
export OPENROUTER_API_KEY="your-key"
mcp-vector-search chat "where is the similarity_threshold parameter set?" --provider openrouter

# Test auto-detection (both keys set)
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
mcp-vector-search chat "test"  # Should use OpenAI

# Test provider override
mcp-vector-search chat "test" --provider openrouter  # Should use OpenRouter
```

## Files Modified

1. **`src/mcp_vector_search/core/llm_client.py`** - Multi-provider LLM client
2. **`src/mcp_vector_search/core/config_utils.py`** - OpenAI key storage functions
3. **`src/mcp_vector_search/config/settings.py`** - New config fields
4. **`src/mcp_vector_search/cli/commands/setup.py`** - Interactive setup for both providers
5. **`src/mcp_vector_search/cli/commands/chat.py`** - Provider selection support

## Design Decisions

### Why Auto-detect OpenAI First?

**Decision**: When both API keys are present, prefer OpenAI by default.

**Rationale**:
1. **Wider Adoption**: More developers already have OpenAI accounts
2. **Simpler**: Direct API vs. proxy service
3. **Performance**: No proxy overhead
4. **User Expectation**: Most expect OpenAI when both are available

**Override**: Users can set `preferred_llm_provider` in config or use `--provider` flag.

### Why Keep OpenRouter?

**Decision**: Maintain full support for OpenRouter alongside OpenAI.

**Rationale**:
1. **Existing Users**: Don't break backward compatibility
2. **Model Access**: OpenRouter provides access to Claude models
3. **Flexibility**: Some users prefer unified billing across providers
4. **Future-Proofing**: Easy to add more providers later

### Why Not Use `litellm` Library?

**Decision**: Direct API integration instead of `litellm`.

**Rationale**:
1. **Zero Dependencies**: Keep package lightweight
2. **Simple Use Case**: Only need chat completions, not streaming/embeddings
3. **Control**: Direct control over error handling and retry logic
4. **Maintenance**: Less external dependency risk

## Future Enhancements

### Potential Additions:

1. **More Providers**:
   - Anthropic Direct API
   - Azure OpenAI
   - Local models (Ollama)

2. **Model Auto-Selection**:
   - Choose model based on query complexity
   - Cost optimization

3. **Streaming Responses**:
   - Real-time response display
   - Better UX for long queries

4. **Caching**:
   - Cache LLM responses for identical queries
   - Reduce API costs

## Notes

- ✅ All syntax checks pass
- ✅ Backward compatible with existing OpenRouter users
- ✅ Environment variables take precedence over config file
- ✅ Secure config file storage (0600 permissions)
- ✅ Clear error messages for missing keys
- ✅ Interactive setup with user-friendly prompts
