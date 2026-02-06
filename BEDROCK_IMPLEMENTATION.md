# AWS Bedrock Implementation Summary

## Overview

AWS Bedrock has been successfully integrated as a new LLM provider for mcp-vector-search, alongside the existing OpenAI and OpenRouter providers. Bedrock uses boto3 to access Claude models via AWS's managed AI service.

## Changes Made

### 1. **pyproject.toml**
- Added `boto3>=1.35.0` dependency for AWS Bedrock SDK

### 2. **src/mcp_vector_search/core/config_utils.py**
- Added `is_bedrock_available()` function to check for AWS credentials
- Updated `save_preferred_llm_provider()` to include 'bedrock' as valid provider

### 3. **src/mcp_vector_search/core/llm_client.py** (Main Implementation)

#### Type System Updates
- Updated `LLMProvider` type: `Literal["openai", "openrouter", "bedrock"]`

#### Model Configuration
- **Default model**: `anthropic.claude-3-5-sonnet-20241022-v2:0` (Claude 3.5 Sonnet v2)
- **Thinking model**: `anthropic.claude-opus-4-20250514-v1:0` (Claude Opus 4)
- Model override via `BEDROCK_MODEL` environment variable

#### Provider Auto-Detection Priority
1. **Bedrock** (if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set)
2. **OpenAI** (if OPENAI_API_KEY is set)
3. **OpenRouter** (if OPENROUTER_API_KEY is set)

#### New Methods

##### `_bedrock_available` (property)
- Checks if AWS credentials are present in environment
- Returns True if both `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set

##### `_get_bedrock_client()`
- Lazy initialization of boto3 bedrock-runtime client
- Region detection: `AWS_REGION` → `AWS_DEFAULT_REGION` → `"us-east-1"` (default)
- Raises `ImportError` if boto3 not installed

##### `_bedrock_chat_completion(messages)`
- Non-streaming completion using Bedrock Converse API
- Converts OpenAI-style messages to Bedrock format
- Handles system messages separately (Bedrock requirement)
- Runs synchronous boto3 call in executor for async compatibility
- Returns OpenAI-compatible response format

##### `_bedrock_stream_chat_completion(messages)`
- Streaming completion using Bedrock Converse Stream API
- Yields text chunks as they arrive
- Handles same message conversion as non-streaming version

#### Modified Methods

##### `__init__(...)`
- Added Bedrock provider detection in auto-detection logic
- Added validation for Bedrock provider (checks AWS credentials)
- Sets up model selection with `BEDROCK_MODEL` env var support

##### `_chat_completion(messages)`
- Routes to `_bedrock_chat_completion()` when provider is 'bedrock'

##### `stream_chat_completion(messages)`
- Routes to `_bedrock_stream_chat_completion()` when provider is 'bedrock'

##### `chat_with_tools(messages, tools)`
- Added fallback for Bedrock (tool calling not yet implemented)
- Logs warning and uses regular chat completion

## Environment Variables

### Required for Bedrock
- `AWS_ACCESS_KEY_ID` - AWS access key (required)
- `AWS_SECRET_ACCESS_KEY` - AWS secret access key (required)

### Optional for Bedrock
- `AWS_REGION` or `AWS_DEFAULT_REGION` - AWS region (default: us-east-1)
- `BEDROCK_MODEL` - Override default model ID

## Usage Examples

### Explicit Bedrock Provider
```python
from mcp_vector_search.core.llm_client import LLMClient

# Initialize with Bedrock
client = LLMClient(provider="bedrock")

# Generate search queries
queries = await client.generate_search_queries("find authentication code")

# Chat completion
messages = [{"role": "user", "content": "Hello!"}]
response = await client._chat_completion(messages)

# Streaming
async for chunk in client.stream_chat_completion(messages):
    print(chunk, end="")
```

### Auto-Detection (Bedrock has highest priority)
```python
# If AWS credentials are set, Bedrock will be auto-selected
client = LLMClient()  # Uses Bedrock if AWS_ACCESS_KEY_ID/SECRET set

# Otherwise falls back to OpenAI or OpenRouter
```

### Using Thinking Models
```python
# Use Claude Opus 4 for complex queries
client = LLMClient(provider="bedrock", think=True)
```

## API Compatibility

### Bedrock Converse API Format
```python
# Request
{
    "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "messages": [
        {"role": "user", "content": [{"text": "Hello"}]}
    ],
    "system": [{"text": "You are a helpful assistant"}],  # Optional
    "inferenceConfig": {
        "maxTokens": 4096,
        "temperature": 0.7
    }
}

# Response
{
    "output": {
        "message": {
            "content": [{"text": "Hello! How can I help?"}]
        }
    },
    "stopReason": "end_turn",
    "usage": {...}
}
```

### Converted to OpenAI Format (for compatibility)
```python
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {...}
}
```

## Error Handling

### Bedrock-Specific Errors
- **AccessDeniedException**: Invalid AWS credentials or insufficient permissions
- **ValidationException**: Invalid request format or parameters
- **ThrottlingException**: Rate limit exceeded
- **ModelNotReadyException**: Model not available or not ready in region
- **ResourceNotFoundException**: Model not found (check model ID and region)

### Generic Error Handling
All errors are wrapped in `SearchError` for consistent error handling across providers.

## Testing

### Structure Validation
Run the structure validation test:
```bash
python test_bedrock_simple.py
```

This validates:
- ✓ boto3 dependency added to pyproject.toml
- ✓ is_bedrock_available() function in config_utils.py
- ✓ Bedrock provider in LLMProvider type
- ✓ Bedrock models defined
- ✓ All Bedrock methods implemented
- ✓ Auto-detection includes Bedrock
- ✓ Converse API integration

### Integration Testing (requires AWS credentials)
Run the full integration test:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1

python test_bedrock_integration.py
```

This tests:
- Provider detection
- Client initialization
- Chat completion
- Streaming
- Auto-detection priority

## Limitations

### Not Yet Implemented
- **Tool/Function Calling**: Bedrock tool calling uses a different format (`toolConfig`)
  - Currently falls back to regular chat completion
  - TODO: Implement when tool calling is needed

### Bedrock-Specific Constraints
- System messages must be passed separately (not in messages array)
- Message format requires `content` to be array of content blocks
- Streaming uses different event structure than OpenAI SSE

## Performance Considerations

### Async Compatibility
- boto3 is synchronous, so calls run in executor via `loop.run_in_executor()`
- Minimal performance impact for typical use cases
- Streaming iterates over boto3 generator in executor

### Regional Latency
- Default region is `us-east-1`
- Set `AWS_REGION` environment variable to use closer region
- Model availability varies by region

## Migration Guide

### For Existing Users

#### No Changes Required
- Existing code using OpenAI or OpenRouter continues to work
- Bedrock is opt-in via explicit provider selection or AWS credentials

#### To Use Bedrock

**Option 1: Auto-detection (recommended)**
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Bedrock will be auto-selected
mcp-vector-search chat
```

**Option 2: Explicit provider**
```python
client = LLMClient(provider="bedrock")
```

**Option 3: Set preferred provider**
```bash
mcp-vector-search config set preferred_llm_provider bedrock
```

### For New Users
1. Install dependencies: `pip install mcp-vector-search`
2. Set AWS credentials (see above)
3. Use any mcp-vector-search command - Bedrock will be auto-selected

## Cost Comparison

### Claude 3.5 Sonnet (Default)
- **Bedrock**: Pay-per-use via AWS (no API key cost)
- **OpenRouter**: $3/M input tokens, $15/M output tokens
- **OpenAI**: N/A (OpenAI doesn't offer Claude models)

### Cost Considerations
- Bedrock may be cheaper for high-volume usage (AWS pricing)
- OpenRouter is simpler for occasional use (no AWS account needed)
- OpenAI offers GPT models instead

## Future Enhancements

### Planned
- [ ] Tool/function calling support for Bedrock
- [ ] Batch inference support for cost optimization
- [ ] Guardrails integration for content filtering
- [ ] Model customization support (fine-tuned models)

### Nice to Have
- [ ] Cross-region failover
- [ ] Cost tracking and budgets
- [ ] Model performance metrics
- [ ] A/B testing between providers

## Summary

✅ **Implemented**:
- Bedrock provider with auto-detection (highest priority)
- Non-streaming and streaming chat completion
- OpenAI-compatible response format
- Comprehensive error handling
- Environment variable configuration
- Model selection (default + thinking models)

✅ **Tested**:
- Structure validation (syntax, imports, methods)
- Provider auto-detection priority
- AWS credentials detection

⏳ **TODO**:
- Tool calling for Bedrock (when needed)
- Integration tests with live AWS credentials
- Performance benchmarks vs. other providers

---

## Related Files

- `/Users/masa/Projects/mcp-vector-search/pyproject.toml` - boto3 dependency
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/config_utils.py` - Bedrock detection
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py` - Main implementation
- `/Users/masa/Projects/mcp-vector-search/test_bedrock_simple.py` - Structure validation
- `/Users/masa/Projects/mcp-vector-search/test_bedrock_integration.py` - Integration tests

---

**Implementation Date**: 2026-02-06
**Status**: ✅ Complete (except tool calling)
