# Chat Command - LLM-Powered Intelligent Code Search

## Overview

The `chat` command provides LLM-powered intelligent code search using OpenRouter API. It allows developers to ask natural language questions about their codebase and receive relevant, contextualized results.

## Features

### 🤖 Natural Language Queries
Ask questions in plain English:
- "where is the similarity_threshold parameter set?"
- "how does the indexing process work?"
- "show me the search ranking algorithm"

### 🎯 Multi-Query Search Strategy
The LLM automatically:
1. Analyzes your natural language query
2. Generates 2-3 targeted search queries
3. Executes each query against the vector database
4. Analyzes all results
5. Selects and ranks the most relevant code snippets

### 📊 Rich Output Format
Results include:
- **Relevance Rating**: High/Medium/Low with visual indicators
- **Contextual Explanation**: Why each result matches your question
- **Syntax-Highlighted Code**: With line numbers
- **Metadata**: Function/class names, file paths, similarity scores

## Setup

### 1. Get OpenRouter API Key
Visit [https://openrouter.ai/keys](https://openrouter.ai/keys) to create a free account and get your API key.

### 2. Set Environment Variable
```bash
export OPENROUTER_API_KEY="your-key-here"
```

Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.):
```bash
echo 'export OPENROUTER_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Usage
```bash
mcp-vector-search chat "where is X defined?"
```

### With Options
```bash
# Limit results
mcp-vector-search chat "find authentication code" --limit 3

# Use faster/cheaper model
mcp-vector-search chat "quick question" --model anthropic/claude-3-haiku

# Custom timeout
mcp-vector-search chat "complex question" --timeout 60

# JSON output
mcp-vector-search chat "find error handling" --json
```

## Command Options

### Required
- `QUERY`: Natural language question about your code

### Optional
- `--project-root, -p`: Project directory (auto-detected if not specified)
- `--limit, -l`: Maximum results (1-20, default: 5)
- `--model, -m`: OpenRouter model to use (default: anthropic/claude-3.5-sonnet)
- `--timeout`: API timeout in seconds (5-120, default: 30)
- `--json`: Output results in JSON format

## Supported Models

### Recommended
- `anthropic/claude-3.5-sonnet` (default) - Best balance of quality and cost
- `anthropic/claude-3-haiku` - Fast and cheap for simple queries

### Other Options
- `anthropic/claude-3-opus` - Highest quality, most expensive
- `openai/gpt-4-turbo` - Alternative high-quality option
- See [OpenRouter Models](https://openrouter.ai/models) for full list

## Examples

### Example 1: Find Parameter Definition
```bash
$ mcp-vector-search chat "where is similarity_threshold set?"

💭 Analyzing query: where is similarity_threshold set?

🔍 Generated 3 search queries:
  1. similarity_threshold default value
  2. similarity_threshold configuration
  3. SemanticSearchEngine init threshold

🔎 Searching codebase...
  • similarity_threshold default value: 12 results
  • similarity_threshold configuration: 8 results
  • SemanticSearchEngine init threshold: 5 results

🤖 Analyzing 25 results...

🎯 Top Results for: where is similarity_threshold set?

📍 Result 1 of 5
📂 src/mcp_vector_search/core/search.py

🟢 Relevance: High
Search query: SemanticSearchEngine init threshold

💡 This is the constructor of SemanticSearchEngine where similarity_threshold
is initialized as an instance variable with a default value.

[Code snippet with syntax highlighting...]

Function: __init__ | Lines: 71-102 | Similarity: 0.847
```

### Example 2: Understand Implementation
```bash
$ mcp-vector-search chat "how does the search ranking algorithm work?"

[LLM analyzes and generates queries like:]
  1. search ranking implementation
  2. rerank results algorithm
  3. similarity score calculation

[Shows relevant code from search.py with explanations]
```

### Example 3: Find Error Handling
```bash
$ mcp-vector-search chat "show me error handling patterns" --limit 3

[Returns top 3 most relevant error handling examples]
```

## Implementation Details

### Architecture

```
┌─────────────────┐
│  User Query     │
│  (Natural Lang) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Client     │  Generate 2-3 targeted
│  (OpenRouter)   │  search queries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │  Execute each query
│  Engine         │  against indexed code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Client     │  Analyze all results,
│  (OpenRouter)   │  rank by relevance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rich Output    │  Display with context
│  Formatter      │  and explanations
└─────────────────┘
```

### Key Components

#### 1. LLM Client (`src/mcp_vector_search/core/llm_client.py`)
- OpenRouter API integration
- Query generation
- Result analysis and ranking
- Error handling (timeouts, rate limits, invalid keys)

#### 2. Chat Command (`src/mcp_vector_search/cli/commands/chat.py`)
- CLI argument parsing
- Orchestrates LLM + search workflow
- Rich terminal output
- JSON export support

#### 3. Integration
- Registered in `src/mcp_vector_search/cli/main.py`
- Uses existing search engine infrastructure
- Leverages vector database for semantic search

### Error Handling

The command handles:
- **Missing API Key**: Clear instructions to get and set key
- **Network Timeouts**: Configurable timeout with helpful error messages
- **Rate Limits**: Retry suggestions and error explanation
- **Invalid Keys**: Authentication error detection
- **No Results**: Helpful suggestions to refine query

### Performance Considerations

- **API Calls**: 2 LLM requests per query (generation + analysis)
- **Cost**: ~$0.01-0.05 per query depending on model
- **Latency**: 5-15 seconds typical (depends on results and model)
- **Optimization**: Results cached during analysis phase

## Cost Estimation

### Per Query Cost (Approximate)

| Model | Cost per Query | Speed |
|-------|---------------|-------|
| claude-3-haiku | $0.005-0.01 | Fast (3-5s) |
| claude-3.5-sonnet | $0.02-0.05 | Medium (5-10s) |
| claude-3-opus | $0.10-0.20 | Slow (10-20s) |

Costs based on:
- Query generation: ~500 tokens
- Result analysis: ~2000-5000 tokens (depends on results)

## Troubleshooting

### "OpenRouter API key not found"
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "Project not initialized"
```bash
mcp-vector-search init
mcp-vector-search index
```

### "No results found"
- Try more general terms
- Ensure code is indexed: `mcp-vector-search status`
- Try alternative phrasing

### Timeout Errors
```bash
# Increase timeout for complex queries
mcp-vector-search chat "query" --timeout 60
```

### Rate Limit Errors
Wait 1-2 minutes and retry. Consider:
- Using --limit to reduce result analysis
- Using claude-3-haiku for faster, cheaper queries

## Future Enhancements

### Planned Features
- [ ] Conversation history (follow-up questions)
- [ ] Multi-turn clarification ("Did you mean X or Y?")
- [ ] Code explanation mode (not just search)
- [ ] Diff analysis ("What changed in this function?")
- [ ] Custom prompt templates

### Integration Opportunities
- [ ] MCP server support (chat via Claude Desktop)
- [ ] Streaming responses for faster feedback
- [ ] Local LLM support (Ollama, llama.cpp)
- [ ] Fine-tuned models for code-specific queries

## Related Commands

- `search` - Traditional semantic search (no LLM)
- `search --context` - Context-aware search
- `search --similar` - Find similar code

## Contributing

To improve the chat command:
1. Add new query generation strategies in `llm_client.py`
2. Enhance result ranking algorithms
3. Add support for new LLM providers
4. Improve error handling and user feedback

## License

Part of MCP Vector Search project. See LICENSE file.

---

**Last Updated**: December 8, 2025
**Version**: 0.14.9+
