# LLM Model Benchmarking Guide

## Overview

The `benchmark_llm_models.py` script tests various OpenRouter LLM models on the chat command to compare:
- **Response quality** (relevance of results)
- **Speed** (latency in seconds)
- **Token usage** (input/output tokens)
- **Cost** (USD per query)

## Prerequisites

1. **OpenRouter API Key**: Get a key from [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. **Indexed Project**: Run `mcp-vector-search init` and `mcp-vector-search index` first
3. **Environment Variable**: Set your API key:
   ```bash
   export OPENROUTER_API_KEY='your-key-here'
   ```

## Usage

### Test All Models (Default)

Benchmark all 7 default models on 3 default queries:

```bash
make benchmark-llm
```

Or directly:

```bash
uv run python scripts/benchmark_llm_models.py
```

**Expected runtime**: ~5-10 minutes (with rate limiting)

### Test Specific Models

Benchmark only fast/cheap models:

```bash
make benchmark-llm-fast
```

Or specify custom models:

```bash
uv run python scripts/benchmark_llm_models.py \
  --models anthropic/claude-3-haiku \
  --models openai/gpt-4o-mini
```

### Test Single Query

Benchmark all models on a single query:

```bash
make benchmark-llm-query QUERY="where is similarity_threshold configured?"
```

Or directly:

```bash
uv run python scripts/benchmark_llm_models.py \
  --query "how does the indexer handle TypeScript files?"
```

### Custom Project Root

Test on a different project:

```bash
uv run python scripts/benchmark_llm_models.py \
  --project-root /path/to/other/project
```

## Default Models

The script tests these models by default:

| Tier | Model | Input | Output | Best For |
|------|-------|-------|--------|----------|
| **Premium** | `anthropic/claude-3.5-sonnet` | $3.00/1M | $15.00/1M | Best quality |
| **Premium** | `openai/gpt-4o` | $2.50/1M | $10.00/1M | GPT-4 quality |
| **Mid-tier** | `anthropic/claude-3-haiku` | $0.25/1M | $1.25/1M | Fast & accurate |
| **Mid-tier** | `openai/gpt-4o-mini` | $0.15/1M | $0.60/1M | GPT-4 speed |
| **Mid-tier** | `google/gemini-flash-1.5` | $0.075/1M | $0.30/1M | Very cheap |
| **Budget** | `meta-llama/llama-3.1-70b-instruct` | $0.52/1M | $0.75/1M | Open source |
| **Budget** | `mistralai/mistral-large` | $2.00/1M | $6.00/1M | European option |

## Default Test Queries

1. `"where is similarity_threshold configured?"`
2. `"how does the indexer handle TypeScript files?"`
3. `"show me examples of error handling in the search module"`

## Output Format

### Per-Query Results Table

```
Query: "where is similarity_threshold configured?"

┌─────────────────────────────────┬─────────┬────────┬────────┬──────────┬─────────┬────────────────┐
│ Model                           │ Time(s) │ Input  │ Output │ Cost($)  │ Quality │ Status         │
├─────────────────────────────────┼─────────┼────────┼────────┼──────────┼─────────┼────────────────┤
│ gemini-flash-1.5                │    0.7s │   1245 │    356 │  $0.0002 │ ★★★☆☆  │ ✓ 5 results    │
│ claude-3-haiku                  │    0.9s │   1245 │    412 │  $0.0008 │ ★★★★☆  │ ✓ 5 results    │
│ gpt-4o-mini                     │    1.1s │   1245 │    389 │  $0.0004 │ ★★★☆☆  │ ✓ 5 results    │
│ claude-3.5-sonnet               │    2.3s │   1245 │    523 │  $0.0115 │ ★★★★★  │ ✓ 5 results    │
│ gpt-4o                          │    2.5s │   1245 │    498 │  $0.0081 │ ★★★★☆  │ ✓ 5 results    │
└─────────────────────────────────┴─────────┴────────┴────────┴──────────┴─────────┴────────────────┘
```

### Summary with Recommendations

```
═══ Benchmark Summary ═══

Performance by Model:

┌─────────────────────────┬──────────┬──────────┬─────────────┬──────────────┐
│ Model                   │ Avg Time │ Avg Cost │ Avg Quality │ Success Rate │
├─────────────────────────┼──────────┼──────────┼─────────────┼──────────────┤
│ gemini-flash-1.5        │     0.7s │ $0.0002  │ ★★★☆☆      │         100% │
│ claude-3-haiku          │     0.9s │ $0.0008  │ ★★★★☆      │         100% │
│ gpt-4o-mini             │     1.1s │ $0.0004  │ ★★★☆☆      │         100% │
│ claude-3.5-sonnet       │     2.3s │ $0.0115  │ ★★★★★      │         100% │
│ gpt-4o                  │     2.5s │ $0.0081  │ ★★★★☆      │         100% │
└─────────────────────────┴──────────┴──────────┴─────────────┴──────────────┘

💡 Recommendations:

  🏃 Fastest: google/gemini-flash-1.5 (0.7s avg)
  💰 Cheapest: google/gemini-flash-1.5 ($0.0002 avg)
  ⭐ Best Quality: anthropic/claude-3.5-sonnet

🎯 Overall Recommendation:
  For speed: Use google/gemini-flash-1.5 (~0.7s per query)
  For cost: Use google/gemini-flash-1.5 (~$0.0002 per query)
  For quality: Use anthropic/claude-3.5-sonnet (best result relevance)
```

## Quality Rating Explanation

Quality is rated with stars (★★★★★) based on:

- **3 stars**: Returned ranked results
- **+1 star**: Found ≥5 relevant results
- **+1 star**: Generated ≥2 search queries

**Examples**:
- `★★★★★` = Perfect (results + good coverage + multiple queries)
- `★★★★☆` = Very good (results + good coverage OR multiple queries)
- `★★★☆☆` = Good (returned results but limited)
- `☆☆☆☆☆` = Failed (error or no results)

## Understanding Results

### Speed vs Quality Tradeoff

- **Fast models** (Gemini Flash, Claude Haiku): 0.7-1.0s, great for interactive use
- **Premium models** (Claude 3.5 Sonnet, GPT-4o): 2-3s, better result relevance

### Cost Analysis

- **Budget**: Gemini Flash (~$0.0002/query) or GPT-4o-mini (~$0.0004/query)
- **Balanced**: Claude Haiku (~$0.0008/query)
- **Premium**: Claude 3.5 Sonnet (~$0.01/query) or GPT-4o (~$0.008/query)

### When to Use Each Model

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Development/testing | `claude-3-haiku` | Fast, cheap, accurate |
| Production chat | `claude-3.5-sonnet` | Best quality responses |
| High-volume queries | `gemini-flash-1.5` | Cheapest, still good quality |
| Complex queries | `claude-3.5-sonnet` | Best understanding of nuance |
| Quick lookups | `gpt-4o-mini` | Fast and inexpensive |

## Troubleshooting

### Rate Limiting

If you see `429 Rate limit exceeded` errors:
- The script includes 1-second delays between requests
- Wait a few minutes before retrying
- Consider testing fewer models at once

### No Results Found

If models return "no results":
- Check your index: `mcp-vector-search status`
- Reindex if needed: `mcp-vector-search index`
- Try broader queries

### API Key Errors

If you see `401 Invalid API key`:
- Verify your API key is correct
- Check environment variable: `echo $OPENROUTER_API_KEY`
- Get a new key at [https://openrouter.ai/keys](https://openrouter.ai/keys)

### Import Errors

If you see module import errors:
- Always use `uv run python` to run the script
- Ensure dependencies are installed: `make dev`

## Advanced Usage

### Add Custom Models

Edit `scripts/benchmark_llm_models.py` to add models:

```python
MODEL_PRICING = {
    "your-provider/your-model": {"input": 0.50, "output": 1.50},
    # ... existing models
}
```

Then run:

```bash
uv run python scripts/benchmark_llm_models.py \
  --models your-provider/your-model
```

### Custom Test Queries

Edit `DEFAULT_TEST_QUERIES` in the script:

```python
DEFAULT_TEST_QUERIES = [
    "your custom query 1",
    "your custom query 2",
]
```

### Export Results

The script prints to stdout, so you can save results:

```bash
make benchmark-llm > benchmark_results.txt
```

Or just the summary:

```bash
make benchmark-llm 2>&1 | tail -50 > summary.txt
```

## Best Practices

1. **Run on Indexed Project**: Always index before benchmarking
2. **Test Multiple Queries**: Single queries can be misleading
3. **Consider Latency**: Fast models improve user experience
4. **Monitor Costs**: Track API usage for production systems
5. **Regular Testing**: Re-run benchmarks when models update

## See Also

- [Chat Command Documentation](../reference/chat-command.md)
- [OpenRouter Models](https://openrouter.ai/models)
- [LLM Client Source](../../src/mcp_vector_search/core/llm_client.py)
