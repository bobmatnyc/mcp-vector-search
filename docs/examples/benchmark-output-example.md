# LLM Benchmark Output Example

This is an example of what the output looks like when running `make benchmark-llm`.

## Example Run

```bash
$ make benchmark-llm
Running LLM model benchmarks...

╔══════════════════════════════════════════════════════════════╗
║           LLM Model Benchmark Results                        ║
╚══════════════════════════════════════════════════════════════╝

Testing query: where is similarity_threshold configured?

  → Testing anthropic/claude-3.5-sonnet... ✓ 2.3s, $0.0115
  → Testing anthropic/claude-3-haiku... ✓ 0.9s, $0.0008
  → Testing openai/gpt-4o... ✓ 2.5s, $0.0081
  → Testing openai/gpt-4o-mini... ✓ 1.1s, $0.0004
  → Testing google/gemini-flash-1.5... ✓ 0.7s, $0.0002
  → Testing meta-llama/llama-3.1-70b-instruct... ✓ 1.8s, $0.0011
  → Testing mistralai/mistral-large... ✓ 1.5s, $0.0063


Query: "where is similarity_threshold configured?"

┌─────────────────────────────────┬─────────┬────────┬────────┬──────────┬─────────┬─────────────────┐
│ Model                           │ Time(s) │ Input  │ Output │ Cost($)  │ Quality │ Status          │
├─────────────────────────────────┼─────────┼────────┼────────┼──────────┼─────────┼─────────────────┤
│ gemini-flash-1.5                │    0.7s │   1245 │    356 │  $0.0002 │ ★★★☆☆  │ ✓ 5 results     │
│ claude-3-haiku                  │    0.9s │   1245 │    412 │  $0.0008 │ ★★★★☆  │ ✓ 5 results     │
│ gpt-4o-mini                     │    1.1s │   1245 │    389 │  $0.0004 │ ★★★☆☆  │ ✓ 5 results     │
│ mistral-large                   │    1.5s │   1245 │    445 │  $0.0063 │ ★★★★☆  │ ✓ 5 results     │
│ llama-3.1-70b-instruct          │    1.8s │   1245 │    423 │  $0.0011 │ ★★★★☆  │ ✓ 5 results     │
│ claude-3.5-sonnet               │    2.3s │   1245 │    523 │  $0.0115 │ ★★★★★  │ ✓ 5 results     │
│ gpt-4o                          │    2.5s │   1245 │    498 │  $0.0081 │ ★★★★☆  │ ✓ 5 results     │
└─────────────────────────────────┴─────────┴────────┴────────┴──────────┴─────────┴─────────────────┘


Testing query: how does the indexer handle TypeScript files?

  → Testing anthropic/claude-3.5-sonnet... ✓ 2.1s, $0.0109
  → Testing anthropic/claude-3-haiku... ✓ 0.8s, $0.0007
  → Testing openai/gpt-4o... ✓ 2.3s, $0.0075
  → Testing openai/gpt-4o-mini... ✓ 1.0s, $0.0003
  → Testing google/gemini-flash-1.5... ✓ 0.6s, $0.0001
  → Testing meta-llama/llama-3.1-70b-instruct... ✓ 1.7s, $0.0010
  → Testing mistralai/mistral-large... ✓ 1.4s, $0.0058


Query: "how does the indexer handle TypeScript files?"

┌─────────────────────────────────┬─────────┬────────┬────────┬──────────┬─────────┬─────────────────┐
│ Model                           │ Time(s) │ Input  │ Output │ Cost($)  │ Quality │ Status          │
├─────────────────────────────────┼─────────┼────────┼────────┼──────────┼─────────┼─────────────────┤
│ gemini-flash-1.5                │    0.6s │   1312 │    334 │  $0.0001 │ ★★★☆☆  │ ✓ 5 results     │
│ claude-3-haiku                  │    0.8s │   1312 │    398 │  $0.0007 │ ★★★★☆  │ ✓ 5 results     │
│ gpt-4o-mini                     │    1.0s │   1312 │    371 │  $0.0003 │ ★★★☆☆  │ ✓ 5 results     │
│ mistral-large                   │    1.4s │   1312 │    429 │  $0.0058 │ ★★★★☆  │ ✓ 5 results     │
│ llama-3.1-70b-instruct          │    1.7s │   1312 │    407 │  $0.0010 │ ★★★★☆  │ ✓ 5 results     │
│ claude-3.5-sonnet               │    2.1s │   1312 │    509 │  $0.0109 │ ★★★★★  │ ✓ 5 results     │
│ gpt-4o                          │    2.3s │   1312 │    482 │  $0.0075 │ ★★★★☆  │ ✓ 5 results     │
└─────────────────────────────────┴─────────┴────────┴────────┼──────────┼─────────┼─────────────────┘


Testing query: show me examples of error handling in the search module

  → Testing anthropic/claude-3.5-sonnet... ✓ 2.4s, $0.0122
  → Testing anthropic/claude-3-haiku... ✓ 0.9s, $0.0009
  → Testing openai/gpt-4o... ✓ 2.6s, $0.0087
  → Testing openai/gpt-4o-mini... ✓ 1.2s, $0.0005
  → Testing google/gemini-flash-1.5... ✓ 0.8s, $0.0002
  → Testing meta-llama/llama-3.1-70b-instruct... ✓ 1.9s, $0.0012
  → Testing mistralai/mistral-large... ✓ 1.6s, $0.0067


Query: "show me examples of error handling in the search module"

┌─────────────────────────────────┬─────────┬────────┬────────┬──────────┬─────────┬─────────────────┐
│ Model                           │ Time(s) │ Input  │ Output │ Cost($)  │ Quality │ Status          │
├─────────────────────────────────┼─────────┼────────┼────────┼──────────┼─────────┼─────────────────┤
│ gemini-flash-1.5                │    0.8s │   1389 │    367 │  $0.0002 │ ★★★★☆  │ ✓ 5 results     │
│ claude-3-haiku                  │    0.9s │   1389 │    428 │  $0.0009 │ ★★★★☆  │ ✓ 5 results     │
│ gpt-4o-mini                     │    1.2s │   1389 │    395 │  $0.0005 │ ★★★☆☆  │ ✓ 5 results     │
│ mistral-large                   │    1.6s │   1389 │    458 │  $0.0067 │ ★★★★☆  │ ✓ 5 results     │
│ llama-3.1-70b-instruct          │    1.9s │   1389 │    437 │  $0.0012 │ ★★★★☆  │ ✓ 5 results     │
│ claude-3.5-sonnet               │    2.4s │   1389 │    541 │  $0.0122 │ ★★★★★  │ ✓ 5 results     │
│ gpt-4o                          │    2.6s │   1389 │    512 │  $0.0087 │ ★★★★☆  │ ✓ 5 results     │
└─────────────────────────────────┴─────────┴────────┴────────┴──────────┴─────────┴─────────────────┘


═══ Benchmark Summary ═══

Performance by Model:

┌─────────────────────────┬──────────┬──────────┬─────────────┬──────────────┐
│ Model                   │ Avg Time │ Avg Cost │ Avg Quality │ Success Rate │
├─────────────────────────┼──────────┼──────────┼─────────────┼──────────────┤
│ gemini-flash-1.5        │     0.7s │ $0.0002  │ ★★★☆☆      │         100% │
│ claude-3-haiku          │     0.9s │ $0.0008  │ ★★★★☆      │         100% │
│ gpt-4o-mini             │     1.1s │ $0.0004  │ ★★★☆☆      │         100% │
│ mistral-large           │     1.5s │ $0.0063  │ ★★★★☆      │         100% │
│ llama-3.1-70b-instruct  │     1.8s │ $0.0011  │ ★★★★☆      │         100% │
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

Benchmark completed!
```

## Key Insights from Example

### Speed Leaders
1. **Google Gemini Flash 1.5**: 0.7s average - 3x faster than premium models
2. **Claude 3 Haiku**: 0.9s average - Best balance of speed and quality
3. **GPT-4o Mini**: 1.1s average - Good OpenAI option for speed

### Cost Leaders
1. **Google Gemini Flash 1.5**: $0.0002/query - 57x cheaper than Claude 3.5 Sonnet
2. **GPT-4o Mini**: $0.0004/query - Good budget option
3. **Claude 3 Haiku**: $0.0008/query - Best value for quality

### Quality Leaders
1. **Claude 3.5 Sonnet**: ★★★★★ - Best result relevance
2. **Claude 3 Haiku**: ★★★★☆ - Very good accuracy
3. **GPT-4o**: ★★★★☆ - Strong performance

### Cost vs Performance Analysis

**Budget Tier (<$0.001/query)**:
- Gemini Flash ($0.0002): Best overall value - fast, cheap, decent quality
- GPT-4o Mini ($0.0004): Slightly better quality than Gemini
- Claude Haiku ($0.0008): Best accuracy in budget tier

**Premium Tier (>$0.007/query)**:
- Claude 3.5 Sonnet ($0.0115): Best quality, slower
- GPT-4o ($0.0081): Good quality, slower
- Mistral Large ($0.0063): Mid-tier option

### Use Case Recommendations

| Scenario | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Development/Testing** | `claude-3-haiku` | Fast (0.9s), cheap ($0.0008), accurate (★★★★☆) |
| **Production Chat** | `claude-3.5-sonnet` | Best quality (★★★★★), worth the cost |
| **High-Volume Queries** | `gemini-flash-1.5` | Cheapest ($0.0002), still good quality |
| **Cost-Conscious Production** | `claude-3-haiku` | Best balance of all factors |
| **Complex/Nuanced Queries** | `claude-3.5-sonnet` | Best understanding of subtlety |
| **Quick Lookups** | `gemini-flash-1.5` | Fastest (0.7s), nearly free |

## How to Read Quality Ratings

The quality rating (★★★★★) is based on:

- **★★★☆☆** (3 stars): Returned ranked results successfully
- **★★★★☆** (4 stars): Returned results + found good coverage (≥5 results)
- **★★★★★** (5 stars): Perfect - results + coverage + multiple search queries

In the example above:
- Claude 3.5 Sonnet: ★★★★★ (perfect execution)
- Claude Haiku, Mistral, Llama: ★★★★☆ (very good)
- Gemini Flash, GPT-4o Mini: ★★★☆☆ (good, found results but limited coverage)

## Interpreting Success Rate

All models showed 100% success rate in this example, meaning:
- No API errors (401, 429, 500)
- All queries generated successfully
- All searches completed
- All results ranked by LLM

If success rate < 100%, possible causes:
- Rate limiting (429 errors)
- API key issues (401 errors)
- Timeout errors
- Model unavailability

## Running Your Own Benchmark

To reproduce these results:

```bash
# Set API key
export OPENROUTER_API_KEY='your-key-here'

# Ensure project is indexed
mcp-vector-search index

# Run full benchmark
make benchmark-llm

# Or test specific models
make benchmark-llm-fast  # Just cheap models

# Or custom query
make benchmark-llm-query QUERY="your custom query"
```

Expected runtime: 5-10 minutes (with rate limiting delays)

## See Also

- [LLM Benchmarking Guide](../guides/llm-benchmarking.md) - Full documentation
- [Chat Command Documentation](../reference/chat-command.md) - Using the chat command
- [OpenRouter Models](https://openrouter.ai/models) - Model details and pricing
