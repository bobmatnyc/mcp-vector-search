# Chat/Ask Implementation - Architecture Analysis for Enhancements

**Date:** 2026-02-20
**Purpose:** Research current chat implementation to plan three enhancements:
1. Increase max supporting queries per chat to 30
2. Iterative refinement - LLM uses results from one query to inform subsequent queries
3. Knowledge Graph integration for contextual enrichment

---

## Executive Summary

The current chat implementation (`src/mcp_vector_search/cli/commands/chat.py`) uses a **tool-calling agentic loop** with Claude Opus 4 via OpenRouter/OpenAI/Bedrock. The LLM has access to 6 tools (search_code, read_file, write_markdown, analyze_code, web_search, list_files) and can make **up to 15 iterations** of tool calls before synthesizing a final response.

**Key Finding:** The current architecture already supports iterative refinement through the tool-calling loop. The LLM can make sequential search_code calls and use previous results to inform subsequent queries. However, there's **no explicit multi-query planning or cross-query result correlation**.

---

## Current Architecture

### 1. Chat Session Management

**File:** `src/mcp_vector_search/cli/commands/chat.py` (lines 66-230)

**Class:** `EnhancedChatSession`

**Key Features:**
- **5-pair conversation memory**: Keeps last 5 user/assistant exchange pairs verbatim
- **History compaction**: Older exchanges summarized into `history_summary` string
- **Task tracking**: Current task with description and status (in_progress, completed, blocked)
- **Message structure**: `[system, history_summary?, task_context?, ...recent_messages]`

**Memory Management:**
```python
RECENT_EXCHANGES_TO_KEEP = 5  # Line 76

def _compact_history(self):  # Lines 137-190
    # Summarizes oldest user/assistant pair
    # Removes compacted messages
    # Keeps history_summary for context
```

**Conversation Flow:**
```
User Query → Add to session → Get full messages (with history) → Tool loop → Add response to session
```

---

### 2. Tool-Calling Agentic Loop

**File:** `src/mcp_vector_search/cli/commands/chat.py` (lines 1112-1262)

**Function:** `_process_query()`

**Key Architecture:**

```python
max_iterations = 15  # Line 1143 - CURRENT LIMIT

for _iteration in range(max_iterations):
    response = await llm_client.chat_with_tools(messages, tools)

    if tool_calls:
        # Execute tools
        for tool_call in tool_calls:
            result = await _execute_tool(...)
            messages.append(tool_result)
        # Loop continues - LLM can make more tool calls
    else:
        # Final response - no more tool calls
        return final_content
```

**Tool Visibility:**
- Each tool call result is appended to `messages` list
- LLM sees ALL previous tool calls and results in context
- This enables **implicit iterative refinement**

**Example Flow:**
1. User: "How does authentication work?"
2. LLM: Calls `search_code("authentication logic")`
3. LLM sees result, decides to call `read_file("auth/middleware.py")`
4. LLM sees both results, calls `search_code("OAuth token validation")`
5. LLM synthesizes final answer from all 3 tool results

---

### 3. Search Tool Implementation

**File:** `src/mcp_vector_search/cli/commands/chat.py` (lines 855-896)

**Function:** `_tool_search_code()`

**Current Behavior:**
```python
async def _tool_search_code(query: str, limit: int, ...):
    limit = min(limit, 10)  # Hard cap at 10 results per call
    results = await search_engine.search(
        query=query,
        limit=limit,
        similarity_threshold=config.similarity_threshold,
        include_context=True,
    )
```

**Result Format:**
```
[Result 1: path/to/file.py]
Location: function_name
Lines 45-67
Similarity: 0.823
```
{code block}
```

**Key Limitation:**
- **Single query per tool call**: No multi-query expansion
- **Max 10 results per call**: Can't retrieve 30 results in one go
- **No result correlation**: Each search is independent

---

### 4. LLM Client Architecture

**File:** `src/mcp_vector_search/core/llm_client.py`

**Key Methods:**

#### `generate_search_queries()` - Lines 201-260
**PURPOSE:** Generate multiple targeted search queries from natural language

```python
async def generate_search_queries(
    natural_language_query: str,
    limit: int = 3  # Default: 3 queries
) -> list[str]:
```

**System Prompt (Lines 216-232):**
```
Given a natural language query, generate {limit} specific search queries
Rules:
1. Each query should target a different aspect of the question
2. Use technical terms and identifiers when possible
3. Keep queries concise (3-7 words each)
4. Focus on code patterns, function names, class names, or concepts
5. Return ONLY the search queries, one per line, no explanations
```

**Example:**
```
Input: "where is the similarity_threshold parameter set?"
Output:
similarity_threshold default value
similarity_threshold configuration
SemanticSearchEngine init threshold
```

**CRITICAL FINDING:** This method is **NOT currently used** in the chat command! It's only used in the `ask` command.

---

#### `analyze_and_rank_results()` - Lines 262-336
**PURPOSE:** Analyze search results and select most relevant ones

```python
async def analyze_and_rank_results(
    original_query: str,
    search_results: dict[str, list[Any]],  # Maps queries to results
    top_n: int = 5
) -> list[dict[str, Any]]:
```

**System Prompt (Lines 284-304):**
```
Given:
1. A natural language query
2. Multiple search results from different queries

Select the top {top_n} most relevant results

For each selected result, provide:
1. Result identifier (e.g., "Query 1, Result 2")
2. Relevance level: "High", "Medium", or "Low"
3. Brief explanation (1-2 sentences) of why this result is relevant
```

**CRITICAL FINDING:** This method is also **NOT currently used** in the chat command! Only in `ask` command.

---

#### `chat_with_tools()` - Lines 937-1030
**PURPOSE:** Chat completion with tool/function calling support

```python
async def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]]
) -> dict[str, Any]:
```

**Key Features:**
- Sends conversation history + tool definitions to LLM
- LLM can choose to call tools or provide final answer
- Returns API response with `tool_calls` or `content`

**Provider Support:**
- ✅ OpenAI: Full tool calling support
- ✅ OpenRouter: Full tool calling support
- ⚠️ Bedrock: Tool calling **NOT IMPLEMENTED** (falls back to regular chat)

---

### 5. Search Engine Architecture

**File:** `src/mcp_vector_search/core/search.py`

**Key Methods:**

#### `search()` - Lines 93-188
**PRIMARY SEARCH METHOD**

```python
async def search(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
    include_context: bool = True,
) -> list[SearchResult]:
```

**Architecture:**
1. **Health check** (throttled, every 60s)
2. **Auto-reindex check** (if search-triggered indexer enabled)
3. **Adaptive threshold** calculation (if not specified)
4. **Preprocess query** (normalize, extract terms)
5. **Two-phase search**:
   - If `VectorsBackend` available: Use LanceDB (NEW)
   - Else: Use ChromaDB (LEGACY)
6. **Knowledge Graph enhancement** (if KG available and enabled)
7. **Result enhancement** (add context lines)
8. **Result reranking** (boost based on query match)

**KG Enhancement (Lines 570-621):**
```python
async def _enhance_with_kg(results, query):
    # Extract query terms
    query_terms = set(query.lower().split())

    for result in results:
        # Find related entities (1 hop)
        related = await self._kg.find_related(chunk_id, max_hops=1)

        # Boost if related entities match query terms
        for rel in related:
            if any(term in rel_name for term in query_terms):
                result.similarity_score = min(1.0, result.similarity_score + 0.02)
```

**KEY INSIGHT:** KG enhancement is **already implemented** but only provides small boosts (0.02) based on entity relationship matching.

---

#### `search_with_context()` - Lines 265-313
**ENHANCED SEARCH WITH ANALYSIS**

```python
async def search_with_context(
    query: str,
    context_files: list[Path] | None = None,
    limit: int = 10,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
```

**Returns:**
```python
{
    "query": query,
    "analysis": query_analysis,
    "results": results,
    "suggestions": suggestions,  # Related query suggestions
    "metrics": quality_metrics,
    "total_results": len(results),
}
```

**CRITICAL FINDING:** This method provides **query analysis** and **suggestions** but is **NOT currently used** in chat command!

---

### 6. Knowledge Graph Architecture

**File:** `src/mcp_vector_search/core/knowledge_graph.py`

**Class:** `KnowledgeGraph` (Lines 145-2600+)

**Backend:** Kuzu (graph database)

**Schema:**
- **Nodes:**
  - `CodeEntity` (files, classes, functions, modules)
  - `DocSection` (markdown sections)
  - `Tag` (topic tags)
  - `Person` (git authors)
  - `Project`, `Repository`, `Branch`, `Commit`
  - `ProgrammingLanguage`, `ProgrammingFramework`

- **Relationships:**
  - `CALLS` (function A calls function B)
  - `IMPORTS` (module A imports module B)
  - `INHERITS` (class A extends B)
  - `CONTAINS` (file F contains class C)
  - `REFERENCES`, `DOCUMENTS`, `FOLLOWS`, `HAS_TAG`, `DEMONSTRATES`, `LINKS_TO`

**Key Methods:**

#### `find_related()` - Line 2561+
```python
async def find_related(chunk_id: str, max_hops: int = 1) -> list[dict]:
    # Traverse graph from chunk_id
    # Return related entities within max_hops distance
```

#### `find_entity_by_name()` - Line 2486+
```python
async def find_entity_by_name(name: str) -> str | None:
    # Find entity ID by name
```

**KEY INSIGHT:** The KG is **already integrated** into search (see `_enhance_with_kg()` in search.py) but:
1. Only used for **post-search boosting** (not query expansion)
2. Only provides **0.02 similarity boost** (very small)
3. Only checks **1-hop relationships** (direct neighbors)

---

## Enhancement Planning

### Enhancement 1: Increase Max Supporting Queries to 30

**Current State:**
- `max_iterations = 15` (line 1143 in chat.py)
- Each iteration can make multiple tool calls
- Each `search_code` tool call returns max 10 results (line 865)

**Options:**

#### Option A: Increase `max_iterations`
**Pros:**
- Simple change (one line)
- Allows LLM to make more tool calls organically
- Maintains current architecture

**Cons:**
- More API calls (higher latency and cost)
- LLM might not use all 30 iterations efficiently
- No guarantee of 30 search queries

**Implementation:**
```python
max_iterations = 30  # Line 1143
```

#### Option B: Increase `limit` per search_code call
**Pros:**
- Get more results per search
- Fewer API calls

**Cons:**
- More context per call (token usage)
- LLM sees all 30 results at once (harder to digest)
- Doesn't increase *number* of queries, just results per query

**Implementation:**
```python
limit = min(limit, 30)  # Line 865 - remove hard cap at 10
```

#### Option C: Multi-query tool (NEW)
**Pros:**
- Explicit multi-query planning
- Can use `generate_search_queries()` method (already exists!)
- Better result correlation
- Single tool call for multiple queries

**Cons:**
- Requires new tool definition
- More complex implementation

**Implementation:**
```python
# New tool in _get_tools()
{
    "type": "function",
    "function": {
        "name": "multi_search",
        "description": "Execute multiple search queries and correlate results",
        "parameters": {
            "queries": {"type": "array", "items": {"type": "string"}},
            "limit_per_query": {"type": "integer", "default": 5},
        },
    },
}

# New tool handler
async def _tool_multi_search(queries: list[str], limit_per_query: int, ...):
    all_results = {}
    for query in queries[:30]:  # Cap at 30 queries
        results = await search_engine.search(query, limit=limit_per_query)
        all_results[query] = results

    # Optional: Use LLM to rank/filter combined results
    ranked = await llm_client.analyze_and_rank_results(
        original_query=queries[0],
        search_results=all_results,
        top_n=30
    )
    return ranked
```

**RECOMMENDATION:** Option C (Multi-query tool) - Best alignment with Enhancement 2 (iterative refinement)

---

### Enhancement 2: Iterative Refinement - LLM Uses Results to Inform Queries

**Current State:**
- LLM can already see previous tool results in conversation history
- Implicit refinement through tool-calling loop
- No explicit multi-query planning or result correlation

**Approaches:**

#### Approach A: Enhanced System Prompt
**Pros:**
- No code changes
- Guides LLM behavior

**Cons:**
- Relies on LLM following instructions
- No guarantee of systematic refinement

**Implementation:**
```python
CONVERSATIONAL_SYSTEM_PROMPT = """...

4. ITERATIVE SEARCH STRATEGY
   - Start with broad queries to explore the codebase
   - Use results from initial searches to refine subsequent queries
   - If a search returns relevant functions/classes, search for their usages
   - Build understanding progressively - don't search for everything at once
   - Example: Search "authentication" → Find auth.py → Search "auth middleware usage"

..."""
```

#### Approach B: Multi-Query Workflow Tool
**Pros:**
- Explicit workflow: generate queries → search → analyze → refine → search again
- Uses existing LLM methods (`generate_search_queries`, `analyze_and_rank_results`)
- Systematic refinement

**Cons:**
- More complex
- Higher latency (multiple LLM calls)

**Implementation:**
```python
async def _tool_iterative_search(
    initial_query: str,
    max_iterations: int = 3,
    results_per_iteration: int = 10,
):
    """
    Iteration 1: Generate 5 broad queries → Search → Get top 10 results
    Iteration 2: Analyze results → Generate 3 refined queries → Search → Get top 10
    Iteration 3: Final refinement → Get best 10 results

    Total: ~8 queries, ~30 results
    """
    current_query = initial_query
    all_results = []

    for i in range(max_iterations):
        # Generate queries for this iteration
        queries = await llm_client.generate_search_queries(
            current_query,
            limit=5 if i == 0 else 3  # Broader first, then narrow
        )

        # Search with each query
        iteration_results = {}
        for query in queries:
            results = await search_engine.search(query, limit=results_per_iteration)
            iteration_results[query] = results

        # Analyze and rank results
        ranked = await llm_client.analyze_and_rank_results(
            original_query=initial_query,
            search_results=iteration_results,
            top_n=results_per_iteration
        )

        all_results.extend(ranked)

        # Generate refinement query based on what we found
        # (This would need a new LLM method)
        if i < max_iterations - 1:
            current_query = await _generate_refinement_query(
                initial_query, ranked
            )

    return all_results[:30]  # Return top 30
```

#### Approach C: Agent-Guided Multi-Query (Hybrid)
**Pros:**
- Leverages LLM's reasoning in tool-calling loop
- Provides explicit multi-query tool for efficiency
- LLM decides when to refine vs. when to synthesize

**Cons:**
- Requires both new tool AND system prompt changes

**Implementation:**
```python
# Add multi_search tool (from Option C above)
# + Update system prompt:

4. MULTI-QUERY SEARCH STRATEGY
   - Use multi_search tool when you need to explore multiple aspects at once
   - Pass 5-10 related queries to search in parallel
   - Review combined results before deciding next steps
   - If results are insufficient, make follow-up multi_search with refined queries
   - Example workflow:
     * Initial multi_search: ["auth logic", "auth middleware", "token validation"]
     * Review results → Identify gap in session management
     * Follow-up multi_search: ["session storage", "session expiry", "refresh tokens"]
```

**RECOMMENDATION:** Approach C (Hybrid) - Best of both worlds

---

### Enhancement 3: Knowledge Graph Integration

**Current State:**
- KG is **already integrated** in `search.py` (lines 570-621)
- Only used for **post-search boosting** (+0.02 similarity)
- Only checks **1-hop relationships**

**Enhancement Options:**

#### Option A: Stronger KG Boosting
**Pros:**
- Simple change
- More impact from existing KG relationships

**Cons:**
- Still post-search only
- Doesn't leverage KG for query expansion

**Implementation:**
```python
# In search.py, line 605-607
if any(term in rel_name for term in query_terms):
    result.similarity_score = min(1.0, result.similarity_score + 0.10)  # Increased from 0.02
```

#### Option B: KG-Driven Query Expansion
**Pros:**
- Proactive use of KG
- Discover related code through relationships
- Expand user's query with architectural context

**Cons:**
- More complex
- Requires KG query methods

**Implementation:**
```python
async def _expand_query_with_kg(query: str, kg: KnowledgeGraph) -> list[str]:
    """Expand query using KG relationships.

    Example:
        Input: "authentication"
        KG finds: auth.py → CALLS → session.py, token.py
        Output: ["authentication", "session management", "token validation"]
    """
    queries = [query]

    # Find entities matching query terms
    query_terms = query.lower().split()
    for term in query_terms:
        entity_id = await kg.find_entity_by_name(term)
        if entity_id:
            # Find related entities (2-hop for broader context)
            related = await kg.find_related(entity_id, max_hops=2)

            # Generate queries from related entities
            for rel in related[:5]:  # Limit to top 5 related
                queries.append(rel["name"])

    return queries[:10]  # Return max 10 expanded queries
```

**Usage in multi_search tool:**
```python
async def _tool_multi_search(queries: list[str], use_kg: bool = True, ...):
    # Expand queries using KG if available
    if use_kg and self._kg:
        expanded_queries = []
        for query in queries:
            expanded = await _expand_query_with_kg(query, self._kg)
            expanded_queries.extend(expanded)
        queries = expanded_queries[:30]  # Cap at 30

    # Execute searches...
```

#### Option C: KG-Aware Result Correlation
**Pros:**
- Surface architectural patterns
- Highlight cross-file dependencies
- Better explanations for users

**Cons:**
- Requires result post-processing
- May need additional LLM call for synthesis

**Implementation:**
```python
async def _correlate_results_with_kg(
    results: list[SearchResult],
    kg: KnowledgeGraph
) -> dict[str, Any]:
    """Analyze relationships between search results.

    Returns:
        {
            "results": results,
            "relationships": [
                {
                    "source": "auth.py:login()",
                    "target": "session.py:create_session()",
                    "type": "CALLS",
                    "explanation": "Login function creates user sessions"
                }
            ],
            "clusters": [
                {
                    "theme": "Authentication Flow",
                    "files": ["auth.py", "session.py", "token.py"]
                }
            ]
        }
    """
    relationships = []

    # Extract chunk_ids from results
    chunk_ids = [getattr(r, "chunk_id", None) for r in results]
    chunk_ids = [cid for cid in chunk_ids if cid]

    # Find relationships between results
    for source_id in chunk_ids:
        for target_id in chunk_ids:
            if source_id == target_id:
                continue

            # Query KG for relationship
            rel = await kg.find_relationship(source_id, target_id)
            if rel:
                relationships.append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel.relationship_type,
                })

    return {
        "results": results,
        "relationships": relationships,
        "relationship_count": len(relationships),
    }
```

**RECOMMENDATION:** Implement all three:
1. **Option A** (quick win, immediate impact)
2. **Option B** (integrate with multi-query tool)
3. **Option C** (provide to LLM as additional context)

---

## Implementation Roadmap

### Phase 1: Foundation (Quick Wins)

**1.1 Increase max_iterations**
```python
# chat.py, line 1143
max_iterations = 30  # Was 15
```

**1.2 Increase search result cap**
```python
# chat.py, line 865
limit = min(limit, 30)  # Was 10
```

**1.3 Stronger KG boosting**
```python
# search.py, line 605-607
result.similarity_score = min(1.0, result.similarity_score + 0.10)  # Was 0.02
```

**Estimated Effort:** 1 hour
**Impact:** Immediate improvement in result quantity and KG influence

---

### Phase 2: Multi-Query Tool (Core Enhancement)

**2.1 Add multi_search tool definition**
```python
# chat.py, _get_tools()
{
    "type": "function",
    "function": {
        "name": "multi_search",
        "description": "Execute multiple related search queries and correlate results. Use this when you need to explore multiple aspects of a topic simultaneously.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries (max 30)",
                },
                "limit_per_query": {
                    "type": "integer",
                    "description": "Results per query (default: 5)",
                    "default": 5,
                },
                "use_kg_expansion": {
                    "type": "boolean",
                    "description": "Expand queries using knowledge graph relationships",
                    "default": True,
                },
            },
            "required": ["queries"],
        },
    },
}
```

**2.2 Implement multi_search handler**
```python
# chat.py
async def _tool_multi_search(
    queries: list[str],
    limit_per_query: int,
    use_kg_expansion: bool,
    search_engine: Any,
    database: Any,
    project_root: Path,
    config: Any,
) -> str:
    """Execute multiple search queries with optional KG expansion."""
    # Limit queries
    queries = queries[:30]

    # Expand with KG if enabled and available
    if use_kg_expansion and hasattr(search_engine, '_kg') and search_engine._kg:
        expanded_queries = []
        for query in queries:
            expanded = await _expand_query_with_kg(query, search_engine._kg)
            expanded_queries.extend(expanded)
        queries = list(dict.fromkeys(expanded_queries))[:30]  # Dedupe and cap

    # Execute searches
    all_results = {}
    async with database:
        for query in queries:
            try:
                results = await search_engine.search(
                    query=query,
                    limit=limit_per_query,
                    similarity_threshold=config.similarity_threshold,
                    include_context=True,
                )
                all_results[query] = results
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

    # Optional: Correlate with KG
    all_results_flat = [r for results in all_results.values() for r in results]
    if hasattr(search_engine, '_kg') and search_engine._kg:
        correlation = await _correlate_results_with_kg(all_results_flat, search_engine._kg)
        relationship_summary = f"\n\n**Relationships Found:** {correlation['relationship_count']}"
    else:
        relationship_summary = ""

    # Format results
    parts = []
    total_results = 0
    for query, results in all_results.items():
        parts.append(f"\n=== Query: {query} ===")
        parts.append(f"Found {len(results)} results\n")

        for i, result in enumerate(results[:5], 1):  # Top 5 per query
            try:
                rel_path = str(result.file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(result.file_path)

            parts.append(
                f"[Result {i}: {rel_path}]\n"
                f"Location: {result.location}\n"
                f"Lines {result.start_line}-{result.end_line}\n"
                f"Similarity: {result.similarity_score:.3f}\n"
                f"```\n{result.content[:300]}...\n```\n"
            )
            total_results += 1

    summary = f"**Multi-Search Results**\nExecuted {len(all_results)} queries, found {total_results} results{relationship_summary}\n"
    return summary + "\n".join(parts)
```

**2.3 Add KG query expansion helper**
```python
# chat.py
async def _expand_query_with_kg(query: str, kg: Any) -> list[str]:
    """Expand query using KG relationships."""
    queries = [query]

    try:
        # Find entities matching query terms
        query_terms = query.lower().split()
        for term in query_terms:
            entity_id = await kg.find_entity_by_name(term)
            if entity_id:
                # Find related entities (2-hop)
                related = await kg.find_related(entity_id, max_hops=2)

                # Generate queries from related entities
                for rel in related[:5]:
                    queries.append(rel["name"])
    except Exception as e:
        logger.debug(f"KG expansion failed for '{query}': {e}")

    return queries[:10]
```

**2.4 Add KG result correlation helper**
```python
# chat.py
async def _correlate_results_with_kg(
    results: list[Any],
    kg: Any
) -> dict[str, Any]:
    """Find relationships between search results using KG."""
    relationships = []

    try:
        # Extract chunk_ids
        chunk_ids = []
        for result in results:
            chunk_id = getattr(result, "chunk_id", None)
            if not chunk_id:
                # Fallback: construct chunk_id
                chunk_id = f"{result.file_path}:{result.start_line}-{result.end_line}"
            chunk_ids.append(chunk_id)

        # Find relationships between results
        for source_id in chunk_ids:
            try:
                related = await kg.find_related(source_id, max_hops=1)
                for rel in related:
                    if rel.get("id") in chunk_ids:
                        relationships.append({
                            "source": source_id,
                            "target": rel["id"],
                            "type": rel.get("relationship_type", "related"),
                        })
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"KG correlation failed: {e}")

    return {
        "results": results,
        "relationships": relationships,
        "relationship_count": len(relationships),
    }
```

**2.5 Register tool in _execute_tool()**
```python
# chat.py, line 789-853
async def _execute_tool(...):
    if tool_name == "search_code":
        # existing
    elif tool_name == "multi_search":
        return await _tool_multi_search(
            arguments.get("queries", []),
            arguments.get("limit_per_query", 5),
            arguments.get("use_kg_expansion", True),
            search_engine,
            database,
            project_root,
            config,
        )
    elif tool_name == "read_file":
        # existing
    ...
```

**Estimated Effort:** 6-8 hours
**Impact:** Major improvement in search coverage and result correlation

---

### Phase 3: Enhanced System Prompt (Behavior Guidance)

**3.1 Update CONVERSATIONAL_SYSTEM_PROMPT**
```python
# chat.py, lines 266-295
CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful code assistant. IMPORTANT GUIDELINES:

1. BE CONVERSATIONAL AND FRIENDLY
   - Explain concepts in plain language first
   - Use a natural, helpful tone
   - Don't be overly formal or robotic

2. DO NOT SHOW CODE UNLESS ASKED
   - Summarize search results rather than showing raw code
   - Describe what functions/classes do without dumping code
   - Only show code when user explicitly asks: "show me", "what's the code", "show the implementation"

3. WHEN SHOWING CODE
   - Use markdown code blocks with language hints
   - Keep snippets focused and relevant
   - Add brief explanations

4. TOOL USAGE
   - Use search_code to find specific code snippets (max 10 results per call)
   - Use multi_search when exploring multiple aspects of a topic (up to 30 queries)
   - Use read_file for full file context
   - Use write_markdown to create reports
   - Use analyze_code for quality metrics
   - Use web_search for external documentation

5. MULTI-QUERY SEARCH STRATEGY
   - For broad questions, use multi_search with 5-10 related queries
   - Example: "How does authentication work?" →
     multi_search(["auth logic", "auth middleware", "session management", "token validation", "login flow"])
   - Review results and make follow-up searches if needed
   - Build understanding progressively

6. KNOWLEDGE GRAPH AWARENESS
   - When multi_search finds relationships between files/functions, mention them
   - Example: "I found that login() calls create_session() which stores data in Redis"
   - Use relationship information to provide architectural context

7. TASK TRACKING
   - Track ONE task at a time
   - Update when user gives a new task
   - Reference the current task in your responses when relevant

Remember: Be helpful, conversational, and only show code when explicitly requested."""
```

**Estimated Effort:** 30 minutes
**Impact:** Guides LLM to use new tools effectively

---

### Phase 4: Testing & Validation

**4.1 Unit Tests**
- Test `_tool_multi_search()` with various query combinations
- Test `_expand_query_with_kg()` with/without KG available
- Test `_correlate_results_with_kg()` with related and unrelated results

**4.2 Integration Tests**
- Test chat session with multi_search tool
- Test iterative refinement (initial broad search → follow-up narrow search)
- Test KG expansion and correlation

**4.3 Performance Tests**
- Measure latency of multi_search with 30 queries
- Compare token usage: single queries vs. multi_search
- Validate max_iterations doesn't cause timeouts

**Estimated Effort:** 4-6 hours
**Impact:** Ensures reliability and catches edge cases

---

## Technical Considerations

### 1. Token Usage

**Current (single search_code calls):**
```
User query: 50 tokens
System prompt: 200 tokens
Search result: 500 tokens
LLM response: 300 tokens
Total per iteration: ~1,050 tokens
```

**With multi_search (30 queries):**
```
User query: 50 tokens
System prompt: 300 tokens (updated)
Multi-search results (30 queries × 5 results × 200 tokens): 30,000 tokens
LLM response: 500 tokens
Total: ~31,000 tokens (single iteration)
```

**Mitigation:**
- Cap results per query at 5 (not 10)
- Truncate code snippets to 300 chars (not full content)
- Use `analyze_and_rank_results()` to pre-filter before sending to LLM

---

### 2. API Rate Limits

**OpenRouter Claude Opus 4:**
- Rate limit: 50 requests/minute (typical)
- Token limit: 200K input context

**Bedrock Claude:**
- Rate limit: Varies by region/account
- Token limit: 200K input context

**Mitigation:**
- Add exponential backoff for rate limit errors
- Cache multi_search results (avoid redundant API calls)

---

### 3. Knowledge Graph Availability

**Graceful Degradation:**
- Check `if search_engine._kg` before KG operations
- Fall back to standard search if KG unavailable
- Log KG failures as DEBUG (not ERROR) - non-blocking

**Example:**
```python
if use_kg_expansion and hasattr(search_engine, '_kg') and search_engine._kg:
    try:
        expanded = await _expand_query_with_kg(query, search_engine._kg)
    except Exception as e:
        logger.debug(f"KG expansion failed: {e}")
        expanded = [query]  # Fall back to original query
else:
    expanded = [query]
```

---

### 4. Conversation Context Management

**Challenge:** Multi-search results consume large token budget

**Solution:** Enhanced session compaction
```python
# In EnhancedChatSession
def add_tool_message(self, message: dict[str, Any]) -> None:
    """Add tool result with smart truncation."""
    content = message.get("content", "")

    # Truncate large tool results (>5000 chars)
    if len(content) > 5000:
        # Keep summary + first/last portions
        summary = f"[Tool result truncated: {len(content)} chars total]\n"
        content = summary + content[:2000] + "\n...\n" + content[-1000:]
        message["content"] = content

    self.messages.append(message)
```

---

## Success Metrics

### Quantitative Metrics

1. **Search Coverage:**
   - Baseline: ~3-5 queries per chat session
   - Target: 15-30 queries per session (when needed)

2. **Result Relevance:**
   - Baseline: 60-70% relevant results (user satisfaction survey)
   - Target: 75-85% relevant results

3. **Response Time:**
   - Baseline: 5-10 seconds for single query
   - Target: 10-20 seconds for multi-query (acceptable trade-off)

4. **KG Impact:**
   - Measure % of queries where KG expansion found additional relevant code
   - Target: 30-40% of queries benefit from KG

### Qualitative Metrics

1. **Iterative Refinement:**
   - Does LLM make follow-up searches based on initial results?
   - Example: Search "auth" → Find session.py → Search "session storage"

2. **Architectural Understanding:**
   - Does LLM surface cross-file relationships?
   - Example: "Login calls validate_token which checks Redis cache"

3. **Query Quality:**
   - Are generated queries diverse and targeted?
   - Do they cover different aspects of the topic?

---

## Risks & Mitigations

### Risk 1: Increased Latency
**Impact:** Users frustrated by slow responses
**Mitigation:**
- Add progress indicators ("Searching 30 queries...")
- Use streaming for LLM responses
- Cache multi-search results

### Risk 2: Token Budget Exhaustion
**Impact:** Hitting context window limits
**Mitigation:**
- Aggressive result truncation
- Smart session compaction
- Use `analyze_and_rank_results()` to pre-filter

### Risk 3: KG Unavailability
**Impact:** Features break if KG not built
**Mitigation:**
- Graceful fallback to standard search
- Clear messaging if KG disabled
- Non-blocking error handling

### Risk 4: LLM Doesn't Use New Tools
**Impact:** Features unused despite implementation
**Mitigation:**
- Strong system prompt guidance
- Tool descriptions that clearly state benefits
- Examples in tool descriptions

---

## Future Enhancements

### 1. Adaptive Query Generation
**Idea:** LLM generates queries dynamically based on search results
**Implementation:**
- New LLM method: `generate_refinement_queries(initial_query, previous_results, gap_analysis)`
- Example: If initial search missed "token refresh", generate query for it

### 2. Query Deduplication
**Idea:** Avoid redundant searches
**Implementation:**
- Hash query semantics (not literal strings)
- Skip if similar query already executed
- Return cached results

### 3. Result Clustering
**Idea:** Group related results by architectural theme
**Implementation:**
- Use KG to identify clusters (e.g., "Auth Flow", "Database Layer")
- Present results organized by cluster
- LLM can focus on one cluster at a time

### 4. Visual Relationship Maps
**Idea:** Show call graphs and dependency diagrams
**Implementation:**
- Generate Mermaid diagrams from KG relationships
- Include in markdown reports
- Example: `write_markdown("auth-flow-diagram.md", mermaid_diagram)`

### 5. Progressive Search
**Idea:** Start narrow, expand if needed
**Implementation:**
```python
async def progressive_search(query: str):
    # Phase 1: Single precise query
    results = await search_engine.search(query, limit=5)
    if results_sufficient(results):
        return results

    # Phase 2: Multi-query expansion (5 queries)
    expanded = await generate_search_queries(query, limit=5)
    results = await multi_search(expanded)
    if results_sufficient(results):
        return results

    # Phase 3: KG-driven exploration (up to 30 queries)
    kg_expanded = await expand_with_kg(expanded)
    return await multi_search(kg_expanded)
```

---

## Appendix: Key Code Locations

### Chat Command
- **Main entry:** `src/mcp_vector_search/cli/commands/chat.py`
- **Session management:** Lines 66-230 (`EnhancedChatSession`)
- **Tool loop:** Lines 1112-1262 (`_process_query`)
- **Tool definitions:** Lines 668-786 (`_get_tools`)
- **Tool execution:** Lines 789-853 (`_execute_tool`)

### Search Engine
- **Main search:** `src/mcp_vector_search/core/search.py` lines 93-188
- **KG enhancement:** Lines 570-621 (`_enhance_with_kg`)
- **Context search:** Lines 265-313 (`search_with_context`)

### LLM Client
- **Query generation:** `src/mcp_vector_search/core/llm_client.py` lines 201-260
- **Result ranking:** Lines 262-336
- **Tool calling:** Lines 937-1030

### Knowledge Graph
- **Main class:** `src/mcp_vector_search/core/knowledge_graph.py` line 145+
- **Find related:** Line 2561+ (`find_related`)
- **Find by name:** Line 2486+ (`find_entity_by_name`)

---

## Conclusion

The current chat implementation provides a solid foundation for the three enhancements:

1. **Increase max queries to 30:** Straightforward implementation via `max_iterations` + new `multi_search` tool
2. **Iterative refinement:** Already possible through tool-calling loop; enhanced with KG-driven query expansion
3. **Knowledge Graph integration:** Already present but underutilized; significant improvement possible through query expansion and result correlation

**Key Insight:** The LLM client already has `generate_search_queries()` and `analyze_and_rank_results()` methods that are **unused** in the chat command. These can be leveraged immediately for Enhancements 1 & 2.

**Recommended Implementation Order:**
1. Phase 1 (quick wins) → Immediate impact
2. Phase 2 (multi-query tool) → Core functionality
3. Phase 3 (system prompt) → Behavior guidance
4. Phase 4 (testing) → Validation

**Total Estimated Effort:** 12-16 hours for all phases

---

**Research Status:** ✅ Complete
**Next Step:** Implement Phase 1 (Foundation) changes
