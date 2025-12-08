"""LLM client for intelligent code search using OpenAI or OpenRouter API."""

import os
import re
from typing import Any, Literal

import httpx
from loguru import logger

from .exceptions import SearchError

# Type alias for provider
LLMProvider = Literal["openai", "openrouter"]


class LLMClient:
    """Client for LLM-powered intelligent search orchestration.

    Supports both OpenAI and OpenRouter APIs:
    1. Generate multiple targeted search queries from natural language
    2. Analyze search results and select most relevant ones
    3. Provide contextual explanations for results

    Provider Selection Priority:
    1. Explicit provider parameter
    2. Preferred provider from config
    3. Auto-detect: OpenAI if available, otherwise OpenRouter
    """

    # Default models for each provider (comparable performance/cost)
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",  # Fast, cheap, comparable to claude-3-haiku
        "openrouter": "anthropic/claude-3-haiku",
    }

    # Advanced "thinking" models for complex queries (--think flag)
    THINKING_MODELS = {
        "openai": "gpt-4o",  # More capable, better reasoning
        "openrouter": "anthropic/claude-sonnet-4",  # Claude Sonnet 4 for deep analysis
    }

    # API endpoints
    API_ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }

    TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        provider: LLMProvider | None = None,
        openai_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        think: bool = False,
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: API key (deprecated, use provider-specific keys)
            model: Model to use (defaults based on provider)
            timeout: Request timeout in seconds
            provider: Explicit provider ('openai' or 'openrouter')
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            openrouter_api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
            think: Use advanced "thinking" model for complex queries

        Raises:
            ValueError: If no API key is found for any provider
        """
        self.think = think
        # Get API keys from environment or parameters
        self.openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Support deprecated api_key parameter (assume OpenRouter for backward compatibility)
        if api_key and not self.openrouter_key:
            self.openrouter_key = api_key

        # Determine which provider to use
        if provider:
            # Explicit provider specified
            self.provider: LLMProvider = provider
            if provider == "openai" and not self.openai_key:
                raise ValueError(
                    "OpenAI provider specified but OPENAI_API_KEY not found. "
                    "Please set OPENAI_API_KEY environment variable."
                )
            elif provider == "openrouter" and not self.openrouter_key:
                raise ValueError(
                    "OpenRouter provider specified but OPENROUTER_API_KEY not found. "
                    "Please set OPENROUTER_API_KEY environment variable."
                )
        else:
            # Auto-detect provider (prefer OpenAI if both are available)
            if self.openai_key:
                self.provider = "openai"
            elif self.openrouter_key:
                self.provider = "openrouter"
            else:
                raise ValueError(
                    "No API key found. Please set OPENAI_API_KEY or OPENROUTER_API_KEY "
                    "environment variable, or pass openai_api_key or openrouter_api_key parameter."
                )

        # Set API key and endpoint based on provider
        # Select model: explicit > env var > thinking model > default model
        if self.provider == "openai":
            self.api_key = self.openai_key
            self.api_endpoint = self.API_ENDPOINTS["openai"]
            default_model = (
                self.THINKING_MODELS["openai"]
                if think
                else self.DEFAULT_MODELS["openai"]
            )
            self.model = model or os.environ.get("OPENAI_MODEL", default_model)
        else:
            self.api_key = self.openrouter_key
            self.api_endpoint = self.API_ENDPOINTS["openrouter"]
            default_model = (
                self.THINKING_MODELS["openrouter"]
                if think
                else self.DEFAULT_MODELS["openrouter"]
            )
            self.model = model or os.environ.get("OPENROUTER_MODEL", default_model)

        self.timeout = timeout

        logger.debug(
            f"Initialized LLM client with provider: {self.provider}, model: {self.model}"
        )

    async def generate_search_queries(
        self, natural_language_query: str, limit: int = 3
    ) -> list[str]:
        """Generate targeted search queries from natural language.

        Args:
            natural_language_query: User's natural language query
            limit: Maximum number of search queries to generate

        Returns:
            List of targeted search queries

        Raises:
            SearchError: If API call fails
        """
        system_prompt = """You are a code search expert. Your task is to convert natural language questions about code into targeted search queries.

Given a natural language query, generate {limit} specific search queries that will help find the relevant code.

Rules:
1. Each query should target a different aspect of the question
2. Use technical terms and identifiers when possible
3. Keep queries concise (3-7 words each)
4. Focus on code patterns, function names, class names, or concepts
5. Return ONLY the search queries, one per line, no explanations

Example:
Input: "where is the similarity_threshold parameter set?"
Output:
similarity_threshold default value
similarity_threshold configuration
SemanticSearchEngine init threshold"""

        user_prompt = f"""Natural language query: {natural_language_query}

Generate {limit} targeted search queries:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt.format(limit=limit)},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._chat_completion(messages)

            # Parse queries from response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            queries = [q.strip() for q in content.strip().split("\n") if q.strip()]

            logger.debug(
                f"Generated {len(queries)} search queries from: '{natural_language_query}'"
            )

            return queries[:limit]

        except Exception as e:
            logger.error(f"Failed to generate search queries: {e}")
            raise SearchError(f"LLM query generation failed: {e}") from e

    async def analyze_and_rank_results(
        self,
        original_query: str,
        search_results: dict[str, list[Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze search results and select the most relevant ones.

        Args:
            original_query: Original natural language query
            search_results: Dictionary mapping search queries to their results
            top_n: Number of top results to return

        Returns:
            List of ranked results with explanations

        Raises:
            SearchError: If API call fails
        """
        # Format results for LLM analysis
        results_summary = self._format_results_for_analysis(search_results)

        system_prompt = """You are a code search expert. Your task is to analyze search results and identify the most relevant ones for answering a user's question.

Given:
1. A natural language query
2. Multiple search results from different queries

Select the top {top_n} most relevant results that best answer the user's question.

For each selected result, provide:
1. Result identifier (e.g., "Query 1, Result 2")
2. Relevance level: "High", "Medium", or "Low"
3. Brief explanation (1-2 sentences) of why this result is relevant

Format your response as:
RESULT: [identifier]
RELEVANCE: [level]
EXPLANATION: [why this matches]

---

Only include the top {top_n} results."""

        user_prompt = f"""Original Question: {original_query}

Search Results:
{results_summary}

Select the top {top_n} most relevant results:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt.format(top_n=top_n)},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._chat_completion(messages)

            # Parse LLM response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            ranked_results = self._parse_ranking_response(
                content, search_results, top_n
            )

            logger.debug(f"Ranked {len(ranked_results)} results from LLM analysis")

            return ranked_results

        except Exception as e:
            logger.error(f"Failed to analyze results: {e}")
            raise SearchError(f"LLM analysis failed: {e}") from e

    async def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make chat completion request to OpenAI or OpenRouter API.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            API response dictionary

        Raises:
            SearchError: If API request fails
        """
        # Build headers based on provider
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter-specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/bobmatnyc/mcp-vector-search"
            headers["X-Title"] = "MCP Vector Search"

        payload = {
            "model": self.model,
            "messages": messages,
        }

        provider_name = self.provider.capitalize()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()
                return response.json()

        except httpx.TimeoutException as e:
            logger.error(f"{provider_name} API timeout after {self.timeout}s")
            raise SearchError(
                f"LLM request timed out after {self.timeout} seconds. "
                "Try a simpler query or check your network connection."
            ) from e

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_msg = f"{provider_name} API error (HTTP {status_code})"

            if status_code == 401:
                env_var = (
                    "OPENAI_API_KEY"
                    if self.provider == "openai"
                    else "OPENROUTER_API_KEY"
                )
                error_msg = f"Invalid {provider_name} API key. Please check {env_var} environment variable."
            elif status_code == 429:
                error_msg = f"{provider_name} API rate limit exceeded. Please wait and try again."
            elif status_code >= 500:
                error_msg = f"{provider_name} API server error. Please try again later."

            logger.error(error_msg)
            raise SearchError(error_msg) from e

        except Exception as e:
            logger.error(f"{provider_name} API request failed: {e}")
            raise SearchError(f"LLM request failed: {e}") from e

    def _format_results_for_analysis(self, search_results: dict[str, list[Any]]) -> str:
        """Format search results for LLM analysis.

        Args:
            search_results: Dictionary mapping search queries to their results

        Returns:
            Formatted string representation of results
        """
        formatted = []

        for i, (query, results) in enumerate(search_results.items(), 1):
            formatted.append(f"\n=== Query {i}: {query} ===")

            if not results:
                formatted.append("  No results found.")
                continue

            for j, result in enumerate(results[:5], 1):  # Top 5 per query
                # Extract key information from SearchResult
                file_path = str(result.file_path)
                similarity = result.similarity_score
                content_preview = result.content[:150].replace("\n", " ")

                formatted.append(
                    f"\n  Result {j}:\n"
                    f"    File: {file_path}\n"
                    f"    Similarity: {similarity:.3f}\n"
                    f"    Preview: {content_preview}..."
                )

                if result.function_name:
                    formatted.append(f"    Function: {result.function_name}")
                if result.class_name:
                    formatted.append(f"    Class: {result.class_name}")

        return "\n".join(formatted)

    def _parse_ranking_response(
        self,
        llm_response: str,
        search_results: dict[str, list[Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Parse LLM ranking response into structured results.

        Args:
            llm_response: Raw LLM response text
            search_results: Original search results dictionary
            top_n: Maximum number of results to return

        Returns:
            List of ranked results with metadata
        """
        ranked = []
        current_result = {}

        for line in llm_response.split("\n"):
            line = line.strip()

            if line.startswith("RESULT:"):
                if current_result:
                    ranked.append(current_result)
                current_result = {"identifier": line.replace("RESULT:", "").strip()}

            elif line.startswith("RELEVANCE:"):
                current_result["relevance"] = line.replace("RELEVANCE:", "").strip()

            elif line.startswith("EXPLANATION:"):
                current_result["explanation"] = line.replace("EXPLANATION:", "").strip()

        # Add last result
        if current_result:
            ranked.append(current_result)

        # Map identifiers back to actual SearchResult objects
        enriched_results = []

        for item in ranked[:top_n]:
            identifier = item.get("identifier", "")

            # Parse identifier (e.g., "Query 1, Result 2" or "Query 1, Result 2 (filename.py)")
            try:
                parts = identifier.split(",")
                query_part = parts[0].replace("Query", "").strip()
                result_part = parts[1].replace("Result", "").strip()

                # Handle case where LLM includes filename in parentheses: "5 (config.py)"
                # Extract just the number
                query_match = re.match(r"(\d+)", query_part)
                result_match = re.match(r"(\d+)", result_part)

                if not query_match or not result_match:
                    logger.warning(
                        f"Could not extract numbers from identifier '{identifier}'"
                    )
                    continue

                query_idx = int(query_match.group(1)) - 1
                result_idx = int(result_match.group(1)) - 1

                # Get corresponding query and result
                queries = list(search_results.keys())
                if query_idx < len(queries):
                    query = queries[query_idx]
                    results = search_results[query]

                    if result_idx < len(results):
                        actual_result = results[result_idx]

                        enriched_results.append(
                            {
                                "result": actual_result,
                                "query": query,
                                "relevance": item.get("relevance", "Medium"),
                                "explanation": item.get(
                                    "explanation", "Relevant to query"
                                ),
                            }
                        )

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse result identifier '{identifier}': {e}")
                continue

        return enriched_results
