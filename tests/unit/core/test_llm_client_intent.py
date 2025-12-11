"""Tests for LLM client intent detection with analyze support."""

from unittest.mock import AsyncMock, patch

import pytest

from mcp_vector_search.core.llm_client import IntentType, LLMClient


@pytest.fixture
def llm_client():
    """Create LLM client for testing."""
    return LLMClient(
        openai_api_key="test-key",
        provider="openai",
        timeout=30.0,
    )


class TestIntentDetection:
    """Test intent detection with analyze support."""

    @pytest.mark.asyncio
    async def test_detect_find_intent(self, llm_client):
        """Test detection of 'find' intent."""
        # Mock API response
        mock_response = {"choices": [{"message": {"content": "find"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("where is the authentication code?")

            assert intent == "find"
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_answer_intent(self, llm_client):
        """Test detection of 'answer' intent."""
        mock_response = {"choices": [{"message": {"content": "answer"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("how does authentication work?")

            assert intent == "answer"
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_analyze_intent_complexity(self, llm_client):
        """Test detection of 'analyze' intent for complexity queries."""
        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            # Test various complexity-related queries
            complexity_queries = [
                "what's the cognitive complexity?",
                "analyze complexity",
                "find complex functions",
                "what's the most complex code?",
            ]

            for query in complexity_queries:
                intent = await llm_client.detect_intent(query)
                assert intent == "analyze", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_detect_analyze_intent_quality(self, llm_client):
        """Test detection of 'analyze' intent for quality queries."""
        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            # Test various quality-related queries
            quality_queries = [
                "find code smells",
                "what are the quality issues?",
                "analyze the code quality",
                "find the worst code",
            ]

            for query in quality_queries:
                intent = await llm_client.detect_intent(query)
                assert intent == "analyze", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_detect_analyze_intent_coupling(self, llm_client):
        """Test detection of 'analyze' intent for coupling queries."""
        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            # Test various coupling-related queries
            coupling_queries = [
                "what are the dependencies?",
                "check circular dependencies",
                "analyze coupling",
                "find tightly coupled code",
            ]

            for query in coupling_queries:
                intent = await llm_client.detect_intent(query)
                assert intent == "analyze", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_detect_analyze_intent_trends(self, llm_client):
        """Test detection of 'analyze' intent for trend queries."""
        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            # Test various trend-related queries
            trend_queries = [
                "is complexity getting worse?",
                "show me quality trends",
                "is the code improving?",
                "track complexity over time",
            ]

            for query in trend_queries:
                intent = await llm_client.detect_intent(query)
                assert intent == "analyze", f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_detect_intent_defaults_to_find_on_unclear(self, llm_client):
        """Test that unclear intent defaults to 'find'."""
        mock_response = {"choices": [{"message": {"content": "unclear"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("some unclear query")

            assert intent == "find"

    @pytest.mark.asyncio
    async def test_detect_intent_defaults_to_find_on_error(self, llm_client):
        """Test that errors default to 'find' intent."""
        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.side_effect = Exception("API error")

            intent = await llm_client.detect_intent("test query")

            assert intent == "find"

    @pytest.mark.asyncio
    async def test_intent_type_is_literal(self):
        """Test that IntentType is correctly typed as Literal."""
        from typing import get_args

        # Verify IntentType includes all three values
        assert "find" in get_args(IntentType)
        assert "answer" in get_args(IntentType)
        assert "analyze" in get_args(IntentType)

    @pytest.mark.asyncio
    async def test_detect_intent_prompt_includes_analyze(self, llm_client):
        """Test that intent detection prompt includes analyze examples."""
        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_response = {"choices": [{"message": {"content": "analyze"}}]}
            mock_chat.return_value = mock_response

            await llm_client.detect_intent("test query")

            # Verify the prompt was called
            call_args = mock_chat.call_args
            messages = call_args[0][0]

            # Check system prompt contains analyze definition
            system_prompt = messages[0]["content"]
            assert "analyze" in system_prompt.lower()
            assert "complexity" in system_prompt.lower()
            assert "code smells" in system_prompt.lower()
            assert "dependencies" in system_prompt.lower()


class TestIntentDetectionEdgeCases:
    """Test edge cases for intent detection."""

    @pytest.mark.asyncio
    async def test_empty_query(self, llm_client):
        """Test intent detection with empty query."""
        mock_response = {"choices": [{"message": {"content": "find"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("")

            assert intent in ("find", "answer", "analyze")

    @pytest.mark.asyncio
    async def test_very_long_query(self, llm_client):
        """Test intent detection with very long query."""
        long_query = "analyze " * 1000  # Very long query

        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent(long_query)

            assert intent == "analyze"

    @pytest.mark.asyncio
    async def test_mixed_intent_query(self, llm_client):
        """Test query with mixed intent keywords."""
        mock_response = {"choices": [{"message": {"content": "analyze"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            # Query contains both "find" and "analyze" keywords
            intent = await llm_client.detect_intent(
                "find the code and analyze its complexity"
            )

            # LLM should pick the most appropriate intent
            assert intent in ("find", "answer", "analyze")

    @pytest.mark.asyncio
    async def test_case_insensitive_response(self, llm_client):
        """Test that intent detection is case-insensitive."""
        mock_response = {"choices": [{"message": {"content": "ANALYZE"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("test query")

            assert intent == "analyze"

    @pytest.mark.asyncio
    async def test_whitespace_in_response(self, llm_client):
        """Test that whitespace is stripped from response."""
        mock_response = {"choices": [{"message": {"content": "  analyze  \n"}}]}

        with patch.object(
            llm_client, "_chat_completion", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = mock_response

            intent = await llm_client.detect_intent("test query")

            assert intent == "analyze"
