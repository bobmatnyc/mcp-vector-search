#!/usr/bin/env python3
"""Verification script for Claude Sonnet 4.6 Bedrock integration.

This script verifies that:
1. Claude Sonnet 4.6 is the default Bedrock model
2. Bedrock credential detection works correctly
3. Custom endpoint support is functional
4. All expected environment variables are supported
"""

import os
import sys

# Add src to path
sys.path.insert(0, "src")

from mcp_vector_search.core.llm_client import LLMClient


def test_default_models():
    """Verify default models are set correctly."""
    print("✓ Testing default models...")

    assert (
        LLMClient.DEFAULT_MODELS["bedrock"]
        == "us.anthropic.claude-sonnet-4-6-20250514-v1:0"
    ), f"Expected Claude Sonnet 4.6, got {LLMClient.DEFAULT_MODELS['bedrock']}"

    assert (
        LLMClient.THINKING_MODELS["bedrock"] == "anthropic.claude-opus-4-20250514-v1:0"
    ), f"Expected Claude Opus 4, got {LLMClient.THINKING_MODELS['bedrock']}"

    print(f"  ✓ Default Bedrock model: {LLMClient.DEFAULT_MODELS['bedrock']}")
    print(f"  ✓ Thinking Bedrock model: {LLMClient.THINKING_MODELS['bedrock']}")


def test_bedrock_detection_with_creds():
    """Test Bedrock availability detection with credentials."""
    print("\n✓ Testing Bedrock detection with credentials...")

    # Save original env vars
    orig_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    orig_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    try:
        # Test 1: No credentials
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        os.environ.pop("ANTHROPIC_BEDROCK_BASE_URL", None)

        client = LLMClient.__new__(LLMClient)
        assert not client._bedrock_available, (
            "Should not detect Bedrock without credentials"
        )
        print("  ✓ Correctly detects no Bedrock credentials")

        # Test 2: With credentials
        os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"

        client = LLMClient.__new__(LLMClient)
        assert client._bedrock_available, "Should detect Bedrock with credentials"
        print("  ✓ Correctly detects Bedrock credentials")

        # Test 3: With custom endpoint
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        os.environ["ANTHROPIC_BEDROCK_BASE_URL"] = "https://custom.endpoint.com"

        client = LLMClient.__new__(LLMClient)
        assert client._bedrock_available, "Should detect Bedrock with custom endpoint"
        print("  ✓ Correctly detects custom Bedrock endpoint")

    finally:
        # Restore original env vars
        if orig_access_key:
            os.environ["AWS_ACCESS_KEY_ID"] = orig_access_key
        else:
            os.environ.pop("AWS_ACCESS_KEY_ID", None)

        if orig_secret_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = orig_secret_key
        else:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

        os.environ.pop("ANTHROPIC_BEDROCK_BASE_URL", None)


def test_provider_priority():
    """Test provider auto-detection priority."""
    print("\n✓ Testing provider priority...")

    # Save original env vars
    orig_openai = os.environ.get("OPENAI_API_KEY")
    orig_openrouter = os.environ.get("OPENROUTER_API_KEY")
    orig_aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    orig_aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

    try:
        # Clear all credentials
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

        # Test 1: Bedrock takes priority
        os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
        os.environ["OPENROUTER_API_KEY"] = "test-or-key"
        os.environ["OPENAI_API_KEY"] = "test-openai-key"

        client = LLMClient()
        assert client.provider == "bedrock", f"Expected bedrock, got {client.provider}"
        print("  ✓ Bedrock has highest priority")

        # Test 2: OpenRouter is second
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

        client = LLMClient()
        assert client.provider == "openrouter", (
            f"Expected openrouter, got {client.provider}"
        )
        print("  ✓ OpenRouter has second priority")

        # Test 3: OpenAI is third
        os.environ.pop("OPENROUTER_API_KEY", None)

        client = LLMClient()
        assert client.provider == "openai", f"Expected openai, got {client.provider}"
        print("  ✓ OpenAI has third priority")

    finally:
        # Restore original env vars
        if orig_openai:
            os.environ["OPENAI_API_KEY"] = orig_openai
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        if orig_openrouter:
            os.environ["OPENROUTER_API_KEY"] = orig_openrouter
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)

        if orig_aws_key:
            os.environ["AWS_ACCESS_KEY_ID"] = orig_aws_key
        else:
            os.environ.pop("AWS_ACCESS_KEY_ID", None)

        if orig_aws_secret:
            os.environ["AWS_SECRET_ACCESS_KEY"] = orig_aws_secret
        else:
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)


def test_model_override():
    """Test model override via environment variable."""
    print("\n✓ Testing model override...")

    orig_openai = os.environ.get("OPENAI_API_KEY")
    orig_bedrock_model = os.environ.get("BEDROCK_MODEL")

    try:
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ.pop("BEDROCK_MODEL", None)

        # Test default model
        client = LLMClient(provider="openai")
        assert client.model == "gpt-4o-mini", (
            f"Expected gpt-4o-mini, got {client.model}"
        )
        print(f"  ✓ Default OpenAI model: {client.model}")

        # Test thinking model
        client = LLMClient(provider="openai", think=True)
        assert client.model == "gpt-4o", f"Expected gpt-4o, got {client.model}"
        print(f"  ✓ Thinking OpenAI model: {client.model}")

        # Test custom model via parameter
        client = LLMClient(provider="openai", model="gpt-4-turbo")
        assert client.model == "gpt-4-turbo", (
            f"Expected gpt-4-turbo, got {client.model}"
        )
        print(f"  ✓ Custom model override: {client.model}")

    finally:
        if orig_openai:
            os.environ["OPENAI_API_KEY"] = orig_openai
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        if orig_bedrock_model:
            os.environ["BEDROCK_MODEL"] = orig_bedrock_model
        else:
            os.environ.pop("BEDROCK_MODEL", None)


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Claude Sonnet 4.6 Bedrock Integration Verification")
    print("=" * 60)

    try:
        test_default_models()
        test_bedrock_detection_with_creds()
        test_provider_priority()
        test_model_override()

        print("\n" + "=" * 60)
        print("✅ All verification tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
