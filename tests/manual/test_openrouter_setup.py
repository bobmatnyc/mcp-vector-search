#!/usr/bin/env python3
"""Manual test script for OpenRouter API key setup function."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_vector_search.cli.commands.setup import setup_openrouter_api_key


def test_without_api_key():
    """Test when API key is not set."""
    print("\n" + "=" * 80)
    print("TEST 1: API Key NOT Set")
    print("=" * 80)

    # Ensure key is not set
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]

    result = setup_openrouter_api_key()

    print(f"\nResult: {result}")
    assert result is False, "Should return False when API key is not set"
    print("✓ Test passed: Returns False when API key not set")


def test_with_api_key():
    """Test when API key is set."""
    print("\n" + "=" * 80)
    print("TEST 2: API Key SET")
    print("=" * 80)

    # Set a dummy key
    os.environ["OPENROUTER_API_KEY"] = "test-key-12345"

    result = setup_openrouter_api_key()

    print(f"\nResult: {result}")
    assert result is True, "Should return True when API key is set"
    print("✓ Test passed: Returns True when API key is set")

    # Clean up
    del os.environ["OPENROUTER_API_KEY"]


if __name__ == "__main__":
    test_without_api_key()
    test_with_api_key()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
