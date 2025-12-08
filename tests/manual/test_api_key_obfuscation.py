#!/usr/bin/env python3
"""Manual test for API key obfuscation logic.

This script tests the _obfuscate_api_key function to ensure it properly
masks API keys for display.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_vector_search.cli.commands.setup import _obfuscate_api_key


def test_obfuscation():
    """Test API key obfuscation with various inputs."""
    test_cases = [
        # (input, expected_output, description)
        ("sk-or-v1-1234567890abcdef", "sk-or-...cdef", "Standard OpenRouter key"),
        ("short", "****...hort", "Short key (<10 chars)"),
        ("1234567890", "123456...7890", "Exactly 10 chars"),
        ("abc", "****...abc", "Very short key (3 chars)"),
        (
            "very_long_api_key_with_many_characters",
            "very_l...ters",
            "Long key (>10 chars)",
        ),
        ("", "****", "Empty string"),
    ]

    print("Testing API Key Obfuscation")
    print("=" * 60)

    all_passed = True
    for api_key, expected, description in test_cases:
        result = _obfuscate_api_key(api_key)
        passed = result == expected

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}: {description}")
        print(f"  Input:    '{api_key}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(test_obfuscation())
