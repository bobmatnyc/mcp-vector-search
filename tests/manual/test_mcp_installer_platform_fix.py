#!/usr/bin/env python3
"""Test script to verify MCPInstaller platform forcing fix.

This test verifies that when a platform is forced (e.g., Platform.CLAUDE_CODE),
the installer correctly detects that specific platform instead of comparing
it to the best detected platform.

Run from project root:
    python tests/manual/test_mcp_installer_platform_fix.py
"""

import sys
from pathlib import Path

# Add vendor path to sys.path
vendor_path = (
    Path(__file__).parent.parent.parent / "vendor" / "py-mcp-installer-service" / "src"
)
sys.path.insert(0, str(vendor_path))

from py_mcp_installer.exceptions import PlatformNotSupportedError
from py_mcp_installer.installer import MCPInstaller
from py_mcp_installer.types import Platform


def test_forced_platform_detection():
    """Test that forcing a platform works when that platform is detectable."""
    print("Testing forced platform detection...")

    try:
        # Try to force CLAUDE_CODE platform
        # This should succeed if claude_code is detectable, even if
        # claude_desktop has higher confidence
        installer = MCPInstaller(platform=Platform.CLAUDE_CODE, verbose=True)

        print(
            f"✅ Successfully initialized with forced platform: {installer.platform_info.platform.value}"
        )
        print(f"   Confidence: {installer.platform_info.confidence:.2f}")
        print(f"   Config path: {installer.platform_info.config_path}")

        return True

    except PlatformNotSupportedError as e:
        print("❌ FAILED: PlatformNotSupportedError raised")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print("❌ FAILED: Unexpected error")
        print(f"   Error: {e}")
        return False


def test_undetectable_platform():
    """Test that forcing an undetectable platform still raises error."""
    print("\nTesting undetectable platform...")

    try:
        # Try to force ANTIGRAVITY platform (should be undetectable)
        installer = MCPInstaller(platform=Platform.ANTIGRAVITY, verbose=True)

        print("❌ FAILED: Should have raised PlatformNotSupportedError")
        print(f"   Got: {installer.platform_info.platform.value}")
        return False

    except PlatformNotSupportedError:
        print("✅ Correctly raised PlatformNotSupportedError for undetectable platform")
        return True
    except Exception as e:
        print("❌ FAILED: Unexpected error")
        print(f"   Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MCPInstaller Platform Forcing Fix - Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Forcing a detectable platform
    results.append(("Forced platform detection", test_forced_platform_detection()))

    # Test 2: Forcing an undetectable platform
    results.append(("Undetectable platform", test_undetectable_platform()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    sys.exit(main())
