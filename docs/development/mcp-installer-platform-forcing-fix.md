# MCPInstaller Platform Forcing Fix

## Problem Description

When a platform was forced in `MCPInstaller` initialization (e.g., `Platform.CLAUDE_CODE`), the code would auto-detect the best platform across all platforms and then compare it to the forced platform. If another platform (e.g., `claude_desktop`) had higher confidence, it would raise `PlatformNotSupportedError` even though the forced platform (`claude_code`) WAS actually supported.

### Root Cause

In `installer.py` lines 139-148, the broken logic was:

```python
if platform:
    # For forced platform, we still need to detect but validate it matches
    detector = PlatformDetector()
    detected_info = detector.detect()  # This gets the BEST match, not the forced one
    if detected_info.platform != platform:  # This fails if best != forced
        raise PlatformNotSupportedError(
            platform.value,
            [p.value for p in Platform if p != Platform.UNKNOWN],
        )
    self._platform_info = detected_info
```

The issue was that `detector.detect()` returns the platform with the **highest confidence** across all platforms, not the confidence for the specific forced platform.

## Solution

### Changes Made

#### 1. Added `detect_for_platform()` method to `PlatformDetector`

**File**: `vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py`

Added a new method that detects platform info specifically for a given platform:

```python
def detect_for_platform(self, platform: Platform) -> PlatformInfo | None:
    """Detect platform info for a specific platform.

    Args:
        platform: The specific platform to detect

    Returns:
        PlatformInfo for the platform, or None if not detectable
    """
    # Map platform to its detector method
    detector_map = {
        Platform.CLAUDE_CODE: self.detect_claude_code,
        Platform.CLAUDE_DESKTOP: self.detect_claude_desktop,
        Platform.CURSOR: self.detect_cursor,
        Platform.AUGGIE: self.detect_auggie,
        Platform.CODEX: self.detect_codex,
        Platform.GEMINI_CLI: self.detect_gemini_cli,
        Platform.WINDSURF: self.detect_windsurf,
        Platform.ANTIGRAVITY: self.detect_antigravity,
    }

    detector = detector_map.get(platform)
    if not detector:
        return None

    confidence, config_path = detector()

    # Determine CLI availability for this platform
    cli_available = False
    if platform in (Platform.CLAUDE_CODE, Platform.CLAUDE_DESKTOP):
        cli_available = resolve_command_path("claude") is not None
    elif platform == Platform.CURSOR:
        cli_available = resolve_command_path("cursor") is not None
    elif platform == Platform.CODEX:
        cli_available = resolve_command_path("codex") is not None
    elif platform == Platform.GEMINI_CLI:
        cli_available = resolve_command_path("gemini") is not None

    return PlatformInfo(
        platform=platform,
        confidence=confidence,
        config_path=config_path,
        cli_available=cli_available,
        scope_support=Scope.BOTH,
    )
```

#### 2. Updated `MCPInstaller.__init__()` to use new method

**File**: `vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py`

Replaced lines 139-148 with the corrected logic:

```python
if platform:
    # For forced platform, detect info specifically for that platform
    detector = PlatformDetector()
    platform_info = detector.detect_for_platform(platform)
    if platform_info is None or platform_info.confidence == 0:
        raise PlatformNotSupportedError(
            platform.value,
            [p.value for p in Platform if p != Platform.UNKNOWN],
        )
    self._platform_info = platform_info
```

Now the code:
1. Detects platform info **specifically for the forced platform**
2. Only raises `PlatformNotSupportedError` if that specific platform has **zero confidence** (cannot be detected at all)
3. Allows forcing a platform even if another platform has higher confidence

## Testing

Created test file: `tests/manual/test_mcp_installer_platform_fix.py`

### Test Results

```
============================================================
MCPInstaller Platform Forcing Fix - Test Suite
============================================================
Testing forced platform detection...
✅ Successfully initialized with forced platform: claude_code
   Confidence: 0.90
   Config path: /Users/masa/.claude.json

Testing undetectable platform...
✅ Correctly raised PlatformNotSupportedError for undetectable platform

============================================================
Test Results Summary
============================================================
✅ PASS: Forced platform detection
✅ PASS: Undetectable platform

Total: 2/2 tests passed
```

### Test Cases Covered

1. **Forced platform with detectable confidence**: Should succeed
2. **Forced platform with zero confidence**: Should raise `PlatformNotSupportedError`

## Behavior Changes

### Before Fix

- Forcing `Platform.CLAUDE_CODE` would fail if `Platform.CLAUDE_DESKTOP` had higher confidence
- Comparison was: `detected_best_platform == forced_platform`
- This was incorrect - we should check if forced platform is detectable, not if it's the best

### After Fix

- Forcing `Platform.CLAUDE_CODE` succeeds as long as `claude_code` is detectable (confidence > 0)
- Comparison is: `forced_platform.confidence > 0`
- This is correct - we only error if the specific forced platform cannot be detected at all

## Impact

- **Positive**: Users can now force a specific platform even when multiple platforms are available
- **No Breaking Changes**: The fix only affects the platform forcing logic when a platform parameter is explicitly provided
- **Backward Compatible**: Auto-detection (no platform parameter) behavior unchanged

## Files Modified

1. `vendor/py-mcp-installer-service/src/py_mcp_installer/installer.py`
   - Updated `__init__()` method (lines 139-148)

2. `vendor/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py`
   - Added `detect_for_platform()` method (52 lines)

3. `tests/manual/test_mcp_installer_platform_fix.py` (new)
   - Created test suite to verify fix

## Related Issues

- Fixes platform forcing bug where forcing a detectable platform would fail if another platform had higher confidence
- Enables proper multi-platform support where users can choose their preferred platform

---

**Date**: December 8, 2025
**Status**: ✅ Implemented and Tested
