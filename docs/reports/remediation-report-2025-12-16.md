# Code Remediation Report
Generated: 2025-12-16

## Summary

- **High Complexity Items**: 44
- **Code Smells Detected**: 633
- **Critical Issues (Errors)**: 340

---

## 🔴 Priority: High Complexity Code

These functions/methods have complexity scores that make them difficult to maintain and test.

| Grade | Name | File | Complexity | Lines |
|-------|------|------|------------|-------|
| 🟠 D | `_run_interactive_test` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 32 | 160 |
| 🟠 D | `BrowserClient` | /Users/masa/Projects/mcp-browser/src/cli/utils/browser_client.py | 34 | 377 |
| 🟠 D | `BrowserMCPServer` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 34 | 470 |
| 🟠 D | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | 37 | 300 |
| 🟠 D | `AppleScriptService` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 33 | 607 |
| 🟠 D | `DaemonClient` | /Users/masa/Projects/mcp-browser/src/services/daemon_client.py | 36 | 267 |
| 🟠 D | `_format_semantic_dom` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 33 | 66 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 27 | 199 |
| 🟡 C | `CommandBuilder` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 27 | 419 |
| 🟡 C | `get_extension_files` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 21 | 64 |
| 🟡 C | `_display_semantic_dom` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 29 | 68 |
| 🟡 C | `_step_content_extraction` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 24 | 164 |
| 🟡 C | `DevelopmentServer` | /Users/masa/Projects/mcp-browser/src/dev_runner.py | 22 | 156 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 27 | 199 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 27 | 199 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 27 | 199 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 27 | 199 |
| 🟡 C | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 27 | 199 |
| 🟡 C | `DOMInteractionService` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | 25 | 479 |
| 🟡 C | `_handle_message` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 22 | 177 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 78 | 567 |
| 🔴 F | `DashboardService` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | 48 | 406 |
| 🔴 F | `ConfigManager` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 50 | 500 |
| 🔴 F | `MCPInstaller` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 58 | 658 |
| 🔴 F | `MCPDoctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 64 | 836 |
| 🔴 F | `MCPInspector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 55 | 609 |
| 🔴 F | `PlatformDetector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 56 | 478 |
| 🔴 F | `ExtensionBuilder` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 61 | 334 |
| 🔴 F | `uninstall` | /Users/masa/Projects/mcp-browser/src/cli/commands/install_legacy.py | 46 | 239 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 78 | 567 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 78 | 567 |
| 🔴 F | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 102 | 1083 |
| 🔴 F | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 49 | 437 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 78 | 567 |
| 🔴 F | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 100 | 1063 |
| 🔴 F | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 48 | 434 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 78 | 567 |
| 🔴 F | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 78 | 567 |
| 🔴 F | `BrowserController` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 79 | 873 |
| 🔴 F | `BrowserService` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | 81 | 673 |
| 🔴 F | `MCPService` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 132 | 1012 |
| 🔴 F | `StorageService` | /Users/masa/Projects/mcp-browser/src/services/storage_service.py | 47 | 305 |
| 🔴 F | `WebSocketService` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 48 | 547 |
| 🔴 F | `MCPBrowserTester` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 92 | 925 |

---

## 🔍 Code Smells

### Critical Issues (Errors)

| Smell | Name | File | Detail |
|-------|------|------|--------|
| 🔴 Deep Nesting | `background-enhanced.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `background.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `handleServerMessage` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background.js | 122 lines (recommended: <50) |
| 🔴 Deep Nesting | `content.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/content.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/manifest.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `popup-enhanced.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/popup-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `popup.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/popup.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `APPLESCRIPT_IMPLEMENTATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/docs/developer/APPLESCRIPT_IMPLEMENTATION_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `AUTO_START_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-browser/docs/developer/AUTO_START_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `BUILD_TRACKING.md` | /Users/masa/Projects/mcp-browser/docs/developer/BUILD_TRACKING.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `DOCTOR_COMMAND_ENHANCEMENT.md` | /Users/masa/Projects/mcp-browser/docs/developer/DOCTOR_COMMAND_ENHANCEMENT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `DOCTOR_QUICK_REFERENCE.md` | /Users/masa/Projects/mcp-browser/docs/developer/DOCTOR_QUICK_REFERENCE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `EXTENSION_CONSOLIDATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/docs/developer/EXTENSION_CONSOLIDATION_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `FIREFOX_PORT.md` | /Users/masa/Projects/mcp-browser/docs/developer/FIREFOX_PORT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `IMPLEMENTATION_COMPLETE.md` | /Users/masa/Projects/mcp-browser/docs/developer/IMPLEMENTATION_COMPLETE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `IMPLEMENTATION_ISSUE_19.md` | /Users/masa/Projects/mcp-browser/docs/developer/IMPLEMENTATION_ISSUE_19.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `IMPLEMENTATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/docs/developer/IMPLEMENTATION_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `PORT_SELECTOR_GUIDE.md` | /Users/masa/Projects/mcp-browser/docs/developer/PORT_SELECTOR_GUIDE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `QA_FIXES_AUTO_START.md` | /Users/masa/Projects/mcp-browser/docs/developer/QA_FIXES_AUTO_START.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASE_AUTOMATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/docs/developer/RELEASE_AUTOMATION_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASE_SCRIPT_SUMMARY.md` | /Users/masa/Projects/mcp-browser/docs/developer/RELEASE_SCRIPT_SUMMARY.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `SAFARI_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-browser/docs/developer/SAFARI_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `STATE_RECOVERY_IMPLEMENTATION.md` | /Users/masa/Projects/mcp-browser/docs/developer/STATE_RECOVERY_IMPLEMENTATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VERIFICATION_CHECKLIST.md` | /Users/masa/Projects/mcp-browser/docs/developer/VERIFICATION_CHECKLIST.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `APPLESCRIPT_QUICK_START.md` | /Users/masa/Projects/mcp-browser/docs/guides/APPLESCRIPT_QUICK_START.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `APPLESCRIPT_USAGE.md` | /Users/masa/Projects/mcp-browser/docs/guides/APPLESCRIPT_USAGE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `INSTALLATION.md` | /Users/masa/Projects/mcp-browser/docs/guides/INSTALLATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASE_QUICK_REFERENCE.md` | /Users/masa/Projects/mcp-browser/docs/guides/releases/RELEASE_QUICK_REFERENCE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASE.md` | /Users/masa/Projects/mcp-browser/docs/guides/releases/RELEASE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `UNINSTALL.md` | /Users/masa/Projects/mcp-browser/docs/guides/UNINSTALL.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp-brower.md` | /Users/masa/Projects/mcp-browser/docs/prd/mcp-brower.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `PROJECT_ORGANIZATION.md` | /Users/masa/Projects/mcp-browser/docs/reference/PROJECT_ORGANIZATION.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `UNINSTALL_TEST_REPORT.md` | /Users/masa/Projects/mcp-browser/docs/testing/UNINSTALL_TEST_REPORT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `UNINSTALL_USAGE_EXAMPLES.md` | /Users/masa/Projects/mcp-browser/docs/testing/UNINSTALL_USAGE_EXAMPLES.md | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `example_usage` | /Users/masa/Projects/mcp-browser/examples/usage-example.py | 119 lines (recommended: <50) |
| 🔴 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/CHANGELOG.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `CONTRIBUTING.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/CONTRIBUTING.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `ARCHITECTURE.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/ARCHITECTURE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `SUMMARY.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/design/SUMMARY.md | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `DIAGRAMS.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `IMPLEMENTATION-PLAN.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/IMPLEMENTATION-PLAN.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `PROJECT-STRUCTURE.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/PROJECT-STRUCTURE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `QUICK-REFERENCE.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/QUICK-REFERENCE.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/README.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `phase3_demo.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/examples/phase3_demo.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `RELEASING.md` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/RELEASING.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `manage_version.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/scripts/manage_version.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `release.sh` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/scripts/release.sh | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `setup.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/setup.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/__init__.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `cli.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/cli.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `command_builder.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `CommandBuilder` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 419 lines (recommended: <50) |
| 🔴 High Complexity | `CommandBuilder` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | Complexity: 27 (recommended: <15) |
| 🔴 God Class | `CommandBuilder` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 419 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `config_manager.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `ConfigManager` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 500 lines (recommended: <50) |
| 🔴 High Complexity | `ConfigManager` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | Complexity: 50 (recommended: <15) |
| 🔴 God Class | `ConfigManager` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 500 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `exceptions.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/exceptions.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `installation_strategy.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `NativeCLIStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 242 lines (recommended: <50) |
| 🔴 Long Method | `JSONManipulationStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 141 lines (recommended: <50) |
| 🔴 Long Method | `TOMLManipulationStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 126 lines (recommended: <50) |
| 🔴 Deep Nesting | `installer.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MCPInstaller` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 658 lines (recommended: <50) |
| 🔴 High Complexity | `MCPInstaller` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | Complexity: 58 (recommended: <15) |
| 🔴 God Class | `MCPInstaller` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 658 lines - consider breaking into smaller classes |
| 🔴 Long Method | `install_server` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 141 lines (recommended: <50) |
| 🔴 Deep Nesting | `mcp_doctor.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `DiagnosticReport` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 120 lines (recommended: <50) |
| 🔴 Long Method | `MCPDoctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 836 lines (recommended: <50) |
| 🔴 High Complexity | `MCPDoctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | Complexity: 64 (recommended: <15) |
| 🔴 God Class | `MCPDoctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 836 lines - consider breaking into smaller classes |
| 🔴 Long Method | `diagnose` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 124 lines (recommended: <50) |
| 🔴 Long Method | `test_server` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 116 lines (recommended: <50) |
| 🔴 Deep Nesting | `mcp_inspector.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `MCPInspector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 609 lines (recommended: <50) |
| 🔴 High Complexity | `MCPInspector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | Complexity: 55 (recommended: <15) |
| 🔴 God Class | `MCPInspector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 609 lines - consider breaking into smaller classes |
| 🔴 Long Method | `inspect` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 117 lines (recommended: <50) |
| 🔴 Long Method | `validate_server` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 110 lines (recommended: <50) |
| 🔴 Deep Nesting | `platform_detector.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | Depth: 9 (recommended: <4) |
| 🔴 Long Method | `PlatformDetector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 478 lines (recommended: <50) |
| 🔴 High Complexity | `PlatformDetector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | Complexity: 56 (recommended: <15) |
| 🔴 God Class | `PlatformDetector` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 478 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/__init__.py | Depth: 10 (recommended: <4) |
| 🔴 Deep Nesting | `claude_code.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/claude_code.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `ClaudeCodeStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/claude_code.py | 201 lines (recommended: <50) |
| 🔴 Deep Nesting | `codex.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/codex.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `CodexStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/codex.py | 159 lines (recommended: <50) |
| 🔴 Deep Nesting | `cursor.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/cursor.py | Depth: 10 (recommended: <4) |
| 🔴 Long Method | `CursorStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platforms/cursor.py | 169 lines (recommended: <50) |
| 🔴 Deep Nesting | `types.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/types.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `utils.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/utils.py | Depth: 9 (recommended: <4) |
| 🔴 Deep Nesting | `test_mcp_doctor.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `TestDiagnosticReport` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 148 lines (recommended: <50) |
| 🔴 Long Method | `TestMCPDoctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 264 lines (recommended: <50) |
| 🔴 Long Method | `TestServerProtocolTests` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 153 lines (recommended: <50) |
| 🔴 Long Method | `TestCLI` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 114 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_platform_detector.py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_platform_detector.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `MAKEFILE_EXTRACTION_REPORT.md` | /Users/masa/Projects/mcp-browser/lib/python-project-template/MAKEFILE_EXTRACTION_REPORT.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/lib/python-project-template/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `STRUCTURE.md` | /Users/masa/Projects/mcp-browser/lib/python-project-template/STRUCTURE.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `TEMPLATE_README.md` | /Users/masa/Projects/mcp-browser/lib/python-project-template/TEMPLATE_README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `VERIFICATION.md` | /Users/masa/Projects/mcp-browser/lib/python-project-template/VERIFICATION.md | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `ExtensionBuilder` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 334 lines (recommended: <50) |
| 🔴 High Complexity | `ExtensionBuilder` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | Complexity: 61 (recommended: <15) |
| 🔴 God Class | `ExtensionBuilder` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 334 lines - consider breaking into smaller classes |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 142 lines (recommended: <50) |
| 🔴 Long Method | `main` | /Users/masa/Projects/mcp-browser/scripts/release.py | 131 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/cli/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `browser.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | Depth: 8 (recommended: <4) |
| 🔴 High Complexity | `_display_semantic_dom` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | Complexity: 29 (recommended: <15) |
| 🔴 Long Method | `_run_interactive_test` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 160 lines (recommended: <50) |
| 🔴 High Complexity | `_run_interactive_test` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | Complexity: 32 (recommended: <15) |
| 🔴 Deep Nesting | `connect.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/connect.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_connect_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/connect.py | 138 lines (recommended: <50) |
| 🔴 Deep Nesting | `demo.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_step_console_logs` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 107 lines (recommended: <50) |
| 🔴 Long Method | `_step_content_extraction` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 164 lines (recommended: <50) |
| 🔴 Long Method | `_step_dom_interaction` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 161 lines (recommended: <50) |
| 🔴 Deep Nesting | `doctor.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_check_console_log_capture` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | 101 lines (recommended: <50) |
| 🔴 Deep Nesting | `extension.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/extension.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `init.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/init.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `init_project_extension` | /Users/masa/Projects/mcp-browser/src/cli/commands/init.py | 195 lines (recommended: <50) |
| 🔴 Long Method | `init_project_extension_interactive` | /Users/masa/Projects/mcp-browser/src/cli/commands/init.py | 139 lines (recommended: <50) |
| 🔴 Deep Nesting | `install_legacy.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/install_legacy.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `install` | /Users/masa/Projects/mcp-browser/src/cli/commands/install_legacy.py | 137 lines (recommended: <50) |
| 🔴 Long Method | `uninstall` | /Users/masa/Projects/mcp-browser/src/cli/commands/install_legacy.py | 239 lines (recommended: <50) |
| 🔴 High Complexity | `uninstall` | /Users/masa/Projects/mcp-browser/src/cli/commands/install_legacy.py | Complexity: 46 (recommended: <15) |
| 🔴 Deep Nesting | `install.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/install.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `install` | /Users/masa/Projects/mcp-browser/src/cli/commands/install.py | 149 lines (recommended: <50) |
| 🔴 Long Method | `uninstall` | /Users/masa/Projects/mcp-browser/src/cli/commands/install.py | 142 lines (recommended: <50) |
| 🔴 Deep Nesting | `quickstart.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/quickstart.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `quickstart` | /Users/masa/Projects/mcp-browser/src/cli/commands/quickstart.py | 163 lines (recommended: <50) |
| 🔴 Deep Nesting | `setup.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/setup.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `setup` | /Users/masa/Projects/mcp-browser/src/cli/commands/setup.py | 120 lines (recommended: <50) |
| 🔴 Long Method | `install_mcp` | /Users/masa/Projects/mcp-browser/src/cli/commands/setup.py | 116 lines (recommended: <50) |
| 🔴 Deep Nesting | `start.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/start.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `start` | /Users/masa/Projects/mcp-browser/src/cli/commands/start.py | 169 lines (recommended: <50) |
| 🔴 Deep Nesting | `status.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/status.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `stop.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/stop.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `tutorial.py` | /Users/masa/Projects/mcp-browser/src/cli/commands/tutorial.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `main.py` | /Users/masa/Projects/mcp-browser/src/cli/main.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/__init__.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `browser_client.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/browser_client.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `BrowserClient` | /Users/masa/Projects/mcp-browser/src/cli/utils/browser_client.py | 377 lines (recommended: <50) |
| 🔴 High Complexity | `BrowserClient` | /Users/masa/Projects/mcp-browser/src/cli/utils/browser_client.py | Complexity: 34 (recommended: <15) |
| 🔴 God Class | `BrowserClient` | /Users/masa/Projects/mcp-browser/src/cli/utils/browser_client.py | 377 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `daemon.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/daemon.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `display.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/display.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `server.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `BrowserMCPServer` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 470 lines (recommended: <50) |
| 🔴 High Complexity | `BrowserMCPServer` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | Complexity: 34 (recommended: <15) |
| 🔴 God Class | `BrowserMCPServer` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 470 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_setup_services` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 101 lines (recommended: <50) |
| 🔴 Deep Nesting | `validation.py` | /Users/masa/Projects/mcp-browser/src/cli/utils/validation.py | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/container/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `service_container.py` | /Users/masa/Projects/mcp-browser/src/container/service_container.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `ServiceContainer` | /Users/masa/Projects/mcp-browser/src/container/service_container.py | 185 lines (recommended: <50) |
| 🔴 Long Method | `DevelopmentServer` | /Users/masa/Projects/mcp-browser/src/dev_runner.py | 156 lines (recommended: <50) |
| 🔴 Deep Nesting | `background-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extension/background-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `background.js` | /Users/masa/Projects/mcp-browser/src/extension/background.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `handleServerMessage` | /Users/masa/Projects/mcp-browser/src/extension/background.js | 122 lines (recommended: <50) |
| 🔴 Deep Nesting | `content.js` | /Users/masa/Projects/mcp-browser/src/extension/content.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-browser/src/extension/manifest.json | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `popup-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extension/popup-enhanced.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `popup.js` | /Users/masa/Projects/mcp-browser/src/extension/popup.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `background-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `PortSelector` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 199 lines (recommended: <50) |
| 🔴 Long Method | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 1083 lines (recommended: <50) |
| 🔴 High Complexity | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | Complexity: 102 (recommended: <15) |
| 🔴 God Class | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 1083 lines - consider breaking into smaller classes |
| 🔴 Long Method | `connectToBackend` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 118 lines (recommended: <50) |
| 🔴 Long Method | `_setupMessageHandler` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 110 lines (recommended: <50) |
| 🔴 Long Method | `connectToServer` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 113 lines (recommended: <50) |
| 🔴 Long Method | `setupWebSocketHandlers` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 140 lines (recommended: <50) |
| 🔴 Long Method | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 437 lines (recommended: <50) |
| 🔴 High Complexity | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | Complexity: 49 (recommended: <15) |
| 🔴 Deep Nesting | `content.js` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/content.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/manifest.json | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `popup-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `background-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `PortSelector` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 199 lines (recommended: <50) |
| 🔴 Long Method | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 1063 lines (recommended: <50) |
| 🔴 High Complexity | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | Complexity: 100 (recommended: <15) |
| 🔴 God Class | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 1063 lines - consider breaking into smaller classes |
| 🔴 Long Method | `connectToBackend` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 118 lines (recommended: <50) |
| 🔴 Long Method | `_setupMessageHandler` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 110 lines (recommended: <50) |
| 🔴 Long Method | `connectToServer` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 113 lines (recommended: <50) |
| 🔴 Long Method | `setupWebSocketHandlers` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 140 lines (recommended: <50) |
| 🔴 Long Method | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 434 lines (recommended: <50) |
| 🔴 High Complexity | `executeCommandOnTab` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | Complexity: 48 (recommended: <15) |
| 🔴 Deep Nesting | `background.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | 300 lines (recommended: <50) |
| 🔴 High Complexity | `ConnectionManager` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | Complexity: 37 (recommended: <15) |
| 🔴 Deep Nesting | `content.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/content.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/manifest.json | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `popup-enhanced.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `popup.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/src/extensions/README.md | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `background.js` | /Users/masa/Projects/mcp-browser/src/extensions/safari/background.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `handleServerMessage` | /Users/masa/Projects/mcp-browser/src/extensions/safari/background.js | 122 lines (recommended: <50) |
| 🔴 Deep Nesting | `CHECKLIST.md` | /Users/masa/Projects/mcp-browser/src/extensions/safari/CHECKLIST.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `manifest.json` | /Users/masa/Projects/mcp-browser/src/extensions/safari/manifest.json | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `popup.js` | /Users/masa/Projects/mcp-browser/src/extensions/safari/popup.js | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/src/extensions/safari/README.md | Depth: 8 (recommended: <4) |
| 🔴 Deep Nesting | `Readability.js` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Depth: 8 (recommended: <4) |
| 🔴 Long Method | `_prepArticle` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 103 lines (recommended: <50) |
| 🔴 Long Method | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 567 lines (recommended: <50) |
| 🔴 High Complexity | `_grabArticle` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Complexity: 78 (recommended: <15) |
| 🔴 Long Method | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 116 lines (recommended: <50) |
| 🔴 Long Method | `_getArticleMetadata` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 107 lines (recommended: <50) |
| 🔴 Long Method | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 199 lines (recommended: <50) |
| 🔴 High Complexity | `_cleanConditionally` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Complexity: 27 (recommended: <15) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/models/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `browser_state.py` | /Users/masa/Projects/mcp-browser/src/models/browser_state.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `BrowserState` | /Users/masa/Projects/mcp-browser/src/models/browser_state.py | 133 lines (recommended: <50) |
| 🔴 Deep Nesting | `console_message.py` | /Users/masa/Projects/mcp-browser/src/models/console_message.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `ConsoleMessage` | /Users/masa/Projects/mcp-browser/src/models/console_message.py | 114 lines (recommended: <50) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/services/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `applescript_service.py` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `AppleScriptService` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 607 lines (recommended: <50) |
| 🔴 High Complexity | `AppleScriptService` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | Complexity: 33 (recommended: <15) |
| 🔴 God Class | `AppleScriptService` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 607 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `browser_controller.py` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `CapabilityDetector` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 130 lines (recommended: <50) |
| 🔴 Long Method | `BrowserController` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 873 lines (recommended: <50) |
| 🔴 High Complexity | `BrowserController` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | Complexity: 79 (recommended: <15) |
| 🔴 God Class | `BrowserController` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 873 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_try_extension` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 102 lines (recommended: <50) |
| 🔴 Long Method | `_try_applescript` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 113 lines (recommended: <50) |
| 🔴 Deep Nesting | `browser_service.py` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `BrowserService` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | 673 lines (recommended: <50) |
| 🔴 High Complexity | `BrowserService` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | Complexity: 81 (recommended: <15) |
| 🔴 God Class | `BrowserService` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | 673 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `daemon_client.py` | /Users/masa/Projects/mcp-browser/src/services/daemon_client.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `DaemonClient` | /Users/masa/Projects/mcp-browser/src/services/daemon_client.py | 267 lines (recommended: <50) |
| 🔴 High Complexity | `DaemonClient` | /Users/masa/Projects/mcp-browser/src/services/daemon_client.py | Complexity: 36 (recommended: <15) |
| 🔴 Deep Nesting | `dashboard_service.py` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `DashboardService` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | 406 lines (recommended: <50) |
| 🔴 High Complexity | `DashboardService` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | Complexity: 48 (recommended: <15) |
| 🔴 God Class | `DashboardService` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | 406 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `dom_interaction_service.py` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `DOMInteractionService` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | 479 lines (recommended: <50) |
| 🔴 God Class | `DOMInteractionService` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | 479 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `mcp_installer_bridge.py` | /Users/masa/Projects/mcp-browser/src/services/mcp_installer_bridge.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `mcp_service.py` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `MCPService` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 1012 lines (recommended: <50) |
| 🔴 High Complexity | `MCPService` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | Complexity: 132 (recommended: <15) |
| 🔴 God Class | `MCPService` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 1012 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_setup_tools` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 246 lines (recommended: <50) |
| 🔴 High Complexity | `_format_semantic_dom` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | Complexity: 33 (recommended: <15) |
| 🔴 Deep Nesting | `storage_service.py` | /Users/masa/Projects/mcp-browser/src/services/storage_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `StorageService` | /Users/masa/Projects/mcp-browser/src/services/storage_service.py | 305 lines (recommended: <50) |
| 🔴 High Complexity | `StorageService` | /Users/masa/Projects/mcp-browser/src/services/storage_service.py | Complexity: 47 (recommended: <15) |
| 🔴 God Class | `StorageService` | /Users/masa/Projects/mcp-browser/src/services/storage_service.py | 305 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `websocket_service.py` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `WebSocketService` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 547 lines (recommended: <50) |
| 🔴 High Complexity | `WebSocketService` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | Complexity: 48 (recommended: <15) |
| 🔴 God Class | `WebSocketService` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 547 lines - consider breaking into smaller classes |
| 🔴 Long Method | `_handle_message` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 177 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_port_selector.js` | /Users/masa/Projects/mcp-browser/tests/fixtures/test_port_selector.js | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test-page.js` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `MCPBrowserTester` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 925 lines (recommended: <50) |
| 🔴 High Complexity | `MCPBrowserTester` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | Complexity: 92 (recommended: <15) |
| 🔴 God Class | `MCPBrowserTester` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 925 lines - consider breaking into smaller classes |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/tests/integration/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_applescript_integration.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_applescript_integration.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_auto_start.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_auto_start.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_browser_control.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_browser_control.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_doctor_command.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_doctor_command.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_implementation.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_implementation.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_services` | /Users/masa/Projects/mcp-browser/tests/integration/test_implementation.py | 149 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_mcp_browser.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_mcp_browser.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_mcp_tools.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_mcp_tools.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_port_mismatch_fix.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_port_mismatch_fix.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_qa_fixes.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_qa_fixes.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_semantic_dom_ws.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_semantic_dom_ws.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_semantic_dom_via_websocket` | /Users/masa/Projects/mcp-browser/tests/integration/test_semantic_dom_ws.py | 178 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_semantic_dom.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_semantic_dom.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_semantic_dom` | /Users/masa/Projects/mcp-browser/tests/integration/test_semantic_dom.py | 115 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_service_integration.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_service_integration.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestServiceIntegration` | /Users/masa/Projects/mcp-browser/tests/integration/test_service_integration.py | 183 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_state_recovery.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_state_recovery.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_state_recovery` | /Users/masa/Projects/mcp-browser/tests/integration/test_state_recovery.py | 103 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_uninstall_integration.py` | /Users/masa/Projects/mcp-browser/tests/integration/test_uninstall_integration.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_applescript_service.py` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/tests/unit/__init__.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_mcp_protocol.py` | /Users/masa/Projects/mcp-browser/tests/unit/test_mcp_protocol.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `test_mcp_server` | /Users/masa/Projects/mcp-browser/tests/unit/test_mcp_protocol.py | 120 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_port_fallback.py` | /Users/masa/Projects/mcp-browser/tests/unit/test_port_fallback.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_service_container.py` | /Users/masa/Projects/mcp-browser/tests/unit/test_service_container.py | Depth: 7 (recommended: <4) |
| 🔴 Long Method | `TestServiceContainer` | /Users/masa/Projects/mcp-browser/tests/unit/test_service_container.py | 150 lines (recommended: <50) |
| 🔴 Deep Nesting | `test_setup.py` | /Users/masa/Projects/mcp-browser/tests/unit/test_setup.py | Depth: 7 (recommended: <4) |
| 🔴 Deep Nesting | `test_uninstall_unit.py` | /Users/masa/Projects/mcp-browser/tests/unit/test_uninstall_unit.py | Depth: 7 (recommended: <4) |

### Warnings

| Smell | Name | File | Detail |
|-------|------|------|--------|
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background-enhanced.js | 63 lines (recommended: <50) |
| 🟡 Long Method | `connectToServer` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background-enhanced.js | 67 lines (recommended: <50) |
| 🟡 Long Method | `tryConnect` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/background.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `updateStatus` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/popup-enhanced.js | 69 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/.mcp-browser/extension/Readability.js | 59 lines (recommended: <50) |
| 🟡 Deep Nesting | `APPLESCRIPT_IMPLEMENTATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/APPLESCRIPT_IMPLEMENTATION_SUMMARY.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `APPLESCRIPT_QUICK_START.md` | /Users/masa/Projects/mcp-browser/APPLESCRIPT_QUICK_START.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `setup-path.sh` | /Users/masa/Projects/mcp-browser/bin/setup-path.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `BUILD_TRACKING.md` | /Users/masa/Projects/mcp-browser/BUILD_TRACKING.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-browser/CHANGELOG.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `CLAUDE.md` | /Users/masa/Projects/mcp-browser/CLAUDE.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `APPLESCRIPT_FALLBACK.md` | /Users/masa/Projects/mcp-browser/docs/APPLESCRIPT_FALLBACK.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `CDP_CONNECTION.md` | /Users/masa/Projects/mcp-browser/docs/CDP_CONNECTION.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `CHANGELOG.md` | /Users/masa/Projects/mcp-browser/docs/CHANGELOG.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `CLI_FEATURES.md` | /Users/masa/Projects/mcp-browser/docs/CLI_FEATURES.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `CODE_STRUCTURE.md` | /Users/masa/Projects/mcp-browser/docs/CODE_STRUCTURE.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `DEPLOYMENT.md` | /Users/masa/Projects/mcp-browser/docs/DEPLOYMENT.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `DEVELOPER.md` | /Users/masa/Projects/mcp-browser/docs/DEVELOPER.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `INSTALLATION.md` | /Users/masa/Projects/mcp-browser/docs/INSTALLATION.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `MCP_BROWSER_TEST_RESULTS.md` | /Users/masa/Projects/mcp-browser/docs/MCP_BROWSER_TEST_RESULTS.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `MCP_SETUP_COMPLETE.md` | /Users/masa/Projects/mcp-browser/docs/MCP_SETUP_COMPLETE.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `QUICK_REFERENCE.md` | /Users/masa/Projects/mcp-browser/docs/QUICK_REFERENCE.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `QUICKSTART.md` | /Users/masa/Projects/mcp-browser/docs/QUICKSTART.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_CHEATSHEET.md` | /Users/masa/Projects/mcp-browser/docs/RELEASE_CHEATSHEET.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_CHECKLIST.md` | /Users/masa/Projects/mcp-browser/docs/RELEASE_CHECKLIST.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_QUICKSTART.md` | /Users/masa/Projects/mcp-browser/docs/RELEASE_QUICKSTART.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_SCRIPT.md` | /Users/masa/Projects/mcp-browser/docs/RELEASE_SCRIPT.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `SAFARI_EXTENSION.md` | /Users/masa/Projects/mcp-browser/docs/SAFARI_EXTENSION.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `TROUBLESHOOTING.md` | /Users/masa/Projects/mcp-browser/docs/TROUBLESHOOTING.md | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `claude-desktop-config.json` | /Users/masa/Projects/mcp-browser/examples/claude-desktop-config.json | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `config-advanced.json` | /Users/masa/Projects/mcp-browser/examples/config-advanced.json | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `config-minimal.json` | /Users/masa/Projects/mcp-browser/examples/config-minimal.json | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `usage-example.py` | /Users/masa/Projects/mcp-browser/examples/usage-example.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `IMPLEMENTATION_COMPLETE.md` | /Users/masa/Projects/mcp-browser/IMPLEMENTATION_COMPLETE.md | Depth: 5 (recommended: <4) |
| 🟡 Long Method | `User` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 97 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 66 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 52 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 62 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 72 lines (recommended: <50) |
| 🟡 Long Method | `Start` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 67 lines (recommended: <50) |
| 🟡 Long Method | `Operation` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 72 lines (recommended: <50) |
| 🟡 Long Method | `MCPInstaller` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/DIAGRAMS.md | 57 lines (recommended: <50) |
| 🟡 Long Method | `py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/IMPLEMENTATION-PLAN.md | 69 lines (recommended: <50) |
| 🟡 Long Method | `py` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/docs/PROJECT-STRUCTURE.md | 88 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/cli.py | 62 lines (recommended: <50) |
| 🟡 Long Method | `cmd_doctor` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/cli.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `_print_report` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/cli.py | 71 lines (recommended: <50) |
| 🟡 Long Method | `build_command` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 59 lines (recommended: <50) |
| 🟡 Long Method | `detect_best_method` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/command_builder.py | 56 lines (recommended: <50) |
| 🟡 Long Method | `write` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `add_server` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `validate` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `migrate_legacy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/config_manager.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `InstallationStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | 68 lines (recommended: <50) |
| 🟡 High Complexity | `NativeCLIStrategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `__init__` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 62 lines (recommended: <50) |
| 🟡 High Complexity | `install_server` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `run_diagnostics` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `_select_strategy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/installer.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `check_platform` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `check_config` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `_generate_recommendations` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_doctor.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `InspectionReport` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/mcp_inspector.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `detect` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `detect_for_platform` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `detect_claude_code` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `detect_claude_desktop` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/platform_detector.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `validate_json_structure` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/src/py_mcp_installer/utils.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `TestDiagnosticIssue` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `TestServerDiagnostic` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `test_test_server_healthy` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `test_cmd_doctor_with_issues` | /Users/masa/Projects/mcp-browser/lib/py-mcp-installer-service/tests/test_mcp_doctor.py | 54 lines (recommended: <50) |
| 🟡 Deep Nesting | `mcp-server.py` | /Users/masa/Projects/mcp-browser/mcp-server.py | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `mcp.json` | /Users/masa/Projects/mcp-browser/mcp.json | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `README.md` | /Users/masa/Projects/mcp-browser/README.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_AUTOMATION_SUMMARY.md` | /Users/masa/Projects/mcp-browser/RELEASE_AUTOMATION_SUMMARY.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_QUICK_REFERENCE.md` | /Users/masa/Projects/mcp-browser/RELEASE_QUICK_REFERENCE.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE_SCRIPT_SUMMARY.md` | /Users/masa/Projects/mcp-browser/RELEASE_SCRIPT_SUMMARY.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `RELEASE.md` | /Users/masa/Projects/mcp-browser/RELEASE.md | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `requirements-dev.txt` | /Users/masa/Projects/mcp-browser/requirements-dev.txt | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `requirements.txt` | /Users/masa/Projects/mcp-browser/requirements.txt | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `build_extension_simple.py` | /Users/masa/Projects/mcp-browser/scripts/build_extension_simple.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `build_extension` | /Users/masa/Projects/mcp-browser/scripts/build_extension_simple.py | 79 lines (recommended: <50) |
| 🟡 Deep Nesting | `build_extension.py` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `get_extension_files` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | 64 lines (recommended: <50) |
| 🟡 High Complexity | `get_extension_files` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | Complexity: 21 (recommended: <15) |
| 🟡 High Complexity | `main` | /Users/masa/Projects/mcp-browser/scripts/build_extension.py | Complexity: 18 (recommended: <15) |
| 🟡 Deep Nesting | `bump_version.py` | /Users/masa/Projects/mcp-browser/scripts/bump_version.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `update_changelog` | /Users/masa/Projects/mcp-browser/scripts/bump_version.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-browser/scripts/bump_version.py | 83 lines (recommended: <50) |
| 🟡 Deep Nesting | `check_changelog.py` | /Users/masa/Projects/mcp-browser/scripts/check_changelog.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `check_version_consistency.py` | /Users/masa/Projects/mcp-browser/scripts/check_version_consistency.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `check_versions` | /Users/masa/Projects/mcp-browser/scripts/check_version_consistency.py | 73 lines (recommended: <50) |
| 🟡 Deep Nesting | `completion.bash` | /Users/masa/Projects/mcp-browser/scripts/completion.bash | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `completion.zsh` | /Users/masa/Projects/mcp-browser/scripts/completion.zsh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `create-safari-extension.sh` | /Users/masa/Projects/mcp-browser/scripts/create-safari-extension.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `demo.sh` | /Users/masa/Projects/mcp-browser/scripts/demo.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `dev-extension.sh` | /Users/masa/Projects/mcp-browser/scripts/dev-extension.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `dev-full.sh` | /Users/masa/Projects/mcp-browser/scripts/dev-full.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `dev-server.sh` | /Users/masa/Projects/mcp-browser/scripts/dev-server.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `extract_changelog.py` | /Users/masa/Projects/mcp-browser/scripts/extract_changelog.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `generate_build_info.py` | /Users/masa/Projects/mcp-browser/scripts/generate_build_info.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `install.sh` | /Users/masa/Projects/mcp-browser/scripts/install.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `release.py` | /Users/masa/Projects/mcp-browser/scripts/release.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `create_github_release` | /Users/masa/Projects/mcp-browser/scripts/release.py | 63 lines (recommended: <50) |
| 🟡 Deep Nesting | `setup-claude-code.sh` | /Users/masa/Projects/mcp-browser/scripts/setup-claude-code.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `sync_extensions.py` | /Users/masa/Projects/mcp-browser/scripts/sync_extensions.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `test_installation.sh` | /Users/masa/Projects/mcp-browser/scripts/test_installation.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `test_mcp_server.sh` | /Users/masa/Projects/mcp-browser/scripts/test_mcp_server.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `update_homebrew_tap.sh` | /Users/masa/Projects/mcp-browser/scripts/update_homebrew_tap.sh | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `setup.py` | /Users/masa/Projects/mcp-browser/setup.py | Depth: 5 (recommended: <4) |
| 🟡 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/src/__init__.py | Depth: 6 (recommended: <4) |
| 🟡 Deep Nesting | `_version.py` | /Users/masa/Projects/mcp-browser/src/_version.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `_extract_content_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `_extract_semantic_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `_display_semantic_dom` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 68 lines (recommended: <50) |
| 🟡 Long Method | `_extract_selector_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `_run_demo_scenario` | /Users/masa/Projects/mcp-browser/src/cli/commands/browser.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `_demo_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 70 lines (recommended: <50) |
| 🟡 Long Method | `_step_verify_connection` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `_step_navigate` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | 69 lines (recommended: <50) |
| 🟡 High Complexity | `_step_content_extraction` | /Users/masa/Projects/mcp-browser/src/cli/commands/demo.py | Complexity: 24 (recommended: <15) |
| 🟡 Long Method | `_doctor_command` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | 99 lines (recommended: <50) |
| 🟡 Long Method | `_check_browser_extension_connection` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | 52 lines (recommended: <50) |
| 🟡 High Complexity | `_check_console_log_capture` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_check_browser_control` | /Users/masa/Projects/mcp-browser/src/cli/commands/doctor.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `find_extension_source` | /Users/masa/Projects/mcp-browser/src/cli/commands/extension.py | 64 lines (recommended: <50) |
| 🟡 Long Method | `install` | /Users/masa/Projects/mcp-browser/src/cli/commands/extension.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `update` | /Users/masa/Projects/mcp-browser/src/cli/commands/extension.py | 54 lines (recommended: <50) |
| 🟡 Long Method | `init` | /Users/masa/Projects/mcp-browser/src/cli/commands/init.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `install_to_platform` | /Users/masa/Projects/mcp-browser/src/cli/commands/install.py | 56 lines (recommended: <50) |
| 🟡 High Complexity | `uninstall` | /Users/masa/Projects/mcp-browser/src/cli/commands/install.py | Complexity: 16 (recommended: <15) |
| 🟡 High Complexity | `quickstart` | /Users/masa/Projects/mcp-browser/src/cli/commands/quickstart.py | Complexity: 18 (recommended: <15) |
| 🟡 Long Method | `install_extension` | /Users/masa/Projects/mcp-browser/src/cli/commands/setup.py | 68 lines (recommended: <50) |
| 🟡 High Complexity | `install_mcp` | /Users/masa/Projects/mcp-browser/src/cli/commands/setup.py | Complexity: 19 (recommended: <15) |
| 🟡 High Complexity | `start` | /Users/masa/Projects/mcp-browser/src/cli/commands/start.py | Complexity: 20 (recommended: <15) |
| 🟡 Long Method | `status` | /Users/masa/Projects/mcp-browser/src/cli/commands/status.py | 96 lines (recommended: <50) |
| 🟡 Long Method | `stop` | /Users/masa/Projects/mcp-browser/src/cli/commands/stop.py | 72 lines (recommended: <50) |
| 🟡 Long Method | `tutorial` | /Users/masa/Projects/mcp-browser/src/cli/commands/tutorial.py | 73 lines (recommended: <50) |
| 🟡 Long Method | `cli` | /Users/masa/Projects/mcp-browser/src/cli/main.py | 57 lines (recommended: <50) |
| 🟡 Long Method | `reference` | /Users/masa/Projects/mcp-browser/src/cli/main.py | 84 lines (recommended: <50) |
| 🟡 Long Method | `completion` | /Users/masa/Projects/mcp-browser/src/cli/main.py | 93 lines (recommended: <50) |
| 🟡 Long Method | `cleanup_stale_servers` | /Users/masa/Projects/mcp-browser/src/cli/utils/daemon.py | 53 lines (recommended: <50) |
| 🟡 Long Method | `cleanup_unregistered_servers` | /Users/masa/Projects/mcp-browser/src/cli/utils/daemon.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `start_daemon` | /Users/masa/Projects/mcp-browser/src/cli/utils/daemon.py | 85 lines (recommended: <50) |
| 🟡 Long Method | `start` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 75 lines (recommended: <50) |
| 🟡 Long Method | `show_status` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `run_mcp_stdio` | /Users/masa/Projects/mcp-browser/src/cli/utils/server.py | 96 lines (recommended: <50) |
| 🟡 Long Method | `check_system_requirements` | /Users/masa/Projects/mcp-browser/src/cli/utils/validation.py | 63 lines (recommended: <50) |
| 🟡 High Complexity | `ServiceContainer` | /Users/masa/Projects/mcp-browser/src/container/service_container.py | Complexity: 20 (recommended: <15) |
| 🟡 Deep Nesting | `dev_runner.py` | /Users/masa/Projects/mcp-browser/src/dev_runner.py | Depth: 6 (recommended: <4) |
| 🟡 High Complexity | `DevelopmentServer` | /Users/masa/Projects/mcp-browser/src/dev_runner.py | Complexity: 22 (recommended: <15) |
| 🟡 Long Method | `run` | /Users/masa/Projects/mcp-browser/src/dev_runner.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-browser/src/extension/background-enhanced.js | 78 lines (recommended: <50) |
| 🟡 Long Method | `connectToServer` | /Users/masa/Projects/mcp-browser/src/extension/background-enhanced.js | 69 lines (recommended: <50) |
| 🟡 Long Method | `tryConnect` | /Users/masa/Projects/mcp-browser/src/extension/background.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `updateStatus` | /Users/masa/Projects/mcp-browser/src/extension/popup-enhanced.js | 69 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/src/extension/Readability.js | 59 lines (recommended: <50) |
| 🟡 High Complexity | `PortSelector` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `disconnectBackend` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 61 lines (recommended: <50) |
| 🟡 High Complexity | `_setupMessageHandler` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_setupCloseHandler` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 64 lines (recommended: <50) |
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/background-enhanced.js | 87 lines (recommended: <50) |
| 🟡 Long Method | `updateOverallStatus` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 60 lines (recommended: <50) |
| 🟡 Long Method | `updateCurrentTabStatus` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 61 lines (recommended: <50) |
| 🟡 Long Method | `updateBackendList` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 75 lines (recommended: <50) |
| 🟡 Long Method | `handleConnectToBackend` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 90 lines (recommended: <50) |
| 🟡 Long Method | `handleScanBackends` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 84 lines (recommended: <50) |
| 🟡 Long Method | `copyDebugInfo` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/popup-enhanced.js | 51 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/src/extensions/chrome/Readability.js | 59 lines (recommended: <50) |
| 🟡 High Complexity | `PortSelector` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `disconnectBackend` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_setupMessageHandler` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `_setupCloseHandler` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 64 lines (recommended: <50) |
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background-enhanced.js | 87 lines (recommended: <50) |
| 🟡 Long Method | `connectToBackend` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | 73 lines (recommended: <50) |
| 🟡 Long Method | `_setupConnectionHandlers` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | 79 lines (recommended: <50) |
| 🟡 Long Method | `probePort` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/background.js | 54 lines (recommended: <50) |
| 🟡 Long Method | `updateOverallStatus` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 60 lines (recommended: <50) |
| 🟡 Long Method | `updateCurrentTabStatus` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 61 lines (recommended: <50) |
| 🟡 Long Method | `updateBackendList` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 75 lines (recommended: <50) |
| 🟡 Long Method | `handleConnectToBackend` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 90 lines (recommended: <50) |
| 🟡 Long Method | `handleScanBackends` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 84 lines (recommended: <50) |
| 🟡 Long Method | `copyDebugInfo` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/popup-enhanced.js | 51 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/src/extensions/firefox/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `tryConnect` | /Users/masa/Projects/mcp-browser/src/extensions/safari/background.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `updateStatus` | /Users/masa/Projects/mcp-browser/src/extensions/safari/popup.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/src/extensions/safari/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `Readability` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 83 lines (recommended: <50) |
| 🟡 Long Method | `_fixRelativeUris` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 80 lines (recommended: <50) |
| 🟡 Long Method | `_getArticleTitle` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `_replaceBrs` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 55 lines (recommended: <50) |
| 🟡 High Complexity | `_initializeNode` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Complexity: 20 (recommended: <15) |
| 🟡 High Complexity | `_getJSONLD` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Complexity: 19 (recommended: <15) |
| 🟡 Long Method | `_unwrapNoscriptImages` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 77 lines (recommended: <50) |
| 🟡 Long Method | `_markDataTables` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 81 lines (recommended: <50) |
| 🟡 High Complexity | `_fixLazyImages` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `parse` | /Users/masa/Projects/mcp-browser/src/extensions/shared/Readability.js | 59 lines (recommended: <50) |
| 🟡 Long Method | `check_browser_availability` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 94 lines (recommended: <50) |
| 🟡 Long Method | `navigate` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `execute_javascript` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 60 lines (recommended: <50) |
| 🟡 Long Method | `click` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `fill_field` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `get_element` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `_execute_applescript` | /Users/masa/Projects/mcp-browser/src/services/applescript_service.py | 67 lines (recommended: <50) |
| 🟡 Long Method | `get_capability_report` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `execute_action` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 61 lines (recommended: <50) |
| 🟡 Long Method | `navigate` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `click` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `fill_field` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `get_element` | /Users/masa/Projects/mcp-browser/src/services/browser_controller.py | 77 lines (recommended: <50) |
| 🟡 Long Method | `extract_content` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | 63 lines (recommended: <50) |
| 🟡 Long Method | `capture_screenshot_via_extension` | /Users/masa/Projects/mcp-browser/src/services/browser_service.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `_get_status` | /Users/masa/Projects/mcp-browser/src/services/dashboard_service.py | 53 lines (recommended: <50) |
| 🟡 High Complexity | `DOMInteractionService` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | Complexity: 25 (recommended: <15) |
| 🟡 Long Method | `_send_dom_command` | /Users/masa/Projects/mcp-browser/src/services/dom_interaction_service.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `_extract_content` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `_format_semantic_dom` | /Users/masa/Projects/mcp-browser/src/services/mcp_service.py | 66 lines (recommended: <50) |
| 🟡 Long Method | `handle_gap_recovery` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | 54 lines (recommended: <50) |
| 🟡 High Complexity | `_handle_message` | /Users/masa/Projects/mcp-browser/src/services/websocket_service.py | Complexity: 22 (recommended: <15) |
| 🟡 Deep Nesting | `__init__.py` | /Users/masa/Projects/mcp-browser/tests/__init__.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `runPortSelectorTests` | /Users/masa/Projects/mcp-browser/tests/fixtures/test_port_selector.js | 89 lines (recommended: <50) |
| 🟡 Long Method | `detectExtension` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 75 lines (recommended: <50) |
| 🟡 High Complexity | `updateExtensionStatus` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | Complexity: 16 (recommended: <15) |
| 🟡 Long Method | `setupEventListeners` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 95 lines (recommended: <50) |
| 🟡 Long Method | `setupDOMTests` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 79 lines (recommended: <50) |
| 🟡 Long Method | `generateManifest` | /Users/masa/Projects/mcp-browser/tests/fixtures/test-page.js | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestBrowserControllerIntegration` | /Users/masa/Projects/mcp-browser/tests/integration/test_applescript_integration.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `TestErrorHandling` | /Users/masa/Projects/mcp-browser/tests/integration/test_applescript_integration.py | 51 lines (recommended: <50) |
| 🟡 Long Method | `test_browser_control` | /Users/masa/Projects/mcp-browser/tests/integration/test_browser_control.py | 90 lines (recommended: <50) |
| 🟡 Long Method | `test_mcp_server` | /Users/masa/Projects/mcp-browser/tests/integration/test_mcp_tools.py | 90 lines (recommended: <50) |
| 🟡 Long Method | `main` | /Users/masa/Projects/mcp-browser/tests/integration/test_qa_fixes.py | 52 lines (recommended: <50) |
| 🟡 Long Method | `test_completion_scripts` | /Users/masa/Projects/mcp-browser/tests/integration/test_uninstall_integration.py | 58 lines (recommended: <50) |
| 🟡 Long Method | `TestBrowserAvailability` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | 76 lines (recommended: <50) |
| 🟡 Long Method | `TestNavigation` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | 67 lines (recommended: <50) |
| 🟡 Long Method | `TestJavaScriptExecution` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | 69 lines (recommended: <50) |
| 🟡 Long Method | `TestFillField` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | 55 lines (recommended: <50) |
| 🟡 Long Method | `TestAppleScriptExecution` | /Users/masa/Projects/mcp-browser/tests/services/test_applescript_service.py | 71 lines (recommended: <50) |
| 🟡 Deep Nesting | `test_applescript_integration.py` | /Users/masa/Projects/mcp-browser/tests/test_applescript_integration.py | Depth: 6 (recommended: <4) |
| 🟡 Long Method | `TestBrowserControllerIntegration` | /Users/masa/Projects/mcp-browser/tests/test_applescript_integration.py | 65 lines (recommended: <50) |
| 🟡 Long Method | `TestErrorHandling` | /Users/masa/Projects/mcp-browser/tests/test_applescript_integration.py | 51 lines (recommended: <50) |
| 🟡 Deep Nesting | `platforms` | lib/py-mcp-installer-service/src/py_mcp_installer/platforms | Depth: 5 (recommended: <4) |

---

## Recommended Actions

1. **Start with Grade F items** - These have the highest complexity and are hardest to maintain
2. **Address Critical code smells** - God Classes and deeply nested code should be refactored
3. **Break down long methods** - Extract helper functions to reduce complexity
4. **Add tests before refactoring** - Ensure behavior is preserved

---

_Generated by MCP Vector Search Visualization_
