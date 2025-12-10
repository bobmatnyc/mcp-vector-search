#!/usr/bin/env python3
"""
Playwright test to verify breadcrumb root display bug fix.
Tests that breadcrumbs correctly show "🏠 Root" as the project root,
not showing subdirectories like "tests/manual" as root level.
"""

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


class BreadcrumbTestResults:
    """Store test results for reporting."""

    def __init__(self):
        self.initial_breadcrumb = None
        self.node_navigation_results = []
        self.root_button_works = None
        self.console_errors = []
        self.screenshots = []
        self.passed = False

    def to_dict(self):
        return {
            "initial_breadcrumb": self.initial_breadcrumb,
            "node_navigation_results": self.node_navigation_results,
            "root_button_works": self.root_button_works,
            "console_errors": self.console_errors,
            "screenshots": self.screenshots,
            "passed": self.passed,
        }


async def capture_console_messages(page: Page, results: BreadcrumbTestResults):
    """Capture console messages, especially errors."""

    def handle_console(msg):
        if msg.type in ["error", "warning"]:
            error_info = {"type": msg.type, "text": msg.text, "location": msg.location}
            results.console_errors.append(error_info)
            print(f"  [CONSOLE {msg.type.upper()}] {msg.text}")

    page.on("console", handle_console)

    def handle_page_error(error):
        error_info = {"type": "page_error", "text": str(error), "location": None}
        results.console_errors.append(error_info)
        print(f"  [PAGE ERROR] {error}")

    page.on("pageerror", handle_page_error)


async def get_breadcrumb_text(page: Page) -> str:
    """Extract the full breadcrumb text from the page."""
    try:
        # Wait for breadcrumb to be visible
        await page.wait_for_selector(".breadcrumb-nav, .breadcrumb-root", timeout=5000)

        # Try multiple selectors to find breadcrumb
        breadcrumb_selectors = [
            ".breadcrumb-nav",
            ".breadcrumb-root",
            "[class*='breadcrumb']",
            "#breadcrumb",
        ]

        for selector in breadcrumb_selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                # Get text from first matching element
                text = await elements[0].inner_text()
                if text:
                    return text.strip()

        # Fallback: try to find any element containing "Root"
        root_elements = await page.query_selector_all("text=/🏠.*Root/")
        if root_elements:
            text = await root_elements[0].inner_text()
            return text.strip()

        return "ERROR: No breadcrumb found"

    except Exception as e:
        return f"ERROR: {str(e)}"


async def get_breadcrumb_parts(page: Page) -> list:
    """Extract breadcrumb parts as a list."""
    text = await get_breadcrumb_text(page)
    if text.startswith("ERROR:"):
        return [text]

    # Split by common separators
    parts = text.replace(" / ", "/").replace(" > ", "/").split("/")
    return [part.strip() for part in parts if part.strip()]


async def click_root_button(page: Page) -> bool:
    """Click the Root breadcrumb button and verify it works."""
    try:
        # Try to click on root button
        root_selectors = [
            "text=/🏠.*Root/",
            ".breadcrumb-root",
            ".breadcrumb-nav >> text=/Root/",
        ]

        for selector in root_selectors:
            try:
                await page.click(selector, timeout=2000)
                await page.wait_for_timeout(500)  # Wait for animation
                return True
            except:
                continue

        return False

    except Exception as e:
        print(f"  [ERROR] Failed to click root button: {e}")
        return False


async def get_visible_nodes(page: Page) -> list:
    """Get list of visible node labels in the graph."""
    try:
        # Wait for SVG to be rendered
        await page.wait_for_selector("svg", timeout=5000)

        # Get all text elements (node labels)
        node_labels = await page.evaluate(
            """
            () => {
                const texts = Array.from(document.querySelectorAll('svg text'));
                return texts
                    .map(t => t.textContent.trim())
                    .filter(t => t.length > 0)
                    .slice(0, 10);  // Limit to first 10 nodes
            }
        """
        )

        return node_labels

    except Exception as e:
        print(f"  [ERROR] Failed to get nodes: {e}")
        return []


async def click_node_by_label(page: Page, label: str) -> bool:
    """Click a node by its label text."""
    try:
        # Try to click the node
        await page.evaluate(
            f"""
            () => {{
                const texts = Array.from(document.querySelectorAll('svg text'));
                const node = texts.find(t => t.textContent.includes('{label}'));
                if (node) {{
                    node.dispatchEvent(new MouseEvent('click', {{ bubbles: true }}));
                    return true;
                }}
                return false;
            }}
        """
        )

        await page.wait_for_timeout(500)  # Wait for breadcrumb update
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to click node '{label}': {e}")
        return False


async def run_breadcrumb_test(
    url: str = "http://localhost:8080",
) -> BreadcrumbTestResults:
    """Run comprehensive breadcrumb test."""

    results = BreadcrumbTestResults()
    screenshot_dir = Path(
        "/Users/masa/Projects/mcp-vector-search/tests/manual/screenshots"
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("🧪 BREADCRUMB ROOT DISPLAY BUG FIX - VERIFICATION TEST")
    print(f"{'=' * 70}\n")

    async with async_playwright() as p:
        # Launch browser
        print("🚀 Launching browser...")
        browser: Browser = await p.chromium.launch(headless=True)
        context: BrowserContext = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page: Page = await context.new_page()

        # Set up console monitoring
        await capture_console_messages(page, results)

        try:
            # Test 1: Initial page load and breadcrumb check
            print("\n📋 TEST 1: Initial Page Load")
            print(f"  → Navigating to {url}")

            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # Wait for visualization to render

            # Capture initial state
            initial_breadcrumb = await get_breadcrumb_text(page)
            results.initial_breadcrumb = initial_breadcrumb

            print("  ✓ Page loaded successfully")
            print(f"  📍 Initial breadcrumb: '{initial_breadcrumb}'")

            # Take screenshot
            screenshot_path = screenshot_dir / "01_initial_load.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            results.screenshots.append(str(screenshot_path))
            print(f"  📸 Screenshot saved: {screenshot_path.name}")

            # Check for the bug: should NOT show subdirectory as root
            if (
                "tests / manual" in initial_breadcrumb
                or "tests/manual" in initial_breadcrumb
            ):
                print("  ❌ BUG DETECTED: Breadcrumb shows subdirectory as root!")
                results.passed = False
            elif "🏠 Root" in initial_breadcrumb or "Root" in initial_breadcrumb:
                print("  ✅ Breadcrumb correctly shows Root")
            else:
                print("  ⚠️  WARNING: Unexpected breadcrumb format")

            # Test 2: Node navigation
            print("\n📋 TEST 2: Node Navigation and Breadcrumb Updates")

            # Get visible nodes
            nodes = await get_visible_nodes(page)
            print(f"  → Found {len(nodes)} visible nodes")

            if nodes:
                # Test clicking first few nodes
                for i, node_label in enumerate(nodes[:3], 1):
                    print(f"\n  → Test {i}: Clicking node '{node_label[:50]}...'")

                    clicked = await click_node_by_label(page, node_label[:30])
                    if clicked:
                        await page.wait_for_timeout(1000)  # Wait for breadcrumb update

                        breadcrumb = await get_breadcrumb_text(page)
                        breadcrumb_parts = await get_breadcrumb_parts(page)

                        result = {
                            "node_label": node_label,
                            "breadcrumb": breadcrumb,
                            "breadcrumb_parts": breadcrumb_parts,
                        }
                        results.node_navigation_results.append(result)

                        print(f"    📍 Breadcrumb: '{breadcrumb}'")
                        print(f"    🔗 Parts: {breadcrumb_parts}")

                        # Take screenshot
                        screenshot_path = screenshot_dir / f"02_node_click_{i}.png"
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                        results.screenshots.append(str(screenshot_path))
                        print(f"    📸 Screenshot: {screenshot_path.name}")

                        # Check if Root is still the first element
                        if breadcrumb_parts and (
                            "Root" in breadcrumb_parts[0] or "🏠" in breadcrumb_parts[0]
                        ):
                            print("    ✅ Root still at beginning of breadcrumb")
                        else:
                            print("    ❌ Root not at beginning of breadcrumb!")
                    else:
                        print("    ⚠️  Failed to click node")
            else:
                print("  ⚠️  No nodes found in visualization")

            # Test 3: Root button functionality
            print("\n📋 TEST 3: Root Button Functionality")
            print("  → Attempting to click 'Root' button")

            root_clicked = await click_root_button(page)
            results.root_button_works = root_clicked

            if root_clicked:
                await page.wait_for_timeout(1000)
                breadcrumb = await get_breadcrumb_text(page)
                print("  ✓ Root button clicked successfully")
                print(f"  📍 Breadcrumb after click: '{breadcrumb}'")

                # Take screenshot
                screenshot_path = screenshot_dir / "03_root_button_click.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                results.screenshots.append(str(screenshot_path))
                print(f"  📸 Screenshot: {screenshot_path.name}")
            else:
                print("  ⚠️  Could not click root button")

            # Final verdict
            print(f"\n{'=' * 70}")
            print("📊 TEST RESULTS SUMMARY")
            print(f"{'=' * 70}\n")

            # Determine pass/fail
            passed = True

            # Check 1: Initial breadcrumb should show Root correctly
            if (
                "tests / manual" in results.initial_breadcrumb
                or "tests/manual" in results.initial_breadcrumb
            ):
                print("  ❌ FAIL: Initial breadcrumb shows subdirectory as root")
                passed = False
            elif "Root" in results.initial_breadcrumb:
                print("  ✅ PASS: Initial breadcrumb shows Root correctly")
            else:
                print("  ⚠️  WARNING: Initial breadcrumb format unexpected")
                passed = False

            # Check 2: Node navigation should maintain Root at start
            if results.node_navigation_results:
                all_have_root = all(
                    any(
                        "Root" in part or "🏠" in part
                        for part in r["breadcrumb_parts"][:1]
                    )
                    for r in results.node_navigation_results
                )

                if all_have_root:
                    print("  ✅ PASS: All breadcrumbs maintain Root at beginning")
                else:
                    print("  ❌ FAIL: Some breadcrumbs missing Root at beginning")
                    passed = False
            else:
                print("  ⚠️  WARNING: No node navigation tests performed")

            # Check 3: Console errors
            if results.console_errors:
                print(
                    f"  ⚠️  WARNING: {len(results.console_errors)} console errors/warnings detected"
                )
                passed = False
            else:
                print("  ✅ PASS: No console errors detected")

            results.passed = passed

            print(f"\n{'=' * 70}")
            if passed:
                print("🎉 FINAL VERDICT: ✅ PASS - Breadcrumb bug is FIXED")
            else:
                print(
                    "⚠️  FINAL VERDICT: ❌ FAIL - Breadcrumb bug still present or new issues"
                )
            print(f"{'=' * 70}\n")

        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            results.passed = False

        finally:
            await browser.close()

    return results


async def main():
    """Main test runner."""
    results = await run_breadcrumb_test()

    # Save results to JSON
    results_path = Path(
        "/Users/masa/Projects/mcp-vector-search/tests/manual/breadcrumb_test_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"💾 Detailed results saved to: {results_path}")
    print("📸 Screenshots saved to: tests/manual/screenshots/")

    # Exit with appropriate code
    sys.exit(0 if results.passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
