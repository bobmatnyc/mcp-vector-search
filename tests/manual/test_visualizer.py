#!/usr/bin/env python3
"""
End-to-end browser test for visualizer.
Tests rendering, console logs, and performance.
"""

import asyncio
import json
import time

from playwright.async_api import async_playwright


async def test_visualizer():
    results = {
        "server_running": False,
        "page_loaded": False,
        "load_time_ms": 0,
        "console_errors": [],
        "console_warnings": [],
        "console_logs": [],
        "svg_elements": 0,
        "node_elements": 0,
        "link_elements": 0,
        "page_title": "",
        "screenshot_saved": False,
        "errors": [],
    }

    async with async_playwright() as p:
        try:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            page = await context.new_page()

            # Capture console messages
            def handle_console(msg):
                msg_type = msg.type
                text = msg.text
                if msg_type == "error":
                    results["console_errors"].append(text)
                elif msg_type == "warning":
                    results["console_warnings"].append(text)
                else:
                    results["console_logs"].append(text)

            page.on("console", handle_console)

            # Capture page errors
            def handle_page_error(error):
                results["errors"].append(str(error))

            page.on("pageerror", handle_page_error)

            # Navigate to visualizer with timeout
            start_time = time.time()
            try:
                response = await page.goto(
                    "http://localhost:8095",
                    wait_until="networkidle",
                    timeout=15000,  # 15 seconds
                )
                results["server_running"] = True
                results["page_loaded"] = response.ok
            except Exception as e:
                results["errors"].append(f"Navigation failed: {str(e)}")
                return results

            # Wait for potential rendering
            await asyncio.sleep(2)

            end_time = time.time()
            results["load_time_ms"] = int((end_time - start_time) * 1000)

            # Get page title
            results["page_title"] = await page.title()

            # Check for SVG elements (the visualization uses D3.js with SVG)
            svg_count = await page.evaluate(
                """
                () => document.querySelectorAll('svg').length
            """
            )
            results["svg_elements"] = svg_count

            # Check for node elements (circles or groups)
            node_count = await page.evaluate(
                """
                () => document.querySelectorAll('.node, circle.node, g.node').length
            """
            )
            results["node_elements"] = node_count

            # Check for link elements (lines or paths)
            link_count = await page.evaluate(
                """
                () => document.querySelectorAll('.link, line.link, path.link').length
            """
            )
            results["link_elements"] = link_count

            # Take screenshot
            await page.screenshot(path="/tmp/visualizer_screenshot.png", full_page=True)
            results["screenshot_saved"] = True

            # Get computed styles to check if elements are visible
            svg_visibility = await page.evaluate(
                """
                () => {
                    const svg = document.querySelector('svg');
                    if (!svg) return null;
                    const style = window.getComputedStyle(svg);
                    return {
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        width: style.width,
                        height: style.height
                    };
                }
            """
            )
            results["svg_visibility"] = svg_visibility

            await browser.close()

        except Exception as e:
            results["errors"].append(f"Test failed: {str(e)}")

    return results


async def main():
    print("Starting visualizer browser test...")
    print("=" * 70)

    results = await test_visualizer()

    # Print results
    print("\n📊 TEST RESULTS")
    print("=" * 70)
    print(f"✓ Server Running: {results['server_running']}")
    print(f"✓ Page Loaded: {results['page_loaded']}")
    print(f"✓ Load Time: {results['load_time_ms']}ms")
    print(f"✓ Page Title: {results['page_title']}")
    print(f"✓ Screenshot Saved: {results['screenshot_saved']}")

    print("\n🎨 VISUALIZATION ELEMENTS")
    print("=" * 70)
    print(f"SVG Elements: {results['svg_elements']}")
    print(f"Node Elements: {results['node_elements']}")
    print(f"Link Elements: {results['link_elements']}")

    if results.get("svg_visibility"):
        print("\n👁️  SVG VISIBILITY")
        print("=" * 70)
        for key, value in results["svg_visibility"].items():
            print(f"{key}: {value}")

    print("\n🖥️  CONSOLE OUTPUT")
    print("=" * 70)
    print(f"Errors: {len(results['console_errors'])}")
    if results["console_errors"]:
        for error in results["console_errors"]:
            print(f"  ❌ {error}")

    print(f"\nWarnings: {len(results['console_warnings'])}")
    if results["console_warnings"]:
        for warning in results["console_warnings"][:5]:  # Limit output
            print(f"  ⚠️  {warning}")

    print(f"\nLogs: {len(results['console_logs'])}")
    if results["console_logs"]:
        for log in results["console_logs"][:10]:  # Limit output
            print(f"  ℹ️  {log}")

    if results["errors"]:
        print("\n❌ ERRORS")
        print("=" * 70)
        for error in results["errors"]:
            print(f"  {error}")

    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("=" * 70)

    success = True
    issues = []

    if not results["server_running"]:
        success = False
        issues.append("Server not running")

    if not results["page_loaded"]:
        success = False
        issues.append("Page failed to load")

    if results["svg_elements"] == 0:
        success = False
        issues.append("No SVG elements found (black screen likely)")

    if results["node_elements"] == 0:
        success = False
        issues.append("No node elements rendered")

    if len(results["console_errors"]) > 0:
        success = False
        issues.append(f"{len(results['console_errors'])} console errors")

    if results["load_time_ms"] > 10000:
        issues.append(f"Slow load time ({results['load_time_ms']}ms)")

    if success:
        print("✅ ALL TESTS PASSED - Visualizer is working correctly!")
    else:
        print("❌ TESTS FAILED")
        for issue in issues:
            print(f"  - {issue}")

    # Save detailed results to JSON
    with open("/tmp/visualizer_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n📄 Detailed results saved to: /tmp/visualizer_test_results.json")
    print("📸 Screenshot saved to: /tmp/visualizer_screenshot.png")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
