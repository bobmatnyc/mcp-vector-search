#!/usr/bin/env python3
"""
Test script for visualization at http://localhost:8090
Uses Playwright to verify visualization functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


async def test_visualization():
    """Test the visualization and capture evidence."""

    # Setup paths
    screenshots_dir = Path("test-screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "url": "http://localhost:8090",
        "initial_node_count": 0,
        "expanded_node_count": 0,
        "console_messages": [],
        "network_requests": [],
        "errors": [],
        "observations": [],
    }

    async with async_playwright() as p:
        # Launch browser with console logging
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        # Capture console messages
        page.on(
            "console",
            lambda msg: results["console_messages"].append(
                {"type": msg.type, "text": msg.text}
            ),
        )

        # Capture console errors
        page.on("pageerror", lambda err: results["errors"].append(str(err)))

        # Capture network requests
        page.on(
            "request",
            lambda req: results["network_requests"].append(
                {
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                }
            ),
        )

        print("🌐 Loading http://localhost:8090...")
        try:
            # Navigate to the page
            response = await page.goto(
                "http://localhost:8090", wait_until="networkidle", timeout=30000
            )
            print(f"✅ Page loaded with status: {response.status}")
            results["observations"].append(
                f"Page loaded successfully with status {response.status}"
            )

        except Exception as e:
            print(f"❌ Failed to load page: {e}")
            results["errors"].append(f"Page load failed: {str(e)}")
            await browser.close()
            return results

        # Wait for the graph to render
        print("⏳ Waiting for graph to render...")
        try:
            await page.wait_for_selector("svg", timeout=10000)
            await asyncio.sleep(2)  # Additional wait for D3 animations
            print("✅ SVG graph found")
        except Exception as e:
            print(f"❌ SVG not found: {e}")
            results["errors"].append(f"SVG not found: {str(e)}")

        # Take initial screenshot
        initial_screenshot = screenshots_dir / f"initial_{timestamp}.png"
        await page.screenshot(path=str(initial_screenshot), full_page=True)
        print(f"📸 Initial screenshot saved: {initial_screenshot}")
        results["screenshots"] = [str(initial_screenshot)]

        # Count initial nodes
        try:
            node_count = await page.evaluate(
                """
                () => document.querySelectorAll('circle').length
            """
            )
            results["initial_node_count"] = node_count
            print(f"📊 Initial node count: {node_count}")
            results["observations"].append(f"Initial node count: {node_count}")
        except Exception as e:
            print(f"❌ Failed to count nodes: {e}")
            results["errors"].append(f"Node count failed: {str(e)}")

        # Get node labels
        try:
            node_labels = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('text')).map(t => t.textContent)
            """
            )
            print(f"📝 Visible node labels: {node_labels[:20]}")  # First 20
            results["initial_node_labels"] = node_labels
            results["observations"].append(f"Found {len(node_labels)} node labels")
        except Exception as e:
            print(f"❌ Failed to get node labels: {e}")
            results["errors"].append(f"Node label extraction failed: {str(e)}")

        # Check if chunk-graph.json loaded
        chunk_graph_loaded = any(
            "chunk-graph.json" in req["url"] for req in results["network_requests"]
        )
        if chunk_graph_loaded:
            print("✅ chunk-graph.json was loaded")
            results["observations"].append("chunk-graph.json loaded successfully")
        else:
            print("⚠️  chunk-graph.json was NOT loaded")
            results["observations"].append("WARNING: chunk-graph.json NOT loaded")

        # Try to find and click the "scripts" directory node
        print("\n🖱️  Attempting to click 'scripts' directory node...")
        try:
            # Method 1: Try to find by text content and dispatch click event
            scripts_clicked = await page.evaluate(
                """
                () => {
                    const textElements = Array.from(document.querySelectorAll('text'));
                    const scriptsText = textElements.find(t => t.textContent === 'scripts');
                    if (scriptsText) {
                        // Find the associated circle (usually in the same group)
                        const group = scriptsText.parentElement;
                        const circle = group.querySelector('circle');
                        if (circle) {
                            // Dispatch a click event
                            const event = new MouseEvent('click', {
                                bubbles: true,
                                cancelable: true,
                                view: window
                            });
                            circle.dispatchEvent(event);
                            return true;
                        }
                    }
                    return false;
                }
            """
            )

            if scripts_clicked:
                print("✅ Clicked 'scripts' node via text search")
                results["observations"].append("Successfully clicked 'scripts' node")

                # Wait for animation
                await asyncio.sleep(2)

                # Take screenshot after click
                after_click_screenshot = (
                    screenshots_dir / f"after_click_{timestamp}.png"
                )
                await page.screenshot(path=str(after_click_screenshot), full_page=True)
                print(f"📸 After-click screenshot saved: {after_click_screenshot}")
                results["screenshots"].append(str(after_click_screenshot))

                # Count nodes after expansion
                expanded_node_count = await page.evaluate(
                    """
                    () => document.querySelectorAll('circle').length
                """
                )
                results["expanded_node_count"] = expanded_node_count
                print(f"📊 Node count after expansion: {expanded_node_count}")
                print(
                    f"➕ New nodes added: {expanded_node_count - results['initial_node_count']}"
                )
                results["observations"].append(
                    f"Expanded from {results['initial_node_count']} to {expanded_node_count} nodes "
                    f"(+{expanded_node_count - results['initial_node_count']} nodes)"
                )

                # Get new node labels
                expanded_labels = await page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll('text')).map(t => t.textContent)
                """
                )
                new_labels = set(expanded_labels) - set(node_labels)
                print(f"📝 New visible labels: {list(new_labels)[:20]}")  # First 20
                results["new_node_labels"] = list(new_labels)

                # Check if new nodes are files or directories
                file_count = sum(1 for label in new_labels if "." in label)
                dir_count = len(new_labels) - file_count
                print(f"📁 New directories: {dir_count}, 📄 New files: {file_count}")
                results["observations"].append(
                    f"Expansion revealed {dir_count} directories and {file_count} files"
                )

            else:
                print("⚠️  Could not find/click 'scripts' node via text search")
                results["observations"].append(
                    "WARNING: Could not locate 'scripts' node"
                )

                # Try alternative method: click first directory-looking node
                print("🔍 Attempting to click first circle element...")
                alternative_clicked = await page.evaluate(
                    """
                    () => {
                        const circles = document.querySelectorAll('circle');
                        if (circles.length > 0) {
                            const event = new MouseEvent('click', {
                                bubbles: true,
                                cancelable: true,
                                view: window
                            });
                            circles[0].dispatchEvent(event);
                            return true;
                        }
                        return false;
                    }
                """
                )

                if alternative_clicked:
                    print("✅ Clicked first circle node as fallback")
                    await asyncio.sleep(2)

                    fallback_screenshot = (
                        screenshots_dir / f"fallback_click_{timestamp}.png"
                    )
                    await page.screenshot(path=str(fallback_screenshot), full_page=True)
                    print(f"📸 Fallback screenshot saved: {fallback_screenshot}")
                    results["screenshots"].append(str(fallback_screenshot))

                    expanded_node_count = await page.evaluate(
                        """
                        () => document.querySelectorAll('circle').length
                    """
                    )
                    results["expanded_node_count"] = expanded_node_count
                    results["observations"].append(
                        f"Fallback click: expanded to {expanded_node_count} nodes"
                    )

        except Exception as e:
            print(f"❌ Click interaction failed: {e}")
            results["errors"].append(f"Click failed: {str(e)}")

        # Analyze console messages
        print("\n📋 Console Messages:")
        for msg in results["console_messages"]:
            print(f"  [{msg['type']}] {msg['text']}")

        # Check for JavaScript errors
        if results["errors"]:
            print("\n❌ JavaScript Errors:")
            for error in results["errors"]:
                print(f"  {error}")
        else:
            print("\n✅ No JavaScript errors detected")

        # Wait a bit for final inspection
        print("\n⏳ Waiting 3 seconds before closing browser...")
        await asyncio.sleep(3)

        await browser.close()

    # Save results to JSON
    results_file = screenshots_dir / f"test_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Test results saved: {results_file}")

    return results


def print_summary(results):
    """Print a summary of the test results."""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"URL: {results['url']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"\nInitial nodes: {results['initial_node_count']}")
    print(f"Expanded nodes: {results['expanded_node_count']}")
    print(
        f"New nodes added: {results['expanded_node_count'] - results['initial_node_count']}"
    )

    print(f"\n📸 Screenshots: {len(results.get('screenshots', []))}")
    for screenshot in results.get("screenshots", []):
        print(f"  - {screenshot}")

    print(f"\n📝 Observations: {len(results['observations'])}")
    for obs in results["observations"]:
        print(f"  ✓ {obs}")

    if results["errors"]:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  ! {error}")
    else:
        print("\n✅ No errors detected")

    print("\n" + "=" * 80)


async def main():
    """Main entry point."""
    print("🧪 Starting Visualization Test")
    print("=" * 80)

    # Check if Playwright is installed
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("❌ Playwright not installed!")
        print("Run: pip install playwright && playwright install chromium")
        return

    results = await test_visualization()
    print_summary(results)

    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
