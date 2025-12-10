#!/usr/bin/env python3
"""Test script to verify the visualization is working correctly."""

import sys

from playwright.sync_api import sync_playwright


def test_visualization():
    """Test the visualization page for correct rendering."""
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Capture console messages
        console_messages = []
        page.on(
            "console",
            lambda msg: console_messages.append({"type": msg.type, "text": msg.text}),
        )

        # Capture errors
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        try:
            # Navigate to the visualization
            print("Navigating to http://localhost:8080...")
            page.goto("http://localhost:8080", wait_until="networkidle", timeout=10000)

            # Wait a bit for D3.js to render
            page.wait_for_timeout(2000)

            # Take screenshot
            screenshot_path = (
                "/Users/masa/Projects/mcp-vector-search/visualization_test.png"
            )
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"✅ Screenshot saved to: {screenshot_path}")

            # Check for SVG elements
            svg_count = page.locator("svg").count()
            print(f"\n📊 SVG elements found: {svg_count}")

            if svg_count > 0:
                # Check for nodes
                node_count = page.locator("circle, rect, ellipse").count()
                print(f"📍 Graph nodes found: {node_count}")

                # Check for text labels
                text_count = page.locator("svg text").count()
                print(f"🏷️  Text labels found: {text_count}")

                # Get SVG dimensions to verify it's not empty
                svg = page.locator("svg").first
                if svg:
                    bbox = svg.bounding_box()
                    if bbox:
                        print(f"📐 SVG dimensions: {bbox['width']}x{bbox['height']}")

            # Print console messages
            print("\n🖥️  Console Messages:")
            if console_messages:
                for msg in console_messages:
                    icon = (
                        "❌"
                        if msg["type"] == "error"
                        else "⚠️"
                        if msg["type"] == "warning"
                        else "ℹ️"
                    )
                    print(f"  {icon} [{msg['type']}] {msg['text']}")
            else:
                print("  ✅ No console messages (clean)")

            # Print errors
            print("\n🐛 JavaScript Errors:")
            if errors:
                for err in errors:
                    print(f"  ❌ {err}")
            else:
                print("  ✅ No JavaScript errors")

            # Verify page title
            title = page.title()
            print(f"\n📄 Page Title: {title}")

            # Check if the page has content
            body_text = page.locator("body").inner_text()
            print(f"\n📝 Page has content: {len(body_text) > 0}")

            # Final verdict
            print("\n" + "=" * 60)
            if errors:
                print("❌ FAILED: JavaScript errors detected")
                return False
            elif svg_count == 0:
                print("❌ FAILED: No SVG elements found")
                return False
            elif node_count == 0:
                print("⚠️  WARNING: SVG present but no graph nodes found")
                return False
            else:
                print("✅ SUCCESS: Visualization is rendering correctly!")
                print(f"   - {svg_count} SVG element(s)")
                print(f"   - {node_count} graph node(s)")
                print(f"   - {text_count} label(s)")
                print("   - No JavaScript errors")
                return True

        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False
        finally:
            browser.close()


if __name__ == "__main__":
    success = test_visualization()
    sys.exit(0 if success else 1)
