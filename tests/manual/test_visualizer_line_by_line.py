#!/usr/bin/env python3
"""
Test to get exact line number of JavaScript error.
"""

import asyncio

from playwright.async_api import async_playwright


async def test_with_debugger():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False
        )  # Non-headless for better error info
        context = await browser.new_context()
        page = await context.new_page()

        errors_with_location = []

        # More detailed error handler
        async def handle_console(msg):
            if msg.type == "error":
                try:
                    location = msg.location
                    print("\n❌ Console Error:")
                    print(f"   Text: {msg.text}")
                    print(f"   URL: {location.get('url', 'unknown')}")
                    print(f"   Line: {location.get('lineNumber', 'unknown')}")
                    print(f"   Column: {location.get('columnNumber', 'unknown')}")
                    errors_with_location.append(msg)
                except Exception as e:
                    print(f"Error handling console message: {e}")

        async def handle_page_error(error):
            print("\n❌ Page Error:")
            print(f"   Message: {error}")

        page.on("console", handle_console)
        page.on("pageerror", handle_page_error)

        print("Navigating to http://localhost:8095...")
        try:
            await page.goto(
                "http://localhost:8095", wait_until="domcontentloaded", timeout=10000
            )
            print("Page loaded, waiting for JS execution...")
            await asyncio.sleep(5)

            # Check if visualization rendered
            svg_count = await page.evaluate("document.querySelectorAll('svg').length")
            node_count = await page.evaluate(
                "document.querySelectorAll('.node').length"
            )

            print("\n📊 Results:")
            print(f"   SVG elements: {svg_count}")
            print(f"   Node elements: {node_count}")

        except Exception as e:
            print(f"Navigation error: {e}")

        print("\nPress Ctrl+C to close browser...")
        await asyncio.sleep(30)  # Keep browser open for manual inspection

        await browser.close()


asyncio.run(test_with_debugger())
