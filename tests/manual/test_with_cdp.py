#!/usr/bin/env python3
"""
Test using Chrome DevTools Protocol for detailed error information.
"""

import asyncio
import json

from playwright.async_api import async_playwright


async def test_with_cdp():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()

        # Enable more detailed error reporting
        cdp = await page.context.new_cdp_session(page)
        await cdp.send("Runtime.enable")
        await cdp.send("Log.enable")

        runtime_exceptions = []

        async def handle_exception(params):
            exception_details = params.get("exceptionDetails", {})
            runtime_exceptions.append(
                {
                    "text": exception_details.get("text", "Unknown error"),
                    "line": exception_details.get("lineNumber", "unknown"),
                    "column": exception_details.get("columnNumber", "unknown"),
                    "url": exception_details.get("url", "unknown"),
                    "exception": str(exception_details.get("exception", {})),
                }
            )
            print("\n🐛 Runtime Exception Caught:")
            print(f"   Text: {exception_details.get('text', 'Unknown')}")
            print(f"   Line: {exception_details.get('lineNumber', 'unknown')}")
            print(f"   Column: {exception_details.get('columnNumber', 'unknown')}")
            print(f"   URL: {exception_details.get('url', 'unknown')}")
            if "exception" in exception_details:
                exc = exception_details["exception"]
                print(f"   Exception: {exc.get('description', 'No description')}")

        cdp.on("Runtime.exceptionThrown", handle_exception)

        print("Navigating to visualizer...")
        try:
            response = await page.goto(
                "http://localhost:8095", wait_until="networkidle", timeout=15000
            )
            print(f"✓ Page loaded with status: {response.status}")

            await asyncio.sleep(3)

            # Check visualization state
            result = await page.evaluate(
                """
                () => {
                    return {
                        svgCount: document.querySelectorAll('svg').length,
                        nodeCount: document.querySelectorAll('.node').length,
                        scriptTags: document.querySelectorAll('script').length,
                        hasD3: typeof d3 !== 'undefined',
                        bodyText: document.body.textContent.substring(0, 500)
                    };
                }
            """
            )

            print("\n📊 Page State:")
            print(f"   SVG elements: {result['svgCount']}")
            print(f"   Node elements: {result['nodeCount']}")
            print(f"   Script tags: {result['scriptTags']}")
            print(f"   D3.js loaded: {result['hasD3']}")

            if runtime_exceptions:
                print(f"\n❌ Total Exceptions: {len(runtime_exceptions)}")
                with open("/tmp/exceptions.json", "w") as f:
                    json.dump(runtime_exceptions, f, indent=2)
                print("   Saved to: /tmp/exceptions.json")

        except Exception as e:
            print(f"❌ Error: {e}")

        await browser.close()


asyncio.run(test_with_cdp())
