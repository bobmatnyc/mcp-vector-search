#!/usr/bin/env python3
"""
Detailed browser test to capture exact JavaScript errors.
"""

import asyncio

from playwright.async_api import async_playwright


async def test_visualizer_detailed():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        console_messages = []
        page_errors = []

        # Capture all console messages with full details
        def handle_console(msg):
            console_messages.append(
                {
                    "type": msg.type,
                    "text": msg.text,
                    "location": msg.location,
                    "args": [str(arg) for arg in msg.args],
                }
            )

        # Capture page errors with stack traces
        def handle_error(error):
            page_errors.append(
                {
                    "message": str(error),
                    "name": error.name if hasattr(error, "name") else "Error",
                    "stack": error.stack if hasattr(error, "stack") else "",
                }
            )

        page.on("console", handle_console)
        page.on("pageerror", handle_error)

        # Navigate
        try:
            await page.goto(
                "http://localhost:8095", wait_until="networkidle", timeout=15000
            )
            await asyncio.sleep(3)

            # Check if data was loaded
            graph_data_loaded = await page.evaluate(
                """
                () => {
                    return {
                        hasGraphData: typeof window.currentGraphData !== 'undefined',
                        dataLoaded: window.currentGraphData ? true : false
                    };
                }
            """
            )

            # Get network requests

            # Try to manually fetch the JSON to see the response
            json_response = await page.evaluate(
                """
                async () => {
                    try {
                        const response = await fetch('chunk-graph.json');
                        const text = await response.text();
                        return {
                            ok: response.ok,
                            status: response.status,
                            textLength: text.length,
                            firstChars: text.substring(0, 200),
                            isValidJSON: (() => {
                                try {
                                    JSON.parse(text);
                                    return true;
                                } catch (e) {
                                    return false;
                                }
                            })()
                        };
                    } catch (e) {
                        return { error: e.toString() };
                    }
                }
            """
            )

            print("=" * 70)
            print("CONSOLE MESSAGES:")
            print("=" * 70)
            for msg in console_messages:
                print(f"\n[{msg['type'].upper()}] {msg['text']}")
                if msg["location"]:
                    print(f"  Location: {msg['location']}")

            print("\n" + "=" * 70)
            print("PAGE ERRORS:")
            print("=" * 70)
            for error in page_errors:
                print(f"\n{error['name']}: {error['message']}")
                if error["stack"]:
                    print(f"Stack: {error['stack']}")

            print("\n" + "=" * 70)
            print("JSON FETCH TEST:")
            print("=" * 70)
            print(f"Response: {json_response}")

            print("\n" + "=" * 70)
            print("GRAPH DATA STATUS:")
            print("=" * 70)
            print(f"Graph data loaded: {graph_data_loaded}")

        except Exception as e:
            print(f"Test failed: {e}")

        await browser.close()


asyncio.run(test_visualizer_detailed())
