#!/usr/bin/env python3
"""
Investigation script for visualization issue - Why are nodes not showing?
"""

import asyncio
import json

from playwright.async_api import async_playwright


async def investigate():
    async with async_playwright() as p:
        # Launch browser with more visibility
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        # Enable console logging
        console_logs = []
        errors = []

        def log_console(msg):
            console_logs.append(f"[{msg.type}] {msg.text}")
            print(f"CONSOLE [{msg.type}]: {msg.text}")

        def log_error(error):
            errors.append(str(error))
            print(f"ERROR: {error}")

        page.on("console", log_console)
        page.on("pageerror", log_error)

        print("=" * 80)
        print("OPENING VISUALIZATION...")
        print("=" * 80)

        # Navigate to the page
        await page.goto("http://localhost:8082", wait_until="networkidle")

        # Wait a bit for initialization
        await asyncio.sleep(3)

        print("\n" + "=" * 80)
        print("CHECKING DATA LOADING...")
        print("=" * 80)

        # Check data variables
        all_nodes_length = await page.evaluate(
            "() => window.allNodes ? window.allNodes.length : 'undefined'"
        )
        all_links_length = await page.evaluate(
            "() => window.allLinks ? window.allLinks.length : 'undefined'"
        )
        data_loaded = await page.evaluate("() => typeof window.data")
        cy_exists = await page.evaluate("() => typeof window.cy")
        cy_nodes_count = await page.evaluate(
            "() => window.cy ? window.cy.nodes().length : 'no cy'"
        )

        print(f"allNodes length: {all_nodes_length}")
        print(f"allLinks length: {all_links_length}")
        print(f"data loaded: {data_loaded}")
        print(f"cy exists: {cy_exists}")
        print(f"cy nodes count: {cy_nodes_count}")

        print("\n" + "=" * 80)
        print("CHECKING CY-CONTAINER...")
        print("=" * 80)

        # Check cy-container
        cy_container_exists = await page.evaluate(
            "() => !!document.getElementById('cy-container')"
        )
        if cy_container_exists:
            cy_container_info = await page.evaluate(
                """() => {
                const container = document.getElementById('cy-container');
                const rect = container.getBoundingClientRect();
                const styles = window.getComputedStyle(container);
                return {
                    exists: true,
                    display: styles.display,
                    visibility: styles.visibility,
                    opacity: styles.opacity,
                    zIndex: styles.zIndex,
                    width: rect.width,
                    height: rect.height,
                    top: rect.top,
                    left: rect.left,
                    hasChildren: container.children.length,
                    innerHTML: container.innerHTML.substring(0, 200)
                };
            }"""
            )
            print(json.dumps(cy_container_info, indent=2))
        else:
            print("cy-container DOES NOT EXIST!")

        print("\n" + "=" * 80)
        print("CHECKING SVG-CONTAINER...")
        print("=" * 80)

        svg_container_info = await page.evaluate(
            """() => {
            const container = document.querySelector('.graph-container svg');
            if (!container) return { exists: false };
            const rect = container.getBoundingClientRect();
            return {
                exists: true,
                width: rect.width,
                height: rect.height,
                nodeCount: container.querySelectorAll('circle').length,
                linkCount: container.querySelectorAll('line').length
            };
        }"""
        )
        print(json.dumps(svg_container_info, indent=2))

        print("\n" + "=" * 80)
        print("CHECKING CURRENT LAYOUT...")
        print("=" * 80)

        current_layout = await page.evaluate(
            "() => window.currentLayout || 'undefined'"
        )
        print(f"Current layout: {current_layout}")

        print("\n" + "=" * 80)
        print("CHECKING VISIBLE NODES...")
        print("=" * 80)

        visible_nodes = await page.evaluate(
            "() => window.visibleNodes ? window.visibleNodes.size : 'undefined'"
        )
        print(f"Visible nodes set size: {visible_nodes}")

        print("\n" + "=" * 80)
        print("JAVASCRIPT ERRORS:")
        print("=" * 80)
        for error in errors:
            print(f"  - {error}")

        print("\n" + "=" * 80)
        print("TAKING SCREENSHOTS...")
        print("=" * 80)

        # Take screenshots
        await page.screenshot(
            path="/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/debug-fullpage.png",
            full_page=True,
        )
        print("Saved: debug-fullpage.png")

        # Screenshot of just the graph area
        graph_container = await page.query_selector(".graph-container")
        if graph_container:
            await graph_container.screenshot(
                path="/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization/debug-graph-area.png"
            )
            print("Saved: debug-graph-area.png")

        print("\n" + "=" * 80)
        print("CHECKING IF NODES ARE RENDERED BUT HIDDEN...")
        print("=" * 80)

        # Try to find any circles or nodes
        circles = await page.evaluate(
            "() => document.querySelectorAll('circle').length"
        )
        canvas = await page.evaluate("() => document.querySelectorAll('canvas').length")
        print(f"SVG circles found: {circles}")
        print(f"Canvas elements found: {canvas}")

        # Check if it's using D3 force layout
        using_d3 = await page.evaluate("() => typeof window.simulation !== 'undefined'")
        print(f"Using D3 simulation: {using_d3}")

        if using_d3:
            sim_status = await page.evaluate(
                """() => {
                if (window.simulation) {
                    return {
                        alpha: window.simulation.alpha(),
                        nodes: window.simulation.nodes().length
                    };
                }
                return null;
            }"""
            )
            print(f"D3 Simulation status: {sim_status}")

        print("\n" + "=" * 80)
        print("INVESTIGATION COMPLETE")
        print("=" * 80)
        print("\nCheck the screenshots in .mcp-vector-search/visualization/")
        print("Press Enter to close browser...")
        input()

        await browser.close()


if __name__ == "__main__":
    asyncio.run(investigate())
