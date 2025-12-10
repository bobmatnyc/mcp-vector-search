#!/usr/bin/env python3
"""
Comprehensive Playwright test for code graph visualization.
Tests UI controls, graph rendering, data loading, and JavaScript errors.
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def test_visualization():
    """Test the code graph visualization at http://localhost:8082"""

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        # Collect console messages
        console_messages = []
        errors = []

        def handle_console(msg):
            console_messages.append(
                {"type": msg.type, "text": msg.text, "location": msg.location}
            )
            if msg.type in ["error", "warning"]:
                errors.append(
                    {"type": msg.type, "text": msg.text, "location": msg.location}
                )

        page.on("console", handle_console)
        page.on(
            "pageerror",
            lambda err: errors.append({"type": "pageerror", "text": str(err)}),
        )

        print("=" * 80)
        print("TESTING CODE GRAPH VISUALIZATION")
        print("=" * 80)

        # Navigate to page
        print("\n[1] Navigating to http://localhost:8082...")
        try:
            response = await page.goto(
                "http://localhost:8082", wait_until="networkidle", timeout=30000
            )
            print(f"    ✅ Page loaded: {response.status} {response.status_text}")
        except Exception as e:
            print(f"    ❌ Failed to load page: {e}")
            await browser.close()
            return

        # Wait a bit for JavaScript to initialize
        await page.wait_for_timeout(2000)

        # Take initial screenshot
        screenshot_dir = Path(
            "/Users/masa/Projects/mcp-vector-search/docs/research/screenshots"
        )
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        screenshot_path = screenshot_dir / "visualization_initial.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"\n[2] Screenshot saved: {screenshot_path}")

        # Check UI Controls
        print("\n[3] Checking UI Controls...")

        # Layout selector
        layout_selector = await page.query_selector("#layoutSelect")
        if layout_selector:
            is_visible = await layout_selector.is_visible()
            print(f"    ✅ Layout selector: {'Visible' if is_visible else 'Hidden'}")
        else:
            print("    ❌ Layout selector: Not found")

        # Edge filters
        edge_filters = await page.query_selector_all("input[type='checkbox']")
        print(
            f"    {'✅' if edge_filters else '❌'} Edge filter checkboxes: {len(edge_filters)} found"
        )

        # Legend
        legend = await page.query_selector("#legend")
        if legend:
            is_visible = await legend.is_visible()
            await legend.inner_html()
            print(f"    ✅ Legend: {'Visible' if is_visible else 'Hidden'}")
        else:
            print("    ❌ Legend: Not found")

        # Check Graph Container
        print("\n[4] Checking Graph Container...")

        cy_container = await page.query_selector("#cy")
        if cy_container:
            is_visible = await cy_container.is_visible()
            box = await cy_container.bounding_box()
            print("    ✅ Cytoscape container (#cy): Found")
            print(f"       Visible: {is_visible}")
            if box:
                print(f"       Dimensions: {box['width']}x{box['height']}")
                print(f"       Position: ({box['x']}, {box['y']})")

            # Check if container has any children
            children = await page.query_selector_all("#cy > *")
            print(f"       Child elements: {len(children)}")
        else:
            print("    ❌ Cytoscape container (#cy): Not found")

        # Check SVG elements
        print("\n[5] Checking SVG/Canvas Elements...")

        svg_elements = await page.query_selector_all("svg")
        canvas_elements = await page.query_selector_all("canvas")
        print(f"    SVG elements: {len(svg_elements)}")
        print(f"    Canvas elements: {len(canvas_elements)}")

        if canvas_elements:
            for i, canvas in enumerate(canvas_elements):
                box = await canvas.bounding_box()
                if box:
                    print(
                        f"       Canvas {i}: {box['width']}x{box['height']} at ({box['x']}, {box['y']})"
                    )

        # Check Data Loading
        print("\n[6] Checking Data Loading...")

        # Check if chunk-graph.json was loaded
        try:
            graph_data = await page.evaluate(
                """
                () => {
                    // Try to access global variables
                    if (typeof window.cy !== 'undefined') {
                        return {
                            cyExists: true,
                            nodeCount: window.cy.nodes().length,
                            edgeCount: window.cy.edges().length
                        };
                    }
                    return { cyExists: false };
                }
            """
            )

            if graph_data.get("cyExists"):
                print("    ✅ Cytoscape instance exists")
                print(f"       Nodes: {graph_data.get('nodeCount', 0)}")
                print(f"       Edges: {graph_data.get('edgeCount', 0)}")
            else:
                print("    ❌ Cytoscape instance not found in window object")
        except Exception as e:
            print(f"    ❌ Error checking Cytoscape data: {e}")

        # Check for data loading in console
        data_loaded = any(
            "loaded" in msg["text"].lower() and "data" in msg["text"].lower()
            for msg in console_messages
        )
        print(
            f"    {'✅' if data_loaded else '⚠️'} Data loading messages in console: {data_loaded}"
        )

        # Check Layout Execution
        print("\n[7] Checking Layout Execution...")

        try:
            layout_info = await page.evaluate(
                """
                () => {
                    if (typeof window.cy !== 'undefined') {
                        // Get positions of first few nodes
                        const nodes = window.cy.nodes().slice(0, 5);
                        const positions = nodes.map(n => ({
                            id: n.id(),
                            x: n.position('x'),
                            y: n.position('y')
                        }));
                        return {
                            hasPositions: positions.every(p => !isNaN(p.x) && !isNaN(p.y)),
                            positions: positions
                        };
                    }
                    return { hasPositions: false };
                }
            """
            )

            if layout_info.get("hasPositions"):
                print("    ✅ Layout executed: Nodes have positions")
                print("       Sample positions:")
                for pos in layout_info.get("positions", [])[:3]:
                    print(f"         {pos['id']}: ({pos['x']:.2f}, {pos['y']:.2f})")
            else:
                print("    ❌ Layout not executed: Nodes missing positions")
        except Exception as e:
            print(f"    ❌ Error checking layout: {e}")

        # Check Console Errors
        print("\n[8] Browser Console Analysis...")

        print(f"    Total console messages: {len(console_messages)}")
        print(f"    Errors/Warnings: {len(errors)}")

        if errors:
            print("\n    JavaScript Errors/Warnings:")
            for err in errors:
                print(f"      [{err['type'].upper()}] {err['text']}")
                if "location" in err and err["location"]:
                    loc = err["location"]
                    print(
                        f"        at {loc.get('url', 'unknown')}:{loc.get('lineNumber', '?')}"
                    )

        # Check Network Requests
        print("\n[9] Checking Network Requests...")

        # Monitor network for chunk-graph.json
        network_requests = []

        async def handle_request(request):
            network_requests.append(
                {
                    "url": request.url,
                    "method": request.method,
                    "resource_type": request.resource_type,
                }
            )

        page.on("request", handle_request)

        # Trigger a refresh to capture network
        await page.reload(wait_until="networkidle")
        await page.wait_for_timeout(2000)

        print(f"    Total network requests: {len(network_requests)}")

        # Check for chunk-graph.json
        graph_request = next(
            (r for r in network_requests if "chunk-graph.json" in r["url"]), None
        )
        if graph_request:
            print(f"    ✅ chunk-graph.json requested: {graph_request['url']}")
        else:
            print("    ❌ chunk-graph.json NOT requested")

        # Take final screenshot
        screenshot_path_final = screenshot_dir / "visualization_final.png"
        await page.screenshot(path=str(screenshot_path_final), full_page=True)
        print(f"\n[10] Final screenshot saved: {screenshot_path_final}")

        # Check if nodes are actually visible
        print("\n[11] Visual Node Detection...")

        try:
            visual_check = await page.evaluate(
                """
                () => {
                    if (typeof window.cy === 'undefined') {
                        return { status: 'no_cy' };
                    }

                    const nodes = window.cy.nodes();
                    if (nodes.length === 0) {
                        return { status: 'no_nodes', nodeCount: 0 };
                    }

                    // Check if nodes are rendered
                    const rendered = nodes.every(n => n.visible());

                    // Get zoom and pan
                    const zoom = window.cy.zoom();
                    const pan = window.cy.pan();

                    // Get extent of graph
                    const extent = window.cy.elements().boundingBox();

                    return {
                        status: 'ok',
                        nodeCount: nodes.length,
                        allVisible: rendered,
                        zoom: zoom,
                        pan: pan,
                        extent: extent
                    };
                }
            """
            )

            if visual_check.get("status") == "no_cy":
                print("    ❌ Cytoscape not initialized")
            elif visual_check.get("status") == "no_nodes":
                print("    ❌ No nodes in graph")
            else:
                print(f"    ✅ Nodes in graph: {visual_check.get('nodeCount')}")
                print(f"    ✅ All nodes visible: {visual_check.get('allVisible')}")
                print(f"    Zoom level: {visual_check.get('zoom', 'N/A')}")
                print(f"    Pan: {visual_check.get('pan', 'N/A')}")
                extent = visual_check.get("extent", {})
                if extent:
                    print(
                        f"    Graph extent: ({extent.get('x1', '?')}, {extent.get('y1', '?')}) to ({extent.get('x2', '?')}, {extent.get('y2', '?')})"
                    )
        except Exception as e:
            print(f"    ❌ Error checking visual state: {e}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if errors:
            print(f"\n❌ ISSUES FOUND: {len(errors)} errors/warnings")
            print("\nTop Issues:")
            for err in errors[:5]:
                print(f"  • {err['text'][:100]}")
        else:
            print("\n✅ No JavaScript errors detected")

        print(f"\nScreenshots saved to: {screenshot_dir}")
        print("  - visualization_initial.png")
        print("  - visualization_final.png")

        # Save detailed report
        report_path = Path(
            "/Users/masa/Projects/mcp-vector-search/docs/research/visualization_test_report.json"
        )
        report = {
            "console_messages": console_messages,
            "errors": errors,
            "network_requests": network_requests,
            "test_timestamp": page.evaluate("() => new Date().toISOString()"),
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_path}")

        # Keep browser open for manual inspection
        print("\n" + "=" * 80)
        print("Browser will remain open for 30 seconds for manual inspection...")
        print("=" * 80)
        await page.wait_for_timeout(30000)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_visualization())
