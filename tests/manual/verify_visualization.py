#!/usr/bin/env python3
"""
Manual verification script for visualization fix.
Tests that graph nodes are visible and properly initialized.
"""

import json
import time

from playwright.sync_api import sync_playwright


def verify_visualization():
    """Verify the visualization at http://localhost:8082"""

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        print("🚀 Navigating to http://localhost:8082...")
        page.goto("http://localhost:8082", wait_until="networkidle", timeout=60000)

        # Wait for page to load
        print("⏳ Waiting for visualization to initialize...")
        time.sleep(5)

        # Take initial screenshot
        page.screenshot(
            path="/Users/masa/Projects/mcp-vector-search/tests/manual/screenshot_initial.png",
            full_page=True,
        )
        print("📸 Initial screenshot saved to tests/manual/screenshot_initial.png")

        # Check console for errors
        console_messages = []
        page.on(
            "console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}")
        )

        # Verify data initialization
        print("\n" + "=" * 60)
        print("VERIFICATION 1: Data Initialization")
        print("=" * 60)

        all_nodes_length = page.evaluate("allNodes ? allNodes.length : 0")
        all_links_length = page.evaluate("allLinks ? allLinks.length : 0")
        cy_nodes_length = page.evaluate("cy ? cy.nodes().length : 0")
        cy_edges_length = page.evaluate("cy ? cy.edges().length : 0")

        print(f"✓ allNodes.length: {all_nodes_length} (expected: 1449)")
        print(f"✓ allLinks.length: {all_links_length} (expected: ~360000)")
        print(f"✓ cy.nodes().length: {cy_nodes_length} (expected: matches allNodes)")
        print(f"✓ cy.edges().length: {cy_edges_length}")

        # Verify graph visibility
        print("\n" + "=" * 60)
        print("VERIFICATION 2: Graph Visibility")
        print("=" * 60)

        # Check if canvas/SVG elements exist
        canvas_exists = page.evaluate("document.querySelector('canvas') !== null")
        cy_container_exists = page.evaluate("document.querySelector('#cy') !== null")

        print(f"✓ Canvas element exists: {canvas_exists}")
        print(f"✓ Cytoscape container exists: {cy_container_exists}")

        # Get bounding box of graph container
        if cy_container_exists:
            container_info = page.evaluate(
                """
                () => {
                    const container = document.querySelector('#cy');
                    const rect = container.getBoundingClientRect();
                    return {
                        width: rect.width,
                        height: rect.height,
                        visible: rect.width > 0 && rect.height > 0
                    };
                }
            """
            )
            print(
                f"✓ Container dimensions: {container_info['width']}x{container_info['height']}"
            )
            print(f"✓ Container visible: {container_info['visible']}")

        # Verify controls
        print("\n" + "=" * 60)
        print("VERIFICATION 3: Controls")
        print("=" * 60)

        layout_selector = page.query_selector("#layoutSelect")
        edge_filter = page.query_selector("#edgeFilter")
        legend = page.query_selector(".legend")

        print(f"✓ Layout selector found: {layout_selector is not None}")
        print(f"✓ Edge filter found: {edge_filter is not None}")
        print(f"✓ Legend found: {legend is not None}")

        if layout_selector:
            current_layout = page.evaluate(
                "document.querySelector('#layoutSelect').value"
            )
            print(f"✓ Current layout: {current_layout}")

        # Take final screenshot
        page.screenshot(
            path="/Users/masa/Projects/mcp-vector-search/tests/manual/screenshot_final.png",
            full_page=True,
        )
        print("\n📸 Final screenshot saved to tests/manual/screenshot_final.png")

        # Test interactivity
        print("\n" + "=" * 60)
        print("VERIFICATION 4: Interactivity")
        print("=" * 60)

        # Try to click a node
        try:
            click_result = page.evaluate(
                """
                () => {
                    if (cy && cy.nodes().length > 0) {
                        const node = cy.nodes()[0];
                        node.emit('tap');
                        return {success: true, nodeId: node.id()};
                    }
                    return {success: false, error: 'No nodes available'};
                }
            """
            )
            print(f"✓ Node click test: {click_result}")
        except Exception as e:
            print(f"✗ Node click test failed: {e}")

        # Check if sidebar updates
        time.sleep(1)
        sidebar_content = page.evaluate(
            "document.querySelector('.sidebar') ? document.querySelector('.sidebar').textContent : 'No sidebar'"
        )
        print(f"✓ Sidebar content length: {len(sidebar_content)} chars")

        # Generate report
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)

        success_criteria = {
            "allNodes.length === 1449": all_nodes_length == 1449,
            "allLinks.length > 0": all_links_length > 0,
            "Graph visually rendered": cy_container_exists and canvas_exists,
            "Controls functional": layout_selector is not None
            and edge_filter is not None,
            "Cytoscape initialized": cy_nodes_length > 0,
        }

        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {criterion}")

        all_passed = all(success_criteria.values())

        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 ALL VERIFICATION CHECKS PASSED")
        else:
            print("⚠️  SOME VERIFICATION CHECKS FAILED")
        print("=" * 60)

        # Save detailed report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_initialization": {
                "allNodes_length": all_nodes_length,
                "allLinks_length": all_links_length,
                "cy_nodes_length": cy_nodes_length,
                "cy_edges_length": cy_edges_length,
            },
            "visibility": {
                "canvas_exists": canvas_exists,
                "cy_container_exists": cy_container_exists,
                "container_info": container_info if cy_container_exists else None,
            },
            "controls": {
                "layout_selector": layout_selector is not None,
                "edge_filter": edge_filter is not None,
                "legend": legend is not None,
                "current_layout": current_layout if layout_selector else None,
            },
            "success_criteria": success_criteria,
            "all_passed": all_passed,
        }

        with open(
            "/Users/masa/Projects/mcp-vector-search/tests/manual/verification_report.json",
            "w",
        ) as f:
            json.dump(report, f, indent=2)

        print("\n📄 Detailed report saved to tests/manual/verification_report.json")

        # Keep browser open for 5 seconds to allow manual inspection
        print("\n👀 Keeping browser open for 5 seconds for manual inspection...")
        time.sleep(5)

        browser.close()

        return all_passed


if __name__ == "__main__":
    try:
        success = verify_visualization()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
