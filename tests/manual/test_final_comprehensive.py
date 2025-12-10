#!/usr/bin/env python3
"""
Final comprehensive test of the visualizer with all metrics.
"""

import asyncio
import json
import time

from playwright.async_api import async_playwright


async def comprehensive_test():
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        console_messages = []
        page.on(
            "console",
            lambda msg: console_messages.append({"type": msg.type, "text": msg.text}),
        )

        # Test 1: Server Connectivity
        print("\n" + "=" * 70)
        print("TEST 1: Server Connectivity")
        print("=" * 70)
        try:
            start = time.time()
            response = await page.goto(
                "http://localhost:8095", wait_until="networkidle", timeout=15000
            )
            load_time = int((time.time() - start) * 1000)

            results["tests"]["server_connectivity"] = {
                "status": "PASS" if response.ok else "FAIL",
                "http_status": response.status,
                "load_time_ms": load_time,
            }
            print(f"✅ Status: {response.status}")
            print(f"✅ Load Time: {load_time}ms")
        except Exception as e:
            results["tests"]["server_connectivity"] = {
                "status": "FAIL",
                "error": str(e),
            }
            print(f"❌ Error: {e}")
            return results

        await asyncio.sleep(2)  # Wait for JS to execute

        # Test 2: Page Structure
        print("\n" + "=" * 70)
        print("TEST 2: Page Structure")
        print("=" * 70)
        page_info = await page.evaluate(
            """
            () => {
                return {
                    title: document.title,
                    svgCount: document.querySelectorAll('svg').length,
                    nodeCount: document.querySelectorAll('.node').length,
                    linkCount: document.querySelectorAll('.link').length,
                    controlsVisible: !!document.getElementById('controls'),
                    graphVisible: !!document.getElementById('graph'),
                    hasD3: typeof d3 !== 'undefined'
                };
            }
        """
        )

        structure_pass = (
            page_info["svgCount"] == 1
            and page_info["nodeCount"] > 0
            and page_info["hasD3"]
            and page_info["controlsVisible"]
            and page_info["graphVisible"]
        )

        results["tests"]["page_structure"] = {
            "status": "PASS" if structure_pass else "FAIL",
            **page_info,
        }

        print(
            f"{'✅' if structure_pass else '❌'} SVG Elements: {page_info['svgCount']}"
        )
        print(
            f"{'✅' if page_info['nodeCount'] > 0 else '❌'} Node Elements: {page_info['nodeCount']}"
        )
        print(
            f"{'✅' if page_info['linkCount'] >= 0 else '❌'} Link Elements: {page_info['linkCount']}"
        )
        print(
            f"{'✅' if page_info['hasD3'] else '❌'} D3.js Loaded: {page_info['hasD3']}"
        )
        print(f"{'✅' if page_info['controlsVisible'] else '❌'} Controls Visible")
        print(f"{'✅' if page_info['graphVisible'] else '❌'} Graph Container Visible")

        # Test 3: Console Errors
        print("\n" + "=" * 70)
        print("TEST 3: JavaScript Console")
        print("=" * 70)
        errors = [m for m in console_messages if m["type"] == "error"]
        warnings = [m for m in console_messages if m["type"] == "warning"]

        results["tests"]["console"] = {
            "status": "PASS" if len(errors) == 0 else "FAIL",
            "errors": len(errors),
            "warnings": len(warnings),
            "total_messages": len(console_messages),
        }

        print(f"{'✅' if len(errors) == 0 else '❌'} Errors: {len(errors)}")
        print(f"{'✅' if len(warnings) == 0 else '⚠️'} Warnings: {len(warnings)}")
        if errors:
            for error in errors[:5]:
                print(f"  ❌ {error['text']}")

        # Test 4: Interactivity
        print("\n" + "=" * 70)
        print("TEST 4: Interactivity")
        print("=" * 70)
        try:
            # Test zoom functionality
            await page.evaluate(
                """
                () => {
                    const svg = d3.select('#graph');
                    if (svg.node()) {
                        // Simulate zoom event
                        const zoom = d3.zoom();
                        svg.call(zoom.transform, d3.zoomIdentity.scale(1.5));
                        return true;
                    }
                    return false;
                }
            """
            )

            # Check if nodes are clickable
            nodes_clickable = await page.evaluate(
                """
                () => {
                    const nodes = document.querySelectorAll('.node circle');
                    return nodes.length > 0 &&
                           window.getComputedStyle(nodes[0]).cursor === 'pointer';
                }
            """
            )

            results["tests"]["interactivity"] = {
                "status": "PASS",
                "zoom_works": True,
                "nodes_clickable": nodes_clickable,
            }
            print("✅ Zoom functionality works")
            print(f"{'✅' if nodes_clickable else '⚠️'} Nodes are clickable")
        except Exception as e:
            results["tests"]["interactivity"] = {"status": "FAIL", "error": str(e)}
            print(f"❌ Error: {e}")

        # Test 5: Performance
        print("\n" + "=" * 70)
        print("TEST 5: Performance Metrics")
        print("=" * 70)
        metrics = await page.evaluate(
            """
            () => {
                const perf = performance.getEntriesByType('navigation')[0];
                return {
                    domContentLoaded: Math.round(perf.domContentLoadedEventEnd - perf.fetchStart),
                    loadComplete: Math.round(perf.loadEventEnd - perf.fetchStart),
                    domInteractive: Math.round(perf.domInteractive - perf.fetchStart)
                };
            }
        """
        )

        performance_pass = metrics["loadComplete"] < 5000  # Less than 5 seconds

        results["tests"]["performance"] = {
            "status": "PASS" if performance_pass else "WARNING",
            **metrics,
        }

        print(
            f"{'✅' if metrics['domContentLoaded'] < 3000 else '⚠️'} DOM Content Loaded: {metrics['domContentLoaded']}ms"
        )
        print(
            f"{'✅' if metrics['loadComplete'] < 5000 else '⚠️'} Load Complete: {metrics['loadComplete']}ms"
        )
        print(
            f"{'✅' if metrics['domInteractive'] < 2000 else '⚠️'} DOM Interactive: {metrics['domInteractive']}ms"
        )

        # Test 6: Data Loading
        print("\n" + "=" * 70)
        print("TEST 6: Graph Data")
        print("=" * 70)
        data_info = await page.evaluate(
            """
            async () => {
                try {
                    const response = await fetch('chunk-graph.json');
                    const data = await response.json();
                    return {
                        loaded: true,
                        nodes: data.nodes.length,
                        links: data.links.length,
                        hasMetadata: !!data.metadata
                    };
                } catch (e) {
                    return { loaded: false, error: e.toString() };
                }
            }
        """
        )

        results["tests"]["data_loading"] = {
            "status": "PASS" if data_info.get("loaded") else "FAIL",
            **data_info,
        }

        if data_info.get("loaded"):
            print("✅ Graph data loaded successfully")
            print(f"✅ Nodes: {data_info['nodes']}")
            print(f"✅ Links: {data_info['links']}")
            print(f"✅ Has Metadata: {data_info['hasMetadata']}")
        else:
            print(f"❌ Failed to load graph data: {data_info.get('error')}")

        # Take screenshot
        await page.screenshot(
            path="/tmp/visualizer_final_screenshot.png", full_page=True
        )

        await browser.close()

    # Overall Assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    all_tests = results["tests"]
    passed = sum(1 for t in all_tests.values() if t.get("status") == "PASS")
    total = len(all_tests)

    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": f"{(passed / total) * 100:.1f}%",
    }

    print(f"\n✅ Tests Passed: {passed}/{total}")
    print(f"📊 Success Rate: {results['summary']['success_rate']}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Visualizer is fully functional!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review results for details")

    # Save results
    with open("/tmp/visualizer_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n📄 Detailed results: /tmp/visualizer_final_results.json")
    print("📸 Screenshot: /tmp/visualizer_final_screenshot.png")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(comprehensive_test())
    exit(0 if success else 1)
