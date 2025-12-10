#!/usr/bin/env python3
"""
Test the root breadcrumb navigation fix.

This test verifies that clicking the "Root" breadcrumb properly resets
the visualization to Phase 1 (initial overview), showing ONLY top-level
directories without any fragments (expanded children).

Issue: Previously, clicking root would show fragments of previously expanded
nodes instead of a clean grid of top-level directories.

Fix: Enhanced StateManager.reset() and _showAllRootNodes() to explicitly hide
all non-root nodes when returning to Phase 1.
"""

import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright


async def test_root_breadcrumb_reset(url: str = "http://localhost:8080"):
    """Test that clicking root breadcrumb shows clean Phase 1 view."""

    print(f"\n{'=' * 70}")
    print("🧪 ROOT BREADCRUMB RESET - VERIFICATION TEST")
    print(f"{'=' * 70}\n")
    print("This test verifies the fix for the root breadcrumb navigation issue.")
    print("Previously, clicking 'Root' would show fragments of expanded children.")
    print("After the fix, it should show ONLY top-level directories in a clean grid.\n")

    results = {
        "initial_nodes": [],
        "after_expansion": [],
        "after_reset": [],
        "passed": False,
        "errors": [],
    }

    async with async_playwright() as p:
        print("🚀 Launching browser...")
        browser = await p.chromium.launch(headless=False)  # Use headless=False to watch
        page: Page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        # Capture console logs
        console_logs = []

        def handle_console(msg):
            console_logs.append(f"[{msg.type}] {msg.text}")
            if "StateManager" in msg.text or "Reset" in msg.text:
                print(f"  📝 {msg.text}")

        page.on("console", handle_console)

        try:
            # Step 1: Load visualization
            print("\n📋 STEP 1: Load Visualization")
            print(f"  → Navigating to {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # Wait for rendering
            print("  ✓ Page loaded")

            # Get initial nodes (should be only root directories)
            initial_nodes = await page.evaluate(
                """
                () => {
                    if (!stateManager) return {error: "StateManager not found"};
                    const visible = stateManager.getVisibleNodes();
                    const nodes = allNodes.filter(n => visible.includes(n.id));
                    return {
                        count: nodes.length,
                        types: nodes.reduce((acc, n) => {
                            acc[n.type] = (acc[n.type] || 0) + 1;
                            return acc;
                        }, {}),
                        sample_names: nodes.slice(0, 5).map(n => n.name)
                    };
                }
            """
            )

            results["initial_nodes"] = initial_nodes
            print("\n  Initial State:")
            print(f"    Visible nodes: {initial_nodes.get('count', 'ERROR')}")
            print(f"    Node types: {initial_nodes.get('types', {})}")
            print(f"    Sample names: {initial_nodes.get('sample_names', [])}")

            # Step 2: Expand a directory
            print("\n📋 STEP 2: Expand a Directory")
            expand_result = await page.evaluate(
                """
                () => {
                    // Find first directory node
                    const dirNode = allNodes.find(n => n.type === 'directory');
                    if (!dirNode) return {error: "No directory found"};

                    console.log('[TEST] Expanding directory:', dirNode.name);

                    // Simulate click to expand
                    const event = new MouseEvent('click', {bubbles: true, cancelable: true});
                    const nodeElement = document.querySelector(`[data-id="${dirNode.id}"]`);
                    if (nodeElement) {
                        nodeElement.dispatchEvent(event);
                    } else {
                        // Fallback: manually expand
                        expandNodeV2(dirNode.id, 'directory');
                    }

                    return {expanded: dirNode.name, id: dirNode.id};
                }
            """
            )

            print(f"  → Expanded directory: {expand_result.get('expanded', 'ERROR')}")
            await page.wait_for_timeout(1500)  # Wait for expansion animation

            # Get state after expansion
            after_expansion = await page.evaluate(
                """
                () => {
                    if (!stateManager) return {error: "StateManager not found"};
                    const visible = stateManager.getVisibleNodes();
                    const nodes = allNodes.filter(n => visible.includes(n.id));
                    return {
                        count: nodes.length,
                        types: nodes.reduce((acc, n) => {
                            acc[n.type] = (acc[n.type] || 0) + 1;
                            return acc;
                        }, {}),
                        expansion_path: stateManager.expansionPath,
                        view_mode: stateManager.viewMode
                    };
                }
            """
            )

            results["after_expansion"] = after_expansion
            print("\n  After Expansion:")
            print(f"    Visible nodes: {after_expansion.get('count', 'ERROR')}")
            print(f"    Node types: {after_expansion.get('types', {})}")
            print(f"    Expansion path: {after_expansion.get('expansion_path', [])}")
            print(f"    View mode: {after_expansion.get('view_mode', 'ERROR')}")

            # Step 3: Click Root breadcrumb
            print("\n📋 STEP 3: Click Root Breadcrumb")
            print("  → Looking for root breadcrumb button...")

            # Find and click root breadcrumb
            try:
                await page.click("text=/🏠.*Root/", timeout=3000)
                print("  ✓ Clicked root breadcrumb")
            except:
                # Fallback: call function directly
                await page.evaluate("resetToListViewV2()")
                print("  ✓ Called resetToListViewV2() directly")

            await page.wait_for_timeout(1500)  # Wait for reset animation

            # Get state after reset
            after_reset = await page.evaluate(
                """
                () => {
                    if (!stateManager) return {error: "StateManager not found"};
                    const visible = stateManager.getVisibleNodes();
                    const nodes = allNodes.filter(n => visible.includes(n.id));

                    // Check for fragments (non-root nodes that shouldn't be visible)
                    const fragments = nodes.filter(n => {
                        // A fragment is a non-structural node or a child node
                        const isStructural = n.type === 'directory' || n.type === 'file' || n.type === 'subproject';
                        if (!isStructural) return true;

                        // Check if it has a parent (shouldn't be visible in Phase 1)
                        const hasParent = allLinks.some(link => {
                            const targetId = link.target.id || link.target;
                            return targetId === n.id &&
                                   (link.type === 'dir_containment' ||
                                    link.type === 'file_containment' ||
                                    link.type === 'dir_hierarchy');
                        });

                        return hasParent;  // If it has a parent, it's a fragment in Phase 1
                    });

                    return {
                        count: nodes.length,
                        types: nodes.reduce((acc, n) => {
                            acc[n.type] = (acc[n.type] || 0) + 1;
                            return acc;
                        }, {}),
                        expansion_path: stateManager.expansionPath,
                        view_mode: stateManager.viewMode,
                        fragments_count: fragments.length,
                        fragments: fragments.map(n => ({id: n.id, name: n.name, type: n.type}))
                    };
                }
            """
            )

            results["after_reset"] = after_reset
            print("\n  After Reset to Root:")
            print(f"    Visible nodes: {after_reset.get('count', 'ERROR')}")
            print(f"    Node types: {after_reset.get('types', {})}")
            print(f"    Expansion path: {after_reset.get('expansion_path', [])}")
            print(f"    View mode: {after_reset.get('view_mode', 'ERROR')}")
            print(
                f"    Fragments detected: {after_reset.get('fragments_count', 'ERROR')}"
            )

            if after_reset.get("fragments", []):
                print("\n  ⚠️  Fragment nodes found (should be hidden):")
                for frag in after_reset["fragments"][:5]:
                    print(f"      - {frag['name']} ({frag['type']})")

            # Step 4: Verify fix
            print(f"\n{'=' * 70}")
            print("📊 VERIFICATION RESULTS")
            print(f"{'=' * 70}\n")

            passed = True
            errors = []

            # Check 1: Expansion path should be empty
            if after_reset.get("expansion_path"):
                print("  ❌ FAIL: Expansion path not cleared")
                print("      Expected: []")
                print(f"      Got: {after_reset['expansion_path']}")
                errors.append("Expansion path not cleared")
                passed = False
            else:
                print("  ✅ PASS: Expansion path cleared")

            # Check 2: View mode should be tree_root
            if after_reset.get("view_mode") != "tree_root":
                print("  ❌ FAIL: View mode not reset to tree_root")
                print("      Expected: tree_root")
                print(f"      Got: {after_reset['view_mode']}")
                errors.append("View mode not reset")
                passed = False
            else:
                print("  ✅ PASS: View mode reset to tree_root")

            # Check 3: No fragments (non-root nodes) should be visible
            if after_reset.get("fragments_count", 0) > 0:
                print(
                    f"  ❌ FAIL: Fragments detected ({after_reset['fragments_count']} nodes)"
                )
                print("      These are child nodes that should be hidden in Phase 1")
                errors.append(
                    f"{after_reset['fragments_count']} fragment nodes visible"
                )
                passed = False
            else:
                print("  ✅ PASS: No fragments detected (clean Phase 1 view)")

            # Check 4: Should return to similar node count as initial
            initial_count = initial_nodes.get("count", 0)
            reset_count = after_reset.get("count", 0)

            # Allow some tolerance (within 20% difference)
            if abs(reset_count - initial_count) > initial_count * 0.2:
                print("  ⚠️  WARNING: Node count differs significantly")
                print(f"      Initial: {initial_count}")
                print(f"      After reset: {reset_count}")
                errors.append(f"Node count mismatch: {initial_count} vs {reset_count}")
            else:
                print(
                    f"  ✅ PASS: Node count consistent ({initial_count} → {reset_count})"
                )

            results["passed"] = passed
            results["errors"] = errors

            print(f"\n{'=' * 70}")
            if passed:
                print(
                    "🎉 FINAL VERDICT: ✅ PASS - Root breadcrumb reset works correctly"
                )
                print(
                    "   Clicking 'Root' now shows a clean Phase 1 grid with no fragments!"
                )
            else:
                print("⚠️  FINAL VERDICT: ❌ FAIL - Issues detected")
                print(f"   Errors: {', '.join(errors)}")
            print(f"{'=' * 70}\n")

        except Exception as e:
            print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            results["passed"] = False
            results["errors"].append(str(e))

        finally:
            # Save console logs
            results["console_logs"] = console_logs[-50:]  # Last 50 logs

            await browser.close()

    return results


async def main():
    """Main test runner."""
    results = await test_root_breadcrumb_reset()

    # Save results
    results_path = Path(__file__).parent / "root_breadcrumb_reset_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
