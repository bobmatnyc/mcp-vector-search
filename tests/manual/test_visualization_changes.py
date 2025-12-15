"""
Manual test script for visualization changes verification.

Tests:
1. Filter button graph redraw functionality
2. Section nav hidden for reports
3. No JavaScript errors during interactions
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import Browser, ConsoleMessage, Page, async_playwright


class VisualizationTester:
    """Test harness for visualization changes."""

    def __init__(self, base_url: str = "http://localhost:8502"):
        self.base_url = base_url
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.console_messages: list[dict] = []
        self.test_results: list[dict] = []
        self.screenshots_dir = Path(__file__).parent / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)

    async def setup(self):
        """Initialize browser and page."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

        # Listen to console messages
        self.page.on("console", self._handle_console_message)

        # Listen to page errors
        self.page.on("pageerror", self._handle_page_error)

        print(f"✓ Browser launched, navigating to {self.base_url}")
        await self.page.goto(self.base_url)
        await self.page.wait_for_load_state("networkidle")
        print("✓ Page loaded")

    def _handle_console_message(self, msg: ConsoleMessage):
        """Capture console messages."""
        self.console_messages.append(
            {"type": msg.type, "text": msg.text, "location": msg.location}
        )

    def _handle_page_error(self, error):
        """Capture page errors."""
        self.console_messages.append(
            {"type": "error", "text": str(error), "location": {"url": self.page.url}}
        )

    async def screenshot(self, name: str):
        """Take a screenshot."""
        path = self.screenshots_dir / f"{name}.png"
        await self.page.screenshot(path=str(path))
        print(f"  📸 Screenshot saved: {path}")
        return path

    async def wait_for_graph_render(self, timeout: int = 3000):
        """Wait for D3 graph to finish rendering."""
        await self.page.wait_for_timeout(timeout)

    async def test_1_filter_buttons_graph_redraw(self):
        """Test 1: Filter buttons cause graph redraw, not just hiding.
        Also verifies that tree remains collapsed (not expanding to massive radial view).
        """
        print("\n" + "=" * 70)
        print("TEST 1: Filter Button Graph Redraw + Collapsed State")
        print("=" * 70)

        result = {
            "test": "Filter Button Graph Redraw + Collapsed State",
            "passed": True,
            "details": [],
        }

        # Take initial screenshot
        await self.screenshot("01_initial_full_graph")
        await self.wait_for_graph_render()

        # Get initial node count
        initial_nodes = await self.page.evaluate(
            """
            () => document.querySelectorAll('.node').length
        """
        )
        print(f"  Initial nodes visible: {initial_nodes}")
        result["details"].append(f"Initial nodes: {initial_nodes}")

        # Click "Code" filter button
        print("\n  Testing 'Code' filter...")
        code_button = self.page.locator('button:has-text("Code")')
        await code_button.click()
        await self.wait_for_graph_render()
        await self.screenshot("02_code_filter_active")

        # Check that graph redrawn (fewer nodes, no display:none)
        code_nodes = await self.page.evaluate(
            """
            () => {
                const nodes = document.querySelectorAll('.node');
                const visible = Array.from(nodes).filter(n =>
                    window.getComputedStyle(n).display !== 'none'
                );
                return {
                    total: nodes.length,
                    visible: visible.length,
                    hidden: nodes.length - visible.length
                };
            }
        """
        )
        print(f"  After Code filter: {code_nodes}")
        result["details"].append(f"Code filter: {code_nodes}")

        # CRITICAL: Verify tree stays collapsed (not expanding to massive view)
        # Should only see root + first-level children, not thousands of nodes
        if code_nodes["visible"] > 1000:
            result["passed"] = False
            result["details"].append(
                f"❌ FAIL: Too many nodes expanded ({code_nodes['visible']}). Tree should stay collapsed."
            )
            print(
                f"  ❌ FAIL: Massive expansion detected ({code_nodes['visible']} nodes)"
            )
        else:
            result["details"].append(
                f"✓ Tree stayed collapsed ({code_nodes['visible']} nodes)"
            )
            print(f"  ✓ Tree stayed collapsed ({code_nodes['visible']} nodes)")

        # Verify no documentation files visible
        doc_nodes = await self.page.evaluate(
            """
            () => {
                const nodes = document.querySelectorAll('.node');
                const docNodes = Array.from(nodes).filter(n => {
                    const text = n.textContent || '';
                    return text.match(/\\.(md|txt|rst)$/i);
                });
                return docNodes.length;
            }
        """
        )

        if doc_nodes > 0:
            result["passed"] = False
            result["details"].append(
                f"❌ FAIL: {doc_nodes} doc nodes still visible with Code filter"
            )
            print(f"  ❌ FAIL: {doc_nodes} doc nodes still visible")
        else:
            result["details"].append("✓ No doc nodes visible with Code filter")
            print("  ✓ No doc nodes visible")

        # Click "Docs" filter
        print("\n  Testing 'Docs' filter...")
        docs_button = self.page.locator('button:has-text("Docs")')
        await docs_button.click()
        await self.wait_for_graph_render()
        await self.screenshot("03_docs_filter_active")

        docs_nodes = await self.page.evaluate(
            """
            () => {
                const nodes = document.querySelectorAll('.node');
                return {
                    total: nodes.length,
                    visible: Array.from(nodes).filter(n =>
                        window.getComputedStyle(n).display !== 'none'
                    ).length
                };
            }
        """
        )
        print(f"  After Docs filter: {docs_nodes}")
        result["details"].append(f"Docs filter: {docs_nodes}")

        # Click "All" to reset
        print("\n  Testing 'All' filter (reset)...")
        all_button = self.page.locator('button:has-text("All")')
        await all_button.click()
        await self.wait_for_graph_render()
        await self.screenshot("04_all_filter_restored")

        all_nodes = await self.page.evaluate(
            """
            () => document.querySelectorAll('.node').length
        """
        )
        print(f"  After All filter: {all_nodes} nodes")
        result["details"].append(f"All filter: {all_nodes} nodes")

        if all_nodes != initial_nodes:
            result["passed"] = False
            result["details"].append(
                f"❌ FAIL: Node count mismatch ({all_nodes} vs {initial_nodes})"
            )
            print(f"  ❌ FAIL: Expected {initial_nodes} nodes, got {all_nodes}")
        else:
            result["details"].append("✓ Full graph restored")
            print("  ✓ Full graph restored")

        self.test_results.append(result)
        print(f"\n  Result: {'✓ PASS' if result['passed'] else '❌ FAIL'}")
        return result

    async def test_2_section_nav_hidden_for_reports(self):
        """Test 2: Section nav dropdown hidden for reports, visible for code."""
        print("\n" + "=" * 70)
        print("TEST 2: Section Nav Hidden for Reports")
        print("=" * 70)

        result = {
            "test": "Section Nav Hidden for Reports",
            "passed": True,
            "details": [],
        }

        # Test Complexity report
        print("\n  Testing Complexity report...")
        complexity_button = self.page.locator('button:has-text("Complexity")')
        await complexity_button.click()
        await self.page.wait_for_timeout(1000)
        await self.screenshot("05_complexity_report")

        section_nav_visible = await self.page.evaluate(
            """
            () => {
                const dropdown = document.querySelector('select[id*="section"]');
                if (!dropdown) return false;
                return window.getComputedStyle(dropdown).display !== 'none';
            }
        """
        )

        if section_nav_visible:
            result["passed"] = False
            result["details"].append(
                "❌ FAIL: Section nav visible for Complexity report"
            )
            print("  ❌ FAIL: Section nav should be hidden")
        else:
            result["details"].append("✓ Section nav hidden for Complexity report")
            print("  ✓ Section nav hidden")

        # Test Code Smells report
        print("\n  Testing Code Smells report...")
        smells_button = self.page.locator('button:has-text("Code Smells")')
        await smells_button.click()
        await self.page.wait_for_timeout(1000)
        await self.screenshot("06_code_smells_report")

        section_nav_visible = await self.page.evaluate(
            """
            () => {
                const dropdown = document.querySelector('select[id*="section"]');
                if (!dropdown) return false;
                return window.getComputedStyle(dropdown).display !== 'none';
            }
        """
        )

        if section_nav_visible:
            result["passed"] = False
            result["details"].append(
                "❌ FAIL: Section nav visible for Code Smells report"
            )
            print("  ❌ FAIL: Section nav should be hidden")
        else:
            result["details"].append("✓ Section nav hidden for Code Smells report")
            print("  ✓ Section nav hidden")

        # Test Dependencies report
        print("\n  Testing Dependencies report...")
        deps_button = self.page.locator('button:has-text("Dependencies")')
        await deps_button.click()
        await self.page.wait_for_timeout(1000)
        await self.screenshot("07_dependencies_report")

        section_nav_visible = await self.page.evaluate(
            """
            () => {
                const dropdown = document.querySelector('select[id*="section"]');
                if (!dropdown) return false;
                return window.getComputedStyle(dropdown).display !== 'none';
            }
        """
        )

        if section_nav_visible:
            result["passed"] = False
            result["details"].append(
                "❌ FAIL: Section nav visible for Dependencies report"
            )
            print("  ❌ FAIL: Section nav should be hidden")
        else:
            result["details"].append("✓ Section nav hidden for Dependencies report")
            print("  ✓ Section nav hidden")

        # Test code chunk viewing (section nav should be visible)
        print("\n  Testing code chunk viewing...")
        # First, reset to All filter
        all_button = self.page.locator('button:has-text("All")')
        await all_button.click()
        await self.wait_for_graph_render()

        # Click on a code chunk node
        code_node = self.page.locator(".node").first
        await code_node.click()
        await self.page.wait_for_timeout(1000)
        await self.screenshot("08_code_chunk_viewer")

        section_nav_visible = await self.page.evaluate(
            """
            () => {
                const dropdown = document.querySelector('select[id*="section"]');
                if (!dropdown) return false;
                return window.getComputedStyle(dropdown).display !== 'none';
            }
        """
        )

        if not section_nav_visible:
            result["passed"] = False
            result["details"].append(
                "❌ FAIL: Section nav hidden for code chunk viewing"
            )
            print("  ❌ FAIL: Section nav should be visible for code")
        else:
            result["details"].append("✓ Section nav visible for code chunk viewing")
            print("  ✓ Section nav visible for code")

        self.test_results.append(result)
        print(f"\n  Result: {'✓ PASS' if result['passed'] else '❌ FAIL'}")
        return result

    async def test_3_no_javascript_errors(self):
        """Test 3: No JavaScript errors during interactions."""
        print("\n" + "=" * 70)
        print("TEST 3: No JavaScript Errors")
        print("=" * 70)

        result = {"test": "No JavaScript Errors", "passed": True, "details": []}

        # Filter console messages for errors
        errors = [msg for msg in self.console_messages if msg["type"] == "error"]
        warnings = [msg for msg in self.console_messages if msg["type"] == "warning"]

        print(f"\n  Total console messages: {len(self.console_messages)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")

        if errors:
            result["passed"] = False
            result["details"].append(
                f"❌ FAIL: {len(errors)} JavaScript errors detected"
            )
            print("\n  JavaScript Errors:")
            for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
                print(f"    {i}. {error['text']}")
                result["details"].append(f"Error: {error['text']}")
        else:
            result["details"].append("✓ No JavaScript errors detected")
            print("  ✓ No JavaScript errors detected")

        # Show warnings (informational)
        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings[:3], 1):  # Show first 3 warnings
                print(f"    {i}. {warning['text']}")

        self.test_results.append(result)
        print(f"\n  Result: {'✓ PASS' if result['passed'] else '❌ FAIL'}")
        return result

    async def generate_report(self):
        """Generate test report."""
        print("\n" + "=" * 70)
        print("TEST REPORT")
        print("=" * 70)

        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)

        print(f"\nTests Passed: {passed}/{total}")
        print("\nDetailed Results:")

        for i, result in enumerate(self.test_results, 1):
            status = "✓ PASS" if result["passed"] else "❌ FAIL"
            print(f"\n{i}. {result['test']}: {status}")
            for detail in result["details"]:
                print(f"   {detail}")

        # Save report to JSON
        report_path = Path(__file__).parent / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": total - passed,
                    },
                    "tests": self.test_results,
                    "console_messages": self.console_messages[:50],  # First 50 messages
                },
                f,
                indent=2,
            )

        print(f"\n📄 Full report saved: {report_path}")
        print(f"📸 Screenshots saved: {self.screenshots_dir}")

        return passed == total

    async def cleanup(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()
        print("\n✓ Browser closed")

    async def run_all_tests(self):
        """Run all test cases."""
        try:
            await self.setup()

            await self.test_1_filter_buttons_graph_redraw()
            await self.test_2_section_nav_hidden_for_reports()
            await self.test_3_no_javascript_errors()

            all_passed = await self.generate_report()

            return all_passed

        finally:
            await self.cleanup()


async def main():
    """Main test runner."""
    tester = VisualizationTester()
    all_passed = await tester.run_all_tests()

    if all_passed:
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
