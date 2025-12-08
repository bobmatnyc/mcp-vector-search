#!/usr/bin/env python3
"""Comprehensive test runner for MCP Vector Search."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_command(cmd: list[str], description: str) -> dict[str, Any]:
    """Run a command and return results."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.perf_counter()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        end_time = time.perf_counter()
        duration = end_time - start_time

        success = result.returncode == 0

        if success:
            print(f"✅ {description} - Passed ({duration:.2f}s)")
        else:
            print(f"❌ {description} - Failed ({duration:.2f}s)")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")

        return {
            "description": description,
            "success": success,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time

        print(f"❌ {description} - Error ({duration:.2f}s): {e}")

        return {
            "description": description,
            "success": False,
            "duration": duration,
            "error": str(e),
        }


def run_unit_tests(test_pattern: str = None) -> list[dict[str, Any]]:
    """Run unit tests."""
    results = []

    # Base pytest command
    cmd = ["python3", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    # Add coverage if available
    try:
        subprocess.run(
            ["python3", "-m", "pytest_cov", "--version"],
            capture_output=True,
            check=True,
        )
        cmd.extend(["--cov=src/mcp_vector_search", "--cov-report=term-missing"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📝 Coverage not available (install pytest-cov for coverage reports)")

    result = run_command(cmd, "Unit Tests")
    results.append(result)

    return results


def run_integration_tests(test_pattern: str = None) -> list[dict[str, Any]]:
    """Run integration tests."""
    results = []

    cmd = ["python3", "-m", "pytest", "tests/integration/", "-v", "--tb=short"]

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    result = run_command(cmd, "Integration Tests")
    results.append(result)

    return results


def run_e2e_tests(test_pattern: str = None) -> list[dict[str, Any]]:
    """Run end-to-end tests."""
    results = []

    cmd = ["python3", "-m", "pytest", "tests/e2e/", "-v", "--tb=short"]

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    result = run_command(cmd, "End-to-End Tests")
    results.append(result)

    return results


def run_performance_tests() -> list[dict[str, Any]]:
    """Run performance tests."""
    results = []

    # Connection pooling performance test
    cmd = ["python3", "scripts/test_connection_pooling.py"]
    result = run_command(cmd, "Connection Pooling Performance Test")
    results.append(result)

    # Reindexing workflow test
    cmd = ["python3", "scripts/test_reindexing_workflow.py"]
    result = run_command(cmd, "Reindexing Workflow Test")
    results.append(result)

    return results


def run_linting() -> list[dict[str, Any]]:
    """Run linting checks."""
    results = []

    # Check if tools are available
    linting_tools = [
        (["python3", "-m", "flake8", "--version"], "flake8"),
        (["python3", "-m", "black", "--version"], "black"),
        (["python3", "-m", "isort", "--version"], "isort"),
        (["python3", "-m", "mypy", "--version"], "mypy"),
    ]

    available_tools = []
    for cmd, tool_name in linting_tools:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            available_tools.append(tool_name)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"📝 {tool_name} not available")

    # Run available linting tools
    if "flake8" in available_tools:
        cmd = ["python3", "-m", "flake8", "src/", "tests/", "--max-line-length=100"]
        result = run_command(cmd, "Flake8 Linting")
        results.append(result)

    if "black" in available_tools:
        cmd = ["python3", "-m", "black", "--check", "src/", "tests/"]
        result = run_command(cmd, "Black Code Formatting Check")
        results.append(result)

    if "isort" in available_tools:
        cmd = ["python3", "-m", "isort", "--check-only", "src/", "tests/"]
        result = run_command(cmd, "Import Sorting Check")
        results.append(result)

    if "mypy" in available_tools:
        cmd = ["python3", "-m", "mypy", "src/mcp_vector_search/"]
        result = run_command(cmd, "Type Checking")
        results.append(result)

    return results


def run_smoke_tests() -> list[dict[str, Any]]:
    """Run basic smoke tests."""
    results = []

    # Test basic imports
    cmd = ["python3", "-c", "import mcp_vector_search; print('✅ Import successful')"]
    result = run_command(cmd, "Basic Import Test")
    results.append(result)

    # Test CLI help
    cmd = ["python3", "-m", "mcp_vector_search", "--help"]
    result = run_command(cmd, "CLI Help Test")
    results.append(result)

    # Test simple functionality
    cmd = ["python3", "tests/test_simple.py"]
    result = run_command(cmd, "Simple Functionality Test")
    results.append(result)

    return results


def print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print test summary."""
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r["success"])
    failed_tests = total_tests - passed_tests
    total_time = sum(r["duration"] for r in all_results)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    print(f"Total Time: {total_time:.2f}s")

    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for result in all_results:
            if not result["success"]:
                print(f"  - {result['description']}")

    print("\n" + "=" * 60)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run MCP Vector Search tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--pattern", type=str, help="Test pattern to match")
    parser.add_argument(
        "--fast", action="store_true", help="Run fast tests only (unit + smoke)"
    )

    args = parser.parse_args()

    # If no specific test type is specified, run all
    if not any(
        [
            args.unit,
            args.integration,
            args.e2e,
            args.performance,
            args.lint,
            args.smoke,
            args.fast,
        ]
    ):
        args.all = True

    print("🧪 MCP Vector Search Test Suite")
    print("=" * 60)

    all_results = []

    # Run smoke tests first if requested
    if args.smoke or args.all or args.fast:
        results = run_smoke_tests()
        all_results.extend(results)

    # Run unit tests
    if args.unit or args.all or args.fast:
        results = run_unit_tests(args.pattern)
        all_results.extend(results)

    # Run integration tests
    if args.integration or args.all:
        results = run_integration_tests(args.pattern)
        all_results.extend(results)

    # Run E2E tests
    if args.e2e or args.all:
        results = run_e2e_tests(args.pattern)
        all_results.extend(results)

    # Run performance tests
    if args.performance or args.all:
        results = run_performance_tests()
        all_results.extend(results)

    # Run linting
    if args.lint or args.all:
        results = run_linting()
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Exit with appropriate code
    failed_tests = sum(1 for r in all_results if not r["success"])
    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()
