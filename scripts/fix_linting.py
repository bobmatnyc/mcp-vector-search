#!/usr/bin/env python3
"""Fix linting issues automatically."""

import subprocess
from pathlib import Path


def run_command(cmd: list[str], cwd: Path = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def main():
    """Fix linting issues."""
    project_root = Path.cwd()
    src_dir = project_root / "src" / "mcp_vector_search"

    print("🔧 Fixing linting issues...")

    # 1. Fix ruff issues automatically
    print("  📝 Running ruff format...")
    exit_code, stdout, stderr = run_command(
        ["uv", "run", "ruff", "format", str(src_dir)], cwd=project_root
    )

    if exit_code == 0:
        print("    ✅ Ruff format completed")
    else:
        print(f"    ❌ Ruff format failed: {stderr}")

    # 2. Fix ruff check issues automatically
    print("  🔍 Running ruff check --fix...")
    exit_code, stdout, stderr = run_command(
        ["uv", "run", "ruff", "check", "--fix", str(src_dir)], cwd=project_root
    )

    if exit_code == 0:
        print("    ✅ Ruff check --fix completed")
    else:
        print(f"    ⚠️  Some ruff issues remain: {stderr}")

    # 3. Check remaining issues
    print("  🔍 Checking remaining issues...")
    exit_code, stdout, stderr = run_command(
        ["uv", "run", "ruff", "check", str(src_dir)], cwd=project_root
    )

    if exit_code == 0:
        print("    ✅ All ruff issues fixed!")
    else:
        print(f"    ⚠️  {stdout.count('error')} issues remain (may need manual fixing)")
        print("    Run 'make lint' to see details")

    # 4. Run mypy check
    print("  🔍 Running mypy check...")
    exit_code, stdout, stderr = run_command(
        ["uv", "run", "mypy", str(src_dir), "--ignore-missing-imports"],
        cwd=project_root,
    )

    if exit_code == 0:
        print("    ✅ MyPy check passed")
    else:
        print(f"    ⚠️  MyPy issues found: {stderr}")

    print("\n🎉 Linting fix completed!")
    print("💡 Run 'make lint' to verify all issues are resolved")


if __name__ == "__main__":
    main()
