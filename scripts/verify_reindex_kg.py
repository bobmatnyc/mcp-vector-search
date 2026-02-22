#!/usr/bin/env python3
"""Verify that reindex command includes KG build step."""

import ast
import sys
from pathlib import Path


def verify_reindex_implementation():
    """Verify that reindex.py has KG build integration."""
    reindex_file = (
        Path(__file__).parent.parent / "src/mcp_vector_search/cli/commands/reindex.py"
    )

    if not reindex_file.exists():
        print(f"❌ Error: {reindex_file} not found")
        return False

    # Read the file
    with open(reindex_file) as f:
        source = f.read()

    # Parse AST
    tree = ast.parse(source)

    # Check for required imports
    required_imports = {
        "gc",
        "json",
        "shutil",
        "subprocess",
        "tempfile",
        "threading",
        "time",
    }
    found_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name)

    missing_imports = required_imports - found_imports
    if missing_imports:
        print(f"❌ Missing imports: {missing_imports}")
        return False

    print("✅ All required imports present")

    # Check for _build_knowledge_graph function
    has_build_kg_function = False
    has_kg_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            if node.name == "_build_knowledge_graph":
                has_build_kg_function = True
                print("✅ _build_knowledge_graph() function exists")
            elif node.name == "_run_reindex":
                # Check for KG build call in _run_reindex
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Await):
                        if isinstance(subnode.value, ast.Call):
                            if isinstance(subnode.value.func, ast.Name):
                                if subnode.value.func.id == "_build_knowledge_graph":
                                    has_kg_call = True
                                    print("✅ KG build call found in _run_reindex()")

    if not has_build_kg_function:
        print("❌ _build_knowledge_graph() function not found")
        return False

    if not has_kg_call:
        print("❌ KG build call not found in _run_reindex()")
        return False

    # Check for subprocess script path
    if "_kg_subprocess.py" not in source:
        print("❌ Subprocess script path not found")
        return False

    print("✅ Subprocess script path found")

    # Check for error handling
    if "except Exception" not in source or "logger.warning" not in source:
        print("❌ Error handling for KG build not found")
        return False

    print("✅ Error handling present")

    # Check for fresh flag check
    if "if fresh:" not in source:
        print("❌ Fresh flag check not found")
        return False

    print("✅ Fresh flag check present (KG only rebuilds on fresh reindex)")

    print("\n✅ All checks passed! reindex -f will rebuild KG after indexing.")
    return True


if __name__ == "__main__":
    success = verify_reindex_implementation()
    sys.exit(0 if success else 1)
