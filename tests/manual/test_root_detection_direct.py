#!/usr/bin/env python3
"""Test root node detection logic directly from graph data"""

import json
from collections import defaultdict
from pathlib import Path


def test_root_node_detection():
    """Test that root nodes are correctly identified using link-based detection"""

    # Find the graph data file
    graph_file = Path("chunk-graph.json")
    if not graph_file.exists():
        print("Error: chunk-graph.json not found. Please run indexing first.")
        return 1

    print("Loading graph data from chunk-graph.json...")
    with open(graph_file) as f:
        data = json.load(f)

    print(f"\nTotal nodes: {len(data['nodes'])}")
    print(f"Total links: {len(data['links'])}")

    # Link-based root detection (same as JavaScript)
    target_ids = {link["target"] for link in data["links"]}
    root_nodes = [node for node in data["nodes"] if node["id"] not in target_ids]

    print(f"\n{'=' * 60}")
    print(f"ROOT NODES FOUND: {len(root_nodes)}")
    print(f"{'=' * 60}")

    # Group by type
    by_type = defaultdict(list)
    for node in root_nodes:
        by_type[node["type"]].append(node["name"])

    # Display results
    print("\nRoot Nodes by Type:")
    for node_type, names in sorted(by_type.items()):
        print(f"\n{node_type.upper()}: {len(names)} nodes")
        for name in sorted(names):
            print(f"  - {name}")

    # Count all nodes by type for comparison
    all_by_type = defaultdict(int)
    for node in data["nodes"]:
        all_by_type[node["type"]] += 1

    print(f"\n{'=' * 60}")
    print("ALL NODES BY TYPE (for comparison):")
    print(f"{'=' * 60}")
    for node_type, count in sorted(all_by_type.items()):
        print(f"{node_type}: {count}")

    # Success criteria
    print(f"\n{'=' * 60}")
    print("SUCCESS CRITERIA:")
    print(f"{'=' * 60}")

    has_directories = "directory" in by_type and len(by_type["directory"]) > 0
    has_files = "file" in by_type and len(by_type["file"]) > 0
    no_functions = "function" not in by_type or len(by_type["function"]) == 0
    no_classes = "class" not in by_type or len(by_type["class"]) == 0

    print(
        f"✓ Has directory root nodes: {has_directories} (found {len(by_type.get('directory', []))})"
    )
    print(f"✓ Has file root nodes: {has_files} (found {len(by_type.get('file', []))})")
    print(
        f"✓ No function root nodes: {no_functions} (found {len(by_type.get('function', []))})"
    )
    print(
        f"✓ No class root nodes: {no_classes} (found {len(by_type.get('class', []))})"
    )

    all_pass = has_directories and has_files and no_functions and no_classes

    print(f"\n{'=' * 60}")
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print(f"{'=' * 60}")
        print("\nThe link-based root detection correctly identifies:")
        print(f"  - {len(by_type['directory'])} top-level directories")
        print(f"  - {len(by_type['file'])} root-level files")
        print("  - 0 functions (they have parent files)")
        print("  - 0 classes (they have parent files)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"{'=' * 60}")
        return 1


if __name__ == "__main__":
    try:
        exit(test_root_node_detection())
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
