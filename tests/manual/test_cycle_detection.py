"""Manual test for cycle detection algorithm.

This test verifies that the three-color DFS cycle detection correctly identifies
only TRUE circular dependencies, not false positives.
"""

from dataclasses import dataclass


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    chunk_id: str
    id: str


def detect_cycles_test(chunks: list, caller_map: dict) -> list[list[str]]:
    """Test version of detect_cycles with same logic."""
    cycles_found = []
    # Three-color constants for DFS cycle detection
    white, gray, black = 0, 1, 2
    color = {chunk.chunk_id or chunk.id: white for chunk in chunks}

    def dfs(node_id: str, path: list) -> None:
        """DFS with three-color marking for accurate cycle detection."""
        if color.get(node_id, white) == black:
            # Already fully explored, no cycle here
            return

        if color.get(node_id, white) == gray:
            # Found a TRUE cycle! Node is in current path
            try:
                cycle_start = path.index(node_id)
                cycle_nodes = path[cycle_start:] + [node_id]  # Include back edge
                # Only record if cycle length > 1 (avoid self-loops unless intentional)
                if len(set(cycle_nodes)) > 1:
                    cycles_found.append(cycle_nodes)
            except ValueError:
                pass  # Node not in path (shouldn't happen)
            return

        # Mark as currently exploring
        color[node_id] = gray
        path.append(node_id)

        # Follow outgoing edges (external_callers → caller_id)
        if node_id in caller_map:
            for caller_info in caller_map[node_id]:
                caller_id = caller_info["chunk_id"]
                dfs(caller_id, path[:])  # Pass copy of path

        # Mark as fully explored
        path.pop()
        color[node_id] = black

    # Run DFS from each unvisited node
    for chunk in chunks:
        chunk_id = chunk.chunk_id or chunk.id
        if color.get(chunk_id, white) == white:
            dfs(chunk_id, [])

    return cycles_found


def test_no_cycle():
    """Test: A → B → C (no cycle)."""
    chunks = [
        MockChunk(chunk_id="A", id="A"),
        MockChunk(chunk_id="B", id="B"),
        MockChunk(chunk_id="C", id="C"),
    ]
    caller_map = {
        "A": [{"chunk_id": "B"}],
        "B": [{"chunk_id": "C"}],
    }
    cycles = detect_cycles_test(chunks, caller_map)
    assert len(cycles) == 0, f"Expected no cycles, found: {cycles}"
    print("✓ Test passed: A → B → C (no cycle)")


def test_true_cycle():
    """Test: A → B → C → A (true cycle)."""
    chunks = [
        MockChunk(chunk_id="A", id="A"),
        MockChunk(chunk_id="B", id="B"),
        MockChunk(chunk_id="C", id="C"),
    ]
    caller_map = {
        "A": [{"chunk_id": "B"}],
        "B": [{"chunk_id": "C"}],
        "C": [{"chunk_id": "A"}],
    }
    cycles = detect_cycles_test(chunks, caller_map)
    assert len(cycles) == 1, f"Expected 1 cycle, found: {len(cycles)}"
    print("✓ Test passed: A → B → C → A (true cycle)")


def test_diamond_no_cycle():
    """Test: Diamond pattern (A → B, A → C, B → D, C → D) - no cycle."""
    chunks = [
        MockChunk(chunk_id="A", id="A"),
        MockChunk(chunk_id="B", id="B"),
        MockChunk(chunk_id="C", id="C"),
        MockChunk(chunk_id="D", id="D"),
    ]
    caller_map = {
        "A": [{"chunk_id": "B"}, {"chunk_id": "C"}],
        "B": [{"chunk_id": "D"}],
        "C": [{"chunk_id": "D"}],
    }
    cycles = detect_cycles_test(chunks, caller_map)
    assert len(cycles) == 0, f"Expected no cycles (diamond pattern), found: {cycles}"
    print("✓ Test passed: Diamond pattern A → B → D, A → C → D (no cycle)")


def test_self_loop():
    """Test: A → A (self-loop cycle)."""
    chunks = [MockChunk(chunk_id="A", id="A")]
    caller_map = {
        "A": [{"chunk_id": "A"}],
    }
    cycles = detect_cycles_test(chunks, caller_map)
    # Self-loops are filtered out (len(set(cycle_nodes)) > 1 check)
    assert len(cycles) == 0, f"Expected 0 cycles (self-loop filtered), found: {cycles}"
    print("✓ Test passed: A → A (self-loop correctly filtered)")


def test_multiple_paths_no_cycle():
    """Test: Multiple paths to same node (not a cycle)."""
    chunks = [
        MockChunk(chunk_id="A", id="A"),
        MockChunk(chunk_id="B", id="B"),
        MockChunk(chunk_id="C", id="C"),
        MockChunk(chunk_id="D", id="D"),
    ]
    caller_map = {
        "A": [{"chunk_id": "C"}],
        "B": [{"chunk_id": "C"}],
        "C": [{"chunk_id": "D"}],
    }
    cycles = detect_cycles_test(chunks, caller_map)
    assert len(cycles) == 0, (
        f"Expected no cycles (multiple paths to C), found: {cycles}"
    )
    print("✓ Test passed: A → C ← B → C → D (multiple paths, no cycle)")


def test_complex_cycle():
    """Test: Complex graph with cycle A → B → C → D → B."""
    chunks = [
        MockChunk(chunk_id="A", id="A"),
        MockChunk(chunk_id="B", id="B"),
        MockChunk(chunk_id="C", id="C"),
        MockChunk(chunk_id="D", id="D"),
    ]
    caller_map = {
        "A": [{"chunk_id": "B"}],
        "B": [{"chunk_id": "C"}],
        "C": [{"chunk_id": "D"}],
        "D": [{"chunk_id": "B"}],  # Back edge creating cycle
    }
    cycles = detect_cycles_test(chunks, caller_map)
    assert len(cycles) == 1, f"Expected 1 cycle (B → C → D → B), found: {len(cycles)}"
    print("✓ Test passed: A → B → C → D → B (complex cycle)")


if __name__ == "__main__":
    print("\n🔬 Testing Cycle Detection Algorithm\n")
    test_no_cycle()
    test_true_cycle()
    test_diamond_no_cycle()
    test_self_loop()
    test_multiple_paths_no_cycle()
    test_complex_cycle()
    print("\n✅ All cycle detection tests passed!\n")
