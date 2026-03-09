"""Shared fixtures for e2e tests against the live project index."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path("/Users/masa/Projects/mcp-vector-search")
INDEX_DIR = PROJECT_ROOT / ".mcp-vector-search"


@pytest.fixture(scope="module")
def live_index_available():
    """Skip all tests if the live LanceDB index is not present."""
    if not (INDEX_DIR / "lance").exists():
        pytest.skip("Live index not available — run `mvs index` first")
    return True


@pytest.fixture(scope="module")
def live_kg_available():
    """Skip KG tests if the live Kuzu knowledge graph is not present."""
    kg_path = INDEX_DIR / "knowledge_graph" / "code_kg"
    if not kg_path.exists():
        pytest.skip("Live KG not available — run `mvs kg build` first")
    return True
