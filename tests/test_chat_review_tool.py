"""Test review_code tool integration in chat system."""

from mcp_vector_search.cli.commands.chat import _get_tools


def test_review_code_tool_registered():
    """Test that review_code tool is properly registered."""
    tools = _get_tools()
    review_tool = next(
        (t for t in tools if t["function"]["name"] == "review_code"), None
    )

    assert review_tool is not None, "review_code tool not found in tools list"

    # Check tool structure
    assert review_tool["type"] == "function"
    func = review_tool["function"]

    # Verify description
    assert "AI-powered code review" in func["description"]
    assert "security" in func["description"]
    assert "architecture" in func["description"]
    assert "performance" in func["description"]

    # Verify parameters
    params = func["parameters"]["properties"]
    assert "review_type" in params
    assert "scope" in params
    assert "max_chunks" in params

    # Verify enum values for review_type
    assert params["review_type"]["enum"] == ["security", "architecture", "performance"]

    # Verify required fields
    assert func["parameters"]["required"] == ["review_type"]


def test_review_code_tool_in_tool_list():
    """Test review_code appears after query_knowledge_graph in tool list."""
    tools = _get_tools()
    tool_names = [t["function"]["name"] for t in tools]

    assert "review_code" in tool_names
    assert "query_knowledge_graph" in tool_names

    # Ensure review_code comes after query_knowledge_graph
    kg_index = tool_names.index("query_knowledge_graph")
    review_index = tool_names.index("review_code")
    assert review_index > kg_index
