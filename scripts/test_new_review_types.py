#!/usr/bin/env python3
"""Quick test script to verify new review types work correctly."""

from mcp_vector_search.analysis.review.engine import REVIEW_SEARCH_QUERIES
from mcp_vector_search.analysis.review.models import ReviewType
from mcp_vector_search.analysis.review.prompts import REVIEW_PROMPTS, get_review_prompt


def test_review_types():
    """Test that all review types are properly configured."""
    print("Testing new review types...\n")

    # Expected review types
    expected_types = [
        "security",
        "architecture",
        "performance",
        "quality",
        "testing",
        "documentation",
    ]

    # Test 1: ReviewType enum has all types
    actual_types = [rt.value for rt in ReviewType]
    print(f"✓ ReviewType enum has {len(actual_types)} types: {actual_types}")
    assert set(actual_types) == set(expected_types), (
        f"Missing types: {set(expected_types) - set(actual_types)}"
    )

    # Test 2: All prompts are registered
    prompt_types = [rt.value for rt in REVIEW_PROMPTS.keys()]
    print(f"✓ Prompts registered for {len(prompt_types)} types: {prompt_types}")
    assert set(prompt_types) == set(expected_types), (
        f"Missing prompts: {set(expected_types) - set(prompt_types)}"
    )

    # Test 3: All search queries are registered
    query_types = [rt.value for rt in REVIEW_SEARCH_QUERIES.keys()]
    print(f"✓ Search queries registered for {len(query_types)} types: {query_types}")
    assert set(query_types) == set(expected_types), (
        f"Missing queries: {set(expected_types) - set(query_types)}"
    )

    # Test 4: Verify each prompt is properly formatted
    for review_type in ReviewType:
        prompt = get_review_prompt(review_type)

        # Check prompt has required placeholders
        assert "{code_context}" in prompt, (
            f"{review_type.value} prompt missing {{code_context}}"
        )
        assert "{kg_relationships}" in prompt, (
            f"{review_type.value} prompt missing {{kg_relationships}}"
        )

        # Check prompt has severity criteria
        assert "Severity Criteria" in prompt or "severity" in prompt.lower(), (
            f"{review_type.value} prompt missing severity info"
        )

        # Check prompt requests JSON output
        assert "JSON" in prompt or "json" in prompt, (
            f"{review_type.value} prompt doesn't specify JSON output"
        )

        print(
            f"  ✓ {review_type.value}: prompt formatted correctly ({len(prompt)} chars)"
        )

    # Test 5: Verify search queries are sensible
    for review_type, queries in REVIEW_SEARCH_QUERIES.items():
        assert len(queries) > 0, f"{review_type.value} has no search queries"
        assert all(isinstance(q, str) and len(q) > 0 for q in queries), (
            f"{review_type.value} has invalid queries"
        )
        print(f"  ✓ {review_type.value}: {len(queries)} search queries")

    print("\n✅ All tests passed!")
    print("\nNew review types:")
    print("  - quality: Code quality and maintainability")
    print("  - testing: Test coverage and quality")
    print("  - documentation: Documentation quality")


if __name__ == "__main__":
    test_review_types()
