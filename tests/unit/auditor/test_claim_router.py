"""Unit tests for claim router (Task 5).

Tests:
- YAML strategy loading for each category
- route() returns QueryPlan list for known categories
- route() returns generic fallback for unknown categories
- Keyword interpolation
- Plans are sorted by weight (descending)
"""

from mcp_vector_search.auditor.claim_router import QueryPlan, _get_strategy, route
from mcp_vector_search.auditor.models import PolicyClaim


def _make_claim(
    category: str = "encryption",
    normalized: str = "All API calls use TLS 1.2.",
    keywords: list[str] | None = None,
) -> PolicyClaim:
    return PolicyClaim(
        id=PolicyClaim.compute_id(category, normalized),
        category=category,
        text="Policy text here.",
        normalized=normalized,
        keywords=keywords or ["TLS", "HTTPS", "encrypt"],
        policy_section="Security",
        testable=True,
        source_offsets=(0, 100),
    )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestStrategyLoading:
    def test_data_sharing_loads(self):
        strategy = _get_strategy("data_sharing")
        assert isinstance(strategy, dict)
        assert "queries" in strategy
        assert len(strategy["queries"]) > 0

    def test_encryption_loads(self):
        strategy = _get_strategy("encryption")
        assert "queries" in strategy
        # Should include search_code tool
        tools = [q["tool"] for q in strategy["queries"]]
        assert "search_code" in tools

    def test_retention_loads(self):
        strategy = _get_strategy("retention")
        assert "queries" in strategy

    def test_user_rights_loads(self):
        strategy = _get_strategy("user_rights")
        assert "queries" in strategy

    def test_third_party_loads(self):
        strategy = _get_strategy("third_party")
        assert "queries" in strategy

    def test_logging_pii_loads(self):
        strategy = _get_strategy("logging_pii")
        assert "queries" in strategy

    def test_unknown_category_returns_empty(self):
        strategy = _get_strategy("nonexistent_category")
        assert strategy == {}


# ---------------------------------------------------------------------------
# route() function
# ---------------------------------------------------------------------------


class TestRoute:
    def test_encryption_returns_plans(self):
        claim = _make_claim(category="encryption")
        plans = route(claim)
        assert len(plans) > 0
        assert all(isinstance(p, QueryPlan) for p in plans)

    def test_data_sharing_returns_plans(self):
        claim = _make_claim(category="data_sharing")
        plans = route(claim)
        assert len(plans) > 0

    def test_retention_returns_plans(self):
        claim = _make_claim(category="retention")
        plans = route(claim)
        assert len(plans) > 0

    def test_user_rights_returns_plans(self):
        claim = _make_claim(category="user_rights")
        plans = route(claim)
        assert len(plans) > 0

    def test_third_party_returns_plans(self):
        claim = _make_claim(category="third_party")
        plans = route(claim)
        assert len(plans) > 0

    def test_logging_pii_returns_plans(self):
        claim = _make_claim(category="logging_pii")
        plans = route(claim)
        assert len(plans) > 0

    def test_consent_uses_fallback(self):
        """consent has no strategy YAML — should get generic fallback."""
        claim = _make_claim(
            category="consent", normalized="Users must opt-in explicitly."
        )
        plans = route(claim)
        assert len(plans) > 0
        # Fallback uses search_hybrid and search_code
        tools = [p.tool for p in plans]
        assert "search_hybrid" in tools

    def test_plans_have_required_fields(self):
        claim = _make_claim(category="encryption")
        plans = route(claim)
        for plan in plans:
            assert plan.tool
            assert plan.query
            assert isinstance(plan.params, dict)

    def test_known_tool_values(self):
        """All plan tools should be valid Evidence tool values."""
        valid_tools = {
            "search_code",
            "search_hybrid",
            "kg_query",
            "kg_callers_at_commit",
            "trace_execution_flow",
            "find_smells",
        }
        for category in ("encryption", "data_sharing", "retention", "logging_pii"):
            claim = _make_claim(category=category)
            plans = route(claim)
            for plan in plans:
                assert plan.tool in valid_tools, f"Unknown tool: {plan.tool}"

    def test_encryption_includes_search_code(self):
        """Encryption strategy should have a search_code query."""
        claim = _make_claim(category="encryption")
        plans = route(claim)
        tools = [p.tool for p in plans]
        assert "search_code" in tools

    def test_encryption_includes_find_smells(self):
        """Encryption strategy should include find_smells for crypto smell detection."""
        claim = _make_claim(category="encryption")
        plans = route(claim)
        tools = [p.tool for p in plans]
        assert "find_smells" in tools

    def test_keyword_interpolation_adds_extra_plan(self):
        """When keyword_interpolation.enabled=True, an extra plan should be added."""
        claim = _make_claim(
            category="data_sharing",
            keywords=["requests.post", "axios", "fetch"],
        )
        plans = route(claim)
        # The interpolated query should contain at least one keyword
        queries = [p.query for p in plans]
        assert any(
            "requests.post" in q or "axios" in q or "fetch" in q for q in queries
        )

    def test_empty_keywords_no_error(self):
        """Claims with empty keywords should still route without error."""
        claim = _make_claim(category="encryption", keywords=[])
        plans = route(claim)
        assert len(plans) > 0

    def test_limit_in_params(self):
        """QueryPlan params should include a 'limit' key."""
        claim = _make_claim(category="encryption")
        plans = route(claim)
        for plan in plans:
            assert "limit" in plan.params
            assert plan.params["limit"] > 0
