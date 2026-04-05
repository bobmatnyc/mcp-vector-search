"""Claim router: maps PolicyClaims to search QueryPlans.

Task 5: Routes each claim to a list of QueryPlans based on its category.
Strategies are loaded from YAML files in the strategies/ directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .models import PolicyClaim

_STRATEGIES_DIR = Path(__file__).parent / "strategies"

# Categories that have YAML strategy files
_KNOWN_CATEGORIES = {
    "data_sharing",
    "encryption",
    "retention",
    "user_rights",
    "third_party",
    "logging_pii",
}


@dataclass
class QueryPlan:
    """A planned search query to be executed by the evidence collector.

    Attributes:
        tool: The search tool to invoke.
        query: The query string to run.
        params: Optional extra parameters for the tool.
    """

    tool: str
    query: str
    params: dict[str, Any] = field(default_factory=dict)


def _load_strategy(category: str) -> dict[str, Any]:
    """Load and parse a strategy YAML file for the given category.

    Args:
        category: Category name matching a YAML filename (without .yaml).

    Returns:
        Parsed strategy dictionary, or empty dict if not found.
    """
    strategy_path = _STRATEGIES_DIR / f"{category}.yaml"
    if not strategy_path.exists():
        logger.debug("No strategy file for category '%s' — skipping", category)
        return {}

    try:
        raw = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse strategy file %s: %s", strategy_path, exc)
        return {}


@lru_cache(maxsize=32)
def _get_strategy(category: str) -> dict[str, Any]:
    """Cache-backed strategy loader.

    Args:
        category: Category name.

    Returns:
        Parsed strategy dict (cached after first load).
    """
    return _load_strategy(category)


def route(claim: PolicyClaim) -> list[QueryPlan]:
    """Route a claim to a list of QueryPlans based on its category.

    Each QueryPlan represents one search tool invocation. Plans are ordered
    by descending weight (highest priority first).

    For categories without a strategy file, falls back to a generic
    search_hybrid query using the claim's normalized text.

    Args:
        claim: The PolicyClaim to route.

    Returns:
        List of QueryPlan objects to execute.
    """
    strategy = _get_strategy(claim.category)

    if not strategy:
        # Generic fallback for categories without a strategy YAML
        logger.debug(
            "No strategy for category '%s' — using generic fallback", claim.category
        )
        return [
            QueryPlan(
                tool="search_hybrid",
                query=claim.normalized,
                params={"limit": 10},
            ),
            QueryPlan(
                tool="search_code",
                query=" ".join(claim.keywords[:5])
                if claim.keywords
                else claim.normalized,
                params={"limit": 10},
            ),
        ]

    queries = strategy.get("queries", [])
    keyword_interp = strategy.get("keyword_interpolation", {})
    use_keyword_interp = keyword_interp.get("enabled", False)
    kw_template = keyword_interp.get("template", "{keywords}")

    plans: list[tuple[float, QueryPlan]] = []

    for q in queries:
        tool = q.get("tool", "search_hybrid")
        query_str = q.get("query", "")
        weight = float(q.get("weight", 1.0))
        plans.append(
            (weight, QueryPlan(tool=tool, query=query_str, params={"limit": 10}))
        )

    # Add a keyword-interpolated query if enabled and claim has keywords
    if use_keyword_interp and claim.keywords:
        kw_string = " ".join(claim.keywords[:8])
        interpolated_query = kw_template.replace("{keywords}", kw_string)
        plans.append(
            (
                0.5,
                QueryPlan(
                    tool="search_hybrid", query=interpolated_query, params={"limit": 10}
                ),
            )
        )

    # Sort by weight descending, strip weights
    plans.sort(key=lambda t: t[0], reverse=True)
    return [plan for _, plan in plans]
