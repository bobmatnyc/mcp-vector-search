"""AI-powered code review system for MCP Vector Search.

This package provides specialized code review capabilities using vector search,
knowledge graphs, and LLM analysis.

Key Components:
- ReviewEngine: Orchestrates review pipeline (search → KG → LLM analysis)
- ReviewType: Security, Architecture, Performance review types
- ReviewResult: Structured review output with findings
- ReviewFinding: Individual finding with severity, category, and recommendations

Usage:
    from mcp_vector_search.analysis.review import ReviewEngine, ReviewType

    engine = ReviewEngine(search_engine, knowledge_graph, llm_client, project_root)
    result = await engine.run_review(ReviewType.SECURITY, scope="src/auth")

    for finding in result.findings:
        print(f"{finding.severity}: {finding.title}")
"""

from .engine import ReviewEngine
from .models import ReviewFinding, ReviewResult, ReviewType, Severity
from .prompts import get_review_prompt

__all__ = [
    "ReviewEngine",
    "ReviewType",
    "ReviewResult",
    "ReviewFinding",
    "Severity",
    "get_review_prompt",
]
