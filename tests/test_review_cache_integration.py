"""Integration tests for review cache with ReviewEngine.

These tests verify cache behavior at the engine level without requiring
full LLM integration (which has prompt formatting complexity).
"""

from mcp_vector_search.analysis.review.cache import ReviewCache


def test_cache_workflow(tmp_path):
    """Test basic cache workflow: store, retrieve, invalidate."""
    cache = ReviewCache(tmp_path)

    # Simulate reviewing a file
    file_path = "src/auth.py"
    content = "def authenticate(user, password): pass"
    content_hash = ReviewCache.compute_hash(content)
    review_type = "security"

    # First access (miss)
    cached = cache.get(file_path, content_hash, review_type)
    assert cached is None

    # Store findings
    findings = [
        {
            "title": "Weak Password Check",
            "description": "Password validation is too weak",
            "severity": "high",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Authentication",
            "recommendation": "Use bcrypt for password hashing",
            "confidence": 0.95,
        }
    ]
    cache.set(file_path, content_hash, review_type, findings, "gpt-4")

    # Second access (hit)
    cached = cache.get(file_path, content_hash, review_type)
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Weak Password Check"


def test_cache_stats_workflow(tmp_path):
    """Test cache statistics over multiple operations."""
    cache = ReviewCache(tmp_path)

    # Initially empty
    stats = cache.stats()
    assert stats["total_entries"] == 0

    # Add entries for multiple files
    for i in range(5):
        file_path = f"file{i}.py"
        content_hash = ReviewCache.compute_hash(f"content{i}")
        findings = [
            {
                "title": f"Issue {i}",
                "description": "Desc",
                "severity": "high",
                "file_path": file_path,
                "start_line": 1,
                "end_line": 1,
                "category": "Test",
                "recommendation": "Fix",
                "confidence": 0.9,
            }
        ]
        cache.set(file_path, content_hash, "security", findings, "gpt-4")

    # Check stats
    stats = cache.stats()
    assert stats["total_entries"] == 5
    assert stats["by_type"]["security"] == 5

    # Clear and verify
    cleared = cache.clear()
    assert cleared == 5

    stats = cache.stats()
    assert stats["total_entries"] == 0


def test_cache_content_change_detection(tmp_path):
    """Test that content changes are detected via hash."""
    cache = ReviewCache(tmp_path)

    file_path = "src/auth.py"
    content_v1 = "def authenticate(user, password): pass"
    content_v2 = "def authenticate(user, password): return True"

    hash_v1 = ReviewCache.compute_hash(content_v1)
    hash_v2 = ReviewCache.compute_hash(content_v2)

    # Hashes should be different
    assert hash_v1 != hash_v2

    # Store findings for v1
    findings = [
        {
            "title": "Issue",
            "description": "Desc",
            "severity": "high",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Test",
            "recommendation": "Fix",
            "confidence": 0.9,
        }
    ]
    cache.set(file_path, hash_v1, "security", findings, "gpt-4")

    # Query with v1 hash (hit)
    assert cache.get(file_path, hash_v1, "security") is not None

    # Query with v2 hash (miss - content changed)
    assert cache.get(file_path, hash_v2, "security") is None


def test_cache_clear_by_review_type(tmp_path):
    """Test selective cache clearing by review type."""
    cache = ReviewCache(tmp_path)

    file_path = "test.py"
    content_hash = ReviewCache.compute_hash("content")

    findings = [
        {
            "title": "Issue",
            "description": "Desc",
            "severity": "high",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Test",
            "recommendation": "Fix",
            "confidence": 0.9,
        }
    ]

    # Store multiple review types
    cache.set(file_path, content_hash, "security", findings, "gpt-4")
    cache.set(file_path, content_hash, "architecture", findings, "gpt-4")
    cache.set(file_path, content_hash, "performance", findings, "gpt-4")

    # Clear only security reviews
    cleared = cache.clear(review_type="security")
    assert cleared == 1

    # Verify security is gone, others remain
    assert cache.get(file_path, content_hash, "security") is None
    assert cache.get(file_path, content_hash, "architecture") is not None
    assert cache.get(file_path, content_hash, "performance") is not None
