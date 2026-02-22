"""Tests for review cache system."""

import sqlite3

import pytest

from mcp_vector_search.analysis.review.cache import ReviewCache


@pytest.fixture
def temp_project_root(tmp_path):
    """Create temporary project root for cache testing."""
    return tmp_path


@pytest.fixture
def cache(temp_project_root):
    """Create ReviewCache instance."""
    return ReviewCache(temp_project_root)


def test_cache_initialization(cache, temp_project_root):
    """Test cache database initialization."""
    db_path = temp_project_root / ".mcp-vector-search" / "reviews.db"
    assert db_path.exists()

    # Verify tables exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check review_cache table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='review_cache'"
        )
        assert cursor.fetchone() is not None

        # Check cache_stats table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cache_stats'"
        )
        assert cursor.fetchone() is not None


def test_cache_miss(cache):
    """Test cache miss when no entry exists."""
    content_hash = ReviewCache.compute_hash("some code content")
    result = cache.get("test.py", content_hash, "security")
    assert result is None


def test_cache_hit(cache):
    """Test cache hit after storing findings."""
    file_path = "src/auth.py"
    content = "def authenticate(user, password): pass"
    content_hash = ReviewCache.compute_hash(content)
    review_type = "security"

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
            "cwe_id": "CWE-521",
            "code_snippet": "def authenticate(user, password): pass",
            "related_files": [],
        }
    ]

    cache.set(file_path, content_hash, review_type, findings, "gpt-4")

    # Retrieve from cache
    cached = cache.get(file_path, content_hash, review_type)

    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Weak Password Check"
    assert cached[0]["severity"] == "high"
    assert cached[0]["confidence"] == 0.95


def test_cache_different_content_hash(cache):
    """Test that changing content results in cache miss."""
    file_path = "src/auth.py"
    content1 = "def authenticate(user, password): pass"
    content2 = "def authenticate(user, password): return True"

    hash1 = ReviewCache.compute_hash(content1)
    hash2 = ReviewCache.compute_hash(content2)

    # Store findings for first version
    findings = [
        {
            "title": "Issue",
            "description": "Desc",
            "severity": "high",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Test",
            "recommendation": "Fix it",
            "confidence": 0.9,
        }
    ]
    cache.set(file_path, hash1, "security", findings, "gpt-4")

    # Try to retrieve with different hash (modified content)
    cached = cache.get(file_path, hash2, "security")
    assert cached is None  # Cache miss for different content


def test_cache_different_review_type(cache):
    """Test that different review types are stored separately."""
    file_path = "src/auth.py"
    content = "def authenticate(user, password): pass"
    content_hash = ReviewCache.compute_hash(content)

    # Store security findings
    security_findings = [
        {
            "title": "Security Issue",
            "description": "Desc",
            "severity": "high",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Security",
            "recommendation": "Fix",
            "confidence": 0.9,
        }
    ]
    cache.set(file_path, content_hash, "security", security_findings, "gpt-4")

    # Store architecture findings (same file, same content, different type)
    arch_findings = [
        {
            "title": "Architecture Issue",
            "description": "Desc",
            "severity": "medium",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Architecture",
            "recommendation": "Refactor",
            "confidence": 0.8,
        }
    ]
    cache.set(file_path, content_hash, "architecture", arch_findings, "gpt-4")

    # Retrieve each independently
    cached_security = cache.get(file_path, content_hash, "security")
    cached_arch = cache.get(file_path, content_hash, "architecture")

    assert cached_security is not None
    assert cached_arch is not None
    assert cached_security[0]["title"] == "Security Issue"
    assert cached_arch[0]["title"] == "Architecture Issue"


def test_cache_invalidate_file(cache):
    """Test invalidating all cache entries for a file."""
    file_path = "src/auth.py"
    content = "def authenticate(user, password): pass"
    content_hash = ReviewCache.compute_hash(content)

    # Store multiple review types
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

    cache.set(file_path, content_hash, "security", findings, "gpt-4")
    cache.set(file_path, content_hash, "architecture", findings, "gpt-4")

    # Verify both are cached
    assert cache.get(file_path, content_hash, "security") is not None
    assert cache.get(file_path, content_hash, "architecture") is not None

    # Invalidate file
    cache.invalidate_file(file_path)

    # Verify both are gone
    assert cache.get(file_path, content_hash, "security") is None
    assert cache.get(file_path, content_hash, "architecture") is None


def test_cache_clear_all(cache):
    """Test clearing all cache entries."""
    # Store multiple entries
    for i in range(3):
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

    # Clear all
    cleared = cache.clear()
    assert cleared == 3

    # Verify all are gone
    for i in range(3):
        file_path = f"file{i}.py"
        content_hash = ReviewCache.compute_hash(f"content{i}")
        assert cache.get(file_path, content_hash, "security") is None


def test_cache_clear_by_type(cache):
    """Test clearing cache entries by review type."""
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

    # Store different review types
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


def test_cache_stats(cache):
    """Test cache statistics."""
    # Initially empty
    stats = cache.stats()
    assert stats["total_entries"] == 0
    assert stats["size_bytes"] > 0  # DB file exists

    # Add some entries
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
    assert stats["size_bytes"] > 0
    assert "security" in stats["by_type"]
    assert stats["by_type"]["security"] == 5


def test_compute_hash_deterministic():
    """Test that hash computation is deterministic."""
    content = "def foo(): pass"
    hash1 = ReviewCache.compute_hash(content)
    hash2 = ReviewCache.compute_hash(content)
    assert hash1 == hash2


def test_compute_hash_different_content():
    """Test that different content produces different hashes."""
    hash1 = ReviewCache.compute_hash("content1")
    hash2 = ReviewCache.compute_hash("content2")
    assert hash1 != hash2


def test_cache_empty_findings(cache):
    """Test caching empty findings list."""
    file_path = "clean.py"
    content_hash = ReviewCache.compute_hash("perfect code")

    cache.set(file_path, content_hash, "security", [], "gpt-4")

    cached = cache.get(file_path, content_hash, "security")
    assert cached is not None
    assert len(cached) == 0


def test_cache_multiple_findings(cache):
    """Test caching multiple findings for same chunk."""
    file_path = "complex.py"
    content_hash = ReviewCache.compute_hash("complex code")

    findings = [
        {
            "title": f"Issue {i}",
            "description": f"Description {i}",
            "severity": "high",
            "file_path": file_path,
            "start_line": i * 10,
            "end_line": i * 10 + 5,
            "category": "Test",
            "recommendation": f"Fix {i}",
            "confidence": 0.9,
        }
        for i in range(10)
    ]

    cache.set(file_path, content_hash, "security", findings, "gpt-4")

    cached = cache.get(file_path, content_hash, "security")
    assert cached is not None
    assert len(cached) == 10


def test_cache_upsert_behavior(cache):
    """Test that caching same key replaces old value."""
    file_path = "test.py"
    content_hash = ReviewCache.compute_hash("content")

    # Store initial findings
    findings1 = [
        {
            "title": "Issue 1",
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
    cache.set(file_path, content_hash, "security", findings1, "gpt-4")

    # Store updated findings (same key)
    findings2 = [
        {
            "title": "Issue 2",
            "description": "Updated desc",
            "severity": "medium",
            "file_path": file_path,
            "start_line": 1,
            "end_line": 1,
            "category": "Test",
            "recommendation": "New fix",
            "confidence": 0.95,
        }
    ]
    cache.set(file_path, content_hash, "security", findings2, "gpt-4")

    # Should have updated value
    cached = cache.get(file_path, content_hash, "security")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Issue 2"
    assert cached[0]["severity"] == "medium"

    # Should still only be 1 entry in DB
    stats = cache.stats()
    assert stats["total_entries"] == 1
