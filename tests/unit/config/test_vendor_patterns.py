"""Tests for vendor patterns module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.config.vendor_patterns import (
    VendorPatternConverter,
    VendorPatternsManager,
)


class TestVendorPatternConverter:
    """Tests for VendorPatternConverter."""

    def test_simple_directory_pattern(self):
        """Test converting (^|/)pattern/ to glob."""
        converter = VendorPatternConverter()
        result = converter.regex_to_glob("(^|/)node_modules/")
        assert "node_modules" in result or "**/node_modules" in result

    def test_file_extension_pattern(self):
        """Test converting file extension patterns."""
        converter = VendorPatternConverter()
        result = converter.regex_to_glob(r"(^|/)jquery\.min\.js$")
        assert any("jquery.min.js" in p for p in result)

    def test_wildcard_conversion(self):
        """Test converting .* to *."""
        converter = VendorPatternConverter()
        result = converter.regex_to_glob(r"(^|/)vendor\.js$")
        assert any("vendor.js" in p for p in result)

    def test_complex_pattern_skipped(self):
        """Test that complex regex patterns are skipped."""
        converter = VendorPatternConverter()
        # Pattern with groups, alternations, etc should be skipped
        result = converter.regex_to_glob(r"(jquery|react|angular)\.min\.js$")
        # Should return empty list or minimal results
        assert len(result) <= 1


class TestVendorPatternsManager:
    """Tests for VendorPatternsManager."""

    @pytest.mark.asyncio
    async def test_download_vendor_yml(self, tmp_path):
        """Test downloading and caching vendor.yml."""
        manager = VendorPatternsManager(tmp_path)

        # Mock the HTTP response
        mock_yaml_content = """
- (^|/)node_modules/
- (^|/)vendor/
- \\.min\\.js$
"""
        mock_response = MagicMock()
        mock_response.text = mock_yaml_content
        mock_response.headers = {"etag": "test123", "last-modified": "Mon, 01 Jan 2024"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await manager.download_vendor_yml()

            assert isinstance(result, list)
            assert manager.vendor_cache_path.exists()
            assert manager.metadata_path.exists()

    def test_load_cached_vendor_yml(self, tmp_path):
        """Test loading cached vendor.yml."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake cache
        cache_content = """
# Source: test
# Downloaded: 2024-01-01

- (^|/)node_modules/
- (^|/)vendor/
"""
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        manager.vendor_cache_path.write_text(cache_content)

        result = manager.load_cached_vendor_yml()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_metadata(self, tmp_path):
        """Test retrieving metadata."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake metadata
        metadata = {
            "source_url": "https://example.com/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "test123",
        }
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        result = manager.get_metadata()
        assert result == metadata

    def test_convert_to_glob_patterns(self, tmp_path):
        """Test converting vendor.yml patterns to glob."""
        manager = VendorPatternsManager(tmp_path)

        vendor_data = [
            "(^|/)node_modules/",
            "(^|/)vendor/",
            r"\\.min\\.js$",
        ]

        result = manager.convert_to_glob_patterns(vendor_data)
        assert isinstance(result, list)
        assert len(result) > 0
        # Check that some patterns were converted
        assert any("node_modules" in p for p in result)

    @pytest.mark.asyncio
    async def test_get_vendor_patterns_with_cache(self, tmp_path):
        """Test getting vendor patterns from cache."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake cache
        cache_content = """
- (^|/)node_modules/
- (^|/)vendor/
"""
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        manager.vendor_cache_path.write_text(cache_content)

        result = await manager.get_vendor_patterns()
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_check_for_updates_no_cache(self, tmp_path):
        """Test checking for updates when no cache exists."""
        manager = VendorPatternsManager(tmp_path)

        has_updates = await manager.check_for_updates()
        # Should return True (update available) when no cache exists
        assert has_updates is True

    @pytest.mark.asyncio
    async def test_check_for_updates_with_cache(self, tmp_path):
        """Test checking for updates with existing cache."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake metadata
        metadata = {
            "source_url": "https://example.com/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "test123",
            "last_modified": "Mon, 01 Jan 2024",
        }
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Mock HTTP HEAD request
        mock_response = MagicMock()
        mock_response.headers = {
            "etag": "test123",  # Same ETag = no update
            "last-modified": "Mon, 01 Jan 2024",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response
            )

            has_updates = await manager.check_for_updates()
            # Should return False (no updates) when ETag matches
            assert has_updates is False
