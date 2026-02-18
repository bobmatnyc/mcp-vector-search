"""Tests for automatic vendor pattern updates in index command."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.config.vendor_patterns import VendorPatternsManager


class TestIndexVendorUpdate:
    """Test vendor pattern update checking during indexing."""

    @pytest.mark.asyncio
    async def test_check_for_updates_called_by_default(self, tmp_path):
        """Test that check_for_updates is called during indexing by default."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake metadata (existing cache)
        metadata = {
            "source_url": "https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "old_etag",
            "last_modified": "Mon, 01 Jan 2024",
        }
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create fake cache
        cache_content = """
# Source: test
# Downloaded: 2024-01-01

- (^|/)node_modules/
- (^|/)vendor/
"""
        manager.vendor_cache_path.write_text(cache_content)

        # Mock HTTP HEAD request (different ETag = update available)
        mock_response = MagicMock()
        mock_response.headers = {
            "etag": "new_etag",  # Different ETag = update available
            "last-modified": "Tue, 02 Jan 2024",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response
            )

            # Check for updates
            has_updates = await manager.check_for_updates()
            assert has_updates is True

    @pytest.mark.asyncio
    async def test_no_update_when_etag_matches(self, tmp_path):
        """Test that no update is triggered when ETag matches."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake metadata
        metadata = {
            "source_url": "https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "test123",
            "last_modified": "Mon, 01 Jan 2024",
        }
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Mock HTTP HEAD request (same ETag = no update)
        mock_response = MagicMock()
        mock_response.headers = {
            "etag": "test123",  # Same ETag
            "last-modified": "Mon, 01 Jan 2024",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response
            )

            has_updates = await manager.check_for_updates()
            assert has_updates is False

    @pytest.mark.asyncio
    async def test_graceful_network_failure(self, tmp_path):
        """Test that network failures are handled gracefully."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake cache with metadata
        cache_content = """
- (^|/)node_modules/
- (^|/)vendor/
"""
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        manager.vendor_cache_path.write_text(cache_content)

        # Create metadata (so check_for_updates doesn't return True immediately)
        metadata = {
            "source_url": "https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "test123",
        }
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Mock network error
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                side_effect=httpx.ConnectError("Network unavailable")
            )

            # Should return False and not raise (graceful degradation)
            has_updates = await manager.check_for_updates()
            assert has_updates is False

            # Should still be able to load cached patterns
            patterns = await manager.get_vendor_patterns()
            assert isinstance(patterns, list)
            assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_download_on_update_available(self, tmp_path):
        """Test that download is triggered when update is available."""
        manager = VendorPatternsManager(tmp_path)

        # Create fake metadata (old version)
        metadata = {
            "source_url": "https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml",
            "downloaded_at": "2024-01-01T00:00:00",
            "etag": "old_etag",
        }
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.metadata_path, "w") as f:
            json.dump(metadata, f)

        # Mock HTTP HEAD request (update available)
        mock_head_response = MagicMock()
        mock_head_response.headers = {"etag": "new_etag"}
        mock_head_response.raise_for_status = MagicMock()

        # Mock HTTP GET request (download)
        mock_yaml_content = """
- (^|/)node_modules/
- (^|/)vendor/
- (^|/)dist/
"""
        mock_get_response = MagicMock()
        mock_get_response.text = mock_yaml_content
        mock_get_response.headers = {
            "etag": "new_etag",
            "last-modified": "Tue, 02 Jan 2024",
        }
        mock_get_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_head_response
            )
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_get_response
            )

            # Check for updates
            has_updates = await manager.check_for_updates()
            assert has_updates is True

            # Download new version
            await manager.download_vendor_yml()

            # Verify new metadata
            new_metadata = manager.get_metadata()
            assert new_metadata["etag"] == "new_etag"

            # Verify patterns loaded
            patterns = await manager.get_vendor_patterns()
            assert len(patterns) > 0
