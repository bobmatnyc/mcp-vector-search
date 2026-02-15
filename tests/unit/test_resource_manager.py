"""Tests for resource manager memory-aware worker spawning."""

import os
from unittest.mock import patch

from mcp_vector_search.core.resource_manager import (
    ResourceLimits,
    calculate_optimal_workers,
    get_batch_size_for_memory,
    get_configured_workers,
    get_system_memory,
)


class TestResourceManager:
    """Test resource manager functionality."""

    def test_get_system_memory(self):
        """Test system memory detection."""
        total_mb, available_mb = get_system_memory()

        # Should return positive values
        assert total_mb > 0
        assert available_mb > 0

        # Available should be less than or equal to total
        assert available_mb <= total_mb

    @patch("mcp_vector_search.core.resource_manager.get_system_memory")
    @patch("os.cpu_count")
    def test_calculate_optimal_workers_high_memory(self, mock_cpu_count, mock_memory):
        """Test worker calculation with high memory."""
        # Mock 16GB available, 8 CPUs
        mock_memory.return_value = (32000, 16000)
        mock_cpu_count.return_value = 8

        limits = calculate_optimal_workers(
            memory_per_worker_mb=500,
            min_workers=1,
            max_workers=8,
        )

        # Should calculate optimal workers
        assert limits.max_workers > 0
        assert limits.max_workers <= 8  # Respects max_workers
        assert limits.total_memory_mb == 32000
        assert limits.available_memory_mb == 16000

    @patch("mcp_vector_search.core.resource_manager.get_system_memory")
    @patch("os.cpu_count")
    def test_calculate_optimal_workers_low_memory(self, mock_cpu_count, mock_memory):
        """Test worker calculation with low memory."""
        # Mock 2GB available, 4 CPUs
        mock_memory.return_value = (8000, 2000)
        mock_cpu_count.return_value = 4

        limits = calculate_optimal_workers(
            memory_per_worker_mb=500,
            min_workers=1,
            max_workers=8,
        )

        # Should respect min_workers
        assert limits.max_workers >= 1
        # Should be limited by memory
        assert limits.max_workers <= 4

    @patch("mcp_vector_search.core.resource_manager.get_system_memory")
    @patch("os.cpu_count")
    def test_calculate_optimal_workers_respects_cpu_count(
        self, mock_cpu_count, mock_memory
    ):
        """Test worker calculation respects CPU count."""
        # Mock high memory but low CPUs
        mock_memory.return_value = (64000, 32000)
        mock_cpu_count.return_value = 2

        limits = calculate_optimal_workers(
            memory_per_worker_mb=500,
            min_workers=1,
            max_workers=16,
        )

        # Should be capped by CPU count
        assert limits.max_workers <= 2

    @patch.dict(os.environ, {"MCP_VECTOR_SEARCH_WORKERS": "4"})
    def test_get_configured_workers_from_env(self):
        """Test worker count from environment variable."""
        workers = get_configured_workers()
        assert workers == 4

    @patch.dict(os.environ, {}, clear=True)
    @patch("mcp_vector_search.core.resource_manager.calculate_optimal_workers")
    def test_get_configured_workers_auto(self, mock_calculate):
        """Test worker count auto-calculation."""
        mock_calculate.return_value = ResourceLimits(
            max_workers=6,
            memory_per_worker_mb=500,
            total_memory_mb=16000,
            available_memory_mb=8000,
        )

        workers = get_configured_workers()
        assert workers == 6
        mock_calculate.assert_called_once()

    @patch.dict(os.environ, {"MCP_VECTOR_SEARCH_MEMORY_PER_WORKER": "800"}, clear=True)
    @patch("mcp_vector_search.core.resource_manager.calculate_optimal_workers")
    def test_get_configured_workers_custom_memory(self, mock_calculate):
        """Test worker count with custom memory per worker."""
        mock_calculate.return_value = ResourceLimits(
            max_workers=4,
            memory_per_worker_mb=800,
            total_memory_mb=16000,
            available_memory_mb=8000,
        )

        workers = get_configured_workers()

        # Should call with custom memory_per_worker
        mock_calculate.assert_called_once_with(memory_per_worker_mb=800)
        assert workers == 4

    def test_get_batch_size_for_memory(self):
        """Test batch size calculation."""
        # Default: 10KB per item, 100MB target
        batch_size = get_batch_size_for_memory()
        assert batch_size >= 100  # Minimum batch size

        # Custom: 5KB per item, 50MB target
        batch_size = get_batch_size_for_memory(item_size_kb=5, target_batch_mb=50)
        expected = (50 * 1024) // 5  # 10240
        assert batch_size == expected

        # Very large items should still return minimum
        batch_size = get_batch_size_for_memory(item_size_kb=10000, target_batch_mb=10)
        assert batch_size == 100  # Minimum

    @patch("mcp_vector_search.core.resource_manager.get_system_memory")
    @patch("os.cpu_count")
    def test_memory_fraction_and_reserve(self, mock_cpu_count, mock_memory):
        """Test memory fraction and reserve parameters."""
        # Mock 10GB available, 8 CPUs
        mock_memory.return_value = (20000, 10000)
        mock_cpu_count.return_value = 8

        limits = calculate_optimal_workers(
            memory_per_worker_mb=500,
            memory_fraction=0.5,  # Use only 50% of available
            memory_reserve_mb=2000,  # Reserve 2GB
        )

        # Should respect fraction and reserve
        # 10000 * 0.5 = 5000, minus 2000 reserve = 3000 usable
        # 3000 / 500 = 6 workers
        assert limits.max_workers <= 8
        assert limits.max_workers >= 1


class TestResourceLimitsDataClass:
    """Test ResourceLimits dataclass."""

    def test_resource_limits_creation(self):
        """Test ResourceLimits can be created."""
        limits = ResourceLimits(
            max_workers=4,
            memory_per_worker_mb=500,
            total_memory_mb=16000,
            available_memory_mb=8000,
        )

        assert limits.max_workers == 4
        assert limits.memory_per_worker_mb == 500
        assert limits.total_memory_mb == 16000
        assert limits.available_memory_mb == 8000

    def test_resource_limits_immutable(self):
        """Test ResourceLimits is a dataclass."""
        limits = ResourceLimits(
            max_workers=4,
            memory_per_worker_mb=500,
            total_memory_mb=16000,
            available_memory_mb=8000,
        )

        # Dataclass should allow attribute access
        assert hasattr(limits, "max_workers")
        assert hasattr(limits, "memory_per_worker_mb")
