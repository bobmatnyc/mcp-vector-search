"""Unit tests for memory monitoring."""

from unittest.mock import MagicMock, patch

from mcp_vector_search.core.memory_monitor import MemoryMonitor


class TestMemoryMonitor:
    """Test memory monitor functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        monitor = MemoryMonitor()
        assert monitor.max_memory_gb == 25.0
        assert monitor.max_memory_bytes == 25 * 1024 * 1024 * 1024
        assert monitor.warn_threshold == 0.8
        assert monitor.critical_threshold == 0.9

    def test_initialization_custom(self):
        """Test custom initialization."""
        monitor = MemoryMonitor(max_memory_gb=10.0, warn_threshold_pct=0.7)
        assert monitor.max_memory_gb == 10.0
        assert monitor.max_memory_bytes == 10 * 1024 * 1024 * 1024
        assert monitor.warn_threshold == 0.7
        assert monitor.critical_threshold == 0.9

    def test_initialization_from_env(self, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("MCP_VECTOR_SEARCH_MAX_MEMORY_GB", "15.5")
        monitor = MemoryMonitor()
        assert monitor.max_memory_gb == 15.5

    def test_initialization_invalid_env(self, monkeypatch):
        """Test initialization with invalid environment variable."""
        monkeypatch.setenv("MCP_VECTOR_SEARCH_MAX_MEMORY_GB", "invalid")
        monitor = MemoryMonitor()
        assert monitor.max_memory_gb == 25.0  # Falls back to default

    @patch("psutil.Process")
    def test_get_current_memory_mb(self, mock_process):
        """Test getting current memory in MB."""
        # Mock memory info: 500MB
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 500 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor()
        assert monitor.get_current_memory_mb() == 500.0

    @patch("psutil.Process")
    def test_get_current_memory_gb(self, mock_process):
        """Test getting current memory in GB."""
        # Mock memory info: 2GB
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 2 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor()
        assert monitor.get_current_memory_gb() == 2.0

    @patch("psutil.Process")
    def test_get_memory_usage_pct(self, mock_process):
        """Test getting memory usage percentage."""
        # Mock memory info: 10GB (40% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 10 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        usage_pct = monitor.get_memory_usage_pct()
        assert abs(usage_pct - 0.4) < 0.01  # ~40%

    @patch("psutil.Process")
    def test_check_memory_limit_ok(self, mock_process):
        """Test check_memory_limit when usage is normal."""
        # Mock memory info: 10GB (40% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 10 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        is_ok, usage_pct, status = monitor.check_memory_limit()

        assert is_ok is True
        assert abs(usage_pct - 0.4) < 0.01
        assert status == "ok"

    @patch("psutil.Process")
    def test_check_memory_limit_warning(self, mock_process):
        """Test check_memory_limit when usage is at warning threshold."""
        # Mock memory info: 21GB (84% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 21 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        is_ok, usage_pct, status = monitor.check_memory_limit()

        assert is_ok is True
        assert abs(usage_pct - 0.84) < 0.01
        assert status == "warning"

    @patch("psutil.Process")
    def test_check_memory_limit_critical(self, mock_process):
        """Test check_memory_limit when usage is at critical threshold."""
        # Mock memory info: 23GB (92% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 23 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        is_ok, usage_pct, status = monitor.check_memory_limit()

        assert is_ok is True
        assert abs(usage_pct - 0.92) < 0.01
        assert status == "critical"

    @patch("psutil.Process")
    def test_check_memory_limit_exceeded(self, mock_process):
        """Test check_memory_limit when usage exceeds limit."""
        # Mock memory info: 26GB (104% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 26 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        is_ok, usage_pct, status = monitor.check_memory_limit()

        assert is_ok is False
        assert usage_pct > 1.0
        assert "Memory limit exceeded" in status

    @patch("psutil.Process")
    def test_should_reduce_batch_size_normal(self, mock_process):
        """Test should_reduce_batch_size with normal usage."""
        # Mock memory info: 10GB (40% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 10 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        assert monitor.should_reduce_batch_size() is False

    @patch("psutil.Process")
    def test_should_reduce_batch_size_critical(self, mock_process):
        """Test should_reduce_batch_size with critical usage."""
        # Mock memory info: 23GB (92% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 23 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        assert monitor.should_reduce_batch_size() is True

    @patch("psutil.Process")
    def test_get_adjusted_batch_size_normal(self, mock_process):
        """Test get_adjusted_batch_size with normal usage."""
        # Mock memory info: 10GB (40% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 10 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        assert monitor.get_adjusted_batch_size(1000) == 1000  # No reduction

    @patch("psutil.Process")
    def test_get_adjusted_batch_size_warning(self, mock_process):
        """Test get_adjusted_batch_size with warning usage."""
        # Mock memory info: 21GB (84% of 25GB cap) - hits 80%+ threshold
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 21 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        # At 80%+ usage, batch size reduced to 10% (current_batch_size // 10)
        assert monitor.get_adjusted_batch_size(1000) == 100  # 90% reduction

    @patch("psutil.Process")
    def test_get_adjusted_batch_size_critical(self, mock_process):
        """Test get_adjusted_batch_size with critical usage."""
        # Mock memory info: 23GB (92% of 25GB cap) - hits 90%+ threshold
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 23 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        # At 90%+ usage, batch size reduced to 1% (current_batch_size // 100)
        assert monitor.get_adjusted_batch_size(1000) == 10  # 99% reduction

    @patch("psutil.Process")
    def test_get_adjusted_batch_size_exceeded(self, mock_process):
        """Test get_adjusted_batch_size when limit exceeded."""
        # Mock memory info: 26GB (104% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 26 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        assert monitor.get_adjusted_batch_size(1000, min_batch_size=10) == 10  # Min

    @patch("psutil.Process")
    def test_get_adjusted_batch_size_respects_min(self, mock_process):
        """Test get_adjusted_batch_size respects minimum."""
        # Mock memory info: 23GB (92% of 25GB cap)
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 23 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        assert monitor.get_adjusted_batch_size(100, min_batch_size=50) == 50

    @patch("psutil.Process")
    def test_log_memory_summary(self, mock_process):
        """Test logging memory summary."""
        # Mock memory info: 10GB
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 10 * 1024 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_mem_info

        monitor = MemoryMonitor(max_memory_gb=25.0)
        # Just verify it doesn't crash - log output is handled by loguru
        monitor.log_memory_summary()
