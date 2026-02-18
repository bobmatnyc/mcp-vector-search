"""Memory monitoring and enforcement for indexer operations.

Provides memory cap enforcement with configurable limits and warning thresholds.
"""

import os
from collections.abc import Callable

import psutil
from loguru import logger


class MemoryMonitor:
    """Monitor and enforce memory limits during indexing operations.

    Tracks current process memory usage and enforces a configurable cap with
    warning thresholds at 80% and 90%.
    """

    def __init__(
        self,
        max_memory_gb: float | None = None,
        warn_threshold_pct: float = 0.8,
        critical_threshold_pct: float = 0.9,
    ):
        """Initialize memory monitor.

        Args:
            max_memory_gb: Maximum memory in GB (default: 25GB or from env var)
            warn_threshold_pct: Warning threshold as fraction (default: 0.8 = 80%)
            critical_threshold_pct: Critical threshold as fraction (default: 0.9 = 90%)

        Environment Variables:
            MCP_VECTOR_SEARCH_MAX_MEMORY_GB: Override max memory limit in GB
        """
        # Get max memory from env var or use default
        env_max_memory = os.environ.get("MCP_VECTOR_SEARCH_MAX_MEMORY_GB")
        if env_max_memory:
            try:
                max_memory_gb = float(env_max_memory)
                logger.info(
                    f"Memory cap from environment: {max_memory_gb:.1f}GB (MCP_VECTOR_SEARCH_MAX_MEMORY_GB)"
                )
            except ValueError:
                logger.warning(
                    f"Invalid MCP_VECTOR_SEARCH_MAX_MEMORY_GB value: {env_max_memory}, using default"
                )

        # Default to 25GB if not specified
        if max_memory_gb is None:
            max_memory_gb = 25.0

        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.max_memory_gb = max_memory_gb
        self.warn_threshold = warn_threshold_pct
        self.critical_threshold = critical_threshold_pct

        # Track warning state to avoid log spam
        self._warned = False
        self._critical_warned = False

        # Get process handle once
        self._process = psutil.Process()

        logger.info(
            f"Memory monitor initialized: {max_memory_gb:.1f}GB cap "
            f"(warn: {warn_threshold_pct * 100:.0f}%, critical: {critical_threshold_pct * 100:.0f}%)"
        )

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        mem_info = self._process.memory_info()
        return mem_info.rss / (1024 * 1024)

    def get_current_memory_gb(self) -> float:
        """Get current process memory usage in GB.

        Returns:
            Memory usage in gigabytes
        """
        return self.get_current_memory_mb() / 1024

    def get_memory_usage_pct(self) -> float:
        """Get current memory usage as percentage of cap.

        Returns:
            Memory usage as fraction (0.0 to 1.0+)
        """
        current_bytes = self._process.memory_info().rss
        return current_bytes / self.max_memory_bytes

    def check_memory_limit(self) -> tuple[bool, float, str]:
        """Check if memory usage is within limits.

        Returns:
            Tuple of (is_ok, usage_pct, message)
            - is_ok: True if under critical threshold
            - usage_pct: Current usage as percentage (0.0 to 1.0+)
            - message: Status message with recommendations
        """
        usage_pct = self.get_memory_usage_pct()
        current_gb = self.get_current_memory_gb()

        if usage_pct >= 1.0:
            # At or over limit
            msg = (
                f"⚠️  Memory limit exceeded: {current_gb:.2f}GB / {self.max_memory_gb:.1f}GB ({usage_pct * 100:.1f}%). "
                "Processing will slow down to prevent OOM crash."
            )
            logger.error(msg)
            return False, usage_pct, msg

        elif usage_pct >= self.critical_threshold:
            # Critical threshold (90%)
            if not self._critical_warned:
                msg = (
                    f"⚠️  Memory usage critical: {current_gb:.2f}GB / {self.max_memory_gb:.1f}GB ({usage_pct * 100:.1f}%). "
                    "Approaching memory limit."
                )
                logger.warning(msg)
                self._critical_warned = True
            return True, usage_pct, "critical"

        elif usage_pct >= self.warn_threshold:
            # Warning threshold (80%)
            if not self._warned:
                msg = (
                    f"Memory usage high: {current_gb:.2f}GB / {self.max_memory_gb:.1f}GB ({usage_pct * 100:.1f}%). "
                    "Consider reducing batch size if this persists."
                )
                logger.warning(msg)
                self._warned = True
            return True, usage_pct, "warning"

        else:
            # Normal operation - reset warning flags
            self._warned = False
            self._critical_warned = False
            return True, usage_pct, "ok"

    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced due to memory pressure.

        Returns:
            True if memory usage is at or above critical threshold
        """
        usage_pct = self.get_memory_usage_pct()
        return usage_pct >= self.critical_threshold

    def wait_for_memory_available(
        self, target_pct: float = 0.8, poll_interval_sec: float = 1.0
    ) -> None:
        """Block until memory usage drops below target threshold.

        Used when memory limit is exceeded to apply backpressure.

        Args:
            target_pct: Target memory usage to wait for (default: 0.8 = 80%)
            poll_interval_sec: Polling interval in seconds (default: 1.0)
        """
        import time

        usage_pct = self.get_memory_usage_pct()
        if usage_pct < target_pct:
            return  # Already under target

        logger.info(
            f"Memory usage at {usage_pct * 100:.1f}%, waiting for it to drop below {target_pct * 100:.0f}%..."
        )

        while usage_pct >= target_pct:
            time.sleep(poll_interval_sec)
            usage_pct = self.get_memory_usage_pct()

        logger.info(
            f"Memory usage dropped to {usage_pct * 100:.1f}%, resuming processing"
        )

    def get_adjusted_batch_size(
        self, current_batch_size: int, min_batch_size: int = 1
    ) -> int:
        """Calculate adjusted batch size based on current memory usage.

        Reduces batch size when memory pressure is high.

        Args:
            current_batch_size: Current batch size
            min_batch_size: Minimum allowed batch size (default: 1)

        Returns:
            Adjusted batch size (reduced if under memory pressure)
        """
        usage_pct = self.get_memory_usage_pct()

        if usage_pct >= 1.0:
            # At limit - reduce to minimum
            return min_batch_size
        elif usage_pct >= self.critical_threshold:
            # Critical - reduce by 75%
            return max(min_batch_size, current_batch_size // 4)
        elif usage_pct >= self.warn_threshold:
            # Warning - reduce by 50%
            return max(min_batch_size, current_batch_size // 2)
        else:
            # Normal - no reduction
            return current_batch_size

    def log_memory_summary(self) -> None:
        """Log current memory usage summary."""
        current_gb = self.get_current_memory_gb()
        usage_pct = self.get_memory_usage_pct()
        logger.info(
            f"Memory usage: {current_gb:.2f}GB / {self.max_memory_gb:.1f}GB ({usage_pct * 100:.1f}%)"
        )

    def with_memory_check(
        self, operation: Callable[[], None], operation_name: str = "operation"
    ) -> None:
        """Execute operation with memory limit enforcement.

        Applies backpressure if memory limit is exceeded.

        Args:
            operation: Callable to execute
            operation_name: Name for logging
        """
        # Check before operation
        is_ok, usage_pct, status = self.check_memory_limit()

        if not is_ok:
            logger.warning(
                f"Memory limit exceeded before {operation_name}, applying backpressure"
            )
            self.wait_for_memory_available(target_pct=self.warn_threshold)

        # Execute operation
        operation()

        # Check after operation
        self.check_memory_limit()
