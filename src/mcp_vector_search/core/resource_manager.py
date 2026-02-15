"""Resource management for memory-aware worker spawning."""

import os
from dataclasses import dataclass

import psutil
from loguru import logger


@dataclass
class ResourceLimits:
    """Resource limits for worker spawning."""

    max_workers: int
    memory_per_worker_mb: int
    total_memory_mb: int
    available_memory_mb: int


def get_system_memory() -> tuple[int, int]:
    """Get total and available system memory in MB.

    Returns:
        Tuple of (total_mb, available_mb)
    """
    mem = psutil.virtual_memory()
    total_mb = mem.total // (1024 * 1024)
    available_mb = mem.available // (1024 * 1024)
    return total_mb, available_mb


def calculate_optimal_workers(
    memory_per_worker_mb: int = 500,
    min_workers: int = 1,
    max_workers: int = 8,
    memory_reserve_mb: int = 1000,
    memory_fraction: float = 0.7,
) -> ResourceLimits:
    """Calculate optimal number of workers based on available memory.

    Args:
        memory_per_worker_mb: Memory budget per worker (default: 500MB)
        min_workers: Minimum workers regardless of memory (default: 1)
        max_workers: Maximum workers regardless of memory (default: 8)
        memory_reserve_mb: Memory to reserve for OS/other processes (default: 1GB)
        memory_fraction: Max fraction of available memory to use (default: 0.7)

    Returns:
        ResourceLimits with calculated values
    """
    total_mb, available_mb = get_system_memory()

    # Calculate usable memory
    usable_mb = int(available_mb * memory_fraction) - memory_reserve_mb
    usable_mb = max(usable_mb, memory_per_worker_mb)  # At least one worker's worth

    # Calculate optimal workers
    optimal = usable_mb // memory_per_worker_mb
    optimal = max(min_workers, min(optimal, max_workers))

    # Also check CPU cores
    cpu_count = os.cpu_count() or 4
    optimal = min(optimal, cpu_count)

    limits = ResourceLimits(
        max_workers=optimal,
        memory_per_worker_mb=memory_per_worker_mb,
        total_memory_mb=total_mb,
        available_memory_mb=available_mb,
    )

    logger.info(
        f"Resource limits: {optimal} workers "
        f"({available_mb}MB available, {memory_per_worker_mb}MB per worker)"
    )

    return limits


def get_configured_workers() -> int:
    """Get worker count from config or calculate automatically.

    Environment variables:
        MCP_VECTOR_SEARCH_WORKERS: Override worker count
        MCP_VECTOR_SEARCH_MEMORY_PER_WORKER: Memory per worker in MB
    """
    # Check for explicit override
    override = os.environ.get("MCP_VECTOR_SEARCH_WORKERS")
    if override:
        return int(override)

    # Calculate based on memory
    memory_per_worker = int(
        os.environ.get("MCP_VECTOR_SEARCH_MEMORY_PER_WORKER", "500")
    )
    limits = calculate_optimal_workers(memory_per_worker_mb=memory_per_worker)
    return limits.max_workers


def get_batch_size_for_memory(
    item_size_kb: int = 10, target_batch_mb: int = 100
) -> int:
    """Calculate batch size that fits in target memory.

    Args:
        item_size_kb: Estimated size per item in KB
        target_batch_mb: Target batch memory in MB

    Returns:
        Optimal batch size
    """
    return max(100, (target_batch_mb * 1024) // item_size_kb)
