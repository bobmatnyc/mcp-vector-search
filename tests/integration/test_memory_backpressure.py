"""Integration test to verify memory backpressure doesn't hang the event loop.

This test was added to prevent regression of the 2.5.18 hang bug where blocking
time.sleep() in memory_monitor.wait_for_memory_available() caused event loop deadlock.
"""

import asyncio
import time

import pytest

from mcp_vector_search.core.memory_monitor import MemoryMonitor


@pytest.mark.asyncio
async def test_wait_for_memory_available_is_async():
    """Verify that wait_for_memory_available is an async function."""
    monitor = MemoryMonitor(max_memory_gb=1.0)

    # Should be an async function
    assert asyncio.iscoroutinefunction(monitor.wait_for_memory_available)


@pytest.mark.asyncio
async def test_memory_backpressure_does_not_block_event_loop():
    """Verify that memory backpressure doesn't block the event loop.

    This test simulates memory pressure and verifies that:
    1. Other coroutines can run concurrently
    2. The event loop remains responsive
    3. wait_for_memory_available doesn't use blocking time.sleep()
    """
    # Set very low memory cap to trigger backpressure quickly
    monitor = MemoryMonitor(max_memory_gb=0.001)  # 1MB cap

    # Track if other coroutines can run
    other_task_completed = asyncio.Event()

    async def other_task():
        """This task should complete even while memory is being waited on."""
        await asyncio.sleep(0.1)  # Simulate some work
        other_task_completed.set()

    # Start other task
    task = asyncio.create_task(other_task())

    # This should trigger immediately since we have >1MB memory
    # The important thing is that it uses asyncio.sleep() not time.sleep()
    start_time = time.time()

    try:
        # Wait for memory to drop below impossible threshold
        # This should time out, but the event loop should remain responsive
        await asyncio.wait_for(
            monitor.wait_for_memory_available(target_pct=0.0001),  # ~100KB threshold
            timeout=0.5,  # Short timeout since this will never succeed
        )
    except TimeoutError:
        # Expected - we set an impossible threshold
        pass

    elapsed = time.time() - start_time

    # Verify event loop was responsive (other task completed)
    assert other_task_completed.is_set(), (
        "Event loop was blocked - other tasks couldn't run. "
        "This indicates time.sleep() is being used instead of asyncio.sleep()"
    )

    # Verify timeout worked (no infinite hang)
    assert elapsed < 1.0, f"wait_for_memory_available took {elapsed:.2f}s, expected <1s"

    # Clean up
    await task


@pytest.mark.asyncio
async def test_memory_backpressure_returns_when_threshold_met():
    """Verify that wait_for_memory_available returns immediately if already under threshold."""
    monitor = MemoryMonitor(max_memory_gb=100.0)  # 100GB cap (way over current usage)

    start_time = time.time()

    # Should return immediately since we're well under the threshold
    await monitor.wait_for_memory_available(target_pct=0.9)  # 90GB threshold

    elapsed = time.time() - start_time

    # Should be nearly instant
    assert elapsed < 0.1, (
        f"Should return immediately when under threshold, took {elapsed:.2f}s"
    )


@pytest.mark.asyncio
async def test_multiple_coroutines_can_wait_concurrently():
    """Verify multiple coroutines can wait for memory concurrently without blocking each other."""
    monitor = MemoryMonitor(max_memory_gb=0.001)  # 1MB cap

    completed_tasks = []

    async def wait_task(task_id: int):
        """Simulated task that waits for memory."""
        try:
            await asyncio.wait_for(
                monitor.wait_for_memory_available(target_pct=0.0001),
                timeout=0.2,
            )
        except TimeoutError:
            completed_tasks.append(task_id)

    # Start 5 concurrent tasks
    tasks = [asyncio.create_task(wait_task(i)) for i in range(5)]

    # All should complete via timeout (not hang)
    await asyncio.gather(*tasks)

    # All tasks should have completed
    assert len(completed_tasks) == 5, (
        f"Only {len(completed_tasks)}/5 tasks completed. "
        "Event loop may be blocked by time.sleep()"
    )


@pytest.mark.asyncio
async def test_event_loop_responsiveness_during_memory_wait():
    """Verify event loop remains responsive during memory wait.

    This is a stress test that measures event loop lag while memory
    monitoring is active. If time.sleep() is used, lag will be >1s.
    """
    monitor = MemoryMonitor(max_memory_gb=0.001)  # 1MB cap

    # Track event loop responsiveness
    ping_times = []

    async def event_loop_ping():
        """Ping the event loop every 100ms."""
        for _ in range(5):
            start = time.time()
            await asyncio.sleep(0)  # Yield to event loop
            ping_times.append(time.time() - start)
            await asyncio.sleep(0.1)

    # Start ping task
    ping_task = asyncio.create_task(event_loop_ping())

    # Start memory wait (will timeout)
    wait_task = asyncio.create_task(
        asyncio.wait_for(
            monitor.wait_for_memory_available(target_pct=0.0001),
            timeout=0.6,
        )
    )

    # Run both concurrently
    try:
        await asyncio.gather(ping_task, wait_task)
    except TimeoutError:
        pass  # Expected from wait_task

    # Verify event loop was responsive (low ping times)
    max_ping = max(ping_times) if ping_times else 0
    avg_ping = sum(ping_times) / len(ping_times) if ping_times else 0

    assert max_ping < 0.05, (
        f"Event loop lag detected: max={max_ping:.3f}s. "
        "This indicates blocking time.sleep() is being used."
    )
    assert avg_ping < 0.01, (
        f"Average event loop lag too high: {avg_ping:.3f}s. "
        "This indicates blocking time.sleep() is being used."
    )


def test_sync_context_usage_raises_warning():
    """Verify that wait_for_memory_available can't be called from sync context.

    This test verifies that the function signature is async, preventing
    accidental synchronous usage that would cause the 2.5.18 hang bug.
    """
    monitor = MemoryMonitor(max_memory_gb=1.0)

    # Attempting to call without await should fail
    with pytest.raises(TypeError, match="coroutine"):
        # This should raise: "coroutine 'wait_for_memory_available' was never awaited"
        monitor.wait_for_memory_available()  # Missing await


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
