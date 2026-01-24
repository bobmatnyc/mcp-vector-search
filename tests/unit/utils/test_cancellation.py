"""Tests for cancellation token."""

import signal
import threading
import time

import pytest

from mcp_vector_search.utils.cancellation import (
    CancellationToken,
    OperationCancelledError,
    restore_interrupt_handler,
    setup_interrupt_handler,
)


def test_cancellation_token_basic():
    """Test basic cancellation token functionality."""
    token = CancellationToken()

    # Initially not cancelled
    assert not token.is_cancelled()

    # Cancel it
    token.cancel()
    assert token.is_cancelled()

    # Check raises exception
    with pytest.raises(OperationCancelledError):
        token.check()


def test_cancellation_token_thread_safety():
    """Test that cancellation token is thread-safe."""
    token = CancellationToken()
    errors = []

    def worker():
        try:
            for _ in range(1000):
                token.is_cancelled()
                time.sleep(0.0001)
        except Exception as e:
            errors.append(e)

    # Start multiple threads
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()

    # Cancel while threads are running
    time.sleep(0.01)
    token.cancel()

    # Wait for threads
    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0
    assert token.is_cancelled()


def test_cancellation_callback():
    """Test cancellation callbacks."""
    token = CancellationToken()
    called = []

    def callback1():
        called.append(1)

    def callback2():
        called.append(2)

    token.register_callback(callback1)
    token.register_callback(callback2)

    # Cancel triggers callbacks
    token.cancel()
    assert called == [1, 2]


def test_cancellation_callback_error_handling():
    """Test that callback errors don't break cancellation."""
    token = CancellationToken()
    called = []

    def bad_callback():
        raise ValueError("Callback error")

    def good_callback():
        called.append("good")

    token.register_callback(bad_callback)
    token.register_callback(good_callback)

    # Cancel should succeed despite bad callback
    token.cancel()
    assert token.is_cancelled()
    assert "good" in called


def test_context_manager():
    """Test cancellation token as context manager."""
    token = CancellationToken()

    # Normal exit
    with token:
        assert not token.is_cancelled()
    assert not token.is_cancelled()

    # Exception cancels token
    token2 = CancellationToken()
    try:
        with token2:
            raise ValueError("Test error")
    except ValueError:
        pass

    assert token2.is_cancelled()


def test_interrupt_handler_setup():
    """Test SIGINT handler setup and restoration."""
    token = CancellationToken()

    # Save original handler
    original = signal.getsignal(signal.SIGINT)

    # Setup interrupt handler
    previous = setup_interrupt_handler(token)

    # Handler should be different
    assert signal.getsignal(signal.SIGINT) != original

    # Restore handler
    restore_interrupt_handler(previous)

    # Should be restored
    assert signal.getsignal(signal.SIGINT) == previous


def test_cancellation_check_before_cancel():
    """Test that check() doesn't raise before cancellation."""
    token = CancellationToken()

    # Should not raise
    token.check()
    token.check()
    token.check()

    # Now cancel and it should raise
    token.cancel()
    with pytest.raises(OperationCancelledError):
        token.check()


def test_multiple_cancellations_are_safe():
    """Test that calling cancel() multiple times is safe."""
    token = CancellationToken()

    token.cancel()
    token.cancel()  # Should be safe
    token.cancel()  # Should be safe

    assert token.is_cancelled()
