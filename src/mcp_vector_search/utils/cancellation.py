"""Cancellation token support for interruptible operations."""

import signal
import threading
from collections.abc import Callable


class OperationCancelledError(Exception):
    """Raised when an operation is cancelled."""

    pass


class CancellationToken:
    """Thread-safe cancellation token for interruptible operations.

    Usage:
        token = CancellationToken()

        # Set up signal handler
        def handler(signum, frame):
            logger.info("Interrupt received, cancelling...")
            token.cancel()
        signal.signal(signal.SIGINT, handler)

        # Check cancellation in loops
        for item in items:
            token.check()  # Raises OperationCancelledError if cancelled
            process(item)
    """

    def __init__(self):
        """Initialize cancellation token."""
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    def cancel(self):
        """Cancel the operation and trigger callbacks."""
        with self._lock:
            if self._cancelled:
                return  # Already cancelled
            self._cancelled = True

        # Call callbacks outside the lock to avoid deadlocks
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore callback errors

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled.

        Returns:
            True if cancelled
        """
        with self._lock:
            return self._cancelled

    def check(self):
        """Raise OperationCancelledError if token is cancelled.

        Raises:
            OperationCancelledError: If operation was cancelled
        """
        if self.is_cancelled():
            raise OperationCancelledError("Operation was cancelled")

    def register_callback(self, callback: Callable[[], None]):
        """Register a callback to be called when cancelled.

        Args:
            callback: Function to call on cancellation
        """
        with self._lock:
            self._callbacks.append(callback)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cancel on exception."""
        if exc_type is not None:
            self.cancel()
        return False  # Don't suppress exceptions


def setup_interrupt_handler(cancel_token: CancellationToken) -> Callable:
    """Set up SIGINT handler to cancel operations.

    Args:
        cancel_token: Token to cancel on interrupt

    Returns:
        Previous signal handler (for restoration)
    """

    def handler(signum, frame):
        cancel_token.cancel()

    previous_handler = signal.signal(signal.SIGINT, handler)
    return previous_handler


def restore_interrupt_handler(previous_handler: Callable):
    """Restore previous SIGINT handler.

    Args:
        previous_handler: Handler to restore
    """
    signal.signal(signal.SIGINT, previous_handler)
