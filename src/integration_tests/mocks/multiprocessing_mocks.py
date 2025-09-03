"""Mock implementations for multiprocessing components."""
from typing import Any, Callable, Iterable


class MockPool:
    """
    Mock multiprocessing Pool that executes functions sequentially.

    This can be used directly in @patch decorators with the 'new' parameter:
    @patch('models.agent.Pool', new=MockPool)

    It provides the same interface as multiprocessing.Pool but executes
    functions sequentially instead of in parallel, making tests deterministic
    and avoiding the complexity of actual multiprocessing in test environments.
    """

    def __init__(self, *args, **kwargs):
        """Initialize mock pool (ignores all arguments like processes count)."""
        # No initialization needed for mock

    def map(self, func: Callable, iterable: Iterable) -> list[Any]:
        """Execute function sequentially over iterable instead of in parallel."""
        return [func(item) for item in iterable]

    def __enter__(self):
        """Context manager entry - returns self for 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no cleanup needed for mock."""
        # No cleanup needed for mock
