"""Mock implementations for ColBERT components."""
from unittest.mock import MagicMock

# pylint: disable-next=too-few-public-methods
class MockIndexer:
    """Mock ColBERT Indexer that returns a MagicMock instance."""

    def __init__(self, *args, **kwargs):
        """Initialize with a MagicMock instance."""

# pylint: disable-next=too-few-public-methods
class MockSearcher:
    """Mock ColBERT Searcher that returns a MagicMock instance."""

    def __init__(self, *args, **kwargs):
        """Initialize with a MagicMock instance."""


class MockRun:
    """Mock ColBERT Run context manager."""

    def context(self, config):
        """Mock context method that returns a context manager."""
        return MockRunContext()

    def __call__(self):
        """Make MockRun callable to return itself."""
        return self


class MockRunContext:
    """Mock context manager for ColBERT Run."""

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""

# pylint: disable-next=too-few-public-methods
class MockRunConfig:
    """Mock ColBERT RunConfig."""

    def __init__(self, *args, **kwargs):
        """Initialize mock config."""

# pylint: disable-next=too-few-public-methods
class MockColBERTConfig:
    """Mock ColBERT ColBERTConfig."""

    def __init__(self, *args, **kwargs):
        """Initialize mock config."""


# pylint: disable-next=unused-argument
def create_mock_indexer(*args, **kwargs):
    """Create a mock indexer instance."""
    mock_indexer = MagicMock()
    mock_indexer.index = MagicMock()
    return mock_indexer

# pylint: disable-next=unused-argument
def create_mock_searcher(*args, **kwargs):
    """Create a mock searcher instance."""
    return MagicMock()


# Factory functions for easier patching
def create_mock_run():
    """Create a mock Run instance."""
    return MockRun()


def create_mock_run_config(*args, **kwargs):
    """Create a mock RunConfig instance."""
    return MockRunConfig(*args, **kwargs)


def create_mock_colbert_config(*args, **kwargs):
    """Create a mock ColBERTConfig instance."""
    return MockColBERTConfig(*args, **kwargs)
