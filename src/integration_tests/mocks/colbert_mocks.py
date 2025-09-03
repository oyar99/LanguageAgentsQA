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


# pylint: disable-next=unused-argument
def create_mock_indexer(*args, **kwargs):
    """Create a mock indexer instance."""
    return MagicMock()

# pylint: disable-next=unused-argument
def create_mock_searcher(*args, **kwargs):
    """Create a mock searcher instance."""
    return MagicMock()
