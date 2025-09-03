"""Worker mocks for integration tests."""
from unittest.mock import MagicMock


def create_mock_worker():
    """
    Create a mock worker that sets up the module-level attributes.
    Tests must configure the search return value themselves.

    Returns:
        MagicMock: A module mock with searcher and lock attributes
    """
    mock_module = MagicMock()
    mock_module.searcher = MagicMock()
    mock_module.lock = MagicMock()

    mock_module.init_agent_worker = MagicMock()

    return mock_module
