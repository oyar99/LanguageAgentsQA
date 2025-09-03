"""Mock implementations for file I/O operations."""
import json
import os
import tempfile
from unittest.mock import mock_open as mo
from typing import Optional
import shutil
from integration_tests.fixtures.musique_data import get_musique_data, get_musique_corpus

# Store original open function before it gets mocked
_ORIGINAL_OPEN = open


def _get_test_output_dirs():
    """Get or create test output directories."""
    if not hasattr(_get_test_output_dirs, 'temp_dir'):
        _get_test_output_dirs.temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(_get_test_output_dirs.temp_dir, 'output')
        _get_test_output_dirs.qa_output_dir = os.path.join(
            output_dir, 'qa_jobs')
        _get_test_output_dirs.retrieval_output_dir = os.path.join(
            output_dir, 'retrieval_jobs')

        # Create the directories
        os.makedirs(_get_test_output_dirs.qa_output_dir, exist_ok=True)
        os.makedirs(_get_test_output_dirs.retrieval_output_dir, exist_ok=True)

    return (
        _get_test_output_dirs.temp_dir,
        _get_test_output_dirs.qa_output_dir,
        _get_test_output_dirs.retrieval_output_dir
    )


def mock_open(file_path, *args, **kwargs):
    """Mock open() function for use in @patch decorators with test fixtures."""
    file_path_str = str(file_path)

    # Check for musique dataset file
    if 'musique_dev.json' in file_path_str:
        return mo(read_data=json.dumps(get_musique_data())).return_value

    # Check for musique corpus file
    if 'musique_corpus.json' in file_path_str:
        return mo(read_data=json.dumps(get_musique_corpus())).return_value

    # For all other files, use real open
    return _ORIGINAL_OPEN(file_path, *args, **kwargs)


def mock_qa_output_path(postfix: Optional[str] = None) -> str:
    """Mock function for get_qa_output_path for use in @patch decorators."""
    _, qa_output_dir, _ = _get_test_output_dirs()
    filename = (
        'qa_results_test_run.jsonl' if postfix is None else f'qa_results_test_run_{postfix}.jsonl'
    )
    return os.path.join(qa_output_dir, filename)


def mock_retrieval_output_path() -> str:
    """Mock function for get_retrieval_output_path for use in @patch decorators."""
    _, _, retrieval_output_dir = _get_test_output_dirs()
    return os.path.join(retrieval_output_dir, 'retrieval_results_test_run.jsonl')


def cleanup_test() -> None:
    """Clean up test fixtures and temp directory."""
    if hasattr(_get_test_output_dirs, 'temp_dir'):
        shutil.rmtree(_get_test_output_dirs.temp_dir, ignore_errors=True)
        delattr(_get_test_output_dirs, 'temp_dir')
        delattr(_get_test_output_dirs, 'qa_output_dir')
        delattr(_get_test_output_dirs, 'retrieval_output_dir')
