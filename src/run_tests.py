#!/usr/bin/env python3
"""
Test runner script that properly handles logger initialization and cleanup.

This script ensures that:
1. Environment variables are set before any imports
2. Logger cleanup happens only once after all tests complete
"""
import os
import sys
import unittest

from logger.logger import MainProcessLogger

# Set test environment variables BEFORE any other imports
os.environ.setdefault('DISABLE_FILE_LOGGING', '1')
os.environ.setdefault('ENABLE_CONSOLE_LOGGING', '1')

def main():
    """Run all tests with proper setup and cleanup."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    MainProcessLogger().shutdown()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
