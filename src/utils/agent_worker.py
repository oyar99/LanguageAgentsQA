"""
Initializes the agent worker with a logger and a lock.
This is used to ensure that the worker can safely access shared resources.

This module should be imported as follows

import utils.agent_worker as worker

worker.lock
worker.searcher
"""

from queue import Queue

from logger.logger import Logger

def init_agent_worker(q: Queue, l):  # type: ignore
    """
    Initializes the worker
    """
    Logger(q)
    # pylint: disable-next=global-variable-undefined
    global searcher
    # pylint: disable-next=global-variable-undefined
    global lock
    lock = l
    searcher = None
