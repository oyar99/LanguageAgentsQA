"""Logger module."""
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
import os
import uuid
from abc import ABC

from utils.singleton import Singleton


class BaseLogger(ABC, metaclass=Singleton):
    """
    A base logger abstract class to define the interface for logging.
    """

    def __init__(self):
        self._run_id = str(uuid.uuid4())
        self._logger = logging.getLogger(__name__)
        self._log_level = os.getenv("SCRIPT_LOG_LEVEL", "INFO").upper()
        self._logger.setLevel(level=self._log_level)

    def info(self, message: str) -> None:
        """
        Logs a message at the INFO level.

        Args:
            message (str): message to be logged
        """
        self._logger.info(message)

    def warn(self, message: str) -> None:
        """
        Logs a message at the WARN level.

        Args:
            message (str): message to be logged
        """
        self._logger.warning(message)

    def debug(self, message: str) -> None:
        """
        Logs a message at the DEBUG level.

        Args:
            message (str): message to be logged
        """
        self._logger.debug(message)

    def error(self, message: str) -> None:
        """
        Logs a message at the ERROR level.

        Args:
            message (str): message to be logged
        """
        self._logger.error(message)

    def get_run_id(self) -> str:
        """
        Gets the execution run id.

        Returns:
            run_id (str): run id
        """
        return self._run_id


class MainProcessLogger(BaseLogger):
    """
    A logger to send log messages to the queue
    """

    def __init__(self):
        super().__init__()

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(self._log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Create file handler only if file logging is not disabled
        disable_file_logging = os.getenv("DISABLE_FILE_LOGGING", "0") == "1"
        fh = None
        if not disable_file_logging:
            log_dir = os.path.join(os.path.normpath(
                os.getcwd() + os.sep + os.pardir), 'logs')

            log_fname = os.path.join(log_dir, f'app-{self._run_id}.log')
            fh = logging.FileHandler(log_fname, encoding='utf-8')
            fh.setLevel(self._log_level)
            fh.setFormatter(formatter)

        self._q = Queue(-1)

        handlers = []

        enable_console_logging = os.getenv(
            "ENABLE_CONSOLE_LOGGING", "0") == "1"
        if enable_console_logging:
            if not disable_file_logging and fh:
                handlers = [ch, fh]
            else:
                handlers = [ch]
        else:
            if not disable_file_logging and fh:
                handlers = [fh]

        self._ql = QueueListener(self._q, *handlers)
        self._ql.start()

    def get_queue(self) -> Queue:
        """
        Gets the queue used by the logger.

        Returns:
            Queue: the queue used by the logger
        """
        return self._q

    def get_queue_listener(self) -> QueueListener:
        """
        Gets the queue listener used by the logger.

        Returns:
            QueueListener: the queue listener used by the logger
        """
        return self._ql

    def shutdown(self):
        """
        Clean up the logger by stopping the queue listener.
        """
        if self._ql:
            self._ql.stop()
            self._ql = None
        if self._q:
            self._q.close()
            self._q = None


class Logger(BaseLogger):
    """
    A logger to log messages to a queue.
    """

    def __init__(self, q: Queue = None):
        super().__init__()
        # If the queue is not provided, we assume consumer intends to use the main process logger
        # Sub-processes should not use the main process logger, and should always ensure they provide a queue by using
        # helper function worker_init
        queue = q if q is not None else MainProcessLogger().get_queue()

        qh = QueueHandler(queue)

        self._logger.addHandler(qh)


def worker_init(q: Queue):
    """
    Initializes the worker logger with a unique run ID.
    """
    Logger(q)
