from __future__ import annotations
import logging
import sys


_COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
           "ERROR": "\033[31m", "CRITICAL": "\033[35m"}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _RESET)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def get_logger(name: str = "neuratensor", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                                         datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
