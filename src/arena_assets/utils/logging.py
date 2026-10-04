import atexit
import logging
import sys

import enlighten

logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
_logger = logging.getLogger()
_manager = enlighten.get_manager(stream=sys.stderr, enabled=True)


def initialize(silent: bool = False, log_level: str = "WARNING"):
    """Reconfigure the global logger and manager instances.
    Should be called once from the CLI before any other operations.

    Args:
        silent: If True, disable all progress bars and set log level to CRITICAL+1
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _manager, _logger

    _manager = enlighten.get_manager(stream=sys.stderr, enabled=not silent)

    if silent:
        logging.getLogger().setLevel(logging.CRITICAL + 1)
    else:
        numeric_level = getattr(logging, log_level.upper(), logging.WARNING)
        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            stream=sys.stderr,
            force=True,  # Force reconfiguration
        )

    _logger = logging.getLogger()


def get_manager() -> enlighten.Manager:
    """Get the global progress manager instance."""
    return _manager


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance. If name is provided, return a child logger."""
    if name:
        return _logger.getChild(name)
    return _logger


def format_file_size(size_bytes: float) -> str:
    """Format file size in human-readable format."""
    size_bytes = int(size_bytes)

    if size_bytes == 0:
        return "0B"

    units = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)}{units[unit_index]}"
    else:
        return f"{size:.1f}{units[unit_index]}"


def _exit_handler():
    global _manager
    _manager.__exit__()


atexit.register(_exit_handler)
