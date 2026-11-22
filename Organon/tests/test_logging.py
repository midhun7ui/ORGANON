
import pytest

from core.observability.logging.base import Logger


class CustomLogger(Logger):
    """Test implementation of the Logger interface."""

    def debug(self, message, *, context=None):
        pass

    def info(self, message, *, context=None):
        pass

    def warning(self, message, *, context=None):
        pass

    def error(self, message, *, context=None):
        pass

    def critical(self, message, *, context=None):
        pass


def test_logger_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Logger()


def test_incomplete_logger_cannot_be_instantiated():
    class IncompleteLogger(Logger):
        def info(self, message, *, context=None):
            pass

    with pytest.raises(TypeError):
        IncompleteLogger()


def test_custom_logger_can_be_instantiated():
    logger = CustomLogger()

    assert isinstance(logger, Logger)


def test_logger_methods_accept_context():
    logger = CustomLogger()

    logger.debug("Debug message", context={"request_id": "123"})
    logger.info("Info message", context={"pipeline": "ingestion"})
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
