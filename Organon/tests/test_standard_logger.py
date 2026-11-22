
import logging

import pytest

from core.observability.logging.providers.standard import StandardLogger


def test_standard_logger_can_be_created():
    logger = StandardLogger()
    assert isinstance(logger, StandardLogger)


def test_standard_logger_rejects_invalid_level():
    with pytest.raises(ValueError):
        StandardLogger(level="INVALID")


def test_standard_logger_logs_info(caplog):
    logger = StandardLogger(name="organon_test_info")

    with caplog.at_level(logging.INFO, logger="organon_test_info"):
        logger.info("Application started")

    assert "Application started" in caplog.text


def test_standard_logger_logs_error(caplog):
    logger = StandardLogger(name="organon_test_error")

    with caplog.at_level(logging.ERROR, logger="organon_test_error"):
        logger.error("Something went wrong")

    assert "Something went wrong" in caplog.text



def test_standard_logger_includes_context(caplog):
    logger = StandardLogger(name="organon_test_context")

    with caplog.at_level(logging.INFO, logger="organon_test_context"):
        logger.info(
            "Document stored",
            context={"document_id": "doc-101"},
        )

    record = caplog.records[-1]

    assert "Document stored" in record.message
    assert record.organon_context["document_id"] == "doc-101"



def test_standard_logger_stores_structured_context(caplog):
    logger = StandardLogger(name="organon_test_structured")

    context = {
        "document_id": "doc-101",
        "pipeline": "ingestion",
    }

    with caplog.at_level(logging.INFO, logger="organon_test_structured"):
        logger.info("Document stored", context=context)

    record = caplog.records[-1]

    assert record.message == "Document stored"
    assert record.organon_context == context
    assert "context=" not in record.message


def test_standard_logger_accepts_numeric_level():
    logger = StandardLogger(
        name="organon_test_numeric",
        level=logging.DEBUG,
    )

    assert logger._logger.level == logging.DEBUG


def test_standard_logger_accepts_custom_numeric_level():
    logger = StandardLogger(
        name="organon_test_custom_level",
        level=15,
    )

    assert logger._logger.level == 15


@pytest.mark.parametrize("level", [True, False, -1, 3.14, None, []])
def test_standard_logger_rejects_invalid_level_types(level):
    with pytest.raises(ValueError):
        StandardLogger(level=level)


def test_standard_logger_accepts_string_level():
    logger = StandardLogger(
        name="organon_test_string_level",
        level="DEBUG",
    )

    assert logger._logger.level == logging.DEBUG
