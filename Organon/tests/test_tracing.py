from core.observability.tracing.base import Span
import pytest
import pytest

from core.observability.tracing.base import Span, Tracer
class DummySpan(Span):
    """Concrete span implementation for testing."""

    def __init__(self):
        self.ended = False
        self.exception = None
        self.status = None

    def set_attribute(self, key, value):
        pass

    def record_exception(self, exception):
        self.exception = exception

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


def test_span_ends_after_successful_execution():

    span = DummySpan()

    with span:
        pass

    assert span.ended is True
    assert span.exception is None
    assert span.status is None

def test_span_records_exception_and_ends():

    span = DummySpan()
    error = ValueError("Retrieval failed.")

    try:
        with span:
            raise error

    except ValueError:
        pass

    assert span.ended is True
    assert span.exception is error
    assert span.status == "error"

def test_span_does_not_suppress_exception():

    span = DummySpan()

    with pytest.raises(ValueError, match="Retrieval failed."):
        with span:
            raise ValueError("Retrieval failed.")

    assert span.ended is True
    assert span.status == "error"

def test_tracer_requires_current_span_implementation():

    class IncompleteTracer(Tracer):

        def start_span(self, name, *, attributes=None):
            return DummySpan()

    with pytest.raises(TypeError, match="start_as_current_span"):
        IncompleteTracer()