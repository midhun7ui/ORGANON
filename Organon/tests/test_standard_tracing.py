import asyncio
import pytest

from core.observability.tracing.providers.standard import (
    StandardSpan,
    StandardTracer,
)


def test_standard_span_generates_unique_ids():

    span_one = StandardSpan("retrieval")
    span_two = StandardSpan("embedding")

    assert span_one.span_id != span_two.span_id

    assert span_one.name == "retrieval"
    assert span_two.name == "embedding"

def test_standard_span_stores_parent_span_id():

    parent = StandardSpan("retrieval")

    child = StandardSpan(
        "vector_search",
        parent_span_id=parent.span_id,
    )

    assert child.parent_span_id == parent.span_id
    assert child.span_id != parent.span_id

def test_standard_span_stores_attributes():

    span = StandardSpan(
        "retrieval",
        attributes={
            "top_k": 5,
            "operation": "document_search",
        },
    )

    span.set_attribute("result_count", 10)

    assert span.attributes == {
        "top_k": 5,
        "operation": "document_search",
        "result_count": 10,
    }

def test_standard_span_records_exception_type():

    span = StandardSpan("document_retrieval")

    exception = ValueError("Sensitive information")

    span.record_exception(exception)

    assert span.exception_type == "ValueError"
    assert "Sensitive information" not in str(span.exception_type)

def test_standard_span_records_duration_and_ends_once():

    span = StandardSpan("document_retrieval")

    assert span.end_time is None
    assert span.duration_ns is None

    span.end()

    first_end_time = span.end_time
    first_duration = span.duration_ns

    assert first_end_time is not None
    assert first_duration is not None
    assert first_duration >= 0

    span.end()

    assert span.end_time == first_end_time
    assert span.duration_ns == first_duration

def test_standard_tracer_creates_span_without_changing_context():

    tracer = StandardTracer()

    span = tracer.start_span(
        "document_retrieval",
        attributes={"top_k": 5},
    )

    assert isinstance(span, StandardSpan)
    assert span.name == "document_retrieval"
    assert span.parent_span_id is None
    assert span.attributes == {"top_k": 5}

    # Creating a span must not activate it.
    assert tracer.get_current_span() is None

def test_standard_tracer_tracks_nested_spans():

    tracer = StandardTracer()

    with tracer.start_as_current_span("ingestion") as parent:

        assert tracer.get_current_span() is parent

        with tracer.start_as_current_span("embedding") as child:

            assert tracer.get_current_span() is child

            assert child.parent_span_id == parent.span_id

        # After child finishes, parent becomes active again.
        assert tracer.get_current_span() is parent

    # After parent finishes, no span should be active.
    assert tracer.get_current_span() is None

    assert parent.end_time is not None
    assert child.end_time is not None

@pytest.mark.asyncio
async def test_standard_tracer_isolates_async_contexts():

    tracer = StandardTracer()

    async def run_request(name: str) -> str:

        with tracer.start_as_current_span(name) as span:

            await asyncio.sleep(0.01)

            assert tracer.get_current_span() is span

            return span.span_id

    first_id, second_id = await asyncio.gather(
        run_request("request_one"),
        run_request("request_two"),
    )

    assert first_id != second_id
    assert tracer.get_current_span() is None