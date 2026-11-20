import pytest

from core.exceptions import ValidationError
from core import validation


def test_valid_document_inputs():
    validation.validate_document_inputs(
        ids=["doc-1", "doc-2"],
        documents=["Python is great.", "AI is fascinating."],
    )


def test_rejects_empty_documents():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=[],
            documents=[],
        )


def test_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=["doc-1"],
            documents=["Document 1", "Document 2"],
        )


def test_rejects_empty_document_id():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=[""],
            documents=["Document 1"],
        )


def test_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=["doc-1", "doc-1"],
            documents=["Document 1", "Document 2"],
        )


def test_rejects_empty_document_content():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=["doc-1"],
            documents=["   "],
        )


def test_rejects_mismatched_metadata_count():
    with pytest.raises(ValidationError):
        validation.validate_document_inputs(
            ids=["doc-1", "doc-2"],
            documents=["Document 1", "Document 2"],
            metadatas=[{"source": "test"}],
        )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_rejects_invalid_query(query):
    with pytest.raises(ValidationError):
        validation.validate_query_input(query)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        True,
        1.5,
        "5",
    ],
)
def test_rejects_invalid_top_k(top_k):
    with pytest.raises(ValidationError):
        validation.validate_query_input(
            query="What is Python?",
            top_k=top_k,
        )


def test_rejects_invalid_filters():
    with pytest.raises(ValidationError):
        validation.validate_query_input(
            query="What is Python?",
            filters=["invalid"],
        )


def test_valid_query_inputs():
    validation.validate_query_input(
        query="What is Python?",
        top_k=5,
        filters={"category": "programming"},
    )


def test_accepts_valid_metadata():
    validation.validate_metadata(
        [
            {
                "category": "research",
                "year": 2026,
                "score": 0.95,
                "published": True,
            }
        ]
    )


def test_rejects_non_dictionary_metadata():
    with pytest.raises(ValidationError):
        validation.validate_metadata(["invalid"])


def test_rejects_empty_metadata_key():
    with pytest.raises(ValidationError):
        validation.validate_metadata([{" ": "value"}])


def test_rejects_unsupported_metadata_value():
    with pytest.raises(ValidationError):
        validation.validate_metadata([{"tags": ["Python", "AI"]}])


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_rejects_non_finite_metadata_values(value):
    with pytest.raises(ValidationError):
        validation.validate_metadata([{"score": value}])
