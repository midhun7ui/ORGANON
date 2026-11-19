import pytest
from core.vectorstores.validation import validate_search_inputs
from core.vectorstores.validation import validate_upsert_inputs


def test_valid_upsert_inputs():
    validate_upsert_inputs(
        ids=["doc-1", "doc-2"],
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        documents=["Document 1", "Document 2"],
        metadatas=[{"type": "A"}, {"type": "B"}],
    )


def test_mismatched_vector_count():
    with pytest.raises(ValueError):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            documents=["Document 1"],
        )


def test_mismatched_document_count():
    with pytest.raises(ValueError):
        validate_upsert_inputs(
            ids=["doc-1", "doc-2"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            documents=["Document 1"],
        )


def test_mismatched_metadata_count():
    with pytest.raises(ValueError):
        validate_upsert_inputs(
            ids=["doc-1", "doc-2"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            documents=["Document 1", "Document 2"],
            metadatas=[{"type": "A"}],
        )


def test_rejects_empty_id():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=[""],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
        )


def test_rejects_whitespace_id():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=["   "],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
        )


def test_rejects_non_string_id():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=[123],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
        )


def test_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        validate_upsert_inputs(
            ids=["doc-1", "doc-1"],
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            documents=["Document 1", "Document 2"],
        )


def test_rejects_empty_vector():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[]],
            documents=["Document"],
        )


def test_rejects_non_numeric_vector_values():
    with pytest.raises(ValueError, match="finite numbers"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, "invalid"]],
            documents=["Document"],
        )


def test_rejects_boolean_vector_values():
    with pytest.raises(ValueError, match="finite numbers"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, True]],
            documents=["Document"],
        )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_vector_values(invalid_value):
    with pytest.raises(ValueError, match="finite numbers"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, invalid_value]],
            documents=["Document"],
        )


def test_rejects_inconsistent_vector_dimensions():
    with pytest.raises(ValueError, match="same dimensions"):
        validate_upsert_inputs(
            ids=["doc-1", "doc-2"],
            vectors=[[0.1, 0.2], [0.3, 0.4, 0.5]],
            documents=["Document 1", "Document 2"],
        )


def test_accepts_consistent_vector_dimensions():
    validate_upsert_inputs(
        ids=["doc-1", "doc-2"],
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        documents=["Document 1", "Document 2"],
    )


def test_rejects_empty_document():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=[""],
        )


def test_rejects_whitespace_document():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=["   "],
        )


def test_rejects_non_string_document():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=[123],
        )


def test_accepts_valid_metadata():
    validate_upsert_inputs(
        ids=["doc-1"],
        vectors=[[0.1, 0.2]],
        documents=["Document"],
        metadatas=[
            {
                "category": "research",
                "year": 2026,
                "score": 0.95,
                "published": True,
            }
        ],
    )


def test_rejects_non_dictionary_metadata():
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
            metadatas=["invalid"],
        )


def test_rejects_empty_metadata_key():
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
            metadatas=[{" ": "value"}],
        )


def test_rejects_unsupported_metadata_value():
    with pytest.raises(ValueError, match="Unsupported metadata value"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
            metadatas=[{"tags": ["python", "AI"]}],
        )


def test_rejects_non_finite_metadata_float():
    with pytest.raises(ValueError, match="must be finite"):
        validate_upsert_inputs(
            ids=["doc-1"],
            vectors=[[0.1, 0.2]],
            documents=["Document"],
            metadatas=[{"score": float("nan")}],
        )


def test_valid_search_inputs():
    validate_search_inputs(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters={"category": "research"},
    )


def test_rejects_empty_query_vector():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_search_inputs(query_vector=[])


def test_rejects_non_numeric_query_values():
    with pytest.raises(ValueError, match="finite numbers"):
        validate_search_inputs(query_vector=[0.1, "invalid"])


@pytest.mark.parametrize(
    "invalid_value",
    [True, float("nan"), float("inf"), float("-inf")],
)
def test_rejects_invalid_query_vector_values(invalid_value):
    with pytest.raises(ValueError, match="finite numbers"):
        validate_search_inputs(query_vector=[0.1, invalid_value])


@pytest.mark.parametrize("invalid_top_k", [0, -1, 1.5, "5", True])
def test_rejects_invalid_top_k(invalid_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        validate_search_inputs(
            query_vector=[0.1, 0.2],
            top_k=invalid_top_k,
        )


def test_rejects_non_dictionary_filters():
    with pytest.raises(ValueError, match="Filters must be a dictionary"):
        validate_search_inputs(
            query_vector=[0.1, 0.2],
            filters=["category", "research"],
        )


def test_accepts_search_without_filters():
    validate_search_inputs(
        query_vector=[0.1, 0.2],
        top_k=3,
    )
