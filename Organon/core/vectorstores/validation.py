"""
ORGANON - Vector Store Validation

Provides shared validation for vector store operations.
"""

import math
from core.validation import validate_metadata


def validate_upsert_inputs(
    ids: list[str],
    vectors: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Validate inputs before inserting or updating documents."""

    # Validate input lengths.
    if not (len(ids) == len(vectors) == len(documents)):
        raise ValueError("IDs, vectors, and documents must have matching lengths.")

    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("Metadata count must match the number of IDs.")

    # Validate document IDs.
    for doc_id in ids:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document IDs must be non-empty strings.")

    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique within an upsert batch.")

    # Validate embedding vectors.
    for vector in vectors:
        if not vector:
            raise ValueError("Vectors must not be empty.")

        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Vector values must be finite numbers.")

    # All vectors must have equal dimensions.
    if vectors:
        expected_dimension = len(vectors[0])

        for vector in vectors:
            if len(vector) != expected_dimension:
                raise ValueError("All vectors must have the same dimensions.")

    # Validate document content.
    for document in documents:
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Documents must be non-empty strings.")

    # Validate metadata.
    if metadatas is not None:
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                raise ValueError("Each metadata entry must be a dictionary.")

            for key, value in metadata.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Metadata keys must be non-empty strings.")

                if isinstance(value, bool):
                    continue

                if isinstance(value, int):
                    continue

                if isinstance(value, float):
                    if not math.isfinite(value):
                        raise ValueError("Metadata float values must be finite.")
                    continue

                if isinstance(value, str):
                    continue

                raise ValueError(f"Unsupported metadata value type for key '{key}'.")

    validate_metadata(metadatas)


def validate_search_inputs(
    query_vector: list[float],
    top_k: int = 5,
    filters: dict | None = None,
) -> None:
    """Validate inputs before performing vector similarity search."""

    # Query vectors must not be empty.
    if not query_vector:
        raise ValueError("Query vector must not be empty.")

    # Query vector values must be finite numbers.
    for value in query_vector:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Query vector values must be finite numbers.")

    # top_k must be a positive integer.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Filters must be a dictionary when provided.
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("Filters must be a dictionary.")
