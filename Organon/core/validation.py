"""
ORGANON - Core Input Validation

Provides shared validation for inputs before expensive operations
such as embedding generation.
"""

import math

from core.exceptions import ValidationError


def validate_document_inputs(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Validate document inputs before generating embeddings."""

    # IDs and documents must have matching lengths.
    if len(ids) != len(documents):
        raise ValidationError("IDs and documents must have matching lengths.")

    # At least one document must be provided.
    if not documents:
        raise ValidationError("At least one document must be provided.")

    # Document IDs must be non-empty strings.
    for doc_id in ids:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValidationError("Document IDs must be non-empty strings.")

    # Document IDs must be unique.
    if len(ids) != len(set(ids)):
        raise ValidationError("Document IDs must be unique within an upsert batch.")

    # Documents must contain non-whitespace text.
    for document in documents:
        if not isinstance(document, str) or not document.strip():
            raise ValidationError("Documents must be non-empty strings.")

    # Metadata count must match the number of documents.
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValidationError("Metadata count must match the number of IDs.")

    # Validate metadata using shared core validation.
    validate_metadata(metadatas)


def validate_query_input(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> None:
    """Validate retrieval inputs before generating query embeddings."""

    # Query must be a non-empty string.
    if not isinstance(query, str) or not query.strip():
        raise ValidationError("Query must be a non-empty string.")

    # top_k must be a positive integer.
    # Boolean values are rejected because bool is a subclass of int in Python.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError("top_k must be a positive integer.")

    # Filters must be a dictionary when provided.
    if filters is not None and not isinstance(filters, dict):
        raise ValidationError("Filters must be a dictionary.")


def validate_metadata(metadatas: list[dict] | None) -> None:
    """Validate metadata entries and their values."""

    if metadatas is None:
        return

    for metadata in metadatas:
        if not isinstance(metadata, dict):
            raise ValidationError("Each metadata entry must be a dictionary.")

        for key, value in metadata.items():

            if not isinstance(key, str) or not key.strip():
                raise ValidationError("Metadata keys must be non-empty strings.")

            if isinstance(value, bool):
                continue

            if isinstance(value, int):
                continue

            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValidationError("Metadata float values must be finite.")
                continue

            if isinstance(value, str):
                continue

            raise ValidationError(f"Unsupported metadata value type for key '{key}'.")
