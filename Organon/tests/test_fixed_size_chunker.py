from core.chunking.base import Chunker
from core.documents.base import Document
import pytest


def test_fixed_size_chunker_implements_chunker():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)

    assert isinstance(chunker, Chunker)


def test_splits_document_into_fixed_size_chunks():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)

    document = Document(content="HelloWorld")

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == "World"


def test_preserves_document_metadata():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)

    document = Document(
        content="HelloWorld",
        metadata={"source": "example.txt"},
    )

    chunks = chunker.chunk(document)

    assert chunks[0].metadata["source"] == "example.txt"
    assert chunks[1].metadata["source"] == "example.txt"


def test_splits_document_with_overlap():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(
        chunk_size=6,
        overlap=2,
    )

    document = Document(content="ABCDEFGHIJKL")

    chunks = chunker.chunk(document)

    assert chunks[0].content == "ABCDEF"
    assert chunks[1].content == "EFGHIJ"
    assert chunks[2].content == "IJKL"


def test_document_smaller_than_chunk_size():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=100)
    document = Document(content="Hello ORGANON")

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello ORGANON"


def test_document_exactly_matches_chunk_size():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)
    document = Document(content="Hello")

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


def test_empty_document_returns_empty_chunks():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)
    document = Document(content="")

    chunks = chunker.chunk(document)

    assert chunks == []


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
        (True, 0),
        (5, True),
    ],
)
def test_rejects_invalid_configuration(chunk_size, overlap):
    from core.chunking.providers.fixed_size import FixedSizeChunker

    with pytest.raises(ValueError):
        FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

def test_assigns_sequential_chunk_indexes():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)
    document = Document(content="ABCDEFGHIJKL")

    chunks = chunker.chunk(document)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]


def test_does_not_modify_original_metadata():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    original_metadata = {"source": "example.txt"}
    document = Document(
        content="ABCDEFGHIJ",
        metadata=original_metadata,
    )

    chunker = FixedSizeChunker(chunk_size=5)
    chunks = chunker.chunk(document)

    assert document.metadata == original_metadata
    assert "chunk_index" not in document.metadata


def test_chunks_have_independent_metadata():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    document = Document(
        content="ABCDEFGHIJ",
        metadata={"source": "example.txt"},
    )

    chunker = FixedSizeChunker(chunk_size=5)
    chunks = chunker.chunk(document)

    chunks[0].metadata["source"] = "modified.txt"

    assert chunks[1].metadata["source"] == "example.txt"

def test_rejects_invalid_document_input():
    from core.chunking.providers.fixed_size import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=5)

    with pytest.raises(TypeError, match="Document"):
        chunker.chunk("This is not a Document")