import pytest

from core.chunking.base import Chunker
from core.documents.base import Document


def test_chunker_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Chunker()


def test_concrete_chunker_must_implement_chunk():
    class CustomChunker(Chunker):
        pass

    with pytest.raises(TypeError):
        CustomChunker()


def test_concrete_chunker_can_implement_chunk():
    class CustomChunker(Chunker):
        def chunk(self, document: Document) -> list[Document]:
            return [
                Document(content="First chunk"),
                Document(content="Second chunk"),
            ]

    chunker = CustomChunker()

    result = chunker.chunk(Document(content="Original document"))

    assert len(result) == 2
    assert result[0].content == "First chunk"
    assert result[1].content == "Second chunk"