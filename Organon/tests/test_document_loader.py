
import pytest

from core.documents.base import Document, DocumentLoader


def test_document_loader_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DocumentLoader()


def test_concrete_loader_must_implement_load():
    class CustomLoader(DocumentLoader):
        pass

    with pytest.raises(TypeError):
        CustomLoader()


@pytest.mark.asyncio
async def test_concrete_loader_can_implement_load():
    class CustomLoader(DocumentLoader):
        async def load(self, source):
            return [Document(content="Loaded successfully")]

    loader = CustomLoader()
    documents = await loader.load("example.txt")

    assert len(documents) == 1
    assert documents[0].content == "Loaded successfully"

