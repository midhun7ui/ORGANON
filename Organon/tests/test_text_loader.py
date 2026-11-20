
import pytest

from core.documents.loaders.text import TextLoader
from core.exceptions import IngestionError


@pytest.mark.asyncio
async def test_load_text_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello ORGANON!", encoding="utf-8")

    loader = TextLoader()
    documents = await loader.load(file_path)

    assert len(documents) == 1
    assert documents[0].content == "Hello ORGANON!"
    assert documents[0].metadata["file_type"] == "txt"


@pytest.mark.asyncio
async def test_load_missing_file():
    loader = TextLoader()

    with pytest.raises(IngestionError):
        await loader.load("missing.txt")


@pytest.mark.asyncio
async def test_reject_unsupported_file(tmp_path):
    file_path = tmp_path / "example.pdf"
    file_path.write_text("PDF content")

    loader = TextLoader()

    with pytest.raises(IngestionError):
        await loader.load(file_path)


@pytest.mark.asyncio
async def test_reject_directory(tmp_path):
    loader = TextLoader()

    with pytest.raises(IngestionError):
        await loader.load(tmp_path)

@pytest.mark.asyncio
async def test_reject_empty_text_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(IngestionError, match="empty"):
        await TextLoader().load(file_path)


@pytest.mark.asyncio
async def test_reject_whitespace_only_file(tmp_path):
    file_path = tmp_path / "whitespace.txt"
    file_path.write_text("   \n\t  ", encoding="utf-8")

    with pytest.raises(IngestionError, match="empty"):
        await TextLoader().load(file_path)


@pytest.mark.asyncio
async def test_load_uppercase_txt_extension(tmp_path):
    file_path = tmp_path / "README.TXT"
    file_path.write_text("ORGANON", encoding="utf-8")

    documents = await TextLoader().load(file_path)

    assert documents[0].content == "ORGANON"


@pytest.mark.asyncio
async def test_reject_invalid_utf8(tmp_path):
    file_path = tmp_path / "invalid.txt"
    file_path.write_bytes(b"\x80\x81\x82")

    with pytest.raises(IngestionError, match="Failed to read"):
        await TextLoader().load(file_path)
