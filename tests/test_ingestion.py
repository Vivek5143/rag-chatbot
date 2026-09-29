from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src.ingestion import (
    ChunkingConfigurationError,
    EmptyPdfTextError,
    PdfNotFoundError,
    PdfValidationError,
    chunk_documents,
    ingest_pdf,
    load_pdf_pages,
)


def test_missing_pdf_has_clear_error(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "Ebook-Agentic-AI.pdf"

    with pytest.raises(PdfNotFoundError, match="Ebook-Agentic-AI.pdf"):
        load_pdf_pages(missing_pdf)


def test_corrupt_pdf_is_rejected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_text("not a valid PDF", encoding="utf-8")

    with pytest.raises(PdfValidationError, match="Unable to open PDF"):
        load_pdf_pages(pdf_path)


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    source_pdf = tmp_path / "source.pdf"
    encrypted_pdf = tmp_path / "encrypted.pdf"
    _write_text_pdf(source_pdf, ["Protected page text"])

    writer = PdfWriter()
    writer.append(str(source_pdf))
    writer.encrypt("test-password")
    with encrypted_pdf.open("wb") as file_handle:
        writer.write(file_handle)

    with pytest.raises(PdfValidationError, match="Encrypted PDFs"):
        load_pdf_pages(encrypted_pdf)


def test_load_pdf_pages_preserves_page_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_text_pdf(pdf_path, ["First source page", "Second source page"])

    pages = load_pdf_pages(pdf_path)

    assert [page.page_content for page in pages] == [
        "First source page",
        "Second source page",
    ]
    assert [page.metadata["page"] for page in pages] == [1, 2]
    assert all(page.metadata["source"] == "sample.pdf" for page in pages)
    assert all(len(page.metadata["document_sha256"]) == 64 for page in pages)


def test_ingest_pdf_creates_stable_embedding_ready_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    page_one = "Agentic AI uses planning and memory. " * 70
    page_two = "Traditional automation follows fixed rules. " * 70
    _write_text_pdf(pdf_path, [page_one, page_two])

    first_run = ingest_pdf(pdf_path, chunk_size=300, chunk_overlap=50)
    second_run = ingest_pdf(pdf_path, chunk_size=300, chunk_overlap=50)

    assert len(first_run) > 2
    assert [chunk.metadata["chunk_id"] for chunk in first_run] == [
        chunk.metadata["chunk_id"] for chunk in second_run
    ]
    assert [chunk.metadata["chunk_index"] for chunk in first_run] == list(
        range(len(first_run))
    )
    assert {chunk.metadata["page"] for chunk in first_run} == {1, 2}
    for chunk in first_run:
        assert chunk.page_content
        assert chunk.metadata["source"] == "source.pdf"
        assert len(chunk.metadata["document_sha256"]) == 64
        assert isinstance(chunk.metadata["chunk_id"], str)
        assert chunk.metadata["chunk_id"]


def test_empty_pdf_text_is_rejected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as file_handle:
        writer.write(file_handle)

    with pytest.raises(EmptyPdfTextError, match="no extractable text"):
        load_pdf_pages(pdf_path)


def test_chunk_documents_rejects_invalid_settings() -> None:
    document = Document(
        page_content="A page with text",
        metadata={"source": "source.pdf", "page": 1, "document_sha256": "a" * 64},
    )

    with pytest.raises(ChunkingConfigurationError, match="smaller"):
        chunk_documents([document], chunk_size=100, chunk_overlap=100)


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    """Create a tiny text-native PDF fixture without adding a test-only dependency."""

    page_ids = [4 + index * 2 for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
            f"/Count {len(pages)} >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for page_id, content_id, text in zip(page_ids, content_ids, pages, strict=True):
        escaped_text = _escape_pdf_text(text)
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\n", " ")
