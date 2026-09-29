"""Explicit, page-aware PDF loading and chunking for the canonical eBook."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PDF_PATH = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class IngestionError(RuntimeError):
    """Base class for PDF ingestion failures that callers can handle safely."""


class PdfNotFoundError(IngestionError, FileNotFoundError):
    """Raised when the approved PDF has not been supplied at the requested path."""


class PdfValidationError(IngestionError):
    """Raised when a path is not a readable, usable PDF document."""


class PdfExtractionError(IngestionError):
    """Raised when PDF text cannot be extracted for embedding preparation."""


class EmptyPdfTextError(PdfExtractionError):
    """Raised when a PDF has no extractable text and requires a different workflow."""


class ChunkingConfigurationError(IngestionError, ValueError):
    """Raised when requested chunk settings are incompatible."""


def load_canonical_pdf_pages() -> list[Document]:
    """Load the approved eBook from its expected project-relative location."""

    return load_pdf_pages(CANONICAL_PDF_PATH)


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Load a PDF into one LangChain document per page with provenance metadata.

    Blank pages are retained so page numbering remains faithful to the source,
    but a document with no extractable text on every page is rejected. This is
    intentionally text-only: it does not attempt OCR or image interpretation.
    """

    path = _validate_pdf_path(pdf_path)
    document_sha256 = _calculate_sha256(path)

    try:
        reader = PdfReader(str(path))
    except (OSError, PdfReadError, ValueError) as error:
        raise PdfValidationError(f"Unable to open PDF: {path}") from error

    if reader.is_encrypted:
        raise PdfValidationError(f"Encrypted PDFs are not supported: {path}")

    if not reader.pages:
        raise PdfValidationError(f"PDF has no pages: {path}")

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            content = (page.extract_text() or "").strip()
        except Exception as error:  # pypdf can raise parser-specific exceptions per page.
            raise PdfExtractionError(
                f"Unable to extract text from page {page_number} of {path}"
            ) from error

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "page": page_number,
                    "document_sha256": document_sha256,
                    "text_extracted": bool(content),
                },
            )
        )

    if not any(document.page_content for document in documents):
        raise EmptyPdfTextError(
            f"PDF contains no extractable text: {path}. OCR is not part of this pipeline."
        )

    return documents


def chunk_documents(
    page_documents: Sequence[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split non-empty page documents into deterministic, embedding-ready chunks."""

    _validate_chunking(chunk_size, chunk_overlap)
    _validate_page_documents(page_documents)

    documents_with_text = [
        document for document in page_documents if document.page_content.strip()
    ]
    if not documents_with_text:
        raise EmptyPdfTextError("No page documents contain extractable text to chunk")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_documents = splitter.split_documents(documents_with_text)

    chunks: list[Document] = []
    for chunk_index, chunk in enumerate(split_documents):
        content = chunk.page_content.strip()
        if not content:
            continue

        metadata = dict(chunk.metadata)
        document_sha256 = str(metadata["document_sha256"])
        page_number = int(metadata["page"])
        metadata["chunk_index"] = chunk_index
        metadata["chunk_id"] = _chunk_id(
            document_sha256=document_sha256,
            page_number=page_number,
            chunk_index=chunk_index,
            content=content,
        )
        chunks.append(Document(page_content=content, metadata=metadata))

    if not chunks:
        raise EmptyPdfTextError("Text splitter did not produce any embedding-ready chunks")

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Explicitly load and chunk a PDF; this function performs no external I/O."""

    return chunk_documents(
        load_pdf_pages(pdf_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise PdfNotFoundError(f"PDF file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfValidationError(f"Expected a .pdf file: {path}")
    return path


def _calculate_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ChunkingConfigurationError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ChunkingConfigurationError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ChunkingConfigurationError("chunk_overlap must be smaller than chunk_size")


def _validate_page_documents(page_documents: Sequence[Document]) -> None:
    if not page_documents:
        raise PdfExtractionError("No page documents were provided for chunking")

    required_metadata = ("source", "page", "document_sha256")
    for document in page_documents:
        missing = [key for key in required_metadata if key not in document.metadata]
        if missing:
            names = ", ".join(missing)
            raise PdfExtractionError(f"Page document is missing required metadata: {names}")
        if not isinstance(document.metadata["page"], int) or document.metadata["page"] < 1:
            raise PdfExtractionError("Page document metadata 'page' must be a positive integer")


def _chunk_id(
    *,
    document_sha256: str,
    page_number: int,
    chunk_index: int,
    content: str,
) -> str:
    payload = "\x1f".join(
        (document_sha256, str(page_number), str(chunk_index), content)
    ).encode("utf-8")
    return sha256(payload).hexdigest()
