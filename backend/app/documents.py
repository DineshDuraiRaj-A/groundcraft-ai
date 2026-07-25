"""
Handles both preset sample documents and user uploads.

Uploads are capped at MAX_CHARS to keep retrieval fast and keep a single
upload from dominating a shared free-tier LLM budget. PDF extraction uses
pypdf (pure Python, no system-level poppler dependency) to keep hosting
on free tiers simple.
"""
from __future__ import annotations
import io
from pathlib import Path

from fastapi import UploadFile

MAX_CHARS = 20_000
SAMPLES_DIR = Path(__file__).parent / "samples"

SAMPLE_DOCS = {
    "refund_policy": {
        "title": "Acme SaaS — Refund Policy.pdf",
        "path": SAMPLES_DIR / "refund_policy.txt",
    },
    "coral_reefs": {
        "title": "Coral Reefs — Overview.pdf",
        "path": SAMPLES_DIR / "coral_reefs.txt",
    },
    "router_manual": {
        "title": "NetLync R400 — Setup Guide.pdf",
        "path": SAMPLES_DIR / "router_manual.txt",
    },
    "remote_work_pro": {
        "title": "Remote Work — The Case for Flexibility.pdf",
        "path": SAMPLES_DIR / "remote_work_pro.txt",
    },
    "remote_work_office": {
        "title": "Remote Work — The Case for the Office.pdf",
        "path": SAMPLES_DIR / "remote_work_office.txt",
    },
}

# Used by feature 10 (bias/perspective probe): pairs of docs covering the
# same topic from different angles, so the same question can be asked
# against each and the answers compared side by side.
PERSPECTIVE_PAIRS = {
    "remote_work": {
        "label": "Remote work productivity",
        "doc_a": "remote_work_pro",
        "doc_b": "remote_work_office",
    },
}


class DocumentTooLargeError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


def list_sample_docs() -> list[dict]:
    return [{"id": doc_id, "title": meta["title"]} for doc_id, meta in SAMPLE_DOCS.items()]


def load_sample_doc(doc_id: str) -> str:
    if doc_id not in SAMPLE_DOCS:
        raise KeyError(f"Unknown sample doc: {doc_id}")
    return SAMPLE_DOCS[doc_id]["path"].read_text(encoding="utf-8")


async def extract_uploaded_text(file: UploadFile) -> str:
    filename = (file.filename or "").lower()
    raw = await file.read()

    if filename.endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        text = _extract_pdf_text(raw)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{file.filename}'. Use .txt, .md, or .pdf."
        )

    text = text.strip()
    if len(text) > MAX_CHARS:
        raise DocumentTooLargeError(
            f"Document is {len(text):,} characters; the limit is {MAX_CHARS:,}. "
            "Try a shorter excerpt (roughly 5 pages)."
        )
    if not text:
        raise UnsupportedFileTypeError(
            "No extractable text found — this may be a scanned/image-only PDF."
        )
    return text


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf is not installed — add it to requirements.txt") from e

    reader = PdfReader(io.BytesIO(raw))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)
