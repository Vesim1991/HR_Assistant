import hashlib

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_pages(pdf_bytes):
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )
    finally:
        document.close()

    return pages


# -----------------------------
# Chunking
# -----------------------------
def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        # Try to end at whitespace
        if end < len(text):

            whitespace = text.rfind(
                " ",
                start,
                end
            )

            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        new_start = max(
            0,
            end - overlap
        )

        ​:contentReference[oaicite:1]{index=1}​
