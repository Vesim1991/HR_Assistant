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
# Cached resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformer model once per Streamlit worker."""
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# PDF + RAG helper functions
# -----------------------------
def extract_pdf_pages(pdf_bytes: bytes):
    """Extract text from each PDF page with PyMuPDF."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )
    finally:
        document.close()

    return pages


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping character chunks.
    Tries to end chunks on whitespace when possible.
    """
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            whitespace = text.rfind(" ", start, end)
            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        new_start = max(0, end - overlap)
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks


def create_chunks(pages):
    """Create chunks while preserving page-number metadata."""
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_chunks = split_text(page_data["text"])

        for chunk_number, text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": text,
                }
            )

    return chunks


def build_faiss_index(chunks):
    """
    Embed chunks and build an in-memory FAISS index.

    Embeddings are normalized, so inner product is equivalent to
    cosine similarity.
    """
    embedding_model = load_embedding_model()

    texts = [item["text"] for item in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(question, index, chunks, top_k=TOP_K):
    """Return the most relevant chunks for the user's question."""
    embedding_model = load_embedding_model()

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def build_context(results):
    """Format retrieved chunks for the LLM."""
    sections = []

    for i, item in enumerate(results, start=1):
        sections.append(
            f"[Source {i} | Page {item['page']} | Chunk {item['chunk']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(sections)


def get_groq_api_key():
    """
    Read GROQ_API_KEY from Streamlit Community Cloud secrets.

    A local environment-variable fallback is intentionally omitted so the
    beginner deployment path stays focused on Streamlit Secrets.
    """
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def ask_groq(question, context, api_key):
    """Ask Groq to answer strictly from retrieved policy context."""
    client = Groq(api_key=api_key)

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy excerpts supplied in the
context.

Rules:
1. Do not invent company rules, benefits, dates, limits, eligibility rules,
   procedures, or legal advice.
2. If the answer is not supported by the supplied context, say:
   "I couldn't find that information in the uploaded HR policy."
3. Be concise but useful.
4. When possible, mention the page number(s) shown in the source labels.
5. If different excerpts appear to conflict, explain the conflict instead of
   choosing one without evidence.
6. The uploaded document is the source of truth for this conversation.
""".strip()

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED HR POLICY CONTEXT:
{context}

Answer the question from the context above.
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return response.choices[0].message.content


def reset_document_state():
    """Clear document-specific data and chat history."""
    keys = [
        "pdf_hash",
        "pdf_name",
        "pages",
        "chunks",
        "faiss_index",
        "messages",
    ]

    for key in keys:
        st.session_state.pop(key, None)


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# UI
# -----------------------------
st.title("📘 HR Policy Assistant")
st.caption(
    "Upload an HR policy PDF and ask questions using RAG, "
    "Sentence Transformers, FAISS, PyMuPDF, and Groq."
)

with st.sidebar:
    st.header("1. Upload policy")

    uploaded_file = st.file_uploader(
        "HR policy PDF",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned image-only PDFs require OCR.",
    )

    st.divider()
    st.subheader("RAG settings")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K,
        help="How many relevant chunks are sent to Groq.",
    )

    st.caption(f"LLM: `{MODEL_NAME}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")

    if st.button("Clear document and chat", use_container_width=True):
        reset_document_state()
        st.rerun()


api_key = get_groq_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it in Streamlit Community Cloud → "
        "App settings → Secrets."
    )
    st.code('GROQ_API_KEY = "gsk_your_key_here"', language="toml")


# -----------------------------
# Process uploaded document
# -----------------------------
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Automatically process only when this is a new/different PDF.
    if st.session_state.get("pdf_hash") != current_hash:
        with st.spinner("Reading and indexing the HR policy..."):
            try:
                pages = extract_pdf_pages(pdf_bytes)

                if not pages:
                    st.error(
                        "No selectable text was found in this PDF. "
                        "It may be a scanned/image-only document."
                    )
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error("I could not create searchable chunks from this PDF.")
                    st.stop()

                index = build_faiss_index(chunks)

                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.messages = []

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

    st.success(
        f"Ready: {st.session_state.pdf_name} · "
        f"{len(st.session_state.pages)} text pages · "
        f"{len(st.session_state.chunks)} chunks"
    )


# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved policy sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Page {source['page']} · "
                        f"similarity {source['score']:.3f}**"
                    )
                    st.write(source["text"])


# -----------------------------
# Ask questions
# -----------------------------
question = st.chat_input(
    "Ask about leave, attendance, benefits, conduct, remote work, etc.",
    disabled=("faiss_index" not in st.session_state or not api_key),
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy and generating an answer..."):
            try:
                results = retrieve(
                    question=question,
                    index=st.session_state.faiss_index,
                    chunks=st.session_state.chunks,
                    top_k=top_k,
                )

                context = build_context(results)
                answer = ask_groq(
                    question=question,
                    context=context,
                    api_key=api_key,
                )

                st.markdown(answer)

                with st.expander("Retrieved policy sources"):
                    for source in results:
                        st.markdown(
                            f"**Page {source['page']} · "
                            f"similarity {source['score']:.3f}**"
                        )
                        st.write(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )

            except Exception as exc:
                st.error(f"Something went wrong while answering: {exc}")
