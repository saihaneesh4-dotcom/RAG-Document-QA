import os

import app.vector_store as vector_store

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts


def process_document(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # Create a fresh index for the new document
    vector_store.index = vector_store.faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_store.add_embeddings(embeddings)

    os.makedirs("data/vectorstore", exist_ok=True)

    index_path = "data/vectorstore/index.faiss"
    chunks_path = "data/vectorstore/chunks.json"

    vector_store.save_index(index_path)
    vector_store.save_chunks(chunks, chunks_path)

    return chunks