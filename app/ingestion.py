import os

import app.vector_store as vector_store

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts


def process_documents(pdf_paths):
    all_chunks = []
    chunk_id = 0

    for pdf_path in pdf_paths:

        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)

        document_name = os.path.basename(pdf_path)

        for chunk in chunks:
            chunk["chunk_id"] = chunk_id
            chunk["document"] = document_name

            all_chunks.append(chunk)

            chunk_id += 1

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = embed_texts(texts)

    # Create a fresh index for all processed documents
    vector_store.index = vector_store.faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_store.add_embeddings(embeddings)

    os.makedirs("data/vectorstore", exist_ok=True)

    index_path = "data/vectorstore/index.faiss"
    chunks_path = "data/vectorstore/chunks.json"

    vector_store.save_index(index_path)
    vector_store.save_chunks(all_chunks, chunks_path)

    return all_chunks