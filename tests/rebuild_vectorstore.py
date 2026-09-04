import os

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts

import app.vector_store as vector_store


# -----------------------------
# Paths
# -----------------------------

pdf_path = "data/uploads/embedded_system_module1.pdf"

index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"


# -----------------------------
# Extract and chunk PDF
# -----------------------------

print("Extracting PDF...")

pages = extract_text_from_pdf(pdf_path)

print(f"Pages extracted: {len(pages)}")

chunks = chunk_pages(pages)

print(f"Chunks created: {len(chunks)}")


# -----------------------------
# Generate embeddings
# -----------------------------

print("\nGenerating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = embed_texts(texts)

print(f"Embedding shape: {embeddings.shape}")


# -----------------------------
# Create new FAISS index
# -----------------------------

print("\nBuilding FAISS index...")

vector_store.index = vector_store.faiss.IndexFlatIP(embeddings.shape[1])

vector_store.add_embeddings(embeddings)


# -----------------------------
# Save everything
# -----------------------------

os.makedirs("data/vectorstore", exist_ok=True)

vector_store.save_index(index_path)
vector_store.save_chunks(chunks, chunks_path)


print("\nVector store rebuilt successfully.")
print(f"Index:  {index_path}")
print(f"Chunks: {chunks_path}")