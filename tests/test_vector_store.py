import os

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts
import app.vector_store as vector_store

pdf_path = "data/uploads/embedded_system_module1.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

texts = [chunk["text"] for chunk in chunks]
embeddings = embed_texts(texts)

vector_store.add_embeddings(embeddings)

index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"

vector_store.save_index(index_path)
vector_store.save_chunks(chunks, chunks_path)

print("Chunks saved:", os.path.exists(chunks_path))
print("Index saved:", os.path.exists(index_path))

vector_store.load_index(index_path)

loaded_chunks = vector_store.load_chunks(chunks_path)

print("Chunks after loading:", len(loaded_chunks))
print("Vectors after loading:", vector_store.index.ntotal)

queries = [
    "What are the five addressing modes in 8051?",
    "List the types of AM in 8051",
    "Types of AM in 8051",
    "Immediate Register Direct Register Indirect Indexed",
    "8051 addressing modes"
]

for query in queries:

    results = vector_store.retrieve(query, loaded_chunks, k=10)

    print("\n" + "=" * 60)
    print("Query:", query)

    for result in results:
        chunk = result["chunk"]

        print(
            f"\nScore: {result['score']:.4f}"
            f"\nChunk ID: {chunk['chunk_id']}"
            f"\nPage: {chunk['page']}"
            f"\nText: {chunk['text'][:250]}"
        )
