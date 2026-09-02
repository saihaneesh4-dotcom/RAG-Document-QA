from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts
import app.vector_store as vector_store


pdf_path = "data/uploads/computer_networks_ass2.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

texts = [chunk["text"] for chunk in chunks]

embeddings = embed_texts(texts)

vector_store.add_embeddings(embeddings)
index_path = "data/vectorstore/index.faiss"

vector_store.save_index(index_path)

vector_store.load_index(index_path)

print("Vectors after loading:", vector_store.index.ntotal)


query = "What is Hamming code used for?"

results = vector_store.retrieve(query, chunks, k=3)


print("Query:", query)

for result in results:
    print("\nScore:", result["score"])
    print("Chunk ID:", result["chunk"]["chunk_id"])
    print("Page:", result["chunk"]["page"])
    print("Text:", result["chunk"]["text"][:300])