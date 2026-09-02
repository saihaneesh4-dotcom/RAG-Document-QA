from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts
from app.vector_store import add_embeddings, retrieve, index


pdf_path = "data/uploads/computer_networks_ass2.pdf"


# 1. Extract text from PDF
pages = extract_text_from_pdf(pdf_path)

# 2. Split extracted text into chunks
chunks = chunk_pages(pages)

# 3. Create embeddings for all chunks
texts = [chunk["text"] for chunk in chunks]
embeddings = embed_texts(texts)

# 4. Add embeddings to FAISS
add_embeddings(embeddings)


# 5. Ask a question
query = "What is Hamming code used for?"

# 6. Retrieve the most relevant chunks
results = retrieve(query, chunks, k=3)


# 7. Display the retrieved results
print("Query:", query)

for result in results:
    print("\nScore:", result["score"])
    print("Chunk ID:", result["chunk"]["chunk_id"])
    print("Page:", result["chunk"]["page"])
    print("Text:", result["chunk"]["text"][:300])

print("\nTotal vectors in FAISS:", index.ntotal)