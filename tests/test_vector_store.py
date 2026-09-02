from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts
from app.vector_store import index, add_embeddings


pdf_path = "data/uploads/computer_networks_ass2.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

texts = [chunk["text"] for chunk in chunks]

embeddings = embed_texts(texts)

add_embeddings(embeddings)


print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Number of vectors in FAISS:", index.ntotal)
print("FAISS vector dimension:", index.d)