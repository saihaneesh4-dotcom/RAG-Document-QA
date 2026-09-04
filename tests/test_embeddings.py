from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts

pdf_path = "data/uploads/embedded_system_module1.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

texts = [chunk["text"] for chunk in chunks]

embeddings = embed_texts(texts)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding:", embeddings[0])