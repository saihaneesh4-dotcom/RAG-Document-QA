import numpy as np

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embeddings import embed_texts


pdf_path = "data/uploads/computer_networks_ass2.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

texts = [chunk["text"] for chunk in chunks]

embeddings = embed_texts(texts)

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First vector norm:", np.linalg.norm(embeddings[0]))
print("First chunk vector, first 10 values:", embeddings[0][:10])