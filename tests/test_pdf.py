from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages

pdf_path = "data/uploads/embedded_system_module1.pdf"

pages = extract_text_from_pdf(pdf_path)

chunks = chunk_pages(pages)

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))

for chunk in chunks[:5]:
    print("\nChunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page"])
    print("Characters:", len(chunk["text"]))
    print("Text:", chunk["text"][:300])