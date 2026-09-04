from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages


pdf_path = "data/uploads/embedded_system_module1.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)


for chunk in chunks:
    if "Generally five interrupt sources" in chunk["text"]:
        print("\nFound interrupt section:")
        print("Start page:", chunk["page"])
        print("End page:", chunk["end_page"])
        print("\nText:")
        print(chunk["text"])