from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages

pdf_path = "data/uploads/computer_networks_ass2.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))

print("\n--- Chunk Information ---")

for i in range(min(10, len(chunks))):
    print(
        f"Chunk {chunks[i]['chunk_id']} "
        f"| Page {chunks[i]['page']} "
        f"| Characters: {len(chunks[i]['text'])}"
    )

print("\n--- Overlap Test ---")

for i in range(len(chunks) - 1):
    current = chunks[i]
    next_chunk = chunks[i + 1]

    if current["page"] == next_chunk["page"]:
        overlap_matches = (
            current["text"][-200:] == next_chunk["text"][:200]
        )

        print(
            f"Chunk {current['chunk_id']} → "
            f"Chunk {next_chunk['chunk_id']} "
            f"| Same page: {current['page']} "
            f"| Overlap correct: {overlap_matches}"
        )