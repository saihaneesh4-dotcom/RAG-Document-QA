from app.ingestion import process_documents


pdf_paths = [
    "data/uploads/computer_networks_ass2.pdf",
    "data/uploads/embedded_system_module1.pdf"
]

chunks = process_documents(pdf_paths)

print(f"Total chunks: {len(chunks)}")

print("\nFirst 5 chunks:")

for chunk in chunks[:5]:
    print(
        f"Chunk {chunk['chunk_id']} | "
        f"{chunk['document']} | "
        f"Pages {chunk['page']}-{chunk.get('end_page', chunk['page'])}"
    )