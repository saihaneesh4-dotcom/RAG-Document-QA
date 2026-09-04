from app.ingestion import process_document


pdf_path = "data/uploads/computer_networks_ass2.pdf"

process_document(pdf_path)

print("Document processed and vector store saved successfully.")