import app.vector_store as vector_store
from app.rag import answer_question


index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"


# Load the existing vector store
vector_store.load_index(index_path)
chunks = vector_store.load_chunks(chunks_path)


# Ask a question
question = input("Enter your question: ")


# Generate an answer using RAG
result = answer_question(question, chunks, k=3)


print("\nAnswer:")
print(result["answer"])


print("\nSources:")

for source in result["sources"]:
    chunk = source["chunk"]

    start_page = chunk["page"]
    end_page = chunk.get("end_page", start_page)

    if start_page == end_page:
        page_label = f"Page {start_page}"
    else:
        page_label = f"Pages {start_page}-{end_page}"

    print(
        f"{page_label} "
        f"(Score: {source['score']:.4f})"
    )