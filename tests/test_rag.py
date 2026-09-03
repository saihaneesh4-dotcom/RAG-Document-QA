from app.rag import answer_question
import app.vector_store as vector_store


index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"

vector_store.load_index(index_path)
chunks = vector_store.load_chunks(chunks_path)

question = "What is Hamming code used for?"

result = answer_question(question, chunks, k=3)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(
        f"Chunk {source['chunk']['chunk_id']} "
        f"(Page {source['chunk']['page']}, "
        f"Score {source['score']:.4f})"
    )