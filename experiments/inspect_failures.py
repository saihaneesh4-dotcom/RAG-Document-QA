import app.vector_store as vector_store

index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"

vector_store.load_index(index_path)
chunks = vector_store.load_chunks(chunks_path)

questions = [
    "What are the interrupt sources in the 8051?",
    "What is the purpose of the stack pointer in ARM?"
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = vector_store.retrieve(question, chunks, k=20)

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(
            f"\nRank {rank} | "
            f"Page {chunk['page']} | "
            f"Score {result['score']:.4f}"
        )

        print(chunk["text"][:300].replace("\n", " "))