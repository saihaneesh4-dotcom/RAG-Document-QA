import app.vector_store as vector_store


vector_store.load_index(
    "data/vectorstore/index.faiss"
)

chunks = vector_store.load_chunks(
    "data/vectorstore/chunks.json"
)

results = vector_store.retrieve(
    "What is Hamming code?",
    chunks,
    k=5
)

for i, result in enumerate(results, start=1):

    chunk = result["chunk"]

    print("\n" + "=" * 80)

    print(
        f"Rank: {i}\n"
        f"Score: {result['score']:.4f}\n"
        f"Document: {chunk['document']}\n"
        f"Pages: {chunk['page']}-{chunk.get('end_page', chunk['page'])}"
    )

    print("\nText:")
    print(chunk["text"])