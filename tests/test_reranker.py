from sentence_transformers import CrossEncoder

import app.vector_store as vector_store


index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"

vector_store.load_index(index_path)
chunks = vector_store.load_chunks(chunks_path)


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


test_cases = [
    {"question": "What are the five addressing modes in 8051?", "expected_pages": [74]},
    {"question": "What is the difference between a microprocessor and a microcontroller?", "expected_pages": [36, 37]},
    {"question": "What is the function of DPTR?", "expected_pages": [68]},
    {"question": "What is the accumulator?", "expected_pages": [65]},
    {"question": "What are the ports of the 8051?", "expected_pages": [42]},
    {"question": "What is immediate addressing mode?", "expected_pages": [76]},
    {"question": "What is direct addressing mode?", "expected_pages": [78]},
    {"question": "What is register addressing mode?", "expected_pages": [77]},
    {"question": "What is indexed addressing mode?", "expected_pages": [82]},
    {"question": "How does ARM handle an exception?", "expected_pages": [118, 119, 120, 121, 122, 123, 124]},
]


for test in test_cases:
    question = test["question"]
    expected_pages = test["expected_pages"]

    # First retrieve a larger candidate pool using dense retrieval
    candidates = vector_store.retrieve(question, chunks, k=20)

    # Create question-chunk pairs for the cross-encoder
    pairs = [
        [question, result["chunk"]["text"]]
        for result in candidates
    ]

    # Score each candidate
    scores = reranker.predict(pairs)

    # Sort candidates by cross-encoder score
    ranked_results = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    retrieved_pages = [
        result["chunk"]["page"]
        for result, score in ranked_results[:10]
    ]

    rank = None

    for position, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            rank = position
            break

    print("\n" + "=" * 70)
    print("Question:", question)
    print("Expected pages:", expected_pages)
    print("Reranked pages:", retrieved_pages)

    if rank is not None:
        print("Correct page found at rank:", rank)
    else:
        print("Correct page NOT found in top 10")