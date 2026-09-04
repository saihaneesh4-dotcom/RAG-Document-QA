from rank_bm25 import BM25Okapi

import app.vector_store as vector_store


index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"


# Load the existing chunks
chunks = vector_store.load_chunks(chunks_path)


# Prepare chunk text for BM25
tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]


# Build the BM25 index
bm25 = BM25Okapi(tokenized_chunks)


# Test questions
test_cases = [
    {
        "question": "What are the five addressing modes in 8051?",
        "expected_pages": [74],
    },
    {
        "question": "What is the difference between a microprocessor and a microcontroller?",
        "expected_pages": [36, 37],
    },
    {
        "question": "What is the function of DPTR?",
        "expected_pages": [68],
    },
    {
        "question": "What is the accumulator?",
        "expected_pages": [65],
    },
    {
        "question": "What are the ports of the 8051?",
        "expected_pages": [42],
    },
    {
        "question": "What is immediate addressing mode?",
        "expected_pages": [76],
    },
    {
        "question": "What is direct addressing mode?",
        "expected_pages": [78],
    },
    {
        "question": "What is register addressing mode?",
        "expected_pages": [77],
    },
    {
        "question": "What is indexed addressing mode?",
        "expected_pages": [82],
    },
    {
        "question": "How does ARM handle an exception?",
        "expected_pages": [118, 119, 120, 121, 122, 123, 124],
    },
]


for test in test_cases:

    question = test["question"]
    expected_pages = test["expected_pages"]

    tokenized_query = question.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:10]

    retrieved_pages = [
        chunks[index]["page"]
        for index in ranked_indices
    ]

    rank = None

    for position, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            rank = position
            break

    print("\n" + "=" * 70)
    print("Question:", question)
    print("Expected pages:", expected_pages)
    print("Retrieved pages:", retrieved_pages)

    if rank is not None:
        print("Correct page found at rank:", rank)
    else:
        print("Correct page NOT found in top 10")