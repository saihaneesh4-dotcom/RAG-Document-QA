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


def get_rank(results, expected_pages):
    for position, result in enumerate(results, start=1):
        if result["chunk"]["page"] in expected_pages:
            return position

    return None


def calculate_metrics(ranks):
    total = len(ranks)

    recall_at_3 = sum(
        rank is not None and rank <= 3
        for rank in ranks
    ) / total

    recall_at_5 = sum(
        rank is not None and rank <= 5
        for rank in ranks
    ) / total

    recall_at_10 = sum(
        rank is not None and rank <= 10
        for rank in ranks
    ) / total

    mrr = sum(
        1 / rank if rank is not None else 0
        for rank in ranks
    ) / total

    return recall_at_3, recall_at_5, recall_at_10, mrr


dense_ranks = []
reranked_ranks = []


for test in test_cases:
    question = test["question"]
    expected_pages = test["expected_pages"]

    # Dense retrieval
    dense_results = vector_store.retrieve(
        question,
        chunks,
        k=20
    )

    # Cross-encoder reranking
    pairs = [
        [question, result["chunk"]["text"]]
        for result in dense_results
    ]

    scores = reranker.predict(pairs)

    reranked_results = [
        result
        for result, score in sorted(
            zip(dense_results, scores),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    dense_top10 = dense_results[:10]
    reranked_top10 = reranked_results[:10]

    dense_rank = get_rank(dense_top10, expected_pages)
    reranked_rank = get_rank(reranked_top10, expected_pages)

    dense_ranks.append(dense_rank)
    reranked_ranks.append(reranked_rank)

    print("\n" + "=" * 70)
    print("Question:", question)
    print("Expected pages:", expected_pages)

    print("Dense rank:", dense_rank)
    print("Reranked rank:", reranked_rank)


dense_metrics = calculate_metrics(dense_ranks)
reranked_metrics = calculate_metrics(reranked_ranks)


print("\n" + "=" * 70)
print("FINAL RETRIEVAL COMPARISON")

print("\nDense Retrieval:")
print("Recall@3 :", f"{dense_metrics[0]:.2%}")
print("Recall@5 :", f"{dense_metrics[1]:.2%}")
print("Recall@10:", f"{dense_metrics[2]:.2%}")
print("MRR      :", f"{dense_metrics[3]:.4f}")

print("\nDense + Cross-Encoder:")
print("Recall@3 :", f"{reranked_metrics[0]:.2%}")
print("Recall@5 :", f"{reranked_metrics[1]:.2%}")
print("Recall@10:", f"{reranked_metrics[2]:.2%}")
print("MRR      :", f"{reranked_metrics[3]:.4f}")