import app.vector_store as vector_store
from sentence_transformers import CrossEncoder


# -----------------------------------
# Test questions
# -----------------------------------

test_cases = [
    {"question": "What are the main characteristics of an embedded system?", "expected_pages": [8, 9]},
    {"question": "What is a programmable logic device and what are its major types?", "expected_pages": [15]},
    {"question": "What is an application specific integrated circuit?", "expected_pages": [16]},
    {"question": "What is the difference between RISC and CISC?", "expected_pages": [90]},
    {"question": "What are the registers available in the ARM processor?", "expected_pages": [101, 102]},
    {"question": "What are the different processor modes in ARM?", "expected_pages": [105]},
    {"question": "What are the different states of an ARM processor?", "expected_pages": [108]},
    {"question": "How does pipelining work in an ARM processor?", "expected_pages": [110, 111, 113, 114]},
    {"question": "How is memory organized in the 8051?", "expected_pages": [57, 58, 60, 61, 62]},
    {"question": "What are the interrupt sources in the 8051?", "expected_pages": [56]},
    {"question": "What are the I/O ports of the 8051?", "expected_pages": [42, 43]},
    {"question": "What are the different types of ARM exceptions?", "expected_pages": [118, 121, 122, 123]},
    {"question": "How does ARM return from an exception?", "expected_pages": [124]},
    {"question": "Which ARM register is used as the stack pointer?", "expected_pages": [101]},
    {"question": "What is the capital of Japan?", "expected_pages": []},
]


# -----------------------------------
# Load vector store
# -----------------------------------

index_path = "data/vectorstore/index.faiss"
chunks_path = "data/vectorstore/chunks.json"

vector_store.load_index(index_path)
chunks = vector_store.load_chunks(chunks_path)


# -----------------------------------
# Load Cross-Encoder
# -----------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------------
# Check whether a chunk contains
# one of the expected pages
# -----------------------------------

def chunk_contains_expected_page(chunk, expected_pages):
    start_page = chunk["page"]
    end_page = chunk.get("end_page", start_page)

    return any(
        start_page <= page <= end_page
        for page in expected_pages
    )


# -----------------------------------
# Evaluation
# -----------------------------------

dense_ranks = []
reranked_ranks = []

dense_hits_at_3 = 0
dense_hits_at_5 = 0
dense_hits_at_10 = 0

reranked_hits_at_3 = 0
reranked_hits_at_5 = 0
reranked_hits_at_10 = 0


for case in test_cases:

    question = case["question"]
    expected_pages = case["expected_pages"]

    # -----------------------------------
    # Dense retrieval
    # -----------------------------------

    dense_results = vector_store.retrieve(
        question,
        chunks,
        k=20
    )

    dense_rank = None

    for rank, result in enumerate(dense_results, start=1):

        if chunk_contains_expected_page(
            result["chunk"],
            expected_pages
        ):
            dense_rank = rank
            break

    # -----------------------------------
    # Cross-Encoder reranking
    # -----------------------------------

    pairs = [
        (question, result["chunk"]["text"])
        for result in dense_results
    ]

    reranker_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(dense_results, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_rank = None

    for rank, (result, score) in enumerate(
        reranked_results,
        start=1
    ):

        if chunk_contains_expected_page(
            result["chunk"],
            expected_pages
        ):
            reranked_rank = rank
            break

    # -----------------------------------
    # Record ranks
    # -----------------------------------

    dense_ranks.append(dense_rank)
    reranked_ranks.append(reranked_rank)

    # -----------------------------------
    # Calculate Recall
    # -----------------------------------

    if expected_pages:

        if dense_rank is not None:

            if dense_rank <= 3:
                dense_hits_at_3 += 1

            if dense_rank <= 5:
                dense_hits_at_5 += 1

            if dense_rank <= 10:
                dense_hits_at_10 += 1

        if reranked_rank is not None:

            if reranked_rank <= 3:
                reranked_hits_at_3 += 1

            if reranked_rank <= 5:
                reranked_hits_at_5 += 1

            if reranked_rank <= 10:
                reranked_hits_at_10 += 1

    # -----------------------------------
    # Print individual result
    # -----------------------------------

    print("\nQuestion:", question)
    print("Expected pages:", expected_pages)
    print("Dense rank:", dense_rank)
    print("Dense + Cross-Encoder rank:", reranked_rank)


# -----------------------------------
# Metrics
# -----------------------------------

positive_cases = sum(
    1
    for case in test_cases
    if case["expected_pages"]
)


dense_recall_3 = dense_hits_at_3 / positive_cases
dense_recall_5 = dense_hits_at_5 / positive_cases
dense_recall_10 = dense_hits_at_10 / positive_cases

reranked_recall_3 = reranked_hits_at_3 / positive_cases
reranked_recall_5 = reranked_hits_at_5 / positive_cases
reranked_recall_10 = reranked_hits_at_10 / positive_cases


# -----------------------------------
# MRR
# -----------------------------------

def calculate_mrr(ranks):

    reciprocal_ranks = []

    for rank in ranks:

        if rank is None:
            reciprocal_ranks.append(0)

        else:
            reciprocal_ranks.append(1 / rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


positive_dense_ranks = [
    rank
    for rank, case in zip(dense_ranks, test_cases)
    if case["expected_pages"]
]

positive_reranked_ranks = [
    rank
    for rank, case in zip(reranked_ranks, test_cases)
    if case["expected_pages"]
]


dense_mrr = calculate_mrr(positive_dense_ranks)
reranked_mrr = calculate_mrr(positive_reranked_ranks)


# -----------------------------------
# Final results
# -----------------------------------

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\nDense Retrieval")
print(f"Recall@3:  {dense_recall_3:.2%}")
print(f"Recall@5:  {dense_recall_5:.2%}")
print(f"Recall@10: {dense_recall_10:.2%}")
print(f"MRR:       {dense_mrr:.4f}")

print("\nDense + Cross-Encoder")
print(f"Recall@3:  {reranked_recall_3:.2%}")
print(f"Recall@5:  {reranked_recall_5:.2%}")
print(f"Recall@10: {reranked_recall_10:.2%}")
print(f"MRR:       {reranked_mrr:.4f}")