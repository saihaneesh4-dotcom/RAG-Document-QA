import app.vector_store as vector_store


chunks_path = "data/vectorstore/chunks.json"
index_path = "data/vectorstore/index.faiss"

vector_store.load_index(index_path)

chunks = vector_store.load_chunks(chunks_path)


test_cases = [
    {
        "question": "What are the five addressing modes in 8051?",
        "expanded_query": "8051 addressing modes Immediate Register Direct Register Indirect Indexed",
        "expected_pages": [74]
    },
    {
        "question": "What is the difference between a microprocessor and a microcontroller?",
        "expanded_query": "microprocessor microcontroller differences characteristics architecture memory input output peripherals",
        "expected_pages": [36, 37]
    },
    {
        "question": "What is the function of DPTR?",
        "expanded_query": "8051 DPTR Data Pointer function register",
        "expected_pages": [68]
    },
    {
        "question": "What is the accumulator?",
        "expanded_query": "8051 accumulator ACC accumulator register function",
        "expected_pages": [65]
    },
    {
        "question": "What are the ports of the 8051?",
        "expanded_query": "8051 ports Port 0 Port 1 Port 2 Port 3 pins",
        "expected_pages": [42]
    },
    {
        "question": "What is immediate addressing mode?",
        "expanded_query": "8051 immediate addressing mode immediate data operand",
        "expected_pages": [76]
    },
    {
        "question": "What is direct addressing mode?",
        "expanded_query": "8051 direct addressing mode direct address internal RAM SFR",
        "expected_pages": [78]
    },
    {
        "question": "What is register addressing mode?",
        "expanded_query": "8051 register addressing mode R0 R1 R2 R3 R4 R5 R6 R7 register",
        "expected_pages": [77]
    },
    {
        "question": "What is indexed addressing mode?",
        "expanded_query": "8051 indexed addressing mode DPTR accumulator code memory MOVC",
        "expected_pages": [82]
    },
    {
        "question": "How does ARM handle an exception?",
        "expanded_query": "ARM exception handling exceptions processor modes CPSR SPSR vector address IRQ FIQ reset",
        "expected_pages": [118, 119, 120, 121, 122, 123, 124]
    }
]


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


original_ranks = []
expanded_ranks = []


for test in test_cases:
    question = test["question"]
    expanded_query = test["expanded_query"]
    expected_pages = test["expected_pages"]

    original_results = vector_store.retrieve(
        question,
        chunks,
        k=10
    )

    expanded_results = vector_store.retrieve(
        expanded_query,
        chunks,
        k=10
    )

    original_pages = [
        result["chunk"]["page"]
        for result in original_results
    ]

    expanded_pages = [
        result["chunk"]["page"]
        for result in expanded_results
    ]

    original_rank = None
    expanded_rank = None

    for position, page in enumerate(original_pages, start=1):
        if page in expected_pages:
            original_rank = position
            break

    for position, page in enumerate(expanded_pages, start=1):
        if page in expected_pages:
            expanded_rank = position
            break

    original_ranks.append(original_rank)
    expanded_ranks.append(expanded_rank)

    print("\n" + "=" * 70)
    print("Question:", question)
    print("Expected pages:", expected_pages)

    print("\nOriginal query:")
    print("Rank:", original_rank)
    print("Pages:", original_pages)

    print("\nExpanded query:")
    print("Rank:", expanded_rank)
    print("Pages:", expanded_pages)


original_metrics = calculate_metrics(original_ranks)
expanded_metrics = calculate_metrics(expanded_ranks)


print("\n" + "=" * 70)
print("OVERALL METRICS")


print("\nOriginal Query:")
print("Recall@3 :", f"{original_metrics[0]:.2%}")
print("Recall@5 :", f"{original_metrics[1]:.2%}")
print("Recall@10:", f"{original_metrics[2]:.2%}")
print("MRR      :", f"{original_metrics[3]:.4f}")


print("\nExpanded Query:")
print("Recall@3 :", f"{expanded_metrics[0]:.2%}")
print("Recall@5 :", f"{expanded_metrics[1]:.2%}")
print("Recall@10:", f"{expanded_metrics[2]:.2%}")
print("MRR      :", f"{expanded_metrics[3]:.4f}")