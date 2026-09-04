import json
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.embeddings import embed_texts


# -----------------------------------
# Load vector store
# -----------------------------------

index = faiss.read_index("data/vectorstore/index.faiss")

with open("data/vectorstore/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# -----------------------------------
# Prepare BM25
# -----------------------------------

tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# -----------------------------------
# Load Cross-Encoder
# -----------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------------
# Test question
# -----------------------------------

question = "What are the interrupt sources in the 8051?"


# -----------------------------------
# Dense retrieval
# -----------------------------------

query_embedding = embed_texts([question])

dense_scores, dense_indices = index.search(
    query_embedding,
    20
)

dense_results = []

for score, index_id in zip(dense_scores[0], dense_indices[0]):
    dense_results.append({
        "chunk": chunks[index_id],
        "score": float(score)
    })


# -----------------------------------
# BM25 retrieval
# -----------------------------------

bm25_scores = bm25.get_scores(
    question.lower().split()
)

bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:20]


# -----------------------------------
# Reciprocal Rank Fusion
# -----------------------------------

rrf_scores = {}

k = 60

for rank, result in enumerate(dense_results, start=1):
    chunk_id = result["chunk"]["chunk_id"]
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)


for rank, index_id in enumerate(bm25_indices, start=1):
    chunk_id = chunks[index_id]["chunk_id"]
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)


hybrid_chunk_ids = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)[:20]


hybrid_results = [
    chunks[chunk_id]
    for chunk_id in hybrid_chunk_ids
]


# -----------------------------------
# Cross-Encoder reranking
# -----------------------------------

pairs = [
    (question, chunk["text"])
    for chunk in hybrid_results
]

cross_scores = reranker.predict(pairs)

final_results = sorted(
    zip(hybrid_results, cross_scores),
    key=lambda x: x[1],
    reverse=True
)


# -----------------------------------
# Print results
# -----------------------------------

print("\n" + "=" * 70)
print("DENSE TOP 20")
print("=" * 70)

for rank, result in enumerate(dense_results, start=1):
    chunk = result["chunk"]
    print(
        f"Rank {rank:2} | "
        f"Page {chunk['page']:3} | "
        f"Score {result['score']:.4f}"
    )


print("\n" + "=" * 70)
print("HYBRID + CROSS-ENCODER TOP 20")
print("=" * 70)

for rank, (chunk, score) in enumerate(final_results, start=1):
    print(
        f"Rank {rank:2} | "
        f"Page {chunk['page']:3} | "
        f"Score {score:.4f}"
    )