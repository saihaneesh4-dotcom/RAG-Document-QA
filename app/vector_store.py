import faiss
import os
import json

dimension = 384

index = faiss.IndexFlatIP(dimension)


def add_embeddings(embeddings):
    index.add(embeddings)


def search(query_embedding, chunks, k=3):
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id == -1:
            continue

        chunk = chunks[index_id]

        results.append({
            "score": float(score),
            "chunk": chunk
        })

    return results

def retrieve(query, chunks, k=3):
    from app.embeddings import embed_texts

    query_embedding = embed_texts([query])

    return search(query_embedding, chunks, k)

def save_index(path):
    faiss.write_index(index, path)

def load_index(path):
    global index
    index = faiss.read_index(path)

def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)