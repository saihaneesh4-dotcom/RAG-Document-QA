import faiss

dimension = 384

index = faiss.IndexFlatIP(dimension)


def add_embeddings(embeddings):
    index.add(embeddings)