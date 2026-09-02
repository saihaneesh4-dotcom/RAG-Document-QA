def chunk_pages(pages, chunk_size=1000, overlap=200):
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]

        if not text.strip():
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk_text
            })

            chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks