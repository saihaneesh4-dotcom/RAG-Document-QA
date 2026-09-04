def chunk_pages(pages, chunk_size=1000, overlap=200):
    chunks = []
    chunk_id = 0

    i = 0

    while i < len(pages):
        page = pages[i]
        text = page["text"]

        if not text.strip():
            i += 1
            continue

        # If the next page starts with "Cont.", combine it
        # with the current page so continued sections stay together.
        if i + 1 < len(pages):
            next_page = pages[i + 1]

            if next_page["text"].strip().lower().startswith("cont."):
                text = text + "\n" + next_page["text"]
                end_page = next_page["page"]
                i += 2
            else:
                end_page = page["page"]
                i += 1
        else:
            end_page = page["page"]
            i += 1

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "end_page": end_page,
                "text": chunk_text
            })

            chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks